"""Implementation of Stochastically Stable Distribution (SSD) analysis.

This module implements the SSD algorithm for computing stochastically stable
distributions of perturbed Markov processes, with applications to evolutionary
game theory and multiagent learning.

Based on:
"An Algorithm for Computing Stochastically Stable Distributions with
Applications to Multiagent Learning in Repeated Games"
by John R. Wicks and Amy Greenwald.
https://arxiv.org/pdf/1207.1424
"""

from numbers import Number
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from numpy.polynomial import Polynomial as P
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from open_spiel.python.egt import utils
from open_spiel.python.egt.kosaraju import Graph
from open_spiel.python.egt import alpharank

def _convert_alpharank_to_ssd_format(
    payoff_tables: List[Any],
    payoffs_are_hpt_format: bool) -> Dict[str, Any]:
  """Converts Alpha-Rank payoff tables to SSD format."""
  num_strats = utils.get_num_strats_per_population(payoff_tables,
                                                   payoffs_are_hpt_format)
  if payoffs_are_hpt_format:
    num_profiles = len(payoff_tables[0])
  elif len(payoff_tables) == 1:
    num_profiles = payoff_tables[0].shape[0]
  else:
    num_profiles = utils.get_num_profiles(
        [table.shape[0] for table in payoff_tables])
  return {
      "payoff_tables": payoff_tables,
      "payoffs_are_hpt_format": payoffs_are_hpt_format,
      "num_populations": len(payoff_tables),
      "num_strats_per_population": num_strats,
      "num_profiles": num_profiles,
  }

def _validate_ssd_distribution(distribution: np.ndarray,
                               tolerance: float = 1e-6) -> bool:
  return (np.all(distribution >= -tolerance) and
          abs(np.sum(distribution) - 1.0) <= tolerance)

def normalize(vector: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
  """Normalize a vector so that it sums to 1."""
  if sp.issparse(vector):
    arr = np.array(vector.toarray(), dtype=float).flatten()
  else:
    arr = np.array(vector, dtype=float).flatten()
  total = np.sum(arr)
  if total == 0:
    return arr
  return arr / total


def _select_stationary_vector(eigenvalues, eigenvectors):
  """Select eigenvector associated with the largest real eigenvalue."""
  eigenvalues = np.asarray(eigenvalues)
  idx = np.argmax(np.real(eigenvalues))
  vec = np.real(eigenvectors[:, idx])
  return vec


# TODO: check if this works
def _stable_distribution(matrix, tol: float = 1e-9):
  """Computes the normalized eigenvector for eigenvalue 1 for a unichain Markov matrix."""
  is_sparse = sp.issparse(matrix)
  if is_sparse:
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    col_sums = np.asarray(matrix.sum(axis=0)).reshape(-1)
    target_dense = None
  else:
    target_dense = np.asarray(matrix, dtype=float)
    row_sums = target_dense.sum(axis=1)
    col_sums = target_dense.sum(axis=0)

  rows_stochastic = np.allclose(row_sums, 1.0, atol=tol)
  cols_stochastic = np.allclose(col_sums, 1.0, atol=tol)

  # fix for small matrix games ?
  use_transpose = True
  if cols_stochastic:
    use_transpose = False
  elif rows_stochastic:
    use_transpose = True

  if is_sparse:
    target_matrix = matrix.T if use_transpose else matrix
    eig_vals, eig_vecs = spla.eigs(target_matrix, k=1, which="LM")
  else:
    if use_transpose:
      target_matrix = target_dense.T
    else:
      target_matrix = target_dense
    eig_vals, eig_vecs = np.linalg.eig(target_matrix)

  vec = _select_stationary_vector(eig_vals, eig_vecs)
  return normalize(vec)


def _poly_eval(arg, val):
  return np.polyval(arg, val) if isinstance(arg, np.poly1d) else arg

_eval_poly_array = np.frompyfunc(_poly_eval, 2, 1)

def _zero_test(value):
  return value == 0

def _poly_zero_test(poly):
  """Returns true if the polynomial is 0."""
  if isinstance(poly, np.poly1d):
    return poly == np.poly1d([0])
  return poly == 0


_POLY_ZERO = np.poly1d([0])
_POLY_ONE = np.poly1d([1])


def _ensure_poly1d(value: Any) -> np.poly1d:
  """Return a numpy.poly1d representation for the provided value."""
  if isinstance(value, np.poly1d):
    return value
  if isinstance(value, P):
    return np.poly1d(value.coef)
  if isinstance(value, np.ndarray):
    if value.size == 1:
      return _ensure_poly1d(value.item())
  if isinstance(value, Number):
    return np.poly1d([float(value)])
  if value is None:
    return _POLY_ZERO
  return np.poly1d(value)


def _evaluate_sparse_constant(matrix: Any,
                              eps_value: float = 0.0) -> sp.csr_matrix:
  """Evaluate a (possibly SparsePolyMatrix) sparse polynomial matrix at epsilon."""
  if _is_sparse_poly_matrix(matrix):
    rows = []
    cols = []
    data = []
    for (row, col), entry in matrix.items():
      value = float(_ensure_poly1d(entry)(eps_value))
      if value == 0:
        continue
      rows.append(row)
      cols.append(col)
      data.append(value)
    return sp.csr_matrix((data, (rows, cols)), shape=matrix.shape)

  if not matrix.nnz: # not custom class
    return sp.csr_matrix(matrix.shape, dtype=float)
  coo = matrix.tocoo()
  data = np.empty_like(coo.data, dtype=float)
  for idx, entry in enumerate(coo.data):
    poly = _ensure_poly1d(entry)
    data[idx] = float(poly(eps_value))
  return sp.csr_matrix((data, (coo.row, coo.col)), shape=matrix.shape)

def _set_sparse_value(matrix: Any,
                      row: int,
                      col: int,
                      value: Any,
                      tol: float = 1e-16):
  """Assign or remove (when value=0) to row, col in a sparse matrix."""
  if _is_sparse_poly_matrix(matrix):
    matrix.set(row, col, value, tol)
    return
  if isinstance(matrix, sp.dok_matrix):
    if _is_zero_poly_entry(value, tol=tol):
      if (row, col) in matrix:
        del matrix[row, col]
    else:
      matrix[row, col] = value
    return
  raise TypeError("Unsupported sparse structure in _set_sparse_value")


class SparsePolyMatrix:
  """dictionary based sparse matrix to store polynomial entries."""

  __slots__ = ("shape", "_data")

  def __init__(self, shape: Tuple[int, int],
               data: Optional[Dict[Tuple[int, int], Any]] = None):
    self.shape = shape
    self._data = {} if data is None else dict(data)

  def copy(self):
    return SparsePolyMatrix(self.shape, self._data)

  def get(self, key: Tuple[int, int], default: Any = _POLY_ZERO) -> Any:
    return self._data.get(key, default)

  def set(self, row: int, col: int, value: Any, tol: float = 1e-16):
    if _is_zero_poly_entry(value, tol=tol):
      self._data.pop((row, col), None)
    else:
      self._data[(row, col)] = value

  def items(self):
    return self._data.items()

  def values(self):
    return self._data.values()

  def nnz(self) -> int:
    return len(self._data)

  def __getitem__(self, key: Tuple[int, int]) -> Any:
    if isinstance(key, Tuple[int, int]) and len(key) == 2:
      return self.get((key[0], key[1]), _POLY_ZERO)
    if isinstance(key, Tuple[np.ndarray]):
      print("AM IN HERE")
    raise TypeError("SparsePolyMatrix can get 2d index or np.ix_ output only")

  def __setitem__(self, key: Tuple[int, int], value: Any):
    if isinstance(key, tuple) and len(key) == 2:
      self.set(key[0], key[1], value)
      return
    raise TypeError("SparsePolyMatrix can set 2d index only")

def _is_sparse_poly_matrix(matrix: Any) -> bool:
  return isinstance(matrix, SparsePolyMatrix)

def _sparse_poly_indicator(matrix: SparsePolyMatrix) -> sp.csr_matrix:
  """Creates a spare adjacency matrix"""
  rows = []
  cols = []
  for (row, col), value in matrix.items():
    if not _poly_zero_test(value):
      rows.append(row)
      cols.append(col)
  data = np.ones(len(rows), dtype=float)
  return sp.csr_matrix((data, (rows, cols)), shape=matrix.shape)


def _make_linear_poly(a0: float, a1: float, tol: float = 1e-14):
  """Returns either the linear polynomial a0 + a1 * eps or just a0 if a1 approx 0."""
  if abs(a1) <= tol:
    return float(a0)
  return np.poly1d([a1, a0])


def _shift_exponent(poly, degree):
  """Lowers the degree of poly."""
  coefs = list(poly.c)
  poly_degree = len(coefs) - 1
  if poly_degree < degree:
    return np.poly1d([])
  del coefs[poly_degree - degree + 1: poly_degree + 1]
  return np.poly1d(coefs)


def _cost_resistance(poly):
  """Return the cost and resistance of poly."""
  poly = _ensure_poly1d(poly)
  coefs = poly.c
  if len(coefs) == 1 and coefs[0] == 0:
    return {"cost": 1, "resistance": "infinity"}
  i = len(coefs) - 1
  while i > 0 and coefs[i] == 0:
    i -= 1
  return {"cost": coefs[i], "resistance": len(coefs) - i - 1}


def _non_uniform_scale(matrix):
  if _is_sparse_poly_matrix(matrix) or sp.issparse(matrix):
    return _non_uniform_scale_sparse(matrix)
  return _non_uniform_scale_dense(matrix)


def _non_uniform_scale_dense(matrix):
  result = {"dim": matrix.shape[1]}
  dim_range = range(result["dim"])
  min_resistance = -1
  max_cost_sum = 0
  non_transient_cols = list(dim_range)

  for col in dim_range:
    min_col_resistance = -1
    col_cost_sum = 0
    for row in dim_range:
      if row == col:
        continue
      cr = _cost_resistance(matrix[row, col])
      if cr["resistance"] == "infinity":
        continue
      if cr["resistance"] == 0:
        min_col_resistance = 0
        break
      if min_col_resistance < 0 or cr["resistance"] < min_col_resistance:
        min_col_resistance = cr["resistance"]
        col_cost_sum = cr["cost"]
      elif cr["resistance"] == min_col_resistance:
        col_cost_sum += cr["cost"]

    if min_col_resistance == 0:
      non_transient_cols.remove(col)
    elif min_col_resistance > 0:
      if min_resistance < 0 or min_col_resistance < min_resistance:
        min_resistance = min_col_resistance
        max_cost_sum = col_cost_sum
      elif min_col_resistance == min_resistance:
        max_cost_sum += col_cost_sum

  if not non_transient_cols:
    raise ValueError("No scaling is possible.")

  result["mat"] = np.array(matrix, copy=True)
  new_mat = result["mat"]
  result["D"] = np.array(np.identity(result["dim"]), dtype=object)
  d_matrix = result["D"]
  del result["dim"]

  f_cost = 2 * max_cost_sum
  tmp = [f_cost]
  tmp.extend([0] * min_resistance)
  scaling_poly = np.poly1d(tmp)
  for col in dim_range:
    if col not in non_transient_cols:
      d_matrix[col, col] = scaling_poly

  poly_one = np.poly1d([1])
  for col in dim_range:
    if col in non_transient_cols:
      for row in dim_range:
        if row == col:
          new_mat[row, col] = poly_one + (poly_one / f_cost) * _shift_exponent(
              matrix[row, col] - poly_one, min_resistance)
        else:
          new_mat[row, col] = (poly_one / f_cost) * _shift_exponent(
              matrix[row, col], min_resistance)
  return result


def _non_uniform_scale_sparse(matrix):
  source = matrix
  dim = source.shape[0]
  column_entries = defaultdict(list)
  for (row, col), value in source.items():
    if row == col:
      continue
    column_entries[col].append((row, value))

  min_resistance = -1
  max_cost_sum = 0
  non_transient_cols = set(range(dim))

  for col in range(dim):
    min_col_resistance = -1
    col_cost_sum = 0
    for row, value in column_entries.get(col, []):
      cr = _cost_resistance(value)
      if cr["resistance"] == "infinity":
        continue
      if cr["resistance"] == 0:
        min_col_resistance = 0
        break
      if min_col_resistance < 0 or cr["resistance"] < min_col_resistance:
        min_col_resistance = cr["resistance"]
        col_cost_sum = cr["cost"]
      elif cr["resistance"] == min_col_resistance:
        col_cost_sum += cr["cost"]

    if min_col_resistance == 0:
      non_transient_cols.discard(col)
    elif min_col_resistance > 0:
      if min_resistance < 0 or min_col_resistance < min_resistance:
        min_resistance = min_col_resistance
        max_cost_sum = col_cost_sum
      elif min_col_resistance == min_resistance:
        max_cost_sum += col_cost_sum

  if not non_transient_cols:
    raise ValueError("No scaling is possible.")

  new_mat = source.copy()
  identity = np.array(np.identity(dim), dtype=object)
  f_cost = 2 * max_cost_sum
  tmp = [f_cost]
  tmp.extend([0] * min_resistance)
  scaling_poly = np.poly1d(tmp)

  for col in range(dim):
    if col not in non_transient_cols:
      identity[col, col] = scaling_poly
      continue
    diag_orig = _ensure_poly1d(source.get((col, col), _POLY_ZERO))
    diag_new = _POLY_ONE + (_POLY_ONE / f_cost) * _shift_exponent(
        diag_orig - _POLY_ONE, min_resistance)
    _set_sparse_value(new_mat, col, col, diag_new)
    for row, value in column_entries.get(col, []):
      updated = (_POLY_ONE / f_cost) * _shift_exponent(
          _ensure_poly1d(value), min_resistance)
      _set_sparse_value(new_mat, row, col, updated)

    return {
        "mat": new_mat,
        "D": identity
    }


def _reduce(matrix, states_to_remove, M0=None):
  """Eliminate selected states from matrix."""
  if _is_sparse_poly_matrix(matrix) or sp.issparse(matrix):
    return _reduce_sparse(matrix, states_to_remove, M0)
  return _reduce_dense(matrix, states_to_remove, M0)


def _reduce_dense(matrix, states_to_remove, M0=None):
  if not states_to_remove:
    raise ValueError("states_to_remove cannot be empty.")

  dim = matrix.shape[1]
  if M0 is None:
    M0 = np.array(_eval_poly_array(matrix, 0), dtype=float)

  complement = list(set(range(dim)).symmetric_difference(set(states_to_remove)))
  perm = list(complement)
  perm.extend(states_to_remove)
  P_matrix = np.zeros((dim, dim))
  for col in range(dim):
    P_matrix[perm[col], col] = 1

  Mssbar = matrix[np.ix_(states_to_remove, complement)]
  lambdass_inv = np.linalg.inv(M0[np.ix_(states_to_remove, states_to_remove)] -
                               np.identity(len(states_to_remove)))
  inclusion = P_matrix @ np.block([[np.identity(len(complement))],
                                   [-lambdass_inv @ Mssbar]])
  tmp = matrix[np.ix_(complement, complement)] - matrix[np.ix_(
      complement, states_to_remove)] @ lambdass_inv @ Mssbar

  dim_tmp = tmp.shape[1]
  zero = np.poly1d([0])
  one = np.poly1d([1])
  for col in range(dim_tmp):
    col_sum = zero
    for row in range(dim_tmp):
      if row == col:
        continue
      cr = _cost_resistance(tmp[row, col])
      if cr["resistance"] == "infinity":
        tmp[row, col] = zero
      else:
        coeffs = [0] * (cr["resistance"] + 1)
        coeffs[0] = cr["cost"]
        tmp[row, col] = np.poly1d(coeffs)
      col_sum = col_sum + tmp[row, col]
    tmp[col, col] = one - col_sum

  return {"i": inclusion, "mat": tmp}

def _reduce_sparse(matrix, states_to_remove, M0=None):
  if not states_to_remove:
    raise ValueError("states_to_remove cannot be empty.")

  dim = matrix.shape[1]
  if M0 is None:
    M0 = _evaluate_sparse_constant(matrix, 0)

  complement = list(set(range(dim)).symmetric_difference(set(states_to_remove)))
  perm = list(complement)
  perm.extend(states_to_remove)
  P_matrix = sp.dok_matrix((dim, dim), dtype=float)
  for col in range(dim):
    P_matrix[perm[col], col] = 1

  Mssbar = matrix[np.ix_(states_to_remove, complement)]
  lambdass_inv = spla.inv(M0[np.ix_(states_to_remove, states_to_remove)] -
                               sp.identity(len(states_to_remove)))
  inclusion = P_matrix @ sp.block_array([[sp.identity(len(complement))],
                                   [-lambdass_inv @ Mssbar]])
  tmp = matrix[np.ix_(complement, complement)] - matrix[np.ix_(
      complement, states_to_remove)] @ lambdass_inv @ Mssbar

  dim_tmp = tmp.shape[1]
  zero = np.poly1d([0])
  one = np.poly1d([1])
  for col in range(dim_tmp):
    col_sum = zero
    for row in range(dim_tmp):
      if row == col:
        continue
      cr = _cost_resistance(tmp[row, col])
      if cr["resistance"] == "infinity":
        tmp[row, col] = zero
      else:
        coeffs = [0] * (cr["resistance"] + 1)
        coeffs[0] = cr["cost"]
        tmp[row, col] = np.poly1d(coeffs)
      col_sum = col_sum + tmp[row, col]
    tmp[col, col] = one - col_sum

  return {"i": inclusion, "mat": tmp}



def _ssd_step(matrix):
  """Performs the next possible reduction step in the SSD algorithm."""
  result = {}
  if _is_sparse_poly_matrix(matrix) or sp.issparse(matrix):
    M0 = _evaluate_sparse_constant(matrix)
  else:
    M0 = np.array(_eval_poly_array(matrix, 0), dtype=float)

  graph = Graph(M0, _zero_test)
  communicating = graph.CommunicatingClasses()

  if len(communicating) == 1:
    result["stab"] = _stable_distribution(M0)
    return result

  closed = graph.ClosedClasses()
  max_size = 0
  max_class = None
  num_non_trivial = 0
  for cls in closed.values():
    class_len = len(cls)
    if class_len > 1:
      num_non_trivial += 1
    if class_len > max_size:
      max_size = class_len
      max_class = cls

  states = list(max_class)
  states.pop(0)
  if num_non_trivial > 0:
    return _reduce(matrix, states, M0)

  return _non_uniform_scale(matrix)


def _ssd_iterate(matrix):
  """Recursively apply SSD_step until convergence."""
  result = _ssd_step(matrix)
  if "stab" in result:
    return result["stab"]
  if "i" in result:
    return result["i"] @ _ssd_iterate(result["mat"])
  return result["D"] @ _ssd_iterate(result["mat"])


def _compute_ssd(matrix):
  """Computes the SSD of a PMM."""
  if _is_sparse_poly_matrix(matrix):
    graph = Graph(_sparse_poly_indicator(matrix), _zero_test)
  else:
    graph = Graph(matrix, _poly_zero_test)
  if len(graph.CommunicatingClasses().keys()) > 1:
    raise ValueError("Input must be unichain.")

  stab = _ssd_iterate(matrix)

  if sp.issparse(stab):
    vec = np.array(stab.toarray(), dtype=float).flatten()
  else:
    vec = np.array(_eval_poly_array(stab, 0), dtype=float)
  return normalize(vec)



def _is_zero_poly_entry(value,
                        tol: float = 1e-12,
                        ignore_infinitesimal: bool = False) -> bool:
  """Check whether a polynomial matrix entry is approx zero."""
  if value is None:
    return True
  if isinstance(value, np.poly1d):
    if ignore_infinitesimal:
      constant_term = float(value(0.0))
      return abs(constant_term) <= tol
    coeffs = np.asarray(value.coeffs, dtype=float)
    return np.all(np.abs(coeffs) <= tol)
  if isinstance(value, Number):
    return abs(value) <= tol
  try:
    coeffs = np.asarray(value, dtype=float)
    if coeffs.size == 0:
      return True
    return np.all(np.abs(coeffs) <= tol)
  except Exception:
    return False


def _matrix_nnz_and_total(matrix: np.ndarray,
                          tol: float = 1e-12,
                          ignore_infinitesimal: bool = False) -> tuple[int, int]:
  """Returns non-zero count and total entries for a polynomial matrix."""
  if matrix is None:
    return 0, 0
  if _is_sparse_poly_matrix(matrix):
    shape = matrix.shape
    if ignore_infinitesimal:
      nnz = 0
      for value in matrix.values():
        if abs(_ensure_poly1d(value)(0.0)) > tol:
          nnz += 1
    else:
      nnz = matrix.nnz()
    return nnz, int(shape[0] * shape[1])
  if sp.issparse(matrix):
    shape = matrix.shape
    return int(matrix.getnnz()), int(shape[0] * shape[1])

  shape = getattr(matrix, "shape", None)
  if shape is None or len(shape) != 2:
    return 0, 0
  total = int(shape[0] * shape[1])

  nnz = 0
  for i, j in np.ndindex(shape):
      entry = matrix[i, j]
      if not _is_zero_poly_entry(entry, tol=tol,
                                ignore_infinitesimal=ignore_infinitesimal):
          nnz += 1
  return nnz, total


# TODO: for code review: Go over this and check.
def _construct_polynomial_transition_matrix(
    base: Union[np.ndarray, sp.spmatrix],
    perturbation_strength: float,
    use_sparse: bool) -> Union[np.ndarray, sp.spmatrix]:
  """Create polynomial transition matrix from a base.
  Here, we use Alpha-Rank's evolutionary dynamics with
  eps = 0 as the base matrix. Then, we add epsilon perturbations
  according to the SSD algorithm and with poly types.
  """
  size = base.shape[0]
  if use_sparse:
    base_sparse = base if sp.issparse(base) else sp.csc_matrix(base)
    matrix = SparsePolyMatrix((size, size))
    for j in range(size):
      noise_target = (j + 1) % size if size > 1 else j
      col_sum = _POLY_ZERO
      processed_rows = set()
      col = base_sparse.getcol(j)
      for idx, row in enumerate(col.indices):
        if row == j:
          continue
        a0 = float(col.data[idx])
        noise_val = 1.0 if row == noise_target else 0.0
        entry = _make_linear_poly(a0,
                                  perturbation_strength * (noise_val - a0))
        if _poly_zero_test(entry):
          continue
        matrix.set(row, j, entry)
        col_sum = col_sum + entry
        processed_rows.add(row)

      if size > 1 and noise_target not in processed_rows and noise_target != j:
        entry = _make_linear_poly(0.0, perturbation_strength)
        if not _poly_zero_test(entry):
          matrix.set(noise_target, j, entry)
          col_sum = col_sum + entry

      matrix.set(j, j, _POLY_ONE - col_sum)

    return matrix

  # Dense case
  matrix = np.zeros((size, size), dtype=object)
  U_val = 1.0 / float(size) if size else 0.0
  for j in range(size):
    for i in range(size):
      a0 = float(base[i, j])
      noise_val = U_val
      a1 = perturbation_strength * (noise_val - a0)
      # base[i, j] + epsilon * (U_val - base[i, j])
      # = (1 - epsilon) * base[i, j] + epsilon * U_val
      entry = _make_linear_poly(a0, a1)
      if isinstance(entry, np.poly1d):
        if _poly_zero_test(entry):
          continue
      elif abs(entry) <= 1e-16:
        continue
      matrix[i, j] = entry

  for j in range(size):
    col_sum = _POLY_ZERO
    for i in range(size):
      if i == j:
        continue
      col_sum = col_sum + _ensure_poly1d(matrix[i, j])
    matrix[j, j] = _POLY_ONE - col_sum

  return matrix


def _construct_perturbed_markov_matrix_ev_dyn(
    payoff_tables: List[Any],
    payoffs_are_hpt_format: bool,
    perturbation_strength: float = 1.0,
    use_sparse: bool = False) -> Union[np.ndarray, sp.spmatrix, SparsePolyMatrix]:
  """Constructs perturbed Markov matrix using evolutionary dynamics."""

  game_info = _convert_alpharank_to_ssd_format(
      payoff_tables, payoffs_are_hpt_format)

  num_populations = game_info["num_populations"]

  m = 10.0
  alpha = 50.0

  if num_populations == 1:
    game_is_constant_sum, payoff_sum = utils.check_is_constant_sum(
        payoff_tables[0], payoffs_are_hpt_format)
    # Use evolutionary dynamics as base with use_inf_alpha False
    c_base, _ = alpharank._get_singlepop_transition_matrix(
        payoff_tables[0], payoffs_are_hpt_format,
        m=m, alpha=alpha,
        game_is_constant_sum=game_is_constant_sum,
        use_local_selection_model=True,
        payoff_sum=payoff_sum,
        use_inf_alpha=False,
        inf_alpha_eps=0.0,
        use_sparse=use_sparse)
    # c_base is a (num_strats x num_strats) matrix with numeric entries.
    base = c_base if sp.issparse(c_base) else np.array(c_base, dtype=float)
  else:
    c_base, _ = alpharank._get_multipop_transition_matrix(
        payoff_tables, payoffs_are_hpt_format,
        m=m, alpha=alpha,
        use_inf_alpha=False,
        inf_alpha_eps=0.0,
        use_sparse=use_sparse)
    base = c_base if sp.issparse(c_base) else np.array(c_base, dtype=float)

  # arank: c[current, next]
  # SSD: M[next, current]
  base = base.transpose() if sp.issparse(base) else base.T
  return _construct_polynomial_transition_matrix(base, perturbation_strength,
                                                 use_sparse)


def compute_ssd(payoff_tables: List[Any],
                perturbation_strength: float = 1,
                verbose: bool = False,
                **kwargs) -> np.ndarray:
  """Compute stochastically stable distribution for given payoff tables.
  
  Args:
    payoff_tables: List of game payoff tables.
    verbose: Set to True to print intermediate results.
    
  Returns:
    Stochastically stable distribution as numpy array.
  """
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)


  num_strats_per_population = utils.get_num_strats_per_population(
      payoff_tables, payoffs_are_hpt_format)
  if np.array_equal(num_strats_per_population,
                    np.ones(len(num_strats_per_population))):
    rhos = np.asarray([[1]])
    return rhos

  if verbose:
    print("Constructing perturbed Markov matrix")
    print('num_strats_per_population:', num_strats_per_population)
  use_sparse = bool(kwargs.get("use_sparse", False))
  matrix = _construct_perturbed_markov_matrix_ev_dyn(
      payoff_tables, payoffs_are_hpt_format, perturbation_strength,
      use_sparse=use_sparse)

  if verbose:
    print(f"Matrix size: {matrix.shape}")
  nnz, total = _matrix_nnz_and_total(
      matrix, ignore_infinitesimal=use_sparse)
  density = (nnz / total) if total else 0.0
  if verbose and use_sparse:
    density_label = "non inf. density" if use_sparse else "density"
    print("SSD matrix sparsity: {}/{} non-zero entries ({:.2f}% {}).".format(
        nnz, total, density * 100.0, density_label))

  ssd_distribution = _compute_ssd(matrix)

  if not _validate_ssd_distribution(ssd_distribution):
    warnings.warn("ssd distribution validation failed, normalizing")
    ssd_distribution = normalize(ssd_distribution)

  return ssd_distribution

