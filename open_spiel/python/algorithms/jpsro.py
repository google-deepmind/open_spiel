# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joint Policy-Space Response Oracles.

An implementation of JSPRO, described in https://arxiv.org/abs/2106.09435.

Bibtex / Cite:

```
@misc{marris2021multiagent,
    title={Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium
           Meta-Solvers},
    author={Luke Marris and Paul Muller and Marc Lanctot and Karl Tuyls and
            Thore Graepel},
    year={2021},
    eprint={2106.09435},
    archivePrefix={arXiv},
    primaryClass={cs.MA}
}
```
"""

import itertools
import string

from absl import logging

import cvxpy as cp
import numpy as np
import scipy as sp

from open_spiel.python import policy
from open_spiel.python.algorithms import projected_replicator_dynamics
from open_spiel.python.egt import alpharank as alpharank_lib
import pyspiel


DEFAULT_ECOS_SOLVER_KWARGS = dict(
    solver="ECOS",
    max_iters=100000000,
    abstol=1e-7,
    reltol=1e-7,
    feastol=1e-7,
    abstol_inacc=1e-7,
    reltol_inacc=1e-7,
    feastol_inacc=1e-7,
    verbose=False,
)
DEFAULT_OSQP_SOLVER_KWARGS = dict(
    solver="OSQP",
    max_iter=1000000000,
    eps_abs=1e-8,
    eps_rel=1e-8,
    eps_prim_inf=1e-8,
    eps_dual_inf=1e-8,
    polish_refine_iter=100,
    check_termination=1000,
    sigma=1e-7,  # Default 1e-6
    delta=1e-7,  # Default 1e-06
    verbose=False,
)
DEFAULT_CVXOPT_SOLVER_KWARGS = dict(
    solver="CVXOPT",
    maxiters=200000,
    abstol=5e-8,
    reltol=5e-8,
    feastol=5e-8,
    refinement=10,
    verbose=False,
)
INIT_POLICIES = (
    "uniform",  # Unopinionated but slower to evaluate.
    "random_deterministic",  # Faster to evaluate but requires samples.
)
UPDATE_PLAYERS_STRATEGY = (
    "all",
    "cycle",
    "random",
)
BRS = (
    "cce",
    "ce",
)
BR_SELECTIONS = (
    "all",  # All policies.
    "all_novel",  # All novel policies.
    "random",  # Random.
    "random_novel",  # Random novel BR (one that has not be considered before).
    "largest_gap",  # The BR with the largest gap.
)
META_SOLVERS = (
    "uni",  # Uniform.
    "undominated_uni",  # Uniform over undominated strategies.
    "rj",  # Random joint.
    "undominated_rj",  # Random joint.
    "rd",  # Random dirichlet.
    "undominated_rd",  # Random dirichlet.
    "prd",  # Prodected replicator dynamics.
    "alpharank",  # Alpha-Rank
    "mgce",  # Maximum gini CE.
    "min_epsilon_mgce",  # Min Epsilon Maximum gini CE.
    "approx_mgce",  # Approximate Maximum gini CE.
    "rmwce",  # Random maximum welfare CE.
    "mwce",  # Maximum welfare CE.
    "rvce",  # Random vertex CE.
    "mgcce",  # Maximum gini CCE.
    "min_epsilon_mgcce",  # Min Epsilon Maximum gini CCE.
    "approx_mgcce",  # Approximate Maximum gini CE.
    "rmwcce",  # Random maximum welfare CCE.
    "mwcce",  # Maximum welfare CCE.
    "rvcce",  # Random vertex CCE.
)
LOG_STRING = """
Iteration {iteration: 6d}
=== ({game})
Player            {player}
BRs               {brs}
Num Policies      {num_policies}
Unique Policies   {unique}
--- ({train_meta_solver})
Train Value       {train_value}
Train Gap         {train_gap}
--- ({eval_meta_solver})
Eval Value        {eval_value}
Eval Gap          {eval_gap}
"""
DIST_TOL = 1e-8
GAP_TOL = 1e-8
RETURN_TOL = 1e-12


## Meta Solvers.


# Helper Functions - Dominated strategy elimination.
def _eliminate_dominated_payoff(
    payoff, epsilon, action_labels=None, action_repeats=None, weakly=False):
  """Eliminate epsilon dominated strategies."""
  num_players = payoff.shape[0]
  eliminated = True
  if action_labels is None:
    action_labels = [np.arange(na, dtype=np.int32) for na in payoff.shape[1:]]
  if action_repeats is not None:
    action_repeats = [ar for ar in action_repeats]
  while eliminated:
    eliminated = False
    for p in range(num_players):
      if epsilon > 0.0:
        continue
      num_actions = payoff.shape[1:]
      if num_actions[p] <= 1:
        continue
      for a in range(num_actions[p]):
        index = [slice(None) for _ in range(num_players)]
        index[p] = slice(a, a+1)
        if weakly:
          diff = payoff[p] <= payoff[p][tuple(index)]
        else:
          diff = payoff[p] < payoff[p][tuple(index)]
        axis = tuple(range(p)) + tuple(range(p+1, num_players))
        less = np.all(diff, axis=axis)
        less[a] = False  # Action cannot eliminate itself.
        if np.any(less):
          nonzero = np.nonzero(less)
          payoff = np.delete(payoff, nonzero, axis=p+1)
          action_labels[p] = np.delete(action_labels[p], nonzero)
          if action_repeats is not None:
            action_repeats[p] = np.delete(action_repeats[p], nonzero)
          eliminated = True
          break
  return payoff, action_labels, action_repeats


def _reconstruct_dist(eliminated_dist, action_labels, num_actions):
  """Returns reconstructed dist from eliminated_dist and action_labels.

  Redundant dist elements are given values 0.

  Args:
    eliminated_dist: Array of shape [A0E, A1E, ...].
    action_labels: List of length N and shapes [[A0E], [A1E], ...].
    num_actions: List of length N and values [A0, A1, ...].

  Returns:
    reconstructed_dist: Array of shape [A0, A1, ...].
  """
  reconstructed_payoff = np.zeros(num_actions)
  reconstructed_payoff[np.ix_(*action_labels)] = eliminated_dist
  return reconstructed_payoff


def _eliminate_dominated_decorator(func):
  """Wrap eliminate dominated."""
  def wrapper(payoff, per_player_repeats, *args, eliminate_dominated=True,
              **kwargs):
    epsilon = getattr(kwargs, "epsilon", 0.0)
    if not eliminate_dominated:
      return func(payoff, *args, **kwargs)
    num_actions = payoff.shape[1:]
    eliminated_payoff, action_labels, eliminated_action_repeats = _eliminate_dominated_payoff(
        payoff, epsilon, action_repeats=per_player_repeats)
    eliminated_dist, meta = func(
        eliminated_payoff, eliminated_action_repeats, *args, **kwargs)
    meta["eliminated_dominated_dist"] = eliminated_dist
    meta["eliminated_dominated_payoff"] = eliminated_payoff
    dist = _reconstruct_dist(
        eliminated_dist, action_labels, num_actions)
    return dist, meta
  return wrapper


# Optimization.
def _try_two_solvers(func, *args, **kwargs):
  try:
    logging.debug("Trying CVXOPT.", flush=True)
    kwargs_ = {"solver_kwargs": DEFAULT_CVXOPT_SOLVER_KWARGS, **kwargs}
    res = func(*args, **kwargs_)
  except:  # pylint: disable=bare-except
    logging.debug("CVXOPT failed. Trying OSQP.", flush=True)
    kwargs_ = {"solver_kwargs": DEFAULT_OSQP_SOLVER_KWARGS, **kwargs}
    res = func(*args, **kwargs_)
  return res


# Helper Functions - CCEs.
def _indices(p, a, num_players):
  return [a if p_ == p else slice(None) for p_ in range(num_players)]


def _sparse_indices_generator(player, action, num_actions):
  indices = [(action,) if p == player else range(na)
             for p, na in enumerate(num_actions)]
  return itertools.product(*indices)


def _partition_by_player(val, p_vec, num_players):
  """Partitions a value by the players vector."""
  parts = []
  for p in range(num_players):
    inds = p_vec == p
    if inds.size > 0:
      parts.append(val[inds])
    else:
      parts.append(None)
  return parts


def _cce_constraints(payoff, epsilons, remove_null=True, zero_tolerance=1e-8):
  """Returns the coarse correlated constraints.

  Args:
    payoff: A [NUM_PLAYER, NUM_ACT_0, NUM_ACT_1, ...] shape payoff tensor.
    epsilons: Per player floats corresponding to the epsilon.
    remove_null: Remove null rows of the constraint matrix.
    zero_tolerance: Zero out elements with small value.

  Returns:
    a_mat: The gain matrix for deviting to an action or shape [SUM(A), PROD(A)].
    meta: Dictionary containing meta information.
  """
  num_players = payoff.shape[0]
  num_actions = payoff.shape[1:]
  num_dists = int(np.prod(num_actions))

  cor_cons = int(np.sum(num_actions))

  a_mat = np.zeros([cor_cons] + list(num_actions))
  p_vec = np.zeros([cor_cons], dtype=np.int32)
  i_vec = np.zeros([cor_cons], dtype=np.int32)
  con = 0
  for p in range(num_players):
    for a1 in range(num_actions[p]):
      a1_inds = _indices(p, a1, num_players)
      for a0 in range(num_actions[p]):
        a0_inds = _indices(p, a0, num_players)
        a_mat[con][a0_inds] += payoff[p][a1_inds]
      a_mat[con] -= payoff[p]
      a_mat[con] -= epsilons[p]

      p_vec[con] = p
      i_vec[con] = a0

      con += 1

  a_mat = np.reshape(a_mat, [cor_cons, num_dists])
  a_mat[np.abs(a_mat) < zero_tolerance] = 0.0
  if remove_null:
    null_cons = np.any(a_mat != 0.0, axis=-1)
    redundant_cons = np.max(a_mat, axis=1) >= 0
    nonzero_mask = null_cons & redundant_cons
    a_mat = a_mat[nonzero_mask, :].copy()
    p_vec = p_vec[nonzero_mask].copy()
    i_vec = i_vec[nonzero_mask].copy()

  meta = dict(
      p_vec=p_vec,
      i_vec=i_vec,
      epsilons=epsilons,
  )

  return a_mat, meta


def _ace_constraints(payoff, epsilons, remove_null=True, zero_tolerance=0.0):
  """Returns sparse alternate ce constraints Ax - epsilon <= 0.

  Args:
    payoff: Dense payoff tensor.
    epsilons: Scalar epsilon approximation.
    remove_null: Whether to remove null row constraints.
    zero_tolerance: Smallest absolute value.

  Returns:
    a_csr: Sparse gain matrix from switching from one action to another.
    e_vec: Epsilon vector.
    meta: Dictionary containing meta information.
  """
  num_players = payoff.shape[0]
  num_actions = payoff.shape[1:]
  num_dists = int(np.prod(num_actions))

  num_cons = 0
  for p in range(num_players):
    num_cons += num_actions[p] * (num_actions[p] - 1)

  a_dok = sp.sparse.dok_matrix((num_cons, num_dists))
  e_vec = np.zeros([num_cons])
  p_vec = np.zeros([num_cons], dtype=np.int32)
  i_vec = np.zeros([num_cons, 2], dtype=np.int32)

  num_null_cons = None
  num_redundant_cons = None
  num_removed_cons = None

  if num_cons > 0:
    con = 0
    for p in range(num_players):
      generator = itertools.permutations(range(num_actions[p]), 2)
      for a0, a1 in generator:
        a0_inds = _sparse_indices_generator(p, a0, num_actions)
        a1_inds = _sparse_indices_generator(p, a1, num_actions)

        for a0_ind, a1_ind in zip(a0_inds, a1_inds):
          a0_ind_flat = np.ravel_multi_index(a0_ind, num_actions)
          val = payoff[p][a1_ind] - payoff[p][a0_ind]
          if abs(val) > zero_tolerance:
            a_dok[con, a0_ind_flat] = val

        e_vec[con] = epsilons[p]
        p_vec[con] = p
        i_vec[con] = [a0, a1]
        con += 1

    a_csr = a_dok.tocsr()
    if remove_null:
      null_cons = np.logical_or(
          a_csr.max(axis=1).todense() != 0.0,
          a_csr.min(axis=1).todense() != 0.0)
      null_cons = np.ravel(null_cons)
      redundant_cons = np.ravel(a_csr.max(axis=1).todense()) >= e_vec
      nonzero_mask = null_cons & redundant_cons
      a_csr = a_csr[nonzero_mask, :]
      e_vec = e_vec[nonzero_mask].copy()
      p_vec = p_vec[nonzero_mask].copy()
      i_vec = i_vec[nonzero_mask].copy()
      num_null_cons = np.sum(~null_cons)
      num_redundant_cons = np.sum(~redundant_cons)
      num_removed_cons = np.sum(~nonzero_mask)

  else:
    a_csr = a_dok.tocsr()

  meta = dict(
      p_vec=p_vec,
      i_vec=i_vec,
      epsilons=epsilons,
      num_null_cons=num_null_cons,
      num_redundant_cons=num_redundant_cons,
      num_removed_cons=num_removed_cons,
  )

  return a_csr, e_vec, meta


def _get_repeat_factor(action_repeats):
  """Returns the repeat factors for the game."""
  num_players = len(action_repeats)
  out_labels = string.ascii_lowercase[:len(action_repeats)]
  in_labels = ",".join(out_labels)
  repeat_factor = np.ravel(np.einsum(
      "{}->{}".format(in_labels, out_labels), *action_repeats))
  indiv_repeat_factors = []
  for player in range(num_players):
    action_repeats_ = [
        np.ones_like(ar) if player == p else ar
        for p, ar in enumerate(action_repeats)]
    indiv_repeat_factor = np.ravel(np.einsum(
        "{}->{}".format(in_labels, out_labels), *action_repeats_))
    indiv_repeat_factors.append(indiv_repeat_factor)
  return repeat_factor, indiv_repeat_factors


# Solvers.
def _linear(
    payoff,
    a_mat,
    e_vec,
    action_repeats=None,
    solver_kwargs=None,
    cost=None):
  """Returns linear solution.

  This is a linear program.

  Args:
    payoff: A [NUM_PLAYER, NUM_ACT_0, NUM_ACT_1, ...] shape payoff tensor.
    a_mat: Constaint matrix.
    e_vec: Epsilon vector.
    action_repeats: List of action repeat counts.
    solver_kwargs: Solver kwargs.
    cost: Cost function of same shape as payoff.

  Returns:
    An epsilon-correlated equilibrium.
  """
  num_players = payoff.shape[0]
  num_actions = payoff.shape[1:]
  num_dists = int(np.prod(num_actions))

  if solver_kwargs is None:
    solver_kwargs = DEFAULT_ECOS_SOLVER_KWARGS

  if a_mat.shape[0] > 0:
    # Variables.
    x = cp.Variable(num_dists, nonneg=True)

    # Classifier.
    epsilon_dists = cp.matmul(a_mat, x) - e_vec

    # Constraints.
    dist_eq_con = cp.sum(x) == 1
    cor_lb_con = epsilon_dists <= 0

    # Objective.
    if cost is None:
      player_totals = [
          cp.sum(cp.multiply(payoff[p].flat, x)) for p in range(num_players)]
      reward = cp.sum(player_totals)
    else:
      reward = cp.sum(cp.multiply(cost.flat, x))
    obj = cp.Maximize(reward)

    prob = cp.Problem(obj, [
        dist_eq_con,
        cor_lb_con,
    ])

    # Solve.
    prob.solve(**solver_kwargs)
    status = prob.status

    # Distribution.
    dist = np.reshape(x.value, num_actions)

    # Other.
    val = reward.value
  else:
    if action_repeats is not None:
      repeat_factor, _ = _get_repeat_factor(action_repeats)
      x = repeat_factor / np.sum(repeat_factor)
    else:
      x = np.ones([num_dists]) / num_dists
    val = 0.0  # Fix me.
    dist = np.reshape(x, num_actions)
    status = None

  meta = dict(
      x=x,
      a_mat=a_mat,
      val=val,
      status=status,
      payoff=payoff,
      consistent=True,
      unique=False,
  )

  return dist, meta


def _qp_cce(
    payoff,
    a_mats,
    e_vecs,
    assume_full_support=False,
    action_repeats=None,
    solver_kwargs=None,
    min_epsilon=False):
  """Returns the correlated equilibrium with maximum Gini impurity.

  Args:
    payoff: A [NUM_PLAYER, NUM_ACT_0, NUM_ACT_1, ...] shape payoff tensor.
    a_mats: A [NUM_CON, PROD(A)] shape gain tensor.
    e_vecs: Epsilon vector.
    assume_full_support: Whether to ignore beta values.
    action_repeats: Vector of action repeats for each player.
    solver_kwargs: Additional kwargs for solver.
    min_epsilon: Whether to minimize epsilon.

  Returns:
    An epsilon-correlated equilibrium.
  """
  num_players = payoff.shape[0]
  num_actions = payoff.shape[1:]
  num_dists = int(np.prod(num_actions))

  if solver_kwargs is None:
    solver_kwargs = DEFAULT_OSQP_SOLVER_KWARGS

  epsilon = None
  nonzero_cons = [a_mat.shape[0] > 0 for a_mat in a_mats if a_mat is not None]
  if any(nonzero_cons):
    x = cp.Variable(num_dists, nonneg=(not assume_full_support))
    if min_epsilon:
      epsilon = cp.Variable(nonpos=True)
      e_vecs = [epsilon] * num_players

    if action_repeats is not None:
      repeat_factor, _ = _get_repeat_factor(action_repeats)
      x_repeated = cp.multiply(x, repeat_factor)
      dist_eq_con = cp.sum(x_repeated) == 1
      cor_lb_cons = [
          cp.matmul(a_mat, cp.multiply(x, repeat_factor)) <= e_vec
          for a_mat, e_vec in
          zip(a_mats, e_vecs) if a_mat.size > 0]
      eye = sp.sparse.diags(repeat_factor)
    else:
      repeat_factor = 1
      x_repeated = x
      dist_eq_con = cp.sum(x_repeated) == 1
      cor_lb_cons = [
          cp.matmul(a_mat, x) <= e_vec for a_mat, e_vec in
          zip(a_mats, e_vecs) if a_mat.size > 0]
      eye = sp.sparse.eye(num_dists)

    # This is more memory efficient than using cp.sum_squares.
    cost = 1 - cp.quad_form(x, eye)
    if min_epsilon:
      cost -= cp.multiply(2, epsilon)

    obj = cp.Maximize(cost)
    prob = cp.Problem(obj, [dist_eq_con] + cor_lb_cons)
    cost_value = prob.solve(**solver_kwargs)
    status = prob.status
    alphas = [cor_lb_con.dual_value for cor_lb_con in cor_lb_cons]
    lamb = dist_eq_con.dual_value

    val = cost.value
    x = x_repeated.value
    dist = np.reshape(x, num_actions)
  else:
    cost_value = 0.0
    val = 1 - 1 / num_dists
    if action_repeats is not None:
      repeat_factor, _ = _get_repeat_factor(action_repeats)
      x = repeat_factor / np.sum(repeat_factor)
    else:
      x = np.ones([num_dists]) / num_dists
    dist = np.reshape(x, num_actions)
    status = None
    alphas = [np.zeros([])]
    lamb = None

  meta = dict(
      x=x,
      a_mats=a_mats,
      status=status,
      cost=cost_value,
      val=val,
      alphas=alphas,
      lamb=lamb,
      unique=True,
      min_epsilon=None if epsilon is None else epsilon.value,
  )
  return dist, meta


def _qp_ce(
    payoff,
    a_mats,
    e_vecs,
    assume_full_support=False,
    action_repeats=None,
    solver_kwargs=None,
    min_epsilon=False):
  """Returns the correlated equilibrium with maximum Gini impurity.

  Args:
    payoff: A [NUM_PLAYER, NUM_ACT_0, NUM_ACT_1, ...] shape payoff tensor.
    a_mats: A [NUM_CON, PROD(A)] shape gain tensor.
    e_vecs: Epsilon vector.
    assume_full_support: Whether to ignore beta values.
    action_repeats: Vector of action repeats for each player.
    solver_kwargs: Additional kwargs for solver.
    min_epsilon: Whether to minimize epsilon.

  Returns:
    An epsilon-correlated equilibrium.
  """
  num_players = payoff.shape[0]
  num_actions = payoff.shape[1:]
  num_dists = int(np.prod(num_actions))

  if solver_kwargs is None:
    solver_kwargs = DEFAULT_OSQP_SOLVER_KWARGS

  epsilon = None
  nonzero_cons = [a_mat.shape[0] > 0 for a_mat in a_mats if a_mat is not None]
  if any(nonzero_cons):
    x = cp.Variable(num_dists, nonneg=(not assume_full_support))
    if min_epsilon:
      epsilon = cp.Variable(nonpos=True)
      e_vecs = [epsilon] * num_players

    if action_repeats is not None:
      repeat_factor, indiv_repeat_factors = _get_repeat_factor(
          action_repeats)
      x_repeated = cp.multiply(x, repeat_factor)
      dist_eq_con = cp.sum(x_repeated) == 1
      cor_lb_cons = [
          cp.matmul(a_mat, cp.multiply(x, rf)) <= e_vec for a_mat, e_vec, rf in
          zip(a_mats, e_vecs, indiv_repeat_factors) if a_mat.size > 0]
      eye = sp.sparse.diags(repeat_factor)
    else:
      repeat_factor = 1
      x_repeated = x
      dist_eq_con = cp.sum(x_repeated) == 1
      cor_lb_cons = [
          cp.matmul(a_mat, x) <= e_vec for a_mat, e_vec in
          zip(a_mats, e_vecs) if a_mat.size > 0]
      eye = sp.sparse.eye(num_dists)

    # This is more memory efficient than using cp.sum_squares.
    cost = 1 - cp.quad_form(x, eye)
    if min_epsilon:
      cost -= cp.multiply(2, epsilon)

    obj = cp.Maximize(cost)
    prob = cp.Problem(obj, [dist_eq_con] + cor_lb_cons)
    cost_value = prob.solve(**solver_kwargs)
    status = prob.status
    alphas = [cor_lb_con.dual_value for cor_lb_con in cor_lb_cons]
    lamb = dist_eq_con.dual_value

    val = cost.value
    x = x_repeated.value
    dist = np.reshape(x, num_actions)
  else:
    cost_value = 0.0
    val = 1 - 1 / num_dists
    if action_repeats is not None:
      repeat_factor, indiv_repeat_factors = _get_repeat_factor(
          action_repeats)
      x = repeat_factor / np.sum(repeat_factor)
    else:
      x = np.ones([num_dists]) / num_dists
    dist = np.reshape(x, num_actions)
    status = None
    alphas = [np.zeros([])]
    lamb = None

  meta = dict(
      x=x,
      a_mats=a_mats,
      status=status,
      cost=cost_value,
      val=val,
      alphas=alphas,
      lamb=lamb,
      unique=True,
      min_epsilon=None if epsilon is None else epsilon.value,
  )
  return dist, meta


def _expand_meta_game(meta_game, per_player_repeats):
  num_players = meta_game.shape[0]
  for player in range(num_players):
    meta_game = np.repeat(meta_game, per_player_repeats[player], axis=player+1)
  return meta_game


def _unexpand_meta_dist(meta_dist, per_player_repeats):
  num_players = len(meta_dist.shape)
  for player in range(num_players):
    meta_dist = np.add.reduceat(
        meta_dist, [0] + np.cumsum(per_player_repeats[player]).tolist()[:-1],
        axis=player)
  return meta_dist


# Meta-solvers - Baselines.
def _uni(meta_game, per_player_repeats, ignore_repeats=False):
  """Uniform."""
  if ignore_repeats:
    num_policies = meta_game.shape[1:]
    num_dists = np.prod(num_policies)
    meta_dist = np.full(num_policies, 1./num_dists)
  else:
    outs = [ppr / np.sum(ppr) for ppr in per_player_repeats]
    labels = string.ascii_lowercase[:len(outs)]
    comma_labels = ",".join(labels)
    meta_dist = np.einsum("{}->{}".format(comma_labels, labels), *outs)
  return meta_dist, dict()


@_eliminate_dominated_decorator
def _undominated_uni(meta_game, per_player_repeats, ignore_repeats=False):
  """Undominated uniform."""
  return _uni(meta_game, per_player_repeats, ignore_repeats=ignore_repeats)


def _rj(meta_game, per_player_repeats, ignore_repeats=False):
  """Random joint."""
  ignore_repeats = True
  pvals, _ = _uni(
      meta_game, per_player_repeats, ignore_repeats=ignore_repeats)
  meta_dist = np.reshape(
      np.random.multinomial(1, pvals.flat), pvals.shape).astype(np.float64)
  return meta_dist, dict()


@_eliminate_dominated_decorator
def _undominated_rj(meta_game, per_player_repeats, ignore_repeats=False):
  """Undominated random joint."""
  return _rj(meta_game, per_player_repeats, ignore_repeats=ignore_repeats)


def _rd(meta_game, per_player_repeats, ignore_repeats=False):
  """Random dirichlet."""
  ignore_repeats = True
  if ignore_repeats:
    num_policies = meta_game.shape[1:]
    alpha = np.ones(num_policies)
  else:
    outs = [ppr for ppr in per_player_repeats]
    labels = string.ascii_lowercase[:len(outs)]
    comma_labels = ",".join(labels)
    alpha = np.einsum("{}->{}".format(comma_labels, labels), *outs)
  meta_dist = np.reshape(
      np.random.dirichlet(alpha.flat), alpha.shape).astype(np.float64)
  return meta_dist, dict()


@_eliminate_dominated_decorator
def _undominated_rd(meta_game, per_player_repeats, ignore_repeats=False):
  """Undominated random dirichlet."""
  return _rd(meta_game, per_player_repeats, ignore_repeats=ignore_repeats)


def _prd(meta_game, per_player_repeats, ignore_repeats=False):
  """Projected replicator dynamics."""
  if not ignore_repeats:
    meta_game = _expand_meta_game(meta_game, per_player_repeats)
  meta_dist = projected_replicator_dynamics.projected_replicator_dynamics(
      meta_game)
  labels = string.ascii_lowercase[:len(meta_dist)]
  comma_labels = ",".join(labels)
  meta_dist = np.einsum("{}->{}".format(comma_labels, labels), *meta_dist)
  meta_dist[meta_dist < DIST_TOL] = 0.0
  meta_dist /= np.sum(meta_dist)
  meta_dist = _unexpand_meta_dist(meta_dist, per_player_repeats)
  return meta_dist, dict()


@_eliminate_dominated_decorator
def _alpharank(meta_game, per_player_repeats, ignore_repeats=False):
  """AlphaRank."""
  if not ignore_repeats:
    meta_game = _expand_meta_game(meta_game, per_player_repeats)
  meta_dist = alpharank_lib.sweep_pi_vs_epsilon([mg for mg in meta_game])
  meta_dist[meta_dist < DIST_TOL] = 0.0
  meta_dist /= np.sum(meta_dist)
  meta_dist = np.reshape(meta_dist, meta_game.shape[1:])
  if not ignore_repeats:
    meta_dist = _unexpand_meta_dist(meta_dist, per_player_repeats)
  return meta_dist, dict()


# Meta-solvers - CEs.
@_eliminate_dominated_decorator
def _mgce(meta_game, per_player_repeats, ignore_repeats=False):
  """Maximum Gini CE."""
  a_mat, e_vec, meta = _ace_constraints(
      meta_game, [0.0] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  a_mats = _partition_by_player(
      a_mat, meta["p_vec"], len(per_player_repeats))
  e_vecs = _partition_by_player(
      e_vec, meta["p_vec"], len(per_player_repeats))
  dist, _ = _try_two_solvers(
      _qp_ce,
      meta_game, a_mats, e_vecs,
      action_repeats=(None if ignore_repeats else per_player_repeats))
  return dist, dict()


@_eliminate_dominated_decorator
def _min_epsilon_mgce(meta_game, per_player_repeats, ignore_repeats=False):
  """Min Epsilon Maximum Gini CE."""
  a_mat, e_vec, meta = _ace_constraints(
      meta_game, [0.0] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  a_mats = _partition_by_player(
      a_mat, meta["p_vec"], len(per_player_repeats))
  e_vecs = _partition_by_player(
      e_vec, meta["p_vec"], len(per_player_repeats))
  dist, _ = _try_two_solvers(
      _qp_ce,
      meta_game, a_mats, e_vecs,
      action_repeats=(None if ignore_repeats else per_player_repeats),
      min_epsilon=True)
  return dist, dict()


@_eliminate_dominated_decorator
def _approx_mgce(meta_game, per_player_repeats, ignore_repeats=False,
                 epsilon=0.01):
  """Approximate Maximum Gini CE."""
  a_mat, e_vec, meta = _ace_constraints(
      meta_game, [0.0] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  max_ab = 0.0
  if a_mat.size:
    max_ab = np.max(a_mat.mean(axis=1))
  a_mat, e_vec, meta = _ace_constraints(
      meta_game, [epsilon * max_ab] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  a_mats = _partition_by_player(
      a_mat, meta["p_vec"], len(per_player_repeats))
  e_vecs = _partition_by_player(
      e_vec, meta["p_vec"], len(per_player_repeats))
  dist, _ = _try_two_solvers(
      _qp_ce,
      meta_game, a_mats, e_vecs,
      action_repeats=(None if ignore_repeats else per_player_repeats))
  return dist, dict()


@_eliminate_dominated_decorator
def _rmwce(meta_game, per_player_repeats, ignore_repeats=False):
  """Random maximum welfare CE."""
  del ignore_repeats
  num_players = len(per_player_repeats)
  cost = np.ravel(np.sum(meta_game, axis=0))
  cost += np.ravel(np.random.normal(size=meta_game.shape[1:])) * 1e-6
  a_mat, e_vec, _ = _ace_constraints(
      meta_game, [0.0] * num_players, remove_null=True,
      zero_tolerance=1e-8)
  x, _ = _linear(meta_game, a_mat, e_vec, cost=cost)
  dist = np.reshape(x, meta_game.shape[1:])
  return dist, dict()


@_eliminate_dominated_decorator
def _mwce(meta_game, per_player_repeats, ignore_repeats=False):
  """Maximum welfare CE."""
  del ignore_repeats
  num_players = len(per_player_repeats)
  cost = np.ravel(np.sum(meta_game, axis=0))
  a_mat, e_vec, _ = _ace_constraints(
      meta_game, [0.0] * num_players, remove_null=True,
      zero_tolerance=1e-8)
  x, _ = _linear(meta_game, a_mat, e_vec, cost=cost)
  dist = np.reshape(x, meta_game.shape[1:])
  return dist, dict()


@_eliminate_dominated_decorator
def _rvce(meta_game, per_player_repeats, ignore_repeats=False):
  """Random vertex CE."""
  del ignore_repeats
  num_players = len(per_player_repeats)
  cost = np.ravel(np.random.normal(size=meta_game.shape[1:]))
  a_mat, e_vec, _ = _ace_constraints(
      meta_game, [0.0] * num_players, remove_null=True,
      zero_tolerance=1e-8)
  x, _ = _linear(meta_game, a_mat, e_vec, cost=cost)
  dist = np.reshape(x, meta_game.shape[1:])
  return dist, dict()


# Meta-solvers - CCEs.
def _mgcce(meta_game, per_player_repeats, ignore_repeats=False):
  """Maximum Gini CCE."""
  a_mat, meta = _cce_constraints(
      meta_game, [0.0] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  a_mats = _partition_by_player(
      a_mat, meta["p_vec"], len(per_player_repeats))
  dist, _ = _try_two_solvers(
      _qp_cce,
      meta_game, a_mats, [0.0] * len(per_player_repeats),
      action_repeats=(None if ignore_repeats else per_player_repeats))
  return dist, dict()


def _min_epsilon_mgcce(meta_game, per_player_repeats, ignore_repeats=False):
  """Min Epsilon Maximum Gini CCE."""
  a_mat, meta = _cce_constraints(
      meta_game, [0.0] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  a_mats = _partition_by_player(
      a_mat, meta["p_vec"], len(per_player_repeats))
  dist, _ = _try_two_solvers(
      _qp_cce,
      meta_game, a_mats, [0.0] * len(per_player_repeats),
      action_repeats=(None if ignore_repeats else per_player_repeats),
      min_epsilon=True)
  return dist, dict()


def _approx_mgcce(meta_game, per_player_repeats, ignore_repeats=False,
                  epsilon=0.01):
  """Maximum Gini CCE."""
  a_mat, meta = _cce_constraints(
      meta_game, [0.0] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  max_ab = 0.0
  if a_mat.size:
    max_ab = np.max(a_mat.mean(axis=1))
  a_mat, meta = _cce_constraints(
      meta_game, [epsilon * max_ab] * len(per_player_repeats), remove_null=True,
      zero_tolerance=1e-8)
  a_mats = _partition_by_player(
      a_mat, meta["p_vec"], len(per_player_repeats))
  dist, _ = _try_two_solvers(
      _qp_cce,
      meta_game, a_mats, [0.0] * len(per_player_repeats),
      action_repeats=(None if ignore_repeats else per_player_repeats))
  return dist, dict()


def _rmwcce(meta_game, per_player_repeats, ignore_repeats=False):
  """Random maximum welfare CCE."""
  del ignore_repeats
  num_players = len(per_player_repeats)
  cost = np.ravel(np.sum(meta_game, axis=0))
  cost += np.ravel(np.random.normal(size=meta_game.shape[1:])) * 1e-6
  a_mat, _ = _cce_constraints(
      meta_game, [0.0] * num_players, remove_null=True,
      zero_tolerance=1e-8)
  e_vec = np.zeros([a_mat.shape[0]])
  x, _ = _linear(meta_game, a_mat, e_vec, cost=cost)
  dist = np.reshape(x, meta_game.shape[1:])
  return dist, dict()


def _mwcce(meta_game, per_player_repeats, ignore_repeats=False):
  """Maximum welfare CCE."""
  del ignore_repeats
  num_players = len(per_player_repeats)
  cost = np.ravel(np.sum(meta_game, axis=0))
  a_mat, _ = _cce_constraints(
      meta_game, [0.0] * num_players, remove_null=True,
      zero_tolerance=1e-8)
  e_vec = np.zeros([a_mat.shape[0]])
  x, _ = _linear(meta_game, a_mat, e_vec, cost=cost)
  dist = np.reshape(x, meta_game.shape[1:])
  return dist, dict()


def _rvcce(meta_game, per_player_repeats, ignore_repeats=False):
  """Random vertex CCE."""
  del ignore_repeats
  num_players = len(per_player_repeats)
  cost = np.ravel(np.random.normal(size=meta_game.shape[1:]))
  a_mat, _ = _cce_constraints(
      meta_game, [0.0] * num_players, remove_null=True,
      zero_tolerance=1e-8)
  e_vec = np.zeros([a_mat.shape[0]])
  x, _ = _linear(meta_game, a_mat, e_vec, cost=cost)
  dist = np.reshape(x, meta_game.shape[1:])
  return dist, dict()


# Flags to functions.
_FLAG_TO_FUNC = dict(
    uni=_uni,
    undominated_uni=_undominated_uni,
    rj=_rj,
    undominated_rj=_undominated_rj,
    rd=_rd,
    undominated_rd=_undominated_rd,
    prd=_prd,
    alpharank=_alpharank,
    mgce=_mgce,
    min_epsilon_mgce=_min_epsilon_mgce,
    approx_mgce=_approx_mgce,
    rmwce=_rmwce,
    mwce=_mwce,
    rvce=_rvce,
    mgcce=_mgcce,
    min_epsilon_mgcce=_min_epsilon_mgcce,
    approx_mgcce=_approx_mgcce,
    rmwcce=_rmwcce,
    mwcce=_mwcce,
    rvcce=_rvcce,
)


## PSRO Functions.


def intilize_policy(game, player, policy_init):
  """Returns initial policy."""
  if policy_init == "uniform":
    new_policy = policy.TabularPolicy(game, players=(player,))

  elif policy_init == "random_deterministic":
    new_policy = policy.TabularPolicy(game, players=(player,))
    for i in range(new_policy.action_probability_array.shape[0]):
      new_policy.action_probability_array[i] = np.random.multinomial(
          1, new_policy.action_probability_array[i]).astype(np.float64)

  else:
    raise ValueError(
        "policy_init must be a valid initialization strategy: %s. "
        "Received: %s" % (INIT_POLICIES, policy_init))

  return new_policy


def add_new_policies(
    per_player_new_policies,
    per_player_gaps,
    per_player_repeats,
    per_player_policies,
    joint_policies,
    joint_returns,
    game,
    br_selection):
  """Adds novel policies from new policies."""
  num_players = len(per_player_new_policies)
  per_player_num_novel_policies = [0 for _ in range(num_players)]

  # Update policies and policy counts.
  for player in range(num_players):
    new_policies = per_player_new_policies[player]
    new_gaps = per_player_gaps[player]

    repeat_policies = []
    repeat_gaps = []
    repeat_ids = []
    novel_policies = []
    novel_gaps = []
    for new_policy, new_gap in zip(new_policies, new_gaps):
      for policy_id, policy_ in enumerate(per_player_policies[player]):
        if np.all(  # New policy is not novel.
            new_policy.action_probability_array ==
            policy_.action_probability_array):  # pytype: disable=attribute-error  # py39-upgrade
          logging.debug("Player %d's new policy is not novel.", player)
          repeat_policies.append(new_policy)
          repeat_gaps.append(new_gap)
          repeat_ids.append(policy_id)
          break
      else:  # New policy is novel.
        logging.debug("Player %d's new policy is novel.", player)
        novel_policies.append(new_policy)
        novel_gaps.append(new_gap)

    add_novel_policies = []
    add_repeat_ids = []
    if (novel_policies or repeat_policies):
      if br_selection == "all":
        add_novel_policies.extend(novel_policies)
        add_repeat_ids.extend(repeat_ids)
      elif br_selection == "all_novel":
        add_novel_policies.extend(novel_policies)
      elif br_selection == "random":
        index = np.random.randint(0, len(repeat_policies) + len(novel_policies))
        if index < len(novel_policies):
          add_novel_policies.append(novel_policies[index])
        else:
          add_repeat_ids.append(repeat_ids[index - len(novel_policies)])
      elif br_selection == "random_novel":
        if novel_policies:
          index = np.random.randint(0, len(novel_policies))
          add_novel_policies.append(novel_policies[index])
        else:  # Fall back on random.
          index = np.random.randint(0, len(repeat_policies))
          add_repeat_ids.append(repeat_ids[index])
      elif br_selection == "largest_gap":
        if novel_policies:
          index = np.argmax(novel_gaps)
          if novel_gaps[index] == 0.0:  # Fall back to random when zero.
            index = np.random.randint(0, len(novel_policies))
          add_novel_policies.append(novel_policies[index])
        else:  # Fall back on random.
          index = np.random.randint(0, len(repeat_policies))
          add_repeat_ids.append(repeat_ids[index])
      else:
        raise ValueError("Unrecognized br_selection method: %s"
                         % br_selection)

    for add_repeat_id in add_repeat_ids:
      per_player_repeats[player][add_repeat_id] += 1

    for add_novel_policy in add_novel_policies:
      per_player_policies[player].append(add_novel_policy)  # Add new policy.
      per_player_repeats[player].append(1)  # Add new count.
      per_player_num_novel_policies[player] += 1

  # Add new joint policies.
  for pids in itertools.product(*[
      range(len(policies)) for policies in per_player_policies]):
    if pids in joint_policies:
      continue
    logging.debug("Evaluating novel joint policy: %s.", pids)
    policies = [
        policies[pid] for pid, policies in zip(pids, per_player_policies)]
    python_tabular_policy = policy.merge_tabular_policies(
        policies, game)
    pyspiel_tabular_policy = policy.python_policy_to_pyspiel_policy(
        python_tabular_policy)
    joint_policies[pids] = pyspiel_tabular_policy
    joint_returns[pids] = [
        0.0 if abs(er) < RETURN_TOL else er
        for er in pyspiel.expected_returns(
            game.new_initial_state(), pyspiel_tabular_policy, -1, True)]

  return per_player_num_novel_policies


def add_meta_game(
    meta_games,
    per_player_policies,
    joint_returns):
  """Returns a meta-game tensor."""
  per_player_num_policies = [
      len(policies) for policies in per_player_policies]
  shape = [len(per_player_num_policies)] + per_player_num_policies
  meta_game = np.zeros(shape)
  for pids in itertools.product(*[
      range(np_) for np_ in per_player_num_policies]):
    meta_game[(slice(None),) + pids] = joint_returns[pids]
  meta_games.append(meta_game)
  return meta_games


def add_meta_dist(
    meta_dists, meta_values, meta_solver, meta_game, per_player_repeats,
    ignore_repeats):
  """Returns meta_dist."""
  num_players = meta_game.shape[0]
  meta_solver_func = _FLAG_TO_FUNC[meta_solver]
  meta_dist, _ = meta_solver_func(
      meta_game, per_player_repeats, ignore_repeats=ignore_repeats)
  # Clean dist.
  meta_dist = meta_dist.astype(np.float64)
  meta_dist[meta_dist < DIST_TOL] = 0.0
  meta_dist[meta_dist > 1.0] = 1.0
  meta_dist /= np.sum(meta_dist)
  meta_dist[meta_dist > 1.0] = 1.0
  meta_dists.append(meta_dist)
  meta_value = np.sum(
      meta_dist * meta_game, axis=tuple(range(1, num_players + 1)))
  meta_values.append(meta_value)
  return meta_dist


def find_best_response(
    game, meta_dist, meta_game, iteration, joint_policies,
    target_equilibrium, update_players_strategy):
  """Returns new best response policies."""
  num_players = meta_game.shape[0]
  per_player_num_policies = meta_dist.shape[:]

  # Player update strategy.
  if update_players_strategy == "all":
    players = list(range(num_players))
  elif update_players_strategy == "cycle":
    players = [iteration % num_players]
  elif update_players_strategy == "random":
    players = [np.random.randint(0, num_players)]
  else:
    raise ValueError(
        "update_players_strategy must be a valid player update strategy: "
        "%s. Received: %s" % (UPDATE_PLAYERS_STRATEGY, update_players_strategy))

  # Find best response.
  per_player_new_policies = []
  per_player_deviation_incentives = []

  if target_equilibrium == "cce":
    for player in range(num_players):
      if player in players:
        joint_policy_ids = itertools.product(*[
            (np_-1,) if p_ == player else range(np_) for p_, np_
            in enumerate(per_player_num_policies)])
        joint_policies_slice = [
            joint_policies[jpid] for jpid in joint_policy_ids]
        meta_dist_slice = np.sum(meta_dist, axis=player)
        meta_dist_slice[meta_dist_slice < DIST_TOL] = 0.0
        meta_dist_slice[meta_dist_slice > 1.0] = 1.0
        meta_dist_slice /= np.sum(meta_dist_slice)
        meta_dist_slice = meta_dist_slice.flat

        mu = [(p, mp) for mp, p in zip(joint_policies_slice, meta_dist_slice)
              if p > 0]
        info = pyspiel.cce_dist(game, mu, player, prob_cut_threshold=0.0)

        new_policy = policy.pyspiel_policy_to_python_policy(
            game, info.best_response_policies[0], players=(player,))
        on_policy_value = np.sum(meta_game[player] * meta_dist)
        deviation_incentive = max(
            info.best_response_values[0] - on_policy_value, 0)
        if deviation_incentive < GAP_TOL:
          deviation_incentive = 0.0

        per_player_new_policies.append([new_policy])
        per_player_deviation_incentives.append([deviation_incentive])
      else:
        per_player_new_policies.append([])
        per_player_deviation_incentives.append([])

  elif target_equilibrium == "ce":
    for player in range(num_players):
      if player in players:
        per_player_new_policies.append([])
        per_player_deviation_incentives.append([])

        for pid in range(per_player_num_policies[player]):
          joint_policy_ids = itertools.product(*[
              (pid,) if p_ == player else range(np_) for p_, np_
              in enumerate(per_player_num_policies)])
          joint_policies_slice = [
              joint_policies[jpid] for jpid in joint_policy_ids]
          inds = tuple((pid,) if player == p_ else slice(None)
                       for p_ in range(num_players))
          meta_dist_slice = np.ravel(meta_dist[inds]).copy()
          meta_dist_slice[meta_dist_slice < DIST_TOL] = 0.0
          meta_dist_slice[meta_dist_slice > 1.0] = 1.0
          meta_dist_slice_sum = np.sum(meta_dist_slice)

          if meta_dist_slice_sum > 0.0:
            meta_dist_slice /= meta_dist_slice_sum
            mu = [(p, mp) for mp, p in
                  zip(joint_policies_slice, meta_dist_slice)
                  if p > 0]
            info = pyspiel.cce_dist(game, mu, player, prob_cut_threshold=0.0)

            new_policy = policy.pyspiel_policy_to_python_policy(
                game, info.best_response_policies[0], players=(player,))
            on_policy_value = np.sum(
                np.ravel(meta_game[player][inds]) * meta_dist_slice)
            deviation_incentive = max(
                info.best_response_values[0] - on_policy_value, 0)
            if deviation_incentive < GAP_TOL:
              deviation_incentive = 0.0

            per_player_new_policies[-1].append(new_policy)
            per_player_deviation_incentives[-1].append(
                meta_dist_slice_sum * deviation_incentive)

      else:
        per_player_new_policies.append([])
        per_player_deviation_incentives.append([])

  else:
    raise ValueError(
        "target_equilibrium must be a valid best response strategy: %s. "
        "Received: %s" % (BRS, target_equilibrium))

  return per_player_new_policies, per_player_deviation_incentives


## Main Loop.


def initialize(game, train_meta_solver, eval_meta_solver, policy_init,
               ignore_repeats, br_selection):
  """Return initialized data structures."""
  num_players = game.num_players()

  # Initialize.
  iteration = 0
  per_player_repeats = [[] for _ in range(num_players)]
  per_player_policies = [[] for _ in range(num_players)]
  joint_policies = {}  # Eg. (1, 0): Joint policy.
  joint_returns = {}
  meta_games = []
  train_meta_dists = []
  eval_meta_dists = []
  train_meta_values = []
  eval_meta_values = []
  train_meta_gaps = []
  eval_meta_gaps = []

  # Initialize policies.
  per_player_new_policies = [
      [intilize_policy(game, player, policy_init)]
      for player in range(num_players)]
  per_player_gaps_train = [[1.0] for player in range(num_players)]
  per_player_num_novel_policies = add_new_policies(
      per_player_new_policies, per_player_gaps_train, per_player_repeats,
      per_player_policies, joint_policies, joint_returns, game, br_selection)
  del per_player_num_novel_policies
  add_meta_game(
      meta_games,
      per_player_policies,
      joint_returns)
  add_meta_dist(
      train_meta_dists, train_meta_values, train_meta_solver,
      meta_games[-1], per_player_repeats, ignore_repeats)
  add_meta_dist(
      eval_meta_dists, eval_meta_values, eval_meta_solver,
      meta_games[-1], per_player_repeats, ignore_repeats)

  return (
      iteration,
      per_player_repeats,
      per_player_policies,
      joint_policies,
      joint_returns,
      meta_games,
      train_meta_dists,
      eval_meta_dists,
      train_meta_values,
      eval_meta_values,
      train_meta_gaps,
      eval_meta_gaps)


def initialize_callback_(
    iteration,
    per_player_repeats,
    per_player_policies,
    joint_policies,
    joint_returns,
    meta_games,
    train_meta_dists,
    eval_meta_dists,
    train_meta_values,
    eval_meta_values,
    train_meta_gaps,
    eval_meta_gaps,
    game):
  """Callback which allows initializing from checkpoint."""
  del game
  checkpoint = None
  return (
      iteration,
      per_player_repeats,
      per_player_policies,
      joint_policies,
      joint_returns,
      meta_games,
      train_meta_dists,
      eval_meta_dists,
      train_meta_values,
      eval_meta_values,
      train_meta_gaps,
      eval_meta_gaps,
      checkpoint)


def callback_(
    iteration,
    per_player_repeats,
    per_player_policies,
    joint_policies,
    joint_returns,
    meta_games,
    train_meta_dists,
    eval_meta_dists,
    train_meta_values,
    eval_meta_values,
    train_meta_gaps,
    eval_meta_gaps,
    kwargs,
    checkpoint):
  """Callback for updating checkpoint."""
  del iteration, per_player_repeats, per_player_policies, joint_policies
  del joint_returns, meta_games, train_meta_dists, eval_meta_dists
  del train_meta_values, eval_meta_values, train_meta_gaps, eval_meta_gaps
  del kwargs
  return checkpoint


def run_loop(
    game,
    game_name,
    seed=0,
    iterations=40,
    policy_init="uniform",
    update_players_strategy="all",
    target_equilibrium="cce",
    br_selection="largest_gap",
    train_meta_solver="mgcce",
    eval_meta_solver="mwcce",
    ignore_repeats=False,
    initialize_callback=None,
    callback=None):
  """Runs JPSRO."""
  if initialize_callback is None:
    initialize_callback = initialize_callback_
  if callback is None:
    callback = callback_
  kwargs = dict(
      game=game,
      game_name=game_name,
      seed=seed,
      iterations=iterations,
      policy_init=policy_init,
      update_players_strategy=update_players_strategy,
      target_equilibrium=target_equilibrium,
      br_selection=br_selection,
      train_meta_solver=train_meta_solver,
      eval_meta_solver=eval_meta_solver,
      ignore_repeats=ignore_repeats,
  )

  # Set seed.
  np.random.seed(seed)

  # Some statistics.
  num_players = game.num_players()  # Look in the game.

  # Initialize.
  values = initialize(game, train_meta_solver, eval_meta_solver, policy_init,
                      ignore_repeats, br_selection)

  # Initialize Callback.
  (iteration,
   per_player_repeats,
   per_player_policies,
   joint_policies,
   joint_returns,
   meta_games,
   train_meta_dists,
   eval_meta_dists,
   train_meta_values,
   eval_meta_values,
   train_meta_gaps,
   eval_meta_gaps,
   checkpoint) = initialize_callback(*values, game)

  # Run JPSRO.
  while iteration <= iterations:
    logging.debug("Beginning JPSRO iteration %03d", iteration)
    per_player_new_policies, per_player_gaps_train = find_best_response(
        game, train_meta_dists[-1], meta_games[-1], iteration, joint_policies,
        target_equilibrium, update_players_strategy)
    train_meta_gaps.append([sum(gaps) for gaps in per_player_gaps_train])
    _, per_player_gaps_eval = find_best_response(
        game, eval_meta_dists[-1], meta_games[-1], iteration, joint_policies,
        target_equilibrium, update_players_strategy)
    eval_meta_gaps.append([sum(gaps) for gaps in per_player_gaps_eval])
    per_player_num_novel_policies = add_new_policies(
        per_player_new_policies, per_player_gaps_train, per_player_repeats,
        per_player_policies, joint_policies, joint_returns, game, br_selection)
    del per_player_num_novel_policies
    add_meta_game(
        meta_games,
        per_player_policies,
        joint_returns)
    add_meta_dist(
        train_meta_dists, train_meta_values, train_meta_solver,
        meta_games[-1], per_player_repeats, ignore_repeats)
    add_meta_dist(
        eval_meta_dists, eval_meta_values, eval_meta_solver,
        meta_games[-1], per_player_repeats, ignore_repeats)

    # Stats.
    per_player_num_policies = train_meta_dists[-1].shape[:]
    log_string = LOG_STRING.format(
        iteration=iteration,
        game=game_name,
        player=("{: 12d}" * num_players).format(*list(range(num_players))),
        brs="",
        num_policies=("{: 12d}" * num_players).format(*[
            sum(ppr) for ppr in per_player_repeats]),
        unique=("{: 12d}" * num_players).format(*per_player_num_policies),
        train_meta_solver=train_meta_solver,
        train_value=("{: 12g}" * num_players).format(*train_meta_values[-1]),
        train_gap=("{: 12g}" * num_players).format(*train_meta_gaps[-1]),
        eval_meta_solver=eval_meta_solver,
        eval_value=("{: 12g}" * num_players).format(*eval_meta_values[-1]),
        eval_gap=("{: 12g}" * num_players).format(*eval_meta_gaps[-1]),
    )
    logging.info(log_string)

    # Increment.
    iteration += 1

    # Callback.
    checkpoint = callback(
        iteration,
        per_player_repeats,
        per_player_policies,
        joint_policies,
        joint_returns,
        meta_games,
        train_meta_dists,
        eval_meta_dists,
        train_meta_values,
        eval_meta_values,
        train_meta_gaps,
        eval_meta_gaps,
        kwargs,
        checkpoint)
