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

"""LP Solver for two-player zero-sum games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cvxopt
import numpy as np
from open_spiel.python.egt import utils
import pyspiel

# Constants that determine the type of objective (max vs. min) and type of
# constraints (<=, >=, =).
OBJ_MAX = 1
OBJ_MIN = 2
CONS_TYPE_LEQ = 3
CONS_TYPE_GEQ = 4
CONS_TYPE_EQ = 5

# Constants that determine the type of dominance to find.
DOMINANCE_STRICT = 1
DOMINANCE_VERY_WEAK = 2
DOMINANCE_WEAK = 3


class _Variable(object):
  """A variable in an LP."""

  def __init__(self, vid, lb=None, ub=None):
    """Creates a variable in a linear program.

    Args:
      vid: (integer) the variable id (should be unique for each variable)
      lb: the lower bound on the variable's value (None means no lower bound)
      ub: the upper bound on the variable's valie (None means no upper bound)
    """
    self.vid = vid
    self.lb = lb
    self.ub = ub


class _Constraint(object):
  """A constraint in an LP."""

  def __init__(self, cid, ctype):
    """Creates a constraint in a linear program.

    Args:
      cid: (integer) the constraint id (should be unique for each constraint)
      ctype: the constraint type (CONS_TYPE_{LEQ, GEQ, EQ})
    """
    self.cid = cid
    self.ctype = ctype
    self.coeffs = {}  # var label -> value
    self.rhs = None


class LinearProgram(object):
  """A object used to provide a user-friendly API for building LPs."""

  def __init__(self, objective):
    assert objective == OBJ_MIN or objective == OBJ_MAX
    self._valid_constraint_types = [CONS_TYPE_EQ, CONS_TYPE_LEQ, CONS_TYPE_GEQ]
    self._objective = objective
    self._obj_coeffs = {}  # var label -> value
    self._vars = {}  # var label -> var
    self._cons = {}  # cons label -> constraint
    self._var_list = []
    self._leq_cons_list = []
    self._eq_cons_list = []

  def add_or_reuse_variable(self, label, lb=None, ub=None):
    """Adds a variable to this LP, or reuses one if the label exists.

    If the variable already exists, simply checks that the upper and lower
    bounds are the same as previously specified.

    Args:
      label: a label to assign to this constraint
      lb: a lower-bound value for this variable
      ub: an upper-bound value for this variable
    """
    var = self._vars.get(label)
    if var is not None:
      # Do not re-add, but ensure it's the same
      assert var.lb == lb and var.ub == ub
      return
    var = _Variable(len(self._var_list), lb, ub)
    self._vars[label] = var
    self._var_list.append(var)

  def add_or_reuse_constraint(self, label, ctype):
    """Adds a constraint to this LP, or reuses one if the label exists.

     If the constraint is already present, simply checks it's the same type as
     previously specified.

    Args:
      label: a label to assign to this constraint
      ctype: the constraint type (in CONS_TYPE_{LEQ,GEQ,EQ})
    """
    assert ctype in self._valid_constraint_types
    cons = self._cons.get(label)
    if cons is not None:
      # Do not re-add, but ensure it's the same type
      assert cons.ctype == ctype
      return
    if ctype == CONS_TYPE_LEQ or ctype == CONS_TYPE_GEQ:
      cons = _Constraint(len(self._leq_cons_list), ctype)
      self._cons[label] = cons
      self._leq_cons_list.append(cons)
    elif ctype == CONS_TYPE_EQ:
      cons = _Constraint(len(self._eq_cons_list), ctype)
      self._cons[label] = cons
      self._eq_cons_list.append(cons)
    else:
      assert False, "Unknown constraint type"

  def set_obj_coeff(self, var_label, coeff):
    """Sets a coefficient of a variable in the objective."""
    self._obj_coeffs[var_label] = coeff

  def set_cons_coeff(self, cons_label, var_label, coeff):
    """Sets a coefficient of a constraint in the LP."""
    self._cons[cons_label].coeffs[var_label] = coeff

  def add_to_cons_coeff(self, cons_label, var_label, add_coeff):
    """Sets a coefficient of a constraint in the LP."""
    val = self._cons[cons_label].coeffs.get(var_label)
    if val is None:
      val = 0
    self._cons[cons_label].coeffs[var_label] = val + add_coeff

  def set_cons_rhs(self, cons_label, value):
    """Sets the right-hand side of a constraint."""
    self._cons[cons_label].rhs = value

  def get_var_id(self, label):
    var = self._vars.get(label)
    assert var is not None
    return var.vid

  def get_num_cons(self):
    return len(self._leq_cons_list), len(self._eq_cons_list)

  def solve(self, solver=None):
    """Solves the LP.

    Args:
      solver: the solver to use ('blas', 'lapack', 'glpk'). Defaults to None,
        which then uses the cvxopt internal default.

    Returns:
      The solution as a dict of var label -> value, one for each variable.
    """
    # From http://cvxopt.org/userguide/coneprog.html#linear-programming,
    # CVXOPT uses the formulation:
    #    minimize: c^t x
    #       s.t.   Gx <= h
    #              Ax = b
    #
    # Here:
    #  - x is the vector the variables
    #  - c is the vector of objective coefficients
    #  - G is the matrix of LEQ (and GEQ) constraint coefficients
    #  - h is the vector or right-hand side values of the LEQ/GEQ constraints
    #  - A is the matrix of equality constraint coefficients
    #  - b is the vector of right-hand side values of the equality constraints
    #
    # This function builds these sparse matrices from the information it has
    # gathered, flipping signs where necessary, and adding equality constraints
    # for the upper and lower bounds of variables. It then calls the cvxopt
    # solver and maps back the values.
    num_vars = len(self._var_list)
    num_eq_cons = len(self._eq_cons_list)
    num_leq_cons = len(self._leq_cons_list)
    for var in self._var_list:
      if var.lb is not None:
        num_leq_cons += 1
      if var.ub is not None:
        num_leq_cons += 1
    # Make the matrices (some need to be dense).
    c = cvxopt.matrix([0.0] * num_vars)
    h = cvxopt.matrix([0.0] * num_leq_cons)
    g_mat = cvxopt.spmatrix([], [], [], (num_leq_cons, num_vars))
    a_mat = None
    b = None
    if num_eq_cons > 0:
      a_mat = cvxopt.spmatrix([], [], [], (num_eq_cons, num_vars))
      b = cvxopt.matrix([0.0] * num_eq_cons)
    # Objective coefficients: c
    for var_label in self._obj_coeffs:
      value = self._obj_coeffs[var_label]
      vid = self._vars[var_label].vid
      if self._objective == OBJ_MAX:
        c[vid] = -value  # negate the value because it's a max
      else:
        c[vid] = value  # min objective matches cvxopt
    # Inequality constraints: G, h
    row = 0
    for cons in self._leq_cons_list:
      # If it's >= then need to negate all coeffs and the rhs
      if cons.rhs is not None:
        h[row] = cons.rhs if cons.ctype == CONS_TYPE_LEQ else -cons.rhs
      for var_label in cons.coeffs:
        value = cons.coeffs[var_label]
        vid = self._vars[var_label].vid
        g_mat[(row, vid)] = value if cons.ctype == CONS_TYPE_LEQ else -value
      row += 1
    # Inequality constraints: variables upper and lower bounds
    for var in self._var_list:
      if var.lb is not None:  # x_i >= lb has to be -x_i <= -lb
        g_mat[(row, var.vid)] = -1.0
        h[row] = -var.lb
        row += 1
      if var.ub is not None:  # x_i <= ub
        g_mat[(row, var.vid)] = 1.0
        h[row] = var.ub
        row += 1
    # Equality constraints: A, b
    if num_eq_cons > 0:
      row = 0
      for cons in self._eq_cons_list:
        b[row] = cons.rhs if cons.rhs is not None else 0.0
        for var_label in cons.coeffs:
          value = cons.coeffs[var_label]
          vid = self._vars[var_label].vid
          a_mat[(row, vid)] = value
        row += 1
    # Solve!
    if num_eq_cons > 0:
      sol = cvxopt.solvers.lp(c, g_mat, h, a_mat, b, solver=solver)
    else:
      sol = cvxopt.solvers.lp(c, g_mat, h, solver=solver)
    return sol["x"]


def solve_zero_sum_matrix_game(game):
  """Solves a matrix game by using linear programming.

  Args:
    game: a pyspiel MatrixGame

  Returns:
    A 4-tuple containing:
      - p0_sol (array-like): probability distribution over row actions
      - p1_sol (array-like): probability distribution over column actions
      - p0_sol_value, expected value to the first player
      - p1_sol_value, expected value to the second player
  """

  # Solving a game for player i (e.g. row player) requires finding a mixed
  # policy over player i's pure strategies (actions) such that a value of the
  # mixed strategy against every opponent pure strategy is maximized.
  #
  # For more detail, please refer to Sec 4.1 of Shoham & Leyton-Brown, 2009:
  # Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations
  # http://www.masfoundations.org/mas.pdf
  #
  # For the row player the LP looks like:
  #    max V
  #     st. sigma_a1 \dot col_0 >= V
  #         sigma_a2 \dot col_1 >= V
  #              .
  #              .
  #         sigma_am \cot col_n >= V
  #         for all i, sigma_ai >= 0
  #         sigma \dot 1 = 1
  assert isinstance(game, pyspiel.MatrixGame)
  assert game.get_type().information == pyspiel.GameType.Information.ONE_SHOT
  assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
  num_rows = game.num_rows()
  num_cols = game.num_cols()
  cvxopt.solvers.options["show_progress"] = False

  # First, do the row player (player 0).
  lp0 = LinearProgram(OBJ_MAX)
  for r in range(num_rows):  # one var per action / pure strategy
    lp0.add_or_reuse_variable(r, lb=0)
  lp0.add_or_reuse_variable(num_rows)  # V
  lp0.set_obj_coeff(num_rows, 1.0)  # max V
  for c in range(num_cols):
    lp0.add_or_reuse_constraint(c, CONS_TYPE_GEQ)
    for r in range(num_rows):
      lp0.set_cons_coeff(c, r, game.player_utility(0, r, c))
    lp0.set_cons_coeff(c, num_rows, -1.0)  # -V >= 0
  lp0.add_or_reuse_constraint(num_cols + 1, CONS_TYPE_EQ)
  lp0.set_cons_rhs(num_cols + 1, 1.0)
  for r in range(num_rows):
    lp0.set_cons_coeff(num_cols + 1, r, 1.0)
  sol = lp0.solve()
  p0_sol = sol[:-1]
  p0_sol_val = sol[-1]

  # Now, the column player (player 1).
  lp1 = LinearProgram(OBJ_MAX)
  for c in range(num_cols):  # one var per action / pure strategy
    lp1.add_or_reuse_variable(c, lb=0)
  lp1.add_or_reuse_variable(num_cols)  # V
  lp1.set_obj_coeff(num_cols, 1)  # max V
  for r in range(num_rows):
    lp1.add_or_reuse_constraint(r, CONS_TYPE_GEQ)
    for c in range(num_cols):
      lp1.set_cons_coeff(r, c, game.player_utility(1, r, c))
    lp1.set_cons_coeff(r, num_cols, -1.0)  # -V >= 0
  lp1.add_or_reuse_constraint(num_rows + 1, CONS_TYPE_EQ)
  lp1.set_cons_rhs(num_rows + 1, 1.0)
  for c in range(num_cols):
    lp1.set_cons_coeff(num_rows + 1, c, 1.0)
  sol = lp1.solve()
  p1_sol = sol[:-1]
  p1_sol_val = sol[-1]

  return p0_sol, p1_sol, p0_sol_val, p1_sol_val


def is_dominated(action,
                 game_or_payoffs,
                 player,
                 mode=DOMINANCE_STRICT,
                 tol=1e-7,
                 return_mixture=False):
  """Determines whether a pure strategy is dominated by any mixture strategies.

  Args:
    action: index of an action for `player`
    game_or_payoffs: either a pyspiel matrix- or normal-form game, or a payoff
      tensor for `player` with ndim == number of players
    player: index of the player (an integer)
    mode: dominance criterion: strict, weak, or very weak
    tol: tolerance
    return_mixture: whether to return the dominating strategy if one exists

  Returns:
    If `return_mixture`:
      a dominating mixture strategy if one exists, or `None`.
      the strategy is provided as a 1D numpy array of mixture weights.
    Otherwise: True if a dominating strategy exists, False otherwise.
  """
  # For more detail, please refer to Sec 4.5.2 of Shoham & Leyton-Brown, 2009:
  # Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations
  # http://www.masfoundations.org/mas.pdf
  assert mode in (DOMINANCE_STRICT, DOMINANCE_VERY_WEAK, DOMINANCE_WEAK)
  payoffs = utils.game_payoffs_array(game_or_payoffs)[player] if isinstance(
      game_or_payoffs, pyspiel.NormalFormGame) else np.asfarray(game_or_payoffs)

  # Reshape payoffs so rows correspond to `player` and cols to the joint action
  # of all other players
  payoffs = np.moveaxis(payoffs, player, 0)
  payoffs = payoffs.reshape((payoffs.shape[0], -1))
  num_rows, num_cols = payoffs.shape

  cvxopt.solvers.options["show_progress"] = False
  cvxopt.solvers.options["maxtol"] = tol
  cvxopt.solvers.options["feastol"] = tol
  lp = LinearProgram(OBJ_MAX)

  # One var for every row probability, fixed to 0 if inactive
  for r in range(num_rows):
    if r == action:
      lp.add_or_reuse_variable(r, lb=0, ub=0)
    else:
      lp.add_or_reuse_variable(r, lb=0)

  # For the strict LP we normalize the payoffs to be strictly positive
  if mode == DOMINANCE_STRICT:
    to_subtract = payoffs.min() - 1
  else:
    to_subtract = 0
    # For non-strict LPs the probabilities must sum to 1
    lp.add_or_reuse_constraint(num_cols, CONS_TYPE_EQ)
    lp.set_cons_rhs(num_cols, 1)
    for r in range(num_rows):
      if r != action:
        lp.set_cons_coeff(num_cols, r, 1)

  # The main dominance constraint
  for c in range(num_cols):
    lp.add_or_reuse_constraint(c, CONS_TYPE_GEQ)
    lp.set_cons_rhs(c, payoffs[action, c] - to_subtract)
    for r in range(num_rows):
      if r != action:
        lp.set_cons_coeff(c, r, payoffs[r, c] - to_subtract)

  if mode == DOMINANCE_STRICT:
    # Minimize sum of probabilities
    for r in range(num_rows):
      if r != action:
        lp.set_obj_coeff(r, -1)
    mixture = lp.solve()
    if mixture is not None and np.sum(mixture) < 1 - tol:
      mixture = np.squeeze(mixture, 1) / np.sum(mixture)
    else:
      mixture = None

  if mode == DOMINANCE_VERY_WEAK:
    # Check feasibility
    mixture = lp.solve()
    if mixture is not None:
      mixture = np.squeeze(mixture, 1)

  if mode == DOMINANCE_WEAK:
    # Check feasibility and whether there's any advantage
    for r in range(num_rows):
      lp.set_obj_coeff(r, payoffs[r].sum())
    mixture = lp.solve()
    if mixture is not None:
      mixture = np.squeeze(mixture, 1)
      if (np.dot(mixture, payoffs) - payoffs[action]).sum() <= tol:
        mixture = None

  return mixture if return_mixture else (mixture is not None)


def _pure_dominated_from_advantages(advantages, mode, tol=1e-7):
  if mode == DOMINANCE_STRICT:
    return (advantages > tol).all(1)
  if mode == DOMINANCE_WEAK:
    return (advantages >= -tol).all(1) & (advantages.sum(1) > tol)
  if mode == DOMINANCE_VERY_WEAK:
    return (advantages >= -tol).all(1)


def iterated_dominance(game_or_payoffs, mode, tol=1e-7):
  """Reduces a strategy space using iterated dominance.

  See: http://www.smallparty.com/yoram/classes/principles/nash.pdf

  Args:
    game_or_payoffs: either a pyspiel matrix- or normal-form game, or a payoff
      tensor of dimension `num_players` + 1. First dimension is the player,
      followed by the actions of all players, e.g. a 3x3 game (2 players) has
      dimension [2,3,3].
    mode: DOMINANCE_STRICT, DOMINANCE_WEAK, or DOMINANCE_VERY_WEAK
    tol: tolerance

  Returns:
    A tuple (`reduced_game`, `live_actions`).
    * if `game_or_payoffs` is an instance of `pyspiel.MatrixGame`, so is
      `reduced_game`; otherwise `reduced_game` is a payoff tensor.
    * `live_actions` is a tuple of length `num_players`, where
      `live_actions[player]` is a boolean vector of shape `num_actions`;
       `live_actions[player][action]` is `True` if `action` wasn't dominated for
       `player`.
  """
  payoffs = utils.game_payoffs_array(game_or_payoffs) if isinstance(
      game_or_payoffs, pyspiel.NormalFormGame) else np.asfarray(game_or_payoffs)
  live_actions = [
      np.ones(num_actions, np.bool) for num_actions in payoffs.shape[1:]
  ]
  progress = True
  while progress:
    progress = False
    # trying faster method first
    for method in ("pure", "mixed"):
      if progress:
        continue
      for player, live in enumerate(live_actions):
        if live.sum() == 1:
          # one action is dominant
          continue

        # discarding all dominated opponent actions
        payoffs_live = payoffs[player]
        for opponent in range(payoffs.shape[0]):
          if opponent != player:
            payoffs_live = payoffs_live.compress(live_actions[opponent],
                                                 opponent)

        # reshaping to (player_actions, joint_opponent_actions)
        payoffs_live = np.moveaxis(payoffs_live, player, 0)
        payoffs_live = payoffs_live.reshape((payoffs_live.shape[0], -1))

        for action in range(live.size):
          if not live[action]:
            continue
          if method == "pure":
            # mark all actions that `action` dominates
            advantage = payoffs_live[action] - payoffs_live
            dominated = _pure_dominated_from_advantages(advantage, mode, tol)
            dominated[action] = False
            dominated &= live
            if dominated.any():
              progress = True
              live &= ~dominated
              if live.sum() == 1:
                break
          if method == "mixed":
            # test if `action` is dominated by a mixed policy
            mixture = is_dominated(
                live[:action].sum(),
                payoffs_live[live],
                0,
                mode,
                tol,
                return_mixture=True)
            if mixture is None:
              continue
            # if it is, mark any other actions dominated by that policy
            progress = True
            advantage = mixture.dot(payoffs_live[live]) - payoffs_live[live]
            dominated = _pure_dominated_from_advantages(advantage, mode, tol)
            dominated[mixture > tol] = False
            assert dominated[live[:action].sum()]
            live.put(live.nonzero()[0], ~dominated)
            if live.sum() == 1:
              break

  for player, live in enumerate(live_actions):
    payoffs = payoffs.compress(live, player + 1)

  if isinstance(game_or_payoffs, pyspiel.MatrixGame):
    return pyspiel.MatrixGame(game_or_payoffs.get_type(),
                              game_or_payoffs.get_parameters(), [
                                  game_or_payoffs.row_action_name(action)
                                  for action in live_actions[0].nonzero()[0]
                              ], [
                                  game_or_payoffs.col_action_name(action)
                                  for action in live_actions[1].nonzero()[0]
                              ], *payoffs), live_actions
  else:
    return payoffs, live_actions


# TODO(author5): add a function for sequential games using sequence-form LPs.
