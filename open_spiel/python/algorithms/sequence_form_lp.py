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

"""An implementation of sequence-form linear programming.

This is a classic algorithm for solving two-player zero-sum games with imperfect
information. For a general introduction to the concepts, see Sec 5.2.3 of
Shoham & Leyton-Brown '09, Multiagent Systems: Algorithmic, Game-Theoretic, and
Logical Foundations http://www.masfoundations.org/mas.pdf.

In this implementation, we follow closely the construction in Koller, Megiddo,
and von Stengel, Fast Algorithms for Finding Randomized Strategies in Game Trees
http://theory.stanford.edu/~megiddo/pdf/stoc94.pdf. Specifically, we construct
and solve equations (8) and (9) from this paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python import policy
from open_spiel.python.algorithms import lp_solver
import pyspiel

_DELIMITER = " -=- "
_EMPTY_INFOSET_KEYS = ["***EMPTY_INFOSET_P0***", "***EMPTY_INFOSET_P1***"]
_EMPTY_INFOSET_ACTION_KEYS = [
    "***EMPTY_INFOSET_ACTION_P0***", "***EMPTY_INFOSET_ACTION_P1***"
]


def _construct_lps(state, infosets, infoset_actions, infoset_action_maps,
                   chance_reach, lps, parent_is_keys, parent_isa_keys):
  """Build the linear programs recursively from this state.

  Args:
    state: an open spiel state (root of the game tree)
    infosets: a list of dicts, one per player, that maps infostate to an id. The
      dicts are filled by this function and should initially only contain root
      values.
    infoset_actions: a list of dicts, one per player, that maps a string of
      (infostate, action) pair to an id. The dicts are filled by this function
      and should inirially only contain the root values
    infoset_action_maps: a list of dicts, one per player, that maps each
      info_state to a list of (infostate, action) string
    chance_reach: the contribution of chance's reach probability (should start
      at 1).
    lps: a list of linear programs, one per player. The first one will be
      constructred as in Eq (8) of Koller, Megiddo and von Stengel. The second
      lp is Eq (9). Initially these should contain only the root-level
      constraints and variables.
    parent_is_keys: a list of parent information state keys for this state
    parent_isa_keys: a list of parent (infostate, action) keys
  """
  if state.is_terminal():
    returns = state.returns()
    # Left-most term of: -Ay + E^t p >= 0
    lps[0].add_or_reuse_constraint(parent_isa_keys[0], lp_solver.CONS_TYPE_GEQ)
    lps[0].add_to_cons_coeff(parent_isa_keys[0], parent_isa_keys[1],
                             -1.0 * returns[0] * chance_reach)
    # Right-most term of: -Ay + E^t p >= 0
    lps[0].set_cons_coeff(parent_isa_keys[0], parent_is_keys[0], 1.0)
    # Left-most term of: x^t (-A) - q^t F <= 0
    lps[1].add_or_reuse_constraint(parent_isa_keys[1], lp_solver.CONS_TYPE_LEQ)
    lps[1].add_to_cons_coeff(parent_isa_keys[1], parent_isa_keys[0],
                             -1.0 * returns[0] * chance_reach)
    # Right-most term of: x^t (-A) - q^t F <= 0
    lps[1].set_cons_coeff(parent_isa_keys[1], parent_is_keys[1], -1.0)
    return

  if state.is_chance_node():
    for action, prob in state.chance_outcomes():
      new_state = state.child(action)
      _construct_lps(new_state, infosets, infoset_actions, infoset_action_maps,
                     prob * chance_reach, lps, parent_is_keys, parent_isa_keys)
    return

  player = state.current_player()
  info_state = state.information_state_string(player)
  legal_actions = state.legal_actions(player)

  # p and q variables, inequality constraints, and part of equality constraints
  if player == 0:
    # p
    lps[0].add_or_reuse_variable(info_state)
    # -Ay + E^t p >= 0
    lps[0].add_or_reuse_constraint(parent_isa_keys[0], lp_solver.CONS_TYPE_GEQ)
    lps[0].set_cons_coeff(parent_isa_keys[0], parent_is_keys[0], 1.0)
    lps[0].set_cons_coeff(parent_isa_keys[0], info_state, -1.0)
    # x^t E^t = e^t
    lps[1].add_or_reuse_constraint(info_state, lp_solver.CONS_TYPE_EQ)
    lps[1].set_cons_coeff(info_state, parent_isa_keys[0], -1.0)
  else:
    # q
    lps[1].add_or_reuse_variable(info_state)
    # x^t (-A) - q^t F <= 0
    lps[1].add_or_reuse_constraint(parent_isa_keys[1], lp_solver.CONS_TYPE_LEQ)
    lps[1].set_cons_coeff(parent_isa_keys[1], parent_is_keys[1], -1.0)
    lps[1].set_cons_coeff(parent_isa_keys[1], info_state, 1.0)
    # -Fy = -f
    lps[0].add_or_reuse_constraint(info_state, lp_solver.CONS_TYPE_EQ)
    lps[0].set_cons_coeff(info_state, parent_isa_keys[1], -1.0)

  # Add to the infostate maps
  if info_state not in infosets[player]:
    infosets[player][info_state] = len(infosets[player])
  if info_state not in infoset_action_maps[player]:
    infoset_action_maps[player][info_state] = []

  new_parent_is_keys = parent_is_keys[:]
  new_parent_is_keys[player] = info_state

  for action in legal_actions:
    isa_key = info_state + _DELIMITER + str(action)
    if isa_key not in infoset_actions[player]:
      infoset_actions[player][isa_key] = len(infoset_actions[player])
    if isa_key not in infoset_action_maps[player][info_state]:
      infoset_action_maps[player][info_state].append(isa_key)

    # x and y variables, and finish equality constraints coeff
    if player == 0:
      lps[1].add_or_reuse_variable(isa_key, lb=0)  # x
      lps[1].set_cons_coeff(info_state, isa_key, 1.0)  # x^t E^t = e^t
    else:
      lps[0].add_or_reuse_variable(isa_key, lb=0)  # y
      lps[0].set_cons_coeff(info_state, isa_key, 1.0)  # -Fy = -f

    new_parent_isa_keys = parent_isa_keys[:]
    new_parent_isa_keys[player] = isa_key

    new_state = state.child(action)
    _construct_lps(new_state, infosets, infoset_actions, infoset_action_maps,
                   chance_reach, lps, new_parent_is_keys, new_parent_isa_keys)


def solve_zero_sum_game(game, solver=None):
  """Solve the two-player zero-sum game using sequence-form LPs.

  Args:
    game: the spiel game tp solve (must be zero-sum, sequential, and have chance
      mode of deterministic or explicit stochastic).
    solver: a specific solver to use, sent to cvxopt (i.e. 'lapack', 'blas',
      'glpk'). A value of None uses cvxopt's default solver.

  Returns:
    A 4-tuple containing:
      - player 0 value
      - player 1 value
      - player 0 policy: a policy.TabularPolicy for player 0
      - player 1 policy: a policy.TabularPolicy for player 1
  """
  assert game.num_players() == 2
  assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
  assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
  assert (
      game.get_type().chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC
      or game.get_type().chance_mode ==
      pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
  # There are several import matrices and vectors that form the LPs that
  # are built by this function:
  #
  # A is expected payoff to p1 of each (infoset0,action0) + (infoset1,action1)
  #   belong to p1 and p2 respectively, which lead to a terminal state. It has
  #   dimensions (infoset-actions0) x (infoset-actions1)
  # E,F are p1 / p2's strategy matrices (infosets) x (infoset-actions)
  # e,f are infosets+ x 1 column vector of (1 0 0 ... 0)
  # p,q are unconstrained variables each with infosets x 1.
  # x,y are realization plans of size infoset-actions
  #
  # In each of the computations above there is a special "root infoset" and
  # "root infoset-action" denote \emptyset. So the values are actually equal to
  # number of infosets + 1 and infoset-actions + 1.
  #
  # Equation (8) is   min_{y,p} e^T p
  #
  #             s.t.  -Ay + E^t p >= 0
  #                   -Fy          = -f
  #                     y         >= 0
  #
  # Equation (9) is   max_{x,q} -q^T f
  #
  #             s.t.  x^t(-A) - q^t F <= 0
  #                   x^t E^t          = e^t
  #                   x               >= 0
  #
  # So, the first LP has:
  #  - |y| + |p| variables (infoset-actions1 + infosets0)
  #  - infoset-actions0 inequality constraints (other than var lower-bounds)
  #  - infosets1 equality constraints
  #
  # And the second LP has:
  #  - |x| + |q| variables (infoset-actions0 + infosets1)
  #  - infoset-actions1 inequality constraints (other than var lower-bounds)
  #  - infosets0 equality constraints
  infosets = [{_EMPTY_INFOSET_KEYS[0]: 0}, {_EMPTY_INFOSET_KEYS[1]: 0}]
  infoset_actions = [{
      _EMPTY_INFOSET_ACTION_KEYS[0]: 0
  }, {
      _EMPTY_INFOSET_ACTION_KEYS[1]: 0
  }]
  infoset_action_maps = [{}, {}]
  lps = [
      lp_solver.LinearProgram(lp_solver.OBJ_MIN),  # Eq. (8)
      lp_solver.LinearProgram(lp_solver.OBJ_MAX)  # Eq. (9)
  ]
  # Root-level variables and constraints.
  lps[0].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[1], lb=0)  # y root
  lps[0].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[0])  # p root
  lps[1].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[0], lb=0)  # x root
  lps[1].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[1])  # q root
  # objective coefficients
  lps[0].set_obj_coeff(_EMPTY_INFOSET_KEYS[0], 1.0)  # e^t p
  lps[1].set_obj_coeff(_EMPTY_INFOSET_KEYS[1], -1.0)  # -q^t f
  # y_root = 1  (-Fy = -f)
  lps[0].add_or_reuse_constraint(_EMPTY_INFOSET_KEYS[1], lp_solver.CONS_TYPE_EQ)
  lps[0].set_cons_coeff(_EMPTY_INFOSET_KEYS[1], _EMPTY_INFOSET_ACTION_KEYS[1],
                        -1.0)
  lps[0].set_cons_rhs(_EMPTY_INFOSET_KEYS[1], -1.0)
  # x_root = 1  (x^t E^t = e^t)
  lps[1].add_or_reuse_constraint(_EMPTY_INFOSET_KEYS[0], lp_solver.CONS_TYPE_EQ)
  lps[1].set_cons_coeff(_EMPTY_INFOSET_KEYS[0], _EMPTY_INFOSET_ACTION_KEYS[0],
                        1.0)
  lps[1].set_cons_rhs(_EMPTY_INFOSET_KEYS[0], 1.0)
  _construct_lps(game.new_initial_state(), infosets, infoset_actions,
                 infoset_action_maps, 1.0, lps, _EMPTY_INFOSET_KEYS[:],
                 _EMPTY_INFOSET_ACTION_KEYS[:])
  # Solve the programs.
  solutions = [lps[0].solve(solver=solver), lps[1].solve(solver=solver)]
  # Extract the policies (convert from realization plan to behavioral form).
  policies = [policy.TabularPolicy(game), policy.TabularPolicy(game)]
  for i in range(2):
    for info_state in infoset_action_maps[i]:
      total_weight = 0
      num_actions = 0
      for isa_key in infoset_action_maps[i][info_state]:
        total_weight += solutions[1 - i][lps[1 - i].get_var_id(isa_key)]
        num_actions += 1
      unif_pr = 1.0 / num_actions
      state_policy = policies[i].policy_for_key(info_state)
      for isa_key in infoset_action_maps[i][info_state]:
        # The 1 - i here is due to Eq (8) yielding a solution for player 1 and
        # Eq (9) a solution for player 0.
        rel_weight = solutions[1 - i][lps[1 - i].get_var_id(isa_key)]
        _, action_str = isa_key.split(_DELIMITER)
        action = int(action_str)
        pr_action = rel_weight / total_weight if total_weight > 0 else unif_pr
        state_policy[action] = pr_action
  return (solutions[0][lps[0].get_var_id(_EMPTY_INFOSET_KEYS[0])],
          solutions[1][lps[1].get_var_id(_EMPTY_INFOSET_KEYS[1])], policies[0],
          policies[1])
