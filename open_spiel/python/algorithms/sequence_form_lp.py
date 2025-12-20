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

from open_spiel.python import policy
from open_spiel.python.algorithms import lp_solver
import pyspiel

_DELIMITER = " -=- "
_EMPTY_INFOSET_KEYS = ["***EMPTY_INFOSET_P0***", "***EMPTY_INFOSET_P1***"]
_EMPTY_INFOSET_ACTION_KEYS = [
    "***EMPTY_INFOSET_ACTION_P0***", "***EMPTY_INFOSET_ACTION_P1***"
]


def _construct_lps(state, infosets, infoset_actions, infoset_action_maps,
                   chance_reach, lps, parent_is_keys, parent_isa_keys,
                   infostate_parent_sequences):
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
    infostate_parent_sequences: a list of dicts, one per player, that maps
      infostate to the parent sequence key of the opponent.
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
                     prob * chance_reach, lps, parent_is_keys, parent_isa_keys,
                     infostate_parent_sequences)
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
    infostate_parent_sequences[player][info_state] = parent_isa_keys[1 - player]
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
                   chance_reach, lps, new_parent_is_keys, new_parent_isa_keys,
                   infostate_parent_sequences)


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
  # Mapping from infostate to its parent sequence id (opponent's sequence)
  # This tells us which opponent sequence leads to this infostate.
  infostate_parent_sequences = [{}, {}]

  _construct_lps(game.new_initial_state(), infosets, infoset_actions,
                 infoset_action_maps, 1.0, lps, _EMPTY_INFOSET_KEYS[:],
                 _EMPTY_INFOSET_ACTION_KEYS[:], infostate_parent_sequences)
  # Solve the programs.
  primal_solutions = []
  dual_eq_solutions = []
  dual_ineq_solutions = []
  for i in range(2):
    x, y, z = lps[i].solve(solver=solver)
    primal_solutions.append(x)
    dual_eq_solutions.append(y)
    dual_ineq_solutions.append(z)

  # Extract the policies (convert from realization plan to behavioral form).
  policies = [policy.TabularPolicy(game), policy.TabularPolicy(game)]

  # To correctly identify reachable states for player i, we need to know the reach
  # probability of infostates under the current equilibrium strategies.
  # An infostate is reachable for player i if ALL its player-i ancestors have
  # non-zero probability and ALL its player-(1-i) ancestors (sequences) have
  # non-zero realization probability.
  
  reach_probs = [{}, {}] # Key: infostate, Value: reach probability
  reach_probs[0][_EMPTY_INFOSET_KEYS[0]] = 1.0
  reach_probs[1][_EMPTY_INFOSET_KEYS[1]] = 1.0

  # We need to traverse the game tree (or use the infoset maps) to propagate reach probs.
  # Since we have infoset_action_maps, we can propagate top-down.
  # However, the order in infoset_action_maps might not be topological.
  # Let's use a simple topological-like propagation by iterating and repeating if needed, 
  # or better, use the structure if possible. Kuhn is small, we can just use the known structure.
  # Realization plan for Player 0 (x) is in primal_solutions[1]
  # Realization plan for Player 1 (y) is in primal_solutions[0]
  def get_realization_prob(player, isa_key):
    if isa_key in _EMPTY_INFOSET_ACTION_KEYS:
      return 1.0
    if player == 0:
      vid = lps[1].get_var_id(isa_key)
      return primal_solutions[1][vid]
    else:
      vid = lps[0].get_var_id(isa_key)
      return primal_solutions[0][vid]

  for i in range(2):
    for info_state in infoset_action_maps[i]:
      # Reach probability of this infostate for player i is x(parent_isa_of_i).
      # But we also need the reach probability of the OPPONENT'S sequence leading here.
      opponent_isa_key = infostate_parent_sequences[i][info_state]
      
      # Joint reach prob = own_reach * opponent_reach * chance_reach.
      # However, total_weight across actions already includes opponent reach and chance reach 
      # from the objective/constraints!
      # Actually, the sequence-form realization plan x(s,a) for Player 0 ALREADY incorporates
      # player 0's own reach. It does NOT incorporate Player 1's decisions.
      
      # So an infostate is reachable for player i if:
      # 1. Player i's own parent sequence has non-zero realization probability.
      # 2. Player 1-i's parent sequence leading here has non-zero realization probability.
      
      # We just need to check if total_weight > 0 where total_weight is calculated 
      # from the OPPONENT'S realization plans for these sequences.
      # Wait, no. The behavioral policy is x(s,a) / x(s).
      # If x(s) == 0, the state is unreachable by player i's own strategy.
      # If the state is unreachable by player 1-i's strategy, x(s) might still be > 0!
      
      # Total realization probability of this infostate s for player i:
      # x(s) = sum_a x(s,a). This x(s) is in primal_solutions[1] (for P0) or [0] (for P1).
      
      own_weight = 0
      for isa_key in infoset_action_maps[i][info_state]:
        own_weight += get_realization_prob(i, isa_key)
        
      # Opponent realization probability leading to this infostate:
      opponent_weight = get_realization_prob(1 - i, opponent_isa_key)
      
      total_reach = own_weight * opponent_weight
      
      state_policy = policies[i].policy_for_key(info_state)
      if total_reach > 1e-8:
        for isa_key in infoset_action_maps[i][info_state]:
          rel_weight = get_realization_prob(i, isa_key)
          _, action_str = isa_key.split(_DELIMITER)
          action = int(action_str)
          state_policy[action] = rel_weight / own_weight if own_weight > 0 else 1.0/len(infoset_action_maps[i][info_state])
      else:
        # State is unreachable in equilibrium. Use subgame-perfect optimal actions.
        optimal_actions = []
        for isa_key in infoset_action_maps[i][info_state]:
          slack = lps[i].get_slack(isa_key, primal_solutions[i])
          if abs(slack) < 1e-7:
            _, action_str = isa_key.split(_DELIMITER)
            optimal_actions.append(int(action_str))

        state_policy.fill(0.0)
        if not optimal_actions:
          prob = 1.0 / len(infoset_action_maps[i][info_state])
          for isa_key in infoset_action_maps[i][info_state]:
            _, action_str = isa_key.split(_DELIMITER)
            state_policy[int(action_str)] = prob
        else:
          prob = 1.0 / len(optimal_actions)
          for action in optimal_actions:
            state_policy[action] = prob

  return (primal_solutions[0][lps[0].get_var_id(_EMPTY_INFOSET_KEYS[0])],
          primal_solutions[1][lps[1].get_var_id(_EMPTY_INFOSET_KEYS[1])],
          policies[0], policies[1])
