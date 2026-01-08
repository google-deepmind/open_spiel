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
"""Solving strong Stackelberg equilibrium based on linear programming.

Based on [1] "Computing the Optimal Strategy to Commit to", Conitzer & Sandholm,
EC'06
"""

import cvxpy as cp
import numpy as np

from open_spiel.python.algorithms.projected_replicator_dynamics import _simplex_projection
from open_spiel.python.egt.utils import game_payoffs_array


def solve_stackelberg(game, is_first_leader=True):
  """Solves the optimal mixed strategty to commit to for the leader.

  Args:
    game: a pyspiel game,
    is_first_leader: if true, then player 0 is the leader, o.w. player 1 is
      the leader.

  Returns:
    (player0 strategy, player1 strategy, player0 payoff, player1 payoff) at an
    SSE.
  """
  p_mat = game_payoffs_array(game)
  assert len(p_mat) == 2
  if is_first_leader:
    leader_payoff, follower_payoff = p_mat[0], p_mat[1]
  else:
    leader_payoff, follower_payoff = p_mat[1].T, p_mat[0].T

  num_leader_strategies, num_follower_strategies = leader_payoff.shape

  leader_eq_value = -float("inf")
  follower_eq_value = None
  leader_eq_strategy = None
  follower_eq_strategy = None

  for t in range(num_follower_strategies):
    p_s = cp.Variable(num_leader_strategies, nonneg=True)
    constraints = [p_s <= 1, cp.sum(p_s) == 1]
    for t_ in range(num_follower_strategies):
      if t_ == t:
        continue
      constraints.append(
          p_s @ follower_payoff[:, t_] <= p_s @ follower_payoff[:, t]
      )
    prob = cp.Problem(cp.Maximize(p_s @ leader_payoff[:, t]), constraints)
    prob.solve()
    p_s_value = p_s.value
    if p_s_value is None:
      continue
    leader_strategy = _simplex_projection(p_s.value.reshape(-1)).reshape(-1, 1)
    leader_value = leader_strategy.T.dot(leader_payoff)[0, t]
    if leader_value > leader_eq_value:
      leader_eq_strategy = leader_strategy
      follower_eq_strategy = t
      leader_eq_value = leader_value
      follower_eq_value = leader_strategy.T.dot(follower_payoff)[0, t]

  assert leader_eq_strategy is not None, p_mat
  if is_first_leader:
    return (
        leader_eq_strategy.reshape(-1),
        np.identity(num_follower_strategies)[follower_eq_strategy],
        leader_eq_value,
        follower_eq_value,
    )
  else:
    return (np.identity(num_follower_strategies)[follower_eq_strategy],
            leader_eq_strategy.reshape(-1), follower_eq_value, leader_eq_value)
