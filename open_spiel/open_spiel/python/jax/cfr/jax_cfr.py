# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX implementation of the counterfactual regret minimization algorithm usable with GPU acceleration.

Uses same CFR setting as open_spiel.python.algorithms.cfr._CFRSolverBase and the
usability should be interchangable.

The results may slightly differ between these 2 versions due to rounding errors
when computing regrets (rounding regrets smaller than epsilon to zero results in
exactly the same results)

The algorithm performs well in short but wide games, with small amount of
illegal actions and poorly in long games with a lot of illegal actions.
"""

# pylint: disable=g-importing-member

from collections import namedtuple
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python import policy
import pyspiel

JAX_CFR_SIMULTANEOUS_UPDATE = -5


def regret_matching(regret, mask):
  """Computes current policy based on current regrets.

  Args:
    regret: Current regrets in array Fkiat[Isets, Actions]
    mask: Legal action mask Bool[Isets, Actions]

  Returns:
    policy: the policy.
  """
  regret = jnp.maximum(regret, 0) * mask
  total = jnp.sum(regret, axis=-1, keepdims=True)

  return jnp.where(total > 0.0, regret / total, 1.0 / jnp.sum(mask)) * mask


def update_regrets_plus(regret):
  """Clamps the regrets to be non-negative."""
  return regret * (regret > 0)


def update_regrets(regret):
  """Updates the regrets without CFRPlus."""
  return regret


@chex.dataclass(frozen=True)
class JaxCFRConstants:
  """Constants for JaxCFR."""

  players: int
  max_depth: int
  # This includes chance outcomes!  TODO: We could do this separately for each
  # depth to make less computations.
  max_actions: int

  max_iset_depth: chex.ArrayTree = ()  # Is just a list of integers
  isets: chex.ArrayTree = ()  # Is just a list of integers

  depth_history_utility: chex.ArrayTree = ()
  depth_history_iset: chex.ArrayTree = ()
  depth_history_actions: chex.ArrayTree = ()
  depth_history_previous_iset: chex.ArrayTree = ()
  depth_history_previous_action: chex.ArrayTree = ()

  depth_history_next_history: chex.ArrayTree = ()
  depth_history_player: chex.ArrayTree = ()
  depth_history_chance: chex.ArrayTree = ()
  depth_history_previous_history: chex.ArrayTree = ()
  depth_history_action_mask: chex.ArrayTree = ()
  depth_history_chance_probabilities: chex.ArrayTree = ()

  iset_previous_action: chex.ArrayTree = ()
  iset_action_mask: chex.ArrayTree = ()
  iset_action_depth: chex.ArrayTree = ()


class JaxCFR:
  """Class for CFR and CFR.

  First it prepares all the structures in `init`, then it just reuses them
  within jitted function `jit_step`.
  """

  def __init__(
      self,
      game: pyspiel.Game,
      regret_matching_plus=True,
      alternating_updates=True,
      linear_averaging=True,
  ):
    self.game = game
    self._regret_matching_plus = regret_matching_plus
    self._alternating_updates = alternating_updates
    self._linear_averaging = linear_averaging
    self.timestep = 1

    self.init()

  def init(self):
    """Constructor."""

    players = self.game.num_players()
    depth_history_utility = [[] for _ in range(players)]
    depth_history_previous_iset = [[] for _ in range(players)]
    depth_history_previous_action = [[] for _ in range(players)]
    depth_history_iset = [[] for _ in range(players)]
    depth_history_actions = [[] for _ in range(players)]
    depth_history_next_history = []
    depth_history_player = []
    depth_history_chance = []
    depth_history_previous_history = []
    depth_history_action_mask = []
    depth_history_chance_probabilities = []
    # Previous action is mapping of both iset and action!
    iset_previous_action = [[] for _ in range(players)]
    iset_action_mask = [[] for _ in range(players)]
    iset_action_depth = [[] for _ in range(players)]
    ids = [0 for _ in range(players)]
    pl_isets = [{} for _ in range(players)]
    distinct_actions = max(
        self.game.num_distinct_actions(), self.game.max_chance_outcomes()
    )

    for pl in range(players):
      pl_isets[pl][''] = ids[pl]
      ids[pl] += 1
      am = [0] * distinct_actions
      am[0] = 1
      iset_action_mask[pl].append(am)
      iset_previous_action[pl].append(0)
      iset_action_depth[pl].append(0)

    PreviousInfo = namedtuple(
        'PreviousInfo',
        ('actions', 'isets', 'prev_actions', 'history', 'player'),
    )

    def _traverse_tree(state, previous_info, depth, chance=1.0):

      if len(depth_history_next_history) <= depth:
        for pl in range(players):
          depth_history_utility[pl].append([])
          depth_history_previous_iset[pl].append([])
          depth_history_previous_action[pl].append([])
          depth_history_iset[pl].append([])
          depth_history_actions[pl].append([])

        depth_history_next_history.append([])
        depth_history_player.append([])
        depth_history_chance.append([])
        depth_history_previous_history.append([])
        depth_history_action_mask.append([])
        depth_history_chance_probabilities.append([])

      history_id = len(depth_history_previous_history[depth])

      next_history_temp = [0] * distinct_actions
      depth_history_next_history[depth].append(next_history_temp)
      depth_history_player[depth].append(state.current_player())
      depth_history_chance[depth].append(chance)
      depth_history_previous_history[depth].append(previous_info.history)

      actions_mask = [0] * distinct_actions
      for a in state.legal_actions():
        actions_mask[a] = 1
      depth_history_action_mask[depth].append(actions_mask)
      chance_probabilities = [0.0 for _ in range(distinct_actions)]
      if state.is_chance_node():
        for a, prob in state.chance_outcomes():
          chance_probabilities[a] = prob
      elif not state.is_terminal():
        chance_probabilities = [1.0 for _ in range(distinct_actions)]
      else:
        chance_probabilities = [
            1.0 / distinct_actions for _ in range(distinct_actions)
        ]

      depth_history_chance_probabilities[depth].append(chance_probabilities)
      for pl in range(players):
        depth_history_utility[pl][depth].append(
            state.rewards()[pl] if not state.is_chance_node() else 0.0
        )
        depth_history_previous_iset[pl][depth].append(previous_info.isets[pl])
        depth_history_previous_action[pl][depth].append(
            previous_info.actions[pl]
        )
        if state.current_player() == pl:
          iset = state.information_state_string()
          if iset not in pl_isets[pl]:
            pl_isets[pl][iset] = ids[pl]
            ids[pl] += 1
            iset_previous_action[pl].append(previous_info.actions[pl])
            iset_action_mask[pl].append(actions_mask)
            iset_action_depth[pl].append(previous_info.prev_actions[pl])
          depth_history_iset[pl][depth].append(pl_isets[pl][iset])
          depth_history_actions[pl][depth].append([
              i + pl_isets[pl][iset] * distinct_actions
              for i in range(distinct_actions)
          ])
        else:
          depth_history_iset[pl][depth].append(0)
          depth_history_actions[pl][depth].append(
              [0 for _ in range(distinct_actions)]
          )

      for a in state.legal_actions():
        new_chance = chance * chance_probabilities[a]
        assert new_chance > 0.0
        new_actions = tuple(
            previous_info.actions[pl]
            if state.current_player() != pl
            else pl_isets[pl][iset] * distinct_actions + a
            for pl in range(players)
        )
        new_infosets = tuple(
            previous_info.isets[pl]
            if state.current_player() != pl
            else pl_isets[pl][iset]
            for pl in range(players)
        )
        new_prev_actions = tuple(
            previous_info.prev_actions[pl] + int(state.current_player() == pl)
            for pl in range(players)
        )
        new_info = PreviousInfo(
            new_actions,
            new_infosets,
            new_prev_actions,
            history_id,
            state.current_player(),
        )
        new_state = state.clone()
        new_state.apply_action(a)

        # simple workaround if the next element was not visited yet
        next_history_temp[a] = (
            len(depth_history_player[depth + 1])
            if len(depth_history_player) > depth + 1
            else 0
        )

        _traverse_tree(new_state, new_info, depth + 1, new_chance)

    s = self.game.new_initial_state()
    _traverse_tree(
        s,
        PreviousInfo(
            tuple(0 for _ in range(players)),
            tuple(0 for _ in range(players)),
            tuple(0 for _ in range(players)),
            0,
            0,
        ),
        0,
    )

    def convert_to_jax(x):
      return [jnp.asarray(i) for i in x]

    def convert_to_jax_players(x):
      return [[jnp.asarray(i) for i in x[pl]] for pl in range(players)]

    depth_history_utility = convert_to_jax_players(depth_history_utility)
    depth_history_iset = convert_to_jax_players(depth_history_iset)
    depth_history_previous_iset = convert_to_jax_players(
        depth_history_previous_iset
    )
    depth_history_actions = convert_to_jax_players(depth_history_actions)
    depth_history_previous_action = convert_to_jax_players(
        depth_history_previous_action
    )

    depth_history_next_history = convert_to_jax(depth_history_next_history)
    depth_history_player = convert_to_jax(depth_history_player)
    depth_history_chance = convert_to_jax(depth_history_chance)
    depth_history_previous_history = convert_to_jax(
        depth_history_previous_history
    )
    depth_history_chance_probabilities = convert_to_jax(
        depth_history_chance_probabilities
    )
    depth_history_action_mask = convert_to_jax(depth_history_action_mask)

    max_iset_depth = [np.max(iset_action_depth[pl]) for pl in range(players)]
    iset_previous_action = convert_to_jax(iset_previous_action)
    iset_action_mask = convert_to_jax(iset_action_mask)
    iset_action_depth = convert_to_jax(iset_action_depth)

    self.constants = JaxCFRConstants(
        players=players,
        max_depth=int(len(depth_history_utility[0])),
        max_actions=distinct_actions,
        max_iset_depth=max_iset_depth,
        isets=ids,
        depth_history_utility=depth_history_utility,
        depth_history_iset=depth_history_iset,
        depth_history_actions=depth_history_actions,
        depth_history_previous_iset=depth_history_previous_iset,
        depth_history_previous_action=depth_history_previous_action,
        depth_history_next_history=depth_history_next_history,
        depth_history_player=depth_history_player,
        depth_history_chance=depth_history_chance,
        depth_history_previous_history=depth_history_previous_history,
        depth_history_action_mask=depth_history_action_mask,
        depth_history_chance_probabilities=depth_history_chance_probabilities,
        iset_previous_action=iset_previous_action,
        iset_action_mask=iset_action_mask,
        iset_action_depth=iset_action_depth,
    )

    self.regrets = [
        jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)
    ]
    self.averages = [
        jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)
    ]

    self.regret_matching = jax.vmap(regret_matching, 0, 0)
    if self._regret_matching_plus:
      self.update_regrets = jax.vmap(update_regrets_plus, 0, 0)
    else:
      self.update_regrets = jax.vmap(update_regrets, 0, 0)

    self.iset_map = pl_isets

  def multiple_steps(self, iterations: int):
    """Performs several CFR steps.

    Args:
      iterations: Amount of CFR steps, the solver should do.
    """
    for _ in range(iterations):
      self.step()

  def evaluate_and_update_policy(self):
    """Wrapper to step().

    Ensures interchangability with
    open_spiel.python.algorithms.cfr._CFRSolverBase.
    """
    self.step()

  def step(self):
    """Wrapper around the jitted function for performing CFR step."""
    averaging_coefficient = self.timestep if self._linear_averaging else 1
    if self._alternating_updates:
      for player in range(self.constants.players):
        self.regrets, self.averages = self.jit_step(
            self.regrets, self.averages, averaging_coefficient, player
        )

    else:
      self.regrets, self.averages = self.jit_step(
          self.regrets,
          self.averages,
          averaging_coefficient,
          JAX_CFR_SIMULTANEOUS_UPDATE,
      )

    self.timestep += 1

  def propagate_strategy(self, current_strategies):
    """Propagtes the strategies withing infosets.

    Args:
      current_strategies: Current strategies for all players, list[Float[Isets,
        Actions]]
    Returns:
      realization_plans: the realization plans.
    """
    realization_plans = [
        jnp.ones_like(current_strategies[pl])
        for pl in range(self.constants.players)
    ]

    for pl in range(self.constants.players):
      for i in range(0, self.constants.max_iset_depth[pl] + 1):
        realization_plans[pl] = jnp.where(
            self.constants.iset_action_depth[pl][..., jnp.newaxis] == i,
            current_strategies[pl]
            * realization_plans[pl].ravel()[
                self.constants.iset_previous_action[pl]
            ][..., jnp.newaxis],
            realization_plans[pl],
        )

    return realization_plans

  @functools.partial(jax.jit, static_argnums=(0,))
  def jit_step(
      self, regrets, averages, average_policy_update_coefficient, player
  ):
    """Performs the CFR step.

    This consists of:
    1. Computes the current strategies based on regrets
    2. Computes the realization plan for each action from top of the tree down
    3. Compute the counterfactual regrets from bottom of the tree up
    4. Updates regrets and average stretegies

    Args:
      regrets: Cummulative regrets for all players, list[Float[Isets, Actions]]
      averages: Average strategies for all players, list[Float[Isets, Actions]]
      average_policy_update_coefficient: Weight of the average policy update.
        When enabled linear_averging it is equal to current iteration. Otherwise
        1, int
      player: Player for which the update should be done. When alternating
        updates are distables, it is JAX_CFR_SIMULTANEOUS_UPDATE

    Returns:
      regrets: the regrets.
      averages: the averages.
    """
    current_strategies = [
        self.regret_matching(regrets[pl], self.constants.iset_action_mask[pl])
        for pl in range(self.constants.players)
    ]

    realization_plans = self.propagate_strategy(current_strategies)
    iset_reaches = [
        jnp.sum(realization_plans[pl], -1)
        for pl in range(self.constants.players)
    ]
    # In last row, there are only terminal, so we start row before it
    depth_utils = [
        [self.constants.depth_history_utility[pl][-1]]
        for pl in range(self.constants.players)
    ]
    for i in range(self.constants.max_depth - 2, -1, -1):

      each_history_policy = self.constants.depth_history_chance_probabilities[i]
      for pl in range(self.constants.players):
        each_history_policy = each_history_policy * jnp.where(
            self.constants.depth_history_player[i][..., jnp.newaxis] == pl,
            current_strategies[pl][self.constants.depth_history_iset[pl][i]],
            1,
        )

      for pl in range(self.constants.players):
        action_value = jnp.where(
            self.constants.depth_history_player[i][..., jnp.newaxis] == -4,
            self.constants.depth_history_utility[pl][i][..., jnp.newaxis],
            depth_utils[pl][-1][self.constants.depth_history_next_history[i]],
        )
        history_value = jnp.sum(action_value * each_history_policy, -1)
        regret = (
            (action_value - history_value[..., jnp.newaxis])
            * self.constants.depth_history_action_mask[i]
            * (self.constants.depth_history_player[i][..., jnp.newaxis] == pl)
            * self.constants.depth_history_chance[i][..., jnp.newaxis]
        )
        for pl2 in range(self.constants.players):
          if pl != pl2:
            regret = (
                regret
                * realization_plans[pl2].ravel()[
                    self.constants.depth_history_previous_action[pl2][i]
                ][..., jnp.newaxis]
            )
        bin_regrets = jnp.bincount(
            self.constants.depth_history_actions[pl][i].ravel(),
            regret.ravel(),
            length=self.constants.isets[pl] * self.constants.max_actions,
        )
        bin_regrets = bin_regrets.reshape(-1, self.constants.max_actions)
        regrets[pl] = jnp.where(
            jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE),
            regrets[pl] + bin_regrets,
            regrets[pl],
        )
        depth_utils[pl].append(history_value)

    regrets = [
        self.update_regrets(regrets[pl]) for pl in range(self.constants.players)
    ]

    averages = [
        jnp.where(
            jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE),
            averages[pl]
            + current_strategies[pl]
            * iset_reaches[pl][..., jnp.newaxis]
            * average_policy_update_coefficient,
            averages[pl],
        )
        for pl in range(self.constants.players)
    ]

    return regrets, averages

  def average_policy(self):
    """Extracts the average policy from JAX structures into a TabularPolicy."""
    averages = [
        np.asarray(self.averages[pl]) for pl in range(self.constants.players)
    ]
    averages = [
        averages[pl] / np.sum(averages[pl], -1, keepdims=True)
        for pl in range(self.constants.players)
    ]

    avg_strategy = policy.TabularPolicy(self.game)

    for pl in range(2):
      for iset, val in self.iset_map[pl].items():
        if not iset:
          continue
        state_policy = avg_strategy.policy_for_key(iset)
        for i in range(len(state_policy)):
          state_policy[i] = averages[pl][val][i]
    return avg_strategy

