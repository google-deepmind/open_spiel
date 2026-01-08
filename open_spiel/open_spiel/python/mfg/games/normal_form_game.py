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

# Lint as python3
"""Mean Field Normal Form Games / Static Mean-Field Games."""

from typing import Any, List, Mapping

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame  # pylint:disable=g-importing-member
import pyspiel


def coop_reward(last_action, distribution):
  """A game incentivising cooperation."""
  nu_a, nu_b, nu_c, *_ = distribution
  if last_action == 0:
    return 10 * nu_a - 200 / 9 * (nu_a - nu_c) * nu_c - 20 * nu_b
  elif last_action == 1:
    return 20 * (nu_a - nu_b) - 2380 * nu_c
  elif last_action == 2:
    return 2000 / 9 * (nu_a - nu_c) * nu_c
  else:
    raise ValueError("Unknown last action " + str(last_action))


def biased_indirect_rps(last_action, distribution):
  """Biased indirect Rock Paper Scissors."""
  nu_a = 0.7 * distribution[0]
  nu_b = 0.5 * distribution[1]
  nu_c = 0.3 * distribution[2]
  if last_action == 0:
    return nu_b - nu_c
  elif last_action == 1:
    return nu_c - nu_a
  elif last_action == 2:
    return nu_a - nu_b
  else:
    raise ValueError("Unknown last action " + str(last_action))


def dominated_reward_source(last_action, distribution):
  nu_a, nu_b, nu_c, *_ = distribution
  if last_action == 0:
    return nu_a + nu_c
  elif last_action == 1:
    return nu_b
  elif last_action == 2:
    return nu_a + nu_c - 0.25
  else:
    raise ValueError("Unknown last action " + str(last_action))


_NUM_PLAYERS = 1
_NUM_ACTIONS = 3
_DEFAULT_PARAMS = {"num_actions": _NUM_ACTIONS, "reward_function": "coop"}
_GAME_TYPE = pyspiel.GameType(
    short_name="mean_field_nfg",
    long_name="Mean-Field Normal-Form Game",
    dynamics=pyspiel.GameType.Dynamics.MEAN_FIELD,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification=_DEFAULT_PARAMS,
)


class MFGNormalFormGame(pyspiel.Game):
  """A Mean Field Normal Form game.

  A game starts by an initial chance node that select the initial state
  of the MFG.
  Then the game sequentially alternates between:
    - An action selection node (Where the player Id >= 0)
    - A chance node (the player id is pyspiel.PlayerId.CHANCE)
    - A Mean Field node (the player id is pyspiel.PlayerId.MEAN_FIELD)
  """

  # pylint:disable=dangerous-default-value
  def __init__(self, params: Mapping[str, Any] = _DEFAULT_PARAMS):
    game_info = pyspiel.GameInfo(
        num_distinct_actions=_NUM_ACTIONS,
        max_chance_outcomes=_NUM_ACTIONS,
        num_players=_NUM_PLAYERS,
        min_utility=-np.inf,
        max_utility=+np.inf,
        utility_sum=0.0,
        max_game_length=2,
    )
    super().__init__(_GAME_TYPE, game_info, params)
    if params["reward_function"] == "coop":
      self.reward_function = coop_reward
    elif params["reward_function"] == "dom":
      self.reward_function = dominated_reward_source
    elif params["reward_function"] == "biased_indirect_rps":
      self.reward_function = biased_indirect_rps
    else:
      raise ValueError("Unknown reward function " + params["reward_function"])
    self.num_actions = params["num_actions"]
    self.size = 1 + self.num_actions

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return MFGNormalFormState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if (iig_obs_type is None) or (
        iig_obs_type.public_info and not iig_obs_type.perfect_recall
    ):
      return Observer(params, self)
    return IIGObserverForPublicInfoGame(iig_obs_type, params)

  def max_chance_nodes_in_history(self):
    """Maximun chance nodes in game history."""
    return 0


class MFGNormalFormState(pyspiel.State):
  """A Mean Field Normal-Form state."""

  def __init__(self, game, last_action=None):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
    self._last_action = last_action
    self._num_actions = game.num_actions
    self.reward_function = game.reward_function
    self.size = game.size
    self._terminal = False

    # Represents the current probability distribution over game states.
    # Initialized with a uniform distribution.
    self._distribution = [1.0 / self.size for _ in range(self.size)]

  def state_to_str(self, player_id=pyspiel.PlayerId.DEFAULT_PLAYER_ID):
    """A string that uniquely identify a triplet x, t, player_id."""
    if self._last_action is None:
      return "initial"
    else:
      bonus = "_final" if self.is_terminal() else ""
      return str(self._last_action) + bonus

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def _legal_actions(self, player):
    """Returns a list of legal actions for player and MFG nodes."""
    if player == pyspiel.PlayerId.MEAN_FIELD:
      return []
    if (
        player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
        and player == self.current_player()
    ):
      return list(range(self._num_actions))
    raise ValueError(
        f"Unexpected player {player}. "
        "Expected a mean field or current player 0."
    )

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self._player_id == pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "_apply_action should not be called at a MEAN_FIELD state."
      )
    self.return_value = self._rewards()

    assert self._player_id == 0
    # Here the action is between 0 and N-1
    if action < 0 or action > self._num_actions - 1:
      raise ValueError(
          "The action is between 0 and {} at any node".format(
              self._num_actions - 1
          )
      )
    self._last_action = action
    self._player_id = pyspiel.PlayerId.MEAN_FIELD

  def _action_to_string(self, player, action):
    """Action -> string."""
    del player
    return str(action)

  def distribution_support(self):
    """return a list of state string."""
    if self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      return [self.state_to_str()]
    elif self._player_id == pyspiel.PlayerId.MEAN_FIELD:
      return [str(i) for i in range(self._num_actions)]

  def update_distribution(self, distribution):
    """This function is central and specific to the logic of the MFG.

    Args:
      distribution: a distribution to register.  - function should be called
        when the node is in MEAN_FIELD state. - distribution are probabilities
        that correspond to each game state given by distribution_support.
    """
    if self._player_id != pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "update_distribution should only be called at a MEAN_FIELD state."
      )
    self._distribution = distribution.copy()
    self._player_id = pyspiel.PlayerId.TERMINAL

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._player_id == pyspiel.PlayerId.TERMINAL

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return self._player_id

  def _rewards(self):
    """Reward for the player for this state."""
    reward = 0.0
    if self._player_id == pyspiel.PlayerId.TERMINAL:
      reward = self.reward_function(self._last_action, self._distribution)
    return reward

  def rewards(self) -> List[float]:
    """Rewards for all players."""
    # For now, only single-population (single-player) mean field games
    # are supported.
    return [self._rewards()]

  def _returns(self):
    """Returns is the sum of all payoffs collected so far."""
    return self._rewards()

  def returns(self) -> List[float]:
    """Returns for all players."""
    # For now, only single-population (single-player) mean field games
    # are supported.
    return [self._returns()]

  def __str__(self):
    """A string that uniquely identify the current state."""
    return self.state_to_str(player_id=self._player_id)


class Observer:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params, game):
    """Initializes an empty observation tensor."""
    del params

    self.size = game.size
    # +1 to allow t == horizon.
    self.tensor = np.array([])
    self.dict = {}

  def set_from(self, state: MFGNormalFormState, player: int):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    del state
    self.tensor.fill(0)
    # state.x is None for the initial (blank) state, don't set any
    # position bit in that case.
    pass

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return str(state)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, MFGNormalFormGame)
