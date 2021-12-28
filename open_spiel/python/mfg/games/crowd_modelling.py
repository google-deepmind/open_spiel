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
"""Mean Field Crowd Modelling, implemented in Python.

This is a demonstration of implementing a mean field game in Python.

Fictitious play for mean field games: Continuous time analysis and applications,
Perrin & al. 2019 (https://arxiv.org/abs/2007.03458). This game corresponds
to the game in section 4.2.
"""

from typing import Any, List, Mapping
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 1
_SIZE = 10
_HORIZON = 10
_NUM_ACTIONS = 3
_NUM_CHANCE = 3
_EPSILON = 10**(-25)
_DEFAULT_PARAMS = {"size": _SIZE, "horizon": _HORIZON}
_GAME_TYPE = pyspiel.GameType(
    short_name="python_mfg_crowd_modelling",
    long_name="Python Mean Field Crowd Modelling",
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
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS)


class MFGCrowdModellingGame(pyspiel.Game):
  """A Mean Field Crowd Modelling game.


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
        max_chance_outcomes=max(params["size"], _NUM_CHANCE),
        num_players=_NUM_PLAYERS,
        min_utility=-np.inf,
        max_utility=+np.inf,
        utility_sum=0.0,
        max_game_length=params["horizon"])
    super().__init__(_GAME_TYPE, game_info, params)
    self.size = params["size"]
    self.horizon = params["horizon"]

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return MFGCrowdModellingState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return Observer(params, self)
    return IIGObserverForPublicInfoGame(iig_obs_type, params)

  def max_chance_nodes_in_history(self):
    """Maximun chance nodes in game history."""
    return self.horizon + 1


class MFGCrowdModellingState(pyspiel.State):
  """A Mean Field Crowd Modelling state."""

  # Maps legal actions to the corresponding move along the 1-D axis of the game.
  _ACTION_TO_MOVE = {0: -1, 1: 0, 2: 1}
  # Action that corresponds to no displacement.
  _NEUTRAL_ACTION = 1

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._is_chance_init = True  # is true for the first state of the game.
    self._player_id = pyspiel.PlayerId.CHANCE
    self._x = None
    self._t = 0
    # We initialize last_action to the neutral action. This makes sure
    # that the first reward does not include any displacement penalty.
    self._last_action = self._NEUTRAL_ACTION
    self.size = game.size
    self.horizon = game.horizon
    self.return_value = 0.0

    # Represents the current probability distribution over game states.
    # Initialized with a uniform distribution.
    self._distribution = [1. / self.size for i in range(self.size)]

  @property
  def x(self):
    return self._x

  @property
  def t(self):
    return self._t

  def state_to_str(self, x, t, player_id=pyspiel.PlayerId.DEFAULT_PLAYER_ID):
    """A string that uniquely identify a triplet x, t, player_id."""
    if self._is_chance_init:
      return "initial"
    if player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      return str((x, t))
    if player_id == pyspiel.PlayerId.MEAN_FIELD:
      return str((x, t)) + "_a"
    if player_id == pyspiel.PlayerId.CHANCE:
      return str((x, t)) + "_a_mu"
    raise ValueError(
        "player_id is not mean field, chance or default player id.")

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def _legal_actions(self, player):
    """Returns a list of legal actions for player and MFG nodes."""
    if player == pyspiel.PlayerId.MEAN_FIELD:
      return []
    if (player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
        and player == self.current_player()):
      return [0, 1, 2]
    raise ValueError(f"Unexpected player {player}. "
                     "Expected a mean field or current player 0.")

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    if self._is_chance_init:
      return list(enumerate(self._distribution))
    return [(0, 1. / 3.), (1, 1. / 3.), (2, 1. / 3.)]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self._player_id == pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "_apply_action should not be called at a MEAN_FIELD state.")
    self.return_value += self._rewards()
    if self._is_chance_init:
      # Here the action is between 0 and self.size - 1
      if action < 0 or action >= self.size:
        raise ValueError(
            "The action is between 0 and self.size - 1 at an init chance node")
      self._x = action
      self._is_chance_init = False
      self._player_id = 0
    elif self._player_id == pyspiel.PlayerId.CHANCE:
      # Here the action is between 0 and 2
      if action < 0 or action > 2:
        raise ValueError(
            "The action is between 0 and 2 at any chance node")
      self._x = (self.x + self._ACTION_TO_MOVE[action]) % self.size
      self._t += 1
      self._player_id = pyspiel.PlayerId.MEAN_FIELD
    elif self._player_id == 0:
      # Here the action is between 0 and 2
      if action < 0 or action > 2:
        raise ValueError(
            "The action is between 0 and 2 at any chance node")
      self._x = (self.x + self._ACTION_TO_MOVE[action]) % self.size
      self._last_action = action
      self._player_id = pyspiel.PlayerId.CHANCE

  def _action_to_string(self, player, action):
    """Action -> string."""
    del player
    if self.is_chance_node() and self._is_chance_init:
      return f"init_state={action}"
    return str(self._ACTION_TO_MOVE[action])

  def distribution_support(self):
    """return a list of state string."""
    return [
        self.state_to_str(
            i, self.t, player_id=pyspiel.PlayerId.MEAN_FIELD)
        for i in range(self.size)
    ]

  def update_distribution(self, distribution):
    """This function is central and specific to the logic of the MFG.

    Args:
      distribution: a distribution to register.

      - function should be called when the node is in MEAN_FIELD state.
      - distribution are probabilities that correspond to each game state
      given by distribution_support.

    """
    if self._player_id != pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "update_distribution should only be called at a MEAN_FIELD state.")
    self._distribution = distribution.copy()
    self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID

  def is_terminal(self):
    """Returns True if the game is over."""
    return self.t >= self.horizon

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return self._player_id

  def _rewards(self):
    """Reward for the player for this state."""
    if self._player_id == 0:
      r_x = 1 - (1.0 * np.abs(self.x - self.size // 2)) / (self.size // 2)
      r_a = -(1.0 * np.abs(self._ACTION_TO_MOVE[self._last_action])) / self.size
      r_mu = - np.log(self._distribution[self.x] + _EPSILON)
      return r_x + r_a + r_mu
    return 0.0

  def rewards(self) -> List[float]:
    """Rewards for all players."""
    # For now, only single-population (single-player) mean field games
    # are supported.
    return [self._rewards()]

  def _returns(self):
    """Returns is the sum of all payoffs collected so far."""
    return self.return_value + self._rewards()

  def returns(self) -> List[float]:
    """Returns for all players."""
    # For now, only single-population (single-player) mean field games
    # are supported.
    return [self._returns()]

  def __str__(self):
    """A string that uniquely identify the current state."""
    return self.state_to_str(self.x, self.t, player_id=self._player_id)


class Observer:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params, game):
    """Initializes an empty observation tensor."""
    del params

    self.size = game.size
    self.horizon = game.horizon
    # +1 to allow t == horizon.
    self.tensor = np.zeros(self.size + self.horizon + 1, np.float32)
    self.dict = {"x": self.tensor[:self.size], "t": self.tensor[self.size:]}

  def set_from(self, state: MFGCrowdModellingState, player: int):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    self.tensor.fill(0)
    # state.x is None for the initial (blank) state, don't set any
    # position bit in that case.
    if state.x is not None:
      if not 0 <= state.x < self.size:
        raise ValueError(
            f"Expected {state} x position to be in [0, {self.size})")
      self.dict["x"][state.x] = 1
    if not 0 <= state.t <= self.horizon:
      raise ValueError(f"Expected {state} time to be in [0, {self.horizon}]")
    self.dict["t"][state.t] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return str(state)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, MFGCrowdModellingGame)
