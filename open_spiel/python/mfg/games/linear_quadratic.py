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
"""Mean Field Linear Quadratic, implemented in Python.

This is a demonstration of implementing a mean field game in Python.

Fictitious play for mean field games: Continuous time analysis and applications,
Perrin & al. 2019 (https://arxiv.org/abs/2007.03458). This game corresponds
to the game in section 4.1.
"""
import math
from typing import Any, List, Mapping

import numpy as np
import scipy.stats

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 1
_SIZE = 10
_HORIZON = 10
_MEAN_REVERT = 0.0
_VOLATILITY = 1.0
_CROSS_Q = 0.01
_KAPPA = 0.5
_TERMINAL_COST = 1.0
_DELTA_T = 1.0  # 3.0/_HORIZON
_N_ACTIONS_PER_SIDE = 3
_SPATIAL_BIAS = 0

_DEFAULT_PARAMS = {
    "size": _SIZE,
    "horizon": _HORIZON,
    "dt": _DELTA_T,
    "n_actions_per_side": _N_ACTIONS_PER_SIDE,
    "volatility": _VOLATILITY,
    "mean_revert": _MEAN_REVERT,
    "cross_q": _CROSS_Q,
    "kappa": _KAPPA,
    "terminal_cost": _TERMINAL_COST,
    "spatial_bias": _SPATIAL_BIAS
}

_GAME_TYPE = pyspiel.GameType(
    short_name="mean_field_lin_quad",
    long_name="Mean-Field Linear Quadratic Game",
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


class MFGLinearQuadraticGame(pyspiel.Game):
  """A Mean-Field Linear QUadratic game.


    A game starts by an initial chance node that select the initial state
    of the player in the MFG.
    Then the game sequentially alternates between:
      - An action selection node (Where the player Id >= 0)
      - A chance node (the player id is pyspiel.PlayerId.CHANCE)
      - A Mean Field node (the player id is pyspiel.PlayerId.MEAN_FIELD)
  """

  # pylint:disable=dangerous-default-value
  def __init__(self, params: Mapping[str, Any] = _DEFAULT_PARAMS):
    self.size = params.get("size", _SIZE)
    self.horizon = params.get("horizon", _HORIZON)
    self.dt = params.get("dt", _DELTA_T)
    self.n_actions_per_side = params.get("n_actions_per_side",
                                         _N_ACTIONS_PER_SIDE)
    self.volatility = params.get("volatility", _VOLATILITY)
    self.mean_revert = params.get("mean_revert", _MEAN_REVERT)
    self.cross_q = params.get("cross_q", _CROSS_Q)
    self.kappa = params.get("kappa", _KAPPA)
    self.terminal_cost = params.get("terminal_cost", _TERMINAL_COST)
    self.spatial_bias = params.get("spatial_bias", _SPATIAL_BIAS)

    game_info = pyspiel.GameInfo(
        num_distinct_actions=2 * self.n_actions_per_side + 1,
        max_chance_outcomes=2 * self.n_actions_per_side + 1,
        num_players=_NUM_PLAYERS,
        min_utility=-np.inf,
        max_utility=+np.inf,
        utility_sum=0.0,
        max_game_length=self.horizon)
    super().__init__(_GAME_TYPE, game_info, params)

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return MFGLinearQuadraticState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return Observer(params, self)
    return IIGObserverForPublicInfoGame(iig_obs_type, params)

  def max_chance_nodes_in_history(self):
    """Maximun chance nodes in game history."""
    return self.horizon + 1


class MFGLinearQuadraticState(pyspiel.State):
  """A Mean Field Normal-Form state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._player_id = pyspiel.PlayerId.CHANCE

    self._last_action = game.n_actions_per_side
    self.tick = 0
    self.x = None
    self.return_value = 0.0

    self.game = game

    self.size = game.size
    self.horizon = game.horizon
    self.dt = game.dt
    self.n_actions_per_side = game.n_actions_per_side
    self.volatility = game.volatility
    self.mean_revert = game.mean_revert
    self.cross_q = game.cross_q
    self.kappa = game.kappa
    self.terminal_cost = game.terminal_cost

    # Represents the current probability distribution over game states.
    # Initialized with a uniform distribution.
    self._distribution = [1. / self.size for i in range(self.size)]

  def to_string(self):
    return self.state_to_str(self.x, self.tick)

  def state_to_str(self, x, tick, player_id=pyspiel.PlayerId.DEFAULT_PLAYER_ID):
    """A string that uniquely identify a triplet x, t, player_id."""
    if self.x is None:
      return "initial"

    if self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      return "({}, {})".format(x, tick)
    elif self._player_id == pyspiel.PlayerId.MEAN_FIELD:
      return "({}, {})_a".format(x, tick)
    elif self._player_id == pyspiel.PlayerId.CHANCE:
      return "({}, {})_a_mu".format(x, tick)
    raise ValueError(
        "player_id is not mean field, chance or default player id.")

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  @property
  def n_actions(self):
    return 2 * self.n_actions_per_side + 1

  def _legal_actions(self, player):
    """Returns a list of legal actions for player and MFG nodes."""
    if player == pyspiel.PlayerId.MEAN_FIELD:
      return []
    if (player == pyspiel.PlayerId.DEFAULT_PLAYER_ID and
        player == self.current_player()):
      return list(range(self.n_actions))
    raise ValueError(f"Unexpected player {player}. "
                     "Expected a mean field or current player 0.")

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self._player_id == pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "_apply_action should not be called at a MEAN_FIELD state.")
    self.return_value = self._rewards()

    assert self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID or self._player_id == pyspiel.PlayerId.CHANCE

    if self.x is None:
      self.x = action
      self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
      return

    if action < 0 or action >= self.n_actions:
      raise ValueError("The action is between 0 and {} at any node".format(
          self.n_actions))

    move = self.action_to_move(action)
    if self._player_id == pyspiel.PlayerId.CHANCE:
      self.x += move * math.sqrt(self.dt) * self.volatility
      self.x = round(self.x) % self.size
      self._player_id = pyspiel.PlayerId.MEAN_FIELD
      self.tick += 1
    elif self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      dist_mean = (self.distribution_average() - self.x)
      full_move = move
      full_move += self.mean_revert * dist_mean
      full_move *= self.dt
      self.x += round(full_move)
      self.x = round(self.x) % self.size

      self._last_action = action
      self._player_id = pyspiel.PlayerId.CHANCE

  def _action_to_string(self, player, action):
    """Action -> string."""
    del player
    return str(action)

  def action_to_move(self, action):
    return action - self.n_actions_per_side

  def actions_to_position(self):
    return [a - self.n_actions_per_side for a in range(self.n_actions)]

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    if self.x is None:
      return list(enumerate(self._distribution))

    a = np.array(self.actions_to_position())
    gaussian_vals = scipy.stats.norm.cdf(
        a + 0.5, scale=self.volatility) - scipy.stats.norm.cdf(
            a - 0.5, scale=self.volatility)
    gaussian_vals[0] += scipy.stats.norm.cdf(
        a[0] - 0.5, scale=self.volatility) - 0.0
    gaussian_vals[-1] += 1.0 - scipy.stats.norm.cdf(
        a[-1] + 0.5, scale=self.volatility)
    return [
        (act, p) for act, p in zip(list(range(self.n_actions)), gaussian_vals)
    ]

  def distribution_support(self):
    """return a list of state string."""
    return [
        self.state_to_str(i, self.tick, player_id=pyspiel.PlayerId.MEAN_FIELD)
        for i in range(self.size)
    ]

  def distribution_average(self):
    """return the average of the distribution over the states: 0, ..., Size."""
    states = np.arange(self.size)
    pos = states * (self._distribution)
    return np.sum(pos)

  def update_distribution(self, distribution):
    """This function is central and specific to the logic of the MFG.

    Args:
      distribution: a distribution to register.  - function should be called
        when the node is in MEAN_FIELD state. - distribution are probabilities
        that correspond to each game state given by distribution_support.
    """
    if self._player_id != pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "update_distribution should only be called at a MEAN_FIELD state.")
    self._distribution = distribution.copy()
    self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID

  @property
  def t(self):
    return self.tick * self.dt

  def is_terminal(self):
    """Returns True if the game is over."""
    return self.t >= self.horizon

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return int(pyspiel.PlayerId.TERMINAL)
    return int(self._player_id)

  def eta_t(self):
    """Computes the theoretical policy's `eta_t` term."""
    # pylint: disable=invalid-name
    kappa = self.kappa
    K = self.mean_revert
    q = self.cross_q
    c = self.terminal_cost
    T = self.horizon
    t = self.t

    R = (K + q)**2 + (kappa - q**2)
    deltap = -(K + q) + math.sqrt(R)
    deltam = -(K + q) - math.sqrt(R)
    numerator = -(kappa - q**2) * (math.exp(
        (deltap - deltam) * (T - t)) - 1) - c * (
            deltap * math.exp((deltap - deltam) * (T - t)) - deltam)
    denominator = (deltam * math.exp(
        (deltap - deltam) * (T - t)) - deltap) - c * (
            math.exp((deltap - deltam) * (T - t)) - 1)
    return numerator / denominator

  def _rewards(self):
    """Reward for the player for this state."""
    if self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      dist_mean = (self.distribution_average() - self.x)

      move = self.action_to_move(self._last_action)
      action_reward = self.dt / 2 * (-move**2 + 2 * self.cross_q * move *
                                     dist_mean - self.kappa * dist_mean**2)

      if self.is_terminal():
        terminal_reward = -self.terminal_cost * dist_mean**2 / 2.0
        return action_reward + terminal_reward
      return action_reward

    return 0.0

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
    return self.state_to_str(
        x=self.x, tick=self.tick, player_id=self._player_id)


class Observer:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params, game):
    """Initializes an empty observation tensor."""
    del params

    self.size = game.size
    self.horizon = game.horizon
    self.tensor = np.zeros(2, np.float32)
    self.dict = {
        "x": self.tensor[0],
        "t": self.tensor[1],
        "observation": self.tensor
    }

  def set_from(self, state, player: int):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    self.tensor[0] = state.x
    self.tensor[1] = state.t
    # state.x is None for the initial (blank) state, don't set any
    # position bit in that case.
    if state.x is not None:
      if not 0 <= state.x < self.size:
        raise ValueError(
            f"Expected {state} x position to be in [0, {self.size})")
      self.dict["x"] = np.array([state.x])
    if not 0 <= state.t <= self.horizon:
      raise ValueError(f"Expected {state} time to be in [0, {self.horizon}]")
    self.dict["t"] = np.array([state.t])

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return state.to_string()

  def plot_mean_field_flow(self, policy):
    a = policy
    return a


pyspiel.register_game(_GAME_TYPE, MFGLinearQuadraticGame)
