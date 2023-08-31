# Copyright 2023 DeepMind Technologies Limited
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

"""Mean Field Game on periodic domain with aversion cost.

This is a demonstration of implementing a mean field game in Python. The model
is an approximation of a continuous space, continuous time model introduced
to study ergodic MFG with explicit solution in:
Almulla, N.; Ferreira, R.; and Gomes, D. 2017.
Two numerical approaches to stationary mean-field games. Dyn. Games Appl.
7(4):657-682.

See also:
Elie, R., Perolat, J., Laurière, M., Geist, M., & Pietquin, O. (2020, April).
On the convergence of model free learning in mean field games.
In Proceedings of the AAAI Conference on Artificial Intelligence
(Vol. 34, No. 05, pp. 7143-7150).
"""

import functools
import math
from typing import Any, List, Mapping

import numpy as np
import scipy.stats

from open_spiel.python import observation
import pyspiel

_NUM_PLAYERS = 1
_SIZE = 21
_HORIZON = 20
_VOLATILITY = 1.0
_COEF_AVERSION = 1.0
_DELTA_T = 0.01
_X_MIN = 0.0
_X_MAX = 1.0
_N_ACTIONS_PER_SIDE = 10

_DEFAULT_PARAMS = {
    "size": _SIZE,
    "horizon": _HORIZON,
    "dt": _DELTA_T,
    "xmin": _X_MIN,
    "xmax": _X_MAX,
    "n_actions_per_side": _N_ACTIONS_PER_SIDE,
    "volatility": _VOLATILITY,
    "coef_aversion": _COEF_AVERSION,
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_mfg_periodic_aversion",
    long_name="Mean-Field Periodic Aversion Game",
    dynamics=pyspiel.GameType.Dynamics.MEAN_FIELD,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS,
)


@functools.lru_cache(maxsize=None)
def _state_to_str(x, t, player_id):
  """A string that uniquely identifies (x, t, player_id)."""
  if int(player_id) == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
    return f"(t={t}, pos={x})"
  if player_id == pyspiel.PlayerId.MEAN_FIELD:
    return f"(t={t}_a, pos={x})"
  if player_id == pyspiel.PlayerId.CHANCE:
    return f"(t={t}_a_mu, pos={x})"


class MFGPeriodicAversionGame(pyspiel.Game):
  """A Mean-Field Game on periodic domain with crowd aversion cost.

  A game starts by an initial chance node that select the initial state
  of the player in the MFG.
  Then the game sequentially alternates between:
    - An action selection node (where the player id is >= 0)
    - A chance node (the player id is pyspiel.PlayerId.CHANCE)
    - A Mean Field node (the player id is pyspiel.PlayerId.MEAN_FIELD)
  """

  # pylint:disable=dangerous-default-value
  def __init__(self, params: Mapping[str, Any] = _DEFAULT_PARAMS):
    self.size = params.get("size", _SIZE)  # number of states
    self.horizon = params.get("horizon", _HORIZON)  # number of time steps
    self.dt = params.get("dt", _DELTA_T)  # size of one step in time
    self.xmin = params.get("xmin", _X_MIN)  # smallest position
    self.xmax = params.get("xmax", _X_MAX)  # largest position
    self.dx = (self.xmax - self.xmin) / (
        self.size - 1
    )  # size of one step in space
    self.n_actions_per_side = params.get(
        "n_actions_per_side", _N_ACTIONS_PER_SIDE
    )  # number of actions on each side, for both players and noise
    self.volatility = params.get("volatility", _VOLATILITY)
    self.coef_aversion = params.get("coef_aversion", _COEF_AVERSION)

    game_info = pyspiel.GameInfo(
        num_distinct_actions=2 * self.n_actions_per_side + 1,
        max_chance_outcomes=2 * self.n_actions_per_side + 1,
        num_players=_NUM_PLAYERS,
        min_utility=-np.inf,
        max_utility=+np.inf,
        utility_sum=0.0,
        max_game_length=self.horizon,
    )
    super().__init__(_GAME_TYPE, game_info, params)

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return MFGPeriodicAversionState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if (iig_obs_type is None) or (
        iig_obs_type.public_info and not iig_obs_type.perfect_recall
    ):
      return Observer(params, self)
    return observation.IIGObserverForPublicInfoGame(iig_obs_type, params)

  def max_chance_nodes_in_history(self):
    """Maximun chance nodes in game history."""
    return self.horizon + 1


class MFGPeriodicAversionState(pyspiel.State):
  """A Mean Field Normal-Form state.

  In this class, x and action are integers. They are converted, when needed, to
  spatial variables by using a scaling factor representing the size of a step in
  space and by shifting them depending on the minimal allowed value.
  """

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    # Initial state where the initial position is chosen according to
    # an initial distribution.
    self._player_id = pyspiel.PlayerId.CHANCE

    self._last_action = game.n_actions_per_side  # neutral action
    self.tick = 0
    self.x = None
    self.return_value = 0.0

    self.game = game

    self.size = game.size
    self.horizon = game.horizon
    self.dt = game.dt
    self.xmin = game.xmin
    self.xmax = game.xmax
    self.dx = game.dx
    self.da = game.dx
    self.n_actions_per_side = game.n_actions_per_side
    self.volatility = game.volatility
    self.coef_aversion = game.coef_aversion

    # Represents the current probability distribution over game states.
    # Initialized with a uniform distribution.
    self._distribution = [1.0 / self.size for _ in range(self.size)]

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
        "player_id is not mean field, chance or default player id."
    )

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  @property
  def n_actions(self):
    return 2 * self.n_actions_per_side + 1

  def _legal_actions(self, player):
    """Returns a list of legal actions for player and MFG nodes."""
    if player == pyspiel.PlayerId.MEAN_FIELD:
      return []
    if (
        player == pyspiel.PlayerId.DEFAULT_PLAYER_ID
        and player == self.current_player()
    ):
      return list(range(self.n_actions))
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

    assert (
        self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID
        or self._player_id == pyspiel.PlayerId.CHANCE
    )

    if self.x is None:
      self.x = action
      self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID
      return

    if action < 0 or action >= self.n_actions:
      raise ValueError(
          "The action is between 0 and {} at any node".format(self.n_actions)
      )

    self.x = (self.x + action - self.n_actions_per_side) % self.size
    if self._player_id == pyspiel.PlayerId.CHANCE:
      self._player_id = pyspiel.PlayerId.MEAN_FIELD
      self.tick += 1
    elif self._player_id == pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      self._last_action = action
      self._player_id = pyspiel.PlayerId.CHANCE

  def _action_to_string(self, player, action):
    """Action -> string."""
    del player
    return str(action - self.n_actions_per_side)

  def action_to_move(self, action):
    return (action - self.n_actions_per_side) * self.da

  def state_to_position(self, state):
    return state * self.dx + self.xmin

  def position_to_state(self, position):
    return round((position - self.xmin) / self.dx)

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    if self.x is None:
      # Initial distribution
      return list(enumerate(self._distribution))
    actions = np.array(
        [(a - self.n_actions_per_side) * self.da for a in range(self.n_actions)]
    )
    stddev = self.volatility * math.sqrt(self.dt)
    probas = scipy.stats.norm.pdf(actions, scale=stddev)
    probas /= np.sum(probas)
    return [(act, p) for act, p in zip(list(range(self.n_actions)), probas)]

  def distribution_support(self):
    """return a list of state string."""
    return [
        self.state_to_str(i, self.tick, player_id=pyspiel.PlayerId.MEAN_FIELD)
        for i in range(self.size)
    ]

  def get_state_proba(self, state: int) -> float:
    """Gets the probability of a position in the current distrib.

    Args:
      state: state requested.

    Returns:
      The probability for the provided position.
    """
    assert state >= 0, state
    assert state < self.size, state
    # This logic needs to match the ordering defined in distribution_support().
    index = state
    assert 0 <= index < len(self._distribution), (
        f"Invalid index {index} vs dist length:"
        f" {len(self._distribution)}, state={state},"
        f" state={self}"
    )
    return self._distribution[index]

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
    self._player_id = pyspiel.PlayerId.DEFAULT_PLAYER_ID

  @property
  def t(self):
    return self.tick

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
    if self._player_id != pyspiel.PlayerId.DEFAULT_PLAYER_ID:
      return 0.0
    assert self.x is not None
    velocity = self.action_to_move(self._last_action) / self.dt
    action_r = -0.5 * velocity**2
    eps = 1e-15
    mu_x = self.get_state_proba(self.x) / self.dx  # represents the density
    # The density should have an integral equal to 1; here sum_x mu_x * dx = 1
    aversion_r = -np.log(mu_x + eps)
    pos = self.state_to_position(self.x)
    pix2 = 2 * np.pi * pos
    geom_r = (
        self.volatility * 2 * np.pi**2 * np.sin(pix2)
        - 2 * np.pi**2 * np.cos(pix2) ** 2
        + (2 / self.volatility**2) * np.sin(pix2)
    )
    return (action_r + self.coef_aversion * aversion_r + geom_r) * self.dt

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
    return self.state_to_str(
        x=self.x, tick=self.tick, player_id=self._player_id
    )


class Observer:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params, game):
    """Initializes an empty observation tensor."""
    del params

    self.size = game.size
    self.horizon = game.horizon
    # +1 to allow t == horizon.
    self.tensor = np.zeros(self.size + self.horizon + 1, np.float32)
    self.dict = {"x": self.tensor[: self.size], "t": self.tensor[self.size :]}

  def set_from(self, state, player: int):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    self.tensor.fill(0)
    # state.x is None for the initial (blank) state, don't set any
    # position bit in that case.
    if state.x is not None:
      if state.x < 0 or state.x > self.size:
        raise ValueError(
            f"Expected {state} positions to be in [0, {self.size})"
        )
      self.dict["x"][state.x] = 1
    if not 0 <= state.tick <= self.horizon:
      raise ValueError(f"Expected {state} time to be in [0, {self.horizon}]")
    self.dict["t"][state.tick] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return state.to_string()


pyspiel.register_game(_GAME_TYPE, MFGPeriodicAversionGame)
