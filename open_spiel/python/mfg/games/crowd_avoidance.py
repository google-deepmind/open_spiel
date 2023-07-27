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
"""Mean Field Crowd Avoidance game, implemented in Python.

This corresponds to an environment in which two populations try to avoid each
other.

The environment is configurable in the following high-level ways:
- Congestion coefficients matrix.
- Initial distribution.
- Geometry (torus, basic square).
"""

import enum
import functools
import math
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from open_spiel.python import observation
import pyspiel
from open_spiel.python.utils import shared_value


class Geometry(enum.IntEnum):
  SQUARE = 0
  TORUS = 1


_DEFAULT_SIZE = 7
_DEFAULT_HORIZON = 10
_NUM_ACTIONS = 5
_NUM_CHANCE = 5
_DEFAULT_CONGESTION_MATRIX = np.array(
    # The first population feels congestion with respect to the second one,
    # and vice-versa.
    [[0, 1], [1, 0]]
)
_DEFAULT_NUM_PLAYERS = 2
# Each population starts in a corner.
_DEFAULT_INIT_DISTRIB = np.array([
    # First population
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Second population
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])


def grid_to_forbidden_states(grid):
  """Converts a grid into string representation of forbidden states.

  Args:
    grid: Rows of the grid. '#' character denotes a forbidden state. All rows
      should have the same number of columns, i.e. cells.

  Returns:
    String representation of forbidden states in the form of x (column) and y
    (row) pairs, e.g. [1|1;0|2].
  """
  forbidden_states = []
  num_cols = len(grid[0])
  for y, row in enumerate(grid):
    assert len(row) == num_cols, f"Number of columns should be {num_cols}."
    for x, cell in enumerate(row):
      if cell == "#":
        forbidden_states.append(f"{x}|{y}")
  return "[" + ";".join(forbidden_states) + "]"


def pairs_string_to_list(positions: str) -> List[np.ndarray]:
  """Converts a string representing positions into a list of positions."""
  pos = positions[1:-1]  # remove [ and ]
  split = pos.split(";")
  return [np.array([i for i in s.split("|")]) for s in split]


forbidden_states_grid = [
    "#######",
    "#  #  #",
    "#     #",
    "#  #  #",
    "#     #",
    "#  #  #",
    "#######",
]
_DEFAULT_FORBIDDEN_STATES = grid_to_forbidden_states(forbidden_states_grid)

forbidden_states_indicator = np.array(
    [
        [math.nan if c == "#" else 0 for c in [*row]]
        for row in forbidden_states_grid
    ]
)

_DEFAULT_PROBA_NOISE = 0.5

_DEFAULT_GEOMETRY = Geometry.SQUARE

_DEFAULT_COEF_CONGESTION = 0.0

_DEFAULT_COEF_TARGET = 1.0

_DEFAULT_PARAMS = {
    "size": _DEFAULT_SIZE,
    "horizon": _DEFAULT_HORIZON,
    "players": _DEFAULT_NUM_PLAYERS,
    # The congestion matrix is represented as a string containing a
    # space-separated list of values.
    # Its size defines the number of populations in the mean field game.
    "congestion_matrix": " ".join(
        str(v) for v in _DEFAULT_CONGESTION_MATRIX.flatten()
    ),
    "geometry": _DEFAULT_GEOMETRY,
    "init_distrib": " ".join(str(v) for v in _DEFAULT_INIT_DISTRIB.flatten()),
    # Probability that the transition is affected by noise
    "proba_noise": _DEFAULT_PROBA_NOISE,
    # Weight of congestion term in the reward
    "coef_congestion": _DEFAULT_COEF_CONGESTION,
    "forbidden_states": _DEFAULT_FORBIDDEN_STATES,
    "coef_target": _DEFAULT_COEF_TARGET,
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_mfg_crowd_avoidance",
    long_name="Python Mean Field Crowd Avoidance",
    dynamics=pyspiel.GameType.Dynamics.MEAN_FIELD,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    # We cannot pass math.inf here, so we pass a very high integer value.
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS,
)


def get_param(param_name, params):
  return params.get(param_name, _DEFAULT_PARAMS[param_name])


@functools.lru_cache(maxsize=None)
def _state_to_str(x, y, t, population, player_id):
  """A string that uniquely identify (pos, t, population, player_id)."""
  if int(player_id) >= 0:
    return f"(pop={population}, t={t}, pos=[{x} {y}])"
  if player_id == pyspiel.PlayerId.MEAN_FIELD:
    return f"(pop={population}, t={t}_a, pos=[{x} {y}])"
  if player_id == pyspiel.PlayerId.CHANCE:
    return f"(pop={population}, t={t}_a_mu, pos=[{x} {y}])"


class MFGCrowdAvoidanceGame(pyspiel.Game):
  """Multi-population MFG."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params: Mapping[str, Any] = _DEFAULT_PARAMS):
    self.size = get_param("size", params)
    self.horizon = get_param("horizon", params)
    flat_congestion_matrix = np.fromstring(
        get_param("congestion_matrix", params), dtype=np.float64, sep=" "
    )
    num_players = get_param("players", params)
    if len(flat_congestion_matrix) != num_players**2:
      raise ValueError(
          "Congestion matrix passed in flat representation does not represent "
          f"a square matrix: {flat_congestion_matrix}"
      )
    self.congestion_matrix = flat_congestion_matrix.reshape(
        [num_players, num_players]
    )
    self.geometry = get_param("geometry", params)
    num_states = self.size**2
    game_info = pyspiel.GameInfo(
        num_distinct_actions=_NUM_ACTIONS,
        max_chance_outcomes=max(num_states, _NUM_CHANCE),
        num_players=num_players,
        min_utility=-np.inf,
        max_utility=+np.inf,
        utility_sum=None,
        max_game_length=self.horizon,
    )
    self.proba_noise = get_param("proba_noise", params)
    self.coef_congestion = get_param("coef_congestion", params)
    self.forbidden_states = pairs_string_to_list(
        get_param("forbidden_states", params)
    )
    self.coef_target = get_param("coef_target", params)
    # TODO(lauriere): should be given as a parameter of the model.
    self.target_positions = np.array([[5, 3], [1, 3]])

    # Represents the current probability distribution over game states
    # (when grouped for each population).
    str_init_distrib = get_param("init_distrib", params)
    if str_init_distrib:
      flat_init_distrib = np.fromstring(
          str_init_distrib, dtype=np.float64, sep=" "
      )
      if len(flat_init_distrib) != num_players * self.size**2:
        raise ValueError(
            "Initial distribution matrix passed in flat representation does"
            f" not represent a sequence of square matrices: {flat_init_distrib}"
        )
      self.initial_distribution = flat_init_distrib
    else:
      # Initialized with a uniform distribution.
      self.initial_distribution = [1.0 / num_states] * (
          num_states * num_players
      )
    super().__init__(_GAME_TYPE, game_info, params)

  def new_initial_state(self):
    """Returns a new population-less blank state.

    This state is provided for some internal operations that use blank
    states (e.g. cloning), but cannot be used to play the game, i.e.
    ApplyAction() will fail. Proper playable states should be
    instantiated with new_initial_state_for_population().
    """
    return MFGCrowdAvoidanceState(self)

  def max_chance_nodes_in_history(self):
    """Maximun chance nodes in game history."""
    return self.horizon + 1

  def new_initial_state_for_population(self, population):
    """State corresponding to the start of a game for a given population."""
    return MFGCrowdAvoidanceState(self, population)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if (iig_obs_type is None) or (
        iig_obs_type.public_info and not iig_obs_type.perfect_recall
    ):
      return Observer(params, self)
    return observation.IIGObserverForPublicInfoGame(iig_obs_type, params)


def pos_to_merged(pos: np.ndarray, size: int) -> int:
  """Converts a [x, y] position into a single integer."""
  assert (pos >= 0).all(), pos
  assert (pos < size).all(), pos
  return pos[0] + pos[1] * size


def merged_to_pos(merged_pos: int, size: int) -> np.ndarray:
  """Inverse of pos_to_merged()."""
  assert 0 <= merged_pos < size * size
  return np.array([merged_pos % size, merged_pos // size])


class MFGCrowdAvoidanceState(pyspiel.State):
  """State for the avoidance MFG."""

  # Maps legal actions to the corresponding move on the grid of the game.
  _ACTION_TO_MOVE = {
      0: np.array([0, 0]),
      1: np.array([1, 0]),
      2: np.array([0, 1]),
      3: np.array([0, -1]),
      4: np.array([-1, 0]),
  }
  # Action that corresponds to no displacement.
  _NEUTRAL_ACTION = 0

  def __init__(self, game, population=None):
    """Constructor; should only be called by Game.new_initial_state.*.

    Args:
      game: MFGCrowdAvoidanceGame for which a state should be created.
      population: ID of the population to create this state for. Must be in [0,
        num_players()) or None. States with population=None cannot be used to
        perform game actions.
    """
    super().__init__(game)
    # Initial state where the initial position is chosen according to
    # an initial distribution.
    self._is_position_init = True
    self._player_id = pyspiel.PlayerId.CHANCE
    # Population this state corresponds to. Can be None, in which
    # case, ApplyAction() is forbidden.
    self._population = population
    if self._population is not None:
      assert 0 <= self._population < self.num_players()
    # When set, <int>[2] numpy array representing the x, y position on the grid.
    self._pos = None  # type: Optional[np.ndarray]
    self._t = 0
    self.size = game.size
    # Number of states in the grid.
    self.num_states = self.size**2
    self.horizon = game.horizon
    self.congestion_matrix = game.congestion_matrix
    self.geometry = game.geometry
    self._returns = np.zeros([self.num_players()], dtype=np.float64)
    self._distribution = shared_value.SharedValue(game.initial_distribution)
    self.proba_noise = game.proba_noise
    self.coef_congestion = game.coef_congestion
    self.forbidden_states = game.forbidden_states
    self.coef_target = game.coef_target
    self.target_positions = game.target_positions

  @property
  def population(self):
    return self._population

  @property
  def pos(self):
    return self._pos

  @property
  def t(self):
    return self._t

  def state_to_str(self, pos, t, population, player_id=0):
    """A string that uniquely identify (pos, t, population, player_id)."""
    if self._is_position_init:
      return f"position_init_{population}"
    assert isinstance(pos, np.ndarray), f"Got type {type(pos)}"
    assert len(pos.shape) == 1, f"Got {len(pos.shape)}, expected 1 (pos={pos})."
    assert pos.shape[0] == 2, f"Got {pos.shape[0]}, expected 2 (pos={pos})."
    return _state_to_str(pos[0], pos[1], t, population, player_id)

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def mean_field_population(self):
    return self._population

  def _legal_actions(self, player):
    """Returns a list of legal actions for player and MFG nodes."""
    if player == pyspiel.PlayerId.MEAN_FIELD:
      return []
    if player >= 0 and player == self.current_player():
      return list(self._ACTION_TO_MOVE)
    raise ValueError(
        f"Unexpected player {player}."
        "Expected a mean field or current player >=0."
    )

  def chance_outcomes(self) -> List[Tuple[int, float]]:
    """Returns the possible chance outcomes and their probabilities."""
    if self._is_position_init:
      if (
          self._population is None
          or not 0 <= self._population < self.num_players()
      ):
        raise ValueError(f"Invalid population {self._population}")
      p = self._population % 2
      dist = self._distribution.value
      dist_p = dist[p * self.num_states : (p + 1) * self.num_states]
      pos_indices_flat = np.nonzero(dist_p)[0]
      pos_indices = [
          np.array([i % self.size, (i - i % self.size) // self.size])
          for i in pos_indices_flat
      ]
      # Beware: In the initial distribution representation, x and y correspond
      # respectively to the row and the column, but in the state representation,
      # they correspond to the column and the row.
      return [
          (pos_to_merged(i, self.size), dist_p[i[1] * self.size + i[0]])
          for i in pos_indices
      ]
    return [
        (0, 1.0 - self.proba_noise),
        (1, self.proba_noise / 4.0),
        (2, self.proba_noise / 4.0),
        (3, self.proba_noise / 4.0),
        (4, self.proba_noise / 4.0),
    ]

  def update_pos(self, action):
    """Updates the position of the player given a move action."""
    if action < 0 or action >= len(self._ACTION_TO_MOVE):
      raise ValueError(
          f"The action must be between 0 and {len(self._ACTION_TO_MOVE)}, "
          f"got {action}"
      )
    candidate_pos = self._pos + self._ACTION_TO_MOVE[action]
    # if candidate_pos in self.forbidden_states:
    # if np.any(np.all(candidate_pos == self.forbidden_states, axis=1)):
    if any(np.array_equal(candidate_pos, x) for x in self.forbidden_states):
      candidate_pos = self._pos
    elif self.geometry == Geometry.TORUS:
      candidate_pos += self.size
      candidate_pos %= self.size
    else:
      assert (
          self.geometry == Geometry.SQUARE
      ), f"Invalid geometry {self.geometry}"
      # Keep the position within the bounds of the square.
      candidate_pos = np.minimum(candidate_pos, self.size - 1)
      candidate_pos = np.maximum(candidate_pos, 0)
    self._pos = candidate_pos

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self._population is None:
      raise ValueError(
          "Attempting to perform an action with a population-less state."
      )
    if self._player_id == pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "_apply_action should not be called at a MEAN_FIELD state."
      )
    self._returns += np.array(self.rewards())
    if self._is_position_init:
      self._pos = merged_to_pos(action, self.size)
      self._is_position_init = False
      self._player_id = self._population
    elif self._player_id == pyspiel.PlayerId.CHANCE:
      self.update_pos(action)
      self._t += 1
      self._player_id = pyspiel.PlayerId.MEAN_FIELD
    elif int(self._player_id) >= 0:
      assert self._player_id == self._population, (
          f"Invalid decision player id {self._player_id} "
          f"expected {self._population}"
      )
      self.update_pos(action)
      self._player_id = pyspiel.PlayerId.CHANCE
    else:
      raise ValueError(f"Unexpected state. Player id: {self._player_id}")

  def _action_to_string(self, player, action):
    """Action -> string."""
    del player
    if self.is_chance_node() and self._is_position_init:
      return f"init_position={action}"
    return str(self._ACTION_TO_MOVE[action])

  def distribution_support(self):
    """Returns a list of state string."""
    support = []
    for x in range(self.size):
      for y in range(self.size):
        for population in range(self.num_players()):
          support.append(
              self.state_to_str(
                  np.array([x, y]),
                  self._t,
                  population,
                  player_id=pyspiel.PlayerId.MEAN_FIELD,
              )
          )
    return support

  def get_pos_proba(self, pos: np.ndarray, population: int) -> float:
    """Gets the probability of a pos and population in the current distrib.

    Args:
      pos: 2D position.
      population: Population requested.

    Returns:
      The probability for the provided position and population.
    """
    assert (pos >= 0).all(), pos
    assert (pos < self.size).all(), pos
    assert 0 <= population < self.num_players(), population
    # This logic needs to match the ordering defined in distribution_support().
    index = population + self.num_players() * (pos[1] + self.size * pos[0])
    assert 0 <= index < len(self._distribution.value), (
        f"Invalid index {index} vs dist length:"
        f" {len(self._distribution.value)}, population={population}, pos={pos},"
        f" state={self}"
    )
    return self._distribution.value[index]

  def update_distribution(self, distribution):
    """This function is central and specific to the logic of the MFG.

    It should only be called when the node is in MEAN_FIELD state.

    Args:
      distribution: List of floats that should contain the probability of each
        state returned by distribution_support().
    """
    expected_dist_size = self.num_states * self.num_players()
    assert len(distribution) == expected_dist_size, (
        "Unexpected distribution length "
        f"{len(distribution)} != {expected_dist_size}"
    )
    if self._player_id != pyspiel.PlayerId.MEAN_FIELD:
      raise ValueError(
          "update_distribution should only be called at a MEAN_FIELD state."
      )
    self._distribution = shared_value.SharedValue(distribution)
    self._player_id = self._population

  def is_terminal(self):
    """Returns True if the game is over."""
    return self.t >= self.horizon

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return self._player_id

  def rewards(self) -> List[float]:
    """Crowd avoidance rewards for all populations.

    Returns:
      One float per population.
    """
    if int(self._player_id) < 0:
      return [0.0] * self.num_players()
    densities = np.array(
        [
            self.get_pos_proba(self._pos, population)
            for population in range(self.num_players())
        ],
        dtype=np.float64,
    )
    rew = -self.coef_congestion * np.dot(self.congestion_matrix, densities)
    # Rewards for target positions.
    rew[0] += self.coef_target * np.array_equal(
        self._pos, self.target_positions[0]
    )
    rew[1] += self.coef_target * np.array_equal(
        self._pos, self.target_positions[1]
    )
    return list(rew)

  def returns(self) -> List[float]:
    """Returns is the sum of all payoffs collected so far."""
    return list(self._returns + np.array(self.rewards()))

  def __str__(self):
    """A string that uniquely identify the current state."""
    return self.state_to_str(
        self._pos, self._t, self._population, player_id=self._player_id
    )


class Observer:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params, game):
    """Initializes an empty observation tensor."""
    del params

    self.size = game.size
    self.horizon = game.horizon
    # +1 to allow t == horizon.
    self.tensor = np.zeros(2 * self.size + self.horizon + 1, np.float32)
    self.dict = {
        "x": self.tensor[: self.size],
        "y": self.tensor[self.size : self.size * 2],
        "t": self.tensor[self.size * 2 :],
    }

  def set_from(self, state: MFGCrowdAvoidanceState, player: int):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    self.tensor.fill(0)
    # state.pos is None for the initial (blank) state, don't set any
    # position bit in that case.
    if state.pos is not None:
      if not (state.pos >= 0).all() or not (state.pos < self.size).all():
        raise ValueError(
            f"Expected {state} positions to be in [0, {self.size})"
        )
      self.dict["x"][state.pos[0]] = 1
      self.dict["y"][state.pos[1]] = 1
    if not 0 <= state.t <= self.horizon:
      raise ValueError(f"Expected {state} time to be in [0, {self.horizon}]")
    self.dict["t"][state.t] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return str(state)


pyspiel.register_game(_GAME_TYPE, MFGCrowdAvoidanceGame)
