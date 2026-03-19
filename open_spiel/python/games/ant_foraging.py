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

"""Ant Foraging on a Grid - A multi-agent foraging simulation game.

This game simulates ants foraging for food on a grid. Multiple ants start from
a nest and must navigate to find food sources, then return the food to the
nest. The game incorporates pheromone trails that ants can leave and follow.

Key mechanics:
- Grid-based world with nest(s), food source(s), and obstacles
- Ants can move in 4 directions (up, down, left, right) or stay
- Pheromones decay over time and influence ant movement heuristics
- Cooperative: all ants work together to maximize food collected
- Terminal condition: fixed number of turns or all food collected

This is a good testbed for multi-agent coordination and swarm intelligence
algorithms.
"""

# pylint: disable=protected-access


import enum

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel


class CellType(enum.IntEnum):
  """Types of cells on the grid."""

  EMPTY = 0
  NEST = 1
  FOOD = 2
  OBSTACLE = 3


class Action(enum.IntEnum):
  """Possible actions for an ant."""

  STAY = 0
  UP = 1
  DOWN = 2
  LEFT = 3
  RIGHT = 4


# Movement deltas for each action (row, col)
_ACTION_DELTAS = {
    Action.STAY: (0, 0),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

# Default game parameters
_DEFAULT_PARAMS = {
    "grid_size": 8,
    "num_ants": 2,
    "num_food": 3,
    "max_turns": 50,
    "pheromone_decay": 0.9,
}

_NUM_ACTIONS = len(Action)


def _make_game_type(params):
  """Create the game type with the given parameters."""
  num_ants = params.get("num_ants", _DEFAULT_PARAMS["num_ants"])
  return pyspiel.GameType(
      short_name="python_ant_foraging",
      long_name="Python Ant Foraging",
      dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
      chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
      information=pyspiel.GameType.Information.PERFECT_INFORMATION,
      utility=pyspiel.GameType.Utility.IDENTICAL,
      reward_model=pyspiel.GameType.RewardModel.TERMINAL,
      max_num_players=num_ants,
      min_num_players=num_ants,
      provides_information_state_string=True,
      provides_information_state_tensor=False,
      provides_observation_string=True,
      provides_observation_tensor=True,
      parameter_specification={},
  )


# Default game type for registration
_GAME_TYPE = _make_game_type(_DEFAULT_PARAMS)


def _make_game_info(params):
  """Create game info based on parameters."""
  num_ants = params.get("num_ants", _DEFAULT_PARAMS["num_ants"])
  num_food = params.get("num_food", _DEFAULT_PARAMS["num_food"])
  max_turns = params.get("max_turns", _DEFAULT_PARAMS["max_turns"])
  return pyspiel.GameInfo(
      num_distinct_actions=_NUM_ACTIONS,
      max_chance_outcomes=0,
      num_players=num_ants,
      min_utility=0.0,
      max_utility=float(num_food),
      utility_sum=None,  # Not constant sum
      max_game_length=max_turns * num_ants,
  )


class AntForagingGame(pyspiel.Game):
  """The Ant Foraging game."""

  def __init__(self, params=None):
    """Initialize the game with given parameters."""
    params = params or {}
    self._grid_size = params.get("grid_size", _DEFAULT_PARAMS["grid_size"])
    self._num_ants = params.get("num_ants", _DEFAULT_PARAMS["num_ants"])
    self._num_food = params.get("num_food", _DEFAULT_PARAMS["num_food"])
    self._max_turns = params.get("max_turns", _DEFAULT_PARAMS["max_turns"])
    self._pheromone_decay = params.get(
        "pheromone_decay", _DEFAULT_PARAMS["pheromone_decay"]
    )

    game_type = _make_game_type(params)
    game_info = _make_game_info(params)
    super().__init__(game_type, game_info, params)

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return AntForagingState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an observer object for the game."""
    if (iig_obs_type is None) or (
        iig_obs_type.public_info and not iig_obs_type.perfect_recall
    ):
      return AntForagingObserver(self._grid_size, self._num_ants, params)
    return IIGObserverForPublicInfoGame(iig_obs_type, params)

  @property
  def grid_size(self):
    return self._grid_size

  @property
  def num_ants(self):
    return self._num_ants

  @property
  def num_food(self):
    return self._num_food

  @property
  def max_turns(self):
    return self._max_turns

  @property
  def pheromone_decay(self):
    return self._pheromone_decay


class AntForagingState(pyspiel.State):
  """State of the Ant Foraging game."""

  def __init__(self, game):
    """Constructor - sets up the initial game state."""
    super().__init__(game)
    self._game = game
    self._grid_size = game.grid_size
    self._num_ants = game.num_ants
    self._num_food = game.num_food
    self._max_turns = game.max_turns
    self._pheromone_decay = game.pheromone_decay

    # Initialize the grid
    self._grid = np.zeros((self._grid_size, self._grid_size), dtype=np.int32)

    # Place nest in center
    nest_pos = self._grid_size // 2
    self._nest_pos = (nest_pos, nest_pos)
    self._grid[self._nest_pos] = CellType.NEST

    # Place food sources randomly (avoiding nest and edges)
    self._food_positions = []
    np.random.seed(42)  # Deterministic for reproducibility
    attempts = 0
    while len(self._food_positions) < self._num_food and attempts < 100:
      row = np.random.randint(1, self._grid_size - 1)
      col = np.random.randint(1, self._grid_size - 1)
      if (row, col) != self._nest_pos and (
          row,
          col,
      ) not in self._food_positions:
        self._food_positions.append((row, col))
        self._grid[row, col] = CellType.FOOD
      attempts += 1

    # Initialize ant positions (all start at nest)
    self._ant_positions = [self._nest_pos for _ in range(self._num_ants)]

    # Track which ants are carrying food
    self._carrying_food = [False] * self._num_ants

    # Pheromone grids (one for "to food" and one for "to nest")
    self._pheromone_to_food = np.zeros(
        (self._grid_size, self._grid_size), dtype=np.float32
    )
    self._pheromone_to_nest = np.zeros(
        (self._grid_size, self._grid_size), dtype=np.float32
    )

    # Game state tracking
    self._current_player = 0
    self._turn_count = 0
    self._food_collected = 0
    self._is_terminal = False

  def current_player(self):
    """Returns the current player, or TERMINAL if the game is over."""
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    return self._current_player

  def _legal_actions(self, player):
    """Returns the list of legal actions for a player."""
    if player != self._current_player:
      return []

    actions = [Action.STAY]
    row, col = self._ant_positions[player]

    # Check each direction
    for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
      dr, dc = _ACTION_DELTAS[action]
      new_row, new_col = row + dr, col + dc
      if self._is_valid_position(new_row, new_col):
        actions.append(action)

    return sorted(actions)

  def _is_valid_position(self, row, col):
    """Check if a position is within bounds and not an obstacle."""
    if row < 0 or row >= self._grid_size:
      return False
    if col < 0 or col >= self._grid_size:
      return False
    if self._grid[row, col] == CellType.OBSTACLE:
      return False
    return True

  def _apply_action(self, action):
    """Applies the action for the current player."""
    ant_idx = self._current_player
    row, col = self._ant_positions[ant_idx]

    # Move the ant
    dr, dc = _ACTION_DELTAS[action]
    new_row, new_col = row + dr, col + dc

    if self._is_valid_position(new_row, new_col):
      self._ant_positions[ant_idx] = (new_row, new_col)
      row, col = new_row, new_col

    # Check for food pickup
    if not self._carrying_food[ant_idx] and (row, col) in self._food_positions:
      self._carrying_food[ant_idx] = True
      self._food_positions.remove((row, col))
      self._grid[row, col] = CellType.EMPTY
      # Leave pheromone trail indicating food was here
      self._pheromone_to_food[row, col] = 1.0

    # Check for food delivery at nest
    if self._carrying_food[ant_idx] and (row, col) == self._nest_pos:
      self._carrying_food[ant_idx] = False
      self._food_collected += 1
      # Leave pheromone trail to nest
      self._pheromone_to_nest[row, col] = 1.0

    # Update pheromone based on ant's state
    if self._carrying_food[ant_idx]:
      # Ant carrying food leaves "to nest" pheromone
      self._pheromone_to_nest[row, col] = min(
          1.0, self._pheromone_to_nest[row, col] + 0.3
      )
    else:
      # Ant searching leaves "to food" pheromone when near food
      if self._pheromone_to_food[row, col] > 0:
        self._pheromone_to_food[row, col] = min(
            1.0, self._pheromone_to_food[row, col] + 0.1
        )

    # Move to next player
    self._current_player = (self._current_player + 1) % self._num_ants

    # If we've cycled through all ants, that's one turn
    if self._current_player == 0:
      self._turn_count += 1
      # Decay pheromones at end of each round
      self._pheromone_to_food *= self._pheromone_decay
      self._pheromone_to_nest *= self._pheromone_decay

    # Check terminal conditions
    if self._turn_count >= self._max_turns:
      self._is_terminal = True
    if self._food_collected >= self._num_food:
      self._is_terminal = True

  def _action_to_string(self, player, action):
    """Converts an action to a human-readable string."""
    action_names = {
        Action.STAY: "stay",
        Action.UP: "up",
        Action.DOWN: "down",
        Action.LEFT: "left",
        Action.RIGHT: "right",
    }
    return f"ant{player}:{action_names.get(action, '?')}"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Returns the total reward for each player."""
    # Cooperative game - all players share the same score
    score = float(self._food_collected)
    return [score] * self._num_ants

  def __str__(self):
    """Returns a string representation of the current state."""
    lines = []
    lines.append(
        f"Turn {self._turn_count}/{self._max_turns}, "
        f"Food: {self._food_collected}/{self._num_food}"
    )

    # Build grid visualization
    grid_chars = []
    for row in range(self._grid_size):
      row_chars = []
      for col in range(self._grid_size):
        # Check for ants first
        ant_here = None
        for i, pos in enumerate(self._ant_positions):
          if pos == (row, col):
            ant_here = i
            break

        if ant_here is not None:
          if self._carrying_food[ant_here]:
            row_chars.append(str(ant_here).upper())
          else:
            row_chars.append(str(ant_here))
        elif self._grid[row, col] == CellType.NEST:
          row_chars.append("N")
        elif self._grid[row, col] == CellType.FOOD:
          row_chars.append("F")
        elif self._grid[row, col] == CellType.OBSTACLE:
          row_chars.append("#")
        else:
          # Show pheromone intensity
          pf = self._pheromone_to_food[row, col]
          pn = self._pheromone_to_nest[row, col]
          if pf > 0.5 or pn > 0.5:
            row_chars.append("~")
          elif pf > 0.1 or pn > 0.1:
            row_chars.append(".")
          else:
            row_chars.append(" ")
      grid_chars.append("".join(row_chars))

    # Add border
    border = "+" + "-" * self._grid_size + "+"
    lines.append(border)
    for row_str in grid_chars:
      lines.append("|" + row_str + "|")
    lines.append(border)

    # Add legend
    lines.append("Legend: N=nest, F=food, 0-9=ant, A-Z=ant+food")

    return "\n".join(lines)


class AntForagingObserver:
  """Observer for the Ant Foraging game."""

  def __init__(self, grid_size, num_ants, params=None):
    """Initialize the observer."""
    if params:
      raise ValueError(f"Observation parameters not supported; got {params}")

    self._grid_size = grid_size
    self._num_ants = num_ants

    # Observation tensor layout:
    # - Grid channels: empty, nest, food, obstacle (4)
    # - Ant positions: one channel per ant (num_ants)
    # - Carrying food: one channel per ant (num_ants)
    # - Pheromone channels: to_food, to_nest (2)
    num_channels = 4 + num_ants * 2 + 2
    shape = (num_channels, grid_size, grid_size)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates the observation tensor from the given state."""
    del player  # Observation is the same for all players
    obs = self.dict["observation"]
    obs.fill(0)

    # Grid cell types
    for row in range(self._grid_size):
      for col in range(self._grid_size):
        cell = state._grid[row, col]
        obs[cell, row, col] = 1.0

    # Ant positions
    for i, (row, col) in enumerate(state._ant_positions):
      obs[4 + i, row, col] = 1.0

    # Carrying food
    for i, carrying in enumerate(state._carrying_food):
      if carrying:
        row, col = state._ant_positions[i]
        obs[4 + self._num_ants + i, row, col] = 1.0

    # Pheromones
    obs[-2] = state._pheromone_to_food
    obs[-1] = state._pheromone_to_nest

  def string_from(self, state, player):
    """Returns string observation."""
    del player
    return str(state)


# Register the game with OpenSpiel
pyspiel.register_game(_GAME_TYPE, AntForagingGame)
