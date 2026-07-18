# Copyright 2024 DeepMind Technologies Limited
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

"""Pursuit-evasion game in 2D continuous space.

A two-player zero-sum game where a pursuer (Player 0) attempts to capture an
evader (Player 1) within a bounded 2D space.  Players alternate actions each
round: the pursuer moves first, then the evader.  Both use 9 discrete actions
(8 compass directions + stay).  Capture occurs when the Euclidean distance
between pursuer and evader is less than the capture radius.

Evader bot implementations (``RandomEvaderBot``, ``ConstantVelocityEvaderBot``,
``ZigzagEvaderBot``, ``AdaptiveEvaderBot``) are provided for driving Player 1
externally via ``evaluate_bots`` or similar evaluation loops.

Designed to accompany research on NEAT vs PPO under non-stationary opponent
strategies (IEEE Access, under submission).
"""

import math

import numpy as np

import pyspiel

_NUM_ACTIONS = 9

# Direction vectors (normalised to unit length).  Index = action id.
# 0 = stay, 1-8 = N, NE, E, SE, S, SW, W, NW.
_DIRECTIONS = [
    (0.0, 0.0),
    (0.0, 1.0),
    (math.sqrt(0.5), math.sqrt(0.5)),
    (1.0, 0.0),
    (math.sqrt(0.5), -math.sqrt(0.5)),
    (0.0, -1.0),
    (-math.sqrt(0.5), -math.sqrt(0.5)),
    (-1.0, 0.0),
    (-math.sqrt(0.5), math.sqrt(0.5)),
]

_DIRECTION_NAMES = ["stay", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]

_DEFAULT_PARAMS = {
    "space_size": 10.0,
    "max_steps": 50,
    "capture_radius": 1.0,
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_pursuit_evasion",
    long_name="Python Pursuit-Evasion",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS,
)


class PursuitEvasionGame(pyspiel.Game):
  """Two-player pursuit-evasion game in continuous 2D space.

  Player 0 (pursuer) and Player 1 (evader) alternate actions each round.
  The pursuer acts first.  After both players have acted, capture is checked
  and the step counter advances.  The game ends when the pursuer captures the
  evader (distance < capture_radius) or after ``max_steps`` rounds.
  """

  def __init__(self, params=None):
    if params is None:
      params = {}
    self.space_size = float(params.get("space_size", 10.0))
    self.max_steps = int(params.get("max_steps", 50))
    self.capture_radius = float(params.get("capture_radius", 1.0))
    game_info = pyspiel.GameInfo(
        num_distinct_actions=_NUM_ACTIONS,
        max_chance_outcomes=0,
        num_players=2,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=self.max_steps * 2,
    )
    super().__init__(_GAME_TYPE, game_info, params or {})

  def new_initial_state(self):
    return PursuitEvasionState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return PursuitEvasionObserver(iig_obs_type, params)


class PursuitEvasionState(pyspiel.State):
  """State of a two-player pursuit-evasion episode.

  Players alternate: Player 0 (pursuer) acts first, then Player 1 (evader).
  After both have acted in a round, capture is checked and the step counter
  advances.

  Attributes:
    pursuer_x, pursuer_y: current pursuer position.
    evader_x, evader_y: current evader position.
    step: number of completed pursuer-evader rounds.
  """

  def __init__(self, game):
    super().__init__(game)
    self.space_size = game.space_size
    self.max_steps = game.max_steps
    self.capture_radius = game.capture_radius
    self.pursuer_x = 0.0
    self.pursuer_y = 0.0
    self.evader_x = self.space_size
    self.evader_y = self.space_size
    self.step = 0
    self._pursuer_acted = False
    self._is_terminal = False
    self.pursuer_reward = 0.0

  def _clamp(self, val):
    return max(0.0, min(self.space_size, val))

  def current_player(self):
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    elif self._pursuer_acted:
      return 1  # evader's turn
    else:
      return 0  # pursuer's turn

  def _legal_actions(self, player):
    if self._is_terminal:
      return []
    cp = self.current_player()
    if player == cp:
      return list(range(_NUM_ACTIONS))
    return []

  def _apply_action(self, action):
    if self.current_player() == 0:
      # Pursuer moves.
      dx, dy = _DIRECTIONS[action]
      self.pursuer_x = self._clamp(self.pursuer_x + dx)
      self.pursuer_y = self._clamp(self.pursuer_y + dy)
      self._pursuer_acted = True
    else:
      # Evader moves.
      dx, dy = _DIRECTIONS[action]
      self.evader_x = self._clamp(self.evader_x + dx)
      self.evader_y = self._clamp(self.evader_y + dy)
      self._pursuer_acted = False
      self.step += 1
      # Check terminal conditions after both have acted.
      dist = math.sqrt(
          (self.pursuer_x - self.evader_x) ** 2 +
          (self.pursuer_y - self.evader_y) ** 2,
      )
      if dist < self.capture_radius:
        self._is_terminal = True
        self.pursuer_reward = 1.0
      elif self.step >= self.max_steps:
        self._is_terminal = True
        self.pursuer_reward = -1.0

  def _action_to_string(self, player, action):
    name = "Pursuer" if player == 0 else "Evader"
    return f"{name} moves {_DIRECTION_NAMES[action]}"

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    return [self.pursuer_reward, -self.pursuer_reward]

  def _check_player(self, player):
    if player is not None and player < 0:
      raise RuntimeError(f"player >= 0: player = {player}")
    if player is not None and player >= 2:
      raise RuntimeError(f"player < 2: player = {player}")

  def information_state_tensor(self, player=None):
    self._check_player(player)
    gs = self.space_size
    ms = float(self.max_steps)
    return np.array([
        self.pursuer_x / gs,
        self.pursuer_y / gs,
        self.evader_x / gs,
        self.evader_y / gs,
        self.step / ms,
    ], dtype=np.float32)

  def observation_tensor(self, player=None):
    self._check_player(player)
    return self.information_state_tensor(player)

  def __str__(self):
    return (
        f"Pursuer: ({self.pursuer_x:.2f}, {self.pursuer_y:.2f}), "
        f"Evader: ({self.evader_x:.2f}, {self.evader_y:.2f}), "
        f"Step: {self.step}/{self.max_steps}, "
        f"Terminal: {self._is_terminal}"
    )


class PursuitEvasionObserver:
  """Observer for the pursuit-evasion game.

  The observation is a 5-element vector:
    [pursuer_x, pursuer_y, evader_x, evader_y, step_count],
  each normalised by the respective maximum value.
  """

  def __init__(self, iig_obs_type=None, params=None):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self._iig_obs_type = iig_obs_type
    self.tensor = np.zeros(5, np.float32)
    self.dict = {"observation": np.reshape(self.tensor, (5,))}

  def set_from(self, state, player):
    del player
    gs = state.space_size
    ms = float(state.max_steps)
    self.dict["observation"][0] = state.pursuer_x / gs
    self.dict["observation"][1] = state.pursuer_y / gs
    self.dict["observation"][2] = state.evader_x / gs
    self.dict["observation"][3] = state.evader_y / gs
    self.dict["observation"][4] = state.step / ms

  def string_from(self, state, player):
    del player
    if (self._iig_obs_type is not None
        and not self._iig_obs_type.public_info):
      return ""
    return str(state)


# ---------------------------------------------------------------------------
# Evader bots (external strategy implementations).
# ---------------------------------------------------------------------------
# These bots wrap the evader strategies as pyspiel.Bot implementations so
# they can be used with evaluate_bots or any other bot-based evaluation loop.
# In the two-player game the evader is Player 1; use these bots to drive
# that player's actions.

class EvaderBot(pyspiel.Bot):
  """Base class for evader bots."""

  def __init__(self, player_id=1):
    pyspiel.Bot.__init__(self)
    self._player_id = player_id

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def restart_at(self, state):
    pass

  def inform_action(self, state, player_id, action):
    pass


class RandomEvaderBot(EvaderBot):
  """Evader bot that chooses uniformly at random from 8 directions."""

  def __init__(self, player_id=1, rng=None):
    super().__init__(player_id)
    self._rng = rng or np.random.RandomState()

  def step_with_policy(self, state):
    actions = list(range(1, _NUM_ACTIONS))
    p = 1.0 / len(actions)
    policy = [(a, p) for a in actions]
    action = self._rng.choice(actions)
    return policy, action

  def step(self, state):
    return self.step_with_policy(state)[1]


class ConstantVelocityEvaderBot(EvaderBot):
  """Evader bot that always moves East."""

  def step_with_policy(self, state):
    return [(3, 1.0)], 3

  def step(self, state):
    return 3


class ZigzagEvaderBot(EvaderBot):
  """Evader bot that alternates between NE and SE."""

  def __init__(self, player_id=1):
    super().__init__(player_id)
    self._toggle = 0

  def restart_at(self, state):
    self._toggle = 0

  def step_with_policy(self, state):
    action = 2 if self._toggle == 0 else 4
    self._toggle = 1 - self._toggle
    return [(action, 1.0)], action

  def step(self, state):
    return self.step_with_policy(state)[1]


class AdaptiveEvaderBot(EvaderBot):
  """Evader bot that moves directly away from the pursuer.

  Requires that the game state exposes ``pursuer_x``, ``pursuer_y``,
  ``evader_x``, and ``evader_y`` attributes.
  """

  def step_with_policy(self, state):
    vx = state.evader_x - state.pursuer_x
    vy = state.evader_y - state.pursuer_y
    norm = math.sqrt(vx ** 2 + vy ** 2)
    if norm > 1e-8:
      best_action = 1
      best_dot = -1.0
      for i in range(1, _NUM_ACTIONS):
        dx, dy = _DIRECTIONS[i]
        dot = dx * vx + dy * vy
        if dot > best_dot:
          best_dot = dot
          best_action = i
      return [(best_action, 1.0)], best_action
    else:
      actions = list(range(1, _NUM_ACTIONS))
      p = 1.0 / len(actions)
      return [(a, p) for a in actions], actions[0]

  def step(self, state):
    return self.step_with_policy(state)[1]


# Register the game with the OpenSpiel library.
pyspiel.register_game(_GAME_TYPE, PursuitEvasionGame)
