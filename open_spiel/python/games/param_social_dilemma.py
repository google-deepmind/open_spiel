# Copyright 2026 DeepMind Technologies Limited
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

"""Python implementation of parameterized social dilemma games.

This implements a flexible N-player social dilemma game with:
- Variable number of agents (N >= 2)
- Dynamic payoff matrices (can change over time)
- Stochastic rewards (optional noise)
- Support for various social dilemma structures

The game is designed for modern MARL research and benchmarking.
"""

import enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import pyspiel

# Default parameters
_DEFAULT_NUM_PLAYERS = 2
_DEFAULT_NUM_ACTIONS = 2
_DEFAULT_NUM_ROUNDS = 10
_DEFAULT_REWARD_NOISE_STD = 0.0
_DEFAULT_PAYOFF_DYNAMICS = "static"

_DEFAULT_PARAMS = {
    "num_players": _DEFAULT_NUM_PLAYERS,
    "num_actions": _DEFAULT_NUM_ACTIONS,
    "num_rounds": _DEFAULT_NUM_ROUNDS,
    "reward_noise_std": _DEFAULT_REWARD_NOISE_STD,
    "payoff_dynamics": _DEFAULT_PAYOFF_DYNAMICS,
    "dilemma_type": "prisoners_dilemma",  # prisoners_dilemma, stag_hunt, chicken, public_goods, custom
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_param_social_dilemma",
    long_name="Python Parameterized Social Dilemma",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=10,  # Configurable maximum
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class PayoffDynamics(enum.Enum):
  """Types of payoff dynamics."""
  STATIC = "static"  # Payoffs don't change
  CYCLING = "cycling"  # Payoffs cycle through predefined matrices
  DRIFTING = "drifting"  # Payoffs gradually drift over time
  RANDOM = "random"  # Payoffs change randomly each round
  CUSTOM = "custom"  # User-defined function


# Predefined social dilemma payoff structures (2-player, 2-action)
DILEMMA_PAYOFFS = {
    "prisoners_dilemma": {
        # (C, C), (C, D), (D, C), (D, D) for each player
        "payoff_matrix": np.array([
            [[3, 3], [0, 5]],
            [[5, 0], [1, 1]]
        ], dtype=np.float32),
        "description": "Classic Prisoner's Dilemma"
    },
    "stag_hunt": {
        "payoff_matrix": np.array([
            [[4, 4], [0, 3]],
            [[3, 0], [2, 2]]
        ], dtype=np.float32),
        "description": "Stag Hunt coordination game"
    },
    "chicken": {
        "payoff_matrix": np.array([
            [[3, 3], [2, 4]],
            [[4, 2], [1, 1]]
        ], dtype=np.float32),
        "description": "Chicken/Hawk-Dove game"
    },
    "public_goods": {
        "payoff_matrix": np.array([
            [[2, 2], [0, 3]],
            [[3, 0], [1, 1]]
        ], dtype=np.float32),
        "description": "Public Goods Game"
    },
}


class ParamSocialDilemmaGame(pyspiel.Game):
  """Parameterized N-player social dilemma game."""

  def __init__(self, params=None):
    """Initialize the game.
    
    Args:
      params: Dictionary of game parameters:
        - num_players: Number of players (default: 2)
        - num_actions: Number of actions per player (default: 2)
        - num_rounds: Number of rounds (default: 10)
        - reward_noise_std: Standard deviation of Gaussian reward noise (default: 0.0)
        - payoff_dynamics: Type of payoff dynamics (default: "static")
        - dilemma_type: Type of social dilemma (default: "prisoners_dilemma")
        - custom_payoff_matrix: Custom payoff matrix (optional)
        - payoff_matrices_sequence: Sequence of payoff matrices for cycling dynamics (optional)
    """
    if params is None:
      params = _DEFAULT_PARAMS
    else:
      # Merge with defaults
      params = {**_DEFAULT_PARAMS, **params}
    
    self._num_players = params["num_players"]
    self._num_actions = params["num_actions"]
    self._num_rounds = params["num_rounds"]
    self._reward_noise_std = params["reward_noise_std"]
    self._payoff_dynamics = PayoffDynamics(params["payoff_dynamics"])
    self._dilemma_type = params["dilemma_type"]
    
    # Initialize payoff matrix
    if "custom_payoff_matrix" in params and params["custom_payoff_matrix"] is not None:
      self._base_payoff_matrix = np.array(params["custom_payoff_matrix"], dtype=np.float32)
    elif self._dilemma_type in DILEMMA_PAYOFFS:
      # For predefined dilemmas, extend to N players if needed
      base_2p = DILEMMA_PAYOFFS[self._dilemma_type]["payoff_matrix"]
      if self._num_players == 2 and self._num_actions == 2:
        self._base_payoff_matrix = base_2p
      else:
        # Create an N-player extension (simplified)
        self._base_payoff_matrix = self._create_n_player_payoff_matrix()
    else:
      # Default to random payoff matrix
      self._base_payoff_matrix = self._create_n_player_payoff_matrix()
    
    # For cycling dynamics
    if "payoff_matrices_sequence" in params and params["payoff_matrices_sequence"] is not None:
      self._payoff_matrices_sequence = [np.array(m, dtype=np.float32) 
                                        for m in params["payoff_matrices_sequence"]]
    else:
      self._payoff_matrices_sequence = None
    
    # Calculate utility bounds
    max_util = float(np.max(self._base_payoff_matrix)) * self._num_rounds
    min_util = float(np.min(self._base_payoff_matrix)) * self._num_rounds
    
    # Add noise bounds if applicable
    if self._reward_noise_std > 0:
      noise_margin = 3 * self._reward_noise_std * self._num_rounds
      max_util += noise_margin
      min_util -= noise_margin
    
    game_info = pyspiel.GameInfo(
        num_distinct_actions=self._num_actions,
        max_chance_outcomes=0,
        num_players=self._num_players,
        min_utility=min_util,
        max_utility=max_util,
        utility_sum=None,  # General-sum game
        max_game_length=self._num_rounds,
    )
    
    super().__init__(_GAME_TYPE, game_info, params)
  
  def _create_n_player_payoff_matrix(self) -> np.ndarray:
    """Create an N-player payoff matrix.
    
    For N players with A actions each, we need a tensor of shape:
    (A, A, ..., A, N) where there are N dimensions for actions.
    
    For simplicity, we use a mean-field approximation where each player's
    payoff depends on their own action and the average action of others.
    """
    # For N > 2, we use a simplified structure
    # Shape: (num_actions, num_actions, num_players)
    # First dimension: player's own action
    # Second dimension: discretized average of others' actions
    
    rng = np.random.RandomState(42)
    
    if self._num_players == 2:
      # Standard 2-player matrix
      return rng.uniform(0, 5, size=(self._num_actions, self._num_actions, self._num_players))
    else:
      # For N > 2, use mean-field structure
      # Payoff depends on: (own_action, discretized_avg_others_action)
      return rng.uniform(0, 5, size=(self._num_actions, self._num_actions, self._num_players))
  
  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return ParamSocialDilemmaState(self)
  
  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return ParamSocialDilemmaObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params,
        self._num_players,
        self._num_rounds)
  
  def get_payoff_matrix(self, round_num: int) -> np.ndarray:
    """Get the payoff matrix for a specific round.
    
    Args:
      round_num: The current round number
      
    Returns:
      Payoff matrix for this round
    """
    if self._payoff_dynamics == PayoffDynamics.STATIC:
      return self._base_payoff_matrix
    
    elif self._payoff_dynamics == PayoffDynamics.CYCLING:
      if self._payoff_matrices_sequence is not None:
        idx = round_num % len(self._payoff_matrices_sequence)
        return self._payoff_matrices_sequence[idx]
      else:
        return self._base_payoff_matrix
    
    elif self._payoff_dynamics == PayoffDynamics.DRIFTING:
      # Gradually drift from base matrix
      drift_factor = 0.1 * np.sin(2 * np.pi * round_num / self._num_rounds)
      return self._base_payoff_matrix * (1 + drift_factor)
    
    elif self._payoff_dynamics == PayoffDynamics.RANDOM:
      # Random perturbation each round (deterministic based on round)
      rng = np.random.RandomState(round_num)
      noise = rng.normal(0, 0.5, self._base_payoff_matrix.shape)
      return self._base_payoff_matrix + noise
    
    else:
      return self._base_payoff_matrix


class ParamSocialDilemmaState(pyspiel.State):
  """Current state of the parameterized social dilemma game."""

  def __init__(self, game: ParamSocialDilemmaGame):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._game = game
    self._current_round = 0
    self._is_terminal = False
    self._rewards = np.zeros(game._num_players, dtype=np.float32)
    self._returns = np.zeros(game._num_players, dtype=np.float32)
    self._action_history = []  # List of joint actions
    
  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    else:
      return pyspiel.PlayerId.SIMULTANEOUS
  
  def _legal_actions(self, player):
    """Returns a list of legal actions for the player."""
    if self._is_terminal:
      return []
    return list(range(self._game._num_actions))
  
  def _apply_actions(self, actions):
    """Applies the joint actions from all players."""
    assert len(actions) == self._game._num_players
    assert not self._is_terminal
    
    # Store action in history
    self._action_history.append(list(actions))
    
    # Get current payoff matrix
    payoff_matrix = self._game.get_payoff_matrix(self._current_round)
    
    # Compute rewards based on the payoff structure
    rewards = self._compute_rewards(actions, payoff_matrix)
    
    # Add stochastic noise if configured
    if self._game._reward_noise_std > 0:
      noise = np.random.normal(0, self._game._reward_noise_std, 
                              size=self._game._num_players)
      rewards += noise
    
    self._rewards = rewards.astype(np.float32)
    self._returns += self._rewards
    
    # Advance round
    self._current_round += 1
    if self._current_round >= self._game._num_rounds:
      self._is_terminal = True
  
  def _compute_rewards(self, actions: List[int], payoff_matrix: np.ndarray) -> np.ndarray:
    """Compute rewards for all players given their joint actions.
    
    Args:
      actions: List of actions, one per player
      payoff_matrix: Current payoff matrix
      
    Returns:
      Array of rewards for each player
    """
    rewards = np.zeros(self._game._num_players, dtype=np.float32)
    
    if self._game._num_players == 2:
      # Standard 2-player case
      # payoff_matrix shape: (num_actions, num_actions, num_players)
      rewards[0] = payoff_matrix[actions[0], actions[1], 0]
      rewards[1] = payoff_matrix[actions[1], actions[0], 1]
    else:
      # N-player case using mean-field approximation
      # Each player's reward depends on their action and average of others
      for i in range(self._game._num_players):
        own_action = actions[i]
        # Compute average action of others (discretized)
        others_actions = [actions[j] for j in range(self._game._num_players) if j != i]
        avg_others = int(np.round(np.mean(others_actions)))
        avg_others = min(avg_others, self._game._num_actions - 1)
        
        rewards[i] = payoff_matrix[own_action, avg_others, i % 2]
    
    return rewards
  
  def _action_to_string(self, player, action):
    """Convert action to string."""
    return f"Action{action}"
  
  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal
  
  def rewards(self):
    """Reward at the previous step."""
    return self._rewards
  
  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns
  
  def __str__(self):
    """String representation for debugging."""
    lines = [f"Round {self._current_round}/{self._game._num_rounds}"]
    lines.append(f"Returns: {self._returns}")
    if self._action_history:
      lines.append(f"Last actions: {self._action_history[-1]}")
    return "\n".join(lines)
  
  def information_state_string(self, player):
    """Information state string (perfect information in this case)."""
    return self.__str__()
  
  def observation_string(self, player):
    """Observation string for the player."""
    return self.__str__()


class ParamSocialDilemmaObserver:
  """Observer for parameterized social dilemma game."""

  def __init__(self, iig_obs_type, params, num_players, num_rounds):
    """Initializes the observer."""
    self.iig_obs_type = iig_obs_type
    self.params = params
    self.num_players = num_players
    self.num_rounds = num_rounds
    self.dict = {}
  
  def set_from(self, state, player):
    """Updates the observation from the state."""
    pass
  
  def string_from(self, state, player):
    """Observation string from state for a player."""
    return state.observation_string(player)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, ParamSocialDilemmaGame)
