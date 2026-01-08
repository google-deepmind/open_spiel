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

"""Python implementation of parameterized social dilemma games.

This module provides a flexible framework for N-player social dilemma games with:
- Variable number of agents (N ≥ 2)
- Dynamic payoff matrices that can change over time
- Stochastic reward noise
- Configurable game parameters
"""

import enum
import random
from typing import List, Optional, Callable, Dict, Any

import numpy as np
import pyspiel

# Default parameters
_DEFAULT_PARAMS = {
    "num_players": 2,
    "num_actions": 2,
    "payoff_matrix": None,  # Will use default prisoner's dilemma if None
    "termination_probability": 0.125,
    "max_game_length": 9999,
    "payoff_function": None,  # Function for dynamic payoffs
    "reward_noise": None,  # {"type": "gaussian", "std": 0.1} or similar
    "seed": None,
}

# Default prisoner's dilemma payoff matrix for 2 players, 2 actions
# Based on standard PD: T > R > P > S
# T (Temptation) = 5, R (Reward) = 3, P (Punishment) = 1, S (Sucker's payoff) = 0
_DEFAULT_PAYOFF_MATRIX = [
    [[3, 0], [5, 1]],  # Player 0's payoffs when Player 1 chooses action 0 or 1
    [[3, 5], [0, 1]]   # Player 1's payoffs when Player 0 chooses action 0 or 1
]

_GAME_TYPE = pyspiel.GameType(
    short_name="python_param_social_dilemma",
    long_name="Python Parameterized Social Dilemma",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=10,  # Reasonable upper limit
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class Action(enum.IntEnum):
    """Default action names for 2-action games."""
    COOPERATE = 0
    DEFECT = 1


class Chance(enum.IntEnum):
    """Chance outcomes for game termination."""
    CONTINUE = 0
    STOP = 1


class ParamSocialDilemmaGame(pyspiel.Game):
    """Parameterized social dilemma game supporting N players with dynamic payoffs."""

    def __init__(self, params=None):
        """Initialize the game with parameters."""
        if params is None:
            params = _DEFAULT_PARAMS.copy()
        
        # Validate parameters
        num_players = params["num_players"]
        num_actions = params["num_actions"]
        
        if num_players < 2:
            raise ValueError("num_players must be >= 2")
        if num_actions < 2:
            raise ValueError("num_actions must be >= 2")
        
        # Set up payoff matrix
        payoff_matrix = params["payoff_matrix"]
        if payoff_matrix is None:
            payoff_matrix = self._create_default_payoff_matrix(num_players, num_actions)
        
        # Validate payoff matrix dimensions
        self._validate_payoff_matrix(payoff_matrix, num_players, num_actions)
        
        # Calculate utility bounds
        min_payoff = np.min(payoff_matrix)
        max_payoff = np.max(payoff_matrix)
        max_game_length = params["max_game_length"]
        
        super().__init__(
            _GAME_TYPE,
            pyspiel.GameInfo(
                num_distinct_actions=num_actions,
                max_chance_outcomes=2,
                num_players=num_players,
                min_utility=min_payoff * max_game_length,
                max_utility=max_payoff * max_game_length,
                utility_sum=None,
                max_game_length=max_game_length,
            ),
            params,
        )
        
        self._payoff_matrix = np.array(payoff_matrix)
        self._termination_probability = params["termination_probability"]
        self._payoff_function = params["payoff_function"]
        self._reward_noise = params["reward_noise"]
        self._seed = params["seed"]
        
        # Initialize random number generator
        if self._seed is not None:
            self.rng = random.Random(self._seed)
            np.random.seed(self._seed)
        else:
            self.rng = random.Random()

    def _create_default_payoff_matrix(self, num_players: int, num_actions: int) -> List:
        """Create a default N-player prisoner's dilemma-like payoff matrix."""
        if num_players == 2 and num_actions == 2:
            return _DEFAULT_PAYOFF_MATRIX
        
        # For N-player games, create a simple social dilemma structure
        # Players who cooperate (action 0) get moderate payoff
        # Defectors (action 1) get higher payoff but reduce others' payoffs
        payoff_matrix = []
        
        for player in range(num_players):
            player_payoffs = []
            for player_action in range(num_actions):
                action_payoffs = []
                for others_action in range(num_actions ** (num_players - 1)):
                    # Decode others' actions
                    others_actions = []
                    temp = others_action
                    for other in range(num_players):
                        if other != player:
                            others_actions.append(temp % num_actions)
                            temp //= num_actions
                    
                    # Calculate payoff based on cooperation/defection pattern
                    num_cooperators = sum(1 for a in others_actions if a == 0)
                    if player_action == 0:  # Cooperate
                        payoff = 2 + num_cooperators * 0.5
                    else:  # Defect
                        payoff = 4 + num_cooperators * 0.2
                    
                    action_payoffs.append(payoff)
                player_payoffs.append(action_payoffs)
            payoff_matrix.append(player_payoffs)
        
        return payoff_matrix

    def _validate_payoff_matrix(self, payoff_matrix: List, num_players: int, num_actions: int):
        """Validate that the payoff matrix has correct dimensions."""
        if len(payoff_matrix) != num_players:
            raise ValueError(f"Payoff matrix must have {num_players} dimensions")
        
        for player_idx, player_payoffs in enumerate(payoff_matrix):
            if len(player_payoffs) != num_actions:
                raise ValueError(f"Player {player_idx} must have {num_actions} action payoffs")
            
            for action_idx, action_payoffs in enumerate(player_payoffs):
                expected_size = num_actions ** (num_players - 1)
                if len(action_payoffs) != expected_size:
                    raise ValueError(
                        f"Player {player_idx}, action {action_idx} must have {expected_size} payoff entries"
                    )

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return ParamSocialDilemmaState(self, self._termination_probability)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return ParamSocialDilemmaObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)

    def get_payoff_matrix(self, timestep: int = 0) -> np.ndarray:
        """Get the payoff matrix, potentially modified by dynamic function."""
        if self._payoff_function is not None:
            return np.array(self._payoff_function(self._payoff_matrix, timestep))
        return self._payoff_matrix.copy()


class ParamSocialDilemmaState(pyspiel.State):
    """Current state of the parameterized social dilemma game."""

    def __init__(self, game, termination_probability):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._current_iteration = 1
        self._termination_probability = termination_probability
        self._is_chance = False
        self._game_over = False
        self._num_players = game.num_players()
        self._num_actions = game.num_distinct_actions()
        self._rewards = np.zeros(self._num_players)
        self._returns = np.zeros(self._num_players)
        self._action_history = [[] for _ in range(self._num_players)]

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self._is_chance:
            return pyspiel.PlayerId.CHANCE
        else:
            return pyspiel.PlayerId.SIMULTANEOUS

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0
        return list(range(self._num_actions))

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self._is_chance
        return [(Chance.CONTINUE, 1 - self._termination_probability),
                (Chance.STOP, self._termination_probability)]

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        assert self._is_chance and not self._game_over
        self._current_iteration += 1
        self._is_chance = False
        self._game_over = (action == Chance.STOP)
        if self._current_iteration > self.get_game().max_game_length():
            self._game_over = True

    def _apply_actions(self, actions):
        """Applies the specified actions (per player) to the state."""
        assert not self._is_chance and not self._game_over
        assert len(actions) == self._num_players
        
        self._is_chance = True
        
        # Store action history
        for player, action in enumerate(actions):
            self._action_history[player].append(action)
        
        # Get current payoff matrix (potentially dynamic)
        payoff_matrix = self.get_game().get_payoff_matrix(self._current_iteration)
        
        # Calculate rewards for each player
        for player in range(self._num_players):
            # Calculate index for payoff lookup
            payoff_index = 0
            multiplier = 1
            for other_player in range(self._num_players):
                if other_player != player:
                    payoff_index += actions[other_player] * multiplier
                    multiplier *= self._num_actions
            
            # Get base payoff
            base_reward = payoff_matrix[player][actions[player]][payoff_index]
            
            # Add noise if specified
            reward = self._add_noise(base_reward)
            self._rewards[player] = reward
        
        self._returns += self._rewards

    def _add_noise(self, base_reward: float) -> float:
        """Add stochastic noise to rewards if configured."""
        game = self.get_game()
        if game._reward_noise is None:
            return base_reward
        
        noise_config = game._reward_noise
        noise_type = noise_config.get("type", "gaussian")
        
        if noise_type == "gaussian":
            std = noise_config.get("std", 0.1)
            noise = np.random.normal(0, std)
        elif noise_type == "uniform":
            range_val = noise_config.get("range", 0.1)
            noise = np.random.uniform(-range_val, range_val)
        elif noise_type == "discrete":
            noise_values = noise_config.get("values", [-0.1, 0, 0.1])
            noise = game.rng.choice(noise_values)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return base_reward + noise

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return Chance(action).name
        else:
            if self._num_actions == 2:
                return Action(action).name
            else:
                return f"Action{action}"

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def rewards(self):
        """Reward at the previous step."""
        return self._rewards

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self._returns

    def __str__(self):
        """String for debug purposes."""
        action_strings = []
        for player in range(self._num_players):
            history = self.action_history_string(player)
            action_strings.append(f"p{player}:{history}")
        return " ".join(action_strings)

    def action_history_string(self, player):
        """Get the action history for a player as a string."""
        history = self._action_history[player]
        if self._num_actions == 2:
            return "".join(Action(action).name[0] for action in history)
        else:
            return "".join(str(action) for action in history)


class ParamSocialDilemmaObserver:
    """Observer, conforming to the PyObserver interface."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {}

    def set_from(self, state, player):
        """Set the observation from state for a given player."""
        # Create observation tensor with action history and current iteration
        game = state.get_game()
        num_players = game.num_players()
        num_actions = game.num_distinct_actions()
        max_length = game.max_game_length()
        
        # Create observation: [own_history, others_history, iteration]
        obs_size = max_length * num_players + 1
        self.tensor = np.zeros(obs_size, dtype=np.float32)
        
        # Fill action history
        for p in range(num_players):
            history = state._action_history[p]
            for i, action in enumerate(history):
                self.tensor[p * max_length + i] = action
        
        # Add current iteration
        self.tensor[-1] = state._current_iteration / max_length

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        if self.iig_obs_type.public_info:
            histories = []
            for p in range(state.get_game().num_players()):
                histories.append(f"p{p}:{state.action_history_string(p)}")
            return " ".join(histories)
        else:
            return f"p{player}:{state.action_history_string(player)}"


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, ParamSocialDilemmaGame)
