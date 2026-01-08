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

"""Axelrod-style bots for parameterized social dilemma games.

This module implements classic strategies from Robert Axelrod's tournaments,
adapted to work with N-player parameterized social dilemma games.
"""

import abc
import random
from typing import List, Optional, Dict, Any

import numpy as np
import pyspiel

from open_spiel.python.bots import bot
from .param_social_dilemma import Action


class SocialDilemmaBot(bot.Bot):
    """Base class for social dilemma bots."""
    
    def __init__(self, player_id: int, game: pyspiel.Game, **kwargs):
        """Initialize the bot."""
        super().__init__(player_id, game)
        self._num_players = game.num_players()
        self._num_actions = game.num_distinct_actions()
        self._action_history = []
        self._opponent_history = [[] for _ in range(self._num_players - 1)]
        
    def restart_at(self, state):
        """Restart the bot at the given state."""
        self._action_history = []
        self._opponent_history = [[] for _ in range(self._num_players - 1)]
        
    def _get_opponent_actions(self, state) -> List[int]:
        """Get the last actions of all opponents."""
        if len(self._action_history) == 0:
            return [0] * (self._num_players - 1)
        
        # Get the most recent actions from the state history
        actions = []
        for player in range(self._num_players):
            if player != self._player_id:
                # Get the last action of this opponent
                history = state._action_history[player]
                if history:
                    actions.append(history[-1])
                else:
                    actions.append(0)  # Default action
        
        return actions
    
    def _get_last_opponent_action(self, state, opponent_id: int) -> int:
        """Get the last action of a specific opponent."""
        if opponent_id == self._player_id:
            raise ValueError("Cannot get own action as opponent")
        
        history = state._action_history[opponent_id]
        if history:
            return history[-1]
        return 0  # Default to cooperation/first action


class AlwaysCooperateBot(SocialDilemmaBot):
    """Always cooperates (chooses action 0)."""
    
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        return 0  # Always cooperate


class AlwaysDefectBot(SocialDilemmaBot):
    """Always defects (chooses action 1 or last action)."""
    
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        return self._num_actions - 1  # Always defect


class TitForTatBot(SocialDilemmaBot):
    """Cooperates on first move, then copies opponent's last move."""
    
    def __init__(self, player_id: int, game: pyspiel.Game, **kwargs):
        """Initialize the Tit-for-Tat bot."""
        super().__init__(player_id, game, **kwargs)
        self._initial_cooperation = True
        
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        
        if self._initial_cooperation and len(self._action_history) == 0:
            return 0  # Cooperate on first move
        
        # Copy the last action of a random opponent (or average behavior)
        opponent_actions = self._get_opponent_actions(state)
        
        # For 2-player games, just copy the opponent
        if self._num_players == 2:
            return opponent_actions[0]
        
        # For N-player games, use majority or random opponent
        if len(opponent_actions) > 0:
            # Simple strategy: copy a random opponent's last action
            return random.choice(opponent_actions)
        
        return 0  # Default to cooperation


class GrimTriggerBot(SocialDilemmaBot):
    """Cooperates until an opponent defects, then defects forever."""
    
    def __init__(self, player_id: int, game: pyspiel.Game, **kwargs):
        """Initialize the Grim Trigger bot."""
        super().__init__(player_id, game, **kwargs)
        self._triggered = False
        
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        
        if self._triggered:
            return self._num_actions - 1  # Defect forever
        
        # Check if any opponent has ever defected
        for opponent_id in range(self._num_players):
            if opponent_id != self._player_id:
                last_action = self._get_last_opponent_action(state, opponent_id)
                if last_action == self._num_actions - 1:  # If opponent defected
                    self._triggered = True
                    return self._num_actions - 1
        
        return 0  # Cooperate


class GenerousTitForTatBot(SocialDilemmaBot):
    """Like Tit-for-Tat but occasionally cooperates after defection."""
    
    def __init__(self, player_id: int, game: pyspiel.Game, generosity: float = 0.1, **kwargs):
        """Initialize the Generous Tit-for-Tat bot."""
        super().__init__(player_id, game, **kwargs)
        self._generosity = generosity
        self._initial_cooperation = True
        
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        
        if self._initial_cooperation and len(self._action_history) == 0:
            return 0  # Cooperate on first move
        
        opponent_actions = self._get_opponent_actions(state)
        
        # For 2-player games
        if self._num_players == 2:
            last_opponent_action = opponent_actions[0]
            # With probability generosity, cooperate even if opponent defected
            if last_opponent_action == self._num_actions - 1 and random.random() < self._generosity:
                return 0
            return last_opponent_action
        
        # For N-player games
        if len(opponent_actions) > 0:
            last_opponent_action = random.choice(opponent_actions)
            if last_opponent_action == self._num_actions - 1 and random.random() < self._generosity:
                return 0
            return last_opponent_action
        
        return 0


class SuspiciousTitForTatBot(SocialDilemmaBot):
    """Like Tit-for-Tat but defects on first move."""
    
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        
        if len(self._action_history) == 0:
            return self._num_actions - 1  # Defect on first move
        
        # Then play Tit-for-Tat
        opponent_actions = self._get_opponent_actions(state)
        if len(opponent_actions) > 0:
            return random.choice(opponent_actions)
        
        return 0


class RandomBot(SocialDilemmaBot):
    """Chooses actions uniformly at random."""
    
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        return random.randint(0, self._num_actions - 1)


class PavlovBot(SocialDilemmaBot):
    """Win-Stay, Lose-Shift strategy."""
    
    def __init__(self, player_id: int, game: pyspiel.Game, **kwargs):
        """Initialize the Pavlov bot."""
        super().__init__(player_id, game, **kwargs)
        self._last_reward = None
        self._last_action = 0
        
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        
        if len(self._action_history) == 0:
            action = 0  # Start with cooperation
        else:
            # Get the last reward
            rewards = state.rewards()
            last_reward = rewards[self._player_id]
            
            if self._last_reward is not None:
                # If last reward was good, stay with same action
                # If last reward was bad, switch actions
                if last_reward >= self._last_reward:
                    action = self._last_action  # Stay
                else:
                    action = 1 - self._last_action if self._num_actions == 2 else (
                        (self._last_action + 1) % self._num_actions
                    )  # Shift
            else:
                action = self._last_action
        
        self._last_reward = state.rewards()[self._player_id] if len(self._action_history) > 0 else None
        self._last_action = action
        return action


class AdaptiveBot(SocialDilemmaBot):
    """Adapts strategy based on opponent behavior patterns."""
    
    def __init__(self, player_id: int, game: pyspiel.Game, memory_length: int = 10, **kwargs):
        """Initialize the Adaptive bot."""
        super().__init__(player_id, game, **kwargs)
        self._memory_length = memory_length
        self._cooperation_rates = [0.5] * self._num_players  # Estimate of cooperation rates
        
    def step(self, state):
        """Returns the selected action."""
        if state.is_terminal():
            return
        
        # Update cooperation rate estimates
        for opponent_id in range(self._num_players):
            if opponent_id != self._player_id:
                history = state._action_history[opponent_id]
                if len(history) > 0:
                    recent_history = history[-self._memory_length:]
                    cooperation_rate = sum(1 for a in recent_history if a == 0) / len(recent_history)
                    self._cooperation_rates[opponent_id] = cooperation_rate
        
        # If opponents are mostly cooperative, cooperate
        avg_cooperation = np.mean([self._cooperation_rates[i] for i in range(self._num_players) 
                                  if i != self._player_id])
        
        if avg_cooperation > 0.7:
            return 0  # Cooperate
        elif avg_cooperation < 0.3:
            return self._num_actions - 1  # Defect
        else:
            # Mixed strategy based on cooperation rates
            return 0 if random.random() < avg_cooperation else self._num_actions - 1


# Factory function to create bots by name
def create_bot(bot_type: str, player_id: int, game: pyspiel.Game, **kwargs) -> SocialDilemmaBot:
    """Create a bot of the specified type."""
    bot_classes = {
        "always_cooperate": AlwaysCooperateBot,
        "always_defect": AlwaysDefectBot,
        "tit_for_tat": TitForTatBot,
        "grim_trigger": GrimTriggerBot,
        "generous_tit_for_tat": GenerousTitForTatBot,
        "suspicious_tit_for_tat": SuspiciousTitForTatBot,
        "random": RandomBot,
        "pavlov": PavlovBot,
        "adaptive": AdaptiveBot,
    }
    
    if bot_type not in bot_classes:
        raise ValueError(f"Unknown bot type: {bot_type}. Available types: {list(bot_classes.keys())}")
    
    return bot_classes[bot_type](player_id, game, **kwargs)


# Get all available bot types
def get_available_bot_types() -> List[str]:
    """Return a list of all available bot types."""
    return [
        "always_cooperate",
        "always_defect", 
        "tit_for_tat",
        "grim_trigger",
        "generous_tit_for_tat",
        "suspicious_tit_for_tat",
        "random",
        "pavlov",
        "adaptive",
    ]
