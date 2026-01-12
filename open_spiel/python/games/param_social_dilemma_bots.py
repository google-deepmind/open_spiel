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

"""Simple bot examples for parameterized social dilemma game.

This module demonstrates how to use different types of bots/policies
in the parameterized social dilemma game, useful for MARL research.
"""

import numpy as np
import pyspiel


class AlwaysCooperateBot:
  """Bot that always cooperates (plays action 0)."""
  
  def __init__(self, player_id):
    self.player_id = player_id
  
  def step(self, state):
    """Returns the cooperation action (0)."""
    return 0
  
  def __str__(self):
    return f"AlwaysCooperate(P{self.player_id})"


class AlwaysDefectBot:
  """Bot that always defects (plays action 1)."""
  
  def __init__(self, player_id):
    self.player_id = player_id
  
  def step(self, state):
    """Returns the defect action (1)."""
    return 1
  
  def __str__(self):
    return f"AlwaysDefect(P{self.player_id})"


class TitForTatBot:
  """Bot that copies the opponent's last action.
  
  Starts by cooperating, then mirrors opponent's previous move.
  Note: Only works for 2-player games.
  """
  
  def __init__(self, player_id):
    self.player_id = player_id
    self.opponent_id = 1 - player_id  # Assumes 2 players
    self.last_opponent_action = 0  # Start with cooperation
  
  def step(self, state):
    """Returns the opponent's last action."""
    # Get action history
    if hasattr(state, '_action_history') and len(state._action_history) > 0:
      self.last_opponent_action = state._action_history[-1][self.opponent_id]
    
    return self.last_opponent_action
  
  def __str__(self):
    return f"TitForTat(P{self.player_id})"


class RandomBot:
  """Bot that plays random actions."""
  
  def __init__(self, player_id, num_actions=2, seed=None):
    self.player_id = player_id
    self.num_actions = num_actions
    self.rng = np.random.RandomState(seed)
  
  def step(self, state):
    """Returns a random action."""
    return self.rng.randint(0, self.num_actions)
  
  def __str__(self):
    return f"Random(P{self.player_id})"


class EpsilonGreedyBot:
  """Bot that cooperates with probability epsilon, defects otherwise."""
  
  def __init__(self, player_id, epsilon=0.8, seed=None):
    self.player_id = player_id
    self.epsilon = epsilon
    self.rng = np.random.RandomState(seed)
  
  def step(self, state):
    """Returns cooperation with probability epsilon."""
    if self.rng.random() < self.epsilon:
      return 0  # Cooperate
    else:
      return 1  # Defect
  
  def __str__(self):
    return f"EpsilonGreedy(P{self.player_id}, ε={self.epsilon})"


class AdaptiveBot:
  """Bot that adapts its cooperation rate based on opponent behavior.
  
  Increases cooperation if opponent cooperates, decreases if opponent defects.
  """
  
  def __init__(self, player_id, initial_coop_prob=0.5, learning_rate=0.1, seed=None):
    self.player_id = player_id
    self.opponent_id = 1 - player_id  # Assumes 2 players
    self.coop_prob = initial_coop_prob
    self.learning_rate = learning_rate
    self.rng = np.random.RandomState(seed)
  
  def step(self, state):
    """Returns action based on adaptive cooperation probability."""
    # Update cooperation probability based on opponent's last action
    if hasattr(state, '_action_history') and len(state._action_history) > 0:
      last_opponent_action = state._action_history[-1][self.opponent_id]
      
      if last_opponent_action == 0:  # Opponent cooperated
        # Increase cooperation probability
        self.coop_prob = min(1.0, self.coop_prob + self.learning_rate)
      else:  # Opponent defected
        # Decrease cooperation probability
        self.coop_prob = max(0.0, self.coop_prob - self.learning_rate)
    
    # Choose action based on current cooperation probability
    if self.rng.random() < self.coop_prob:
      return 0  # Cooperate
    else:
      return 1  # Defect
  
  def __str__(self):
    return f"Adaptive(P{self.player_id}, p_coop={self.coop_prob:.2f})"


def simulate_game(game, bots, verbose=True):
  """Simulate a game with the given bots.
  
  Args:
    game: OpenSpiel game instance
    bots: List of bot instances, one per player
    verbose: Whether to print game progress
    
  Returns:
    Final returns for each player
  """
  state = game.new_initial_state()
  
  if verbose:
    print(f"\nSimulating game with {len(bots)} players:")
    for i, bot in enumerate(bots):
      print(f"  Player {i}: {bot}")
    print()
  
  round_num = 0
  while not state.is_terminal():
    # Get actions from all bots
    actions = [bot.step(state) for bot in bots]
    
    if verbose:
      action_strs = [f"P{i}={a}" for i, a in enumerate(actions)]
      print(f"Round {round_num + 1}: {', '.join(action_strs)}")
    
    # Apply actions
    state.apply_actions(actions)
    
    if verbose:
      print(f"  Rewards: {state.rewards()}")
      print(f"  Returns: {state.returns()}")
    
    round_num += 1
  
  if verbose:
    print(f"\nFinal returns: {state.returns()}\n")
  
  return state.returns()


def example_bot_tournament():
  """Run a tournament between different bot types."""
  print("="*70)
  print("  BOT TOURNAMENT - PARAMETERIZED SOCIAL DILEMMA")
  print("="*70)
  
  # Create game
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 10,
      "dilemma_type": "prisoners_dilemma"
  })
  
  # Define bot types
  bot_classes = [
      (AlwaysCooperateBot, {}),
      (AlwaysDefectBot, {}),
      (TitForTatBot, {}),
      (EpsilonGreedyBot, {"epsilon": 0.8}),
      (RandomBot, {"seed": 42}),
  ]
  
  # Run round-robin tournament
  results = {}
  
  for i, (BotClass1, kwargs1) in enumerate(bot_classes):
    for j, (BotClass2, kwargs2) in enumerate(bot_classes):
      if i <= j:  # Only run each matchup once (including self-play)
        bot1 = BotClass1(0, **kwargs1)
        bot2 = BotClass2(1, **kwargs2)
        
        print(f"\nMatch: {bot1} vs {bot2}")
        print("-" * 50)
        
        returns = simulate_game(game, [bot1, bot2], verbose=False)
        
        matchup_name = f"{bot1} vs {bot2}"
        results[matchup_name] = returns
        
        print(f"Results: {bot1} = {returns[0]:.1f}, {bot2} = {returns[1]:.1f}")
  
  print("\n" + "="*70)
  print("  TOURNAMENT RESULTS")
  print("="*70)
  
  for matchup, returns in results.items():
    print(f"{matchup}: [{returns[0]:.1f}, {returns[1]:.1f}]")


def example_n_player_bots():
  """Example with N-player game (N > 2)."""
  print("\n" + "="*70)
  print("  N-PLAYER GAME WITH BOTS")
  print("="*70)
  
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 4,
      "num_actions": 2,
      "num_rounds": 5,
      "dilemma_type": "public_goods"
  })
  
  # Create mixed bot population
  bots = [
      AlwaysCooperateBot(0),
      AlwaysDefectBot(1),
      RandomBot(2, seed=42),
      EpsilonGreedyBot(3, epsilon=0.7, seed=43)
  ]
  
  simulate_game(game, bots, verbose=True)


def example_dynamic_environment():
  """Example with dynamic payoffs and adaptive bots."""
  print("\n" + "="*70)
  print("  DYNAMIC ENVIRONMENT WITH ADAPTIVE BOTS")
  print("="*70)
  
  # Create game with drifting payoffs
  game = pyspiel.load_game("python_param_social_dilemma", {
      "num_players": 2,
      "num_actions": 2,
      "num_rounds": 20,
      "payoff_dynamics": "drifting",
      "dilemma_type": "prisoners_dilemma"
  })
  
  bots = [
      AdaptiveBot(0, initial_coop_prob=0.5, learning_rate=0.1, seed=42),
      TitForTatBot(1)
  ]
  
  print("Adaptive bot adjusts its cooperation based on opponent's behavior")
  print("in a non-stationary environment.\n")
  
  simulate_game(game, bots, verbose=True)


if __name__ == "__main__":
  example_bot_tournament()
  example_n_player_bots()
  example_dynamic_environment()
  
  print("\n" + "#"*70)
  print("#  All bot examples completed!")
  print("#"*70 + "\n")
