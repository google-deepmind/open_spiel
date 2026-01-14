#!/usr/bin/env python3
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

"""Example script demonstrating basic usage of the parameterized social dilemma game.

This script shows how to:
1. Create games with different configurations
2. Play basic games manually
3. Use built-in bots
4. Analyze game outcomes
"""

import numpy as np
import pyspiel

from open_spiel.python.games.param_social_dilemma_bots import create_bot, get_available_bot_types


def basic_2player_example():
    """Demonstrate basic 2-player prisoner's dilemma."""
    print("=== Basic 2-Player Prisoner's Dilemma ===")
    
    # Create a 2-player game
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "termination_probability": 0.2,  # 20% chance to end each round
        "max_game_length": 10
    })
    
    state = game.new_initial_state()
    
    print(f"Game created: {game.get_type().long_name}")
    print(f"Players: {game.num_players()}")
    print(f"Actions per player: {game.num_distinct_actions()}")
    print()
    
    # Play a few rounds manually
    round_num = 1
    while not state.is_terminal() and round_num <= 5:
        print(f"Round {round_num}:")
        
        # Both players cooperate
        actions = [0, 0]  # [Player 0, Player 1]
        print(f"  Actions: {['Cooperate' if a == 0 else 'Defect' for a in actions]}")
        
        state.apply_actions(actions)
        rewards = state.rewards()
        print(f"  Rewards: {rewards}")
        
        returns = state.returns()
        print(f"  Total Returns: {returns}")
        
        # Handle chance node
        if state.current_player() == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            print(f"  Chance outcomes: {outcomes}")
            
            # For demo, always continue if possible
            if outcomes[0][1] > 0:  # If there's a chance to continue
                state.apply_action(0)  # CONTINUE
                print("  Game continues...")
            else:
                state.apply_action(1)  # STOP
                print("  Game terminated!")
                break
        
        round_num += 1
        print()
    
    print("Final game state:")
    print(f"  Returns: {state.returns()}")
    print(f"  History: {state}")
    print()


def n_player_example():
    """Demonstrate N-player social dilemma."""
    print("=== N-Player Social Dilemma ===")
    
    # Create a 4-player game
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 4,
        "num_actions": 2,
        "termination_probability": 0.1,
        "max_game_length": 8
    })
    
    state = game.new_initial_state()
    
    print(f"Created {game.num_players()}-player game")
    print()
    
    # Play a few rounds with different strategies
    strategies = [
        "Always Cooperate",  # Player 0
        "Always Defect",     # Player 1  
        "Tit-for-Tat",       # Player 2
        "Random"             # Player 3
    ]
    
    round_num = 1
    while not state.is_terminal() and round_num <= 4:
        print(f"Round {round_num}:")
        
        # Generate actions based on strategies
        actions = []
        for player, strategy in enumerate(strategies):
            if strategy == "Always Cooperate":
                actions.append(0)
            elif strategy == "Always Defect":
                actions.append(1)
            elif strategy == "Tit-for-Tat":
                if round_num == 1:
                    actions.append(0)  # Cooperate first
                else:
                    # Copy a random opponent's last action
                    last_actions = []
                    for other in range(game.num_players()):
                        if other != player and len(state._action_history[other]) > 0:
                            last_actions.append(state._action_history[other][-1])
                    actions.append(last_actions[0] if last_actions else 0)
            elif strategy == "Random":
                actions.append(np.random.randint(0, 2))
        
        action_names = ['Cooperate' if a == 0 else 'Defect' for a in actions]
        print(f"  Actions: {dict(zip(range(4), zip(strategies, action_names)))}")
        
        state.apply_actions(actions)
        rewards = state.rewards()
        print(f"  Rewards: {rewards}")
        
        returns = state.returns()
        print(f"  Total Returns: {returns}")
        
        # Handle chance node
        if state.current_player() == pyspiel.PlayerId.CHANCE:
            if np.random.random() > 0.1:  # 90% chance to continue
                state.apply_action(0)
                print("  Game continues...")
            else:
                state.apply_action(1)
                print("  Game terminated!")
                break
        
        round_num += 1
        print()
    
    print("Final rankings:")
    final_returns = state.returns()
    sorted_players = sorted(range(len(final_returns)), key=lambda i: final_returns[i], reverse=True)
    for rank, player in enumerate(sorted_players):
        print(f"  {rank+1}. Player {player} ({strategies[player]}): {final_returns[player]}")
    print()


def dynamic_payoffs_example():
    """Demonstrate dynamic payoff matrices."""
    print("=== Dynamic Payoffs Example ===")
    
    def escalating_tension(base_matrix, timestep):
        """Payoffs that become more competitive over time."""
        # As time goes on, defection becomes more tempting
        tension_factor = 1 + 0.2 * timestep
        
        new_matrix = []
        for player_payoffs in base_matrix:
            new_player_payoffs = []
            for action_payoffs in player_payoffs:
                new_action_payoffs = []
                for payoff in action_payoffs:
                    # Increase payoff for defection, decrease for cooperation
                    if payoff > 2:  # Likely a defection payoff
                        new_action_payoffs.append(payoff * tension_factor)
                    else:  # Likely a cooperation payoff
                        new_action_payoffs.append(payoff / tension_factor)
                new_player_payoffs.append(new_action_payoffs)
            new_matrix.append(new_player_payoffs)
        
        return new_matrix
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "termination_probability": 0.0,  # Never terminate for this demo
        "payoff_function": escalating_tension
    })
    
    state = game.new_initial_state()
    
    print("Game with escalating tension over time...")
    print("Both players always cooperate:")
    print()
    
    for round_num in range(1, 6):
        state.apply_actions([0, 0])  # Both cooperate
        rewards = state.rewards()
        returns = state.returns()
        
        print(f"Round {round_num}:")
        print(f"  Rewards: {rewards}")
        print(f"  Total Returns: {returns}")
        
        state.apply_action(0)  # Continue
    
    print()


def stochastic_rewards_example():
    """Demonstrate stochastic rewards."""
    print("=== Stochastic Rewards Example ===")
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "termination_probability": 0.0,
        "reward_noise": {"type": "gaussian", "std": 0.5},
        "seed": 42
    })
    
    state = game.new_initial_state()
    
    print("Game with Gaussian noise (std=0.5)")
    print("Both players always cooperate:")
    print()
    
    for round_num in range(1, 6):
        state.apply_actions([0, 0])  # Both cooperate
        rewards = state.rewards()
        returns = state.returns()
        
        print(f"Round {round_num}:")
        print(f"  Noisy Rewards: {rewards}")
        print(f"  Total Returns: {returns}")
        
        state.apply_action(0)  # Continue
    
    print()


def bot_tournament_example():
    """Demonstrate a tournament between different bot strategies."""
    print("=== Axelrod-Style Bot Tournament ===")
    
    available_bots = get_available_bot_types()
    print(f"Available bots: {available_bots}")
    print()
    
    # Create a tournament between selected bots
    bot_types = ["always_cooperate", "always_defect", "tit_for_tat", "grim_trigger", 
                "generous_tit_for_tat", "random"]
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "termination_probability": 0.1,
        "max_game_length": 20,
        "seed": 42
    })
    
    # Round-robin tournament
    scores = {bot_type: 0 for bot_type in bot_types}
    
    print("Round-robin tournament (10 rounds each):")
    print()
    
    for i, bot1_type in enumerate(bot_types):
        for j, bot2_type in enumerate(bot_types[i:], i):
            if i == j:
                continue
            
            # Play multiple games
            total_score1 = 0
            total_score2 = 0
            
            for game_num in range(10):
                # Create bots
                bot1 = create_bot(bot1_type, 0, game)
                bot2 = create_bot(bot2_type, 1, game)
                
                state = game.new_initial_state()
                
                # Play until termination
                while not state.is_terminal():
                    actions = [bot1.step(state), bot2.step(state)]
                    state.apply_actions(actions)
                    
                    if state.current_player() == pyspiel.PlayerId.CHANCE:
                        outcomes = state.chance_outcomes()
                        # Use actual chance outcome
                        if np.random.random() < outcomes[0][1]:  # Continue probability
                            state.apply_action(0)
                        else:
                            state.apply_action(1)
                            break
                
                final_returns = state.returns()
                total_score1 += final_returns[0]
                total_score2 += final_returns[1]
            
            avg_score1 = total_score1 / 10
            avg_score2 = total_score2 / 10
            
            scores[bot1_type] += avg_score1
            scores[bot2_type] += avg_score2
            
            print(f"{bot1_type} vs {bot2_type}: {avg_score1:.1f} - {avg_score2:.1f}")
    
    print()
    print("Final tournament rankings:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (bot_type, score) in enumerate(sorted_scores):
        print(f"  {rank+1}. {bot_type}: {score:.1f}")
    print()


def custom_payoff_matrix_example():
    """Demonstrate custom payoff matrices."""
    print("=== Custom Payoff Matrix Example ===")
    
    # Create a custom payoff matrix for a coordination game
    # Both players want to choose the same action
    coordination_payoff = [
        [[5, 0], [0, 5]],  # Player 0: high payoff if match, low if mismatch
        [[5, 0], [0, 5]]   # Player 1: same structure
    ]
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "payoff_matrix": coordination_payoff,
        "termination_probability": 0.3
    })
    
    state = game.new_initial_state()
    
    print("Coordination game - players get high payoff if they choose the same action")
    print()
    
    # Demonstrate different action combinations
    action_combinations = [
        ([0, 0], "Both choose action 0"),
        ([1, 1], "Both choose action 1"), 
        ([0, 1], "Mismatched actions"),
        ([1, 0], "Mismatched actions")
    ]
    
    for actions, description in action_combinations:
        state = game.new_initial_state()  # Reset state
        state.apply_actions(actions)
        rewards = state.rewards()
        
        print(f"{description}: {actions} -> Rewards: {rewards}")
    
    print()


def main():
    """Run all examples."""
    print("Parameterized Social Dilemma Game Examples")
    print("=" * 50)
    print()
    
    basic_2player_example()
    n_player_example()
    dynamic_payoffs_example()
    stochastic_rewards_example()
    bot_tournament_example()
    custom_payoff_matrix_example()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()
