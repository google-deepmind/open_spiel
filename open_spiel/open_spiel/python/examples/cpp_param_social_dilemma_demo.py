#!/usr/bin/env python3
"""Comprehensive example demonstrating both C++ and Python implementations of parameterized social dilemma.

This script shows the equivalence and capabilities of both implementations.
"""

import pyspiel
import numpy as np

def demonstrate_cpp_implementation():
    """Demonstrate the C++ implementation."""
    print("=== C++ Implementation Demo ===")
    
    # Create a 2-player game
    game = pyspiel.load_game("param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "termination_probability": 0.2,
        "max_game_length": 10
    })
    
    print(f"Game: {game.get_type().long_name}")
    print(f"Players: {game.num_players()}")
    print(f"Actions: {game.num_distinct_actions()}")
    print(f"Max utility: {game.max_utility():.2f}")
    print(f"Min utility: {game.min_utility():.2f}")
    print()
    
    # Play a few rounds
    state = game.new_initial_state()
    
    strategies = ["Tit-for-Tat", "Always Cooperate"]
    actions = [[0, 1], [0, 0]]  # [Tit-for-Tat, Always Cooperate]
    
    round_num = 1
    while not state.is_terminal() and round_num <= 5:
        print(f"Round {round_num}:")
        
        # Choose actions based on strategies
        if round_num == 1:
            current_actions = actions[0]  # Tit-for-Tat cooperates first
        else:
            # Simple strategy for demo
            current_actions = [0, 0]  # Both cooperate
        
        action_names = ['Cooperate' if a == 0 else 'Defect' for a in current_actions]
        print(f"  Actions: {list(zip(strategies, action_names))}")
        
        state.apply_actions(current_actions)
        rewards = state.rewards()
        returns = state.returns()
        
        print(f"  Rewards: {rewards}")
        print(f"  Total Returns: {returns}")
        
        # Handle chance node
        if state.current_player() == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            continue_prob = outcomes[0][1]
            if np.random.random() < continue_prob:
                state.apply_action(0)  # CONTINUE
                print("  Game continues...")
            else:
                state.apply_action(1)  # STOP
                print("  Game terminated!")
                break
        
        round_num += 1
        print()
    
    print("Final state:")
    print(f"  Returns: {state.returns()}")
    print(f"  History: {state.to_string()}")
    print()


def compare_implementations():
    """Compare C++ and Python implementations for equivalence."""
    print("=== Implementation Comparison ===")
    
    # Create equivalent games
    params = {
        "num_players": 2,
        "num_actions": 2,
        "termination_probability": 0.0,  # Never terminate for comparison
        "max_game_length": 5
    }
    
    cpp_game = pyspiel.load_game("param_social_dilemma", params)
    py_game = pyspiel.load_game("python_param_social_dilemma", params)
    
    print("Testing action sequences for equivalence:")
    
    test_sequences = [
        ([0, 0], "Both Cooperate"),
        ([0, 1], "P0 Cooperate, P1 Defect"),
        ([1, 0], "P0 Defect, P1 Cooperate"),
        ([1, 1], "Both Defect")
    ]
    
    for actions, description in test_sequences:
        print(f"\n{description}:")
        
        # Test C++ implementation
        cpp_state = cpp_game.new_initial_state()
        cpp_state.apply_actions(actions)
        cpp_rewards = cpp_state.rewards()
        
        # Test Python implementation
        py_state = py_game.new_initial_state()
        py_state.apply_actions(actions)
        py_rewards = py_state.rewards()
        
        print(f"  C++ rewards: {cpp_rewards}")
        print(f"  Python rewards: {py_rewards}")
        
        # Check equivalence (allowing for small floating point differences)
        equivalent = True
        for i in range(2):
            if abs(cpp_rewards[i] - py_rewards[i]) > 1e-6:
                equivalent = False
                break
        
        print(f"  Equivalent: {'✓' if equivalent else '✗'}")


def demonstrate_advanced_features():
    """Demonstrate advanced features of the C++ implementation."""
    print("=== Advanced Features Demo ===")
    
    # Test with reward noise
    print("1. Stochastic Rewards (Gaussian noise):")
    game = pyspiel.load_game("param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "reward_noise_std": 0.5,
        "reward_noise_type": "gaussian",
        "seed": 42
    })
    
    state = game.new_initial_state()
    state.apply_actions([0, 0])  # Both cooperate
    
    rewards = state.rewards()
    print(f"   Base payoff: 3.0, Noisy reward: {rewards}")
    print()
    
    # Test custom payoff matrix
    print("2. Custom Payoff Matrix (Coordination Game):")
    coordination_payoff = [5, 0, 0, 5, 5, 0, 0, 5]  # Both get 5 if match, 0 otherwise
    
    game = pyspiel.load_game("param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "payoff_matrix": coordination_payoff
    })
    
    state = game.new_initial_state()
    
    test_cases = [
        ([0, 0], "Both choose action 0"),
        ([1, 1], "Both choose action 1"),
        ([0, 1], "Mismatched actions"),
        ([1, 0], "Mismatched actions")
    ]
    
    for actions, description in test_cases:
        state = game.new_initial_state()  # Reset
        state.apply_actions(actions)
        rewards = state.rewards()
        print(f"   {description}: {actions} -> {rewards}")
    print()
    
    # Test 3-player game
    print("3. Three-Player Game:")
    game = pyspiel.load_game("param_social_dilemma", {
        "num_players": 3,
        "num_actions": 2,
        "termination_probability": 0.0
    })
    
    state = game.new_initial_state()
    state.apply_actions([0, 0, 0])  # All cooperate
    rewards = state.rewards()
    print(f"   All cooperate: {rewards}")
    
    state = game.new_initial_state()  # Reset
    state.apply_actions([1, 0, 1])  # Mixed actions
    rewards = state.rewards()
    print(f"   Mixed actions: {rewards}")
    print()


def demonstrate_observation_features():
    """Demonstrate observation and information state features."""
    print("=== Observation Features Demo ===")
    
    game = pyspiel.load_game("param_social_dilemma", {
        "num_players": 2,
        "num_actions": 2,
        "max_game_length": 5
    })
    
    state = game.new_initial_state()
    
    # Play a few rounds
    action_sequence = [[0, 1], [1, 0], [0, 0]]
    
    for round_num, actions in enumerate(action_sequence, 1):
        state.apply_actions(actions)
        
        print(f"Round {round_num}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {state.rewards()}")
        print(f"  Returns: {state.returns()}")
        
        # Show information states
        for player in range(2):
            info_state = state.information_state_string(player)
            print(f"  P{player} info: {info_state}")
        
        # Show observation tensor for player 0
        obs_tensor = state.observation_tensor(0)
        print(f"  P0 observation size: {len(obs_tensor)}")
        
        state.apply_action(0)  # Continue
        print()


def demonstrate_performance_characteristics():
    """Demonstrate performance characteristics."""
    print("=== Performance Characteristics ===")
    
    # Test different game sizes
    sizes = [
        (2, 2, "2-player, 2-action"),
        (3, 2, "3-player, 2-action"),
        (4, 2, "4-player, 2-action"),
        (2, 3, "2-player, 3-action")
    ]
    
    for num_players, num_actions, description in sizes:
        game = pyspiel.load_game("param_social_dilemma", {
            "num_players": num_players,
            "num_actions": num_actions
        })
        
        state = game.new_initial_state()
        
        # Time state creation
        import time
        start_time = time.time()
        for _ in range(1000):
            test_state = game.new_initial_state()
            test_state.apply_actions([0] * num_players)
        creation_time = (time.time() - start_time) / 1000
        
        # Time action application
        start_time = time.time()
        for _ in range(1000):
            state.apply_actions([0] * num_players)
        action_time = (time.time() - start_time) / 1000
        
        print(f"{description}:")
        print(f"  State creation: {creation_time*1000:.3f}μs")
        print(f"  Action application: {action_time*1000:.3f}μs")
        print()


def main():
    """Run all demonstrations."""
    print("Parameterized Social Dilemma - C++ Implementation Demo")
    print("=" * 60)
    print()
    
    demonstrate_cpp_implementation()
    compare_implementations()
    demonstrate_advanced_features()
    demonstrate_observation_features()
    demonstrate_performance_characteristics()
    
    print("🎉 All demonstrations completed!")
    print()
    print("Key Benefits of C++ Implementation:")
    print("  ✓ High performance for large-scale simulations")
    print("  ✓ Full OpenSpiel integration")
    print("  ✓ Compatible with existing C++ algorithms")
    print("  ✓ Memory efficient state representation")
    print("  ✓ Thread-safe execution")
    print()
    print("Both C++ and Python implementations are available!")
    print("- Use 'param_social_dilemma' for C++ version")
    print("- Use 'python_param_social_dilemma' for Python version")


if __name__ == "__main__":
    main()
