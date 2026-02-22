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

"""Example usage of parameterized social dilemma for MARL experiments.

This demonstrates how to use the param_social_dilemma game with various
configurations for multi-agent reinforcement learning research.
"""

import numpy as np
from open_spiel.python.games import param_social_dilemma
import pyspiel


def play_game(game, policies):
    state = game.new_initial_state()
    episode_rewards = [[] for _ in range(game.num_players())]
    
    while not state.is_terminal():
        actions = []
        for player in range(game.num_players()):
            legal_actions = state.legal_actions(player)
            action = policies[player](state, legal_actions)
            actions.append(action)
        
        state.apply_actions(actions)
        rewards = state.rewards()
        
        for player, reward in enumerate(rewards):
            episode_rewards[player].append(reward)
    
    return episode_rewards, state.returns()


def random_policy(state, legal_actions):
    return np.random.choice(legal_actions)


def always_cooperate(state, legal_actions):
    return 0


def always_defect(state, legal_actions):
    return 1


def tit_for_tat(state, legal_actions):
    history = state.full_history()
    if not history:
        return 0
    
    opponent_actions = [h for h in history if h.player != state.current_player()]
    if not opponent_actions:
        return 0
    
    last_opponent_action = opponent_actions[-1].action
    return last_opponent_action


def main():
    print("=" * 70)
    print("Parameterized Social Dilemma - MARL Examples")
    print("=" * 70)
    
    print("\n1. Basic 3-player game with deterministic rewards")
    print("-" * 70)
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 3,
        "max_game_length": 5
    })
    
    policies = [always_cooperate, always_defect, random_policy]
    episode_rewards, final_returns = play_game(game, policies)
    
    for player in range(game.num_players()):
        print(f"Player {player} - Total return: {final_returns[player]:.2f}")
    
    print("\n2. 5-player game with stochastic rewards")
    print("-" * 70)
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 5,
        "max_game_length": 10,
        "reward_noise_std": 0.1
    })
    
    policies = [random_policy] * 5
    num_episodes = 100
    avg_returns = np.zeros(5)
    
    for episode in range(num_episodes):
        _, returns = play_game(game, policies)
        avg_returns += returns
    
    avg_returns /= num_episodes
    print(f"Average returns over {num_episodes} episodes:")
    for player in range(game.num_players()):
        print(f"  Player {player}: {avg_returns[player]:.2f}")
    
    print("\n3. 4-player game with dynamic payoffs")
    print("-" * 70)
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 4,
        "max_game_length": 8,
        "dynamic_payoffs": True,
        "payoff_change_prob": 0.1
    })
    
    policies = [always_cooperate, always_defect, random_policy, random_policy]
    num_episodes = 50
    
    all_returns = []
    for episode in range(num_episodes):
        _, returns = play_game(game, policies)
        all_returns.append(returns)
    
    all_returns = np.array(all_returns)
    print(f"Statistics over {num_episodes} episodes:")
    for player in range(game.num_players()):
        mean = np.mean(all_returns[:, player])
        std = np.std(all_returns[:, player])
        print(f"  Player {player}: mean={mean:.2f}, std={std:.2f}")
    
    print("\n4. Custom payoff matrix (2-player Prisoner's Dilemma)")
    print("-" * 70)
    custom_payoff = np.zeros((2, 2, 2))
    custom_payoff[0, 0] = [3, 3]
    custom_payoff[0, 1] = [0, 5]
    custom_payoff[1, 0] = [5, 0]
    custom_payoff[1, 1] = [1, 1]
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "payoff_matrix": str(custom_payoff.tolist()),
        "max_game_length": 10
    })
    
    scenarios = [
        ("Both Cooperate", [always_cooperate, always_cooperate]),
        ("Both Defect", [always_defect, always_defect]),
        ("Mixed Strategy", [always_cooperate, always_defect]),
    ]
    
    for scenario_name, policies in scenarios:
        _, returns = play_game(game, policies)
        print(f"{scenario_name}:")
        print(f"  Player 0: {returns[0]:.2f}, Player 1: {returns[1]:.2f}")
    
    print("\n5. Varying number of players - scalability test")
    print("-" * 70)
    for num_players in [2, 3, 5, 8]:
        game = pyspiel.load_game("python_param_social_dilemma", {
            "num_players": num_players,
            "max_game_length": 5
        })
        
        policies = [random_policy] * num_players
        _, returns = play_game(game, policies)
        avg_return = np.mean(returns)
        
        print(f"{num_players} players - Average return: {avg_return:.2f}")
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
