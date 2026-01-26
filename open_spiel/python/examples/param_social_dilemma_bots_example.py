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

"""Axelrod tournament example for param_social_dilemma.

This demonstrates bot-vs-bot tournaments using Axelrod-style strategies.
"""

import numpy as np
from open_spiel.python.games import param_social_dilemma_bots
import pyspiel


def play_game_with_bots(game, bots):
    state = game.new_initial_state()
    
    for bot in bots:
        bot.restart_at(state)
    
    while not state.is_terminal():
        actions = [bot.step(state) for bot in bots]
        state.apply_actions(actions)
    
    return state.returns()


def run_tournament(num_rounds=100):
    print("=" * 70)
    print("Axelrod-Style Tournament - Parameterized Social Dilemma")
    print("=" * 70)
    
    bot_classes = [
        ("AlwaysCooperate", param_social_dilemma_bots.AlwaysCooperateBot),
        ("AlwaysDefect", param_social_dilemma_bots.AlwaysDefectBot),
        ("TitForTat", param_social_dilemma_bots.TitForTatBot),
        ("GrimTrigger", param_social_dilemma_bots.GrimTriggerBot),
        ("Pavlov", param_social_dilemma_bots.PavlovBot),
        ("TitForTwoTats", param_social_dilemma_bots.TitForTwoTatsBot),
        ("Gradual", param_social_dilemma_bots.GradualBot),
    ]
    
    num_bots = len(bot_classes)
    scores = np.zeros((num_bots, num_bots))
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 2,
        "max_game_length": 10
    })
    
    print(f"\nRunning {num_rounds} rounds for each pairing...")
    print("-" * 70)
    
    for i, (name1, bot_class1) in enumerate(bot_classes):
        for j, (name2, bot_class2) in enumerate(bot_classes):
            total_score = 0
            for _ in range(num_rounds):
                bot1 = bot_class1(player_id=0, num_players=2) if bot_class1.__name__ != "AlwaysCooperateBot" and bot_class1.__name__ != "AlwaysDefectBot" else bot_class1(player_id=0)
                bot2 = bot_class2(player_id=1, num_players=2) if bot_class2.__name__ != "AlwaysCooperateBot" and bot_class2.__name__ != "AlwaysDefectBot" else bot_class2(player_id=1)
                
                returns = play_game_with_bots(game, [bot1, bot2])
                total_score += returns[0]
            
            scores[i, j] = total_score / num_rounds
    
    print("\nTournament Results:")
    print("-" * 70)
    print(f"{'Strategy':<18}", end="")
    for name, _ in bot_classes:
        print(f"{name[:12]:<14}", end="")
    print()
    print("-" * 70)
    
    for i, (name, _) in enumerate(bot_classes):
        print(f"{name:<18}", end="")
        for j in range(num_bots):
            print(f"{scores[i, j]:>6.2f}        ", end="")
        print()
    
    avg_scores = np.mean(scores, axis=1)
    print("-" * 70)
    print("\nAverage Scores:")
    print("-" * 70)
    rankings = sorted(enumerate(avg_scores), key=lambda x: x[1], reverse=True)
    
    for rank, (idx, score) in enumerate(rankings, 1):
        name = bot_classes[idx][0]
        print(f"{rank}. {name:<18} {score:>6.2f}")
    
    print("\n" + "=" * 70)


def demonstrate_n_player_tournament():
    print("\n\n" + "=" * 70)
    print("N-Player Tournament (3 players)")
    print("=" * 70)
    
    game = pyspiel.load_game("python_param_social_dilemma", {
        "num_players": 3,
        "max_game_length": 10
    })
    
    scenarios = [
        ("All Cooperate", [
            param_social_dilemma_bots.AlwaysCooperateBot(0),
            param_social_dilemma_bots.AlwaysCooperateBot(1),
            param_social_dilemma_bots.AlwaysCooperateBot(2)
        ]),
        ("All Defect", [
            param_social_dilemma_bots.AlwaysDefectBot(0),
            param_social_dilemma_bots.AlwaysDefectBot(1),
            param_social_dilemma_bots.AlwaysDefectBot(2)
        ]),
        ("Two Cooperators vs One Defector", [
            param_social_dilemma_bots.AlwaysCooperateBot(0),
            param_social_dilemma_bots.AlwaysCooperateBot(1),
            param_social_dilemma_bots.AlwaysDefectBot(2)
        ]),
        ("TitForTat vs AlwaysCooperate vs AlwaysDefect", [
            param_social_dilemma_bots.TitForTatBot(0, 3),
            param_social_dilemma_bots.AlwaysCooperateBot(1),
            param_social_dilemma_bots.AlwaysDefectBot(2)
        ]),
        ("Mixed Strategies", [
            param_social_dilemma_bots.TitForTatBot(0, 3),
            param_social_dilemma_bots.GrimTriggerBot(1, 3),
            param_social_dilemma_bots.PavlovBot(2, 3)
        ]),
    ]
    
    print("\nScenario Outcomes:")
    print("-" * 70)
    
    for scenario_name, bots in scenarios:
        returns = play_game_with_bots(game, bots)
        print(f"\n{scenario_name}:")
        for i, ret in enumerate(returns):
            bot_name = type(bots[i]).__name__.replace("Bot", "")
            print(f"  Player {i} ({bot_name:<15}): {ret:>6.2f}")
    
    print("\n" + "=" * 70)


def main():
    run_tournament(num_rounds=100)
    demonstrate_n_player_tournament()


if __name__ == "__main__":
    main()
