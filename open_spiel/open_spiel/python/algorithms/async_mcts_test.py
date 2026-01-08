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

import math
import random

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import async_mcts
from open_spiel.python.algorithms import evaluate_bots
import pyspiel

UCT_C = math.sqrt(2)


def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  raise ValueError("invalid action string: {}".format(action_str))


def search_tic_tac_toe_state(initial_actions):
  game = pyspiel.load_game("tic_tac_toe")
  state = game.new_initial_state()
  for action_str in initial_actions.split(" "):
    state.apply_action(_get_action(state, action_str))
  rng = np.random.RandomState(42)
  bot = async_mcts.MCTSBot(
      game,
      UCT_C,
      max_simulations=10000,
      solve=True,
      random_state=rng,
      evaluator=async_mcts.RandomRolloutEvaluator(random_state=rng),
  )
  return bot.mcts_search(state), state


def make_node(action, player=0, prior=1, **kwargs):
  node = async_mcts.SearchNode(action, player, prior)
  for k, v in kwargs.items():
    setattr(node, k, v)
  return node


class MctsBotTest(absltest.TestCase):

  def test_can_play_tic_tac_toe(self):
    game = pyspiel.load_game("tic_tac_toe")
    max_simulations = 100
    evaluator = async_mcts.RandomRolloutEvaluator()
    bots = [
        async_mcts.MCTSBot(game, UCT_C, max_simulations, evaluator),
        async_mcts.MCTSBot(game, UCT_C, max_simulations, evaluator),
    ]
    v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    self.assertEqual(v[0] + v[1], 0)

  def test_can_play_both_sides(self):
    game = pyspiel.load_game("tic_tac_toe")
    bot = async_mcts.MCTSBot(
        game,
        UCT_C,
        max_simulations=100,
        evaluator=async_mcts.RandomRolloutEvaluator(),
    )
    bots = [bot, bot]
    v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    self.assertEqual(v[0] + v[1], 0)

  def test_can_play_single_player(self):
    game = pyspiel.load_game("catch")
    max_simulations = 100
    evaluator = async_mcts.RandomRolloutEvaluator()
    bots = [async_mcts.MCTSBot(game, UCT_C, max_simulations, evaluator)]
    v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    self.assertGreater(v[0], 0)

  def test_throws_on_simultaneous_game(self):
    game = pyspiel.load_game("matrix_mp")
    evaluator = async_mcts.RandomRolloutEvaluator()
    with self.assertRaises(ValueError):
      async_mcts.MCTSBot(game, UCT_C, max_simulations=100, evaluator=evaluator)

  def test_can_play_three_player_stochastic_games(self):
    game = pyspiel.load_game("pig(players=3,winscore=20,horizon=30)")
    max_simulations = 100
    evaluator = async_mcts.RandomRolloutEvaluator()
    bots = [
        async_mcts.MCTSBot(game, UCT_C, max_simulations, evaluator),
        async_mcts.MCTSBot(game, UCT_C, max_simulations, evaluator),
        async_mcts.MCTSBot(game, UCT_C, max_simulations, evaluator),
    ]
    v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    self.assertEqual(sum(v), 0)

  def assertBestChild(self, choice, children):
    # If this causes flakiness, the key in `SearchNode.best_child` is bad.
    random.shuffle(children)
    root = make_node(-1, children=children)
    self.assertEqual(root.best_child().action, choice)

  def test_choose_most_visited_when_not_solved(self):
    self.assertBestChild(
        0,
        [
            make_node(0, explore_count=50, total_reward=30),
            make_node(1, explore_count=40, total_reward=40),
        ],
    )

  def test_choose_win_over_most_visited(self):
    self.assertBestChild(
        1,
        [
            make_node(0, explore_count=50, total_reward=30),
            make_node(1, explore_count=40, total_reward=40, outcome=[1]),
        ],
    )

  def test_choose_best_over_good(self):
    self.assertBestChild(
        1,
        [
            make_node(0, explore_count=50, total_reward=30, outcome=[0.5]),
            make_node(1, explore_count=40, total_reward=40, outcome=[0.8]),
        ],
    )

  def test_choose_bad_over_worst(self):
    self.assertBestChild(
        0,
        [
            make_node(0, explore_count=50, total_reward=30, outcome=[-0.5]),
            make_node(1, explore_count=40, total_reward=40, outcome=[-0.8]),
        ],
    )

  def test_choose_positive_reward_over_promising(self):
    self.assertBestChild(
        1,
        [
            make_node(0, explore_count=50, total_reward=40),  # more promising
            make_node(
                1, explore_count=10, total_reward=1, outcome=[0.1]
            ),  # solved
        ],
    )

  def test_choose_most_visited_over_loss(self):
    self.assertBestChild(
        0,
        [
            make_node(0, explore_count=50, total_reward=30),
            make_node(1, explore_count=40, total_reward=40, outcome=[-1]),
        ],
    )

  def test_choose_most_visited_over_draw(self):
    self.assertBestChild(
        0,
        [
            make_node(0, explore_count=50, total_reward=30),
            make_node(1, explore_count=40, total_reward=40, outcome=[0]),
        ],
    )

  def test_choose_uncertainty_over_most_visited_loss(self):
    self.assertBestChild(
        1,
        [
            make_node(0, explore_count=50, total_reward=30, outcome=[-1]),
            make_node(1, explore_count=40, total_reward=40),
        ],
    )

  def test_choose_slowest_loss(self):
    self.assertBestChild(
        1,
        [
            make_node(0, explore_count=50, total_reward=10, outcome=[-1]),
            make_node(1, explore_count=60, total_reward=15, outcome=[-1]),
        ],
    )


if __name__ == "__main__":
  absltest.main()
