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

# Lint as: python3
"""Tests for open_spiel.python.algorithms.alpha_zero.evaluator."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import model as model_lib
import pyspiel


def build_model(game):
  return model_lib.Model.build_model(
      "mlp", game.observation_tensor_shape(), game.num_distinct_actions(),
      nn_width=64, nn_depth=2, weight_decay=1e-4, learning_rate=0.01, path=None)


class EvaluatorTest(absltest.TestCase):

  def test_evaluator_caching(self):
    game = pyspiel.load_game("tic_tac_toe")
    model = build_model(game)
    evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)

    state = game.new_initial_state()
    obs = state.observation_tensor()
    act_mask = state.legal_actions_mask()
    action = state.legal_actions()[0]
    policy = np.zeros(len(act_mask), dtype=float)
    policy[action] = 1
    train_inputs = [model_lib.TrainInput(obs, act_mask, policy, value=1)]

    value = evaluator.evaluate(state)
    self.assertEqual(value[0], -value[1])
    value = value[0]

    value2 = evaluator.evaluate(state)[0]
    self.assertEqual(value, value2)

    prior = evaluator.prior(state)
    prior2 = evaluator.prior(state)
    np.testing.assert_array_equal(prior, prior2)

    info = evaluator.cache_info()
    self.assertEqual(info.misses, 1)
    self.assertEqual(info.hits, 3)

    for _ in range(20):
      model.update(train_inputs)

    # Still equal due to not clearing the cache
    value3 = evaluator.evaluate(state)[0]
    self.assertEqual(value, value3)

    info = evaluator.cache_info()
    self.assertEqual(info.misses, 1)
    self.assertEqual(info.hits, 4)

    evaluator.clear_cache()

    info = evaluator.cache_info()
    self.assertEqual(info.misses, 0)
    self.assertEqual(info.hits, 0)

    # Now they differ from before
    value4 = evaluator.evaluate(state)[0]
    value5 = evaluator.evaluate(state)[0]
    self.assertNotEqual(value, value4)
    self.assertEqual(value4, value5)

    info = evaluator.cache_info()
    self.assertEqual(info.misses, 1)
    self.assertEqual(info.hits, 1)

    value6 = evaluator.evaluate(game.new_initial_state())[0]
    self.assertEqual(value4, value6)

    info = evaluator.cache_info()
    self.assertEqual(info.misses, 1)
    self.assertEqual(info.hits, 2)

  def test_works_with_mcts(self):
    game = pyspiel.load_game("tic_tac_toe")
    model = build_model(game)
    evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    bot = mcts.MCTSBot(
        game, 1., 20, evaluator, solve=False, dirichlet_noise=(0.25, 1.))
    root = bot.mcts_search(game.new_initial_state())
    self.assertEqual(root.explore_count, 20)


if __name__ == "__main__":
  absltest.main()
