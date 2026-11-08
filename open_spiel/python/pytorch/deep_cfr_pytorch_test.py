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

from absl import app
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from open_spiel.python import policy
import pyspiel
from open_spiel.python.pytorch import deep_cfr

SEED = 24984617


def _poisoned_like(leaf: np.ndarray, capacity: int) -> np.ndarray:
  """Returns a `capacity`-row array shaped like `leaf`, filled with garbage."""
  shape = (capacity, *leaf.shape)
  if np.issubdtype(leaf.dtype, np.floating):
    return np.full(shape, np.nan, dtype=leaf.dtype)
  return np.zeros(shape, dtype=leaf.dtype)


class _NaNInitializedBufferSolver(deep_cfr.DeepCFRSolver):
  """Creates replay buffers whose not-yet-written rows are NaN.

  The real buffers are allocated uninitialized, so those rows may hold
  arbitrary values; NaN makes any read of them deterministic and visible.
  """

  def _get_buffer_init(
      self,
      capacity: int,
      data: deep_cfr.AdvantageMemory | deep_cfr.StrategyMemory,
  ) -> deep_cfr.ReservoirBuffer:
    storage = type(data)(*(_poisoned_like(leaf, capacity) for leaf in data))
    return deep_cfr.ReservoirBuffer(np.array(capacity), storage)


def _strategy_memory(value: int) -> deep_cfr.StrategyMemory:
  return deep_cfr.StrategyMemory(
      info_state=np.full(2, value, dtype=np.float32),
      iteration=np.array([value]),
      strategy_action_probs=np.full(3, value, dtype=np.float32),
  )


class DeepCFRPyTorchTest(parameterized.TestCase):

  def test_reservoir_buffer_experience_holds_only_appended_rows(self):
    capacity = 5
    buffer = deep_cfr.ReservoirBuffer.init(capacity, _strategy_memory(0))
    self.assertEmpty(buffer.experience.info_state)

    for value in range(3):
      buffer.append(_strategy_memory(value))
    buffer.shuffle()
    for leaf in buffer.experience:
      self.assertLen(leaf, 3)
    for leaf in buffer.experience:
      self.assertCountEqual(leaf[:, 0].tolist(), [0, 1, 2])

    num_appends = 20
    for value in range(3, num_appends):
      buffer.append(_strategy_memory(value))
    for leaf in buffer.experience:
      self.assertLen(leaf, capacity)
      self.assertContainsSubset(leaf[:, 0].tolist(), range(num_appends))

    buffer.clear()
    self.assertEmpty(buffer.experience.info_state)

  def test_reservoir_buffer_shuffle_keeps_rows_aligned(self):
    buffer = deep_cfr.ReservoirBuffer.init(100, _strategy_memory(0))
    for value in range(50):
      buffer.append(_strategy_memory(value))
    for _ in range(3):
      buffer.shuffle()
    experience = buffer.experience
    row_ids = experience.iteration[:, 0]
    self.assertCountEqual(row_ids.tolist(), range(50))
    self.assertFalse(np.array_equal(row_ids, np.arange(50)))
    np.testing.assert_array_equal(experience.info_state[:, 0], row_ids)
    np.testing.assert_array_equal(
        experience.strategy_action_probs[:, 0], row_ids)

  def test_full_batch_training_ignores_unfilled_buffer_rows(self):
    game = pyspiel.load_game('kuhn_poker')
    capacity = 1000
    deep_cfr_solver = _NaNInitializedBufferSolver(
        game,
        policy_network_layers=(8, 4),
        advantage_network_layers=(4, 2),
        num_iterations=2,
        num_traversals=2,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=capacity,
        seed=SEED,
    )
    _, advantage_losses, policy_loss = deep_cfr_solver.solve()
    self.assertLess(len(deep_cfr_solver.strategy_buffer), capacity)
    self.assertTrue(np.isfinite(policy_loss))
    for losses in advantage_losses.values():
      self.assertTrue(np.all(np.isfinite(losses)))
    tabular_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities)
    self.assertTrue(
        np.all(np.isfinite(tabular_policy.action_probability_array)))
    conv = pyspiel.nash_conv(
        game, policy.python_policy_to_pyspiel_policy(tabular_policy))
    self.assertTrue(np.isfinite(conv))

  @parameterized.parameters('leduc_poker', 'kuhn_poker', 'liars_dice')
  def test_deep_cfr_runs(self, game_name):
    game = pyspiel.load_game(game_name)
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(8, 4),
        advantage_network_layers=(4, 2),
        num_iterations=2,
        num_traversals=2,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1e7,
        seed=SEED,
    )
    deep_cfr_solver.solve()

  def test_matching_pennies_3p(self):
    game = pyspiel.load_game_as_turn_based('matching_pennies_3p')
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(16, 8),
        advantage_network_layers=(32, 16),
        num_iterations=2,
        num_traversals=2,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=int(1e7),
    )
    deep_cfr_solver.solve()
    conv = pyspiel.nash_conv(
        game,
        policy.python_policy_to_pyspiel_policy(
            policy.tabular_policy_from_callable(
                game, deep_cfr_solver.action_probabilities)))
    logging.info('Deep CFR in Matching Pennies 3p. NashConv: %.2f', conv)


def main(_):
  torch.manual_seed(SEED)
  absltest.main()


if __name__ == '__main__':
  # Necessary to run main via app.run for internal tests.
  app.run(main)
