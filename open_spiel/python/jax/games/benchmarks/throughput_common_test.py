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

"""The throughput harness: flag parsing, result schema and the CPU guard."""

from absl import flags
from absl.testing import absltest

from open_spiel.python.jax.games import no_limit_holdem
from open_spiel.python.jax.games.benchmarks import no_limit_holdem_throughput
from open_spiel.python.jax.games.benchmarks import throughput_common

throughput_common.define_flags()
flags.FLAGS.mark_as_parsed()


class ThroughputCommonTest(absltest.TestCase):
  """The shared throughput benchmark code."""

  def test_parse_batch_sizes(self):
    """Batch sizes parse and validate."""
    self.assertEqual(
        throughput_common.parse_batch_sizes(["256", "1024"]), [256, 1024]
    )
    with self.assertRaisesRegex(ValueError, "positive"):
      throughput_common.parse_batch_sizes(["0"])

  def test_no_limit_holdem_smoke_result_schema(self):
    """A no-limit hold'em smoke run has the result schema."""
    spec = throughput_common.BenchmarkSpec(
        game_name="no_limit_holdem",
        jax_game_factory=no_limit_holdem.NoLimitHoldem,
        pyspiel_game_string=no_limit_holdem.UNIVERSAL_POKER_GAME_STRING,
        terminal_fn=lambda state: state.is_terminal,
        current_player_fn=lambda state: state.current_player,
    )
    result = throughput_common.run_benchmark(
        spec,
        batch_sizes=[2],
        steps=2,
        num_runs=1,
        compare_cpp=False,
        allow_cpu=True,
        cpp_games=0,
        include_returns=False,
        include_training_tensors=True,
    )
    self.assertEqual(result["schema_version"], throughput_common.SCHEMA_VERSION)
    self.assertEqual(result["game"], "no_limit_holdem")
    self.assertIsNone(result["cpp_baseline"])
    self.assertLen(result["results"], 1)
    row = result["results"][0]
    self.assertEqual(row["batch_size"], 2)
    self.assertEqual(row["steps"], 2)
    self.assertEqual(row["num_runs"], 1)
    self.assertFalse(row["include_returns"])
    self.assertFalse(result["config"]["include_returns"])
    self.assertTrue(row["include_training_tensors"])
    self.assertTrue(result["config"]["include_training_tensors"])
    self.assertGreater(row["steps_per_second"], 0.0)
    self.assertGreater(row["games_per_second"], 0.0)
    self.assertGreaterEqual(row["completion_rate"], 0.0)
    self.assertLessEqual(row["completion_rate"], 1.0)

  def test_no_limit_holdem_throughput_spec_uses_dense_mask_sampling(self):
    """The no-limit hold'em spec samples from the dense mask."""
    spec = no_limit_holdem_throughput.benchmark_spec(use_showdown_cache=True)
    self.assertEqual(spec.game_name, "no_limit_holdem_cache")

  def test_include_returns_requires_completed_batch(self):
    """include_returns requires every rollout to finish."""
    spec = throughput_common.BenchmarkSpec(
        game_name="no_limit_holdem",
        jax_game_factory=no_limit_holdem.NoLimitHoldem,
        pyspiel_game_string=no_limit_holdem.UNIVERSAL_POKER_GAME_STRING,
        terminal_fn=lambda state: state.is_terminal,
        current_player_fn=lambda state: state.current_player,
    )
    with self.assertRaisesRegex(RuntimeError, "requires every rollout"):
      throughput_common.run_benchmark(
          spec,
          batch_sizes=[2],
          steps=2,
          num_runs=1,
          compare_cpp=False,
          allow_cpu=True,
          cpp_games=0,
          include_returns=True,
          include_training_tensors=False,
      )


if __name__ == "__main__":
  absltest.main()
