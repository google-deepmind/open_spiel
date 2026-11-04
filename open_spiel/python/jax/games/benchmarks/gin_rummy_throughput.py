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

r"""Random-play throughput of gin rummy.

Example:
  python -m open_spiel.python.jax.games.benchmarks.gin_rummy_throughput \
      --batch_sizes=256,1024,4096,16384 --steps=200 --num_runs=5
"""

from absl import app
from absl import flags

from open_spiel.python.jax.games import gin_rummy
from open_spiel.python.jax.games.benchmarks import throughput_common


def benchmark_spec() -> throughput_common.BenchmarkSpec:
  """Returns the gin rummy benchmark spec.

  Returns:
    The BenchmarkSpec.
  """
  game = gin_rummy.GinRummy()
  return throughput_common.BenchmarkSpec(
      game_name="gin_rummy",
      jax_game_factory=gin_rummy.GinRummy,
      pyspiel_game_string="gin_rummy",
      terminal_fn=lambda state: state["done"],
      current_player_fn=game.current_player,
  )


def define_flags() -> None:
  """Defines the benchmark flags."""
  throughput_common.define_flags()


def main(argv):
  """Runs the benchmark and prints or writes the result.

  Args:
    argv: Unused.
  """
  del argv
  fv = flags.FLAGS
  result = throughput_common.run_benchmark(
      benchmark_spec(),
      batch_sizes=throughput_common.parse_batch_sizes(fv.batch_sizes),
      steps=fv.steps,
      num_runs=fv.num_runs,
      compare_cpp=fv.compare_cpp,
      allow_cpu=fv.allow_cpu,
      cpp_games=fv.cpp_games,
      cpp_max_steps_per_game=fv.cpp_max_steps_per_game,
      include_returns=fv.include_returns,
  )
  throughput_common.print_report(result)
  throughput_common.write_json(result, fv.output_json)


if __name__ == "__main__":
  define_flags()
  app.run(main)
