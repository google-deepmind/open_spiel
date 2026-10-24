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

r"""Random-play throughput of no-limit hold'em.

Example:
  python -m open_spiel.python.jax.games.benchmarks.no_limit_holdem_throughput \
      --batch_sizes=256,1024,4096,16384 --steps=200 --num_runs=5
"""

from absl import app
from absl import flags

from open_spiel.python.jax.games import no_limit_holdem
from open_spiel.python.jax.games.benchmarks import throughput_common


def benchmark_spec(
    use_showdown_cache: bool,
    track_information_state_history: bool = False,
) -> throughput_common.BenchmarkSpec:
  """Returns the no-limit hold'em benchmark spec.

  Args:
    use_showdown_cache: Whether to use the showdown lookup table.
    track_information_state_history: Whether to track the action history.

  Returns:
    The BenchmarkSpec.
  """
  suffix = "cache" if use_showdown_cache else "no_cache"
  return throughput_common.BenchmarkSpec(
      game_name=f"no_limit_holdem_{suffix}",
      jax_game_factory=lambda: no_limit_holdem.NoLimitHoldem(
          use_showdown_cache=use_showdown_cache,
          track_information_state_history=track_information_state_history,
      ),
      pyspiel_game_string=no_limit_holdem.UNIVERSAL_POKER_GAME_STRING,
      terminal_fn=lambda state: state.is_terminal,
      current_player_fn=lambda state: state.current_player,
  )


def define_flags() -> None:
  """Defines the benchmark flags."""
  throughput_common.define_flags()
  fv = flags.FLAGS
  if "use_showdown_cache" not in fv:
    flags.DEFINE_boolean(
        "use_showdown_cache",
        False,
        "Use the precomputed five-card score LUT for terminal showdowns.",
    )
  if "track_information_state_history" not in fv:
    flags.DEFINE_boolean(
        "track_information_state_history",
        False,
        "Track OpenSpiel-compatible information-state action history during "
        "the throughput benchmark. The default benchmark path keeps "
        "observations/legal masks/returns but omits information states for "
        "training-style throughput.",
    )


def main(argv):
  """Runs the benchmark and prints or writes the result.

  Args:
    argv: Unused.
  """
  del argv
  fv = flags.FLAGS
  result = throughput_common.run_benchmark(
      benchmark_spec(
          fv.use_showdown_cache,
          track_information_state_history=fv.track_information_state_history,
      ),
      batch_sizes=throughput_common.parse_batch_sizes(fv.batch_sizes),
      steps=fv.steps,
      num_runs=fv.num_runs,
      compare_cpp=fv.compare_cpp,
      allow_cpu=fv.allow_cpu,
      cpp_games=fv.cpp_games,
      cpp_max_steps_per_game=fv.cpp_max_steps_per_game,
      include_returns=fv.include_returns,
      include_training_tensors=fv.include_training_tensors,
  )
  throughput_common.print_report(result)
  throughput_common.write_json(result, fv.output_json)


if __name__ == "__main__":
  define_flags()
  app.run(main)
