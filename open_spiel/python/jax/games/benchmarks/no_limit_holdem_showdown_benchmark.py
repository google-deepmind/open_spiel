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

r"""Microbenchmark for no-limit hold'em 7-card hand scoring.

Example:
  python -m \
    open_spiel.python.jax.games.benchmarks.no_limit_holdem_showdown_benchmark \
    --batch_size=1048576 --num_runs=10
"""

import json
import os
import statistics
import time
from typing import Any

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import no_limit_holdem
from open_spiel.python.jax.games.benchmarks import throughput_common


def define_flags() -> None:
  """Registers benchmark flags once."""
  fv = flags.FLAGS
  if "batch_size" not in fv:
    flags.DEFINE_integer(
        "batch_size", 262144, "Number of random 7-card hands per timed run."
    )
  if "num_runs" not in fv:
    flags.DEFINE_integer("num_runs", 10, "Post-JIT timed scoring runs.")
  if "allow_cpu" not in fv:
    flags.DEFINE_boolean(
        "allow_cpu",
        False,
        "Permit CPU execution. Canonical benchmark numbers should use GPU.",
    )
  if "output_json" not in fv:
    flags.DEFINE_string(
        "output_json", None, "Optional path to write benchmark metrics as JSON."
    )
  if "seed" not in fv:
    flags.DEFINE_integer("seed", 0, "PRNG seed for input hands.")
  if "evaluator" not in fv:
    flags.DEFINE_enum(
        "evaluator",
        "cache",
        ["cache", "no_cache"],
        "Five-card evaluator to benchmark through the 7-card path.",
    )


def sample_random_hands(seed: int, batch_size: int) -> jax.Array:
  """Samples `batch_size` unique-card 7-card hands.

  Args:
    seed: The seed.
    batch_size: The number of hands.

  Returns:
    i32[batch_size, 7].

  Raises:
    ValueError: `batch_size` is not positive.
  """
  if batch_size <= 0:
    raise ValueError(f"--batch_size must be positive, got {batch_size}")
  keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
  hands = jax.vmap(
      lambda key: jax.random.permutation(key, no_limit_holdem.NUM_CARDS)[:7]
  )(keys)
  return hands.astype(jnp.int32)


def run_benchmark(
    *,
    seed: int,
    batch_size: int,
    num_runs: int,
    allow_cpu: bool,
    evaluator: str,
) -> dict[str, Any]:
  """Runs the scoring microbenchmark.

  Args:
    seed: The seed.
    batch_size: The number of hands.
    num_runs: The timed runs.
    allow_cpu: Whether a CPU may be used.
    evaluator: "cache" or "no_cache".

  Returns:
    JSON-ready metrics.

  Raises:
    ValueError: `num_runs` is not positive or `evaluator` is unknown.
  """
  if num_runs <= 0:
    raise ValueError(f"--num_runs must be positive, got {num_runs}")
  if evaluator not in ("cache", "no_cache"):
    raise ValueError(f"unknown evaluator: {evaluator!r}")
  device = throughput_common.ensure_gpu_or_allowed(allow_cpu)
  hands = jax.device_put(sample_random_hands(seed, batch_size), device)
  jax.block_until_ready(hands)
  score_one = (
      no_limit_holdem.best_seven_card_score
      if evaluator == "cache"
      else no_limit_holdem.best_seven_card_score_uncached
  )

  @jax.jit
  def score_batch(batch_hands):
    """Scores a batch of hands.

    Args:
      batch_hands: i32[B, 7].

    Returns:
      The scores.
    """
    return jax.vmap(score_one)(batch_hands)

  t0 = time.perf_counter()
  scores = score_batch(hands)
  jax.block_until_ready(scores)
  jit_compile_seconds = time.perf_counter() - t0

  durations = []
  for _ in range(num_runs):
    t0 = time.perf_counter()
    scores = score_batch(hands)
    jax.block_until_ready(scores)
    durations.append(time.perf_counter() - t0)

  steady_seconds = float(statistics.median(durations))
  return {
      "schema_version": throughput_common.SCHEMA_VERSION,
      "benchmark": "no_limit_holdem_showdown",
      "jax_device": str(device),
      "jax_platform": device.platform,
      "config": {
          "seed": seed,
          "batch_size": batch_size,
          "num_runs": num_runs,
          "evaluator": evaluator,
      },
      "jit_compile_seconds": jit_compile_seconds,
      "steady_seconds_median": steady_seconds,
      "hands_per_second": batch_size / steady_seconds,
      "score_checksum": int(np.asarray(jnp.sum(scores))),
  }


def print_report(result: dict[str, Any]) -> None:
  """Prints a compact benchmark report.

  Args:
    result: The run_benchmark result.
  """
  config = result["config"]
  print("=" * 80)
  print("JAX no-limit hold'em showdown benchmark")
  print("=" * 80)
  print(f"device: {result['jax_device']}")
  print(
      "config: "
      f"batch_size={config['batch_size']}, num_runs={config['num_runs']}, "
      f"seed={config['seed']}, evaluator={config['evaluator']}"
  )
  print(f"jit_compile_seconds: {result['jit_compile_seconds']:.6f}")
  print(f"steady_seconds_median: {result['steady_seconds_median']:.6f}")
  print(f"hands_per_second: {result['hands_per_second']:,.0f}")
  print(f"score_checksum: {result['score_checksum']}")


def main(argv):
  """Runs the benchmark and prints or writes the result.

  Args:
    argv: Unused.
  """
  del argv
  fv = flags.FLAGS
  result = run_benchmark(
      seed=fv.seed,
      batch_size=fv.batch_size,
      num_runs=fv.num_runs,
      allow_cpu=fv.allow_cpu,
      evaluator=fv.evaluator,
  )
  print_report(result)
  if fv.output_json:
    os.makedirs(os.path.dirname(fv.output_json) or ".", exist_ok=True)
    with open(fv.output_json, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2, sort_keys=True)
      f.write("\n")


if __name__ == "__main__":
  define_flags()
  app.run(main)
