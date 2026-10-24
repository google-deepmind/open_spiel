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

"""The random-play throughput harness shared by the game benchmarks.

A benchmark supplies a `BenchmarkSpec`; `run_benchmark` times a jitted,
vmapped scan of uniformly random legal moves for each batch size, and
optionally a single-thread pyspiel baseline.

  result = run_benchmark(spec, batch_sizes=[1024], steps=200, num_runs=5,
                         compare_cpp=True, allow_cpu=False, cpp_games=128,
                         cpp_max_steps_per_game=1000, include_returns=False)
  print_report(result)
"""

from collections.abc import Callable, Sequence
import dataclasses
import json
import os
import time
from typing import Any

from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib

SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZES = "256,1024,4096,16384"
DEFAULT_STEPS = 200
DEFAULT_NUM_RUNS = 5
DEFAULT_CPP_GAMES = 128
DEFAULT_CPP_MAX_STEPS_PER_GAME = 1000


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
  """Game-specific inputs for the shared random-play benchmark.

  Attributes:
    current_player_fn: Maps a state to its current player.
    game_name: The name the results are reported under.
    jax_game_factory: Builds the JAX game.
    pyspiel_game_string: The pyspiel game string of the C++ baseline.
    terminal_fn: Maps a state to its bool terminal flag.
  """

  game_name: str
  jax_game_factory: Callable[[], game_lib.Game]
  pyspiel_game_string: str
  terminal_fn: Callable[[Any], jax.Array]
  current_player_fn: Callable[[Any], jax.Array]


def define_flags() -> None:
  """Registers the shared benchmark flags once."""
  fv = flags.FLAGS
  if "batch_sizes" not in fv:
    flags.DEFINE_list(
        "batch_sizes",
        DEFAULT_BATCH_SIZES.split(","),
        "Comma-separated batch sizes to benchmark.",
    )
  if "steps" not in fv:
    flags.DEFINE_integer("steps", DEFAULT_STEPS, "Steps per rollout.")
  if "num_runs" not in fv:
    flags.DEFINE_integer(
        "num_runs", DEFAULT_NUM_RUNS, "Timed post-JIT runs per batch size."
    )
  if "compare_cpp" not in fv:
    flags.DEFINE_boolean(
        "compare_cpp",
        True,
        "Also run a sequential pyspiel C++ random-play baseline.",
    )
  if "allow_cpu" not in fv:
    flags.DEFINE_boolean(
        "allow_cpu",
        False,
        "Permit CPU execution. By default benchmarks require a GPU or TPU "
        "device.",
    )
  if "cpp_games" not in fv:
    flags.DEFINE_integer(
        "cpp_games",
        DEFAULT_CPP_GAMES,
        "Sequential pyspiel games for the C++ comparison anchor.",
    )
  if "cpp_max_steps_per_game" not in fv:
    flags.DEFINE_integer(
        "cpp_max_steps_per_game",
        DEFAULT_CPP_MAX_STEPS_PER_GAME,
        "Maximum pyspiel steps per game before treating the random game as "
        "truncated. This keeps no-limit poker random betting anchors finite.",
    )
  if "output_json" not in fv:
    flags.DEFINE_string(
        "output_json",
        None,
        "Optional path to write the full benchmark result JSON.",
    )
  if "include_returns" not in fv:
    flags.DEFINE_boolean(
        "include_returns",
        False,
        "Also materialize game.returns for every final state in the batch. "
        "Use this to include terminal payoff/showdown work in the benchmark.",
    )
  if "include_training_tensors" not in fv:
    flags.DEFINE_boolean(
        "include_training_tensors",
        False,
        "Materialize observation_tensor and legal_actions_mask at every "
        "nonterminal step: the per-step tensors a learner reads, without "
        "storing trajectories.",
    )


def parse_batch_sizes(raw_batch_sizes: Sequence[str]) -> list[int]:
  """Parses and validates positive batch sizes.

  Args:
    raw_batch_sizes: The --batch_sizes items.

  Returns:
    The batch sizes.

  Raises:
    ValueError: there is none, or one is not positive.
  """
  batch_sizes = [int(item) for item in raw_batch_sizes if str(item).strip()]
  if not batch_sizes:
    raise ValueError("--batch_sizes must include at least one positive integer")
  if any(batch_size <= 0 for batch_size in batch_sizes):
    raise ValueError(f"--batch_sizes must be positive: {batch_sizes}")
  return batch_sizes


def ensure_gpu_or_allowed(allow_cpu: bool) -> jax.Device:
  """Returns the benchmark device, enforcing accelerator-first execution.

  Args:
    allow_cpu: Whether a CPU may be used when no accelerator is available.

  Returns:
    The first GPU or TPU, else the first device when `allow_cpu`.

  Raises:
    RuntimeError: no accelerator is available and `allow_cpu` is False.
  """
  devices = jax.devices()
  accelerator_devices = [
      device for device in devices if device.platform in ("gpu", "tpu")
  ]
  if accelerator_devices:
    return accelerator_devices[0]
  if allow_cpu:
    return devices[0]
  raise RuntimeError(
      "No JAX GPU or TPU device is available. Re-run with --allow_cpu for a "
      "CPU smoke run; throughput is meaningful only on an accelerator."
  )


def _make_run_batch(
    spec: BenchmarkSpec,
    batch_size: int,
    steps: int,
    include_returns: bool,
    include_training_tensors: bool,
):
  """Builds a JIT-compiled random-play rollout.

  Args:
    spec: The benchmark spec.
    batch_size: Rollouts per batch.
    steps: Steps per rollout.
    include_returns: Whether the rollout also returns the returns.
    include_training_tensors: Whether the rollout also checksums the training
      tensors.

  Returns:
    The jitted rollout, a function of a PRNG key.
  """
  game = spec.jax_game_factory()

  def sample_from_mask(key: jax.Array, mask: jax.Array) -> jax.Array:
    """Samples uniformly from a boolean mask with a static action dimension.

    Args:
      key: The PRNG key.
      mask: The mask; an all-False mask samples from every action.

    Returns:
      i32[], the sampled action.
    """
    mask = mask.astype(jnp.int32)
    count = jnp.sum(mask)
    safe_mask = jnp.where(count > 0, mask, jnp.ones_like(mask))
    safe_count = jnp.sum(safe_mask)
    rank = jax.random.randint(key, (), 0, safe_count)
    cumulative = jnp.cumsum(safe_mask)
    return jnp.argmax(cumulative > rank).astype(jnp.int32)

  @jax.jit
  def run_batch(key: jax.Array):
    """Plays `batch_size` random rollouts of the configured length.

    Args:
      key: The PRNG key.

    Returns:
      The terminal flags, then the returns and the tensor checksum when they
      are requested.
    """
    init_keys = jax.random.split(key, batch_size)
    states = jax.vmap(lambda _: game.new_initial_state())(init_keys)

    def scan_step(carry, xs):
      """Advances every rollout one step.

      Args:
        carry: (states, keys, tensor checksum).
        xs: The scan's xs, None (unused).

      Returns:
        ((states, keys, tensor checksum), None) for `jax.lax.scan`.
      """
      del xs
      states, keys, tensor_checksum = carry

      def one_step(state, key):
        """Advances one rollout one step; a no-op at a terminal state.

        Args:
          state: The state.
          key: The rollout's key.

        Returns:
          (next state, next key, the step's tensor checksum).
        """

        def active_step():
          """Samples and applies an action at a live state.

          Returns:
            (next state, next key, the step's tensor checksum).
          """
          next_key, action_key = jax.random.split(key)
          if include_training_tensors:
            observing_player = jnp.maximum(spec.current_player_fn(state), 0)
            observation = game.observation_tensor(state, observing_player)
            legal_mask = game.legal_actions_mask(state)
            step_checksum = jnp.sum(observation) + jnp.sum(
                legal_mask.astype(jnp.float32)
            )
          else:
            step_checksum = jnp.float32(0.0)
          player_key, chance_key = jax.random.split(action_key)
          player_action = sample_from_mask(
              player_key, game.legal_actions_mask(state)
          )
          chance_action = sample_from_mask(
              chance_key, game.chance_outcomes(state) > 0
          )
          action = jnp.where(
              spec.current_player_fn(state) == game_lib.CHANCE_PLAYER_ID,
              chance_action,
              player_action,
          )
          return game.apply_action(state, action), next_key, step_checksum

        next_state, key, step_checksum = jax.lax.cond(
            spec.terminal_fn(state),
            lambda: (state, key, jnp.float32(0.0)),
            active_step,
        )
        return next_state, key, step_checksum

      states, keys, step_checksums = jax.vmap(one_step)(states, keys)
      return (states, keys, tensor_checksum + jnp.sum(step_checksums)), None

    (final_states, _, tensor_checksum), _ = jax.lax.scan(
        scan_step, (states, init_keys, jnp.float32(0.0)), None, length=steps
    )
    terminals = spec.terminal_fn(final_states)
    if include_returns:
      zero_returns = jnp.zeros(
          (batch_size, game.num_players), dtype=jnp.float32
      )
      terminals, returns = jax.lax.cond(
          jnp.all(terminals),
          lambda: (terminals, jax.lax.map(game.returns, final_states)),
          lambda: (terminals, zero_returns),
      )
      if include_training_tensors:
        return terminals, returns, tensor_checksum
      return terminals, returns
    if include_training_tensors:
      return terminals, tensor_checksum
    return terminals

  return run_batch


def benchmark_batch(
    spec: BenchmarkSpec,
    batch_size: int,
    steps: int,
    num_runs: int,
    include_returns: bool,
    include_training_tensors: bool = False,
) -> dict[str, Any]:
  """Benchmarks one batch size.

  Args:
    spec: The game's benchmark spec.
    batch_size: The number of rollouts per batch.
    steps: The steps per rollout.
    num_runs: The timed runs.
    include_returns: Whether to compute the terminal returns.
    include_training_tensors: Whether to compute observations and legal masks
      at each step.

  Returns:
    Schema-ready metrics.

  Raises:
    ValueError: `steps` or `num_runs` is not positive.
    RuntimeError: with `include_returns`, a rollout did not reach a terminal
      state.
  """
  if steps <= 0:
    raise ValueError(f"--steps must be positive, got {steps}")
  if num_runs <= 0:
    raise ValueError(f"--num_runs must be positive, got {num_runs}")

  run_batch = _make_run_batch(
      spec, batch_size, steps, include_returns, include_training_tensors
  )

  t0 = time.perf_counter()
  output = run_batch(jax.random.PRNGKey(0))
  jax.block_until_ready(output)
  jit_compile_seconds = time.perf_counter() - t0

  durations = []
  for run_index in range(num_runs):
    t0 = time.perf_counter()
    output = run_batch(jax.random.PRNGKey(run_index + 1))
    jax.block_until_ready(output)
    durations.append(time.perf_counter() - t0)

  done = output[0] if include_returns or include_training_tensors else output
  completion_rate = float(jnp.mean(done))
  if include_returns and completion_rate < 1.0:
    raise RuntimeError(
        "--include_returns requires every rollout to reach a terminal state "
        "so payoff/showdown work is actually materialized. Increase --steps "
        f"or disable --include_returns. completion_rate={completion_rate:.2%}"
    )
  steady_seconds = float(np.median(np.asarray(durations, dtype=np.float64)))
  total_steps = batch_size * steps
  steps_per_second = total_steps / steady_seconds
  return {
      "batch_size": batch_size,
      "steps": steps,
      "num_runs": num_runs,
      "include_returns": include_returns,
      "include_training_tensors": include_training_tensors,
      "jit_compile_seconds": jit_compile_seconds,
      "steady_seconds_median": steady_seconds,
      "steps_per_second": steps_per_second,
      "games_per_second": batch_size / steady_seconds,
      "completion_rate": completion_rate,
  }


def benchmark_cpp(
    pyspiel_game_string: str, num_games: int, max_steps_per_game: int
) -> dict[str, Any] | None:
  """Runs a single-thread pyspiel random-play comparison anchor.

  Args:
    pyspiel_game_string: The pyspiel game.
    num_games: The number of games.
    max_steps_per_game: The step cap per game.

  Returns:
    The metrics, or None when pyspiel or the game is unavailable.

  Raises:
    ValueError: `num_games` or `max_steps_per_game` is not positive.
  """
  if num_games <= 0:
    raise ValueError(f"--cpp_games must be positive, got {num_games}")
  if max_steps_per_game <= 0:
    raise ValueError(
        f"--cpp_max_steps_per_game must be positive, got {max_steps_per_game}"
    )
  try:
    # Lazy: pyspiel is an optional dependency (the C++ baseline only).
    import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # pyrefly: ignore[missing-import]  # noqa: PLC0415
  except ImportError:
    logging.exception("Skipping C++ baseline because pyspiel is unavailable")
    return None

  try:
    game = pyspiel.load_game(pyspiel_game_string)
  except RuntimeError:
    logging.exception(
        "Skipping C++ baseline because pyspiel.load_game failed for %s",
        pyspiel_game_string,
    )
    return None
  rng = np.random.default_rng(0)
  total_steps = 0
  truncated_games = 0
  t0 = time.perf_counter()
  for _ in range(num_games):
    state = game.new_initial_state()
    game_steps = 0
    while not state.is_terminal() and game_steps < max_steps_per_game:
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probabilities = zip(*outcomes)
        action = rng.choice(actions, p=probabilities)
      else:
        legal_actions = state.legal_actions()
        if not legal_actions:
          break
        action = rng.choice(legal_actions)
      state.apply_action(int(action))
      total_steps += 1
      game_steps += 1
    if not state.is_terminal():
      truncated_games += 1
  elapsed = time.perf_counter() - t0
  return {
      "engine": "pyspiel C++",
      "game_string": pyspiel_game_string,
      "num_games": num_games,
      "max_steps_per_game": max_steps_per_game,
      "truncated_games": truncated_games,
      "steps": total_steps,
      "wall_seconds": elapsed,
      "steps_per_second": total_steps / elapsed,
      "games_per_second": num_games / elapsed,
  }


def run_benchmark(
    spec: BenchmarkSpec,
    batch_sizes: Sequence[int],
    steps: int,
    num_runs: int,
    compare_cpp: bool,
    allow_cpu: bool,
    cpp_games: int,
    cpp_max_steps_per_game: int = DEFAULT_CPP_MAX_STEPS_PER_GAME,
    include_returns: bool = False,
    include_training_tensors: bool = False,
) -> dict[str, Any]:
  """Runs a full batch sweep.

  Args:
    spec: The game's benchmark spec.
    batch_sizes: The batch sizes.
    steps: The steps per rollout.
    num_runs: The timed runs per batch size.
    compare_cpp: Whether to run the pyspiel anchor.
    allow_cpu: Whether a CPU may be used.
    cpp_games: The pyspiel anchor's games.
    cpp_max_steps_per_game: The pyspiel anchor's step cap per game.
    include_returns: Whether to compute the terminal returns.
    include_training_tensors: Whether to compute the training tensors.

  Returns:
    A stable JSON-compatible result.
  """
  device = ensure_gpu_or_allowed(allow_cpu)
  results = []
  for batch_size in batch_sizes:
    print(f"running {spec.game_name} batch_size={batch_size}...", flush=True)
    row = benchmark_batch(
        spec,
        batch_size,
        steps,
        num_runs,
        include_returns,
        include_training_tensors,
    )
    print(
        f"completed batch_size={batch_size}: "
        f"{row['steps_per_second']:,.0f} steps/sec",
        flush=True,
    )
    results.append(row)
  cpp_baseline = (
      benchmark_cpp(spec.pyspiel_game_string, cpp_games, cpp_max_steps_per_game)
      if compare_cpp
      else None
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "game": spec.game_name,
      "jax_device": str(device),
      "jax_platform": device.platform,
      "config": {
          "batch_sizes": list(batch_sizes),
          "steps": steps,
          "num_runs": num_runs,
          "compare_cpp": compare_cpp,
          "cpp_games": cpp_games if compare_cpp else 0,
          "cpp_max_steps_per_game": (
              cpp_max_steps_per_game if compare_cpp else 0
          ),
          "include_returns": include_returns,
          "include_training_tensors": include_training_tensors,
      },
      "results": results,
      "cpp_baseline": cpp_baseline,
  }


def print_report(result: dict[str, Any]) -> None:
  """Prints a human-readable report with the same fields as the JSON.

  Args:
    result: The run_benchmark result.
  """
  print("=" * 88)
  print(f"JAX game throughput benchmark: {result['game']}")
  print("=" * 88)
  print(f"device: {result['jax_device']}")
  config = result["config"]
  print(
      f"config: batch_sizes={config['batch_sizes']}, steps={config['steps']}, "
      f"num_runs={config['num_runs']}, compare_cpp={config['compare_cpp']}, "
      f"include_returns={config['include_returns']}, "
      f"include_training_tensors={config['include_training_tensors']}"
  )
  print()
  print(
      f"{'batch':>8} {'steps':>6} {'jit_s':>8} {'steady_s':>10} "
      f"{'steps/sec':>14} {'games/sec':>12} {'done':>7}"
  )
  print("-" * 88)
  for row in result["results"]:
    print(
        f"{row['batch_size']:>8} {row['steps']:>6} "
        f"{row['jit_compile_seconds']:>8.2f} "
        f"{row['steady_seconds_median']:>10.4f} "
        f"{row['steps_per_second']:>14,.0f} "
        f"{row['games_per_second']:>12,.0f} "
        f"{row['completion_rate']:>7.2%}"
    )
  print("-" * 88)
  best = max(result["results"], key=lambda row: row["steps_per_second"])
  print(
      f"best_jax_steps_per_second: {best['steps_per_second']:,.0f} "
      f"(batch={best['batch_size']})"
  )
  if result["cpp_baseline"] is not None:
    cpp = result["cpp_baseline"]
    print(
        "pyspiel_cpp_steps_per_second: "
        f"{cpp['steps_per_second']:,.0f} "
        f"({cpp['steps']} steps / {cpp['wall_seconds']:.3f}s)"
    )
    if cpp["truncated_games"]:
      print(
          "pyspiel_cpp_truncated_games: "
          f"{cpp['truncated_games']} / {cpp['num_games']} "
          f"(max_steps_per_game={cpp['max_steps_per_game']})"
      )
    print(
        "best_jax_over_pyspiel_cpp: "
        f"{best['steps_per_second'] / cpp['steps_per_second']:,.1f}x"
    )


def write_json(result: dict[str, Any], output_json: str | None) -> None:
  """Writes the benchmark result JSON when requested.

  Args:
    result: The run_benchmark result.
    output_json: The path, or None/empty to write nothing.
  """
  if not output_json:
    return
  os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
  with open(output_json, "w", encoding="utf-8") as fp:
    json.dump(result, fp, indent=2, sort_keys=True)
    fp.write("\n")
