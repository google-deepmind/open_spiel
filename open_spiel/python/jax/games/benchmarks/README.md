# Throughput benchmarks

Random-play throughput benchmarks for the JAX games, with an optional
single-thread pyspiel C++ anchor.

## Benchmarks

The commands use the package's import path in an OpenSpiel install
(`open_spiel.python.jax.games`).

```bash
python -m open_spiel.python.jax.games.benchmarks.gin_rummy_throughput \
    --batch_sizes=256,1024,4096,16384 --steps=200 --num_runs=5

python -m open_spiel.python.jax.games.benchmarks.no_limit_holdem_throughput \
    --batch_sizes=256,1024,4096,16384 --steps=80 --num_runs=5

python -m open_spiel.python.jax.games.benchmarks.no_limit_holdem_showdown_benchmark \
    --batch_size=1048576 --num_runs=10 --evaluator=cache
```

The two throughput binaries share one flag surface:

- `--batch_sizes`: comma-separated batch sizes.
- `--steps`: scanned environment steps per timed rollout.
- `--num_runs`: post-JIT timed runs per batch size.
- `--compare_cpp` / `--nocompare_cpp`: run or skip the single-thread pyspiel
  C++ random-play anchor.
- `--include_returns` / `--noinclude_returns`: materialize `game.returns` for
  every final state so payoff/showdown work is included in the timed rollout.
- `--include_training_tensors`: materialize `observation_tensor` and
  `legal_actions_mask` at every nonterminal step (the per-step tensors a
  learner reads), without storing trajectories.
- `--cpp_games`: sequential pyspiel games played for the C++ anchor.
- `--cpp_max_steps_per_game`: pyspiel steps after which a random C++ game
  counts as truncated (keeps no-limit poker's random betting finite).
- `--allow_cpu`: permit a CPU run (see below).
- `--output_json`: write the full result JSON (see "Output schema").

A GPU or TPU is the intended target. The harness fails when no JAX accelerator
is present unless `--allow_cpu` is set for a smoke run.

`no_limit_holdem_showdown_benchmark` is a narrower microbenchmark for the
7-card evaluator used during showdown. It pre-samples random unique-card hands,
times only 7-card scoring, and reports hands/sec. Use `--evaluator=cache` for
the lookup-table path and `--evaluator=no_cache` for the categorical reference
path. For no-limit hold'em full-game runs, `--use_showdown_cache=false`
selects the no-cache variant.

On a TPU, install JAX for TPU first (see the JAX installation guide).

### What the gin rummy numbers mean

`gin_rummy_throughput` scans `--steps` steps of `batch_size` independent games
under `vmap`, choosing uniformly among the legal actions, and reports the JIT
compile time and the median steady-state time of `--num_runs` runs.

- On an accelerator, throughput grows with the batch until the per-step
  deadwood and meld computation saturates the device, then stays flat. The
  JAX game is not designed for CPU; CPU runs are a smoke test.
- The policy is uniform random, which draws and discards aimlessly, so games
  run longer than games between competent players; `completion_rate` reports
  the share that ended within `--steps`.
- The C++ baseline plays games one at a time, because pyspiel has no batched
  API. It anchors the scale of the numbers; it is not a head-to-head
  comparison.
- The first call on a large batch compiles for tens of seconds. Set
  `JAX_COMPILATION_CACHE_DIR` to reuse compiled programs across processes.

## Output schema

Both throughput benchmarks print a table and can write the same JSON schema
with `--output_json=/abs/path/result.json`. The top-level fields are:

- `schema_version`
- `game`
- `jax_device`
- `jax_platform`
- `config`
- `results`
- `cpp_baseline`

Each result row records `batch_size`, `steps`, `num_runs`, `include_returns`,
`jit_compile_seconds`, `steady_seconds_median`, `steps_per_second`,
`games_per_second`, and `completion_rate`.

Results are throughput anchors, not pass/fail thresholds: compare runs on the
same class of accelerator before drawing conclusions.
