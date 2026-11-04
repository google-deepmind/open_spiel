# No-Limit Hold'em Hand Evaluator

`no_limit_holdem` scores 5-card poker hands with a precomputed lookup
table and scores 7-card hands by evaluating all 21 five-card subsets.

## Table Key

The table uses a combinadic key over the sorted rank multiset plus the flush
flag. For sorted ranks `r0 <= r1 <= r2 <= r3 <= r4`, first adjust the ranks to
the strictly increasing sequence:

```text
a0 = r0, a1 = r1 + 1, a2 = r2 + 2, a3 = r3 + 3, a4 = r4 + 4
```

The key is:

```text
(C(a0, 1) + C(a1, 2) + C(a2, 3) + C(a3, 4) + C(a4, 5)) * 2 + flush
```

This maps every 5-card rank pattern and flush/non-flush state to a stable table
slot, so no collision resolution or prime-product hash table is needed. A
Cactus-Kev-style evaluator would use a smaller logical hand-rank space, but this
direct table keeps the JAX runtime path simple: sort five ranks, test flush,
sum five cached binomial values, and gather one packed score.

## Build And Runtime Cost

The score table is built at import time as a NumPy `int32` array, following the
cached-constants rule in `docs/cached_constants.md`. JAX sees it as an
XLA constant only when `_score_5` is traced.

- Entries: `C(13 + 5 - 1, 5) * 2 = 12,376`
- Score table size: about 48 KiB (`int32`)
- Import work: enumerate valid sorted rank patterns once, score flush and
  non-flush variants with a vectorized NumPy implementation, and fill the table
  by rank/flush key

## Correctness

The table stores the same packed category-and-kicker integers as the
categorical JAX evaluator (`best_seven_card_score_uncached`).
`no_limit_holdem_test` walks all 2,598,960 five-card hands and checks the table
against the vectorized categorical reference that builds it, then checks
that reference against the categorical JAX evaluator across every hand
category. Equal packed integers give `best_seven_card_score` the same total
order as the categorical evaluator. Because the table and both evaluators
share one scoring rule, known-answer tests against pyspiel (every hand class,
and kickers such as a four-of-a-kind kicker held in the hole) check the rule
itself.

## Benchmarking

Run the focused showdown benchmark with:

```bash
python -m open_spiel.python.jax.games.benchmarks.no_limit_holdem_showdown_benchmark \
    --batch_size=1048576 --num_runs=10 --evaluator=cache
```

Use `--evaluator=no_cache` to benchmark the categorical evaluator on the same
random hands.

Run the end-to-end no-limit hold'em random-play benchmark with terminal returns
materialized so showdown scoring is included:

```bash
python -m open_spiel.python.jax.games.benchmarks.no_limit_holdem_throughput \
    --use_showdown_cache=true \
    --batch_sizes=256,1024 --steps=80 --num_runs=5 --include_returns
```

Set `--use_showdown_cache=false` for the no-cache variant. `--include_returns`
is important for this comparison because terminal returns are where no-limit
hold'em calls the 7-card showdown evaluator.

Throughput is meaningful on an accelerator; `--allow_cpu` is for smoke checks.
