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

r"""The first thing to run: PPO learns tic-tac-toe on the CPU in minutes.

  python -m open_spiel.python.scale.examples.quickstart
  python -m open_spiel.python.scale.examples.quickstart \
      --updates=30 --out_dir=/tmp/my_quickstart

It trains the named config
`ppo_tic_tac_toe` (PPO against a uniformly random opponent) through the
command-line trainer's own entry point (`train.run`), the path a run takes
on any machine, at a size that finishes in a few minutes on a laptop CPU.
It then plays the trained policy against the random
opponent on seat-swapped deals, prints its score, and leaves the checkpoint
and metrics in --out_dir (default /tmp/quickstart; running the same command
again resumes from it, a new --out_dir starts afresh).

The score is the trained policy's mean payoff per game (+1 a win, -1 a loss,
0 a draw) with its standard error, and the equivalent points share (wins plus
half the draws). It comes from one short run: an indication that learning
works, not a benchmark result.

The same config trains on a Google Cloud VM with one command, the cloud
launcher (`launch/cloud.py`, `--config=ppo_tic_tac_toe` after its `--`).
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Optional, Sequence

from absl import app
from absl import flags

from open_spiel.python.scale import metrics
from open_spiel.python.scale import train
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import runner

CONFIG = "ppo_tic_tac_toe"
EVAL = "vs_uniform"

# Names of their own: the trainer's flags (num_envs, seed, ...) are defined
# in the same process.
_UPDATES = flags.DEFINE_integer("updates", 20, "PPO updates (the run length)")
_ROWS = flags.DEFINE_integer("rows", 512, "games played in parallel per update")
_PLIES = flags.DEFINE_integer("plies", 32, "plies per game row per update")
_EVAL_DEALS = flags.DEFINE_integer(
    "eval_deals", 512, "seat-swapped deal pairs against the random opponent"
)
_RUN_SEED = flags.DEFINE_integer("run_seed", 0, "the run's seed")
_OUT_DIR = flags.DEFINE_string(
    "out_dir",
    "/tmp/quickstart",
    "checkpoint and metrics directory (a rerun resumes from it)",
)
flags.DEFINE_alias("out", "out_dir")


def quickstart_config(
    updates: int, num_envs: int, num_steps: int, eval_deals: int, seed: int
):
  """Returns ppo_tic_tac_toe resized, with an evaluation against random play.

  The evaluation runs at its first and last update.

  Args:
    updates: The number of updates.
    num_envs: The parallel envs.
    num_steps: The steps per update.
    eval_deals: The evaluation deals.
    seed: The seed.

  Returns:
    The training config.
  """
  cfg = config_common.with_overrides(
      config_common.load_config(CONFIG),
      num_updates=updates,
      num_envs=num_envs,
      num_steps=num_steps,
      seed=seed,
  )
  return dataclasses.replace(
      cfg,
      eval=runner.InLoopEval(EVAL, every=updates, deals=eval_deals, seed=seed),
  )


def last_eval(out: str) -> Optional[dict[str, Any]]:
  """Returns the last vs-random evaluation the run wrote to its metrics.

  Args:
    out: The run directory.

  Returns:
    {mean, se, n_deals, ...}, or None when there is none.
  """
  path = os.path.join(out, "metrics", metrics.JSONL_NAME)
  found = None
  for row in metrics.read_jsonl(path):
    stem = next(
        (k[: -len("mean")] for k in row if k.endswith(f"{EVAL}/mean")), None
    )
    if stem is not None:
      found = {k[len(stem) :]: row[k] for k in row if k.startswith(stem)}
  return found


def run(
    updates: int,
    num_envs: int,
    num_steps: int,
    eval_deals: int,
    seed: int,
    out: str,
) -> dict[str, Any]:
  """Trains and evaluates.

  Args:
    updates: The number of updates.
    num_envs: The parallel envs.
    num_steps: The steps per update.
    eval_deals: The evaluation deals.
    seed: The seed.
    out: The run directory.

  Returns:
    The last evaluation with the wall time ("seconds").

  Raises:
    RuntimeError: the run recorded no evaluation.
  """
  t0 = time.monotonic()
  cfg = quickstart_config(updates, num_envs, num_steps, eval_deals, seed)
  train.run(cfg, checkpoint_dir=out)
  ev = last_eval(out)
  if ev is None:
    raise RuntimeError(f"the run recorded no eval/{EVAL} metrics")
  ev["seconds"] = time.monotonic() - t0
  return ev


def main(argv: Sequence[str]) -> None:
  """Trains, evaluates and prints the result.

  Args:
    argv: Unused.
  """
  # The framework needs 64-bit JAX; on before anything is built.
  x64.enable_x64()
  del argv
  out = os.path.abspath(_OUT_DIR.value)
  print(
      f"training PPO on tic-tac-toe against a random opponent ({_UPDATES.value}"
      f" updates of {_ROWS.value} x {_PLIES.value} plies) into {out}"
  )
  ev = run(
      _UPDATES.value,
      _ROWS.value,
      _PLIES.value,
      _EVAL_DEALS.value,
      _RUN_SEED.value,
      out,
  )
  mean, se = float(ev["mean"]), float(ev.get("se", float("nan")))
  print(
      f"vs uniform random, {int(ev.get('n_deals', 0))} seat-swapped deal pairs:"
      f" mean payoff {mean:+.3f} +/- {se:.3f} (points share "
      f"{(1 + mean) / 2:.1%});"
      " one short run, an indication rather than a benchmark"
  )
  print(f"done in {ev['seconds']:.0f} s; checkpoint and metrics in {out}")


if __name__ == "__main__":
  app.run(main)
