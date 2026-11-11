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

r"""Run one training run to completion on a backend, surviving preemptions.

  python -m open_spiel.python.scale.launch.run --backend_config=backend.yaml \
      --run_id=kuhn-a [--max_preemptions=20] [--capacity_wait=10m] \
      [--poll=30s] \
      [--deadline=DURATION] [--plugins=pkg.mod,...] [--retry] [--dry_run] \
      -- --config=mmd_kuhn_poker_jax --seed=0 [any other trainer flag]

The backend config is one file (launch.config: local, gcp or a
registered kind). The run lives under the backend's storage at
<storage>/<run_id>: its supervisor.json (one run, every attempt), its
checkpoint dir ckpt/ and each attempt's log. Running the same command again
continues the run or reports it done; a preempted attempt is relaunched and
resumes from the checkpoint (launch.supervisor). Before anything is created
(a dry run included), a run whose game needs a game package the backend does
not ship is refused (launch.preflight). --dry_run prints the
backend's plan for the first attempt (Backend.plan: for gcp, the VM startup
scripts and gcloud commands) and creates nothing. --plugins: modules the
launcher and the trainer import first (passed on in plugins.ENV); a plugin may
register a backend kind. --deadline (for example 80h) bounds the run's wall
time; by default there is none.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Sequence

from absl import logging

from open_spiel.python.scale import plugins as plugins_lib
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import config as config_lib
from open_spiel.python.scale.launch import preflight
from open_spiel.python.scale.launch import supervisor as supervisor_lib

#: This command line's module (`python -m MODULE`).
MODULE = backend_lib.TRAINER_MODULE.rpartition(".")[0] + ".launch.run"
LOG_PREFIX = "[run]"


def parse_duration(s: str) -> float:
  """Seconds of "<N>s|m|h|d" (for example "10m").

  Args:
    s: the duration.

  Returns:
    its seconds.

  Raises:
    ValueError: s is not "<N>s|m|h|d".
  """
  m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", s.strip())
  if not m:
    raise ValueError(f"duration {s!r} is not <N>s|m|h|d")
  return (
      float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
  )


def main(argv: Sequence[str]) -> int:
  """Runs the command line (module doc); returns the exit code.

  Args:
    argv: the command line, without the program name.

  Returns:
    the exit code.

  Raises:
    SystemExit: there is no `--` before the trainer flags.
  """
  # An entry point (package doc); the trainers it starts inherit it.
  frameworks.enable_x64()
  argv = list(argv)
  if "--" not in argv:
    raise SystemExit(f"usage: python -m {MODULE} [flags] -- <trainer flags>")
  i = argv.index("--")
  ap = argparse.ArgumentParser(prog=MODULE)
  ap.add_argument("--backend_config", required=True)
  ap.add_argument("--run_id", required=True)
  ap.add_argument("--max_preemptions", type=int, default=20)
  ap.add_argument("--capacity_wait", default="10m")
  ap.add_argument("--poll", default="30s")
  ap.add_argument("--deadline", default=None)
  ap.add_argument("--plugins", default="")
  ap.add_argument("--retry", action="store_true")
  ap.add_argument("--dry_run", action="store_true")
  a = ap.parse_args(argv[1:i])
  run = backend_lib.RunSpec(
      a.run_id,
      train_args=tuple(argv[i + 1 :]),
      plugins=tuple(p for p in a.plugins.split(",") if p),
  )
  plugins_lib.load_env()
  plugins_lib.load(run.plugins)
  backend = config_lib.make_backend(
      config_lib.load(a.backend_config), dry_run=a.dry_run
  )
  check_shipped(backend, run, a.backend_config)
  if a.dry_run:
    return dry_run(backend, run)
  sup = supervisor_lib.Supervisor(
      backend,
      max_preemptions=a.max_preemptions,
      capacity_wait_s=parse_duration(a.capacity_wait),
      poll_s=parse_duration(a.poll),
      deadline_s=parse_duration(a.deadline) if a.deadline else None,
      on_event=lambda m: print(
          f"{LOG_PREFIX} {supervisor_lib.utc()} {m}", flush=True
      ),
  )
  return sup.run(run, retry=a.retry)


def check_shipped(
    backend: backend_lib.Backend, run: backend_lib.RunSpec, config_path: str
) -> None:
  """Refuses a run whose game needs a game package the backend lacks.

  Refuses, before any machine exists (and in a dry run), a run whose game
  needs a game package the backend does not ship (launch.preflight).

  Args:
    backend: the backend.
    run: the run.
    config_path: the backend config file, for the fix.

  Raises:
    SystemExit: the game needs a game package the backend does not ship.
  """
  shipped = backend.shipped_packages()
  if shipped is None:
    return
  try:
    game = preflight.check_game_packages(
        run.train_args,
        shipped,
        run.plugins,
        fix=f"Add ${{GAMES_PACKAGE_PATH}} to code.paths in {config_path}.",
    )
  except preflight.GamePackageNotShippedError as e:
    raise SystemExit(f"{LOG_PREFIX} preflight: {e}; nothing created") from e
  if game is not None:
    print(
        f"{LOG_PREFIX} preflight: game {game} needs no unshipped game package"
    )


def dry_run(backend: backend_lib.Backend, run: backend_lib.RunSpec) -> int:
  """Prints the backend's plan of attempt 0; creates nothing.

  Args:
    backend: the backend.
    run: the run.

  Returns:
    0.
  """
  print(
      f"{LOG_PREFIX} dry run: run {run.run_id} at {backend.run_uri(run.run_id)}"
  )
  print(backend.plan(run, 0))
  print(f"{LOG_PREFIX} dry run: nothing created, nothing launched")
  return 0


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  sys.exit(main(sys.argv))
