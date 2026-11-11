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

"""The eval deadline: a hard wall-clock bound on in-loop evals.

The train entry point's --wall_budget stops training at the first iteration
boundary past the budget (league.runner.run), but an in-loop eval (for example a
long MCTS-N match) cannot be interrupted at an iteration boundary. So while a
deadline is set (`active`), every eval rollout checks it between rollout
segments and between row chunks (core.arena_engine, eval.match) and raises
EvalOverBudgetError once it has passed, so the run stops with that error instead
of running past its wall budget. One deadline per process (the
runner sets it around its loop); `check` is a no-op without one.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Callable, Iterator, Optional

#: Plies per rollout segment while a deadline is set (core.arena_engine), so a
#: long match is checked often enough: one MCTS-N ply can take seconds.
SEGMENT_PLIES = 32


class EvalOverBudgetError(RuntimeError):
  """An eval was still running when the run's eval deadline passed."""


_STATE: dict[str, Any] = {"at": None, "what": "", "clock": time.monotonic}


def active() -> bool:
  """Whether an eval deadline is set.

  Returns:
    True inside a `deadline` with a time set.
  """
  return _STATE["at"] is not None


@contextlib.contextmanager
def deadline(
    at: Optional[float], what: str, clock: Callable[[], float] = time.monotonic
) -> Iterator[None]:
  """Evals inside raise EvalOverBudgetError once clock() >= at (None: none).

  Not nestable: one deadline per process.

  Args:
    at: the clock() time of the deadline, or None for no deadline.
    what: what the deadline bounds, for messages.
    clock: the clock `at` is read against.

  Yields:
    None, once, with the deadline set.

  Raises:
    RuntimeError: a deadline is already set.
  """
  if at is None:
    yield
    return
  if _STATE["at"] is not None:
    raise RuntimeError(f"an eval deadline is already set ({_STATE['what']})")
  _STATE.update(at=float(at), what=str(what), clock=clock)
  try:
    yield
  finally:
    _STATE.update(at=None, what="", clock=time.monotonic)


def check(where: str) -> None:
  """Raises EvalOverBudgetError when the deadline has passed (else a no-op).

  Args:
    where: the caller, for the message.

  Raises:
    EvalOverBudgetError: the deadline has passed.
  """
  at = _STATE["at"]
  if at is None:
    return
  now = _STATE["clock"]()
  if now >= at:
    raise EvalOverBudgetError(
        f"{where}: the eval deadline ({_STATE['what']}) passed "
        f"{now - at:.0f} s ago; the eval is stopped"
    )
