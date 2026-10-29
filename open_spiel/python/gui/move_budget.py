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


"""The move budget: how long a controller may take to choose a move.

`run` calls a controller on a thread of its own and waits for it no longer
than the budget, 10 seconds unless the server's `--move-budget-seconds` (or
`set_seconds`) sets another. A request that asks a controller for a move so
waits at most the budget, and the server's other requests are answered
meanwhile on their own threads. Python cannot stop a thread, so stopping is
cooperative: when the budget expires, `run` asks the call to stop, and a call
that may take long (MCTS, between simulations) calls `check` (or reads
`stop_requested`) often and returns or raises soon after. A call that never
checks runs on, on its own copy of the state, until it returns. Either way
what it returns after the budget is discarded; its seat has failed by then,
and is not asked again until the session is undone or reset, or the seat is
given another controller.
"""

from collections.abc import Callable
import dataclasses
import math
import queue
import threading
from typing import TypeVar

# How long a controller may take to choose a move, in seconds, by default.
DEFAULT_SECONDS = 10.0

_T = TypeVar("_T")


class OverBudgetError(Exception):
  """A call did not return within the move budget.

  Attributes:
    seconds: The budget it overran.
  """

  def __init__(self, budget: float) -> None:
    super().__init__(f"No result within the {budget:g} second move budget.")
    self.seconds = budget


@dataclasses.dataclass
class _Budget:
  """The process's move budget.

  Attributes:
    seconds: How long a controller may take to choose a move.
  """

  seconds: float = DEFAULT_SECONDS


_BUDGET = _Budget()


class StoppedError(Exception):
  """A call that `run` asked to stop, its budget expired, checked and stopped.

  Raised by `check` on the call's own thread; `run` has raised
  `OverBudgetError` to its caller by then, so nothing waits for this.
  """

  def __init__(self) -> None:
    super().__init__("The move budget expired; the call was asked to stop.")


class _Current(threading.local):
  """The stop flag of the call `run` runs on this thread, if any.

  Attributes:
    stop: Set when the call's budget expires; None on a thread `run` did not
      start.
  """

  stop: threading.Event | None = None


_CURRENT = _Current()


def stop_requested() -> bool:
  """Whether the call running on this thread has been asked to stop.

  True once the budget of the `run` that started this thread has expired;
  always False on a thread `run` did not start.
  """
  stop = _CURRENT.stop
  return stop is not None and stop.is_set()


def check() -> None:
  """Raises `StoppedError` if the call on this thread was asked to stop.

  A controller that may take long calls this often, between units of its
  work, so that a move that overran its budget stops soon after, and its
  thread exits, rather than computing a result that is discarded.

  Raises:
    StoppedError: If the call's budget has expired.
  """
  if stop_requested():
    raise StoppedError()


def set_seconds(value: float) -> None:
  """Sets the move budget, for every move asked for from now on.

  Args:
    value: The budget in seconds, positive and finite.

  Raises:
    ValueError: If `value` is not positive and finite.
  """
  if not (math.isfinite(value) and value > 0):
    raise ValueError(
        f"The move budget is {value!r} seconds; it must be positive and finite."
    )
  _BUDGET.seconds = float(value)


def seconds() -> float:
  """Returns the move budget in seconds."""
  return _BUDGET.seconds


def run(call: Callable[[], _T], *, name: str) -> _T:
  """Returns what `call` returns, calling it on a thread of its own.

  Args:
    call: What to call.
    name: The thread's name, for logs and debuggers.

  Returns:
    What `call` returned.

  Raises:
    OverBudgetError: If `call` has not returned within the budget. `call` is
      then asked to stop: `check` raises on its thread from then on.
    Exception: Whatever `call` raised, raised again here.
  """
  budget = _BUDGET.seconds
  # (returned, value): the result if `call` returned, else what it raised.
  outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
  stop = threading.Event()

  def target() -> None:
    _CURRENT.stop = stop
    try:
      outcome.put((True, call()))
    # Handed to the waiting thread, which raises it there.
    except Exception as e:  # pylint: disable=broad-exception-caught
      outcome.put((False, e))

  # A daemon, so a call that never returns cannot keep the process alive.
  threading.Thread(target=target, name=name, daemon=True).start()
  try:
    returned, value = outcome.get(timeout=budget)
  except queue.Empty:
    stop.set()
    raise OverBudgetError(budget) from None
  if not returned:
    assert isinstance(value, Exception)
    raise value
  return value  # type: ignore[return-value]
