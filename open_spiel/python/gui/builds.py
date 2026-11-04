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

"""Builds: a seat's controller, built on a thread of its own.

Building a controller can take a while: a bot that plays a trained model
loads it and warms it up, in tens of seconds for a large game. A session so
builds each seat's controller on a thread of its own (`Build`), and waits for
it only briefly (`wait_seconds`, a quarter of a second unless `set_wait_seconds`
says otherwise): a build that finishes by then is used at once, as if built
in the request; one that does not leaves its seat `preparing`, and the
request answers meanwhile. The seat is ready when its build finishes, and
autoplay goes on from there.

A build that takes longer than the build budget (`budget_seconds`, 120
seconds unless the server's `--build-budget-seconds`, or `set_budget_seconds`,
says otherwise) fails with `OverBudgetError`. Python cannot stop a thread, so
the build runs on, and what it returns then is discarded.
"""

from collections.abc import Callable
import dataclasses
import math
import threading
from typing import Generic, TypeVar

# How long a controller may take to be built, in seconds, by default.
DEFAULT_BUDGET_SECONDS = 120.0

# How long a request waits for a build before answering, by default.
DEFAULT_WAIT_SECONDS = 0.25

_T = TypeVar("_T")


class OverBudgetError(Exception):
  """A build did not finish within the build budget.

  Attributes:
    seconds: The budget it overran.
  """

  def __init__(self, budget: float) -> None:
    super().__init__(f"No controller within the {budget:g} second budget.")
    self.seconds = budget


@dataclasses.dataclass
class _Settings:
  """The process's build settings.

  Attributes:
    budget: How long a controller may take to be built, in seconds.
    wait: How long a request waits for a build, in seconds.
  """

  budget: float = DEFAULT_BUDGET_SECONDS
  wait: float = DEFAULT_WAIT_SECONDS


_SETTINGS = _Settings()


def _check_seconds(value: float, what: str, *, zero: bool) -> float:
  """Returns `value` as a float, if it is finite and positive (or `zero`)."""
  if not (math.isfinite(value) and (value > 0 or (zero and value == 0))):
    bound = "non-negative" if zero else "positive"
    raise ValueError(
        f"The {what} is {value!r} seconds; it must be {bound} and finite."
    )
  return float(value)


def set_budget_seconds(value: float) -> None:
  """Sets the build budget, for every build started from now on.

  Raises:
    ValueError: If `value` is not positive and finite.
  """
  _SETTINGS.budget = _check_seconds(value, "build budget", zero=False)


def budget_seconds() -> float:
  """Returns the build budget in seconds."""
  return _SETTINGS.budget


def set_wait_seconds(value: float) -> None:
  """Sets how long a request waits for a build before answering.

  Raises:
    ValueError: If `value` is not non-negative and finite.
  """
  _SETTINGS.wait = _check_seconds(value, "build wait", zero=True)


def wait_seconds() -> float:
  """Returns how long a request waits for a build, in seconds."""
  return _SETTINGS.wait


class Build(Generic[_T]):
  """One call, made on a thread of its own and bounded by the build budget.

  The build finishes once, with what the call returned, what it raised, or
  `OverBudgetError` if the budget ran out first; `on_done` then hears of it,
  on the thread that finished it, exactly once.
  """

  def __init__(
      self,
      call: Callable[[], _T],
      *,
      name: str,
      on_done: Callable[["Build[_T]"], None],
  ) -> None:
    """Prepares the build; `start` starts it.

    Args:
      call: What builds the result.
      name: The thread's name, for logs and debuggers.
      on_done: Called with the build once it has finished.
    """
    self._call = call
    self._name = name
    self._on_done = on_done
    self._lock = threading.Lock()
    self._finished = threading.Event()
    self._result: _T | None = None
    self._error: Exception | None = None
    self._timer: threading.Timer | None = None

  def start(self) -> None:
    """Starts the call, and the budget's clock.

    Raises:
      RuntimeError: If the build has started already.
    """
    with self._lock:
      if self._timer is not None:
        raise RuntimeError(f"{self._name} has started already.")
      budget = _SETTINGS.budget
      self._timer = threading.Timer(
          budget, self._finish, args=(None, OverBudgetError(budget))
      )
    # Daemons, so a build that never returns cannot keep the process alive.
    self._timer.daemon = True
    self._timer.start()
    threading.Thread(target=self._run, name=self._name, daemon=True).start()

  def wait(self, timeout: float) -> bool:
    """Waits up to `timeout` seconds; returns whether the build finished."""
    return self._finished.wait(max(timeout, 0.0))

  @property
  def done(self) -> bool:
    """Whether the build has finished."""
    return self._finished.is_set()

  def outcome(self) -> tuple[_T | None, Exception | None]:
    """Returns what the call returned, or else what made the build fail.

    Returns:
      (result, None) if the call returned in time; (None, error) if it
      raised `error`, or the budget ran out (`OverBudgetError`).

    Raises:
      RuntimeError: If the build has not finished.
    """
    if not self._finished.is_set():
      raise RuntimeError(f"{self._name} has not finished.")
    return self._result, self._error

  def _run(self) -> None:
    """Makes the call, and finishes with what it returned or raised."""
    try:
      result = self._call()
    # Handed to whoever reads the outcome, which raises or records it.
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._finish(None, e)
    else:
      self._finish(result, None)

  def _finish(self, result: _T | None, error: Exception | None) -> None:
    """Records the first outcome, and tells `on_done`; drops any later one."""
    with self._lock:
      if self._finished.is_set():
        return
      self._result, self._error = result, error
      self._finished.set()
      assert self._timer is not None
      self._timer.cancel()
    self._on_done(self)
