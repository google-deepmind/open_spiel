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

"""The preemption notice a training run checkpoints on.

Spot and preemptible machines give a short grace period before they stop.
A run that notices in time writes a resume checkpoint and exits with
PREEMPTED_EXIT, and the next attempt continues from it instead of from the
last immutable checkpoint.

  notice = preemption.Notice()
  stop = preemption.start_notice_sources(notice, preemption.default_sources())
  try:
    ...  # the training loop reads notice() after every iteration
  finally:
    stop()

Notice sources set the notice. A source is registered by name with
register_notice_source(name, start), where start(notice) begins watching and
returns the function that stops it. Built in:

  sigterm   SIGTERM sets the notice instead of killing the process (main
            thread only; the previous handler comes back on stop)
  gce       the Compute Engine metadata server's preemption flag
            (launch.gcp.watch_preemption); off GCE it stops and says so

The default sources are ("sigterm",); set_default_sources changes them for a
deployment. league.runner.run reads the notice after every iteration and
before every evaluation: the evaluations still due are skipped, a resume
checkpoint is written and the run returns preempted; the trainer then exits
PREEMPTED_EXIT (128 + SIGTERM), never 0, so a supervisor does not read a
preempted run as done.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from typing import Callable, Optional, Sequence

from absl import logging

PREEMPTED_EXIT = 128 + int(signal.SIGTERM)  # 143

#: start(notice) -> stop(): begins watching for a preemption, returns its stop.
NoticeSource = Callable[["Notice"], Callable[[], None]]
_SOURCES: dict[str, NoticeSource] = {}
_DEFAULT: list[str] = ["sigterm"]


class Notice:
  """A thread-safe, set-once preemption flag; call it to read it.

  Attributes:
    reason: the first reason the flag was set with, or None while unset.
  """

  def __init__(self):
    self._ev = threading.Event()
    self._lock = threading.Lock()
    self.reason: Optional[str] = None

  def set(self, reason: str) -> None:
    """Sets the flag; the first reason given is kept.

    Args:
      reason: why (for example "sigterm").
    """
    with self._lock:
      if self.reason is None:
        self.reason = str(reason)
    self._ev.set()

  def __call__(self) -> bool:
    """Whether the notice is set."""
    return self._ev.is_set()


@contextlib.contextmanager
def on_sigterm(notice: Notice):
  """Within the block, SIGTERM sets `notice` (logged) instead of exiting.

  Args:
    notice: the Notice to set.

  Yields:
    None, with the handler installed.
  """
  stop = _start_sigterm(notice)
  try:
    yield notice
  finally:
    stop()


def _start_sigterm(notice: Notice) -> Callable[[], None]:
  """Sets `notice` on SIGTERM.

  Args:
    notice: The notice to set.

  Returns:
    A function restoring the previous handler.
  """

  def handler(signum, frame):
    del frame
    notice.set("SIGTERM")
    logging.warning(
        "preemption: signal %d: checkpointing after the iteration in flight",
        signum,
    )

  old = signal.signal(signal.SIGTERM, handler)
  return lambda: signal.signal(signal.SIGTERM, old)


def _start_gce(notice: Notice) -> Callable[[], None]:
  """Sets `notice` when the GCE metadata server announces a preemption.

  Args:
    notice: The notice to set.

  Returns:
    A function stopping the watcher.
  """
  # Lazy: a cycle (gcp imports backend, which imports this module).
  from open_spiel.python.scale.launch import gcp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  th = gcp.watch_preemption(notice)
  return th.stop


def register_notice_source(name: str, start: NoticeSource) -> None:
  """Registers a notice source under `name` (module doc).

  Registering a name twice raises ValueError.

  Args:
    name: the source's name.
    start: start(notice) -> stop function.

  Raises:
    ValueError: the name is empty or already registered.
  """
  if not isinstance(name, str) or not name or "," in name:
    raise ValueError(
        f"notice source name {name!r}: want a non-empty name without commas"
    )
  if name in _SOURCES:
    raise ValueError(f"notice source {name!r} is already registered")
  _SOURCES[name] = start


def sources() -> tuple[str, ...]:
  """The registered notice sources, sorted.

  Returns:
    the names.
  """
  return tuple(sorted(_SOURCES))


def set_default_sources(names: Sequence[str]) -> None:
  """The sources default_sources() returns; each must be registered.

  Args:
    names: registered source names.

  Raises:
    KeyError: a name is not registered.
  """
  names = list(names)
  unknown = [n for n in names if n not in _SOURCES]
  if unknown:
    raise KeyError(
        f"unknown notice sources {unknown}; registered: {list(sources())}"
    )
  _DEFAULT[:] = names


def default_sources() -> tuple[str, ...]:
  """The notice sources a run starts when it names none.

  Returns:
    the names.
  """
  return tuple(_DEFAULT)


def start_notice_sources(
    notice: Notice, names: Sequence[str]
) -> Callable[[], None]:
  """Starts the named sources on `notice`.

  Returns the function that stops them all, in reverse order.

  Args:
    notice: the Notice the sources set.
    names: registered source names.

  Returns:
    the stop function.

  Raises:
    KeyError: an unregistered name (nothing is started).
    BaseException: whatever a source's start raised, after the sources already
        started are stopped.
  """
  names = list(names)
  unknown = [n for n in names if n not in _SOURCES]
  if unknown:
    raise KeyError(
        f"unknown notice sources {unknown}; registered: {list(sources())}"
    )
  stops = []
  try:
    for n in names:
      stops.append(_SOURCES[n](notice))
  except BaseException:
    for stop in reversed(stops):
      stop()
    raise

  def stop_all() -> None:
    for stop in reversed(stops):
      stop()

  return stop_all


register_notice_source("sigterm", _start_sigterm)
register_notice_source("gce", _start_gce)
