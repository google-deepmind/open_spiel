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


"""Servers run as child processes, each in its own process group.

The one-step launcher (`play`) starts the bot server and the GUI server this
way, copies their output with a label, waits for the line each prints once
it listens, and stops them however it ends: a stop interrupted by a second
signal kills the group rather than leave it running.
"""

from collections.abc import Mapping, Sequence
import os
import re
import signal
import subprocess
import threading
from typing import TextIO

# Seconds a server has to exit after SIGTERM before it is killed.
STOP_SECONDS = 10.0

# Seconds a child whose output has closed has to become reapable before it is
# taken to be running still. The kernel closes a dying child's descriptors
# before the child can be reaped, so its output ends a moment before
# `poll()` sees it exit.
REAP_SECONDS = 10.0


class LaunchError(Exception):
  """A server did not start, or the session could not be opened."""


class ChildServer:
  """A server started as a child process, in its own process group."""

  def __init__(
      self,
      label: str,
      argv: list[str],
      env: Mapping[str, str],
      ready: re.Pattern[str],
      out: TextIO,
  ) -> None:
    self.label = label
    self._ready = ready
    self._out = out
    self._url: str | None = None
    self._seen = threading.Event()
    self._eof = threading.Event()
    self.process = subprocess.Popen(  # pylint: disable=consider-using-with
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        env=dict(env),
        start_new_session=True,
    )
    try:
      self._pump = threading.Thread(target=self._read, daemon=True)
      self._pump.start()
    except BaseException:
      signal_group(self.process, signal.SIGKILL)
      raise

  def _read(self) -> None:
    """Copies the child's output and log to `out`; notes its ready line."""
    assert self.process.stdout is not None
    for line in self.process.stdout:
      self._out.write(f"[{self.label} server] {line}")
      self._out.flush()
      match = self._ready.search(line)
      if match and self._url is None:
        self._url = match.group(1).rstrip("/")
        self._seen.set()
    self._eof.set()
    self._seen.set()

  def wait_ready(self, seconds: float) -> str:
    """Returns the URL the child prints once it listens.

    Once the child's output ends without the ready line, the child is reaped
    before the message is chosen, which may take up to `REAP_SECONDS` more.

    Raises:
      LaunchError: If it exits before it listens; if it closes its output
        before it listens and is still running `REAP_SECONDS` later; or if
        it says nothing within `seconds`.
    """
    self._seen.wait(seconds)
    if self._url is not None:
      return self._url
    if self._eof.is_set():
      # Its output has ended, so it is exiting or has exited: reap it before
      # choosing the message, as a single poll() can come too early.
      self._pump.join(REAP_SECONDS)
      try:
        code = self.process.wait(REAP_SECONDS)
      except subprocess.TimeoutExpired:
        raise LaunchError(
            f"The {self.label} server closed its output before it listened "
            f"and is still running {REAP_SECONDS:g} s later."
        ) from None
      raise self._exited(code)
    # Its output is still open after `seconds`, long past any closing race.
    polled = self.process.poll()
    if polled is not None:
      raise self._exited(polled)
    raise LaunchError(
        f"The {self.label} server did not listen within {seconds:g} s."
    )

  def _exited(self, code: int) -> LaunchError:
    """The error for a child that exited with `code` before it listened."""
    return LaunchError(
        f"The {self.label} server exited with code {code} before it "
        "listened; its output is above."
    )

  def stop(self) -> int:
    """Stops the child's process group; returns its exit code.

    If the stop is interrupted (a second Ctrl-C, say), the group is killed
    before the interruption goes on.
    """
    try:
      if self.process.poll() is None:
        signal_group(self.process, signal.SIGTERM)
        try:
          self.process.wait(STOP_SECONDS)
        except subprocess.TimeoutExpired:
          signal_group(self.process, signal.SIGKILL)
          self.process.wait()
      else:
        # Whatever the child started in its group goes with it.
        signal_group(self.process, signal.SIGKILL)
    except BaseException:
      signal_group(self.process, signal.SIGKILL)
      raise
    self._pump.join(STOP_SECONDS)
    return self.process.returncode


def stop_all(children: Sequence[ChildServer | None]) -> None:
  """Stops every child in turn, even if stopping one is interrupted.

  Raises:
    BaseException: The first interruption or error, once every child is
      stopped.
  """
  first: BaseException | None = None
  for child in children:
    if child is None:
      continue
    try:
      child.stop()
    except BaseException as e:  # pylint: disable=broad-exception-caught
      if first is None:
        first = e
  if first is not None:
    raise first


def signal_group(process: subprocess.Popen[str], signum: int) -> None:
  """Sends `signum` to the process group `process` leads, if it exists."""
  try:
    os.killpg(process.pid, signum)
  except ProcessLookupError:
    pass  # The group has exited already.
