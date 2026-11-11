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

"""Run storage: the text objects a supervisor and its attempts share.

A preemption-safe run keeps its state where every attempt can reach it, not
on the machine that launched it. A Backend reads and writes that state
through a Storage: a local directory for local runs, a GCS bucket for cloud
runs, or anything else that implements four methods.

  store = storage.LocalStorage()
  uri = store.join("/data/runs", "kuhn-a", "supervisor.json")
  store.write_text(uri, "{}")
  store.read_text(uri)               # "{}"; None for a missing object

Writes replace the whole object. LocalStorage writes atomically (a temporary
file, then rename); GcsStorage relies on GCS's whole-object writes.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import time
from typing import Any, Callable, Optional, Protocol, Sequence

from absl import logging

# A storage call is tried this many times (backoff 2, 4, 8, 16 s) before it
# raises.
IO_TRIES = 5
# What `gcloud storage` prints for a missing object; any other error (a
# missing bucket, a permission) is not read as a missing object.
NOT_FOUND = ("matched no objects", "No URLs matched")

Runner = Callable[..., subprocess.CompletedProcess[str]]


class Storage(Protocol):
  """Text objects addressed by URI."""

  def read_text(self, uri: str) -> Optional[str]:
    """The object's text, or None when it does not exist.

    Args:
      uri: the object.

    Returns:
      its text, or None.
    """

  def write_text(self, uri: str, text: str) -> None:
    """Creates or replaces the object.

    Args:
      uri: the object.
      text: its text.
    """

  def exists(self, uri: str) -> bool:
    """Whether the object exists.

    Args:
      uri: the object.

    Returns:
      True when it exists.
    """

  def join(self, *parts: str) -> str:
    """A URI from a root and path parts.

    Args:
      *parts: the root, then path parts.

    Returns:
      the URI.
    """


def join(root: str, *parts: str) -> str:
  """root/part/part with single slashes (works for paths and gs:// URIs).

  Args:
    root: a path or gs:// URI.
    *parts: path parts.

  Returns:
    the joined URI.
  """
  out = str(root).rstrip("/")
  for p in parts:
    out = f"{out}/{str(p).strip('/')}"
  return out


class LocalStorage:
  """Storage on the local filesystem; URIs are paths."""

  def read_text(self, uri: str) -> Optional[str]:
    """The file's text, or None when it does not exist.

    Args:
      uri: the file.

    Returns:
      its text, or None.
    """
    if not os.path.exists(uri):
      return None
    with open(uri, encoding="utf-8") as f:
      return f.read()

  def write_text(self, uri: str, text: str) -> None:
    """Writes the file atomically (a temp file, then a rename).

    Args:
      uri: the file.
      text: its text.

    Raises:
      BaseException: whatever the write raised, after the temp file is removed.
    """
    parent = os.path.dirname(uri) or "."
    os.makedirs(parent, exist_ok=True)
    # A unique temporary file per write: two writers of one object never share
    # it.
    tmp = f"{uri}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    try:
      with open(tmp, "x", encoding="utf-8") as f:
        f.write(text)
      os.replace(tmp, uri)
    except BaseException:
      if os.path.exists(tmp):
        os.remove(tmp)
      raise

  def exists(self, uri: str) -> bool:
    """Whether the file exists.

    Args:
      uri: the file.

    Returns:
      True when it exists.
    """
    return os.path.exists(uri)

  def join(self, *parts: str) -> str:
    """A path from a root and path parts.

    Args:
      *parts: the root, then path parts.

    Returns:
      the path.
    """
    return join(*parts)


def run_command(
    argv: Sequence[str],
    *,
    # `input` is the keyword the callers pass (as subprocess.run's).
    input: Optional[str] = None,  # pylint: disable=redefined-builtin
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
  """Argv's completed process, output captured as text.

  Never raises on a nonzero exit.

  Args:
    argv: the command.
    input: text for its stdin.
    timeout: seconds before it is killed.

  Returns:
    the CompletedProcess.
  """
  return subprocess.run(
      list(argv),
      input=input,
      capture_output=True,
      text=True,
      timeout=timeout,
      check=False,
  )


def retried(
    call: Callable[[], subprocess.CompletedProcess[str]],
    what: str,
    *,
    tries: int = IO_TRIES,
    sleep: Callable[[float], Any] = time.sleep,
    not_found_ok: bool = False,
) -> subprocess.CompletedProcess[str]:
  """Calls call() until it succeeds, with exponential backoff.

  call() until it succeeds (or, with not_found_ok, reports a missing
  object), waiting 2, 4, 8, ... s between tries; a timeout counts as a
  failed try.

  A transient storage error does not end a run; a persistent one does.

  Args:
    call: the call, returning a CompletedProcess.
    what: the call, for messages.
    tries: the most tries.
    sleep: the sleep between tries.
    not_found_ok: a missing object counts as success.

  Returns:
    the successful CompletedProcess.

  Raises:
    RuntimeError: after `tries` failures, with the last error.
  """
  assert tries >= 1, tries
  last = ""
  for i in range(tries):
    try:
      p = call()
    except subprocess.TimeoutExpired as e:
      last = f"timed out after {e.timeout}s"
    else:
      if p.returncode == 0 or (
          not_found_ok and any(x in p.stderr for x in NOT_FOUND)
      ):
        return p
      last = f"exit {p.returncode}: {p.stderr.strip()[-1500:]}"
    if i + 1 < tries:
      logging.warning(
          "storage: %s failed (try %d/%d): %s", what, i + 1, tries, last[-300:]
      )
      sleep(2.0 * 2**i)
  raise RuntimeError(f"{what} failed {tries}x: {last}")


class GcsStorage:
  """Storage on gs:// through `gcloud storage`.

  Storage on gs:// through `gcloud storage` (cat, cp -), with no client
  library. Every call is retried (retried) before it raises.

  Attributes:
    tries: attempts per call before it raises.
    sleep: the sleep between attempts.
  """

  def __init__(
      self,
      runner: Runner = run_command,
      *,
      tries: int = IO_TRIES,
      sleep: Callable[[float], Any] = time.sleep,
  ):
    self._runner = runner
    self.tries, self.sleep = int(tries), sleep

  def read_text(self, uri: str) -> Optional[str]:
    """The object's text, or None when it does not exist.

    Args:
      uri: the gs:// object.

    Returns:
      its text, or None.
    """
    p = retried(
        lambda: self._runner(["gcloud", "storage", "cat", uri]),
        f"gcloud storage cat {uri}",
        tries=self.tries,
        sleep=self.sleep,
        not_found_ok=True,
    )
    return None if p.returncode != 0 else p.stdout

  def write_text(self, uri: str, text: str) -> None:
    """Creates or replaces the object.

    Args:
      uri: the gs:// object.
      text: its text.
    """
    retried(
        lambda: self._runner(
            ["gcloud", "storage", "cp", "-", uri, "--quiet"], input=text
        ),
        f"gcloud storage cp - {uri}",
        tries=self.tries,
        sleep=self.sleep,
    )

  def exists(self, uri: str) -> bool:
    """Whether the object exists.

    Args:
      uri: the gs:// object.

    Returns:
      True when it exists.
    """
    return self.read_text(uri) is not None

  def join(self, *parts: str) -> str:
    """A gs:// URI from a root and path parts.

    Args:
      *parts: the root, then path parts.

    Returns:
      the URI.
    """
    return join(*parts)
