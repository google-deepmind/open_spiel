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

"""Counts XLA backend compiles (the "no recompiles across iterations" check).

A jitted function's `_cache_size()` is not a compile count: the C++ dispatch
cache may add an entry for an argument it has not seen in that exact form
(e.g. an array produced by a different program) without tracing or compiling
anything. The compile count here comes from JAX's own monitoring event, fired
once per backend compile.
"""

import contextlib
import threading

from jax import monitoring

BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"

_lock = threading.Lock()
_count = [0]


def _listener(event: str, duration_secs: float, **kwargs) -> None:
  del duration_secs, kwargs
  if event == BACKEND_COMPILE_EVENT:
    with _lock:
      _count[0] += 1


monitoring.register_event_duration_secs_listener(_listener)


def total() -> int:
  """Backend compiles in this process since this module was imported.

  Returns:
    the count.
  """
  with _lock:
    return _count[0]


class Counter:
  """`with Counter() as c: ...; c.count` = backend compiles inside the block.

  Attributes:
    start: the backend compile total at entry.
    count: the compiles inside the block (0 until the block ends).
  """

  def __init__(self):
    self.start = None
    self.count = None

  def __enter__(self):
    self.start = total()
    self.count = 0
    return self

  def __exit__(self, *exc):
    self.count = total() - self.start
    return False


@contextlib.contextmanager
def assert_no_compiles(what: str):
  """Raises AssertionError if anything compiles inside the block.

  Args:
    what: the block, for the message.

  Yields:
    the block's Counter.

  Raises:
    AssertionError: the block compiled.
  """
  with Counter() as c:
    yield c
  if c.count:
    raise AssertionError(
        f"{what}: {c.count} XLA compile(s) where none was expected"
    )
