# Copyright 2019 DeepMind Technologies Limited
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

"""A wrapper around multiprocessing to be compatible at google."""

import contextlib
import multiprocessing
import queue

Empty = queue.Empty

# Without this line, this fails on latest MacOS with Python 3.8. See
# https://github.com/pytest-dev/pytest-flask/issues/104#issuecomment-577908228
# and for more details see
# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
multiprocessing.set_start_method("fork")


# For compatibility so that it works inside Google.
@contextlib.contextmanager
def main_handler():
  yield


class Process(object):
  """A wrapper around `multiprocessing` that allows it to be used at google.

  It spawns a subprocess from the given target function. That function should
  take an additional argument `queue` which will get a bidirectional
  _ProcessQueue for communicating with the parent.
  """

  def __init__(self, target, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}
    elif "queue" in kwargs:
      raise ValueError("`queue` is reserved for use by `Process`.")

    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    self._queue = _ProcessQueue(q1, q2)
    kwargs["queue"] = _ProcessQueue(q2, q1)

    self._process = multiprocessing.Process(
        target=target, args=args, kwargs=kwargs)
    self._process.start()

  def join(self, *args):
    return self._process.join(*args)

  @property
  def exitcode(self):
    return self._process.exitcode

  @property
  def queue(self):
    return self._queue


class _ProcessQueue(object):
  """A bidirectional queue for talking to a subprocess.

  `empty`, `get` and `get_nowait` act on the incoming queue, while
  `full`, `put` and `put_nowait` act on the outgoing queue.

  This class should only be created by the Process object.
  """

  def __init__(self, q_in, q_out):
    self._q_in = q_in
    self._q_out = q_out

  def empty(self):
    return self._q_in.empty()

  def full(self):
    return self._q_out.full()

  def get(self, block=True, timeout=None):
    return self._q_in.get(block=block, timeout=timeout)

  def get_nowait(self):
    return self.get(False)

  def put(self, obj, block=True, timeout=None):
    return self._q_out.put(obj, block=block, timeout=timeout)

  def put_nowait(self, obj):
    return self.put(obj, False)
