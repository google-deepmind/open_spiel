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

"""ReplayBuffer: a trajectory source over the last K iterations' batches.

    buf = ReplayBuffer(capacity_iterations=4, batch_size=256)
    buf.add(batch)                 # one iteration's TrajectoryBatch [B_i, T]
    sample = buf.next(key)         # batch_size rows, uniform over retained rows

Search-based training (AlphaZero) trains on the targets of several recent
iterations, not only the latest. ReplayBuffer keeps the last
`capacity_iterations` batches and samples rows from them.

`add` appends one batch (B_i may vary; T, tree structure, trailing shapes
and dtypes must match the first add); once K batches are held the oldest is
dropped whole. `next(key)` draws batch_size rows uniformly over the retained
rows, without replacement by default, deterministically under the key. Rows
are whole [T] sequences with their init_carry rows, so a recurrent learner
can consume them.

iterations_added is an np.int64 counter. Storage is device arrays,
concatenated lazily once per change.
"""

import collections
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types


def _signature(batch: types.TrajectoryBatch):
  """A batch's (T, treedef, per-leaf trailing shape and dtype).

  Args:
    batch: The batch.

  Returns:
    The signature.
  """
  t = batch.valid.shape[1]
  leaves, treedef = jax.tree_util.tree_flatten(batch)
  return (
      t,
      treedef,
      [(tuple(np.shape(x))[1:], np.dtype(x.dtype)) for x in leaves],
  )


class ReplayBuffer:
  """FIFO over the last capacity_iterations batches (module doc).

  Attributes:
    capacity_iterations: the batches retained.
    batch_size: rows per `next`.
    replace: whether `next` draws rows with replacement.
    iterations_added: batches added so far (np.int64).
  """

  def __init__(
      self, capacity_iterations: int, batch_size: int, replace: bool = False
  ):
    """Builds an empty buffer.

    Args:
      capacity_iterations: the batches retained (>= 1).
      batch_size: rows per `next` (>= 1).
      replace: draw rows with replacement.
    """
    assert int(capacity_iterations) >= 1, capacity_iterations
    assert int(batch_size) >= 1, batch_size
    self.capacity_iterations = int(capacity_iterations)
    self.batch_size = int(batch_size)
    self.replace = bool(replace)
    self._items: collections.deque[types.TrajectoryBatch] = collections.deque()
    self._sig = None
    self._cat: Optional[types.TrajectoryBatch] = None
    self.iterations_added = np.int64(0)

  @property
  def num_iterations(self) -> int:
    """The number of retained adds.

    Returns:
      The count.
    """
    return len(self._items)

  @property
  def num_rows(self) -> int:
    """The number of retained rows.

    Returns:
      The count.
    """
    return sum(int(b.valid.shape[0]) for b in self._items)

  def items(self) -> tuple[types.TrajectoryBatch, ...]:
    """The retained batches, oldest first.

    Returns:
      The batches.
    """
    return tuple(self._items)

  def clear(self) -> None:
    """Drops every retained batch.

    The batch layout and iterations_added stay: the next add must match the
    layout, and the counter goes on.
    """
    self._items.clear()
    self._cat = None

  def restore_iterations_added(self, n: int) -> None:
    """Sets iterations_added to a restored count n (>= the batches held).

    Args:
      n: the restored count.

    Raises:
      ValueError: n is below the number of batches held.
    """
    n = np.int64(n)
    if n < len(self._items):
      raise ValueError(
          f"iterations_added {int(n)} < the {len(self._items)} batches held"
      )
    self.iterations_added = n

  def add(self, batch: types.TrajectoryBatch) -> None:
    """Retains a batch as the newest add, dropping the oldest past capacity.

    Args:
      batch: a TrajectoryBatch with the buffer's layout.
    """
    assert isinstance(batch, types.TrajectoryBatch), type(batch)
    sig = _signature(batch)
    if self._sig is None:
      self._sig = sig
    else:
      assert sig == self._sig, (
          "batch layout (T, tree, trailing shapes, dtypes) differs from the "
          "buffer's: "
          f"{sig} vs {self._sig}"
      )
    self._items.append(batch)
    while len(self._items) > self.capacity_iterations:
      self._items.popleft()
    self._cat = None
    self.iterations_added = np.int64(self.iterations_added + np.int64(1))

  def contents(self) -> types.TrajectoryBatch:
    """Every retained row, oldest iteration first.

    Returns:
      The concatenated rows.
    """
    assert self._items, "replay buffer is empty"
    if self._cat is None:
      self._cat = jax.tree.map(
          lambda *xs: jnp.concatenate(xs, axis=0), *self._items
      )
    return self._cat

  def next(self, key) -> types.TrajectoryBatch:
    """batch_size uniform rows of the retained iterations.

    Args:
      key: the PRNG key of the draw.

    Returns:
      The TrajectoryBatch of the rows.
    """
    data = self.contents()
    n = int(data.valid.shape[0])
    assert (
        self.replace or self.batch_size <= n
    ), f"{n} rows < batch_size {self.batch_size} (sampling without replacement)"
    idx = jax.random.choice(key, n, (self.batch_size,), replace=self.replace)
    return jax.tree.map(lambda x: x[idx], data)
