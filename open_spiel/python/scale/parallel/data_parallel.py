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

"""Data parallelism over a 1-D "batch" mesh: the rollout and the learner update.

Training scales by splitting rows over devices. The Arena's rows and every
update's minibatch rows are split over one mesh axis, "batch", so the same
config runs on any device count and a checkpoint written on one count
resumes on any other.

  dp = data_parallel.DataParallel(data_parallel.batch_mesh())   # every device
  state = dp.replicate(learner.init(key, env))
  state, metrics = dp.update(learner, state, batch, key)

The layout:

  rollout   core.arena.Arena: one shard_map over "batch"; each device steps
            its own B/N rows, with no collective inside the rollout.
  update    DataParallel.update(learner, state, batch, key): the learner's own
            jitted update runs with the batch row-sharded over "batch" and the
            learner state and key replicated. Inside the update the learner
            pins each minibatch's row axis to "batch" (constrain_rows), so
            every device runs the forward and backward pass of its 1/N of
            every minibatch; the cross-device reductions (gradients, masked
            means, advantage normalisation, value-EMA moments) are all-reduces
            that the partitioner inserts where the single-device program
            reduces over rows. The optimizer step runs replicated on the
            reduced gradient.

What is equal and what is not:
  * the rollout at N devices equals the 1-device rollout bit for bit (every
    row draws from its own keys; core.arena);
  * the update at N devices computes the 1-device update's maths over the
    same minibatches (the shuffle is global); only the summation order of the
    reductions differs, so params agree to float rounding (about 1e-7
    relative), not bitwise. Learning across device counts is equal
    statistically.

Why GSPMD for the update and shard_map for the rollout: the rollout is a
long latency-bound loop that must not communicate, and shard_map makes that
a property of the program. The update must communicate, because its losses
reduce over the whole minibatch (masked means, advantage normalisation, the
advantage filter's quantile, the value EMA); writing those reductions as
explicit psums would need a second copy of every loss, while the
partitioner's all-reduce is the psum at the reductions the single-device
program already has.

Global batch semantics: num_envs (Arena rows) and minibatch sizes are global,
and N devices split them. Growing the batch with N is a config change
(num_envs x N), never implicit. Every row-sharded axis must be divisible by
N, or the placement raises ValueError.
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional, Sequence

import jax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import x64

AXIS = "batch"
tree_map = jax.tree_util.tree_map


def batch_mesh(
    num_devices: Optional[int] = None, devices: Optional[Sequence[Any]] = None
) -> Mesh:
  """The 1-D ("batch",) mesh over the first num_devices devices.

  Args:
    num_devices: the device count (default: every device).
    devices: the devices to take them from (default: jax.devices()).

  Returns:
    the Mesh.

  Raises:
    ValueError: fewer devices than num_devices are available.
  """
  devs = list(jax.devices() if devices is None else devices)
  n = len(devs) if num_devices is None else int(num_devices)
  if not 1 <= n <= len(devs):
    raise ValueError(f"num_devices={n}: {len(devs)} devices available")
  return Mesh(np.array(devs[:n]), (AXIS,))


def check_divides(rows: int, num_devices: int, what: str) -> None:
  """Raises ValueError unless `rows` split evenly over the devices.

  num_envs is global. Called before anything compiles.

  Args:
    rows: the global row count.
    num_devices: the device count.
    what: what the rows are, for the message.

  Raises:
    ValueError: rows is not divisible by num_devices.
  """
  if int(num_devices) < 1 or int(rows) % int(num_devices):
    raise ValueError(
        f"{what}: {rows} rows do not split over {num_devices} devices "
        "(num_envs "
        "is the global row count and must be divisible by the device count)"
    )


def active_batch_size() -> int:
  """Size of the "batch" axis of the DataParallel scope's mesh (1 if none).

  Returns:
    the device count of the active scope, 1 without one.
  """
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty or AXIS not in mesh.axis_names:
    return 1
  return int(mesh.shape[AXIS])


def constrain_rows(tree: Any, axis: int = 0) -> Any:
  """Pins `axis` of every leaf to the "batch" axis inside a traced update.

  When a DataParallel scope with more than one device is active, each device
  computes its 1/N of the rows; identity otherwise (one device, or no scope: the
  single-device program, unchanged). Raises ValueError when a leaf's `axis` is
  not divisible by the device count.

  pins `axis` of every leaf to the "batch" mesh axis when a DataParallel scope
  with more than one device is active, so each device computes its 1/N of the
  rows; identity otherwise (one device, or no scope: the single-device
  program, unchanged). Raises ValueError when a leaf's `axis` is not divisible
  by the device count.

  Args:
    tree: a pytree of arrays.
    axis: the row axis.

  Returns:
    the tree, constrained.
  """
  n = active_batch_size()
  if n == 1:
    return tree
  spec = PartitionSpec(*([None] * axis), AXIS)

  def one(x):
    """Constrains one leaf.

    Args:
      x: the leaf.

    Returns:
      the leaf with `axis` sharded over "batch".

    Raises:
      ValueError: the leaf's `axis` does not divide over the devices.
    """
    if x.ndim <= axis or x.shape[axis] % n:
      raise ValueError(
          f"data parallel over {n} devices: a leaf of shape {x.shape} has no "
          f"axis {axis} divisible by {n} (minibatch rows must split evenly)"
      )
    return jax.lax.with_sharding_constraint(x, spec)

  return tree_map(one, tree)


class DataParallel:
  """The one data-parallel driver: rows over "batch", state replicated.

  Attributes:
    mesh: the 1-D "batch" mesh.
    num_devices: the mesh's device count.
  """

  def __init__(self, mesh: Mesh):
    x64.assert_x64("DataParallel")
    assert isinstance(mesh, Mesh), type(mesh)
    if mesh.axis_names != (AXIS,):
      raise ValueError(
          f"DataParallel needs a 1-D ({AXIS!r},) mesh, got {mesh.axis_names}"
      )
    self.mesh = mesh
    self.num_devices = int(mesh.devices.size)
    self._replicated = NamedSharding(mesh, PartitionSpec())

  def rows(self, axis: int = 0) -> NamedSharding:
    """The sharding that splits `axis` over the devices.

    Args:
      axis: the row axis.

    Returns:
      a NamedSharding.
    """
    return NamedSharding(self.mesh, PartitionSpec(*([None] * axis), AXIS))

  def shard_rows(self, tree: Any, axis: int = 0) -> Any:
    """device_put every leaf with `axis` split over the devices.

    Raises unless divisible. Row r of a B-row axis lands on device r // (B/N).

    Args:
      tree: a pytree of arrays.
      axis: the row axis.

    Returns:
      the sharded tree.
    """
    sharding = self.rows(axis)

    def one(x):
      """Places one leaf.

      Args:
        x: the leaf.

      Returns:
        the leaf on the devices.

      Raises:
        ValueError: the leaf's `axis` does not divide over the devices.
      """
      shape = np.shape(x)
      if len(shape) <= axis or shape[axis] % self.num_devices:
        raise ValueError(
            f"cannot split axis {axis} of a {shape} leaf over "
            f"{self.num_devices} devices"
        )
      return jax.device_put(x, sharding)

    return tree_map(one, tree)

  def replicate(self, tree: Any) -> Any:
    """Every leaf replicated on every device.

    Args:
      tree: a pytree of arrays.

    Returns:
      the replicated tree.
    """
    return jax.device_put(tree, self._replicated)

  @contextlib.contextmanager
  def scope(self):
    """Sets the mesh for tracing.

    learner updates traced inside it pin their minibatch rows to "batch"
    (constrain_rows). A jitted loop that calls learner.update (the AlphaZero
    learner's epoch scan, learners.az) must be called inside it.

    Yields:
      this DataParallel.
    """
    with jax.set_mesh(self.mesh):
      yield self

  def update(self, learner: Any, state: Any, batch: Any, key: Any):
    """learner.update(state, batch, key), data parallel.

    batch rows sharded, state and key replicated. Returns what
    learner.update returns (the new state stays replicated).

    Args:
      learner: the learner.
      state: the learner state.
      batch: the TrajectoryBatch.
      key: the update key.

    Returns:
      learner.update's (state, metrics).
    """
    with self.scope():
      return learner.update(
          self.replicate(state), self.shard_rows(batch), self.replicate(key)
      )

  def restore_replicated(self, path: Any, template: Any) -> Any:
    """artifact.restore_tree onto this mesh, every leaf replicated.

    Whatever device count wrote it (orbax reshards to the template's sharding).

    Args:
      path: the saved tree's directory.
      template: a pytree with the saved tree's structure, shapes and dtypes.

    Returns:
      the restored tree.
    """
    return artifact.restore_tree(path, self.replicate(template))

  def restore_rows(self, path: Any, template: Any, axis: int = 0) -> Any:
    """artifact.restore_tree onto this mesh with `axis` row-sharded.

    Args:
      path: the saved tree's directory.
      template: a pytree with the saved tree's structure, shapes and dtypes.
      axis: the row axis.

    Returns:
      the restored tree.
    """
    return artifact.restore_tree(path, self.shard_rows(template, axis))
