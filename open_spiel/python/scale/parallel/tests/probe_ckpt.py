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

"""Subprocess body of sharded_ckpt_test.

orbax save on one device count, restore on another.

  save <dir>     (8 devices) train.run kuhn MMD 4 updates data parallel with
                 checkpoints at 2 and 4; plus a row-sharded tree (f32 and f64
                 leaves, split 8 ways) through artifact.save_tree; reference
                 values to <dir>/ref.npz.
  restore <dir>  (any device count N dividing 64) the learner state restored
                 (run_state.restore, and DataParallel.restore_replicated
                 onto the N-device mesh), the artifact params (load_artifact)
                 and the row tree (DataParallel.restore_rows: device d holds
                 rows d*64/N..) equal the saved values exactly, dtypes kept
                 (int64 counter, f64 leaf); then train.run resumes the run on
                 N devices to update 6.
"""

import dataclasses
import os
import sys

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale import train
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.parallel import data_parallel as dpl
from open_spiel.python.scale.testing import multidevice

ROWS = 64


def config(num_updates):
  """The kuhn MMD config with num_updates updates, a checkpoint every 2.

  Args:
    num_updates: the run length.

  Returns:
    the run config.
  """
  cfg = config_common.with_overrides(
      config_common.load_config("mmd_kuhn_poker_jax"),
      num_envs=64,
      num_steps=16,
      num_updates=num_updates,
  )
  return dataclasses.replace(cfg, keep_every=2)


def flat(tree):
  """A pytree as {key path: host array}.

  Args:
    tree: a pytree of arrays.

  Returns:
    the flat dict.
  """
  return {
      jax.tree_util.keystr(k): multidevice.host(v)
      for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]
  }


def save(root):
  """The save step on 8 devices (module doc).

  Args:
    root: the output directory.
  """
  multidevice.expect_devices(8)
  dp = dpl.DataParallel(dpl.batch_mesh(8))
  cfg = config(4)
  res = train.run(cfg, os.path.join(root, "run"), mesh=dp.mesh)
  state = res.learner.state_dict(res.state)
  for leaf in jax.tree_util.tree_leaves(state):
    assert len(leaf.sharding.device_set) == 8, leaf.sharding
  rows = {
      "rows": np.arange(ROWS * 3, dtype=np.float32).reshape(ROWS, 3) * 0.5,
      "rows64": np.arange(ROWS, dtype=np.float64) + 2.0**-40,
  }
  sharded = dp.shard_rows(rows)
  assert len(sharded["rows64"].sharding.device_set) == 8
  artifact.save_tree(os.path.join(root, "rows"), sharded)
  ref = {"state" + k: v for k, v in flat(state).items()}
  ref.update({"rows" + k: v for k, v in flat(rows).items()})
  np.savez(os.path.join(root, "ref.npz"), **ref)
  print(f"saved on 8 devices: update_step {int(res.state.update_step)}")
  print(multidevice.OK)


def restore(root):
  """The restore step on this process's devices (module doc).

  Args:
    root: the directory save wrote.
  """
  n = len(jax.devices())
  dp = dpl.DataParallel(dpl.batch_mesh(n))
  ref = dict(np.load(os.path.join(root, "ref.npz")))
  cfg = config(4)
  parts = runner.one_slot_parts(cfg)
  env, policy = parts.env, parts.policy
  learner = pg.PGLearner(policy, cfg.learner.learner)
  ckpt = os.path.join(root, "run")
  learners = commit.learner_dir(ckpt, "main", 4)

  def check(tree, prefix, what):
    got = flat(tree)
    for k, v in got.items():
      want = ref[prefix + k]
      assert v.dtype == want.dtype and v.shape == want.shape, (
          what,
          k,
          v.dtype,
          want.dtype,
      )
      np.testing.assert_array_equal(v, want, err_msg=f"{what} {k}")

  state = run_state.restore(learners, 4, learner)
  check(learner.state_dict(state), "state", "run_state.restore")
  assert state.update_step.dtype == jnp.int64 and int(state.update_step) == 4

  template = learner.state_dict(learner.init(jax.random.key(0), env))
  blob = dp.restore_replicated(
      epath.Path(learners) / run_state.LEARNER_DIR / "4", template
  )
  check(blob, "state", "restore_replicated")
  for leaf in jax.tree_util.tree_leaves(blob):
    assert (
        leaf.sharding.is_fully_replicated and len(leaf.sharding.device_set) == n
    ), leaf.sharding

  art = artifact.load_artifact(ckpt, 4)
  got = flat(art.params)
  for k, v in got.items():
    np.testing.assert_array_equal(
        v, ref["state['params']" + k], err_msg=f"artifact {k}"
    )

  tmpl = {
      "rows": np.zeros((ROWS, 3), np.float32),
      "rows64": np.zeros((ROWS,), np.float64),
  }
  rows = dp.restore_rows(os.path.join(root, "rows"), tmpl)
  check(rows, "rows", "restore_rows")
  per = ROWS // n
  dev_index = {d: i for i, d in enumerate(dp.mesh.devices.flat)}
  for shard in rows["rows64"].addressable_shards:
    d = dev_index[shard.device]
    assert range(ROWS)[shard.index[0]] == range(d * per, (d + 1) * per), (
        shard.index,
        d,
    )
  assert rows["rows64"].dtype == jnp.float64

  res = train.run(config(6), ckpt, mesh=dp.mesh)
  assert int(res.state.update_step) == 6, int(res.state.update_step)
  ts = run_state.read_train_state(ckpt)
  assert ts["step"] == 6 and ts["frames"] == 6 * 64 * 16, ts
  assert len(res.metrics) == 2, len(res.metrics)
  print(
      f"restored on {n} devices: learner state, artifact params and row tree "
      "exact; "
      "resumed to update 6"
  )
  print(multidevice.OK)


if __name__ == "__main__":
  # A script: x64 on before any array is made (here, not at import, which
  # changes no process-global state).
  frameworks.enable_x64()
  {"save": save, "restore": restore}[sys.argv[1]](sys.argv[2])
