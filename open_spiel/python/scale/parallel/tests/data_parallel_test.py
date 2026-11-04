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

"""DataParallel contract tests.

in-process on one device, and the 8-device learner probe (probe_learner.py) in a
subprocess.
"""

import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.parallel import compiles
from open_spiel.python.scale.parallel import data_parallel as dpl
from open_spiel.python.scale.testing import multidevice


def test_check_divides():
  """check_divides passes even splits and refuses others."""
  dpl.check_divides(1024, 8, "run")
  dpl.check_divides(1024, 1, "run")
  with pytest.raises(
      ValueError, match="1000 rows do not split over 16 devices"
  ):
    dpl.check_divides(1000, 16, "x")


def test_batch_mesh_is_one_axis_named_batch():
  """batch_mesh is one axis named "batch"."""
  mesh = dpl.batch_mesh()
  assert mesh.axis_names == (dpl.AXIS,) and mesh.devices.size == len(
      jax.devices()
  )
  assert dpl.batch_mesh(1).devices.size == 1
  with pytest.raises(ValueError):
    dpl.batch_mesh(len(jax.devices()) + 1)
  with pytest.raises(ValueError):
    dpl.DataParallel(Mesh(np.array(jax.devices()[:1]), ("x",)))


def test_constrain_rows_is_identity_without_a_scope_and_on_one_device():
  """constrain_rows is the identity without a scope and on one device."""
  x = {"a": jnp.ones((4, 3)), "b": jnp.zeros((4,))}
  assert dpl.constrain_rows(x) is x
  dp = dpl.DataParallel(dpl.batch_mesh(1))
  with dp.scope():
    assert dpl.constrain_rows(x, axis=0) is x


def test_shard_rows_and_replicate_place_on_the_mesh():
  """shard_rows and replicate place leaves on the mesh."""
  dp = dpl.DataParallel(dpl.batch_mesh(1))
  y = dp.shard_rows({"r": np.arange(6, dtype=np.int64)})
  assert y["r"].dtype == jnp.int64 and y[
      "r"
  ].sharding.spec == dpl.PartitionSpec(dpl.AXIS)
  z = dp.replicate(jnp.float32(2.0))
  assert z.sharding.is_fully_replicated
  with pytest.raises(ValueError):
    dp.shard_rows(np.float32(1.0))  # a scalar has no row axis


def test_one_device_update_is_the_plain_update_bit_for_bit():
  """Known answer: on one device DataParallel adds nothing to the program."""
  cfg = config_common.with_overrides(
      config_common.load_config("mmd_kuhn_poker_jax"),
      num_envs=16,
      num_steps=8,
      num_updates=4,
  )
  dp = dpl.DataParallel(dpl.batch_mesh(1))
  parts = runner.one_slot_parts(cfg, mesh=dp.mesh)
  env, policy = parts.env, parts.policy
  learner = pg.PGLearner(policy, cfg.learner.learner)
  plain = pg.PGLearner(policy, cfg.learner.learner)
  s0 = learner.init(jax.random.key(0), env)
  src = arena_lib.ArenaSource(
      parts.arena, parts.slot_params(s0.params), parts.table, parts.num_steps
  )
  batch = src.next(jax.random.key(1))
  s_dp, m_dp = dp.update(learner, s0, batch, jax.random.key(2))
  s_pl, m_pl = plain.update(s0, batch, jax.random.key(2))
  multidevice.assert_trees_equal(s_dp, s_pl, "1-device DataParallel vs plain")
  assert m_dp == m_pl


def test_compile_counter_counts_backend_compiles():
  """The compile counter counts backend compiles."""
  f = jax.jit(lambda x: x * 3.0 + 1.0)
  x = jnp.arange(7, dtype=jnp.float32)
  with compiles.Counter() as c:
    f(x).block_until_ready()
  assert c.count == 1, c.count
  with compiles.assert_no_compiles("second call"):
    f(x).block_until_ready()
  with pytest.raises(AssertionError):
    with compiles.assert_no_compiles("new shape"):
      f(jnp.arange(9, dtype=jnp.float32)).block_until_ready()


def test_pg_data_parallel_on_8_devices():
  """PG data parallel on 8 devices equals 1 device (probe_learner pg)."""
  out = multidevice.run_probe(
      subprocess_env.module("parallel.tests.probe_learner"), 8, "pg"
  )
  for line in (
      "refusals: indivisible rows refused",
      "pg mmd_flat_kuhn: 8-device update == 1-device update",
      "pg ppo_env_axis_leduc: 8-device update == 1-device update",
      "hlo: all-reduce",
      "compile: iterations 2-5 compiled none",
  ):
    assert line in out, (line, out)


def test_az_and_bc_data_parallel_on_8_devices():
  """AlphaZero and BC data parallel on 8 devices (probe_learner az)."""
  out = multidevice.run_probe(
      subprocess_env.module("parallel.tests.probe_learner"), 8, "az"
  )
  assert "az: self-play counters equal" in out, out
  assert "bc: 2x2 data parallel == 1 device" in out, out


def test_training_on_8_devices_matches_1_device_statistically():
  """Training on 8 devices matches 1 device statistically."""
  out = multidevice.run_probe(
      subprocess_env.module("parallel.tests.probe_learner"), 8, "stat"
  )
  for seed in (0, 1, 2):
    assert f"stat seed {seed}:" in out, out


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
