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

"""Subprocess body for arena_sharding_test: XLA_FLAGS set before jax loads.

Runs the same collect on a 4-device mesh and on a 1-device mesh and asserts
the batch axis is sharded over 4 devices and the results are identical.
"""

import os

from absl import app
from absl import logging
import jax
from jax.sharding import Mesh
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.testing import fixtures


def _require_four_host_devices() -> None:
  """Fails unless XLA_FLAGS forces 4 host devices (read at backend start).

  Raises:
    AssertionError: XLA_FLAGS does not force 4 host devices.
  """
  assert "--xla_force_host_platform_device_count=4" in os.environ.get(
      "XLA_FLAGS", ""
  ), "set XLA_FLAGS=--xla_force_host_platform_device_count=4 before running"


def main(argv):
  """Checks the 4-device collect against the 1-device collect.

  Args:
    argv: unused.

  Raises:
    AssertionError: a batch that does not split over 4 devices was accepted.
  """
  del argv
  devices = jax.devices()
  assert len(devices) == 4, devices
  env = kuhn.KuhnEnv()
  pol = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  params = [
      jax.tree_util.tree_map(
          lambda x: x[None], pol.make_params(b=np.array([0.0, 0.7], np.float32))
      )
  ]
  b, t = 64, 16
  table = arena_lib.self_play_table(b, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(0), b, 2)

  results = {}
  for name, devs in (("four", devices), ("one", devices[:1])):
    ar = arena_lib.Arena(
        env,
        [arena_lib.Slot(pol, 1, types.SAMPLE)],
        mesh=Mesh(np.array(devs), ("batch",)),
    )
    state, batch, log = ar.collect(params, table, dk, pk, t)
    results[name] = (batch, log)
    n_dev = len(batch.action.sharding.device_set)
    assert n_dev == len(devs), (name, batch.action.sharding)
    assert len(state.env_state.p0_card.sharding.device_set) == len(devs)
    logging.info(
        "%s: action sharded over %d devices: %s",
        name,
        n_dev,
        batch.action.sharding,
    )
  for x, y in zip(
      jax.tree_util.tree_leaves(results["four"]),
      jax.tree_util.tree_leaves(results["one"]),
  ):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  # Batch not divisible by the device count fails loud.
  ar4 = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  try:
    ar4.validate(params, arena_lib.self_play_table(6, 2))
  except ValueError:
    pass
  else:
    raise AssertionError("batch 6 on 4 devices was accepted")
  logging.info("SHARDING_PROBE_OK")


if __name__ == "__main__":
  # A script: checked and set here, before any JAX backend starts or array is
  # made, never at import.
  _require_four_host_devices()
  frameworks.enable_x64()
  app.run(main)
