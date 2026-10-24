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

"""Subprocess probe: orbax round-trips params sharded over 4 devices.

Subprocess probe (4 forced host devices): params sharded over a 4-device
mesh are written by orbax and restored with the same values and dtypes.
"""

import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.testing import fixtures


def main(tmp):
  """Writes and restores sharded params, asserting values and dtypes.

  Args:
    tmp: the scratch directory to write the artifact under.
  """
  assert len(jax.devices()) == 4, jax.devices()
  mesh = Mesh(np.array(jax.devices()), ("batch",))
  pol = fixtures.DtypeProbePolicy(
      "kuhn_poker", 2, 2, w64_dtype="float64", n32=8, n64=8
  )
  params = pol.make_params(
      w32=np.arange(8, dtype=np.float32),
      w64=np.arange(8, dtype=np.float64) + 2.0**-40,
      count=np.int64(2**33),
  )
  sharded = jax.tree_util.tree_map(
      lambda x: jax.device_put(x, NamedSharding(mesh, PartitionSpec("batch")))
      if x.ndim
      else jax.device_put(x, NamedSharding(mesh, PartitionSpec())),
      params,
  )
  assert len(sharded["w64"].sharding.device_set) == 4
  artifact.write_artifact(
      tmp,
      5,
      pol,
      sharded,
      rules_sha="r",
      config_hash="c",
      algo="t",
      env_name="kuhn_poker",
  )
  got = artifact.load_artifact(tmp, 5).params
  for k, want in params.items():
    assert got[k].dtype == want.dtype, (k, got[k].dtype, want.dtype)
    np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(want))
  assert got["w64"].dtype == jnp.float64
  print("CKPT_SHARDED_OK")


if __name__ == "__main__":
  # A script: x64 on before any array is made (not at import, which changes
  # no process-global state).
  frameworks.enable_x64()
  with fixtures.registered():
    main(sys.argv[1])
