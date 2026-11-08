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

"""load_artifact and load_manifest beyond artifact_test.

Pins: round trips are bit-exact (NaN payloads, -0.0 and denormals survive;
a NetworkPolicy's params restore exactly); a missing manifest raises;
parents and lineage are mandatory together; load_manifest equals the
manifest write_artifact returned; gs:// path strings keep their scheme.
"""

import json
import sys

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.networks.tests import helpers
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.testing import fixtures

_W = dict(
    rules_sha="r" * 8, config_hash="c" * 8, algo="test", env_name="kuhn_poker"
)

# float32 / float64 bit patterns a lossy path would change: quiet NaN with a
# payload, negative NaN, -0.0, the smallest denormal, max, an ordinary value.
_F32_BITS = np.array(
    [0x7FC00001, 0xFFC00123, 0x80000000, 0x00000001, 0x7F7FFFFF, 0x3F9DF3B6],
    np.uint32,
)
_F64_BITS = np.array(
    [
        0x7FF8000000000ABC,
        0x8000000000000000,
        0x0000000000000001,
        0x3FF0000000000001,
    ],
    np.uint64,
)


def _bits(x):
  """The bit pattern of `x` as unsigned integers.

  Args:
    x: A float array.

  Returns:
    The uint view of `x`.
  """
  x = np.asarray(x)
  return x.view({4: np.uint32, 8: np.uint64}[x.dtype.itemsize])


def test_round_trip_is_bit_exact(tmp_path):
  """Round trip is bit exact."""
  pol = fixtures.DtypeProbePolicy("kuhn_poker", 2, 2, n32=6, n64=4)
  params = pol.make_params(
      w32=_F32_BITS.view(np.float32),
      w64=_F64_BITS.view(np.float64),
      count=np.int64(2**62 + 1),
  )
  artifact.write_artifact(str(tmp_path), 5, pol, params, **_W)
  got = artifact.load_artifact(str(tmp_path), 5).params
  np.testing.assert_array_equal(_bits(got["w32"]), _F32_BITS)
  np.testing.assert_array_equal(_bits(got["w64"]), _F64_BITS)
  assert got["count"].dtype == jnp.int64 and int(got["count"]) == 2**62 + 1


def _kuhn_mlp():
  """A small Kuhn MLP network policy.

  Returns:
    The policy.
  """
  net = {
      "compute_dtype": "float32",
      "torso": {
          "kind": "mlp",
          "widths": [16, 16],
          "layout": "shared",
          "activation": "relu",
      },
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
  return policy_lib.NetworkPolicy(
      "kuhn_poker", "kuhn_onehot", 2, 2, helpers.KUHN_ONEHOT, [12], net
  )


def test_network_policy_round_trip_is_bit_exact(tmp_path):
  """Network policy round trip is bit exact."""
  pol = _kuhn_mlp()
  p0 = pol.init_params(jax.random.key(3))
  leaves, treedef = jax.tree.flatten(p0)
  keys = jax.random.split(jax.random.key(4), len(leaves))
  params = jax.tree.unflatten(
      treedef,
      [
          x + jax.random.normal(k, x.shape, x.dtype)
          for x, k in zip(leaves, keys)
      ],
  )
  artifact.write_artifact(str(tmp_path), 11, pol, params, **_W)
  art = artifact.load_artifact(str(tmp_path), 11)
  assert art.policy.spec == pol.spec
  assert jax.tree.structure(art.params) == jax.tree.structure(params)
  for a, b in zip(jax.tree.leaves(art.params), jax.tree.leaves(params)):
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(_bits(a), _bits(b))


def _tabular():
  """A Kuhn tabular policy and its params.

  Returns:
    (policy, params).
  """
  index = tabular.__name__.rpartition(".policies.")[0] + ".envs.kuhn:info_index"
  pol = tabular.TabularPolicy("kuhn_poker", 12, 2, 2, index)
  return pol, pol.params_from_logits(jnp.arange(24.0).reshape(12, 2))


def test_missing_manifest_raises(tmp_path):
  """Missing manifest raises."""
  with pytest.raises(FileNotFoundError):
    artifact.load_artifact(str(tmp_path), 7)
  pol, params = _tabular()
  artifact.write_artifact(str(tmp_path), 7, pol, params, **_W)
  (tmp_path / "7" / artifact.MANIFEST_FILENAME).unlink()
  with pytest.raises(FileNotFoundError):
    artifact.load_artifact(str(tmp_path), 7)
  with pytest.raises(FileNotFoundError):
    artifact.load_manifest(str(tmp_path), 7)


def test_lineage_is_mandatory_on_warm_start(tmp_path):
  """Lineage is mandatory on warm start."""
  pol, params = _tabular()
  with pytest.raises(artifact.ArtifactError, match="lineage"):
    artifact.write_artifact(
        str(tmp_path / "a"), 1, pol, params, parents=("9@100",), **_W
    )
  with pytest.raises(artifact.ArtifactError, match="parents"):
    artifact.write_artifact(
        str(tmp_path / "b"),
        1,
        pol,
        params,
        lineage=artifact.lineage("gs://b/runs/9", 100),
        **_W
    )
  with pytest.raises(artifact.ArtifactError, match="lineage"):
    artifact.write_artifact(
        str(tmp_path / "c"),
        1,
        pol,
        params,
        parents=("9@100",),
        lineage=dict(source="gs://b/runs/9"),
        **_W
    )
  lin = artifact.lineage("gs://b/runs/9", 100, run="9")
  man = artifact.write_artifact(
      str(tmp_path / "d"), 1, pol, params, parents=("9@100",), lineage=lin, **_W
  )
  assert man.parents == ("9@100",) and man.provenance["lineage"] == lin
  path = tmp_path / "d" / "1" / artifact.MANIFEST_FILENAME
  m = json.loads(path.read_text())
  assert m["provenance"]["lineage"] == lin
  m["provenance"][
      "lineage"
  ] = None  # a hand-edited manifest that drops the lineage
  path.write_text(json.dumps(m))
  with pytest.raises(artifact.ArtifactError, match="lineage"):
    artifact.load_artifact(str(tmp_path / "d"), 1)
  written = artifact.write_artifact(str(tmp_path / "e"), 1, pol, params, **_W)
  assert not written.parents


def test_load_manifest_equals_the_written_manifest(tmp_path):
  """Load manifest equals the written manifest."""
  pol, params = _tabular()
  man = artifact.write_artifact(
      str(tmp_path), 3, pol, params, run_id="77", git_commit="g", **_W
  )
  assert artifact.load_manifest(str(tmp_path), 3) == man
  assert artifact.load_artifact(str(tmp_path), 3).manifest == man
  assert (man.run_id, man.git_commit) == ("77", "g")
  named = artifact.write_artifact(
      str(tmp_path / "n"), 3, pol, params, run_id="ppo-a", **_W
  )
  assert named.run_id == "ppo-a"
  with pytest.raises(artifact.ArtifactError, match="run_id"):
    artifact.write_artifact(
        str(tmp_path / "bad"), 3, pol, params, run_id="", **_W
    )


def test_gs_paths_keep_their_scheme():
  """Gs paths keep their scheme."""
  assert artifact.norm_path("gs://bucket/runs/1/") == "gs://bucket/runs/1"
  assert artifact.norm_path("gs://bucket//a/b") == "gs://bucket/a/b"
  assert artifact.norm_path("/tmp/x/../y/") == "/tmp/y"
  # The etils path of the normalized string: some etils builds map gs:// to a
  # local mount, so the expected form is etils' own, not a literal.
  want = str(epath.Path("gs://bucket/runs/1"))
  assert str(artifact.ckpt_root("gs://bucket/runs/1/")) == want
  with pytest.raises(ValueError):
    artifact.norm_path("")
  with pytest.raises(ValueError, match="scheme"):
    artifact.norm_path("s3://bucket/x")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
