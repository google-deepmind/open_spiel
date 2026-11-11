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

"""The JAX params digest: tree_sha256, computed in memory.

Pins: a known answer spelled out from the definition; the digest changes with
any leaf's bits, dtype, shape or key path; write_artifact never reads the
written params back; load_artifact checks the digest against the restored
tree.
"""

import hashlib
import shutil
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import codecs
from open_spiel.python.scale.core.tests import numpy_policies
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform

KUHN_INDEX = artifact.__name__.rpartition(".ckpt.")[0] + ".envs.kuhn:info_index"
_W = dict(
    rules_sha="r" * 8, config_hash="c" * 8, algo="test", env_name="kuhn_poker"
)


def _tabular(scale=1.0):
  """A Kuhn tabular policy and params with logits scaled by `scale`.

  Args:
    scale: The logit scale.

  Returns:
    (policy, params).
  """
  pol = tabular.TabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX)
  return pol, pol.params_from_logits(
      jnp.arange(24.0, dtype=jnp.float32).reshape(12, 2) * scale
  )


@pytest.fixture(name="default_tag")
def _default_tag(monkeypatch):
  # A deployment's plugin may rebind the tag in this process; the known
  # answers below are the library's own.
  monkeypatch.setattr(codecs, "TREE_SHA_TAG", b"params_tree_sha256\0")


@pytest.mark.usefixtures("default_tag")
def test_tree_sha256_known_answer():
  """Tree sha256 known answer."""
  tree = {
      "b": np.array([1, -2], np.int64),
      "a": np.array([[0.5], [-0.0]], np.float32),
  }
  h = hashlib.sha256(b"params_tree_sha256\0")
  for key, dtype, shape, raw in (
      ("['a']", "float32", "[2, 1]", np.array([0.5, -0.0], "<f4").tobytes()),
      ("['b']", "int64", "[2]", np.array([1, -2], "<i8").tobytes()),
  ):
    h.update(
        key.encode() + b"\0" + dtype.encode() + b"\0" + shape.encode() + b"\0"
    )
    h.update(hashlib.sha256(raw).digest())
  assert artifact.tree_sha256(tree) == h.hexdigest()
  jtree = {"b": jnp.asarray(tree["b"]), "a": jnp.asarray(tree["a"])}
  # a jax leaf hashes as its numpy bits
  assert artifact.tree_sha256(jtree) == h.hexdigest()


@pytest.mark.usefixtures("default_tag")
def test_tree_sha256_sees_bits_dtype_shape_and_paths():
  """Tree sha256 sees bits dtype shape and paths."""
  base = {"w": np.array([0.0, 1.0], np.float32)}
  ref = artifact.tree_sha256(base)
  for other in (
      {"w": np.array([-0.0, 1.0], np.float32)},  # one bit
      {"w": np.array([0.0, 1.0], np.float64)},  # dtype
      {"w": np.array([[0.0, 1.0]], np.float32)},  # shape
      {"v": np.array([0.0, 1.0], np.float32)},  # key path
      {"w": np.array([0.0, 1.0], np.float32), "x": np.zeros((0,), np.float32)},
  ):
    assert artifact.tree_sha256(other) != ref, other
  assert (
      artifact.tree_sha256({})
      == hashlib.sha256(b"params_tree_sha256\0").hexdigest()
  )


@pytest.mark.usefixtures("default_tag")
def test_flat_sha256_known_answer_matches_tree_sha256_keys_sorted():
  """Flat sha256 known answer matches tree sha256 keys sorted."""
  flat = {
      "b": np.array([1, -2], np.int64),
      "a": np.array([[0.5], [-0.0]], np.float32),
  }
  h = hashlib.sha256(b"params_tree_sha256\0")
  for key, dtype, shape, raw in (
      ("a", "float32", "[2, 1]", np.array([0.5, -0.0], "<f4").tobytes()),
      ("b", "int64", "[2]", np.array([1, -2], "<i8").tobytes()),
  ):
    h.update(
        key.encode() + b"\0" + dtype.encode() + b"\0" + shape.encode() + b"\0"
    )
    h.update(hashlib.sha256(raw).digest())
  assert codecs.flat_sha256(flat) == h.hexdigest()
  with pytest.raises(artifact.ArtifactError, match="flat mapping"):
    codecs.flat_sha256([np.zeros(2)])


def test_numpy_params_codec_round_trips_and_verifies(tmp_path):
  """Numpy params codec round trips and verifies."""
  pol = numpy_policies.NumpyHistoryPolicy("kuhn_poker", 2, 2)
  codec = codecs.params_codec("numpy")
  assert (
      codec is codecs.params_codec_by_name("numpy.npz")
      and codec.suffix == ".npz"
  )
  params = pol.make_params(c=[1.0, 2.0], b=[0.5, -0.5])
  path = tmp_path / "p.npz"
  digest = codec.write(path, pol, params)
  assert digest == codecs.flat_sha256(params)
  got = codec.read(path, pol, digest)
  assert sorted(got) == ["b", "c", "w"]
  for k, v in params.items():
    assert got[k].dtype == v.dtype and np.array_equal(got[k], v)
  with pytest.raises(artifact.ArtifactError, match="already exists"):
    codec.write(path, pol, params)
  with pytest.raises(artifact.ArtifactError, match="does not match"):
    codec.read(path, pol, "0" * 64)
  with pytest.raises(artifact.ArtifactError, match="architecture wants"):
    codec.write(
        tmp_path / "q.npz", pol, dict(params, b=np.zeros(3, np.float32))
    )
  with pytest.raises(artifact.ArtifactError, match="architecture keys"):
    codec.write(tmp_path / "r.npz", pol, {"c": params["c"]})


def _no_readback(monkeypatch):
  def refuse(*a, **k):
    raise AssertionError("params were read back during the write")

  monkeypatch.setattr(codecs, "restore_tree", refuse)


def test_write_hashes_params_in_memory(tmp_path, monkeypatch):
  """Write hashes params in memory."""
  pol, params = _tabular()
  with monkeypatch.context() as m:
    _no_readback(m)
    man = artifact.write_artifact(str(tmp_path), 4, pol, params, **_W)
  assert man.params_sha256 == artifact.tree_sha256(params)
  assert artifact.load_manifest(str(tmp_path), 4) == man
  art = artifact.load_artifact(str(tmp_path), 4)
  assert (
      art.manifest == man
      and artifact.tree_sha256(art.params) == man.params_sha256
  )


def test_empty_params_digest_round_trips(tmp_path, monkeypatch):
  """Empty params digest round trips."""
  with monkeypatch.context() as m:
    _no_readback(m)
    man = artifact.write_artifact(
        str(tmp_path), 0, uniform.UniformPolicy("kuhn_poker", 2, 2), {}, **_W
    )
  assert man.params_sha256 == artifact.tree_sha256({})
  assert not artifact.load_artifact(str(tmp_path), 0).params


def test_other_params_under_a_manifest_raise(tmp_path):
  """Other params under a manifest raise."""
  pol, params = _tabular()
  artifact.write_artifact(str(tmp_path / "a"), 1, pol, params, **_W)
  _, other = _tabular(scale=2.0)
  artifact.write_artifact(str(tmp_path / "b"), 1, pol, other, **_W)

  shutil.rmtree(tmp_path / "a" / "1" / "params")
  shutil.copytree(
      tmp_path / "b" / "1" / "params", tmp_path / "a" / "1" / "params"
  )
  with pytest.raises(artifact.ArtifactError, match="tree_sha256"):
    artifact.load_artifact(str(tmp_path / "a"), 1)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
