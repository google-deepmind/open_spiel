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

"""The native artifact format: layout, round trip, validation, extension points.

Pins: a step is <dir>/<step>/manifest.json plus params written by the
framework's codec; leaf dtypes round-trip exactly; a tampered, incomplete or
mismatched step is refused; extensions must be registered to be written;
registered formats are tried before the native one and the default writer
can be switched; a crashed write is moved aside, never deleted.
"""

import dataclasses
import json
import os
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import codecs
from open_spiel.python.scale.ckpt import formats
from open_spiel.python.scale.core import types
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import fixtures

KUHN_INDEX = artifact.__name__.rpartition(".ckpt.")[0] + ".envs.kuhn:info_index"
_W = dict(
    rules_sha="r" * 8, config_hash="c" * 8, algo="test", env_name="kuhn_poker"
)
F64_ONLY = 1.0 + 2.0**-40  # not representable in f32


def _tabular():
  """A Kuhn tabular policy and its params.

  Returns:
    (policy, params).
  """
  pol = tabular.TabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX)
  return pol, pol.params_from_logits(jnp.arange(24.0).reshape(12, 2))


def _write_tabular(tmp_path, step=7, **kw):
  """Writes a tabular artifact at `step`.

  Args:
    tmp_path: The folder.
    step: The step.
    **kw: Overrides of the write_artifact arguments.

  Returns:
    (policy, params, manifest).
  """
  pol, params = _tabular()
  kw = dict(
      dict(
          _W,
          default_rule=types.SAMPLE,
          git_commit="abc",
          run_id="123",
          parents=("1@2",),
          lineage=artifact.lineage("gs://b/runs/1", 2),
      ),
      **kw,
  )
  man = artifact.write_artifact(str(tmp_path), step, pol, params, **kw)
  return pol, params, man


def _manifest(tmp_path, step):
  """The parsed manifest.json of `step`.

  Args:
    tmp_path: The artifact folder.
    step: The step.

  Returns:
    The manifest dict.
  """
  return json.loads(
      (tmp_path / str(step) / artifact.MANIFEST_FILENAME).read_text()
  )


def _rewrite_manifest(tmp_path, step, fn):
  path = tmp_path / str(step) / artifact.MANIFEST_FILENAME
  m = json.loads(path.read_text())
  fn(m)
  path.write_text(json.dumps(m))


def test_round_trip_and_layout(tmp_path):
  """Round trip and layout."""
  pol, params, man = _write_tabular(tmp_path)
  art = artifact.load_artifact(str(tmp_path), 7)
  assert art.manifest == man
  assert (
      man.schema_version == artifact.FORMAT_VERSION
      and man.arch_name == "tabular"
  )
  assert man.default_rule == types.SAMPLE and man.run_id == "123"
  assert man.parents == ("1@2",) and man.step == 7
  assert type(art.policy) is type(pol) and art.policy.spec == pol.spec
  np.testing.assert_array_equal(
      np.asarray(art.params["logits"]), np.asarray(params["logits"])
  )
  m = _manifest(tmp_path, 7)
  # Reaches private _MANIFEST_KEYS of the code under test.
  assert sorted(m) == sorted(artifact._MANIFEST_KEYS)  # pylint: disable=protected-access
  assert (m["format"], m["schema_version"]) == (artifact.FORMAT_TAG, 1)
  assert m["params_codec"] == "jax.orbax" and m["decider"] == "network"
  assert (
      m["provenance"]["config_hash"] == "c" * 8
      and m["provenance"]["git_commit"] == "abc"
  )
  assert m["provenance"]["lineage"] == {"source": "gs://b/runs/1", "step": 2}
  pdir = tmp_path / "7" / "params"
  assert pdir.is_dir() and (pdir / "_CHECKPOINT_METADATA").exists()
  assert artifact.load_manifest(str(tmp_path), 7) == man


def test_a_manifest_under_another_format_tag_is_refused(tmp_path):
  """A manifest under another format tag is refused."""
  _write_tabular(tmp_path)
  assert _manifest(tmp_path, 7)["format"] == "artifact"
  path = tmp_path / "7" / artifact.MANIFEST_FILENAME
  m = json.loads(path.read_text())
  path.write_text(json.dumps(dict(m, format="acme.artifact")))
  with pytest.raises(artifact.ArtifactError, match="'artifact'"):
    artifact.load_manifest(str(tmp_path), 7)

  class AcmeFormat(artifact.NativeFormat):
    """The native format under another format tag."""

    tag = "acme.artifact"

  assert AcmeFormat().load_manifest(str(tmp_path), 7).step == 7


def test_a_decider_that_disagrees_with_the_rebuilt_policy_is_refused(tmp_path):
  """A decider that disagrees with the rebuilt policy is refused."""
  _write_tabular(tmp_path)
  path = tmp_path / "7" / artifact.MANIFEST_FILENAME
  m = json.loads(path.read_text())
  path.write_text(json.dumps(dict(m, decider="search")))
  with pytest.raises(artifact.ArtifactError, match="decider"):
    artifact.load_artifact(str(tmp_path), 7)
  path.write_text(json.dumps(dict(m, decider="oracle")))
  with pytest.raises(artifact.ArtifactError, match="decider"):
    artifact.load_manifest(str(tmp_path), 7)


def test_exists(tmp_path):
  """Exists."""
  assert not artifact.exists(str(tmp_path)) and not artifact.exists(
      str(tmp_path), 7
  )
  _write_tabular(tmp_path)
  assert artifact.exists(str(tmp_path)) and artifact.exists(str(tmp_path), 7)
  assert not artifact.exists(str(tmp_path), 8)


def test_missing_files_raise(tmp_path):
  """Missing files raise."""
  with pytest.raises(FileNotFoundError):
    artifact.load_artifact(str(tmp_path), 0)
  _write_tabular(tmp_path)
  with pytest.raises(FileNotFoundError):
    artifact.load_artifact(str(tmp_path), 8)
  shutil.rmtree(tmp_path / "7" / "params")
  with pytest.raises(FileNotFoundError):
    artifact.load_artifact(str(tmp_path), 7)


def test_steps_are_written_once(tmp_path):
  """Steps are written once."""
  pol, params, _ = _write_tabular(tmp_path)
  with pytest.raises(artifact.ArtifactError, match="write-once"):
    artifact.write_artifact(str(tmp_path), 7, pol, params, **_W)


def test_params_that_do_not_match_the_digest_raise(tmp_path):
  """Params that do not match the digest raise."""
  pol, _, _ = _write_tabular(tmp_path)
  other = tmp_path / "other"
  artifact.write_artifact(
      str(other),
      7,
      pol,
      pol.params_from_logits(jnp.zeros((12, 2), jnp.float32)),
      **_W,
  )
  pdir = tmp_path / "7" / "params"
  shutil.rmtree(pdir)
  shutil.copytree(other / "7" / "params", pdir)
  with pytest.raises(artifact.ArtifactError, match="tree_sha256"):
    artifact.load_artifact(str(tmp_path), 7)
  assert (
      artifact.load_artifact(str(tmp_path), 7, verify=False).params is not None
  )
  p = sorted(x for x in pdir.rglob("*") if x.is_file())[0]
  # a corrupt file, whatever orbax makes of it
  p.write_bytes(b"x" + p.read_bytes())
  with pytest.raises(artifact.ArtifactError, match="digest"):
    artifact.load_artifact(str(tmp_path), 7)


@pytest.mark.parametrize(
    "field", ["arch_name", "params_sha256", "schema_version"]
)
def test_an_incomplete_or_foreign_manifest_raises(tmp_path, field):
  """An incomplete or foreign manifest raises."""
  _write_tabular(tmp_path)
  if field == "schema_version":
    _rewrite_manifest(tmp_path, 7, lambda m: m.update(schema_version=2))
  else:
    _rewrite_manifest(tmp_path, 7, lambda m: m.pop(field))
  with pytest.raises(artifact.ArtifactError, match=field):
    artifact.load_artifact(str(tmp_path), 7)


def test_f64_and_f32_leaves_restore_with_their_dtypes(tmp_path):
  """F64 and f32 leaves restore with their dtypes."""
  pol = fixtures.DtypeProbePolicy("kuhn_poker", 2, 2, w64_dtype="float64")
  params = pol.make_params(
      w32=np.array([0.5, -1.5, 2.0], np.float32),
      w64=np.array([F64_ONLY, -F64_ONLY], np.float64),
      count=np.int64(2**40 + 3),
  )
  artifact.write_artifact(str(tmp_path), 1, pol, params, **_W)
  got = artifact.load_artifact(str(tmp_path), 1).params
  assert got["w32"].dtype == jnp.float32 and got["w64"].dtype == jnp.float64
  assert got["count"].dtype == jnp.int64 and int(got["count"]) == 2**40 + 3
  np.testing.assert_array_equal(np.asarray(got["w64"]), [F64_ONLY, -F64_ONLY])


def test_stored_dtype_differing_from_architecture_raises(tmp_path):
  """A stored dtype that differs from the architecture raises.

  Orbax would cast an f64 leaf to an f32 target silently; the stored dtypes
  are compared with the architecture's first.

  Args:
    tmp_path: The test's temporary directory.
  """
  pol = fixtures.DtypeProbePolicy("kuhn_poker", 2, 2, w64_dtype="float64")
  artifact.write_artifact(str(tmp_path), 1, pol, pol.init_params(), **_W)
  _rewrite_manifest(
      tmp_path, 1, lambda m: m["arch_kwargs"].update(w64_dtype="float32")
  )
  with pytest.raises(artifact.ArtifactError, match="float64"):
    artifact.load_artifact(str(tmp_path), 1)


def test_writing_params_that_differ_from_the_architecture_raises(tmp_path):
  """Writing params that differ from the architecture raises."""
  pol = fixtures.DtypeProbePolicy("kuhn_poker", 2, 2, w64_dtype="float64")
  bad = dict(pol.init_params(), w64=jnp.zeros((2,), jnp.float32))
  with pytest.raises(artifact.ArtifactError, match="w64"):
    artifact.write_artifact(str(tmp_path), 1, pol, bad, **_W)
  assert not artifact.exists(str(tmp_path), 1)


def test_sharded_params_round_trip_over_four_devices(tmp_path):
  """Sharded params round trip over four devices."""
  env = {
      "XLA_FLAGS": (
          os.environ.get("XLA_FLAGS", "")
          + " --xla_force_host_platform_device_count=4"
      ).strip(),
      "JAX_PLATFORMS": "cpu",
  }
  r = subprocess_env.run_python(
      ["-m", subprocess_env.module("ckpt.tests.sharded_probe"), str(tmp_path)],
      env=env,
  )
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
  assert "CKPT_SHARDED_OK" in r.stdout, r.stdout[-2000:]


def test_empty_params_round_trip(tmp_path):
  """A params-free architecture (uniform) is still a loadable artifact."""
  pol = uniform.UniformPolicy("kuhn_poker", 2, 2)
  man = artifact.write_artifact(str(tmp_path), 0, pol, {}, **_W)
  art = artifact.load_artifact(str(tmp_path), 0)
  assert not art.params and art.manifest.arch_name == "uniform"
  assert _manifest(tmp_path, 0)["decider"] == "scripted"
  assert man.params_sha256 == artifact.tree_sha256({})


def test_save_restore_tree_round_trip_and_refuse_narrowing(tmp_path):
  """save_tree / restore_tree round-trip bitwise and refuse narrowing.

  save_tree / restore_tree: bitwise round trip keeping int64 past 2**31 and
  float64; write-once; a stored dtype differing from the template raises.

  Args:
    tmp_path: The test's temporary directory.
  """
  tree = {
      "step": jnp.asarray(2**31 + 5, jnp.int64),
      "w": jnp.asarray([1.0 + 2**-40, -0.0], jnp.float64),
      "p": jnp.arange(3, dtype=jnp.float32),
  }
  path = tmp_path / "tree"
  artifact.save_tree(path, tree)
  back = artifact.restore_tree(path, tree)
  for k, v in tree.items():
    assert back[k].dtype == v.dtype, k
    np.testing.assert_array_equal(np.asarray(back[k]), np.asarray(v))
  with pytest.raises(artifact.ArtifactError, match="write-once"):
    artifact.save_tree(path, tree)
  with pytest.raises(artifact.ArtifactError, match="float64"):
    artifact.restore_tree(path, dict(tree, w=jnp.zeros((2,), jnp.float32)))


# -- orbax commit markers ------------------------------------------------------


def _gcs_semantics(monkeypatch, *params_dirs):
  """Gives every path gs:// commit semantics; params_dirs lose their marker.

  A checkpoint directory is then complete only with its commit_success.txt.
  The seam is this package's own (codecs.needs_commit_file), not orbax's, so
  the check behaves the same under every orbax release. Whether a local save
  leaves a commit_success.txt behind is orbax's choice (0.12.3 and later
  write one on every backend), so the marker is deleted from each of
  `params_dirs`: each then stands for a local write copied to gs:// without
  its marker, under any orbax release.

  Args:
    monkeypatch: The test's monkeypatch fixture.
    *params_dirs: orbax checkpoint directories to strip of the marker.
  """
  monkeypatch.setattr(codecs, "needs_commit_file", lambda _: True)
  for d in params_dirs:
    assert d.is_dir(), d
    (d / codecs.COMMIT_SUCCESS_FILE).unlink(missing_ok=True)


def test_params_finalized_follows_orbax_commit_semantics(tmp_path, monkeypatch):
  """Params finalized follows orbax commit semantics."""
  p = tmp_path / "t"
  artifact.save_tree(p, {"w": jnp.arange(3.0, dtype=jnp.float32)})
  assert artifact.params_finalized(p)  # local: the atomic rename finalised it
  tmp = tmp_path / ("u" + codecs.TMP_DIR_SUFFIX + "-123")
  shutil.copytree(p, tmp)
  assert not artifact.params_finalized(tmp)  # an orbax temporary
  assert not artifact.params_finalized(tmp_path / "absent")
  _gcs_semantics(monkeypatch, p)
  assert not artifact.params_finalized(p)  # a local write copied to gs:// as is
  (p / "commit_success.txt").write_text("ok")
  assert artifact.params_finalized(p)


def test_unfinalized_params_refused_by_manifest_and_load(tmp_path, monkeypatch):
  """Unfinalized params refused by manifest and load."""
  _write_tabular(tmp_path)
  _gcs_semantics(monkeypatch, tmp_path / "7" / "params")
  with pytest.raises(
      artifact.ArtifactError, match="incomplete orbax checkpoint"
  ):
    artifact.load_manifest(str(tmp_path), 7)
  with pytest.raises(
      artifact.ArtifactError, match="incomplete orbax checkpoint"
  ):
    artifact.load_artifact(str(tmp_path), 7)
  assert artifact.load_manifest(str(tmp_path), 7, verify=False).step == 7
  (tmp_path / "7" / "params" / "commit_success.txt").write_text("ok")
  assert artifact.load_artifact(str(tmp_path), 7).manifest.step == 7


def test_empty_params_marker_needs_no_commit_marker(tmp_path, monkeypatch):
  """Empty params marker needs no commit marker."""
  artifact.write_artifact(
      str(tmp_path), 1, uniform.UniformPolicy("kuhn_poker", 2, 2), {}, **_W
  )
  _gcs_semantics(monkeypatch)
  assert artifact.load_manifest(str(tmp_path), 1).step == 1
  assert not artifact.load_artifact(str(tmp_path), 1).params


# -- extensions ----------------------------------------------------------------


def test_value_scale_is_a_built_in_extension(tmp_path):
  """Value scale is a built in extension."""
  pol, params = _tabular()
  artifact.write_artifact(
      str(tmp_path / "a"),
      1,
      pol,
      params,
      extensions={"value_scale": 50.0},
      **_W,
  )
  artifact.write_artifact(str(tmp_path / "b"), 1, pol, params, **_W)
  assert artifact.value_scale(str(tmp_path / "a"), 1) == 50.0
  assert _manifest(tmp_path / "a", 1)["extensions"] == {"value_scale": 50.0}
  # not recorded: unknown units
  assert artifact.value_scale(str(tmp_path / "b"), 1) is None
  for bad in (0.0, -1.0, float("nan"), float("inf"), True, "50"):
    with pytest.raises(artifact.ArtifactError, match="value_scale"):
      artifact.write_artifact(
          str(tmp_path / f"bad{bad}"),
          1,
          pol,
          params,
          extensions={"value_scale": bad},
          **_W,
      )
  _rewrite_manifest(
      tmp_path / "a", 1, lambda m: m["extensions"].update(value_scale=-2.0)
  )
  with pytest.raises(artifact.ArtifactError, match="value_scale"):
    artifact.load_manifest(str(tmp_path / "a"), 1)


def test_unregistered_extensions_are_refused_on_write_and_kept_on_read(
    tmp_path,
):
  """Unregistered extensions are refused on write and kept on read."""
  pol, params = _tabular()
  with pytest.raises(artifact.ArtifactError, match="unregistered"):
    artifact.write_artifact(
        str(tmp_path), 1, pol, params, extensions={"nope": 1}, **_W
    )
  assert not artifact.exists(str(tmp_path), 1)
  artifact.write_artifact(str(tmp_path), 1, pol, params, **_W)
  _rewrite_manifest(
      tmp_path, 1, lambda m: m["extensions"].update(other_tool={"x": 1})
  )
  assert artifact.load_manifest(str(tmp_path), 1).extensions == {
      "other_tool": {"x": 1}
  }


def test_register_meta_extension_validates(tmp_path):
  """Register meta extension validates."""

  def check(v):
    if v not in ("a", "b"):
      raise ValueError(f"{v!r} not a or b")

  artifact.register_meta_extension("artifact_test.choice", check)
  # same validator: no-op
  artifact.register_meta_extension("artifact_test.choice", check)
  with pytest.raises(ValueError, match="already registered"):
    artifact.register_meta_extension("artifact_test.choice", lambda v: None)
  pol, params = _tabular()
  man = artifact.write_artifact(
      str(tmp_path),
      1,
      pol,
      params,
      extensions={"artifact_test.choice": "a"},
      **_W,
  )
  assert man.extensions == {"artifact_test.choice": "a"}
  with pytest.raises(artifact.ArtifactError, match="not a or b"):
    artifact.write_artifact(
        str(tmp_path),
        2,
        pol,
        params,
        extensions={"artifact_test.choice": "c"},
        **_W,
    )


def test_a_training_only_wrapper_is_not_publishable(tmp_path):
  """A training only wrapper is not publishable."""
  pol = uniform.UniformPolicy("kuhn_poker", 2, 2)

  wrapper = uniform.UniformPolicy("kuhn_poker", 2, 2)
  wrapper.spec = dataclasses.replace(wrapper.spec, publishable=False)
  with pytest.raises(artifact.ArtifactError, match="never published"):
    artifact.write_artifact(str(tmp_path), 1, wrapper, {}, **_W)
  artifact.write_artifact(str(tmp_path), 1, pol, {}, **_W)


# -- formats -------------------------------------------------------------------


class _JsonFormat:
  """A toy format: params as JSON lists beside a JSON manifest (tabular)."""

  name = "artifact_test.json"

  def _path(self, d, step):
    """The toy format's file of `step`.

    Args:
      d: The artifact folder.
      step: The step.

    Returns:
      The path.
    """
    return artifact.ckpt_root(d) / f"toy_{int(step)}.json"

  def detect(self, d, step):
    """Whether step `step` of `d` is a toy JSON artifact.

    Args:
      d: the artifact directory.
      step: the step id.

    Returns:
      True when the step's JSON file exists.
    """
    return self._path(d, step).exists()

  def has_any(self, d):
    """Whether `d` holds any toy JSON artifact.

    Args:
      d: the artifact directory.

    Returns:
      True when a toy_* entry exists under the checkpoint root.
    """
    root = artifact.ckpt_root(d)
    return root.exists() and any(
        c.name.startswith("toy_") for c in root.iterdir()
    )

  def load_manifest(self, d, step, *, verify=True):
    """Reads the manifest of a toy JSON artifact.

    Args:
      d: the artifact directory.
      step: the step id.
      verify: unused; the toy format has nothing to verify.

    Returns:
      The step's types.Manifest.
    """
    del verify
    m = json.loads(self._path(d, step).read_text())["manifest"]
    return types.Manifest(
        schema_version=0,
        game=m["game"],
        rules_sha=m["rules_sha"],
        obs_format=m["obs_format"],
        arch_name=m["arch_name"],
        arch_kwargs=m["arch_kwargs"],
        framework=str(m["framework"]),
        capability=types.Capability(m["capability"]),
        default_rule=types.ARGMAX,
        params_sha256="",
        step=int(step),
        parents=(),
    )

  def load_artifact(self, d, step, *, verify=True):
    """Reads a toy JSON artifact.

    Args:
      d: the artifact directory.
      step: the step id.
      verify: passed to load_manifest.

    Returns:
      The step's types.Artifact.
    """
    manifest = self.load_manifest(d, step, verify=verify)
    policy = artifact.build_policy(manifest)
    logits = np.asarray(
        json.loads(self._path(d, step).read_text())["logits"], np.float32
    )
    return types.Artifact(
        manifest=manifest,
        params=policy.params_from_logits(logits),
        policy=policy,
        path=artifact.norm_path(d),
    )

  def write(self, d, step, policy, params, manifest):
    """Writes a toy JSON artifact.

    Args:
      d: the artifact directory.
      step: the step id.
      policy: the policy whose spec is recorded.
      params: the params; their logits are stored.
      manifest: the manifest to record.

    Returns:
      The written manifest.
    """
    s = policy.spec
    blob = dict(
        manifest=dict(
            game=s.game,
            rules_sha=manifest.rules_sha,
            obs_format=s.obs_format,
            arch_name=s.arch_name,
            arch_kwargs=dict(s.arch_kwargs),
            framework=s.framework,
            capability=s.capability.value,
        ),
        logits=np.asarray(params["logits"]).tolist(),
    )
    artifact.atomic_write(self._path(d, step), json.dumps(blob).encode())
    return self.load_manifest(d, step)

  def clear_partial(self, d, step, tag):
    """Clears nothing: the toy format writes one file atomically.

    Args:
      d: the artifact directory.
      step: the step id.
      tag: the partial-write tag.
    """
    del d, step, tag
    return None


_TOY = _JsonFormat()


def test_a_registered_format_is_read_and_can_be_the_default_writer(tmp_path):
  """A registered format is read and can be the default writer."""
  formats.register_format(_TOY)
  formats.register_format(_TOY)  # the same object again: a no-op
  assert formats.names()[-1] == formats.NATIVE and _TOY.name in formats.names()
  pol, params = _tabular()
  formats.set_default_writer(_TOY.name)
  try:
    man = artifact.write_artifact(str(tmp_path), 3, pol, params, **_W)
  finally:
    formats.set_default_writer(formats.NATIVE)
  assert man.schema_version == 0 and (tmp_path / "toy_3.json").exists()
  assert not (tmp_path / "3").exists()
  art = artifact.load_artifact(str(tmp_path), 3)
  np.testing.assert_array_equal(
      np.asarray(art.params["logits"]), np.asarray(params["logits"])
  )
  assert artifact.exists(str(tmp_path), 3)
  # the native writer again
  artifact.write_artifact(str(tmp_path), 4, pol, params, **_W)
  assert (
      artifact.load_manifest(str(tmp_path), 4).schema_version
      == artifact.FORMAT_VERSION
  )
  with pytest.raises(KeyError, match="no artifact format"):
    formats.set_default_writer("no.such.format")


def test_clear_partial_moves_a_dead_write_aside(tmp_path):
  """Clear partial moves a dead write aside."""
  pol, params, _ = _write_tabular(tmp_path)
  # complete: left alone
  assert artifact.clear_partial(str(tmp_path), 7, "a1") is None
  # no manifest
  codecs.params_codec("jax").write(tmp_path / "8" / "params", pol, params)
  with pytest.raises(artifact.ArtifactError, match="without a manifest"):
    artifact.write_artifact(str(tmp_path), 8, pol, params, **_W)
  aside = artifact.clear_partial(str(tmp_path), 8, "a1")
  assert aside == str(tmp_path / "unpublished" / "8.a1") and os.path.isdir(
      aside
  )
  artifact.write_artifact(str(tmp_path), 8, pol, params, **_W)
  assert artifact.load_artifact(str(tmp_path), 8).manifest.step == 8


def test_unknown_framework_has_no_codec():
  """Unknown framework has no codec."""
  with pytest.raises(artifact.ArtifactError, match="no params codec"):
    codecs.params_codec("no_such_framework")
  assert jax.numpy is jnp


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
