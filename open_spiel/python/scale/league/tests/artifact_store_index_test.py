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

"""The ArtifactStore's index, without jax.

The ArtifactStore's index without jax: register / ids / entries / get /
unregister read artifacts through ckpt.artifact.load_manifest only.

Artifacts here are manifests a stub load_manifest returns (no params, no
framework), so the file runs where jax cannot be imported
(jax_free_control_test). Pins the ids, the persisted index JSON, the
identity fields read from the manifest, and the conflict and corruption
errors; load and slot are artifact_store_test.
"""

import json
import os
import shutil
import sys
from typing import Any

import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import types
from open_spiel.python.scale.league import artifact_store as store_lib


@pytest.fixture(name="manifests")
def _manifests_fixture(monkeypatch):
  """Path -> {step: Manifest}; the store's load_manifest reads it.

  Args:
    monkeypatch: The pytest fixture.

  Returns:
    The table the fake load_manifest reads.
  """
  table: dict[Any, Any] = {}

  def load_manifest(path, step):
    """Looks up a manifest in the table.

    Args:
      path: the artifact directory.
      step: the step id.

    Returns:
      The registered types.Manifest.

    Raises:
      FileNotFoundError: no manifest is registered at path@step.
    """
    got = table.get(artifact.norm_path(path).rstrip("/"), {}).get(int(step))
    if got is None:
      raise FileNotFoundError(f"{path}@{step}")
    return got

  monkeypatch.setattr(store_lib.artifact, "load_manifest", load_manifest)
  return table


def _add(table, path, step, digest, run=42, arch_name="stub"):
  """Registers a fake manifest for path@step.

  Args:
    table: The manifest table.
    path: The artifact folder.
    step: The step.
    digest: The params digest.
    run: The run id, or None.
    arch_name: The arch.

  Returns:
    The path.
  """
  path = str(path)
  os.makedirs(path, exist_ok=True)
  table.setdefault(artifact.norm_path(path).rstrip("/"), {})[int(step)] = (
      types.Manifest(
          schema_version=2,
          game="kuhn_poker",
          rules_sha="r" * 8,
          obs_format="kuhn_onehot",
          arch_name=arch_name,
          arch_kwargs={"width": 3},
          framework="jax",
          capability=types.Capability.HOST_DRIVEN,
          default_rule=types.ARGMAX,
          params_sha256=digest,
          run_id=None if run is None else str(run),
          step=int(step),
          parents=(),
      )
  )
  return path


def test_register_list_get_by_id(tmp_path, manifests):
  """Register list get by id."""
  run = _add(manifests, tmp_path / "run", 1, "a" * 64)
  _add(manifests, tmp_path / "run", 2, "b" * 64)
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  assert not st.ids()
  assert st.register(run, 1) == "42@1"
  assert st.register(run, 2) == "42@2"
  assert st.ids() == ("42@1", "42@2")
  e = st.get("42@2")
  assert (
      e.path,
      e.step,
      e.run_id,
      e.game,
      e.arch_name,
      e.framework,
      e.capability,
  ) == (run, 2, "42", "kuhn_poker", "stub", "jax", "host_driven")
  assert (
      e.params_sha256 == "b" * 64 and st.get("42@1").params_sha256 == "a" * 64
  )
  assert [x.artifact_id for x in st.entries(game="kuhn_poker")] == [
      "42@1",
      "42@2",
  ]
  assert not st.entries(game="leduc_poker") and not st.entries(
      arch_name="other"
  )
  with pytest.raises(KeyError, match="42@3"):
    st.get("42@3")
  idx = json.loads((tmp_path / "league" / "artifact_index.json").read_text())
  assert idx["schema"] == 1 and sorted(idx["artifacts"]) == ["42@1", "42@2"]
  assert store_lib.ArtifactStore(str(tmp_path / "league")).ids() == (
      "42@1",
      "42@2",
  )


def test_ids_sort_numeric_run_ids_before_names(tmp_path, manifests):
  """Ids sort numeric run ids before names."""
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  st.register(_add(manifests, tmp_path / "a", 10, "a" * 64, run=9), 10)
  st.register(_add(manifests, tmp_path / "b", 2, "b" * 64, run=100), 2)
  st.register(
      _add(manifests, tmp_path / "c", 5, "c" * 64, run=None),
      5,
      artifact_id="main@5",
  )
  assert st.ids() == ("9@10", "100@2", "main@5")


def test_register_is_idempotent_and_conflicts_raise(tmp_path, manifests):
  """Register is idempotent and conflicts raise."""
  a = _add(manifests, tmp_path / "a", 1, "a" * 64)
  b = _add(manifests, tmp_path / "b", 1, "b" * 64)
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  assert st.register(a, 1) == "42@1"
  assert st.register(a + "/", 1) == "42@1"
  with pytest.raises(ValueError, match="42@1"):
    st.register(b, 1)
  with pytest.raises(ValueError, match="run id"):
    st.register(_add(manifests, tmp_path / "n", 3, "n" * 64, run=None), 3)
  with pytest.raises(ValueError, match="step"):
    st.register(str(tmp_path / "n"), 3, artifact_id="x@4")
  with pytest.raises(ValueError, match="run id"):
    st.register(
        _add(manifests, tmp_path / "m", 3, "m" * 64, run=7),
        3,
        artifact_id="8@3",
    )


def test_unregister_only_under_the_given_dir(tmp_path, manifests):
  """Unregister only under the given dir."""
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  st.register(
      _add(
          manifests, tmp_path / "run" / "members" / "m", 1, "a" * 64, run=None
      ),
      1,
      artifact_id="m@1",
  )
  st.register(
      _add(manifests, tmp_path / "other", 1, "b" * 64, run=None),
      1,
      artifact_id="o@1",
  )
  with pytest.raises(ValueError, match="not under"):
    st.unregister("o@1", under=str(tmp_path / "run" / "members"))
  e = st.unregister("m@1", under=str(tmp_path / "run" / "members"))
  assert e.artifact_id == "m@1" and st.ids() == ("o@1",)
  assert (tmp_path / "run" / "members" / "m").exists()  # files stay
  with pytest.raises(KeyError):
    st.unregister("m@1", under=str(tmp_path / "run"))


def test_corrupt_or_foreign_index_raises(tmp_path):
  """Corrupt or foreign index raises."""
  (tmp_path / "artifact_index.json").write_text("{not json")
  with pytest.raises(store_lib.Error):
    store_lib.ArtifactStore(str(tmp_path)).ids()
  (tmp_path / "artifact_index.json").write_text(
      json.dumps({"schema": 99, "artifacts": {}})
  )
  with pytest.raises(store_lib.Error, match="schema"):
    store_lib.ArtifactStore(str(tmp_path)).ids()
  shutil.rmtree(tmp_path)
  tmp_path.mkdir()
  (tmp_path / "artifact_index.json").write_text(
      json.dumps({"schema": 1, "artifacts": []})
  )
  with pytest.raises(store_lib.Error, match="mapping"):
    store_lib.ArtifactStore(str(tmp_path)).ids()


def test_parse_artifact_id():
  """Parse artifact id."""
  assert store_lib.parse_artifact_id("123@3000") == ("123", 3000)
  for bad in ("42", "@3", "42@", "42@x", "a b@3", "42@-1", 7):
    with pytest.raises(ValueError):
      store_lib.parse_artifact_id(bad)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
