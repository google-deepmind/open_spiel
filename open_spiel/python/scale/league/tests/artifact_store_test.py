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

"""ArtifactStore: register, list and load artifacts by id.

ArtifactStore: register / list / load artifacts by id ("<run id>@<step>"),
and load them as an opponent slot for the Arena.

Pins: ids are "<run id>@<step>" (or an explicit "<name>@<step>" when the
artifact has no run id); the index persists at
<root>/artifact_index.json; registering is idempotent and a conflicting
claim on an id raises; load goes through load_artifact and refuses an
artifact whose params_sha256 changed since registration; `slot` stacks K
same-architecture artifacts into one Arena Slot (known answer: each stacked
member plays exactly its stored argmax policy).
"""

import json
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.league import artifact_store as store_lib
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform

_W = dict(
    rules_sha="r" * 8, config_hash="c" * 8, algo="test", env_name="kuhn_poker"
)


def _pol():
  """A Kuhn tabular policy.

  Returns:
    The policy.
  """
  return tabular.TabularPolicy(
      "kuhn_poker", 12, 2, 2, f"{kuhn.__name__}:info_index"
  )


def _write(path, step, prefer, run_id=42):
  """A tabular kuhn artifact whose argmax action is `prefer` everywhere.

  Args:
    path: The folder.
    step: The step.
    prefer: The preferred action.
    run_id: The run id, or None.

  Returns:
    The path.
  """
  pol = _pol()
  logits = np.zeros((12, 2), np.float32)
  logits[:, prefer] = 10.0
  artifact.write_artifact(
      str(path),
      step,
      pol,
      pol.params_from_logits(logits),
      run_id=None if run_id is None else str(run_id),
      default_rule=types.ARGMAX,
      **_W,
  )
  return str(path)


def test_register_list_load_by_id(tmp_path):
  """Register list load by id."""
  run = _write(tmp_path / "run", 1, 0)
  _write(tmp_path / "run", 2, 1)
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  assert not st.ids()
  assert st.register(run, 1) == "42@1"
  assert st.register(run, 2) == "42@2"
  assert st.ids() == ("42@1", "42@2")
  e = st.get("42@2")
  assert (e.path, e.step, e.run_id, e.game, e.arch_name) == (
      run,
      2,
      "42",
      "kuhn_poker",
      "tabular",
  )
  assert e.params_sha256 == artifact.load_manifest(run, 2).params_sha256
  assert [x.artifact_id for x in st.entries(game="kuhn_poker")] == [
      "42@1",
      "42@2",
  ]
  assert not st.entries(game="leduc_poker")
  art = st.load("42@2")
  assert art.manifest.step == 2 and art.manifest.run_id == "42"
  assert int(jnp.argmax(art.params["logits"][0])) == 1
  with pytest.raises(KeyError, match="42@3"):
    st.get("42@3")
  with pytest.raises(KeyError):
    st.load("42@3")
  # The index persists and a second instance sees the same ids.
  idx = json.loads((tmp_path / "league" / "artifact_index.json").read_text())
  assert sorted(idx["artifacts"]) == ["42@1", "42@2"]
  assert store_lib.ArtifactStore(str(tmp_path / "league")).ids() == (
      "42@1",
      "42@2",
  )


def test_register_is_idempotent_and_conflicts_raise(tmp_path):
  """Register is idempotent and conflicts raise."""
  a = _write(tmp_path / "a", 1, 0)
  b = _write(tmp_path / "b", 1, 1)  # same run id and step, different artifact
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  assert st.register(a, 1) == "42@1"
  assert st.register(a + "/", 1) == "42@1"  # same path, normalised
  assert st.ids() == ("42@1",)
  with pytest.raises(ValueError, match="42@1"):
    st.register(b, 1)


def test_ids_need_a_run_id_or_an_explicit_name(tmp_path):
  """Ids need a run id or an explicit name."""
  p = _write(tmp_path / "no_run_id", 3, 0, run_id=None)
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  with pytest.raises(ValueError, match="run id"):
    st.register(p, 3)
  assert st.register(p, 3, artifact_id="bc_seed0@3") == "bc_seed0@3"
  with pytest.raises(ValueError, match="step"):
    st.register(p, 3, artifact_id="bc_seed0@4")
  q = _write(tmp_path / "x", 5, 0, run_id=42)
  with pytest.raises(ValueError, match="run id"):
    st.register(q, 5, artifact_id="43@5")
  for bad in ("42", "@3", "42@", "42@x", "a b@3", "42@-1"):
    with pytest.raises(ValueError):
      store_lib.parse_artifact_id(bad)
  assert store_lib.parse_artifact_id("123@3000") == ("123", 3000)


def test_load_refuses_an_artifact_changed_since_registration(tmp_path):
  """Load refuses an artifact changed since registration."""
  run = _write(tmp_path / "run", 1, 0)
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  st.register(run, 1)
  shutil.rmtree(tmp_path / "run")
  # a different artifact at the same path and step
  _write(tmp_path / "run", 1, 1)
  with pytest.raises(artifact.ArtifactError, match="registered"):
    store_lib.ArtifactStore(str(tmp_path / "league")).load("42@1")


def test_corrupt_index_raises(tmp_path):
  """Corrupt index raises."""
  (tmp_path / "artifact_index.json").write_text("{not json")
  with pytest.raises(store_lib.Error):
    store_lib.ArtifactStore(str(tmp_path)).ids()
  (tmp_path / "artifact_index.json").write_text(
      json.dumps({"schema": 99, "artifacts": {}})
  )
  with pytest.raises(store_lib.Error, match="schema"):
    store_lib.ArtifactStore(str(tmp_path)).ids()


def test_slot_stacks_opponents_known_answer(tmp_path):
  """Two stored opponents in one slot: member m always plays action m."""
  run = _write(tmp_path / "run", 1, 0)
  _write(tmp_path / "run", 2, 1)
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  st.register(run, 1)
  st.register(run, 2)
  opp, opp_params = st.slot(["42@1", "42@2"])
  assert opp.num_members == 2 and opp.rule == types.ARGMAX
  assert opp_params["logits"].shape == (2, 12, 2)
  np.testing.assert_array_equal(
      np.asarray(opp_params["logits"][1]),
      np.asarray(st.load("42@2").params["logits"]),
  )
  env = kuhn.KuhnEnv()
  me = arena_lib.Slot(
      uniform.UniformPolicy("kuhn_poker", 2, 2), 1, types.SAMPLE
  )
  b = 16
  seat_slot = np.array([[0, 1], [1, 0]] * (b // 2), np.int32)
  seat_member = np.zeros((b, 2), np.int32)
  for r in range(b):
    opp_seat = 1 if r % 2 == 0 else 0
    seat_member[r, opp_seat] = (r // 2) % 2
  persp = np.array([0, 1] * (b // 2), np.int32)
  table = arena_lib.SlotTable(seat_slot, seat_member, persp)
  dk, pk = arena_lib.independent_keys(jax.random.key(0), b, 2)
  params = [jax.tree_util.tree_map(lambda x: jnp.stack([x]), {}), opp_params]
  _, _, log = arena_lib.Arena(env, [me, opp]).collect(params, table, dk, pk, 12)
  slot, member, action = (
      np.asarray(log.slot),
      np.asarray(log.member),
      np.asarray(log.action),
  )
  mask = slot == 1
  assert (
      mask.sum() > 0 and (member[mask] == 0).any() and (member[mask] == 1).any()
  )
  np.testing.assert_array_equal(action[mask], member[mask])
  assert not bool(np.asarray(log.bad).any())


def test_slot_rejects_mixed_architectures_and_rules(tmp_path):
  """Slot rejects mixed architectures and rules."""
  run = _write(tmp_path / "run", 1, 0)
  pol = uniform.UniformPolicy("kuhn_poker", 2, 2)
  artifact.write_artifact(str(tmp_path / "u"), 1, pol, {}, run_id="7", **_W)
  st = store_lib.ArtifactStore(str(tmp_path / "league"))
  st.register(run, 1)
  st.register(str(tmp_path / "u"), 1)
  with pytest.raises(ValueError, match="architecture"):
    st.slot(["42@1", "7@1"])
  with pytest.raises(ValueError):
    st.slot([])
  opp, _ = st.slot(["42@1"], rule=types.SAMPLE)
  assert opp.rule == types.SAMPLE


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
