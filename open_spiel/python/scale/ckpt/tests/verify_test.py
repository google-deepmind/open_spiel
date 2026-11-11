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

"""Verification compares serialized host values.

Pins: snapshot.equal is bitwise and structural; run_state.verify compares a
learner state on disk with learner.state_dict(state) without restoring it
into the learner (equal -> True, any changed value -> False); a registered
leaf codec makes a foreign leaf type comparable.
"""

import json
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.ckpt import snapshot
from open_spiel.python.scale.league.tests import league_fixtures


def test_snapshot_equal_is_bitwise_and_structural():
  """Snapshot equal is bitwise and structural."""
  a = snapshot.to_host({
      "w": jnp.asarray([1.0, jnp.nan], jnp.float32),
      "n": [1, 2.5, None],
      "c": np.int64(3),
  })
  b = snapshot.to_host({
      "w": np.asarray([1.0, np.nan], np.float32),
      "n": (1, 2.5, None),
      "c": np.int64(3),
  })
  assert snapshot.equal(a, b)  # same NaN bits; tuple == list
  assert not snapshot.equal(a, dict(b, c=np.int64(4)))
  assert not snapshot.equal(a, dict(b, w=np.asarray([1.0, np.nan], np.float64)))
  assert not snapshot.equal(a, dict(b, extra=1))
  assert not snapshot.equal(a, dict(b, n=[1, 2.5]))
  # NaN scalars
  assert snapshot.equal(
      {"s": np.float32(np.nan), "f": float("nan")},
      {"s": np.float32(np.nan), "f": float("nan")},
  )
  assert not snapshot.equal({"s": np.float32(0.0)}, {"s": np.float32(-0.0)})
  with pytest.raises(TypeError):
    snapshot.to_host({"x": object()})


class _Boxed:
  """A foreign leaf type: an array numpy cannot convert directly.

  Attributes:
    data: the boxed array.
  """

  def __init__(self, data):
    self.data = np.asarray(data)


def test_a_registered_leaf_codec_converts_foreign_leaves():
  """A registered leaf codec converts foreign leaves."""
  snapshot.register_leaf_codec(
      lambda x: isinstance(x, _Boxed),
      lambda x: (x.data.copy(), "boxed"),
      lambda a, meta: _Boxed(a),
      name="verify_test.boxed",
  )
  with pytest.raises(ValueError, match="already registered"):
    snapshot.register_leaf_codec(
        lambda x: False, None, None, name="verify_test.boxed"
    )
  host = snapshot.to_host({"b": _Boxed([1, 2])})
  assert snapshot.equal(host, {"b": np.asarray([1, 2])})
  codec = snapshot.leaf_codec_for(_Boxed([0]))
  arr, meta = codec.encode(_Boxed([3]))
  assert meta == "boxed" and isinstance(
      snapshot.leaf_codec("verify_test.boxed").decode(arr, meta), _Boxed
  )


def test_jax_learner_state_verified_without_restoring(tmp_path, monkeypatch):
  """Jax learner state verified without restoring."""
  pol = league_fixtures.kuhn_policy()
  lr = league_fixtures.br_learner(pol)
  s = lr.init_from_params(league_fixtures.init_params(pol, 0))
  run_state.save(str(tmp_path), 2, lr, s)

  def refuse(blob):
    raise AssertionError("verification must not restore into the live learner")

  monkeypatch.setattr(lr, "load_state", refuse)
  assert run_state.verify(str(tmp_path), 2, lr, s)
  other = lr.init_from_params(league_fixtures.init_params(pol, 1))
  assert not run_state.verify(str(tmp_path), 2, lr, other)
  with pytest.raises(FileExistsError):
    run_state.save(str(tmp_path), 2, lr, s)


def test_check_resume_requires_the_kind_and_the_config():
  """Check resume requires the kind and the config."""
  ts = {"kind": "pg", "step": 4, "resume_hash": "a" * 16}
  assert run_state.check_resume("d", ts, "pg", "a" * 16, 10) == 4
  with pytest.raises(ValueError, match="names no learner kind"):
    run_state.check_resume(
        "d", {"step": 4, "resume_hash": "a" * 16}, "pg", "a" * 16, 10
    )
  with pytest.raises(ValueError, match="'bc'"):
    run_state.check_resume("d", ts, "bc", "a" * 16, 10)
  with pytest.raises(ValueError, match="different config"):
    run_state.check_resume("d", ts, "pg", "b" * 16, 10)
  with pytest.raises(ValueError, match="length"):
    run_state.check_resume("d", ts, "pg", "a" * 16, 3)


def test_train_state_records_are_versioned_and_migrated(tmp_path, monkeypatch):
  """Train state records are versioned and migrated."""
  rec = {
      "kind": "pg",
      "step": 1,
      "resume_hash": "h",
      "config_hash": "c",
      "config": {},
  }
  run_state.write_train_state(str(tmp_path), rec)
  assert run_state.read_train_state(str(tmp_path)) == dict(
      rec, record_version=run_state.RECORD_VERSION
  )
  with pytest.raises(ValueError, match="needs"):
    run_state.write_train_state(str(tmp_path), {"kind": "pg"})
  with pytest.raises(ValueError, match="record_version"):
    run_state.write_train_state(str(tmp_path), dict(rec, record_version=99))

  path = tmp_path / run_state.TRAIN_STATE
  path.write_text(json.dumps(dict(rec, record_version=0)))
  with pytest.raises(ValueError, match="record_version 0"):
    run_state.read_train_state(str(tmp_path))
  monkeypatch.setattr(run_state, "_MIGRATIONS", {})

  def from_v0(r):
    """Migrates a version-0 record.

    Args:
      r: the version-0 record.

    Returns:
      The record at the current version, marked migrated.
    """
    return dict(r, record_version=run_state.RECORD_VERSION, migrated=True)

  run_state.register_record_migration(0, from_v0)
  # the same function again: a no-op
  run_state.register_record_migration(0, from_v0)
  assert run_state.read_train_state(str(tmp_path))["migrated"]
  with pytest.raises(ValueError, match="already"):
    run_state.register_record_migration(0, lambda r: r)
  with pytest.raises(ValueError, match="current"):
    run_state.register_record_migration(run_state.RECORD_VERSION, from_v0)
  # returns no version
  run_state.register_record_migration(None, dict)
  path.write_text(json.dumps(rec))
  with pytest.raises(ValueError, match="returned"):
    run_state.read_train_state(str(tmp_path))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
