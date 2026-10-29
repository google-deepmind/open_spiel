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

"""sources.shards: write -> read and the reader's refusals.

sources.shards: write -> read is bitwise equal (tree, dtypes, values) for a
kuhn DAgger batch; the manifest carries game, rules_sha, info_schema_version
and the producer id, and the reader refuses any mismatch, a corrupted shard,
an encoded-observation batch, a missing manifest and a second writer on the
same directory.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources import shards
from open_spiel.python.scale.sources.tests import helpers
from open_spiel.python.scale.testing import fixtures

RULES = "kuhn-rules-sha-test"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


def _batches(env, n=2, student=None, sp=None, batch=8, num_steps=6, seed=0):
  """Batches from a Kuhn dagger source with a random teacher.

  Args:
    env: The env.
    n: The batches.
    student: The student (default tabular).
    sp: The student params (default init).
    batch: The rows.
    num_steps: The plies.
    seed: The seed.

  Returns:
    The batches.
  """
  pol = helpers.kuhn_tabular(env)
  logits = jnp.asarray(
      np.random.default_rng(seed).normal(size=(12, 2)), jnp.float32
  )
  student = student or pol
  sp = sp if sp is not None else pol.init_params()
  src = dagger.DaggerSource(
      env,
      student,
      pol,
      student_params=sp,
      teacher_params=pol.params_from_logits(logits),
      table=arena_lib.self_play_table(batch, 2, all_seats=True),
      num_steps=num_steps,
      beta=schedules.constant(0.5),
  )
  return [src.next(jax.random.key(seed * 100 + i)) for i in range(n)]


def _write(path, env, batches, **kw):
  """Writes `batches` as shards at `path`.

  Args:
    path: The folder.
    env: The env.
    batches: The batches.
    **kw: ShardWriter overrides.

  Returns:
    The path.
  """
  args = dict(rules_sha=RULES, info_schema_version=1, producer="123@100")
  args.update(kw)
  with shards.ShardWriter(str(path), env, **args) as w:
    for b in batches:
      w.write(b)
  return str(path)


def _reader(path, env, **kw):
  """A ShardReader of `path`.

  Args:
    path: The shard folder.
    env: The env.
    **kw: ShardReader overrides.

  Returns:
    The reader.
  """
  args = dict(rules_sha=RULES, info_schema_version=1)
  args.update(kw)
  return shards.ShardReader(path, env, **args)


def _concat(batches):
  """Concatenates batches along the rows.

  Args:
    batches: The batches.

  Returns:
    The batch.
  """
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batches)


def test_round_trip_bitwise(env, tmp_path):
  """Round trip bitwise."""
  bs = _batches(env)
  path = _write(tmp_path / "d", env, bs)
  r = _reader(path, env)
  assert r.num_rows == 2 * 16 and r.num_steps == 6
  assert r.producer == "123@100"
  helpers.assert_tree_equal(r.read_all(), _concat(bs))
  helpers.assert_tree_equal(r.rows(16, 16), bs[1])


def test_manifest_records_identity_and_infostate_leaves(env, tmp_path):
  """Manifest records identity and infostate leaves."""
  path = _write(tmp_path / "d", env, _batches(env, n=1))
  m = json.loads(
      open(os.path.join(path, shards.MANIFEST), encoding="utf-8").read()
  )
  assert (
      m["format"] == shards.MANIFEST_FORMAT
      and m["format_version"] == shards.FORMAT_VERSION
  )
  assert m["game"] == "kuhn_poker" and m["rules_sha"] == RULES
  assert m["info_schema_version"] == 1 and m["producer"] == "123@100"
  paths = [leaf["path"] for leaf in m["leaves"]]
  for f in kuhn.KuhnInfo._fields:  # the InfoState fields themselves, not an obs
    assert f".steps.info.{f}" in paths, f
  assert len(m["shards"]) == 1 and m["shards"][0]["rows"] == 16
  assert len(m["shards"][0]["sha256"]) == 64


@pytest.mark.parametrize(
    "kw,err",
    [
        (dict(rules_sha="other"), shards.Error),
        (dict(info_schema_version=2), shards.Error),
    ],
)
def test_reader_refuses_identity_mismatch(env, tmp_path, kw, err):
  """Reader refuses identity mismatch."""
  path = _write(tmp_path / "d", env, _batches(env, n=1))
  with pytest.raises(err):
    _reader(path, env, **kw)


def test_reader_refuses_other_game(env, tmp_path):
  """Reader refuses other game."""
  path = _write(tmp_path / "d", env, _batches(env, n=1))
  with pytest.raises(shards.Error):
    _reader(path, leduc.LeducEnv())


def test_reader_refuses_corrupt_shard(env, tmp_path):
  """Reader refuses corrupt shard."""
  path = _write(tmp_path / "d", env, _batches(env, n=1))
  f = os.path.join(path, "shard-00000.npz")
  data = bytearray(open(f, "rb").read())
  data[len(data) // 2] ^= 0xFF
  open(f, "wb").write(bytes(data))
  with pytest.raises(shards.Error):
    _reader(path, env)


def test_missing_manifest_raises(env, tmp_path):
  """Missing manifest raises."""
  with pytest.raises(FileNotFoundError):
    _reader(str(tmp_path / "nothing"), env)


def test_writer_refuses_obs_batch(env, tmp_path):
  """Writer refuses obs batch."""
  b = _batches(env, n=1)[0]
  obs = jax.nn.one_hot(kuhn.info_index(b.steps.info), 12, dtype=jnp.float32)
  bad = b._replace(steps=b.steps._replace(info=obs))
  with pytest.raises(shards.Error):
    _write(tmp_path / "d", env, [bad])


def test_writer_refuses_structure_change_between_writes(env, tmp_path):
  """Writer refuses structure change between writes."""
  b = _batches(env, n=1)[0]
  b2 = b._replace(labels={**b.labels, "extra": b.valid})
  with pytest.raises(shards.Error):
    _write(tmp_path / "d", env, [b, b2])


def test_writer_is_write_once(env, tmp_path):
  """Writer is write once."""
  path = _write(tmp_path / "d", env, _batches(env, n=1))
  with pytest.raises(shards.Error):
    _write(path, env, _batches(env, n=1))


def test_writer_requires_producer_and_rules(env, tmp_path):
  """Writer requires producer and rules."""
  with pytest.raises(AssertionError):
    shards.ShardWriter(
        str(tmp_path / "a"),
        env,
        rules_sha="",
        info_schema_version=1,
        producer="x@1",
    )
  with pytest.raises(AssertionError):
    shards.ShardWriter(
        str(tmp_path / "b"),
        env,
        rules_sha=RULES,
        info_schema_version=1,
        producer="",
    )


def test_artifact_id_forms():
  """Artifact id forms."""
  assert shards.artifact_id("321@3000") == "321@3000"
  man = types.Manifest(
      schema_version=2,
      game="g",
      rules_sha="r",
      obs_format="o",
      arch_name="a",
      arch_kwargs={},
      framework="jax",
      capability=types.Capability.NATIVE_JAX,
      default_rule=types.ARGMAX,
      params_sha256="s",
      run_id="321",
      step=3000,
      parents=(),
  )
  assert shards.artifact_id(man) == "321@3000"
  assert (
      shards.artifact_id(man._replace(run_id="kuhn-a")) == "kuhn-a@3000"
  )  # a named run
  art = types.Artifact(
      manifest=man._replace(run_id=None),
      params=None,
      policy=None,
      path="gs://b/x",
  )
  assert shards.artifact_id(art) == "gs://b/x@3000"
  with pytest.raises(AssertionError):
    shards.artifact_id(man._replace(run_id=None))


def test_next_samples_rows_deterministically(env, tmp_path):
  """Next samples rows deterministically."""
  bs = _batches(env)
  path = _write(tmp_path / "d", env, bs)
  r = _reader(path, env, batch_size=8)
  a, b = r.next(jax.random.key(0)), r.next(jax.random.key(0))
  helpers.assert_tree_equal(a, b)
  c = r.next(jax.random.key(1))
  assert not np.array_equal(
      np.asarray(a.steps.info.card), np.asarray(c.steps.info.card)
  ) or not np.array_equal(np.asarray(a.action), np.asarray(c.action))
  # Every sampled row is a stored row, rows not repeated within a draw.
  allb = _concat(bs)
  rows = r.last_rows
  assert len(set(rows.tolist())) == 8
  helpers.assert_tree_equal(c, jax.tree.map(lambda x: x[rows], allb))


def test_stateful_init_carry_round_trip(env, tmp_path):
  """Stateful init carry round trip."""
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  bs = _batches(env, n=2, student=st, sp=st.make_params())
  path = _write(tmp_path / "d", env, bs, carry_owner=st.spec)
  with pytest.raises(shards.Error):
    _reader(path, env)  # a stored carry needs its structure
  like = st.init_carry(
      st.make_params(), 1, jnp.zeros((1,), jnp.int32), jax.random.key(0)
  )
  r = _reader(path, env, init_carry_like=like)
  helpers.assert_tree_equal(r.read_all(), _concat(bs))


# --- a producer's carry is not another network's ------------------------------


def test_unaligned_valid_plies_known_answer():
  """Unaligned valid plies known answer."""
  nm = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 1]], bool)
  valid = np.array(
      [[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1]], bool
  )
  steps = types.SeatStep(
      info=None,
      legal_mask=None,
      seat=None,
      to_move=None,
      new_hand=nm,
      new_match=nm,
      last_action=None,
      last_actor=None,
  )
  got = shards.unaligned_valid_plies(steps, valid)
  assert got.dtype == np.int64
  np.testing.assert_array_equal(got, [0, 1, 2, 0])


def test_writer_refuses_a_carry_without_its_owner(env, tmp_path):
  """Writer refuses a carry without its owner."""
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  bs = _batches(env, n=1, student=st, sp=st.make_params())
  with pytest.raises(shards.Error, match="carry_owner"):
    _write(tmp_path / "d", env, bs)
  path = _write(tmp_path / "e", env, bs, carry_owner=st.spec)
  m = json.loads(
      open(os.path.join(path, shards.MANIFEST), encoding="utf-8").read()
  )
  assert m["carry_owner"] == shards.carry_owner_id(st.spec)
  assert m["carry_owner"]["arch_name"] == st.spec.arch_name


def _like(pol):
  """A batch-of-one carry of `pol`.

  Args:
    pol: The policy.

  Returns:
    The carry.
  """
  return pol.init_carry(
      pol.init_params(), 1, jnp.zeros((1,), jnp.int32), jax.random.key(0)
  )


def test_owner_student_reuses_the_stored_carry(env, tmp_path):
  """Owner student reuses the stored carry."""
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  bs = _batches(env, n=2, student=st, sp=st.make_params())
  path = _write(tmp_path / "d", env, bs, carry_owner=st.spec)
  r = _reader(path, env, init_carry_like=_like(st), student=st.spec)
  assert r.carry == "stored"
  helpers.assert_tree_equal(r.read_all(), _concat(bs))


def _mid_match(b, k=1):
  """B with every [N, T] leaf rotated k plies left, so rows open mid-hand.

  Args:
    b: The batch.
    k: The plies to rotate.

  Returns:
    The rotated batch.
  """

  def rot(x):
    """Rotates a leaf k plies left.

    Args:
      x: A leaf, [N, T, ...].

    Returns:
      The rotated leaf.
    """
    return jnp.roll(x, -k, axis=1)

  keep = {"init_carry", "bootstrap_step"}
  return b._replace(**{
      f: jax.tree.map(rot, getattr(b, f)) for f in b._fields if f not in keep
  })


def test_other_recurrent_student_rebuilds_or_refuses(env, tmp_path):
  """Other recurrent student rebuilds or refuses.

  A stored HistoryPolicy carry fed to a DIFFERENT stateful network (same tree
  shapes, other arch_kwargs): never reused. Batch 0 starts every row at a match
  start (reset-aligned): accepted with init_carry (). A batch whose rows open
  mid-hand: refused.

  Args:
    env: the kuhn env (fixture).
    tmp_path: pytest's temporary directory.
  """
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  other = fixtures.HistoryPolicy(
      "kuhn_poker", env.num_actions, env.num_players, cross_hand=True
  )
  assert other.spec.arch_kwargs != st.spec.arch_kwargs
  (b0,) = _batches(env, n=1, student=st, sp=st.make_params())
  b1 = _mid_match(b0)
  assert np.asarray(b0.steps.new_match[:, 0]).all()
  assert not shards.unaligned_valid_plies(b0.steps, b0.valid).any()
  assert shards.unaligned_valid_plies(
      b1.steps, b1.valid
  ).any()  # the precondition
  ok = _write(tmp_path / "ok", env, [b0], carry_owner=st.spec)
  r = _reader(ok, env, init_carry_like=_like(other), student=other.spec)
  assert r.carry == "student_init" and not r.read_all().init_carry
  helpers.assert_tree_equal(r.read_all(), b0._replace(init_carry=()))
  bad = _write(tmp_path / "bad", env, [b0, b1], carry_owner=st.spec)
  with pytest.raises(shards.Error, match="reset-aligned"):
    _reader(bad, env, init_carry_like=_like(other), student=other.spec)


def test_stateless_producer_feeds_a_recurrent_student_from_resets(
    env, tmp_path
):
  """Stateless producer feeds a recurrent student from resets.

  A stateless teacher's shard stores no carry: a recurrent student reads it when
  the rows start at resets (init_carry (), BC initialises it) and is refused on
  mid-match rows; a stateless student reads either.

  Args:
    env: the kuhn env (fixture).
    tmp_path: pytest's temporary directory.
  """
  (b0,) = _batches(env, n=1)
  b1 = _mid_match(b0)
  assert not jax.tree.leaves(b0.init_carry)
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  ok = _write(tmp_path / "ok", env, [b0])
  assert _reader(ok, env, student=st.spec).carry == "student_init"
  bad = _write(tmp_path / "bad", env, [b0, b1])
  with pytest.raises(shards.Error, match="stores no carry"):
    _reader(bad, env, student=st.spec)
  tab = helpers.kuhn_tabular(env)
  assert _reader(bad, env, student=tab.spec).carry == "none"


def test_stateless_student_ignores_a_stored_carry(env, tmp_path):
  """Stateless student ignores a stored carry."""
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  bs = _batches(env, n=2, student=st, sp=st.make_params())
  path = _write(tmp_path / "d", env, bs, carry_owner=st.spec)
  tab = helpers.kuhn_tabular(env)
  r = _reader(path, env, student=tab.spec)  # no init_carry_like needed
  assert r.carry == "none" and not r.read_all().init_carry


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
