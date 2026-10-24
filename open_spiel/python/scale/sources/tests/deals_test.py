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

"""sources.deals and its shard enforcement.

sources.deals and its shard enforcement: canonical deal hashes, the eval
deal index, the writer dropping and counting eval-deal rows, the reader
refusing an unchecked shard or one holding an eval deal; and the stub env's
InfoState template path.
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import arena_engine
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import stub
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.sources import deals
from open_spiel.python.scale.sources import shards

SEED, PAIRS = 1234567, 6


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A gin env with Oklahoma rules.

  Returns:
    The env.
  """
  return gin.GinEnv(oklahoma_frac=1.0)


def test_canonical_hash_is_seat_independent_and_rules_bound(env):
  """Canonical hash is seat independent and rules bound."""
  st = deals.eval_deal_states(env, SEED, 3)
  rs = env_registry.rules_sha(env)
  h0 = deals.state_hashes(env, rs, st)
  seat1 = jnp.ones((3,), jnp.int32)
  h1 = deals.deal_hashes(
      env.name,
      rs,
      jax.vmap(env.info_state)(st, seat1),
      jax.vmap(env.labels)(st, seat1),
  )
  assert h0 == h1 and len(set(h0)) == 3
  assert deals.state_hashes(env, rs + "x", st) != h0
  with pytest.raises(KeyError, match="no canonical deal form"):
    deals.canon_of("leduc_poker")


def test_index_round_trip(env, tmp_path):
  """Index round trip."""
  idx = deals.DealIndex.from_eval_sets(
      env, [("eval", SEED, PAIRS), ("bias", SEED + 1, 2)]
  )
  assert len(idx.hashes) == PAIRS + 2 and idx.sources[0]["deals"] == PAIRS
  sha = idx.save(str(tmp_path / "i.json"))
  back = deals.DealIndex.load(str(tmp_path / "i.json"))
  assert back.sha256 == sha and back.hashes == idx.hashes


def _batch_with_eval_deal(env, eval_rows=(0,)):
  """A gin batch of 4 hand-aligned rows, some dealt from the eval set.

  A gin batch of 4 hand-aligned rows; rows in eval_rows are dealt from
  the eval set's deal keys (pairs 0, 1, ...).

  Args:
    env: the gin env (fixture).
    eval_rows: the rows dealt from the eval set.

  Returns:
    The TrajectoryBatch.
  """
  dk_eval, _ = arena_engine.deal_keys(jax.random.key(SEED), PAIRS, True)
  dk_eval = dk_eval[::2]
  table = arena_lib.self_play_table(4, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(99), 4, 2)
  for j, r in enumerate(eval_rows):
    dk = dk.at[r].set(dk_eval[j])
  pol = uniform.UniformPolicy("gin_rummy", env.num_actions, 2)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  params = [{}]
  st = ar.reset(params, table, dk, pk)
  _, batch, _ = ar.rollout(params, table, st, 4)
  return batch


def test_writer_drops_and_counts_eval_deals_reader_refuses_unchecked(
    env, tmp_path
):
  """Writer drops and counts eval deals reader refuses unchecked."""
  rs = env_registry.rules_sha(env)
  idx = deals.DealIndex.from_eval_sets(env, [("eval", SEED, PAIRS)])
  batch = _batch_with_eval_deal(env, eval_rows=(0, 2))
  _, hashes, unaligned = deals.batch_hashes(env.name, rs, batch)
  assert not unaligned.any() and idx.hits(hashes).sum() == 2
  p = str(tmp_path / "clean")
  with shards.ShardWriter(
      p,
      env,
      rules_sha=rs,
      info_schema_version=1,
      producer="t",
      deal_exclusion=idx,
  ) as w:
    w.write(batch)
  m = json.loads(open(f"{p}/manifest.json", encoding="utf-8").read())
  assert m["deal_exclusion"] == dict(
      index_sha256=idx.sha256, refused_rows=2, checked_hands=4
  )
  r = shards.ShardReader(
      p, env, rules_sha=rs, info_schema_version=1, deal_exclusion=idx
  )
  assert r.num_rows == 2
  # A shard written without the index: the exit-mode reader refuses it.
  q = str(tmp_path / "unchecked")
  with shards.ShardWriter(
      q, env, rules_sha=rs, info_schema_version=1, producer="t"
  ) as w:
    w.write(batch)
  with pytest.raises(shards.Error, match="without the eval deal index"):
    shards.ShardReader(
        q, env, rules_sha=rs, info_schema_version=1, deal_exclusion=idx
    )
  shards.ShardReader(
      q, env, rules_sha=rs, info_schema_version=1
  )  # no index given: reads
  # A shard holding one eval deal (its manifest forged) is refused on read.
  mq = json.loads(open(f"{q}/manifest.json", encoding="utf-8").read())
  mq["deal_exclusion"] = dict(
      index_sha256=idx.sha256, refused_rows=0, checked_hands=4
  )
  open(f"{q}/manifest.json", "w", encoding="utf-8").write(json.dumps(mq))
  with pytest.raises(shards.Error, match="start from an eval deal"):
    shards.ShardReader(
        q, env, rules_sha=rs, info_schema_version=1, deal_exclusion=idx
    )


def test_writer_refuses_rows_whose_deal_is_not_in_the_row(env, tmp_path):
  """Writer refuses rows whose deal is not in the row."""
  rs = env_registry.rules_sha(env)
  idx = deals.DealIndex.from_eval_sets(env, [("eval", SEED, 2)])
  batch = _batch_with_eval_deal(env, eval_rows=())
  cut = jax.tree_util.tree_map(
      lambda x: x[:, 1:], batch._replace(init_carry=(), bootstrap_step=())
  )
  cut = cut._replace(bootstrap_step=batch.bootstrap_step)
  with pytest.raises(shards.Error, match="before their first hand start"):
    with shards.ShardWriter(
        str(tmp_path / "x"),
        env,
        rules_sha=rs,
        info_schema_version=1,
        producer="t",
        deal_exclusion=idx,
    ) as w:
      w.write(cut)


def test_stub_env_template_feeds_the_shard_path(tmp_path):
  """Stub env template feeds the shard path."""
  info_like = {
      "hand": np.zeros((52,), np.int8),
      "auction": np.zeros((40,), np.int32),
  }
  e = stub.StubEnv("bridge_stub_test", 4, 90, 128, info_like)
  with pytest.raises(NotImplementedError, match="stub"):
    e.init(jax.random.key(0))
  assert set(shards.info_template(e)) == {"hand", "auction"}
  b, t = 3, 2
  info = {
      "hand": jnp.zeros((b, t, 52), jnp.int8),
      "auction": jnp.zeros((b, t, 40), jnp.int32),
  }
  steps = types.SeatStep(
      info=info,
      legal_mask=jnp.ones((b, t, 90), bool),
      seat=jnp.zeros((b, t), jnp.int32),
      to_move=jnp.zeros((b, t), jnp.int32),
      new_hand=jnp.zeros((b, t), bool).at[:, 0].set(True),
      new_match=jnp.zeros((b, t), bool).at[:, 0].set(True),
      last_action=jnp.full((b, t), -1, jnp.int32),
      last_actor=jnp.full((b, t), -1, jnp.int32),
  )
  batch = types.TrajectoryBatch(
      steps=steps,
      init_carry=(),
      action=jnp.zeros((b, t), jnp.int32),
      behavior_logits=jnp.zeros((b, t, 90), jnp.float32),
      behavior_logp=jnp.zeros((b, t), jnp.float32),
      value_pred=jnp.zeros((b, t, 4), jnp.float32),
      target_policy=jnp.zeros((b, t, 90), jnp.float32),
      target_value=jnp.zeros((b, t, 4), jnp.float32),
      reward=jnp.zeros((b, t), jnp.float32),
      done=jnp.zeros((b, t), bool),
      valid=jnp.ones((b, t), bool),
      labels={},
      aux={},
      bootstrap_step=jax.tree_util.tree_map(lambda x: x[:, 0], steps),
  )
  p = str(tmp_path / "bridge")
  with shards.ShardWriter(
      p,
      e,
      rules_sha="bridge-rules-x",
      info_schema_version=1,
      producer="labeller@sha",
  ) as w:
    w.write(batch)
  r = shards.ShardReader(
      p, e, rules_sha="bridge-rules-x", info_schema_version=1
  )
  assert r.num_rows == 3
  with pytest.raises(shards.Error, match="rules_sha"):
    shards.ShardReader(p, e, rules_sha="other", info_schema_version=1)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
