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

"""The bridge deal canon on a synthetic bridge fixture.

The bridge deal canon on a synthetic bridge fixture: rotations and seat
permutations share a hash, every ply of a ROW_DEALS game is checked, and a
ShardWriter drops every row of an evaluation deal.
"""

import itertools
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import bridge
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.sources import deals
from open_spiel.python.scale.sources import shards

RS = bridge.RULES_SHA


def random_deals(n, seed):
  """int8[n, 52]: n random deals (the seat dealt each card).

  Args:
    n: the deal count.
    seed: the RNG seed.

  Returns:
    int8[n, 52].
  """
  rng = np.random.default_rng(seed)
  base = np.repeat(np.arange(4, dtype=np.int8), 13)
  return np.stack([rng.permutation(base) for _ in range(n)])


def rotate(deal, k=1):
  """A deal with every seat moved k seats on.

  Args:
    deal: int8[..., 52] deal holders.
    k: the rotation.

  Returns:
    The rotated deal.
  """
  return ((np.asarray(deal).astype(np.int64) + k) % 4).astype(np.int8)


def bridge_batch(deal_holder, plies_per_board=3):
  """T = 1 rows: plies_per_board rows per deal, only the first a hand start.

  Args:
    deal_holder: int8[N, 52] deals.
    plies_per_board: rows per deal.

  Returns:
    The TrajectoryBatch.
  """
  dh = np.repeat(np.asarray(deal_holder, np.int8), plies_per_board, axis=0)
  n = dh.shape[0]
  first = (np.arange(n) % plies_per_board) == 0
  like = bridge.info_template()
  info = {
      k: jnp.asarray(np.broadcast_to(v, (n, 1) + v.shape).copy())
      for k, v in like.items()
  }
  info["hand"] = jnp.asarray((dh == 0)[:, None, :])
  steps = types.SeatStep(
      info=info,
      legal_mask=jnp.ones((n, 1, 90), bool),
      seat=jnp.zeros((n, 1), jnp.int32),
      to_move=jnp.zeros((n, 1), jnp.int32),
      new_hand=jnp.asarray(first[:, None]),
      new_match=jnp.asarray(first[:, None]),
      last_action=jnp.full((n, 1), -1, jnp.int32),
      last_actor=jnp.full((n, 1), -1, jnp.int32),
  )

  def f32(*s):
    """Zeros [n, 1, *s].

    Args:
      *s: The trailing shape.

    Returns:
      The zeros.
    """
    return jnp.zeros((n, 1) + s, jnp.float32)

  return types.TrajectoryBatch(
      steps=steps,
      init_carry=(),
      action=jnp.arange(n, dtype=jnp.int32)[:, None] % 90,
      behavior_logits=f32(90),
      behavior_logp=f32(),
      value_pred=f32(4),
      target_policy=f32(90),
      target_value=f32(4),
      reward=f32(),
      done=jnp.zeros((n, 1), bool),
      valid=jnp.ones((n, 1), bool),
      labels={"deal_holder": jnp.asarray(dh[:, None, :])},
      aux={"search_q": f32(90)},
      bootstrap_step=jax.tree.map(lambda x: x[:, 0], steps),
  )


def eval_index(deal_holder):
  """The DealIndex of the given deals.

  Args:
    deal_holder: int8[N, 52] deals.

  Returns:
    The DealIndex.
  """
  return deals.DealIndex.from_labels(
      "bridge",
      RS,
      {"deal_holder": np.asarray(deal_holder)},
      [dict(name="eval", deals=len(deal_holder))],
  )


# -- the bridge deal canon -----------------------------------------------------


def test_bridge_hash_covers_rotations_and_every_seat_permutation():
  """Bridge hash covers rotations and every seat permutation."""
  d = random_deals(1, 0)[0]
  perms = np.stack(
      [np.asarray(p, np.int8)[d] for p in itertools.permutations(range(4))]
  )
  h = deals.deal_hashes("bridge", RS, None, {"deal_holder": perms})
  assert len(h) == 24 and len(set(h)) == 1
  assert (
      deals.deal_hashes("bridge", RS, None, {"deal_holder": rotate(d)[None]})
      == h[:1]
  )
  other = deals.deal_hashes(
      "bridge", RS, None, {"deal_holder": random_deals(5, 1)}
  )
  assert len(set(other)) == 5 and h[0] not in other
  assert (
      deals.deal_hashes("bridge", RS + "x", None, {"deal_holder": d[None]})
      != h[:1]
  )


def test_bridge_masks_are_the_hands():
  """Bridge masks are the hands."""
  d = random_deals(2, 3)
  masks = bridge.bridge_hand_masks(d)
  assert masks.dtype == np.uint64 and masks.shape == (2, 4)
  for r in range(2):
    for s in range(4):
      want = sum(1 << int(c) for c in np.flatnonzero(d[r] == s))
      assert int(masks[r, s]) == want
  assert (np.bitwise_or.reduce(masks, axis=1) == np.uint64((1 << 52) - 1)).all()


def test_bridge_canon_refuses_a_non_deal():
  """Bridge canon refuses a non deal."""
  d = random_deals(1, 4)
  bad = d.copy()
  bad[0, 0] = (bad[0, 0] + 1) % 4  # 14 cards in one hand, 12 in another
  with pytest.raises(ValueError, match="13 cards per seat"):
    deals.deal_hashes("bridge", RS, None, {"deal_holder": bad})
  with pytest.raises(ValueError, match="outside 0-3"):
    deals.deal_hashes(
        "bridge", RS, None, {"deal_holder": np.full((1, 52), 4, np.int8)}
    )


def test_every_ply_of_a_row_deals_game_is_hashed():
  """Every ply of a row deals game is hashed."""
  d = random_deals(2, 5)
  rows, hashes, unaligned = deals.batch_hashes("bridge", RS, bridge_batch(d, 3))
  assert rows.tolist() == list(range(6)) and not unaligned.any()
  assert len(set(hashes)) == 2


def test_writer_drops_every_row_of_an_eval_deal(tmp_path):
  """Writer drops every row of an eval deal."""
  d = random_deals(3, 6)
  idx = eval_index(rotate(d[1:2], 2))
  p = str(tmp_path / "w")
  with shards.ShardWriter(
      p,
      registry.make("bridge"),
      rules_sha=RS,
      info_schema_version=1,
      producer="t",
      deal_exclusion=idx,
  ) as w:
    w.write(bridge_batch(d, 3))
  m = json.loads(open(f"{p}/manifest.json", encoding="utf-8").read())
  assert m["deal_exclusion"] == dict(
      index_sha256=idx.sha256, refused_rows=3, checked_hands=9
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
