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

"""gin_card_tokens: the gin card-token encoding, from GinInfo.

The input of the gin card-token transformer (`networks.gin_tfv2`): 799
floats, laid out as `encoders.card_tokens_layout` says, with the Oklahoma
knock limit one-hot and no meld-membership block:

    [ card block   52 x (hand, discard_pile, opp_known, is_upcard)
    | global block phase one-hot (8), upcard_norm, deck_count, deadwood,
                   deadwood_with_upcard, knock limit one-hot (11), now
    | touch block  189 x (card + 1, kind, t + 1), oldest first, then zeros ]

The four scalars are bfloat16 values ((upcard + 1) / 53, stock / 52,
deadwood / 100 of the own hand and of hand + upcard), computed in f32,
rounded to bfloat16 and widened back to f32 (`bf16_ratio`). `is_upcard` is
the card equal to the upcard. `now` is the number of events. A touch row is
(card + 1, kind, t + 1) for each event t (kind = 2 * (type - 1) + actor
code, card 0 for a knock) and, right after the observer's own stock draw,
(drawn card + 1, KIND_OBS_STOCK_CARD, t + 1).

Errors are never dropped: a buffer error carried by the InfoState, more
touch rows than 189, a private card on a row that is not the observer's own
stock draw, or a transcript that is not a contiguous prefix of valid codes
makes the whole vector NaN (NaN fails the Arena's bad-ply check in-graph,
with no host sync).

`encode_opp_blind` is the same with the opp_known column zeroed (a view
that ignores which of the opponent's cards are publicly known).
"""

import jax.numpy as jnp

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.jax.games.gin_rummy import deadwood
from open_spiel.python.scale.encoders import card_tokens_layout as ct
from open_spiel.python.scale.encoders import registry

NAME = "gin_card_tokens"

# The GinInfo event transcript's codes (envs.gin defines them; an encoder
# reads the InfoState only and does not import the env, tests pin equality).
#: The most events one hand holds: 1 pass + 2 x (1 take + 29 stock draws + 49
#: upcard draws) + 1 wall knock.
EVENTS_MAX = (
    1
    + 2
    * (
        1
        + (cm.NUM_CARDS - 2 * cm.HAND_SIZE - 1 - cm.WALL_STOCK_SIZE)
        + cm.MAX_NUM_DRAW_UPCARD_ACTIONS
        - 1
    )
    + 1
)
#: GinInfo.ev_actor codes (relative to the observing seat).
ACTOR_PAD, ACTOR_SELF, ACTOR_OPP = 0, 1, 2
#: GinInfo.ev_type codes used here.
EVT_DRAW_STOCK, EVT_KNOCK = 3, 5

NUM_PHASES = 8
KNOCK_LIMIT_VALUES = cm.KNOCK_THRESHOLD + 1  # 11
# 29
MAX_STOCK_DRAWS = cm.NUM_CARDS - 2 * cm.HAND_SIZE - 1 - cm.WALL_STOCK_SIZE
# The format constants the torso reads are defined once, in
# encoders.card_tokens_layout; the encoder derives its own sizes from the game
# and asserts they agree.
TOUCH_ROWS = ct.TOUCH_ROWS
assert EVENTS_MAX + MAX_STOCK_DRAWS == TOUCH_ROWS, (
    EVENTS_MAX,
    MAX_STOCK_DRAWS,
)
assert EVENTS_MAX == ct.EVENTS_MAX and cm.NUM_CARDS == ct.NUM_CARDS
CARD_FIELDS = 4
GLOBAL_FIELDS = NUM_PHASES + 4 + KNOCK_LIMIT_VALUES + 1  # 24
DIM = ct.dim(CARD_FIELDS, GLOBAL_FIELDS)
assert DIM == 799, DIM
KIND_OBS_STOCK_CARD = ct.NUM_KINDS - 1  # 11, the last kind code
assert ct.TOUCH_FIELDS == 3
_BF16 = jnp.bfloat16


def bf16_ratio(num, den: float):
  """Divides in f32, rounds to bfloat16 and widens back to f32.

  Args:
    num: the numerator (any numeric array).
    den: the denominator.

  Returns:
    f32, holding a bfloat16 value.
  """
  return (
      (num.astype(jnp.float32) / jnp.float32(den))
      .astype(_BF16)
      .astype(jnp.float32)
  )


def touch_block(info):
  """The touch rows, the event count and the error flag.

  Args:
    info: one GinInfo.

  Returns:
    (touch rows f32[189 * 3], event count i32, error bool).
  """
  i32 = jnp.int32
  actor = info.ev_actor.astype(i32)
  etype = info.ev_type.astype(i32)
  public = info.ev_public.astype(i32)
  private = info.ev_private.astype(i32)
  t = jnp.arange(EVENTS_MAX, dtype=i32)
  valid = actor != ACTOR_PAD
  count = jnp.sum(valid.astype(i32))
  has_priv = valid & (private != 0)
  hp = has_priv.astype(i32)
  kind_pub = jnp.where(valid, 2 * (etype - 1) + actor, 0)
  row_pub = t + jnp.cumsum(hp) - hp  # t + private rows of earlier events
  row_priv = row_pub + 1
  # Each event's rows by a scatter: the public rows are distinct, the private
  # rows distinct and disjoint from them, so the add is a placement; rows
  # past TOUCH_ROWS (an overflow, flagged in err) and absent events go to the
  # out-of-range index and are dropped.
  oob = jnp.int32(TOUCH_ROWS)
  idx_pub = jnp.where(valid, row_pub, oob)
  idx_priv = jnp.where(has_priv, row_priv, oob)

  def gather(field_pub, field_priv):
    """One touch column: public rows, then private rows.

    Args:
      field_pub: the column's value on each event's public row.
      field_priv: the column's value on each event's private row.

    Returns:
      i32[TOUCH_ROWS].
    """
    z = jnp.zeros((TOUCH_ROWS,), i32)
    z = z.at[idx_pub].add(field_pub.astype(i32), mode="drop")
    return z.at[idx_priv].add(field_priv.astype(i32), mode="drop")

  touch = jnp.stack(
      [
          gather(public, private),
          gather(kind_pub, jnp.full_like(t, KIND_OBS_STOCK_CARD)),
          gather(t + 1, t + 1),
      ],
      axis=-1,
  )
  needed = count + jnp.sum(hp)
  bad_priv = jnp.any(
      has_priv & ~((actor == ACTOR_SELF) & (etype == EVT_DRAW_STOCK))
  )
  rows_any = (actor != 0) | (etype != 0) | (public != 0) | (private != 0)
  bad_layout = (
      jnp.any(valid != (t < count))
      | jnp.any(
          valid & ((actor > ACTOR_OPP) | (etype < 1) | (etype > EVT_KNOCK))
      )
      | jnp.any(~valid & rows_any)
      | jnp.any((public < 0) | (public > cm.NUM_CARDS))
      | jnp.any((private < 0) | (private > cm.NUM_CARDS))
      | (count != info.ev_count.astype(i32))
  )
  err = (needed > TOUCH_ROWS) | bad_priv | bad_layout | (info.ev_error != 0)
  return touch.reshape(-1).astype(jnp.float32), count, err


def _encode(info, opp_known_live: bool):
  """The card-token vector of one GinInfo.

  Args:
    info: one GinInfo.
    opp_known_live: fill the opp_known card column (else zeros).

  Returns:
    f32[799], NaN everywhere on an error.
  """
  i32 = jnp.int32
  seat = info.seat.astype(i32)
  hand = info.hand
  cards = jnp.arange(cm.NUM_CARDS, dtype=i32)
  upcard = info.upcard.astype(i32)
  opp_known = jnp.where(seat == 0, info.known_cards[1], info.known_cards[0])
  opp_col = opp_known.astype(jnp.float32) * jnp.float32(
      1.0 if opp_known_live else 0.0
  )
  card_block = jnp.stack(
      [
          hand.astype(jnp.float32),
          info.discard_pile.astype(jnp.float32),
          opp_col,
          (cards == upcard).astype(jnp.float32),
      ],
      axis=-1,
  )

  dw = deadwood.calculate_deadwood_lut(hand).astype(i32)
  hand_up = jnp.where(upcard >= 0, hand.at[upcard].set(jnp.int8(1)), hand)
  dw_up = jnp.where(
      upcard >= 0, deadwood.calculate_deadwood_lut(hand_up).astype(i32), dw
  )
  touch, count, err = touch_block(info)
  glob = jnp.concatenate([
      (jnp.arange(NUM_PHASES, dtype=i32) == info.phase.astype(i32)).astype(
          jnp.float32
      ),
      jnp.stack([
          bf16_ratio(upcard + 1, 53.0),
          bf16_ratio(info.stock_count.astype(i32), 52.0),
          bf16_ratio(dw, 100.0),
          bf16_ratio(dw_up, 100.0),
      ]),
      (
          jnp.arange(KNOCK_LIMIT_VALUES, dtype=i32)
          == info.knock_card.astype(i32)
      ).astype(jnp.float32),
      count.astype(jnp.float32)[None],
  ])
  x = jnp.concatenate([card_block.reshape(-1), glob, touch])
  return jnp.where(err, jnp.float32(jnp.nan), x)


def encode(info):
  """The card-token vector of one GinInfo, opp_known filled.

  Args:
    info: one (unbatched) GinInfo.

  Returns:
    f32[799], NaN everywhere on an error.
  """
  return _encode(info, opp_known_live=True)


def encode_opp_blind(info):
  """The card-token vector of one GinInfo with the opp_known column zeroed.

  Args:
    info: one (unbatched) GinInfo.

  Returns:
    f32[799], NaN everywhere on an error.
  """
  return _encode(info, opp_known_live=False)


registry.register(
    registry.Encoder(
        name=NAME,
        game="gin_rummy",
        shape=(DIM,),
        dtype=jnp.float32,
        fn=encode,
        reference=(
            "the gin card-token transformer's input (card, global and"
            " event-touch blocks)"
        ),
    )
)
