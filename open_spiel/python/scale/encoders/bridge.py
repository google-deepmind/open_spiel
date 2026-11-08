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

"""Contract bridge encoder over the envs.bridge InfoState.

``public_facts(info)`` replays a seat's view (its action list) into the
public state of the board: the auction's contract, declarer and double
status, who played each card, the tricks and the seat to act. The encoder
``bridge_info_state`` lays those out as one flat f32 vector with every seat
taken relative to the observer (rel = (seat - observer) % 4: 0 me, 1 left
opponent, 2 partner, 3 right opponent), so one network plays all four seats.
Both InfoState forms of envs.bridge are read: the raw action list, and the
compact history (calls, n_calls, play), which is expanded to the raw list
first, so the two encode equal.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import bridge as game
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import bridge

F32 = jnp.float32
I32 = jnp.int32
NUM_CARDS = game.NUM_CARDS
NUM_BIDS = game.NUM_BIDS
_INV_TRICKS = np.float32(1.0 / game.NUM_TRICKS)

PHASE_AUCTION, PHASE_PLAY, PHASE_OVER = 0, 1, 2


class PublicFacts(NamedTuple):
  """What every seat knows about the board, derived from an action list.

  Attributes:
    n_calls: i32, calls made.
    n_played: i32, cards played.
    phase: i32, PHASE_AUCTION / PHASE_PLAY / PHASE_OVER.
    level: i32, 1..7; 0 no bid yet.
    strain: i32, 0..4; -1 no bid yet.
    dbl: i32, 0 undoubled, 1 doubled, 2 redoubled.
    declarer: i32 seat; -1 no bid yet.
    opening_passes: bool[4], the seat passed before the first bid.
    bid_by: bool[35, 4], the seat made bid b.
    doubled_by: bool[35, 4], the seat doubled bid b.
    redoubled_by: bool[35, 4], the seat redoubled bid b.
    card_player: i32[52], the seat that played each card; -1 unplayed.
    card_trick: i32[52], the trick each card was played to; -1 unplayed.
    trick_leader: i32[13], the seat that led each trick; -1 not led.
    decl_tricks: i32, completed tricks won by the declaring side.
    on_move: i32, the seat whose call or card is due (dummy included); -1
      over.
  """

  n_calls: jax.Array  # i32 calls made
  n_played: jax.Array  # i32 cards played
  phase: jax.Array  # i32 PHASE_AUCTION / PHASE_PLAY / PHASE_OVER
  level: jax.Array  # i32 1..7, 0 no bid yet
  strain: jax.Array  # i32 0..4, -1 no bid yet
  dbl: jax.Array  # i32 0 undoubled, 1 doubled, 2 redoubled
  declarer: jax.Array  # i32 seat, -1 no bid yet
  opening_passes: jax.Array  # bool[4] seat passed before the first bid
  bid_by: jax.Array  # bool[35, 4] seat made bid b
  doubled_by: jax.Array  # bool[35, 4] seat doubled bid b
  redoubled_by: jax.Array  # bool[35, 4] seat redoubled bid b
  card_player: jax.Array  # i32[52] seat that played each card, -1 unplayed
  card_trick: jax.Array  # i32[52] trick each card was played to, -1 unplayed
  trick_leader: jax.Array  # i32[13] seat that led each trick, -1 not led
  decl_tricks: jax.Array  # i32 completed tricks won by the declaring side
  # i32 seat whose call or card is due (dummy included), -1 over
  on_move: jax.Array


def public_facts(info) -> PublicFacts:
  """Replays one (unbatched) InfoState's action list; vmap for a batch.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    The PublicFacts of the auction and play so far.
  """
  if "calls" in info:
    acts, count = bridge.expand_history(info)
    acts, count = acts.astype(I32), count.astype(I32)
  else:
    acts = jnp.asarray(info["actions"], I32)
    count = jnp.asarray(info["num_actions"], I32)
  dealer = jnp.asarray(info["dealer"], I32)
  k = jnp.arange(acts.shape[0], dtype=I32)
  valid = k < count
  is_call = valid & (acts >= game.PASS_ACTION)
  n_calls = jnp.sum(is_call).astype(I32)
  n_played = (count - n_calls).astype(I32)
  call = jnp.where(is_call, acts - game.PASS_ACTION, -1)
  caller = (dealer + k) % 4
  is_bid = call >= game.FIRST_BID_CALL
  bid = jnp.where(is_bid, call - game.FIRST_BID_CALL, -1)

  # The auction. Bids only go up, so the last bid is the largest.
  last_bid = jnp.max(bid)
  has_bid = last_bid >= 0
  level = jnp.where(has_bid, last_bid // game.NUM_STRAINS + 1, 0)
  strain = jnp.where(has_bid, last_bid % game.NUM_STRAINS, -1)
  last_bid_k = jnp.max(jnp.where(bid == last_bid, k, -1))
  after = k > last_bid_k
  redoubled = jnp.any(after & (call == game.CALL_REDOUBLE))
  doubled = jnp.any(after & (call == game.CALL_DOUBLE))
  dbl = jnp.where(redoubled, 2, jnp.where(doubled, 1, 0))
  bidder = jnp.where(last_bid_k >= 0, caller[jnp.maximum(last_bid_k, 0)], -1)
  same_strain = (
      is_bid
      & (bid % game.NUM_STRAINS == strain)
      & (game.partnership(caller) == game.partnership(bidder))
  )
  first_k = jnp.min(jnp.where(same_strain, k, acts.shape[0]))
  declarer = jnp.where(
      has_bid, caller[jnp.minimum(first_k, acts.shape[0] - 1)], -1
  )

  seat_hot = jax.nn.one_hot(caller, 4, dtype=jnp.bool_)  # [K, 4]
  first_bid_k = jnp.min(jnp.where(is_bid, k, acts.shape[0]))
  opening_passes = jnp.any(
      seat_hot & ((call == game.CALL_PASS) & (k < first_bid_k))[:, None], axis=0
  )
  # The bid each double / redouble applies to: the last bid before it.
  standing = jax.lax.cummax(bid)
  bid_hot = jax.nn.one_hot(
      jnp.where(is_bid, bid, -1), NUM_BIDS, dtype=jnp.bool_
  )
  stand_hot = jax.nn.one_hot(standing, NUM_BIDS, dtype=jnp.bool_)

  def per_seat(hot, m):  # bool[35, 4]: a call of kind m on bid b by each seat
    """Which seat made a call of kind `m` on each bid.

    Args:
      hot: bool[T, 35], the bid of each call.
      m: bool[T], the calls of the kind.

    Returns:
      bool[35, 4]: a call of kind m on bid b by each seat.
    """
    return jnp.any(
        hot[:, :, None] & m[:, None, None] & seat_hot[:, None, :], axis=0
    )

  bid_by = per_seat(bid_hot, is_bid)
  doubled_by = per_seat(stand_hot, call == game.CALL_DOUBLE)
  redoubled_by = per_seat(stand_hot, call == game.CALL_REDOUBLE)

  passes = (call == game.CALL_PASS) & is_call
  non_pass_k = jnp.max(jnp.where(is_call & ~passes, k, -1))
  trailing = n_calls - 1 - non_pass_k
  auction_over = jnp.where(has_bid, trailing >= 3, n_calls >= 4)

  # The play: card j of the play is acts[n_calls + j].
  j = jnp.arange(NUM_CARDS, dtype=I32)
  cards = acts[jnp.clip(n_calls + j, 0, acts.shape[0] - 1)]
  played = j < n_played
  cards = jnp.where(played, cards, 0).reshape(game.NUM_TRICKS, 4)
  played = played.reshape(game.NUM_TRICKS, 4)

  def trick(leader, xs):
    """One played card: scan step over the play.

    Args:
      leader: the seat that leads the current trick.
      xs: the scanned card and trick facts.

    Returns:
      (next leader, (leader of a led card or -1, declarer side won)).
    """
    tc, tp = xs
    suits = tc % 4
    value = (
        tc // 4
        + game.NUM_RANKS * (suits == suits[0])
        + 2 * game.NUM_RANKS * (suits == strain)
    )
    winner = ((leader + jnp.argmax(value).astype(I32)) % 4).astype(I32)
    complete = jnp.all(tp)
    led = tp[0]
    nxt = jnp.where(complete, winner, leader).astype(I32)
    return nxt, (
        jnp.where(led, leader, -1),
        complete & (game.partnership(winner) == game.partnership(declarer)),
        leader,
    )

  first_leader = ((declarer + 1) % 4).astype(I32)
  last_leader, (trick_leader, won, leaders) = jax.lax.scan(
      trick, first_leader, (cards, played)
  )
  pos = jnp.arange(4, dtype=I32)
  players = (leaders[:, None] + pos[None, :]) % 4  # [13, 4]
  tricks = jnp.broadcast_to(
      jnp.arange(game.NUM_TRICKS, dtype=I32)[:, None], (13, 4)
  )
  flat_cards, flat_played = cards.reshape(-1), played.reshape(-1)
  card_player = (
      jnp.full((NUM_CARDS,), -1, I32)
      .at[jnp.where(flat_played, flat_cards, NUM_CARDS)]
      .set(players.reshape(-1), mode="drop")
  )
  card_trick = (
      jnp.full((NUM_CARDS,), -1, I32)
      .at[jnp.where(flat_played, flat_cards, NUM_CARDS)]
      .set(tricks.reshape(-1), mode="drop")
  )
  decl_tricks = jnp.sum(won).astype(I32)

  over = auction_over & (~has_bid | (n_played == NUM_CARDS))
  phase = jnp.where(
      over, PHASE_OVER, jnp.where(auction_over, PHASE_PLAY, PHASE_AUCTION)
  )
  play_on_move = (last_leader + n_played % 4) % 4
  on_move = jnp.where(
      phase == PHASE_AUCTION,
      (dealer + n_calls) % 4,
      jnp.where(phase == PHASE_PLAY, play_on_move, -1),
  )
  return PublicFacts(
      n_calls=n_calls,
      n_played=n_played,
      phase=phase.astype(I32),
      level=level.astype(I32),
      strain=strain.astype(I32),
      dbl=dbl.astype(I32),
      declarer=declarer.astype(I32),
      opening_passes=opening_passes,
      bid_by=bid_by,
      doubled_by=doubled_by,
      redoubled_by=redoubled_by,
      card_player=card_player,
      card_trick=card_trick,
      trick_leader=trick_leader.astype(I32),
      decl_tricks=decl_tricks,
      on_move=on_move.astype(I32),
  )


def _rel(seat, observer):
  """Seat relative to the observer; -1 (no seat) stays -1.

  Args:
    seat: The seat, or -1.
    observer: The observing seat.

  Returns:
    The relative seat, or -1.
  """
  return jnp.where(seat >= 0, (seat - observer) % 4, -1)


def info_state(info) -> jax.Array:
  """The perfect-recall bridge encoding, every seat relative to the observer.

  Args:
    info: one (unbatched) InfoState.

  Returns:
    f32[DIM]:

      own remaining cards (52) ++ dummy's remaining cards (52, zero until
      dummy is face up) ++ cards played, by rel seat of the player (4 x 52)
      ++ the trick in progress, by rel seat (4 x 52)
      ++ opening passes by rel seat (4) ++ per bid 1C..7NT: bid / doubled /
      redoubled, by rel seat (35 x 3 x 4)
      ++ vulnerable: my side, the other side (2) ++ rel dealer (4)
      ++ contract level (7), strain (5), double status (3), rel declarer (4)
      ++ rel seat to act, dummy included (4) ++ phase: auction, play, over (3)
      ++ tricks won by the declaring side and by the defenders, / 13 (2).

    Unset one-hots (no contract yet; nobody to act at the end) are zeros.
  """
  f = public_facts(info)
  me = jnp.asarray(info["seat"], I32)
  rel_order = (me + jnp.arange(4, dtype=I32)) % 4  # rel r -> absolute seat
  played = f.card_player >= 0
  hand = jnp.asarray(info["hand"], jnp.bool_)
  dummy = jnp.asarray(info["dummy"], jnp.bool_)
  player_rel = _rel(f.card_player, me)
  by_rel = jax.nn.one_hot(player_rel, 4, dtype=F32).T  # [4, 52]; -1 -> zeros
  current = f.n_played // 4
  in_trick = played & (f.card_trick == current) & (f.phase == PHASE_PLAY)
  my_side = game.partnership(me)
  vul = jnp.asarray(info["vul"], jnp.bool_)
  # [35, 3, 4]
  auction = jnp.stack([f.bid_by, f.doubled_by, f.redoubled_by], axis=1)
  defence = jnp.maximum(current - f.decl_tricks, 0)
  out = jnp.concatenate([
      (hand & ~played).astype(F32),
      (dummy & ~played).astype(F32),
      by_rel.reshape(-1),
      (by_rel * in_trick[None, :].astype(F32)).reshape(-1),
      f.opening_passes[rel_order].astype(F32),
      auction[:, :, rel_order].astype(F32).reshape(-1),
      jnp.stack([vul[my_side], vul[1 - my_side]]).astype(F32),
      jax.nn.one_hot(_rel(jnp.asarray(info["dealer"], I32), me), 4, dtype=F32),
      jax.nn.one_hot(f.level - 1, game.NUM_LEVELS, dtype=F32),
      jax.nn.one_hot(f.strain, game.NUM_STRAINS, dtype=F32),
      jax.nn.one_hot(jnp.where(f.level > 0, f.dbl, -1), 3, dtype=F32),
      jax.nn.one_hot(_rel(f.declarer, me), 4, dtype=F32),
      jax.nn.one_hot(_rel(f.on_move, me), 4, dtype=F32),
      jax.nn.one_hot(f.phase, 3, dtype=F32),
      jnp.stack([f.decl_tricks, defence]).astype(F32) * _INV_TRICKS,
  ])
  return out


DIM = (
    2 * NUM_CARDS
    + 2 * 4 * NUM_CARDS
    + 4
    + NUM_BIDS * 3 * 4
    + 2
    + 4
    + game.NUM_LEVELS
    + game.NUM_STRAINS
    + 3
    + 4
    + 4
    + 3
    + 2
)  # 978

INFO_STATE = registry.register(
    registry.Encoder(
        name="bridge_info_state",
        game=bridge.BridgeEnv.name,
        shape=(DIM,),
        dtype=F32,
        fn=info_state,
    )
)
