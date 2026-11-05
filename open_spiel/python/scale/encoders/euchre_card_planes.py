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

"""Euchre card planes over the envs.euchre InfoState: f32[64, 24].

Importing this module registers the encoder; encoders.registry.LAZY names
this module, so a lookup of the name imports it (no import by the caller).

One column per card (card = rank * 4 + suit, ranks 9 T J Q K A, suits
C D H S), one row per channel (CHANNELS), from one seat's view; seats are
relative to the viewer (rel 0 the viewer, 1 its left-hand opponent, 2 its
partner, 3 its right-hand opponent). Only what the seat may know is encoded:
its hand, its own discard (the dealer only), the up-card and whether the
dealer took it or it was turned down, the public bidding, every played card
(who played it, in which trick, whether it was led) and the voids those
plays reveal. Partners share no hidden information.

  25 card planes: my_hand, upcard, upcard_with_dealer, upcard_turned_down,
    my_discard, played_by_rel0..3, played_in_trick0..4, led_card,
    in_current_trick, winning_current_trick, is_trump, eff_rank_over7,
    same_suit_as_upcard, follows_led_suit, void_rel1..3, unseen.
  39 broadcast planes (one value over every card): dealer_rel0..3,
    phase0..4 (first-round bid, second-round bid, the dealer's discard, the
    maker's go-alone decision, play), terminal, actor_rel0..3,
    maker_rel0..3, trump0..3, alone, passed1_rel0..3, passed2_rel0..3,
    tricks_us_over5, tricks_them_over5, leader_rel0..3, n_in_trick_over4,
    trick_num_over5.

Effective suits and ranks: the left bower (the jack of trump's colour) is
trump; within a suit the right bower ranks 7, the left bower 6, else the
printed rank 0..5; before trump is named the printed rank.

The dealer's discard: a seat other than the dealer cannot tell, from its
InfoState, whether the dealer has discarded yet (the discard is the one
hidden action), so before the first card is played it sees the play phase
with the first leader on move from the moment the dealer takes the up-card.
The first leader (never the dealer) decides there, so its decision planes
are exact; only the waiting seats' views of the discard ply differ from a
view that names the discard phase.

``waiting_row`` is a value-row predicate (learners.pg PGConfig.value_rows)
for self-play over all four seats with a round-robin row clock (row r
belongs to seat r % 4; a seat not on move waits on its row): at the ply
where seat Y is about to move after seat X moved, the seats strictly
between X and Y in seat order (all three others when X is Y: the maker's
go-alone decision right after its call) are the seats such a clock reaches
while Y has yet to move. Left out: the first bid of a hand (where the clock
stands then depends on the previous hand, not on this InfoState) and, for a
seat other than the dealer, the plies between the dealer's taking the
up-card and the first lead (the discard is hidden from it, so it cannot
tell the discard ply from the first lead).
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import euchre as game
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import euchre

F32 = jnp.float32
I32 = jnp.int32
NAME = "euchre_card_planes"
N_CARD = game.NUM_CARDS
N_PLAYER = game.NUM_PLAYERS
N_SUIT = game.NUM_SUITS
N_TRICK = game.NUM_TRICKS
N_PLAYS = game.NUM_HAND_CARDS
JACK_RANK = game.JACK_RANK
#: The phases of the phase planes.
PHASE_BID1, PHASE_BID2, PHASE_DISCARD, PHASE_ALONE, PHASE_PLAY, PHASE_OVER = (
    range(6)
)

CARD_PLANES = (
    (
        "my_hand",
        "upcard",
        "upcard_with_dealer",
        "upcard_turned_down",
        "my_discard",
    )
    + tuple(f"played_by_rel{k}" for k in range(N_PLAYER))
    + tuple(f"played_in_trick{t}" for t in range(N_TRICK))
    + ("led_card", "in_current_trick", "winning_current_trick")
    + ("is_trump", "eff_rank_over7", "same_suit_as_upcard", "follows_led_suit")
    + tuple(f"void_rel{k}" for k in (1, 2, 3))
    + ("unseen",)
)
BROADCAST_PLANES = (
    tuple(f"b_dealer_rel{k}" for k in range(N_PLAYER))
    + tuple(f"b_phase{p}" for p in range(5))
    + ("b_terminal",)
    + tuple(f"b_actor_rel{k}" for k in range(N_PLAYER))
    + tuple(f"b_maker_rel{k}" for k in range(N_PLAYER))
    + tuple(f"b_trump{s}" for s in range(N_SUIT))
    + ("b_alone",)
    + tuple(f"b_passed1_rel{k}" for k in range(N_PLAYER))
    + tuple(f"b_passed2_rel{k}" for k in range(N_PLAYER))
    + ("b_tricks_us_over5", "b_tricks_them_over5")
    + tuple(f"b_leader_rel{k}" for k in range(N_PLAYER))
    + ("b_n_in_trick_over4", "b_trick_num_over5")
)
CHANNELS = CARD_PLANES + BROADCAST_PLANES
N_CHANNEL = len(CHANNELS)
assert N_CHANNEL == 64 and len(set(CHANNELS)) == N_CHANNEL


def eff_suit(card, trump):
  """The effective suit: the left bower is trump; trump < 0 the printed suit.

  Args:
    card: i32 card id(s).
    trump: i32 trump suit or -1.

  Returns:
    i32 suit(s).
  """
  s = card % N_SUIT
  left = (card // N_SUIT == JACK_RANK) & (trump >= 0) & (s == 3 - trump)
  return jnp.where(left, trump, s)


def eff_rank(card, trump):
  """The rank within the effective suit: right bower 7, left bower 6, else 0..5.

  Args:
    card: i32 card id(s).
    trump: i32 trump suit or -1.

  Returns:
    i32 rank(s).
  """
  s = card % N_SUIT
  r = card // N_SUIT
  jack = (r == JACK_RANK) & (trump >= 0)
  return jnp.where(
      jack & (s == trump), 7, jnp.where(jack & (s == 3 - trump), 6, r)
  )


def _rel_onehot(seat, viewer):
  """bool[4]: `seat` relative to `viewer`; all False when seat < 0.

  Args:
    seat: i32 seat or -1.
    viewer: i32 seat.

  Returns:
    bool[4].
  """
  rel = (seat - viewer) % N_PLAYER
  return (jnp.arange(N_PLAYER) == rel) & (seat >= 0)


def _facts(info):
  """The bidding, phase and play-history facts of one seat's InfoState.

  Args:
    info: one seat's InfoState (envs.euchre).

  Returns:
    A dict of arrays (see the keys).
  """
  viewer = info["seat"].astype(I32)
  dealer = info["dealer"].astype(I32)
  upcard = info["upcard"].astype(I32)
  num_passes = info["num_passes"].astype(I32)
  trump = info["trump"].astype(I32)
  declarer = info["declarer"].astype(I32)
  go_alone = info["go_alone"].astype(I32)
  discard = info["discard"].astype(I32)
  n_played = info["n_played"].astype(I32)
  play = info["play"].astype(I32)
  play_seat = info["play_seat"].astype(I32)
  trump_set = trump >= 0
  alone = go_alone == 1
  sitter = jnp.where(alone, (declarer + 2) % N_PLAYER, -1)
  n_active = jnp.where(alone, N_PLAYER - 1, N_PLAYER)
  ordered_up = trump_set & (num_passes < N_PLAYER)
  dealer_out = alone & (dealer == sitter)
  takes = (
      ordered_up & ~dealer_out
  )  # the dealer takes the up-card (after go-alone)
  over = trump_set & (go_alone >= 0) & (n_played == N_TRICK * n_active)
  # per play position: trick, place in trick, led card, effective suit / rank
  pos = jnp.arange(N_PLAYS, dtype=I32)
  live = pos < n_played
  card_at = jnp.where(live, play, 0)
  trick_at = pos // n_active
  place_at = pos % n_active
  es_at = eff_suit(card_at, trump)
  er_at = eff_rank(card_at, trump)
  first_at = jnp.clip(trick_at * n_active, 0, N_PLAYS - 1)
  led_es_at = es_at[first_at]
  # the winner so far of each trick (its live cards): trump first, then the
  # led suit
  key = jnp.where(
      es_at == trump, 200 + er_at, jnp.where(es_at == led_es_at, er_at, -1)
  )
  key = jnp.where(live, key, -2)
  t_idx = jnp.arange(N_TRICK, dtype=I32)
  slot = jnp.arange(N_PLAYER, dtype=I32)
  idx = t_idx[:, None] * n_active + slot[None, :]  # [5, 4]
  in_trick = (slot[None, :] < n_active) & (idx < N_PLAYS)
  idx = jnp.clip(idx, 0, N_PLAYS - 1)
  tkey = jnp.where(in_trick, key[idx], -3)
  widx = idx[t_idx, jnp.argmax(tkey, axis=1)]
  winner = play_seat[widx]
  win_card = card_at[widx]
  complete = (t_idx + 1) * n_active <= n_played
  tricks_won = (
      jnp.zeros((N_PLAYER,), I32)
      .at[jnp.where(complete, winner, N_PLAYER)]
      .add(1, mode="drop")
  )
  # per card
  target = jnp.where(live, play, N_CARD)
  seat_of = jnp.full((N_CARD,), -1, I32).at[target].set(play_seat, mode="drop")
  trick_of = jnp.full((N_CARD,), -1, I32).at[target].set(trick_at, mode="drop")
  led_of = (
      jnp.zeros((N_CARD,), jnp.bool_).at[target].set(place_at == 0, mode="drop")
  )
  played = seat_of >= 0
  # voids: a follower that did not follow the led effective suit
  shows_void = live & (place_at != 0) & (es_at != led_es_at)
  vidx = jnp.where(
      shows_void, play_seat * N_SUIT + led_es_at, N_PLAYER * N_SUIT
  )
  void = (
      jnp.zeros((N_PLAYER * N_SUIT,), jnp.bool_)
      .at[vidx]
      .set(True, mode="drop")
      .reshape(N_PLAYER, N_SUIT)
  )
  trick_num = n_played // n_active
  n_in_trick = n_played % n_active
  cur = jnp.minimum(trick_num, N_TRICK - 1)
  last_seat = play_seat[jnp.maximum(n_played - 1, 0)]
  # the phase and the seat on move (module doc: the hidden discard)
  bidding = ~trump_set
  deciding_alone = trump_set & (go_alone < 0)
  discard_pending = (
      trump_set
      & (go_alone >= 0)
      & takes
      & (n_played == 0)
      & (viewer == dealer)
      & (discard < 0)
  )
  phase = jnp.where(
      bidding,
      jnp.where(num_passes < N_PLAYER, PHASE_BID1, PHASE_BID2),
      jnp.where(
          deciding_alone,
          PHASE_ALONE,
          jnp.where(
              over,
              PHASE_OVER,
              jnp.where(discard_pending, PHASE_DISCARD, PHASE_PLAY),
          ),
      ),
  )
  first = (dealer + 1) % N_PLAYER
  first_leader = jnp.where(first == sitter, (first + 1) % N_PLAYER, first)
  nxt = (last_seat + 1) % N_PLAYER
  nxt = jnp.where(nxt == sitter, (nxt + 1) % N_PLAYER, nxt)
  play_actor = jnp.where(
      n_played == 0,
      first_leader,
      jnp.where(
          n_in_trick > 0, nxt, winner[jnp.clip(trick_num - 1, 0, N_TRICK - 1)]
      ),
  )
  actor = jnp.select(
      [
          phase <= PHASE_BID2,
          phase == PHASE_ALONE,
          phase == PHASE_DISCARD,
          phase == PHASE_PLAY,
      ],
      [(dealer + 1 + num_passes) % N_PLAYER, declarer, dealer, play_actor],
      -1,
  )
  # the seat that moved last (-1: unknown from this InfoState; module doc)
  hidden_window = takes & (go_alone >= 0) & (n_played == 0) & (viewer != dealer)
  prev = jnp.select(
      [
          phase <= PHASE_BID2,
          phase == PHASE_ALONE,
          phase == PHASE_DISCARD,
          (phase == PHASE_PLAY) & (n_played > 0),
          (phase == PHASE_PLAY) & ~hidden_window,
      ],
      [
          jnp.where(num_passes > 0, (dealer + num_passes) % N_PLAYER, -1),
          declarer,
          declarer,
          last_seat,
          jnp.where(takes, dealer, declarer),
      ],
      -1,
  )
  leader = jnp.where(
      (phase == PHASE_PLAY) & (n_in_trick > 0),
      play_seat[jnp.minimum(trick_num * n_active, N_PLAYS - 1)],
      -1,
  )
  return dict(
      viewer=viewer,
      dealer=dealer,
      upcard=upcard,
      num_passes=num_passes,
      trump=trump,
      declarer=declarer,
      discard=discard,
      trump_set=trump_set,
      alone=alone,
      ordered_up=ordered_up,
      takes=takes,
      go_alone=go_alone,
      n_played=n_played,
      seat_of=seat_of,
      trick_of=trick_of,
      led_of=led_of,
      played=played,
      void=void,
      trick_num=trick_num,
      n_in_trick=n_in_trick,
      led_suit=led_es_at[jnp.clip(trick_num * n_active, 0, N_PLAYS - 1)],
      win_card=win_card[cur],
      tricks_won=tricks_won,
      phase=phase,
      actor=actor,
      prev=prev,
      leader=leader,
  )


def card_planes(info) -> jax.Array:
  """The card planes of one seat's InfoState (module doc).

  Args:
    info: one seat's InfoState (envs.euchre).

  Returns:
    f32[64, 24].
  """
  f = _facts(info)
  viewer, trump, phase = f["viewer"], f["trump"], f["phase"]
  trump_set = f["trump_set"]
  cards = jnp.arange(N_CARD, dtype=I32)
  is_up = cards == f["upcard"]
  hand = info["hand"]
  played = f["played"]
  in_play = phase == PHASE_PLAY
  upcard_with_dealer = (
      is_up
      & f["takes"]
      & (f["go_alone"] >= 0)
      & ~played[jnp.clip(f["upcard"], 0, N_CARD - 1)]
  )
  turned_down = is_up & (f["num_passes"] >= N_PLAYER)
  my_discard = (
      (cards == f["discard"]) & (viewer == f["dealer"]) & (f["discard"] >= 0)
  )
  played_rel = (f["seat_of"] - viewer) % N_PLAYER
  played_by_rel = [played & (played_rel == k) for k in range(N_PLAYER)]
  played_in = [played & (f["trick_of"] == t) for t in range(N_TRICK)]
  in_trick = (
      played
      & (f["trick_of"] == f["trick_num"])
      & in_play
      & (f["n_in_trick"] > 0)
  )
  winning = (cards == f["win_card"]) & in_trick
  es = eff_suit(cards, trump)
  is_trump = trump_set & (es == trump)
  er = jnp.where(trump_set, eff_rank(cards, trump), cards // N_SUIT)
  same_up = (cards % N_SUIT) == (f["upcard"] % N_SUIT)
  follows = in_trick.any() & (es == f["led_suit"]) & trump_set
  void = f["void"]
  voids = [
      void[(viewer + k) % N_PLAYER][jnp.clip(es, 0, N_SUIT - 1)] & trump_set
      for k in (1, 2, 3)
  ]
  unseen = ~hand & ~played & ~is_up & ~my_discard
  card = [
      hand,
      is_up,
      upcard_with_dealer,
      turned_down,
      my_discard,
      *played_by_rel,
      *played_in,
      f["led_of"] & played,
      in_trick,
      winning,
      is_trump,
      er.astype(F32) / 7.0,
      same_up,
      follows,
      *voids,
      unseen,
  ]
  rel = jnp.arange(N_PLAYER, dtype=I32)
  pass_order = (
      rel - f["dealer"] - 1
  ) % N_PLAYER  # each seat's place in a bidding round
  seats = (viewer + rel) % N_PLAYER
  passed1 = pass_order[seats] < jnp.minimum(f["num_passes"], N_PLAYER)
  passed2 = pass_order[seats] < jnp.maximum(f["num_passes"] - N_PLAYER, 0)
  tw = f["tricks_won"]
  us = (tw[viewer] + tw[(viewer + 2) % N_PLAYER]).astype(F32)
  them = (tw[(viewer + 1) % N_PLAYER] + tw[(viewer + 3) % N_PLAYER]).astype(F32)
  broadcast = jnp.concatenate([
      _rel_onehot(f["dealer"], viewer).astype(F32),
      (jnp.arange(5) == phase).astype(F32),
      (phase == PHASE_OVER).astype(F32)[None],
      _rel_onehot(f["actor"], viewer).astype(F32),
      _rel_onehot(f["declarer"], viewer).astype(F32),
      ((jnp.arange(N_SUIT) == trump) & trump_set).astype(F32),
      f["alone"].astype(F32)[None],
      passed1.astype(F32),
      passed2.astype(F32),
      (us / 5.0)[None],
      (them / 5.0)[None],
      _rel_onehot(f["leader"], viewer).astype(F32),
      (jnp.where(in_play, f["n_in_trick"], 0).astype(F32) / 4.0)[None],
      (jnp.minimum(f["trick_num"], N_TRICK).astype(F32) / N_TRICK)[None],
  ])
  planes = jnp.stack([jnp.asarray(p).astype(F32) for p in card])
  out = jnp.concatenate([
      planes,
      jnp.broadcast_to(broadcast[:, None], (broadcast.shape[0], N_CARD)),
  ])
  assert out.shape == (N_CHANNEL, N_CARD), out.shape
  return out


def waiting_row(info) -> jax.Array:
  """bool: a value row of `info`'s seat (module doc, round-robin clock).

  Args:
    info: one seat's InfoState (envs.euchre).

  Returns:
    bool[].
  """
  f = _facts(info)
  x, y, seat = f["prev"], f["actor"], f["viewer"]
  gap = (y - x - 1) % N_PLAYER
  k = (seat - x - 1) % N_PLAYER
  return (x >= 0) & (y >= 0) & (seat != y) & (k < gap)


CARD_PLANES_ENCODER = registry.register(
    registry.Encoder(
        name=NAME,
        game=euchre.EuchreEnv.name,
        shape=(N_CHANNEL, N_CARD),
        dtype=F32,
        fn=card_planes,
        reference="per-card planes of one seat's view (module doc)",
    )
)
