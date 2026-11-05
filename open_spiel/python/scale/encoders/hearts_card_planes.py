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

"""Hearts card planes over the envs.hearts InfoState: f32[56, 52].

Importing this module registers the encoder; encoders.registry.LAZY names
this module, so a lookup of the name imports it (no import by the caller).

One column per card (card = rank * 4 + suit, suits C D H S), one row per
channel (CHANNELS), from one seat's view; seats are relative to the viewer
(rel 0 the viewer, 1 its left, 2 across, 3 its right). Only what the seat
may know is encoded: its hand and deal, the pass direction, the cards it
picked to pass (and, after the exchange, that they sit with the receiver
until played), the cards it received (after the exchange only), every
played card (who played it, in which trick, whether it was led, who won its
trick), the voids those plays reveal, hearts broken and each seat's points.
Another seat's picks and pick count are never encoded.

  28 card planes: my_hand, my_dealt, my_pass_pick, my_pass_with_receiver,
    received, played_by_rel0..3, played_trick_over13, led_card,
    in_current_trick, winning_current_trick, won_by_rel0..3, suit0..3,
    rank_over12, points_over13, follows_led_suit, void_rel1..3, unseen.
  28 broadcast planes (one value over every card): pass_dir0..3,
    phase_pass, phase_play, terminal, actor_rel0..3, my_picks_over3,
    leader_rel0..3, n_in_trick_over4, trick_num_over13, hearts_broken,
    qs_played, points_rel0..3_over26, moon_live_rel0..3.

The actor before the first lead is the viewer when it holds the two of
clubs, else no seat (nobody may know who holds it). Points are the
InfoState's: trick points during play, the scored points (a moon applied)
at the end. Hearts are broken by a heart or the queen of spades, as at the
default rules.

``waiting_row`` is a value-row predicate (learners.pg PGConfig.value_rows)
for self-play over all four seats with a round-robin row clock: right after
a trick, the seats strictly between the trick's last player and its winner
(in seat order) are the seats such a clock reaches while the winner has yet
to lead. The first lead of a hand is left out (where the clock stands then
depends on the previous hand, not on this InfoState).
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import hearts as game
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import hearts

F32 = jnp.float32
I32 = jnp.int32
NAME = "hearts_card_planes"
N_CARD = game.NUM_CARDS
N_PLAYER = game.NUM_PLAYERS
N_SUIT = game.NUM_SUITS
N_TRICK = game.NUM_TRICKS
N_PASS = game.NUM_CARDS_IN_PASS
TOTAL_POINTS = game.TOTAL_POSITIVE_POINTS

CARD_PLANES = (
    ("my_hand", "my_dealt", "my_pass_pick", "my_pass_with_receiver", "received")
    + tuple(f"played_by_rel{k}" for k in range(N_PLAYER))
    + (
        "played_trick_over13",
        "led_card",
        "in_current_trick",
        "winning_current_trick",
    )
    + tuple(f"won_by_rel{k}" for k in range(N_PLAYER))
    + tuple(f"suit{s}" for s in range(N_SUIT))
    + ("rank_over12", "points_over13", "follows_led_suit")
    + tuple(f"void_rel{k}" for k in (1, 2, 3))
    + ("unseen",)
)
BROADCAST_PLANES = (
    tuple(f"b_pass_dir{d}" for d in range(4))
    + ("b_phase_pass", "b_phase_play", "b_terminal")
    + tuple(f"b_actor_rel{k}" for k in range(N_PLAYER))
    + ("b_my_picks_over3",)
    + tuple(f"b_leader_rel{k}" for k in range(N_PLAYER))
    + (
        "b_n_in_trick_over4",
        "b_trick_num_over13",
        "b_hearts_broken",
        "b_qs_played",
    )
    + tuple(f"b_points_rel{k}_over26" for k in range(N_PLAYER))
    + tuple(f"b_moon_live_rel{k}" for k in range(N_PLAYER))
)
CHANNELS = CARD_PLANES + BROADCAST_PLANES
N_CHANNEL = len(CHANNELS)
assert N_CHANNEL == 56 and len(set(CHANNELS)) == N_CHANNEL


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


def _history(info):
  """The play history's per-card and per-trick facts.

  Args:
    info: one seat's InfoState (envs.hearts).

  Returns:
    A dict of arrays (see the keys).
  """
  n_played = info["n_played"].astype(I32)
  play = info["play"].astype(I32)
  play_seat = info["play_seat"].astype(I32)
  pos = jnp.arange(N_CARD, dtype=I32)
  live = pos < n_played
  safe = jnp.where(live, play, 0)
  suit_at = safe % N_SUIT
  rank_at = safe // N_SUIT
  target = jnp.where(live, play, N_CARD)
  card_pos = jnp.full((N_CARD,), N_CARD, I32).at[target].set(pos, mode="drop")
  seat_of = jnp.full((N_CARD,), -1, I32).at[target].set(play_seat, mode="drop")
  played = card_pos < N_CARD
  card_trick = jnp.where(played, card_pos // N_PLAYER, -1)
  # per trick: led suit, winner, winning card (over the cards played so far)
  suit4 = suit_at.reshape(N_TRICK, N_PLAYER)
  rank4 = rank_at.reshape(N_TRICK, N_PLAYER)
  live4 = live.reshape(N_TRICK, N_PLAYER)
  led4 = suit4[:, 0]
  key = jnp.where(live4 & (suit4 == led4[:, None]), rank4, -1)
  widx = jnp.argmax(key, axis=1)
  t_idx = jnp.arange(N_TRICK, dtype=I32)
  winner = play_seat.reshape(N_TRICK, N_PLAYER)[t_idx, widx]
  win_card = safe.reshape(N_TRICK, N_PLAYER)[t_idx, widx]
  complete = (t_idx + 1) * N_PLAYER <= n_played
  # voids: a seat that did not follow the led suit
  follower = live & (pos % N_PLAYER != 0)
  led_at = led4[pos // N_PLAYER]
  shows_void = follower & (suit_at != led_at)
  vidx = jnp.where(shows_void, play_seat * N_SUIT + led_at, N_PLAYER * N_SUIT)
  void = (
      jnp.zeros((N_PLAYER * N_SUIT,), jnp.bool_)
      .at[vidx]
      .set(True, mode="drop")
      .reshape(N_PLAYER, N_SUIT)
  )
  return dict(
      n_played=n_played,
      play_seat=play_seat,
      card_pos=card_pos,
      seat_of=seat_of,
      played=played,
      card_trick=card_trick,
      led4=led4,
      winner=winner,
      win_card=win_card,
      complete=complete,
      void=void,
  )


def card_planes(info) -> jax.Array:
  """The card planes of one seat's InfoState (module doc).

  Args:
    info: one seat's InfoState (envs.hearts).

  Returns:
    f32[56, 52].
  """
  h = _history(info)
  viewer = info["seat"].astype(I32)
  cards = jnp.arange(N_CARD, dtype=I32)
  suit = cards % N_SUIT
  rank = cards // N_SUIT
  pass_dir = info["pass_dir"].astype(I32)
  received_raw = info["received"]
  n_played = h["n_played"]
  over = n_played == N_CARD
  passing = (pass_dir != 0) & ~received_raw.any() & (n_played == 0)
  in_play = ~passing & ~over
  exchanged = ~passing
  hand = info["hand"]
  my_pick = info["passed"]
  played = h["played"]
  my_pass_with_receiver = my_pick & ~played & exchanged
  received = received_raw & exchanged
  played_rel = (h["seat_of"] - viewer) % N_PLAYER
  played_by_rel = [played & (played_rel == k) for k in range(N_PLAYER)]
  card_trick = h["card_trick"]
  safe_trick = jnp.clip(card_trick, 0, N_TRICK - 1)
  won = played & h["complete"][safe_trick]
  won_seat = h["winner"][safe_trick]
  won_rel = (won_seat - viewer) % N_PLAYER
  won_by_rel = [won & (won_rel == k) for k in range(N_PLAYER)]
  trick_num = n_played // N_PLAYER
  n_in_trick = n_played % N_PLAYER
  cur = jnp.minimum(trick_num, N_TRICK - 1)
  in_trick = played & (card_trick == trick_num) & in_play & (n_in_trick > 0)
  winning = (cards == h["win_card"][cur]) & in_trick
  led_suit = h["led4"][cur]
  follows = in_trick.any() & (suit == led_suit)
  void = h["void"]
  voids = [void[(viewer + k) % N_PLAYER][suit] for k in (1, 2, 3)]
  unseen = ~hand & ~played & ~my_pick
  points_of = jnp.where(suit == game.HEARTS, 1, 0) + jnp.where(
      cards == game.QUEEN_OF_SPADES, 13, 0
  )
  card = [
      hand,
      info["dealt"],
      my_pick,
      my_pass_with_receiver,
      received,
      *played_by_rel,
      jnp.where(played, (card_trick + 1).astype(F32) / N_TRICK, 0.0),
      played & (h["card_pos"] % N_PLAYER == 0),
      in_trick,
      winning,
      *won_by_rel,
      *[suit == k for k in range(N_SUIT)],
      rank.astype(F32) / 12.0,
      points_of.astype(F32) / 13.0,
      follows,
      *voids,
      unseen,
  ]
  n_picked = my_pick.sum(dtype=I32)
  last_seat = h["play_seat"][jnp.maximum(n_played - 1, 0)]
  prev_winner = h["winner"][jnp.clip(trick_num - 1, 0, N_TRICK - 1)]
  play_actor = jnp.where(
      n_in_trick > 0,
      (last_seat + 1) % N_PLAYER,
      jnp.where(
          n_played > 0,
          prev_winner,
          jnp.where(hand[game.TWO_OF_CLUBS], viewer, -1),
      ),
  )
  actor = jnp.where(
      over,
      -1,
      jnp.where(passing, jnp.where(n_picked < N_PASS, viewer, -1), play_actor),
  )
  leader = jnp.where(
      in_play & (n_in_trick > 0),
      h["play_seat"][jnp.minimum(trick_num * N_PLAYER, N_CARD - 1)],
      -1,
  )
  points = info["points"].astype(I32)
  points_rel = jnp.roll(points, -viewer)
  taken = points.sum()
  moon_live = (points_rel > 0) & (points_rel == taken)
  heart_played = (played & (suit == game.HEARTS)).any()
  qs_played = played[game.QUEEN_OF_SPADES]
  broadcast = jnp.concatenate([
      (jnp.arange(4) == pass_dir).astype(F32),
      passing.astype(F32)[None],
      in_play.astype(F32)[None],
      over.astype(F32)[None],
      _rel_onehot(actor, viewer).astype(F32),
      (jnp.where(passing, n_picked, 0).astype(F32) / N_PASS)[None],
      _rel_onehot(leader, viewer).astype(F32),
      (jnp.where(in_play, n_in_trick, 0).astype(F32) / 4.0)[None],
      (jnp.minimum(trick_num, N_TRICK).astype(F32) / N_TRICK)[None],
      (heart_played | qs_played).astype(F32)[None],
      qs_played.astype(F32)[None],
      points_rel.astype(F32) / TOTAL_POINTS,
      moon_live.astype(F32),
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
    info: one seat's InfoState (envs.hearts).

  Returns:
    bool[].
  """
  h = _history(info)
  seat = info["seat"].astype(I32)
  n_played = h["n_played"]
  between = (
      (n_played % N_PLAYER == 0) & (n_played >= N_PLAYER) & (n_played < N_CARD)
  )
  last = h["play_seat"][jnp.maximum(n_played - 1, 0)]
  winner = h["winner"][jnp.clip(n_played // N_PLAYER - 1, 0, N_TRICK - 1)]
  gap = (winner - last - 1) % N_PLAYER
  k = (seat - last - 1) % N_PLAYER
  return between & (k < gap)


CARD_PLANES_ENCODER = registry.register(
    registry.Encoder(
        name=NAME,
        game=hearts.HeartsEnv.name,
        shape=(N_CHANNEL, N_CARD),
        dtype=F32,
        fn=card_planes,
        reference="per-card planes of one seat's view (module doc)",
    )
)
