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

"""Euchre in pure JAX: OpenSpiel's ``euchre``, both game parameters.

Four players in two partnerships (North-South, seats 0 and 2, against
East-West, seats 1 and 3), a 24-card deck (9 to ace), five tricks. Encodings
are OpenSpiel's, so actions, chance outcomes, players, returns and the
information-state tensor compare with OpenSpiel's euchre without translation:

- seats 0..3 are North, East, South, West;
- card = rank * 4 + suit, suit 0..3 = clubs, diamonds, hearts, spades, rank
  0..5 = 9, 10, J, Q, K, A (the jack of a suit is 8 + suit);
- actions 0..23 are cards (dealt by chance, discarded and played by
  players), 24 pass, 25..28 name clubs, diamonds, hearts or spades as trump,
  29 go alone, 30 play with partner; the first chance node's outcomes 0..3
  are the dealer.

Rules version. This is the euchre of OpenSpiel's corrected engine, whose
information-state tensor has 959 entries (released OpenSpiel 1.6 and 2.0
have 935 and take up the up-card before the go-alone decisions):

- the declarer (the player who names trump) says whether they play alone
  right after naming trump, before the dealer takes up an accepted up-card;
  with ``allow_lone_defender`` the defenders then decide in turn (the one
  nearer the dealer's left first; the other is asked only if the first
  declines);
- if trump was named in the first round (the up-card accepted) and the dealer
  plays, the dealer takes the up-card and discards one of the other five
  cards; a dealer who sits out (their partner plays alone) neither takes it
  nor discards;
- the player left of the dealer leads (the next active seat if that one sits
  out); players follow the led suit (the left bower belongs to trump) when
  they can; trump beats the led suit, right bower highest, then left bower;
- returns are zero-sum, the makers' score for both makers and its negative
  for the defenders: 1 for three or four tricks, 2 for all five (4 if the
  declarer played alone), -2 for fewer than three (-4 against a lone
  defender). Eight passes (only without ``stick_the_dealer``) end the hand
  with all returns zero; with ``stick_the_dealer`` the dealer cannot pass the
  eighth bid.

Phases (OpenSpiel's euchre.cc). Chance draws the dealer, uniformly over the
four, then deals 20 cards one at a time (the i-th card dealt goes to seat
(dealer + i) % 4), then turns one more card, the up-card. Bidding starts left
of the dealer: in round one a player may pass or name the up-card's suit, in
round two pass or name another suit.

Information state. ``information_state_tensor`` is OpenSpiel's f32[959]: the
dealer, the up-card, eight bid slots of five (clubs, diamonds, hearts,
spades, pass: each pass sets its own slot's pass entry, a call its slot's
suit), three go-alone bits (the declarer went alone, the first defender went
alone, the second defender went alone), the dealer's discard (the dealer's
view only), the current hand, then the five tricks, each as 7 slots of 24
(N E S W N E S) with the card a seat played in a trick led by seat L in slot
L + (seat - L) % 4. It is all zeros before the up-card is turned; after
eight passes only the dealer, the up-card and the bids are set. OpenSpiel's
euchre has no observation tensor; ``observation_tensor`` is the
information-state tensor (a seat's perfect-recall view).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_CARDS_PER_SUIT = 6
NUM_CARDS = 24
NUM_TRICKS = 5
JACK_RANK = 2
CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3

PASS_ACTION = 24
CLUBS_TRUMP_ACTION = 25
DIAMONDS_TRUMP_ACTION = 26
HEARTS_TRUMP_ACTION = 27
SPADES_TRUMP_ACTION = 28
GO_ALONE_ACTION = 29
PLAY_WITH_PARTNER_ACTION = 30
NUM_DISTINCT_ACTIONS = 31

MAX_BIDS = 8
#: Cards dealt to the hands; the next chance outcome is the up-card.
NUM_HAND_CARDS = NUM_PLAYERS * NUM_TRICKS
TRICK_TENSOR_SIZE = NUM_CARDS * 7  # N E S W N E S
BIDDING_OFFSET = NUM_PLAYERS + NUM_CARDS  # 28
GO_ALONE_OFFSET = BIDDING_OFFSET + (NUM_SUITS + 1) * MAX_BIDS  # 68
DISCARD_OFFSET = GO_ALONE_OFFSET + 3  # 71
HAND_OFFSET = DISCARD_OFFSET + NUM_CARDS  # 95
TRICKS_OFFSET = HAND_OFFSET + NUM_CARDS  # 119
INFO_STATE_SIZE = TRICKS_OFFSET + NUM_TRICKS * TRICK_TENSOR_SIZE  # 959

(
    PHASE_DEALER_SELECTION,
    PHASE_DEAL,
    PHASE_BIDDING,
    PHASE_DISCARD,
    PHASE_GO_ALONE,
    PHASE_PLAY,
    PHASE_OVER,
) = range(7)

PARAMS = ("allow_lone_defender", "stick_the_dealer")


def max_game_length(allow_lone_defender: bool) -> int:
  """OpenSpiel's MaxGameLength: the most player decisions in one hand.

  Eight bids, the declarer's go-alone decision, the two defenders' (with
  `allow_lone_defender`) and 20 plays. A hand with a discard was decided in
  the first round of bidding, so it is never longer.

  Args:
    allow_lone_defender: The game parameter.

  Returns:
    The length.
  """
  return (
      2 * NUM_PLAYERS
      + 1
      + 2 * int(allow_lone_defender)
      + NUM_PLAYERS * NUM_TRICKS
  )


class GameState(NamedTuple):
  """A hand of euchre. -1 marks an unset card, seat, suit or slot."""

  phase: jax.Array  # int8
  # int32: seat, CHANCE_PLAYER_ID, or TERMINAL_PLAYER_ID
  current_player: jax.Array
  is_terminal: jax.Array  # bool
  dealer: jax.Array  # int8
  n_dealt: jax.Array  # int8, the up-card included
  holder: jax.Array  # int8[24]: the seat holding each card now
  initial_deal: jax.Array  # int8[24]: the seat dealt each card (once dealt)
  upcard: jax.Array  # int8
  num_passes: jax.Array  # int8
  trump: jax.Array  # int8, the trump suit
  declarer: jax.Array  # int8
  go_alone: jax.Array  # int8: the declarer's decision, 1 alone, 0 not, -1
  lone_defender: jax.Array  # int8
  active: jax.Array  # bool[4]: the seats that play (not sitting out)
  discard: jax.Array  # int8: the dealer's discard
  play: jax.Array  # int8[20]: the cards in order of play
  play_seat: jax.Array  # int8[20]: who played each of them
  n_played: jax.Array  # int8
  trick_winner: jax.Array  # int8[5]
  points: jax.Array  # int32[4]


def _i8(v) -> jax.Array:
  """`v` as int8.

  Args:
    v: A value or array.

  Returns:
    The int8 array.
  """
  return jnp.asarray(v, jnp.int8)


def _i32(v) -> jax.Array:
  """`v` as int32.

  Args:
    v: A value or array.

  Returns:
    The int32 array.
  """
  return jnp.asarray(v, jnp.int32)


def _cards() -> jax.Array:
  """The card ids, built at use (no array is made at import).

  Returns:
    int32[24], 0..23.
  """
  return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def _where_tree(cond, x, y):
  """`x` where `cond`, else `y`, leaf by leaf.

  Args:
    cond: A bool scalar.
    x: A pytree.
    y: A pytree of the same structure.

  Returns:
    The selected pytree.
  """
  return jax.tree.map(lambda a, b: jnp.where(cond, a, b), x, y)


def card_suit(card):
  """The printed suit of a card (card % 4).

  Args:
    card: The card id(s).

  Returns:
    The suit(s).
  """
  return card % NUM_SUITS


def card_rank(card):
  """The printed rank of a card (card // 4), 0..5 = 9..A.

  Args:
    card: The card id(s).

  Returns:
    The rank(s).
  """
  return card // NUM_SUITS


def same_color_suit(suit):
  """The other suit of the same colour (clubs-spades, diamonds-hearts).

  Args:
    suit: The suit(s).

  Returns:
    The suit(s).
  """
  return 3 - suit


def effective_suit(card, trump):
  """OpenSpiel's CardSuit(card, trump): the left bower counts as trump.

  Args:
    card: The card id(s).
    trump: The trump suit (-1 before trump is named).

  Returns:
    The suit(s) the card belongs to under `trump`.
  """
  suit = card_suit(card)
  left = (card_rank(card) == JACK_RANK) & (same_color_suit(suit) == trump)
  return jnp.where(left & (trump >= 0), trump, suit)


def effective_rank(card, trump):
  """OpenSpiel's CardRank(card, trump): right bower 100, left bower 99.

  Args:
    card: The card id(s).
    trump: The trump suit.

  Returns:
    The rank(s), comparable within one effective suit.
  """
  rank = card_rank(card)
  jack = rank == JACK_RANK
  right = jack & (card_suit(card) == trump)
  left = jack & (same_color_suit(card_suit(card)) == trump) & (trump >= 0)
  return jnp.where(right, 100, jnp.where(left, 99, rank))


def info_state_tensor(
    *,
    seat,
    dealer,
    upcard,
    num_passes,
    trump,
    declarer,
    go_alone,
    lone_defender,
    discard,
    hand,
    play,
    play_seat,
    n_played,
) -> jax.Array:
  """OpenSpiel's euchre information-state tensor from one seat's view.

  Args:
    seat: The observing seat.
    dealer: The dealer.
    upcard: The up-card, -1 before it is turned.
    num_passes: The passes made.
    trump: The trump suit, -1 before it is named.
    declarer: The declarer, -1 before trump is named.
    go_alone: The declarer's decision: 1 alone, 0 with partner, -1 undecided.
    lone_defender: The defender playing alone, else -1.
    discard: The dealer's discard; -1 when there is none. Set only in the
      dealer's view (another seat's tensor ignores it).
    hand: bool[24], the seat's hand now.
    play: int[20], the cards in order of play (-1 pad).
    play_seat: int[20], who played them (-1 pad).
    n_played: The cards played.

  Returns:
    f32[959] (module doc).
  """
  f32 = jnp.float32
  seat, dealer, upcard = _i32(seat), _i32(dealer), _i32(upcard)
  num_passes, trump, declarer = _i32(num_passes), _i32(trump), _i32(declarer)
  go_alone, lone_defender = _i32(go_alone), _i32(lone_defender)
  discard = _i32(discard)
  live = upcard >= 0
  # After eight passes OpenSpiel's tensor stops after the bids.
  more = live & (num_passes < 2 * NUM_PLAYERS)
  slot = jnp.arange(MAX_BIDS)
  bids = jnp.zeros((MAX_BIDS, NUM_SUITS + 1), f32)
  bids = bids.at[:, NUM_SUITS].set((slot < num_passes).astype(f32))
  bids = bids.at[
      jnp.minimum(num_passes, MAX_BIDS - 1), jnp.maximum(trump, 0)
  ].max(((trump >= 0) & more).astype(f32))
  lone = lone_defender >= 0
  go = jnp.stack([
      go_alone == 1,
      lone & (lone_defender == (declarer + 1) % NUM_PLAYERS),
      lone & (lone_defender == (declarer + 3) % NUM_PLAYERS),
  ]).astype(f32)
  disc = jax.nn.one_hot(
      jnp.where((discard >= 0) & (seat == dealer), discard, NUM_CARDS),
      NUM_CARDS,
      dtype=f32,
  )
  t = jnp.arange(NUM_HAND_CARDS)
  play = _i32(play)
  seats = _i32(play_seat)
  n_active = (
      NUM_PLAYERS - (go_alone == 1).astype(jnp.int32) - lone.astype(jnp.int32)
  )
  k = t // n_active
  leader = seats[jnp.minimum(k * n_active, NUM_HAND_CARDS - 1)]
  pos = (
      TRICKS_OFFSET
      + k * TRICK_TENSOR_SIZE
      + (leader + (seats - leader) % NUM_PLAYERS) * NUM_CARDS
      + jnp.maximum(play, 0)
  )
  ok = (t < _i32(n_played)) & more
  out = (
      jnp.zeros((INFO_STATE_SIZE,), f32)
      .at[jnp.where(ok, pos, INFO_STATE_SIZE)]
      .set(1.0, mode="drop")
  )
  more_f = more.astype(f32)
  head = jnp.concatenate([
      jax.nn.one_hot(dealer, NUM_PLAYERS, dtype=f32),
      jax.nn.one_hot(upcard, NUM_CARDS, dtype=f32),
      bids.reshape(-1),
      go * more_f,
      disc * more_f,
      jnp.asarray(hand, f32) * more_f,
  ])
  out = out.at[:TRICKS_OFFSET].set(head)
  return jnp.where(live, out, jnp.zeros_like(out))


class Euchre(game_lib.Game, interop.PyspielInterop):
  """OpenSpiel euchre (module doc); the parameters are OpenSpiel's."""

  def __init__(
      self, allow_lone_defender: bool = False, stick_the_dealer: bool = True
  ):
    values = (allow_lone_defender, stick_the_dealer)
    for name, v in zip(PARAMS, values):
      if not isinstance(v, bool):
        raise TypeError(f"euchre parameter {name} must be a bool, got {v!r}")
    self.params = dict(zip(PARAMS, values))
    self.allow_lone_defender, self.stick_the_dealer = values

  @property
  def num_actions(self) -> int:
    """The 24 cards and the seven calls."""
    return NUM_DISTINCT_ACTIONS

  @property
  def num_players(self) -> int:
    """Four seats."""
    return NUM_PLAYERS

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The information-state tensor shape."""
    return (INFO_STATE_SIZE,)

  @property
  def max_chance_outcomes(self) -> int:
    """The 24 cards (the four dealers use 0..3)."""
    return NUM_CARDS

  @property
  def max_game_length(self) -> int:
    """OpenSpiel's MaxGameLength (player decisions; `max_game_length`)."""
    return max_game_length(self.allow_lone_defender)

  def new_initial_state(self) -> GameState:
    """The state before the dealer is drawn.

    Returns:
      The initial GameState (a chance node).
    """
    return GameState(
        phase=_i8(PHASE_DEALER_SELECTION),
        current_player=_i32(game_lib.CHANCE_PLAYER_ID),
        is_terminal=jnp.asarray(False),
        dealer=_i8(-1),
        n_dealt=_i8(0),
        holder=jnp.full((NUM_CARDS,), -1, jnp.int8),
        initial_deal=jnp.full((NUM_CARDS,), -1, jnp.int8),
        upcard=_i8(-1),
        num_passes=_i8(0),
        trump=_i8(-1),
        declarer=_i8(-1),
        go_alone=_i8(-1),
        lone_defender=_i8(-1),
        active=jnp.ones((NUM_PLAYERS,), jnp.bool_),
        discard=_i8(-1),
        play=jnp.full((NUM_HAND_CARDS,), -1, jnp.int8),
        play_seat=jnp.full((NUM_HAND_CARDS,), -1, jnp.int8),
        n_played=_i8(0),
        trick_winner=jnp.full((NUM_TRICKS,), -1, jnp.int8),
        points=jnp.zeros((NUM_PLAYERS,), jnp.int32),
    )

  # -- transitions --------------------------------------------------------

  def _dealer_step(self, s: GameState, a) -> GameState:
    """The dealer-selection chance outcome `a`.

    Args:
      s: The state.
      a: The dealer.

    Returns:
      The state, now dealing.
    """
    return s._replace(dealer=_i8(a), phase=_i8(PHASE_DEAL))

  def _deal_step(self, s: GameState, a) -> GameState:
    """Deals card `a` to the next seat, or turns it as the up-card.

    Args:
      s: The state.
      a: The card.

    Returns:
      The state; after the up-card, bidding.
    """
    n = _i32(s.n_dealt)
    dealer = _i32(s.dealer)
    is_up = n == NUM_HAND_CARDS
    holder = jnp.where(
        is_up, s.holder, s.holder.at[a].set(_i8((dealer + n) % NUM_PLAYERS))
    )
    return s._replace(
        holder=holder,
        n_dealt=_i8(n + 1),
        initial_deal=jnp.where(is_up, s.holder, s.initial_deal),
        upcard=jnp.where(is_up, _i8(a), s.upcard),
        phase=_i8(jnp.where(is_up, PHASE_BIDDING, PHASE_DEAL)),
        current_player=_i32(
            jnp.where(
                is_up, (dealer + 1) % NUM_PLAYERS, game_lib.CHANCE_PLAYER_ID
            )
        ),
    )

  def _bid_step(self, s: GameState, a) -> GameState:
    """The current player passes or names trump.

    Args:
      s: The state.
      a: The bid.

    Returns:
      The state; terminal after eight passes, else bidding or go-alone.
    """
    p = s.current_player
    is_pass = a == PASS_ACTION
    passes = _i32(s.num_passes) + is_pass.astype(jnp.int32)
    over = is_pass & (passes == 2 * NUM_PLAYERS)
    phase = jnp.where(
        is_pass, jnp.where(over, PHASE_OVER, PHASE_BIDDING), PHASE_GO_ALONE
    )
    cp = jnp.where(
        is_pass,
        jnp.where(over, game_lib.TERMINAL_PLAYER_ID, (p + 1) % NUM_PLAYERS),
        p,
    )
    return s._replace(
        num_passes=_i8(passes),
        trump=jnp.where(is_pass, s.trump, _i8(a - CLUBS_TRUMP_ACTION)),
        declarer=jnp.where(is_pass, s.declarer, _i8(p)),
        phase=_i8(phase),
        current_player=_i32(cp),
    )

  def _start_play(self, s: GameState) -> GameState:
    """OpenSpiel StartPlay: the first active seat left of the dealer leads.

    Args:
      s: The state.

    Returns:
      The state, in play.
    """
    order = (_i32(s.dealer) + 1 + jnp.arange(NUM_PLAYERS)) % NUM_PLAYERS
    first = order[jnp.argmax(s.active[order])]
    return s._replace(phase=_i8(PHASE_PLAY), current_player=_i32(first))

  def _end_go_alone(self, s: GameState) -> GameState:
    """OpenSpiel EndGoAlonePhase: the dealer takes an accepted up-card.

    Args:
      s: The state, every go-alone decision made.

    Returns:
      The state, at the dealer's discard or in play.
    """
    dealer = _i32(s.dealer)
    take = (s.num_passes < NUM_PLAYERS) & s.active[dealer]
    taken = s._replace(
        holder=s.holder.at[_i32(s.upcard)].set(_i8(dealer)),
        phase=_i8(PHASE_DISCARD),
        current_player=dealer,
    )
    return _where_tree(take, taken, self._start_play(s))

  def _go_alone_step(self, s: GameState, a) -> GameState:
    """A go-alone decision by the declarer or (optionally) a defender.

    Args:
      s: The state.
      a: GO_ALONE_ACTION or PLAY_WITH_PARTNER_ACTION.

    Returns:
      The state.
    """
    p = s.current_player
    dealer = _i32(s.dealer)
    declarer = _i32(s.declarer)
    partner = (declarer + 2) % NUM_PLAYERS
    alone = a == GO_ALONE_ACTION
    # The declarer's decision.
    d = s._replace(
        go_alone=_i8(alone),
        active=jnp.where(alone, s.active.at[partner].set(False), s.active),
    )
    if not self.allow_lone_defender:
      return self._end_go_alone(d)
    first = (dealer + 1) % NUM_PLAYERS
    first = jnp.where(
        (first == declarer) | (first == partner),
        (first + 1) % NUM_PLAYERS,
        first,
    )
    d = d._replace(current_player=_i32(first))
    # A defender's decision.
    e_alone = self._end_go_alone(
        s._replace(
            lone_defender=_i8(p),
            active=s.active.at[(p + 2) % NUM_PLAYERS].set(False),
        )
    )
    ask_other = (p == (dealer + 1) % NUM_PLAYERS) | (
        p == (dealer + 2) % NUM_PLAYERS
    )
    e_other = s._replace(current_player=_i32((p + 2) % NUM_PLAYERS))
    e_partner = _where_tree(ask_other, e_other, self._end_go_alone(s))
    e = _where_tree(alone, e_alone, e_partner)
    return _where_tree(s.go_alone < 0, d, e)

  def _discard_step(self, s: GameState, a) -> GameState:
    """The dealer discards card `a`; play starts.

    Args:
      s: The state.
      a: The card.

    Returns:
      The state, in play.
    """
    return self._start_play(
        s._replace(discard=_i8(a), holder=s.holder.at[a].set(-1))
    )

  def _play_step(self, s: GameState, a) -> GameState:
    """The current player plays card `a`; a full trick is scored.

    Args:
      s: The state.
      a: The card.

    Returns:
      The state; after the last trick, terminal with the final points.
    """
    p = s.current_player
    t = _i32(s.n_played)
    n_active = jnp.sum(s.active, dtype=jnp.int32)
    play = s.play.at[t].set(_i8(a))
    play_seat = s.play_seat.at[t].set(_i8(p))
    n = t + 1
    trick_done = n % n_active == 0
    k = t // n_active
    first = k * n_active
    cards = jax.lax.dynamic_slice(_i32(play), (first,), (NUM_PLAYERS,))
    seats = jax.lax.dynamic_slice(_i32(play_seat), (first,), (NUM_PLAYERS,))
    trump = _i32(s.trump)
    suits = effective_suit(cards, trump)
    ranks = effective_rank(cards, trump)
    key = jnp.where(
        suits == trump, 200 + ranks, jnp.where(suits == suits[0], ranks, -1)
    )
    key = jnp.where(jnp.arange(NUM_PLAYERS) < n_active, key, -2)
    winner = seats[jnp.argmax(key)]
    trick_winner = jnp.where(
        trick_done, s.trick_winner.at[k].set(_i8(winner)), s.trick_winner
    )
    order = (p + 1 + jnp.arange(NUM_PLAYERS)) % NUM_PLAYERS
    nxt = order[jnp.argmax(s.active[order])]
    over = n == n_active * NUM_TRICKS
    points = jnp.where(over, self._final_points(s, trick_winner), s.points)
    cp = jnp.where(
        over,
        game_lib.TERMINAL_PLAYER_ID,
        jnp.where(trick_done, winner, nxt),
    )
    return s._replace(
        holder=s.holder.at[a].set(-1),
        play=play,
        play_seat=play_seat,
        n_played=_i8(n),
        trick_winner=trick_winner,
        points=points,
        phase=_i8(jnp.where(over, PHASE_OVER, PHASE_PLAY)),
        current_player=_i32(cp),
    )

  def _final_points(self, s: GameState, trick_winner) -> jax.Array:
    """OpenSpiel ComputeScore.

    Args:
      s: The state (for the declarer and the go-alone decisions).
      trick_winner: int8[5], the seat that took each trick.

    Returns:
      int32[4], each seat's points.
    """
    side = _i32(s.declarer) % 2
    makers = jnp.arange(NUM_PLAYERS) % 2 == side
    w = _i32(trick_winner)
    made = jnp.sum((w >= 0) & (w % 2 == side), dtype=jnp.int32)
    score = jnp.where(
        made <= 2,
        jnp.where(s.lone_defender >= 0, -4, -2),
        jnp.where(made <= 4, 1, jnp.where(s.go_alone == 1, 4, 2)),
    )
    return _i32(jnp.where(makers, score, -score))

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Applies a chance outcome or the current player's action.

    A terminal state is returned unchanged. The action is not validated
    (OpenSpiel's engine rejects an illegal one; use `legal_actions_mask`).

    Args:
      state: The current state.
      action: The chance outcome or action.

    Returns:
      The next state.
    """
    a = _i32(action)
    card = jnp.clip(a, 0, NUM_CARDS - 1)
    candidates = (
        self._dealer_step(state, a),
        self._deal_step(state, card),
        self._bid_step(state, a),
        self._discard_step(state, card),
        self._go_alone_step(state, a),
        self._play_step(state, card),
        state,
    )

    def pick(*leaves):
      """The leaf of the phase's candidate state.

      Args:
        *leaves: The leaf of each candidate, in phase order, then the unchanged
          state's.

      Returns:
        The chosen leaf.
      """
      out = leaves[-1]
      for ph in range(PHASE_PLAY, PHASE_DEALER_SELECTION - 1, -1):
        out = jnp.where(state.phase == ph, leaves[ph], out)
      return out

    nxt = jax.tree.map(pick, *candidates)
    return nxt._replace(is_terminal=nxt.phase == PHASE_OVER)

  # -- queries ------------------------------------------------------------

  def _bid_mask(self, s: GameState) -> jax.Array:
    """OpenSpiel BiddingLegalActions.

    Args:
      s: The state, bidding.

    Returns:
      bool[31].
    """
    acts = jnp.arange(NUM_DISTINCT_ACTIONS)
    passes = _i32(s.num_passes)
    can_pass = jnp.asarray(True)
    if self.stick_the_dealer:
      can_pass = passes != 2 * NUM_PLAYERS - 1
    up_suit = card_suit(_i32(s.upcard))
    named = acts - CLUBS_TRUMP_ACTION
    is_suit = (named >= 0) & (named < NUM_SUITS)
    suits = is_suit & jnp.where(
        passes < NUM_PLAYERS, named == up_suit, named != up_suit
    )
    return suits | ((acts == PASS_ACTION) & can_pass)

  def _play_mask(self, s: GameState) -> jax.Array:
    """OpenSpiel PlayLegalActions: follow the led suit (left bower is trump).

    Args:
      s: The state, at a play.

    Returns:
      bool[24], the legal cards.
    """
    t = _i32(s.n_played)
    n_active = jnp.sum(s.active, dtype=jnp.int32)
    hand = _i32(s.holder) == s.current_player
    leading = t % n_active == 0
    trump = _i32(s.trump)
    lead_card = _i32(s.play[jnp.maximum(t - t % n_active, 0)])
    led = effective_suit(lead_card, trump)
    follow = hand & (effective_suit(_cards(), trump) == led) & ~leading
    return jnp.where(follow.any(), follow, hand)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """The current player's legal actions.

    At chance nodes OpenSpiel's LegalActions (the dealers, the undealt cards).

    Args:
      state: The current state.

    Returns:
      bool[31].
    """
    acts = jnp.arange(NUM_DISTINCT_ACTIONS)
    pad = jnp.zeros((NUM_DISTINCT_ACTIONS - NUM_CARDS,), jnp.bool_)
    hand = _i32(state.holder) == state.current_player
    ph = state.phase
    return jnp.select(
        [
            ph == PHASE_DEALER_SELECTION,
            ph == PHASE_DEAL,
            ph == PHASE_BIDDING,
            ph == PHASE_DISCARD,
            ph == PHASE_GO_ALONE,
            ph == PHASE_PLAY,
        ],
        [
            acts < NUM_PLAYERS,
            jnp.concatenate([state.holder < 0, pad]),
            self._bid_mask(state),
            jnp.concatenate([hand & (_cards() != state.upcard), pad]),
            (acts == GO_ALONE_ACTION) | (acts == PLAY_WITH_PARTNER_ACTION),
            jnp.concatenate([self._play_mask(state), pad]),
        ],
        jnp.zeros((NUM_DISTINCT_ACTIONS,), jnp.bool_),
    )

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """The chance distribution: uniform dealers, then uniform undealt cards.

    Args:
      state: The current state.

    Returns:
      f32[24].
    """
    undealt = (state.holder < 0).astype(jnp.float32)
    deal = undealt / jnp.maximum(undealt.sum(), 1.0)
    dealers = jnp.where(
        _cards() < NUM_PLAYERS, jnp.float32(1.0 / NUM_PLAYERS), jnp.float32(0.0)
    )
    return jnp.where(
        state.phase == PHASE_DEALER_SELECTION, dealers, deal
    ).astype(jnp.float32)

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node.
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Whether the hand is over.

    Args:
      state: The current state.

    Returns:
      bool[].
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Whether the dealer or a card is being dealt.

    Args:
      state: The current state.

    Returns:
      bool[].
    """
    return (state.phase == PHASE_DEALER_SELECTION) | (state.phase == PHASE_DEAL)

  def returns(self, state: GameState) -> jax.Array:
    """The makers' score (negated for the defenders) at the end; zeros before.

    Args:
      state: The current state.

    Returns:
      f32[4].
    """
    return jnp.where(state.is_terminal, state.points, 0).astype(jnp.float32)

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """OpenSpiel's f32[959] information-state tensor (module doc).

    Args:
      state: The current state.
      observing_player: The observing seat.

    Returns:
      f32[959].
    """
    return info_state_tensor(
        seat=observing_player,
        dealer=state.dealer,
        upcard=state.upcard,
        num_passes=state.num_passes,
        trump=state.trump,
        declarer=state.declarer,
        go_alone=state.go_alone,
        lone_defender=state.lone_defender,
        discard=state.discard,
        hand=state.holder == observing_player,
        play=state.play,
        play_seat=state.play_seat,
        n_played=state.n_played,
    )

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """The information-state tensor.

    OpenSpiel euchre has no observation tensor.

    Args:
      state: The current state.
      observing_player: The observing seat.

    Returns:
      f32[959].
    """
    return self.information_state_tensor(state, observing_player)

  def information_state_from_os(self, os_state: Any, player: int) -> jax.Array:
    """Returns OpenSpiel's information-state tensor of `player`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      f32[959].
    """
    return jnp.asarray(os_state.information_state_tensor(player), jnp.float32)

  def observation_from_os(self, os_state: Any, player: int) -> jax.Array:
    """Returns OpenSpiel's information-state tensor of `player`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      f32[959].
    """
    return self.information_state_from_os(os_state, player)
