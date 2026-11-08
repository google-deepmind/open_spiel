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

"""Hearts in pure JAX: OpenSpiel's ``hearts``, all eight game parameters.

Four players, each for themselves, avoid taking hearts (1 point each) and the
queen of spades (13). Encodings are pyspiel's, so actions, chance outcomes,
players, returns and the information-state tensor compare with
``pyspiel.load_game("hearts")`` without translation:

- seats 0..3 are North, East, South, West;
- card = rank * 4 + suit, suit 0..3 = clubs, diamonds, hearts, spades, rank
  0..12 = 2..A (2C is 0, QS is 43, JD is 37);
- actions 0..51 are cards (dealt by chance, passed and played by players);
  the first chance node's outcomes 0..3 are the pass direction (no pass,
  left, across, right).

Phases (pyspiel's hearts.cc). Chance draws the pass direction, uniformly
over the four, then deals 52 cards one at a time (the i-th card dealt goes to
seat i % 4). Unless the direction is "no pass", seats 0, 1, 2, 3 in turn each
pass three cards, one action per card; once West has passed, seat p's cards
go to seat (p + direction) % 4. The holder of 2C leads it. Players follow suit
when they can; the highest card of the suit led wins the trick, and its
winner leads next. Hearts may not be led until broken (a heart was played,
or the queen of spades with ``qs_breaks_hearts``), unless the leader holds
only hearts. With ``no_pts_on_first_trick`` a player void in clubs on the
first trick may not discard a heart or the queen of spades unless that is all
they hold.

Score (pyspiel's ComputeScore). Each trick's points go to its winner. A
player who wins every trick that carries points (any nonzero total other
than the jack of diamonds alone) has shot the moon: their points drop by 26
and everyone else's rise by 26. With ``avoid_all_tricks_bonus`` (and no moon)
a player who won no trick scores -5. Returns are 26 - points, so they are
general-sum: 78 in total on a normal hand, 26 when the moon is shot.

A pyspiel quirk, reproduced. With ``pass_cards=false`` pyspiel's legal
actions at the pass-direction node are "no pass" only, but its chance
outcomes are still the four directions uniformly, and a sampled direction is
applied as is (cards are passed 3/4 of the time). This game reproduces both:
``legal_actions_mask`` at that node holds "no pass" only, ``chance_outcomes``
is uniform over the four.

Information state. ``information_state_tensor`` is pyspiel's, f32[5088]:
the pass direction, the dealt hand, the passed cards, the received cards
(once West has passed), the current hand, a thermometer of each seat's
points (points + 10 ones of 36), then the 13 tricks, each as 7 slots of 52
(N E S W N E S) with the j-th card of a trick led by seat L in slot L + j.
It is all zeros at chance nodes. At a terminal state after a moon pyspiel's
thermometer writes past its 36 cells (into the next blocks); this game clips
each thermometer to its 36 cells, so terminal tensors may differ from
pyspiel's there. pyspiel has no observation tensor; ``observation_tensor``
is the information-state tensor (a seat's perfect-recall view).

Checked move for move against pyspiel's hearts across all option combinations
(openspiel_equivalence).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_CARDS_PER_SUIT = 13
NUM_CARDS = 52
NUM_CARDS_IN_PASS = 3
NUM_TRICKS = NUM_CARDS // NUM_PLAYERS
CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3
TWO_OF_CLUBS = 0
QUEEN_OF_SPADES = 10 * NUM_SUITS + SPADES  # 43
JACK_OF_DIAMONDS = 9 * NUM_SUITS + DIAMONDS  # 37
POINTS_FOR_HEART = 1
POINTS_FOR_QS = 13
POINTS_FOR_JD = -10
TOTAL_POSITIVE_POINTS = 26
MAX_SCORE = TOTAL_POSITIVE_POINTS - POINTS_FOR_JD  # 36
AVOID_ALL_TRICKS_BONUS = -5
PASS_NONE, PASS_LEFT, PASS_ACROSS, PASS_RIGHT = 0, 1, 2, 3
#: Player decisions in the longest hand (pyspiel MaxGameLength): 12 passes, 52
#: plays.
MAX_GAME_LENGTH = NUM_CARDS_IN_PASS * NUM_PLAYERS + NUM_CARDS
TRICK_TENSOR_SIZE = NUM_CARDS * 7
# 5088
INFO_STATE_SIZE = (
    NUM_PLAYERS
    + 4 * NUM_CARDS
    + NUM_PLAYERS * MAX_SCORE
    + NUM_TRICKS * TRICK_TENSOR_SIZE
)

PHASE_PASS_DIR, PHASE_DEAL, PHASE_PASS, PHASE_PLAY, PHASE_OVER = 0, 1, 2, 3, 4

PARAMS = (
    "pass_cards",
    "no_pts_on_first_trick",
    "can_lead_any_club",
    "jd_bonus",
    "avoid_all_tricks_bonus",
    "must_break_hearts",
    "qs_breaks_hearts",
    "can_lead_hearts_instead_of_qs",
)


class GameState(NamedTuple):
  """A hand of hearts. -1 marks an empty card, seat or slot."""

  phase: jax.Array  # int8
  pass_dir: jax.Array  # int8
  # int32: seat, CHANCE_PLAYER_ID, or TERMINAL_PLAYER_ID
  current_player: jax.Array
  is_terminal: jax.Array  # bool
  n_dealt: jax.Array  # int8
  holder: jax.Array  # int8[52]: the seat holding each card now
  # int8[52]: the seat dealt each card (set after the deal)
  initial_deal: jax.Array
  passed: jax.Array  # int8[4, 3]: the cards each seat passed, in order
  n_passed: jax.Array  # int8[4]
  play: jax.Array  # int8[52]: the cards in order of play
  play_seat: jax.Array  # int8[52]: who played each of them
  n_played: jax.Array  # int8
  trick_winner: jax.Array  # int8[13]
  trick_points: jax.Array  # int32[13]
  hearts_broken: jax.Array  # bool
  points: jax.Array  # int32[4]


def _i8(v) -> jax.Array:
  """`v` as int8.

  Args:
    v: A value or array.

  Returns:
    The int8 array.
  """
  return jnp.asarray(v, jnp.int8)


def card_suit(card):
  """The suit of a card (card % 4).

  Args:
    card: The card id(s).

  Returns:
    The suit(s).
  """
  return card % NUM_SUITS


def card_rank(card):
  """The rank of a card (card // 4).

  Args:
    card: The card id(s).

  Returns:
    The rank(s).
  """
  return card // NUM_SUITS


def _cards() -> jax.Array:
  """The card ids, built at use (no array is made at import).

  Returns:
    int32[52], 0..51.
  """
  return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def _suits() -> jax.Array:
  """Each card id's suit.

  Returns:
    int32[52].
  """
  return _cards() % NUM_SUITS


class Hearts(game_lib.Game, interop.PyspielInterop):
  """OpenSpiel hearts (module doc); the parameters are pyspiel's."""

  def __init__(
      self,
      pass_cards: bool = True,
      no_pts_on_first_trick: bool = True,
      can_lead_any_club: bool = False,
      jd_bonus: bool = False,
      avoid_all_tricks_bonus: bool = False,
      must_break_hearts: bool = True,
      qs_breaks_hearts: bool = True,
      can_lead_hearts_instead_of_qs: bool = False,
  ):
    values = (
        pass_cards,
        no_pts_on_first_trick,
        can_lead_any_club,
        jd_bonus,
        avoid_all_tricks_bonus,
        must_break_hearts,
        qs_breaks_hearts,
        can_lead_hearts_instead_of_qs,
    )
    for name, v in zip(PARAMS, values):
      if not isinstance(v, bool):
        raise TypeError(f"hearts parameter {name} must be a bool, got {v!r}")
    self.params = dict(zip(PARAMS, values))
    (
        self.pass_cards,
        self.no_pts_on_first_trick,
        self.can_lead_any_club,
        self.jd_bonus,
        self.avoid_all_tricks_bonus,
        self.must_break_hearts,
        self.qs_breaks_hearts,
        self.can_lead_hearts_instead_of_qs,
    ) = values
    pts = [0] * NUM_CARDS
    for c in range(NUM_CARDS):
      if c % NUM_SUITS == HEARTS:
        pts[c] = POINTS_FOR_HEART
      elif c == QUEEN_OF_SPADES:
        pts[c] = POINTS_FOR_QS
      elif c == JACK_OF_DIAMONDS and jd_bonus:
        pts[c] = POINTS_FOR_JD
    self._card_points = tuple(pts)

  @property
  def num_actions(self) -> int:
    """The 52 cards."""
    return NUM_CARDS

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
    """The 52 cards (four pass directions use 0..3)."""
    return NUM_CARDS

  @property
  def max_game_length(self) -> int:
    """The longest game in plies, chance included."""
    return MAX_GAME_LENGTH

  def card_points(self, card) -> jax.Array:
    """The points of a card under this game's jd_bonus.

    Args:
      card: The card id(s).

    Returns:
      int32 points.
    """
    return jnp.asarray(self._card_points, jnp.int32)[card]

  def new_initial_state(self) -> GameState:
    """The state before the pass direction is drawn.

    Returns:
      The initial GameState (a chance node).
    """
    return GameState(
        phase=_i8(PHASE_PASS_DIR),
        pass_dir=_i8(PASS_NONE),
        current_player=jnp.asarray(game_lib.CHANCE_PLAYER_ID, jnp.int32),
        is_terminal=jnp.asarray(False),
        n_dealt=_i8(0),
        holder=jnp.full((NUM_CARDS,), -1, jnp.int8),
        initial_deal=jnp.full((NUM_CARDS,), -1, jnp.int8),
        passed=jnp.full((NUM_PLAYERS, NUM_CARDS_IN_PASS), -1, jnp.int8),
        n_passed=jnp.zeros((NUM_PLAYERS,), jnp.int8),
        play=jnp.full((NUM_CARDS,), -1, jnp.int8),
        play_seat=jnp.full((NUM_CARDS,), -1, jnp.int8),
        n_played=_i8(0),
        trick_winner=jnp.full((NUM_TRICKS,), -1, jnp.int8),
        trick_points=jnp.zeros((NUM_TRICKS,), jnp.int32),
        hearts_broken=jnp.asarray(not self.must_break_hearts),
        points=jnp.zeros((NUM_PLAYERS,), jnp.int32),
    )

  # -- transitions --------------------------------------------------------

  def _pass_dir_step(self, s: GameState, a) -> GameState:
    """The pass-direction chance outcome `a`.

    Args:
      s: The state.
      a: The pass direction.

    Returns:
      The state, now dealing.
    """
    return s._replace(pass_dir=_i8(a), phase=_i8(PHASE_DEAL))

  def _deal_step(self, s: GameState, a) -> GameState:
    """Deals card `a` to the next seat in turn.

    Args:
      s: The state.
      a: The card dealt.

    Returns:
      The state; after the last card, passing or playing.
    """
    holder = s.holder.at[a].set((s.n_dealt % NUM_PLAYERS).astype(jnp.int8))
    n = (s.n_dealt + 1).astype(jnp.int8)
    done = n == NUM_CARDS
    no_pass = s.pass_dir == PASS_NONE
    phase = jnp.where(
        done, jnp.where(no_pass, PHASE_PLAY, PHASE_PASS), PHASE_DEAL
    )
    cp = jnp.where(
        done,
        jnp.where(no_pass, holder[TWO_OF_CLUBS].astype(jnp.int32), 0),
        game_lib.CHANCE_PLAYER_ID,
    )
    return s._replace(
        holder=holder,
        n_dealt=n,
        initial_deal=jnp.where(done, holder, s.initial_deal),
        phase=_i8(phase),
        current_player=cp.astype(jnp.int32),
    )

  def _pass_step(self, s: GameState, a) -> GameState:
    """The current player passes card `a`.

    The passed cards change hands once the last passer has passed.

    Args:
      s: The state.
      a: The card passed.

    Returns:
      The state.
    """
    p = s.current_player
    k = s.n_passed[p].astype(jnp.int32)
    passed = s.passed.at[p, k].set(_i8(a))
    n_passed = s.n_passed.at[p].add(1)
    holder = s.holder.at[a].set(-1)
    p_next = jnp.where((k + 1) % NUM_CARDS_IN_PASS == 0, p + 1, p)
    all_done = p_next == NUM_PLAYERS
    receiver = (
        jnp.arange(NUM_PLAYERS) + s.pass_dir.astype(jnp.int32)
    ) % NUM_PLAYERS
    recv = (
        jnp.broadcast_to(receiver[:, None], passed.shape)
        .reshape(-1)
        .astype(jnp.int8)
    )
    flat = passed.reshape(-1).astype(jnp.int32)
    dealt = holder.at[jnp.where(flat >= 0, flat, NUM_CARDS)].set(
        recv, mode="drop"
    )
    holder = jnp.where(all_done, dealt, holder)
    cp = jnp.where(all_done, holder[TWO_OF_CLUBS].astype(jnp.int32), p_next)
    return s._replace(
        passed=passed,
        n_passed=n_passed,
        holder=holder,
        phase=_i8(jnp.where(all_done, PHASE_PLAY, PHASE_PASS)),
        current_player=cp.astype(jnp.int32),
    )

  def _play_step(self, s: GameState, a) -> GameState:
    """The current player plays card `a`; a full trick is scored.

    Args:
      s: The state.
      a: The card played.

    Returns:
      The state; after the last trick, terminal with the final points.
    """
    p = s.current_player
    t = s.n_played.astype(jnp.int32)
    holder = s.holder.at[a].set(-1)
    play = s.play.at[t].set(_i8(a))
    play_seat = s.play_seat.at[t].set(_i8(p))
    broken = s.hearts_broken | (card_suit(a) == HEARTS)
    if self.qs_breaks_hearts:
      broken = broken | (a == QUEEN_OF_SPADES)
    n = t + 1
    trick_done = n % NUM_PLAYERS == 0
    k = t // NUM_PLAYERS
    first = k * NUM_PLAYERS
    cards = jax.lax.dynamic_slice(
        play.astype(jnp.int32), (first,), (NUM_PLAYERS,)
    )
    seats = jax.lax.dynamic_slice(
        play_seat.astype(jnp.int32), (first,), (NUM_PLAYERS,)
    )
    led = card_suit(cards[0])
    key = jnp.where(card_suit(cards) == led, card_rank(cards), -1)
    winner = seats[jnp.argmax(key)]
    tpts = jnp.sum(self.card_points(jnp.maximum(cards, 0)), dtype=jnp.int32)
    trick_winner = jnp.where(
        trick_done, s.trick_winner.at[k].set(_i8(winner)), s.trick_winner
    )
    trick_points = jnp.where(
        trick_done, s.trick_points.at[k].set(tpts), s.trick_points
    )
    points = jnp.where(trick_done, s.points.at[winner].add(tpts), s.points)
    over = n == NUM_CARDS
    points = jnp.where(
        over, self._final_points(points, trick_winner, trick_points), points
    )
    cp = jnp.where(
        over,
        game_lib.TERMINAL_PLAYER_ID,
        jnp.where(trick_done, winner, (p + 1) % NUM_PLAYERS),
    )
    return s._replace(
        holder=holder,
        play=play,
        play_seat=play_seat,
        n_played=_i8(n),
        hearts_broken=broken,
        trick_winner=trick_winner,
        trick_points=trick_points,
        points=points,
        phase=_i8(jnp.where(over, PHASE_OVER, PHASE_PLAY)),
        current_player=cp.astype(jnp.int32),
    )

  def _final_points(self, points, trick_winner, trick_points) -> jax.Array:
    """Pyspiel ComputeScore: the moon (per-trick test) and the avoid bonus.

    Args:
      points: int32[4], each seat's points from its tricks.
      trick_winner: int8[13], the seat that took each trick.
      trick_points: int32[13], the points of each trick.

    Returns:
      int32[4], the final points.
    """
    winners = trick_winner.astype(jnp.int32)
    counted = (trick_points != 0) & (trick_points != POINTS_FOR_JD)
    shooter = winners[jnp.argmax(counted)]
    moon = jnp.all(~counted | (winners == shooter))
    seats = jnp.arange(NUM_PLAYERS)
    out = jnp.where(
        moon,
        points
        + jnp.where(
            seats == shooter, -TOTAL_POSITIVE_POINTS, TOTAL_POSITIVE_POINTS
        ),
        points,
    )
    if self.avoid_all_tricks_bonus:
      took = jnp.any(winners[None, :] == seats[:, None], axis=1)
      out = jnp.where(~moon & ~took, out + AVOID_ALL_TRICKS_BONUS, out)
    return out.astype(jnp.int32)

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Applies a chance outcome or the current player's action.

    A terminal state is returned unchanged.

    Args:
      state: The current state.
      action: The chance outcome or action.

    Returns:
      The next state.
    """
    a = jnp.asarray(action, jnp.int32)
    candidates = (
        self._pass_dir_step(state, a),
        self._deal_step(state, a),
        self._pass_step(state, a),
        self._play_step(state, a),
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
      for ph in range(PHASE_PLAY, PHASE_PASS_DIR - 1, -1):
        out = jnp.where(state.phase == ph, leaves[ph], out)
      return out

    nxt = jax.tree.map(pick, *candidates)
    return nxt._replace(is_terminal=nxt.phase == PHASE_OVER)

  # -- queries ------------------------------------------------------------

  def _play_mask(self, s: GameState) -> jax.Array:
    """Pyspiel PlayLegalActions, rule by rule in its order.

    Args:
      s: The state, at a play.

    Returns:
      bool[52], the legal cards.
    """
    p = s.current_player
    t = s.n_played.astype(jnp.int32)
    hand = s.holder.astype(jnp.int32) == p
    leading = t % NUM_PLAYERS == 0
    led = card_suit(
        s.play[jnp.maximum(t - t % NUM_PLAYERS, 0)].astype(jnp.int32)
    )
    follow = hand & (_suits() == led) & ~leading
    first_lead = _cards() == TWO_OF_CLUBS
    if self.can_lead_any_club:
      first_lead = first_lead | (hand & (_suits() == CLUBS))
    no_pts = hand & (_cards() != QUEEN_OF_SPADES) & (_suits() != HEARTS)
    no_pts = (
        no_pts & (t < NUM_PLAYERS)
        if self.no_pts_on_first_trick
        else no_pts & False
    )
    lead = hand & ((_suits() != HEARTS) | s.hearts_broken) & leading
    if self.can_lead_hearts_instead_of_qs:
      only_qs = (jnp.sum(lead) == 1) & lead[QUEEN_OF_SPADES]
      lead = lead & ~only_qs
    out = hand
    for rule in (lead, no_pts):
      out = jnp.where(rule.any(), rule, out)
    out = jnp.where(t == 0, first_lead, out)
    return jnp.where(follow.any(), follow, out)

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """The current player's legal actions.

    At chance nodes pyspiel's LegalActions (the pass directions, the undealt
    cards).

    Args:
      state: The current state.

    Returns:
      bool[52].
    """
    pass_dir = (
        (_cards() < NUM_PLAYERS) if self.pass_cards else (_cards() == PASS_NONE)
    )
    hand = state.holder.astype(jnp.int32) == state.current_player
    ph = state.phase
    return jnp.select(
        [
            ph == PHASE_PASS_DIR,
            ph == PHASE_DEAL,
            ph == PHASE_PASS,
            ph == PHASE_PLAY,
        ],
        [pass_dir, state.holder < 0, hand, self._play_mask(state)],
        jnp.zeros((NUM_CARDS,), jnp.bool_),
    )

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """The chance distribution.

    Uniform over the four pass directions (whatever pass_cards says; module
    doc), then over the undealt cards.

    Args:
      state: The current state.

    Returns:
      f32[52].
    """
    undealt = (state.holder < 0).astype(jnp.float32)
    deal = undealt / jnp.maximum(undealt.sum(), 1.0)
    dirs = jnp.where(
        _cards() < NUM_PLAYERS, jnp.float32(1.0 / NUM_PLAYERS), jnp.float32(0.0)
    )
    return jnp.where(state.phase == PHASE_PASS_DIR, dirs, deal).astype(
        jnp.float32
    )

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node.
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[].
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Whether the pass direction or a card is being dealt.

    Args:
      state: The current state.

    Returns:
      bool[].
    """
    return (state.phase == PHASE_PASS_DIR) | (state.phase == PHASE_DEAL)

  def returns(self, state: GameState) -> jax.Array:
    """26 - points at the end; zeros before.

    Args:
      state: The current state.

    Returns:
      f32[4].
    """
    return jnp.where(
        state.is_terminal, (TOTAL_POSITIVE_POINTS - state.points), 0
    ).astype(jnp.float32)

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Pyspiel's f32[5088] information-state tensor (module doc).

    Args:
      state: The current state.
      observing_player: The observing seat.

    Returns:
      f32[5088].
    """
    return info_state_tensor(
        pass_dir=state.pass_dir,
        dealt=state.initial_deal == observing_player,
        passed=_card_set(state.passed[observing_player]),
        received=received_cards(state, observing_player),
        hand=state.holder == observing_player,
        points=state.points,
        play=state.play,
        play_seat=state.play_seat,
        n_played=state.n_played,
        live=~self.is_chance_node(state),
    )

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """The information-state tensor.

    Pyspiel hearts has no observation tensor.

    Args:
      state: The current state.
      observing_player: The observing seat.

    Returns:
      f32[5088].
    """
    return self.information_state_tensor(state, observing_player)

  def information_state_from_os(self, os_state: Any, player: int) -> jax.Array:
    """Returns pyspiel's information-state tensor of `player`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      f32[5088].
    """
    return jnp.asarray(os_state.information_state_tensor(player), jnp.float32)

  def observation_from_os(self, os_state: Any, player: int) -> jax.Array:
    """Returns pyspiel's information-state tensor of `player`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      f32[5088].
    """
    return self.information_state_from_os(os_state, player)


def _card_set(cards) -> jax.Array:
  """bool[52] of an int array of cards with -1 pads.

  Args:
    cards: Cards, -1 for none.

  Returns:
    bool[52].
  """
  cards = jnp.asarray(cards, jnp.int32).reshape(-1)
  return (
      jnp.zeros((NUM_CARDS,), jnp.bool_)
      .at[jnp.where(cards >= 0, cards, NUM_CARDS)]
      .set(True, mode="drop")
  )


def received_cards(state: GameState, seat) -> jax.Array:
  """The cards `seat` received, once West (the last passer) has passed.

  Args:
    state: The current state.
    seat: The seat.

  Returns:
    bool[52].
  """
  seat = jnp.asarray(seat, jnp.int32)
  passer = (seat - state.pass_dir.astype(jnp.int32) + NUM_PLAYERS) % NUM_PLAYERS
  shown = state.n_passed[NUM_PLAYERS - 1] == NUM_CARDS_IN_PASS
  return _card_set(state.passed[passer]) & shown


def info_state_tensor(
    *,
    pass_dir,
    dealt,
    passed,
    received,
    hand,
    points,
    play,
    play_seat,
    n_played,
    live,
) -> jax.Array:
  """Pyspiel's hearts information-state tensor from one seat's view.

  Args:
    pass_dir: int, 0..3.
    dealt: bool[52], the seat's dealt cards.
    passed: bool[52], the cards it passed.
    received: bool[52], the cards it received.
    hand: bool[52], its hand.
    points: int[4], each seat's points so far.
    play: int[52], the cards in order of play (-1 pad).
    play_seat: int[52], who played them (-1 pad).
    n_played: the cards played.
    live: False at a chance node (the tensor is all zeros there).

  Returns:
    f32[5088] (module doc). Each points thermometer is clipped to its 36 cells.
  """
  f32 = jnp.float32
  head = jnp.concatenate([
      jax.nn.one_hot(jnp.asarray(pass_dir, jnp.int32), NUM_PLAYERS, dtype=f32),
      dealt.astype(f32),
      passed.astype(f32),
      received.astype(f32),
      hand.astype(f32),
      (
          jnp.arange(MAX_SCORE)[None, :]
          < (jnp.asarray(points, jnp.int32) - POINTS_FOR_JD)[:, None]
      )
      .astype(f32)
      .reshape(-1),
  ])
  t = jnp.arange(NUM_CARDS)
  play = jnp.asarray(play, jnp.int32)
  seats = jnp.asarray(play_seat, jnp.int32)
  leader = seats[(t // NUM_PLAYERS) * NUM_PLAYERS]
  pos = (
      (t // NUM_PLAYERS) * TRICK_TENSOR_SIZE
      + (leader + t % NUM_PLAYERS) * NUM_CARDS
      + jnp.maximum(play, 0)
  )
  ok = t < jnp.asarray(n_played, jnp.int32)
  tricks = (
      jnp.zeros((NUM_TRICKS * TRICK_TENSOR_SIZE,), f32)
      .at[jnp.where(ok, pos, NUM_TRICKS * TRICK_TENSOR_SIZE)]
      .set(1.0, mode="drop")
  )
  out = jnp.concatenate([head, tricks])
  return jnp.where(live, out, jnp.zeros_like(out))
