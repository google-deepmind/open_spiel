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

"""Contract bridge (auction and card play) in pure JAX.

This is OpenSpiel's ``bridge`` game with ``use_double_dummy_result=false``:
four players in two partnerships deal, bid and play all thirteen tricks, and
the board is scored with duplicate bridge scoring.

OpenSpiel's default, ``use_double_dummy_result=true``, is another game: the
board ends after the auction and a double-dummy solver's trick count scores
it. This game does not compute that, so it refuses OpenSpiel's default:
``load_game("bridge")`` and ``Bridge()`` raise, and every caller states
``use_double_dummy_result=false``.

Encodings are pyspiel's, so actions, chance outcomes, players and returns
compare with ``pyspiel.load_game("bridge(use_double_dummy_result=false)")``
without translation:

- seats 0..3 are North, East, South, West; partnership = seat & 1 (0 is
  North-South, 1 is East-West);
- card = rank * 4 + suit, suit 0..3 = clubs, diamonds, hearts, spades and
  rank 0..12 = 2..A;
- actions 0..51 are cards (dealt by chance, played by players), 52 pass,
  53 double, 54 redouble and 55..89 the 35 bids 1C, 1D, 1H, 1S, 1NT, 2C, ...,
  7NT;
- strain 0..4 = clubs, diamonds, hearts, spades, no trump.

Phases. A board starts with 52 chance nodes: the i-th dealt card goes to seat
i % 4, as in pyspiel. The dealer calls first, then the auction runs
clockwise. A bid must outrank the last bid; a double applies only to an
undoubled opposing contract and a redouble only to a doubled contract of the
caller's side. Four passes before any bid throw the board in (score 0). After
a bid, three passes end the auction; the declarer is the first player of the
declaring partnership to have named the final strain. Declarer's left-hand
opponent leads; players follow suit when they can; the highest trump, else
the highest card of the suit led, wins the trick and its winner leads next.
Declarer chooses dummy's cards: while dummy is due to play, the current
player is the declarer and the legal actions are dummy's cards.

Returns are duplicate points (Law 77, as pyspiel's ``bridge_scoring.cc``):
the declaring side scores +score and the defenders -score. ``imps`` converts
a point difference between two tables into International Match Points.

Dealer and vulnerability are game parameters, as in pyspiel (``dealer``,
``dealer_vul``, ``non_dealer_vul``). With ``random_dealer_vul=True`` (an
extension with no pyspiel counterpart) the board starts with one
more chance node, uniform over the 16 positions of the duplicate board cycle,
which sets the dealer and both sides' vulnerability.

The observation tensor is pyspiel's (``num_tricks_in_observation`` past tricks
in the card-play view, the auction view otherwise), with the auction seats
taken relative to the dealer (the bidder of call k is ``(dealer + k) % 4``),
so the tensor is correct for every dealer, as in OpenSpiel's current bridge.
OpenSpiel releases up to 2.0.2 place auction calls as if North always dealt;
the two agree whenever North deals. The information state tensor equals the
observation, as in pyspiel. At a terminal state a side with 13 tricks has no
slot in the trick count blocks and that count is omitted.

Everything is per board (no batch axis); ``jax.vmap`` batches it. Every phase
computes its successor and ``jnp.where`` selects one, so ``apply_action`` is
branch-free under ``vmap``.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import interop

NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = 52
NUM_TRICKS = 13
NUM_STRAINS = 5
NUM_LEVELS = 7
NUM_BIDS = NUM_LEVELS * NUM_STRAINS  # 35
NUM_OTHER_CALLS = 3  # pass, double, redouble
NUM_CALLS = NUM_BIDS + NUM_OTHER_CALLS  # 38
NO_TRUMP = 4

PASS_ACTION = NUM_CARDS  # 52
DOUBLE_ACTION = NUM_CARDS + 1  # 53
REDOUBLE_ACTION = NUM_CARDS + 2  # 54

FIRST_BID_ACTION = NUM_CARDS + NUM_OTHER_CALLS  # 55
NUM_ACTIONS = NUM_CARDS + NUM_CALLS  # 90

# A call index is action - 52: 0 pass, 1 double, 2 redouble, 3 + bid index.
CALL_PASS = 0
CALL_DOUBLE = 1
CALL_REDOUBLE = 2
FIRST_BID_CALL = 3

# The longest auction: three opening passes, then for each of the 35 bids
# "bid, pass, pass, double, pass, pass, redouble, pass, pass", then a final
# pass. pyspiel's kMaxAuctionLength.
MAX_AUCTION_LENGTH = NUM_BIDS * (1 + 2 * NUM_PLAYERS) + NUM_PLAYERS  # 319
# Player decisions in the longest board: pyspiel's MaxGameLength.
MAX_GAME_LENGTH = MAX_AUCTION_LENGTH + NUM_CARDS  # 371

PHASE_BOARD = 0  # the duplicate-cycle chance node (random_dealer_vul only)
PHASE_DEAL = 1
PHASE_AUCTION = 2
PHASE_PLAY = 3
PHASE_OVER = 4

UNDOUBLED = 0
DOUBLED = 1
REDOUBLED = 2

NUM_BOARDS_IN_CYCLE = 16
# The duplicate board cycle (pyspiel BridgeGame::NewDuplicateBridgeInitialState
# and the standard board layout): position b (board b + 1) has dealer b % 4;
# these give whether the dealer's side and the other side are vulnerable. The
# 16 positions hold every (dealer, vulnerability) pair exactly once.
BOARD_DEALER_VUL = (
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
    False,
    False,
    True,
    False,
    True,
    True,
)
BOARD_NON_DEALER_VUL = (
    False,
    True,
    True,
    True,
    False,
    False,
    True,
    False,
    True,
    True,
    False,
    True,
    True,
    False,
    False,
    False,
)

# Largest absolute duplicate score, pyspiel's kMaxScore.
MAX_SCORE = 7600

# Lowest point difference worth n IMPs, n = 1..24 (WBF scoring table).
IMP_BAND_FLOORS = (
    20,
    50,
    90,
    130,
    170,
    220,
    270,
    320,
    370,
    430,
    500,
    600,
    750,
    900,
    1100,
    1300,
    1500,
    1750,
    2000,
    2250,
    2500,
    3000,
    3500,
    4000,
)

NUM_OBSERVATION_TYPES = 4  # auction, opening lead, declaring, defending
AUCTION_TENSOR_SIZE = (
    NUM_PLAYERS * (1 + 3 * NUM_BIDS) + NUM_CARDS + 2 * 2
)  # 480


def play_tensor_size(num_tricks_in_observation: int) -> int:
  """Returns the size of pyspiel's card-play observation block.

  The size excludes the 4 types.

  Args:
    num_tricks_in_observation: The number of tricks the block shows.

  Returns:
    The size.
  """
  return (
      NUM_LEVELS
      + NUM_STRAINS
      + NUM_OTHER_CALLS
      + NUM_PLAYERS
      + 2
      + NUM_CARDS
      + NUM_CARDS
      + num_tricks_in_observation * NUM_PLAYERS * NUM_CARDS
      + NUM_TRICKS
      + NUM_TRICKS
  )


class GameState(NamedTuple):
  """One bridge board. Card arrays are indexed by card id.

  Attributes:
    calls: int8[319] call index of each call, -1 padding.
    current_player: int32 pyspiel CurrentPlayer (-1 chance, -4 over).
    dbl: int8 UNDOUBLED / DOUBLED / REDOUBLED.
    deal: int8[52] seat each card was dealt to, -1 not yet dealt.
    dealer: int8 seat.
    decl_tricks: int8 tricks won by the declaring side.
    declarer: int8 seat, -1 none.
    first_strain_bidder: int8[2, 5] per side and strain, -1 none.
    holder: int8[52] seat holding each card, -1 undealt or played.
    is_terminal: bool.
    last_bid: int8 bid index 0..34 of the last bid, -1 none.
    level: int8 contract level 1..7, 0 none.
    n_calls: int16 calls made.
    n_dealt: int8 cards dealt so far.
    n_played: int8 cards played.
    num_passes: int8 consecutive passes.
    on_move: int8 seat whose call or card is due (may be dummy).
    phase: int8 PHASE_*.
    score: int32[4] duplicate points per seat (0 until scored).
    strain: int8 contract strain 0..4, -1 none.
    trick_cards: int8[13, 4] cards of each trick in play order.
    trick_leader: int8[13] seat that led each trick, -1 not led.
    vul: bool[2]: North-South, East-West vulnerable.
  """

  phase: jax.Array
  current_player: jax.Array
  is_terminal: jax.Array
  dealer: jax.Array
  vul: jax.Array
  n_dealt: jax.Array
  deal: jax.Array
  holder: jax.Array
  calls: jax.Array
  n_calls: jax.Array
  last_bid: jax.Array
  dbl: jax.Array
  num_passes: jax.Array
  first_strain_bidder: jax.Array
  level: jax.Array
  strain: jax.Array
  declarer: jax.Array
  trick_cards: jax.Array
  trick_leader: jax.Array
  n_played: jax.Array
  on_move: jax.Array
  decl_tricks: jax.Array
  score: jax.Array


def _i8(v) -> jax.Array:
  """Casts `v` to int8.

  Args:
    v: A value.

  Returns:
    int8 array of `v`.
  """
  return jnp.asarray(v).astype(jnp.int8)


def card_suit(card):
  """Returns the suit of `card`.

  Args:
    card: The card id.

  Returns:
    card % NUM_SUITS.
  """
  return card % NUM_SUITS


def card_rank(card):
  """Returns the rank of `card`.

  Args:
    card: The card id.

  Returns:
    card // NUM_SUITS.
  """
  return card // NUM_SUITS


def partnership(seat):
  """Returns the partnership of `seat`.

  Args:
    seat: The seat.

  Returns:
    0 for North-South, 1 for East-West.
  """
  return seat & 1


def board_dealer_vul(position: int) -> tuple[int, tuple[bool, bool]]:
  """Returns the dealer and vulnerability of a board cycle position.

  Args:
    position: The cycle position, 0..15.

  Returns:
    (dealer, (NS vulnerable, EW vulnerable)).

  Raises:
    ValueError: `position` is not 0..15.
  """
  if not 0 <= position < NUM_BOARDS_IN_CYCLE:
    raise ValueError(f"cycle position must be 0..15, got {position}")
  dealer = position % NUM_PLAYERS
  vul = [False, False]
  vul[partnership(dealer)] = BOARD_DEALER_VUL[position]
  vul[1 - partnership(dealer)] = BOARD_NON_DEALER_VUL[position]
  return dealer, (vul[0], vul[1])


def score(level, strain, dbl, tricks, vulnerable) -> jax.Array:
  """Returns declarer's duplicate score, broadcasting over arrays.

  Term for term this is pyspiel's ``bridge_scoring.cc``: trick points, the
  game or part-score bonus, slam bonuses, the bonus for making a (re)doubled
  contract, overtricks and the doubled undertrick schedule.

  Args:
    level: 1..7; 0 means passed out and scores 0.
    strain: 0..4.
    dbl: UNDOUBLED / DOUBLED / REDOUBLED.
    tricks: 0..13, taken by the declaring side.
    vulnerable: Whether the declaring side is vulnerable.

  Returns:
    int32 score.
  """
  level = jnp.asarray(level, jnp.int32)
  strain = jnp.asarray(strain, jnp.int32)
  dbl = jnp.asarray(dbl, jnp.int32)
  tricks = jnp.asarray(tricks, jnp.int32)
  vul = jnp.asarray(vulnerable, jnp.bool_)
  mult = jnp.where(dbl == DOUBLED, 2, jnp.where(dbl == REDOUBLED, 4, 1))
  half = mult // 2  # 0 undoubled, 1 doubled, 2 redoubled
  per_trick = jnp.where(strain <= 1, 20, 30)
  result = tricks - (6 + level)

  contract_points = (
      level * per_trick + jnp.where(strain == NO_TRUMP, 10, 0)
  ) * mult
  bonus = jnp.where(
      level == 7,
      jnp.where(vul, 2000, 1300),
      jnp.where(
          level == 6,
          jnp.where(vul, 1250, 800),
          jnp.where(contract_points >= 100, jnp.where(vul, 500, 300), 50),
      ),
  )
  over = jnp.maximum(result, 0)
  overtricks = jnp.where(
      dbl == UNDOUBLED, over * per_trick, jnp.where(vul, 100, 50) * over * mult
  )
  made = contract_points + bonus + 50 * half + overtricks

  under = jnp.maximum(-result, 0)
  undoubled = jnp.where(vul, -100, -50) * under
  doubled_vul = -200 - 300 * (under - 1)
  doubled_nonvul = jnp.where(
      under == 1, -100, jnp.where(under == 2, -300, -500 - 300 * (under - 3))
  )
  doubled = jnp.where(vul, doubled_vul, doubled_nonvul) * half
  defeated = jnp.where(dbl == UNDOUBLED, undoubled, doubled)

  out = jnp.where(result >= 0, made, defeated)
  return jnp.where(level == 0, 0, out).astype(jnp.int32)


def imps(diff) -> jax.Array:
  """Returns the IMPs of a point difference, broadcasting.

  Duplicate bridge compares the two tables that played the same board: the
  pair's IMPs are ``imps(NS score at table 1 - NS score at table 2)``.

  Args:
    diff: The point difference.

  Returns:
    int32 IMPs, -24..24.
  """
  diff = jnp.asarray(diff, jnp.int32)
  floors = jnp.asarray(IMP_BAND_FLOORS, jnp.int32)
  n = (jnp.abs(diff)[..., None] >= floors).sum(-1).astype(jnp.int32)
  return jnp.sign(diff).astype(jnp.int32) * n


def dummy(state: GameState) -> jax.Array:
  """Returns the dummy's seat (declarer's partner); meaningless before a bid.

  Args:
    state: The state.

  Returns:
    int32 seat.
  """
  return (state.declarer.astype(jnp.int32) + 2) % NUM_PLAYERS


def dummy_exposed(state: GameState) -> jax.Array:
  """Returns whether dummy's hand is face up (the opening lead was made).

  Args:
    state: The state.

  Returns:
    bool.
  """
  return (state.n_played >= 1) & (state.level > 0)


def action_history(state: GameState) -> tuple[jax.Array, jax.Array]:
  """Returns the player actions so far.

  The calls in order, then the cards in order of play. Chance actions (the
  deal and the board position) are not included.

  Args:
    state: The state.

  Returns:
    (int16[371] player actions, -1 padded; int16 their count).
  """
  n_calls = state.n_calls.astype(jnp.int32)
  n_played = state.n_played.astype(jnp.int32)
  idx = jnp.arange(MAX_GAME_LENGTH, dtype=jnp.int32)
  calls = jnp.concatenate(
      [state.calls.astype(jnp.int32), jnp.full((NUM_CARDS,), -1, jnp.int32)]
  )
  call_actions = jnp.where(idx < n_calls, calls + PASS_ACTION, -1)
  cards = state.trick_cards.astype(jnp.int32).reshape(-1)
  card_pos = jnp.clip(idx - n_calls, 0, NUM_CARDS - 1)
  is_card = (idx >= n_calls) & (idx < n_calls + n_played)
  out = jnp.where(is_card, cards[card_pos], call_actions)
  return out.astype(jnp.int16), (n_calls + n_played).astype(jnp.int16)


def _current_player(phase, on_move, declarer) -> jax.Array:
  """The player to move: declarer for dummy, chance or terminal ids.

  Args:
    phase: The game phase.
    on_move: The seat on move.
    declarer: The declarer's seat.

  Returns:
    i32[], the player id.
  """
  on_move = on_move.astype(jnp.int32)
  declarer = declarer.astype(jnp.int32)
  declaring = (phase == PHASE_PLAY) & (
      partnership(on_move) == partnership(declarer)
  )
  player = jnp.where(declaring, declarer, on_move)
  player = jnp.where(phase <= PHASE_DEAL, game_lib.CHANCE_PLAYER_ID, player)
  return jnp.where(
      phase == PHASE_OVER, game_lib.TERMINAL_PLAYER_ID, player
  ).astype(jnp.int32)


def _led_suit(state: GameState) -> jax.Array:
  """The suit led in the current trick, or -1 outside one.

  Args:
    state: The state.

  Returns:
    i32[], the led suit or -1.
  """
  n = state.n_played.astype(jnp.int32)
  t = jnp.minimum(n // NUM_PLAYERS, NUM_TRICKS - 1)
  lead = state.trick_cards[t, 0].astype(jnp.int32)
  in_trick = (state.phase == PHASE_PLAY) & (n % NUM_PLAYERS > 0)
  return jnp.where(in_trick, card_suit(lead), -1)


def _trick_winner_offset(cards, trump) -> jax.Array:
  """Index 0..3 (play order) of the card that wins a complete trick.

  Args:
    cards: i32[4], the trick's cards in play order.
    trump: The trump suit (NO_TRUMP for none).

  Returns:
    i32[], the play-order index of the winning card.
  """
  cards = cards.astype(jnp.int32)
  suits = card_suit(cards)
  value = (
      card_rank(cards)
      + NUM_RANKS * (suits == suits[0])
      + 2 * NUM_RANKS * (suits == trump)
  )
  return jnp.argmax(value).astype(jnp.int32)


def _scored(state: GameState, level, strain, dbl, declarer, tricks):
  """The per-seat scores of a finished deal.

  Args:
    state: The state.
    level: The contract level.
    strain: The strain.
    dbl: The doubling.
    declarer: The declarer.
    tricks: The declarer's tricks.

  Returns:
    i32[4], each seat's score.
  """
  decl_side = partnership(declarer)
  s = score(level, strain, dbl, tricks, state.vul[decl_side])
  seats = jnp.arange(NUM_PLAYERS, dtype=jnp.int32)
  return jnp.where(partnership(seats) == decl_side, s, -s).astype(jnp.int32)


class Bridge(game_lib.Game, interop.PyspielInterop):
  """Contract bridge, auction and card play, duplicate scoring.

  Attributes:
    dealer: The seat that calls first, 0..3.
    dealer_vul: Whether the dealer's side is vulnerable.
    non_dealer_vul: Whether the other side is vulnerable.
    num_tricks_in_observation: The tricks shown in the card-play observation
      tensor.
    random_dealer_vul: Whether a chance node draws the dealer and vulnerability.
  """

  def __init__(
      self,
      use_double_dummy_result: bool = True,
      dealer: int = 0,
      dealer_vul: bool = False,
      non_dealer_vul: bool = False,
      num_tricks_in_observation: int = 2,
      random_dealer_vul: bool = False,
  ):
    """Builds the game; the parameters are pyspiel ``bridge``'s.

    Args:
      use_double_dummy_result: must be False. pyspiel's default, True,
        replaces the play with a double-dummy solver's result, which this
        pure-JAX game does not compute; the default is pyspiel's, so a caller
        that leaves it unset gets the error rather than a different game.
      dealer: the seat that calls first, 0..3.
      dealer_vul: the dealer's side is vulnerable.
      non_dealer_vul: the other side is vulnerable.
      num_tricks_in_observation: tricks (the current one and the ones before
        it) shown in the card-play observation tensor.
      random_dealer_vul: draw the dealer and vulnerability at a chance node,
        uniformly over the 16-board duplicate cycle; ``dealer``,
        ``dealer_vul`` and ``non_dealer_vul`` must keep their defaults.
    """
    if use_double_dummy_result:
      raise ValueError(
          "this bridge plays the cards: pass use_double_dummy_result=false. "
          "OpenSpiel's default (use_double_dummy_result=true) ends the board "
          "after the auction and scores it with a double-dummy solver, which "
          "this game does not compute"
      )
    if not 0 <= int(dealer) < NUM_PLAYERS:
      raise ValueError(f"dealer must be 0..3, got {dealer}")
    if int(num_tricks_in_observation) < 1:
      raise ValueError(
          "num_tricks_in_observation must be >= 1, got "
          f"{num_tricks_in_observation}"
      )
    if random_dealer_vul and (dealer != 0 or dealer_vul or non_dealer_vul):
      raise ValueError(
          "random_dealer_vul draws the dealer and vulnerability; "
          "leave dealer, dealer_vul and non_dealer_vul unset"
      )
    self.dealer = int(dealer)
    self.dealer_vul = bool(dealer_vul)
    self.non_dealer_vul = bool(non_dealer_vul)
    self.num_tricks_in_observation = int(num_tricks_in_observation)
    self.random_dealer_vul = bool(random_dealer_vul)
    vul = [False, False]
    vul[partnership(self.dealer)] = self.dealer_vul
    vul[1 - partnership(self.dealer)] = self.non_dealer_vul
    self._vul = tuple(vul)
    self._obs_size = NUM_OBSERVATION_TYPES + max(
        play_tensor_size(self.num_tricks_in_observation), AUCTION_TENSOR_SIZE
    )
    self._cycle_dealer = np.arange(NUM_BOARDS_IN_CYCLE) % NUM_PLAYERS
    cycle_vul = np.zeros((NUM_BOARDS_IN_CYCLE, 2), bool)
    for b in range(NUM_BOARDS_IN_CYCLE):
      cycle_vul[b] = board_dealer_vul(b)[1]
    self._cycle_vul = cycle_vul

  @property
  def num_actions(self) -> int:
    """The number of distinct player actions.

    OpenSpiel calls it `num_distinct_actions`.
    """
    return NUM_ACTIONS

  @property
  def num_players(self) -> int:
    """The number of players."""
    return NUM_PLAYERS

  @property
  def obs_shape(self) -> tuple[int, ...]:
    """The shape of `observation_tensor`."""
    return (self._obs_size,)

  @property
  def max_chance_outcomes(self) -> int:
    """The number of distinct chance outcomes; 0 for games without chance."""
    return NUM_CARDS

  @property
  def max_game_length(self) -> int:
    """The longest game in actions, chance included."""
    return MAX_GAME_LENGTH

  def new_initial_state(self) -> GameState:
    """Returns the initial state, a chance node.

    Returns:
      The state before the board position (or the deal).
    """
    phase = PHASE_BOARD if self.random_dealer_vul else PHASE_DEAL
    return GameState(
        phase=_i8(phase),
        current_player=jnp.asarray(game_lib.CHANCE_PLAYER_ID, jnp.int32),
        is_terminal=jnp.asarray(False),
        dealer=_i8(self.dealer),
        vul=jnp.asarray(self._vul, jnp.bool_),
        n_dealt=_i8(0),
        deal=jnp.full((NUM_CARDS,), -1, jnp.int8),
        holder=jnp.full((NUM_CARDS,), -1, jnp.int8),
        calls=jnp.full((MAX_AUCTION_LENGTH,), -1, jnp.int8),
        n_calls=jnp.asarray(0, jnp.int16),
        last_bid=_i8(-1),
        dbl=_i8(UNDOUBLED),
        num_passes=_i8(0),
        first_strain_bidder=jnp.full((2, NUM_STRAINS), -1, jnp.int8),
        level=_i8(0),
        strain=_i8(-1),
        declarer=_i8(-1),
        trick_cards=jnp.full((NUM_TRICKS, NUM_PLAYERS), -1, jnp.int8),
        trick_leader=jnp.full((NUM_TRICKS,), -1, jnp.int8),
        n_played=_i8(0),
        on_move=_i8(self.dealer),
        decl_tricks=_i8(0),
        score=jnp.zeros((NUM_PLAYERS,), jnp.int32),
    )

  # -- transitions --------------------------------------------------------

  def _board_step(self, state: GameState, action) -> GameState:
    """Applies the board chance outcome: dealer and vulnerability.

    Args:
      state: The state.
      action: The board outcome.

    Returns:
      The state, now dealing.
    """
    position = jnp.clip(action, 0, NUM_BOARDS_IN_CYCLE - 1)
    dealer = jnp.asarray(self._cycle_dealer, jnp.int32)[position]
    vul = jnp.asarray(self._cycle_vul)[position]
    return state._replace(
        phase=_i8(PHASE_DEAL), dealer=_i8(dealer), vul=vul, on_move=_i8(dealer)
    )

  def _deal_step(self, state: GameState, action) -> GameState:
    """Deals one card to the next seat.

    Args:
      state: The state.
      action: The card to deal.

    Returns:
      The state after the deal step.
    """
    card = jnp.clip(action, 0, NUM_CARDS - 1)
    n = state.n_dealt.astype(jnp.int32)
    seat = _i8(n % NUM_PLAYERS)
    done = n + 1 == NUM_CARDS
    return state._replace(
        phase=_i8(jnp.where(done, PHASE_AUCTION, PHASE_DEAL)),
        n_dealt=_i8(n + 1),
        deal=state.deal.at[card].set(seat),
        holder=state.holder.at[card].set(seat),
        on_move=state.dealer,
    )

  def _auction_step(self, state: GameState, action) -> GameState:
    """Applies one call of the auction.

    Args:
      state: The state.
      action: The call action.

    Returns:
      The state after the call.
    """
    call = action - PASS_ACTION
    seat = state.on_move.astype(jnp.int32)
    is_pass = call == CALL_PASS
    is_double = call == CALL_DOUBLE
    is_redouble = call == CALL_REDOUBLE
    is_bid = call >= FIRST_BID_CALL
    bid = jnp.clip(call - FIRST_BID_CALL, 0, NUM_BIDS - 1)

    num_passes = jnp.where(is_pass, state.num_passes.astype(jnp.int32) + 1, 0)
    dbl = jnp.where(
        is_double,
        DOUBLED,
        jnp.where(
            is_redouble,
            REDOUBLED,
            jnp.where(is_bid, UNDOUBLED, state.dbl.astype(jnp.int32)),
        ),
    )
    bid_strain = bid % NUM_STRAINS
    side = partnership(seat)
    first = state.first_strain_bidder[side, bid_strain].astype(jnp.int32)
    new_first = jnp.where(first < 0, seat, first)
    fsb = jnp.where(
        is_bid,
        state.first_strain_bidder.at[side, bid_strain].set(_i8(new_first)),
        state.first_strain_bidder,
    )
    level = jnp.where(
        is_bid, bid // NUM_STRAINS + 1, state.level.astype(jnp.int32)
    )
    strain = jnp.where(is_bid, bid_strain, state.strain.astype(jnp.int32))
    declarer = jnp.where(is_bid, new_first, state.declarer.astype(jnp.int32))
    last_bid = jnp.where(is_bid, bid, state.last_bid.astype(jnp.int32))
    n_calls = state.n_calls.astype(jnp.int32)
    calls = state.calls.at[jnp.minimum(n_calls, MAX_AUCTION_LENGTH - 1)].set(
        _i8(call)
    )

    passed_out = is_pass & (num_passes == NUM_PLAYERS)
    to_play = is_pass & (num_passes == NUM_PLAYERS - 1) & (level > 0)
    phase = jnp.where(
        passed_out, PHASE_OVER, jnp.where(to_play, PHASE_PLAY, PHASE_AUCTION)
    )
    on_move = jnp.where(
        to_play, (declarer + 1) % NUM_PLAYERS, (seat + 1) % NUM_PLAYERS
    )
    return state._replace(
        phase=_i8(phase),
        calls=calls,
        n_calls=(n_calls + 1).astype(jnp.int16),
        last_bid=_i8(last_bid),
        dbl=_i8(dbl),
        num_passes=_i8(num_passes),
        first_strain_bidder=fsb,
        level=_i8(level),
        strain=_i8(strain),
        declarer=_i8(declarer),
        on_move=_i8(on_move),
    )

  def _play_step(self, state: GameState, action) -> GameState:
    """Plays one card, scoring the deal after the last trick.

    Args:
      state: The state.
      action: The card action.

    Returns:
      The state after the play.
    """
    card = jnp.clip(action, 0, NUM_CARDS - 1)
    seat = state.on_move.astype(jnp.int32)
    n = state.n_played.astype(jnp.int32)
    t = jnp.minimum(n // NUM_PLAYERS, NUM_TRICKS - 1)
    pos = n % NUM_PLAYERS
    holder = state.holder.at[card].set(_i8(-1))
    trick_cards = state.trick_cards.at[t, pos].set(_i8(card))
    trick_leader = jnp.where(
        pos == 0, state.trick_leader.at[t].set(_i8(seat)), state.trick_leader
    )
    complete = pos == NUM_PLAYERS - 1
    leader = trick_leader[t].astype(jnp.int32)
    winner = (
        leader
        + _trick_winner_offset(trick_cards[t], state.strain.astype(jnp.int32))
    ) % NUM_PLAYERS
    declarer = state.declarer.astype(jnp.int32)
    won = complete & (partnership(winner) == partnership(declarer))
    decl_tricks = state.decl_tricks.astype(jnp.int32) + won
    on_move = jnp.where(complete, winner, (seat + 1) % NUM_PLAYERS)
    over = n + 1 == NUM_CARDS
    final = _scored(
        state,
        state.level.astype(jnp.int32),
        state.strain.astype(jnp.int32),
        state.dbl.astype(jnp.int32),
        declarer,
        decl_tricks,
    )
    return state._replace(
        phase=_i8(jnp.where(over, PHASE_OVER, PHASE_PLAY)),
        holder=holder,
        trick_cards=trick_cards,
        trick_leader=trick_leader,
        n_played=_i8(n + 1),
        on_move=_i8(on_move),
        decl_tricks=_i8(decl_tricks),
        score=jnp.where(over, final, state.score),
    )

  def apply_action(self, state: GameState, action: jax.Array) -> GameState:
    """Applies a chance outcome or the current player's action.

    The action must be legal (``legal_actions_mask``, or a chance outcome
    with nonzero probability); a terminal state is returned unchanged.

    Args:
      state: The state.
      action: i32[], the action.

    Returns:
      The next state.
    """
    action = jnp.asarray(action, jnp.int32)
    phase = state.phase
    candidates = (
        self._board_step(state, action),
        self._deal_step(state, action),
        self._auction_step(state, action),
        self._play_step(state, action),
        state,
    )

    def pick(*leaves):
      """Selects the leaf of the state's phase.

      Args:
        *leaves: One leaf per phase, in phase order.

      Returns:
        The leaf of the current phase.
      """
      out = leaves[-1]
      for p in range(PHASE_PLAY, PHASE_BOARD - 1, -1):
        out = jnp.where(phase == p, leaves[p], out)
      return out

    nxt = jax.tree.map(pick, *candidates)
    return nxt._replace(
        current_player=_current_player(nxt.phase, nxt.on_move, nxt.declarer),
        is_terminal=nxt.phase == PHASE_OVER,
    )

  # -- queries ------------------------------------------------------------

  def legal_actions_mask(self, state: GameState) -> jax.Array:
    """Returns the legal actions of the current player.

    Args:
      state: The state.

    Returns:
      bool[90]; all False at chance and terminal states.
    """
    on_move = state.on_move.astype(jnp.int32)
    in_auction = state.phase == PHASE_AUCTION
    has_bid = state.level > 0
    opposing = partnership(state.declarer.astype(jnp.int32)) != partnership(
        on_move
    )
    can_double = has_bid & opposing & (state.dbl == UNDOUBLED)
    can_redouble = has_bid & ~opposing & (state.dbl == DOUBLED)
    bids = jnp.arange(NUM_BIDS, dtype=jnp.int32) > state.last_bid.astype(
        jnp.int32
    )
    calls = (
        jnp.concatenate([
            jnp.ones((1,), jnp.bool_),
            can_double[None],
            can_redouble[None],
            bids,
        ])
        & in_auction
    )
    in_play = state.phase == PHASE_PLAY
    hand = state.holder.astype(jnp.int32) == on_move
    suits = jnp.arange(NUM_CARDS, dtype=jnp.int32) % NUM_SUITS
    follow = hand & (suits == _led_suit(state))
    cards = jnp.where(follow.any(), follow, hand) & in_play
    return jnp.concatenate([cards, calls])

  def chance_outcomes(self, state: GameState) -> jax.Array:
    """Returns the chance distribution.

    It is uniform over the undealt cards, or over the 16 cycle positions.

    Args:
      state: The state.

    Returns:
      f32[max_chance_outcomes].
    """
    undealt = (state.deal < 0).astype(jnp.float32)
    deal_probs = undealt / jnp.maximum(undealt.sum(), 1.0)
    board_probs = jnp.where(
        jnp.arange(NUM_CARDS) < NUM_BOARDS_IN_CYCLE,
        jnp.float32(1.0 / NUM_BOARDS_IN_CYCLE),
        jnp.float32(0.0),
    )
    return jnp.where(
        state.phase == PHASE_BOARD, board_probs, deal_probs
    ).astype(jnp.float32)

  def current_player(self, state: GameState) -> jax.Array:
    """Returns the player to move.

    Args:
      state: The current state.

    Returns:
      i32[]: the player to move, or `CHANCE_PLAYER_ID` at a chance node. The
      value at a terminal state is game-specific.
    """
    return state.current_player

  def is_terminal(self, state: GameState) -> jax.Array:
    """Returns whether the game is over.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the game is over.
    """
    return state.is_terminal

  def is_chance_node(self, state: GameState) -> jax.Array:
    """Returns whether the next action is a chance outcome.

    Args:
      state: The current state.

    Returns:
      bool[]: True if the next action is a chance outcome.
    """
    return (
        state.current_player == game_lib.CHANCE_PLAYER_ID
    ) & ~state.is_terminal

  def information_state_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns the observation tensor.

    It already holds the whole auction and every trick a seat has seen, as
    pyspiel's information state does.

    Args:
      state: The state.
      observing_player: i32[], the observing player.

    Returns:
      f32[obs_size].
    """
    return self.observation_tensor(state, observing_player)

  def returns(self, state: GameState) -> jax.Array:
    """Returns the duplicate points per seat.

    Args:
      state: The state.

    Returns:
      f32[4]; zeros until the board is scored.
    """
    return state.score.astype(jnp.float32)

  def observation_tensor(
      self, state: GameState, observing_player: jax.Array
  ) -> jax.Array:
    """Returns pyspiel's bridge observation tensor layout.

    The auction view places each call by its bidder relative to the dealer
    (module doc), so it equals OpenSpiel's tensor only when North (seat 0)
    deals.

    Args:
      state: The state.
      observing_player: i32[], the observing player.

    Returns:
      f32[obs_size].
    """
    player = jnp.asarray(observing_player, jnp.int32)
    play = self._play_observation(state, player)
    auction = self._auction_observation(state, player)
    out = jnp.where(state.n_played > 0, play, auction)
    return jnp.where(state.phase <= PHASE_DEAL, jnp.zeros_like(out), out)

  def _auction_observation(self, state: GameState, player) -> jax.Array:
    """The observation of `player` during the auction.

    Args:
      state: The state.
      player: The observing player.

    Returns:
      f32[obs_size], the observation.
    """
    size = self._obs_size
    sink = size  # writes that must not land go to a dropped extra slot
    out = jnp.zeros((size + 1,), jnp.float32)
    out = out.at[jnp.where(state.phase == PHASE_PLAY, 1, 0)].set(1.0)
    side = partnership(player)
    base = NUM_OBSERVATION_TYPES
    out = out.at[base + state.vul[side].astype(jnp.int32)].set(1.0)
    out = out.at[base + 2 + state.vul[1 - side].astype(jnp.int32)].set(1.0)
    base += 4
    calls = state.calls.astype(jnp.int32)
    k = jnp.arange(MAX_AUCTION_LENGTH, dtype=jnp.int32)
    valid = k < state.n_calls.astype(jnp.int32)
    # Call k at its bidder's seat, (dealer + k) % 4, relative to the player.
    rel = (state.dealer.astype(jnp.int32) + k - player) % NUM_PLAYERS
    bid_call = jnp.where(valid & (calls >= FIRST_BID_CALL), calls, 0)
    last_incl = jax.lax.cummax(bid_call)
    last_before = jnp.concatenate([jnp.zeros((1,), jnp.int32), last_incl[:-1]])
    is_pass = calls == CALL_PASS
    is_double = calls == CALL_DOUBLE
    is_redouble = calls == CALL_REDOUBLE
    is_bid = calls >= FIRST_BID_CALL
    block = base + NUM_PLAYERS
    idx = jnp.where(
        is_pass & (last_before == 0),
        base + rel,
        jnp.where(
            is_double,
            block
            + (last_before - FIRST_BID_CALL) * 3 * NUM_PLAYERS
            + NUM_PLAYERS
            + rel,
            jnp.where(
                is_redouble,
                block
                + (last_before - FIRST_BID_CALL) * 3 * NUM_PLAYERS
                + 2 * NUM_PLAYERS
                + rel,
                jnp.where(
                    is_bid,
                    block + (calls - FIRST_BID_CALL) * 3 * NUM_PLAYERS + rel,
                    sink,
                ),
            ),
        ),
    )
    out = out.at[jnp.where(valid, idx, sink)].set(1.0)
    base += NUM_PLAYERS * (1 + 3 * NUM_BIDS)
    hand = (state.holder.astype(jnp.int32) == player).astype(jnp.float32)
    out = out.at[base : base + NUM_CARDS].set(hand)
    return out[:size]

  def _play_observation(self, state: GameState, player) -> jax.Array:
    """The observation of `player` during the play.

    Args:
      state: The state.
      player: The observing player.

    Returns:
      f32[obs_size], the observation.
    """
    size = self._obs_size
    sink = size
    out = jnp.zeros((size + 1,), jnp.float32)
    declarer = state.declarer.astype(jnp.int32)
    defending = (partnership(player) != partnership(declarer)).astype(jnp.int32)
    out = out.at[jnp.where(state.phase == PHASE_PLAY, 2 + defending, sink)].set(
        1.0
    )
    base = NUM_OBSERVATION_TYPES
    out = out.at[base + jnp.maximum(state.level.astype(jnp.int32) - 1, 0)].set(
        1.0
    )
    base += NUM_LEVELS
    out = out.at[base + jnp.maximum(state.strain.astype(jnp.int32), 0)].set(1.0)
    base += NUM_STRAINS
    out = out.at[base + state.dbl.astype(jnp.int32)].set(1.0)
    base += NUM_OTHER_CALLS
    out = out.at[base + (declarer - player) % NUM_PLAYERS].set(1.0)
    base += NUM_PLAYERS
    decl_vul = state.vul[partnership(declarer)].astype(jnp.int32)
    out = out.at[base + decl_vul].set(1.0)
    base += 2
    holder = state.holder.astype(jnp.int32)
    out = out.at[base : base + NUM_CARDS].set(
        (holder == player).astype(jnp.float32)
    )
    base += NUM_CARDS
    out = out.at[base : base + NUM_CARDS].set(
        (holder == dummy(state)).astype(jnp.float32)
    )
    base += NUM_CARDS

    n = state.n_played.astype(jnp.int32)
    current = n // NUM_PLAYERS
    in_trick = n % NUM_PLAYERS
    trick_block = NUM_PLAYERS * NUM_CARDS

    def write_trick(out, t, count, block_base):
      """Writes one trick's first `count` cards, seats relative to `player`.

      Args:
        out: The observation being built.
        t: The trick index; out of range writes nothing.
        count: The number of cards played in the trick.
        block_base: The offset of the trick's block.

      Returns:
        `out` with the cards set.
      """
      t_safe = jnp.clip(t, 0, NUM_TRICKS - 1)
      leader = state.trick_leader[t_safe].astype(jnp.int32)
      cards = state.trick_cards[t_safe].astype(jnp.int32)
      i = jnp.arange(NUM_PLAYERS, dtype=jnp.int32)
      rel = (i + leader - player) % NUM_PLAYERS
      idx = jnp.where(
          (i < count) & (t >= 0) & (t < NUM_TRICKS),
          block_base + rel * NUM_CARDS + cards,
          sink,
      )
      return out.at[idx].set(1.0)

    live = state.phase != PHASE_OVER
    out = write_trick(out, current, jnp.where(live, in_trick, 0), base)
    base += trick_block
    for k in range(self.num_tricks_in_observation - 1):
      out = write_trick(out, current - 1 - k, NUM_PLAYERS, base)
      base += trick_block
    decl_tricks = state.decl_tricks.astype(jnp.int32)
    defence_tricks = current - decl_tricks
    out = out.at[
        jnp.where(decl_tricks < NUM_TRICKS, base + decl_tricks, sink)
    ].set(1.0)
    base += NUM_TRICKS
    out = out.at[
        jnp.where(defence_tricks < NUM_TRICKS, base + defence_tricks, sink)
    ].set(1.0)
    return out[:size]

  def observation_from_os(self, os_state: Any, player: int) -> jax.Array:
    """Returns the JAX game's observation tensor of `player` at `os_state`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The observation tensor the JAX game gives `player` in the same position.

    Raises:
      NotImplementedError: if the game has no pyspiel conversion.
    """
    return jnp.asarray(os_state.observation_tensor(player), jnp.float32)

  def information_state_from_os(self, os_state: Any, player: int) -> jax.Array:
    """Returns the JAX game's information-state tensor of `player`.

    Games whose information state is their observation inherit this
    default, which returns `observation_from_os`.

    Args:
      os_state: The pyspiel state.
      player: The observing player.

    Returns:
      The information-state tensor the JAX game gives `player` in the same
      position.
    """
    return jnp.asarray(os_state.information_state_tensor(player), jnp.float32)
