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

"""N-player Kuhn poker (OpenSpiel's kuhn_poker(players=N)), self-contained JAX.

The smallest imperfect-information game with more than two players, so the
league, the evaluator and the meta-solvers have an exact N-player test game:
its whole tree is a few thousand nodes at N=3 and N=4, and core.game_tree
computes expected returns, per-seat best responses and NashConv exactly.

Rules (OpenSpiel kuhn_poker.cc, followed action for action):

  N seats, a deck of N+1 cards 0..N; each seat antes 1. Chance deals seat
  0's card, then seat 1's, ... (each uniform over the undealt cards; the
  outcome id is the card). Seats act in turn 0, 1, ..., N-1, each pass (0) or
  bet (1). If nobody bets in the first N actions the hand ends and the
  highest card wins the pot. Once seat f is the first to bet, every later
  seat f+1..N-1 and then every earlier seat 0..f-1 acts once more (a bet
  calls, a pass folds); the hand ends after N + f actions. The highest card
  among the seats that bet wins. A seat that bet has 2 chips in, one that did
  not has 1; the winner's return is the pot minus its chips, every other
  seat's is minus its chips (zero-sum).

The game is registered as "kuhn_nplayer" with the option `players` (2..10,
pyspiel's range); it is not "kuhn_poker", so artifacts of the two-player
games never mix with it, and it is not served. At players=2 its returns,
actors and legal sets equal envs.kuhn_jax's (a test checks it); its
InfoState has a bets array of length 2N-1, so it is not the two-player
KuhnInfo. This module imports nothing beyond jax and the package's core: it
is the module rules_sha hashes (`game` is the env itself).

Sides: none declared, so every seat is its own side (Sides.individual(N,
constant_sum=True), the default for a zero-sum env).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import chance

PASS, BET = 0, 1
MIN_PLAYERS, MAX_PLAYERS = 2, 10  # pyspiel kuhn_poker's range


class KuhnNInfo(NamedTuple):
  """What one seat knows: its seat, its own card and the public betting."""

  seat: jax.Array  # i32
  card: jax.Array  # i32 own card 0..N, -1 before it is dealt
  # i32[2N-1]; action k (by seat k % N): 0 pass, 1 bet, -1 not played
  bets: jax.Array
  num_bets: jax.Array  # i32 actions played


class KuhnNState(NamedTuple):
  """The raw state: the deal and the betting.

  Attributes:
    cards: i32[N]; seat i's card, -1 until dealt (dealt in seat order).
    bets: i32[2N-1]; the actions played, -1 after the last.
    num_bets: i32; the actions played.
  """

  # i32[N]; seat i's card, -1 until dealt (dealt in seat order)
  cards: jax.Array
  bets: jax.Array  # i32[2N-1]
  num_bets: jax.Array  # i32


def max_actions(players: int) -> int:
  """The longest hand: N-1 passes, a bet by seat N-1, N-1 responses.

  Args:
    players: N, the number of seats.

  Returns:
    2N - 1.
  """
  return 2 * players - 1


class KuhnNPlayerEnv:
  """types.Env for N-player Kuhn poker (module doc).

  Every method is on one state.

  Attributes:
    players: N, the number of players.
    num_players: N.
    num_cards: N + 1, the deck size.
    max_len: the longest betting sequence (max_actions(N)).
    max_chance_outcomes: the deck size: a deal draws one of num_cards cards.
    game: the env itself: the rules live in this module, which
      registry.rules_sha hashes.
  """

  name = "kuhn_nplayer"
  num_actions = 2
  zero_sum = True

  def __init__(self, players: int = 3):
    if isinstance(players, bool) or not isinstance(players, int):
      raise TypeError(f"kuhn_nplayer players must be an int, got {players!r}")
    if not MIN_PLAYERS <= players <= MAX_PLAYERS:
      raise ValueError(
          f"kuhn_nplayer players must be in [{MIN_PLAYERS}, {MAX_PLAYERS}], "
          f"got {players}"
      )
    self.players = int(players)
    self.num_players = self.players
    self.num_cards = self.players + 1
    self.max_len = max_actions(self.players)
    self.max_chance_outcomes = self.num_cards
    # The rules live in this module, which registry.rules_sha hashes.
    self.game = self

  # -- raw layer ---------------------------------------------------------------
  def root(self) -> KuhnNState:
    """The state before the deal.

    Returns:
      The root state.
    """
    return KuhnNState(
        cards=jnp.full((self.players,), -1, jnp.int32),
        bets=jnp.full((self.max_len,), -1, jnp.int32),
        num_bets=jnp.int32(0),
    )

  def is_chance(self, s: KuhnNState) -> jax.Array:
    """Whether a card is still to be dealt.

    Args:
      s: The state.

    Returns:
      bool.
    """
    return s.cards[-1] < 0

  def apply_chance(self, s: KuhnNState, outcome) -> KuhnNState:
    """Deals card `outcome` to the next seat without one.

    Args:
      s: The state.
      outcome: The card.

    Returns:
      The next state (s itself away from a chance node).
    """
    o = jnp.asarray(outcome, jnp.int32)
    seat = jnp.sum((s.cards >= 0).astype(jnp.int32))
    dealt = s.cards.at[jnp.minimum(seat, self.players - 1)].set(o)
    return s._replace(cards=jnp.where(self.is_chance(s), dealt, s.cards))

  def apply(self, s: KuhnNState, a) -> KuhnNState:
    """The seat to act plays `a` (0 pass, 1 bet).

    Args:
      s: The state.
      a: The action.

    Returns:
      The next state (s itself at a terminal or chance node).
    """
    a = jnp.asarray(a, jnp.int32)
    live = ~self.terminal(s) & ~self.is_chance(s)
    slot = jnp.minimum(s.num_bets, self.max_len - 1)
    bets = jnp.where(live, s.bets.at[slot].set(a), s.bets)
    return s._replace(
        bets=bets, num_bets=jnp.where(live, s.num_bets + 1, s.num_bets)
    )

  # -- resolved layer ----------------------------------------------------------
  def init(self, deal_key) -> KuhnNState:
    """The first decision state, the deal sampled with `deal_key`.

    Args:
      deal_key: The deal's PRNG key.

    Returns:
      The state.
    """
    return chance.resolve(self, self.root(), deal_key)[0]

  def step(self, s, a, chance_key) -> KuhnNState:
    """Plays `a`, then resolves any chance with `chance_key`.

    Args:
      s: The state.
      a: The action.
      chance_key: The PRNG key of the chance after `a`.

    Returns:
      The next decision or terminal state.
    """
    return chance.resolve(self, self.apply(s, a), chance_key)[0]

  # -- rules -------------------------------------------------------------------
  def first_bettor(self, s: KuhnNState) -> jax.Array:
    """The seat of the first bet among the first N actions, or -1.

    Args:
      s: The state.

    Returns:
      i32.
    """
    first = s.bets[: self.players] == BET
    return jnp.where(jnp.any(first), jnp.argmax(first), -1).astype(jnp.int32)

  def did_bet(self, s: KuhnNState) -> jax.Array:
    """Each seat has put a second chip in (OpenSpiel's DidBet).

    Args:
      s: The state.

    Returns:
      bool[N].
    """
    n = self.players
    fb = self.first_bettor(s)
    seats = jnp.arange(n, dtype=jnp.int32)
    ext = jnp.concatenate([s.bets, jnp.full((1,), -1, jnp.int32)])  # length 2N
    first_round = s.bets[seats] == BET  # seat p's action in the first N
    second_round = ext[n + seats] == BET  # seat p < fb's response
    bet = (
        (seats == fb)
        | ((seats > fb) & first_round)
        | ((seats < fb) & second_round)
    )
    return (fb >= 0) & bet

  def terminal(self, s: KuhnNState) -> jax.Array:
    """Whether the hand is over.

    Args:
      s: The state.

    Returns:
      bool.
    """
    fb = self.first_bettor(s)
    done = jnp.where(
        fb < 0, s.num_bets >= self.players, s.num_bets >= self.players + fb
    )
    return ~self.is_chance(s) & done

  def actor(self, s: KuhnNState) -> jax.Array:
    """The seat to act, -1 at a terminal state.

    Args:
      s: The state.

    Returns:
      i32.
    """
    seat = (s.num_bets % self.players).astype(jnp.int32)
    return jnp.where(self.terminal(s), jnp.int32(-1), seat)

  def legal_mask(self, s: KuhnNState) -> jax.Array:
    """Both actions while the hand is played, none otherwise.

    Args:
      s: The state.

    Returns:
      bool[2].
    """
    playing = ~self.terminal(s) & ~self.is_chance(s)
    return jnp.broadcast_to(playing, (self.num_actions,))

  def returns(self, s: KuhnNState) -> jax.Array:
    """Each seat's chips won or lost at a terminal state, zeros before.

    Args:
      s: The state.

    Returns:
      f32[N].
    """
    bet = self.did_bet(s)
    contenders = jnp.where(self.first_bettor(s) >= 0, bet, True)
    winner = jnp.argmax(jnp.where(contenders, s.cards, -1))
    chips = 1.0 + bet.astype(jnp.float32)
    pot = jnp.sum(chips)
    r = jnp.where(jnp.arange(self.players) == winner, pot - chips, -chips)
    return jnp.where(self.terminal(s), r, 0.0).astype(jnp.float32)

  def chance_probs(self, s: KuhnNState) -> jax.Array:
    """The deal's distribution: uniform over the cards not dealt yet.

    Args:
      s: The state.

    Returns:
      f32[N+1].
    """
    cards = jnp.arange(self.num_cards, dtype=jnp.int32)
    dealt = jnp.any(cards[:, None] == s.cards[None, :], axis=1)
    free = (~dealt).astype(jnp.float32)
    return free / jnp.maximum(jnp.sum(free), 1.0)

  # -- what a seat sees --------------------------------------------------------
  def info_state(self, s: KuhnNState, seat) -> KuhnNInfo:
    """What `seat` knows: its seat, its own card and the public betting.

    Args:
      s: The state.
      seat: The seat.

    Returns:
      The KuhnNInfo.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return KuhnNInfo(
        seat=seat,
        card=jnp.asarray(s.cards[seat], jnp.int32),
        bets=jnp.asarray(s.bets, jnp.int32),
        num_bets=jnp.asarray(s.num_bets, jnp.int32),
    )

  def labels(self, s: KuhnNState, seat):
    """The hidden-information labels of `seat`.

    The other seats' cards, seat+1, seat+2, ... mod N.

    Args:
      s: The state.
      seat: The seat.

    Returns:
      {"opp_cards": i32[N-1]}.
    """
    seat = jnp.asarray(seat, jnp.int32)
    others = (
        seat + jnp.arange(1, self.players, dtype=jnp.int32)
    ) % self.players
    return {"opp_cards": s.cards[others].astype(jnp.int32)}

  def visible_action(self, s, a, seat) -> jax.Array:
    """Every seat sees every action.

    Args:
      s: Unused.
      a: The action.
      seat: Unused.

    Returns:
      i32 `a`.
    """
    del s, seat
    return jnp.asarray(a, jnp.int32)

  def world_from_info(self, info) -> Optional[KuhnNState]:
    """No world is built from an info state here.

    Args:
      info: The KuhnNInfo.

    Returns:
      None.
    """
    del info
    return None

  # -- helpers for tests and evaluators ----------------------------------------
  def deal(self, *cards: int) -> KuhnNState:
    """A state with seat i dealt cards[i] (for known-answer tests).

    Args:
      *cards: One distinct card in [0, N] per seat.

    Returns:
      The first decision state.

    Raises:
      ValueError: not N distinct cards in range.
    """
    if (
        len(cards) != self.players
        or len(set(cards)) != self.players
        or not all(0 <= c < self.num_cards for c in cards)
    ):
      raise ValueError(
          f"kuhn_nplayer({self.players}) deals {self.players} distinct cards "
          f"in [0, {self.num_cards}), got {cards}"
      )
    s = self.root()
    for c in cards:
      s = self.apply_chance(s, c)
    return s


def hide_other_cards(
    env: KuhnNPlayerEnv, s: KuhnNState, seat, key
) -> KuhnNState:
  """Privacy hook for check_env: the cards seat cannot see, permuted.

  A random permutation of the cards seat cannot see (the other seats' and the
  undealt one) over those places. Resolved states only (every card dealt).

  Args:
    env: The env.
    s: The state.
    seat: The seat whose view is kept.
    key: The PRNG key of the permutation.

  Returns:
    The state with the hidden cards permuted.
  """
  n = env.players
  seat = jnp.asarray(seat, jnp.int32)
  undealt = jnp.int32(n * (n + 1) // 2) - jnp.sum(s.cards)
  # [N+1]; place N is the undealt card
  places = jnp.concatenate([s.cards, undealt[None]])
  idx = jnp.arange(n + 1, dtype=jnp.int32)
  # every place but seat's, in order
  others = jnp.argsort(idx == seat, stable=True)[:n]
  perm = jax.random.permutation(key, n)
  places = places.at[others].set(places[others[perm]])
  return s._replace(cards=places[:n].astype(jnp.int32))
