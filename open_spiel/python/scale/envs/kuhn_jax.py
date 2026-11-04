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

"""Kuhn poker as a self-contained implementation that needs no game package.

With neither the JAX games nor pyspiel installed, training and evaluation
still run on this game (the `train` entry point with
`--config=mmd_kuhn_poker_jax`).
The rules, the chance layout and the KuhnInfo InfoState are those of the
adapter of the JAX games package's engine, envs.kuhn (game "kuhn_poker"), so
the same encoders read it; with that package installed, either game works.
It is registered as its own game, "kuhn_poker_jax", so artifacts of the two
never mix.

  3 cards (J=0, Q=1, K=2); each seat antes 1. Chance deals seat 0's card
  (3 outcomes, uniform), then seat 1's (the 2 remaining, uniform). Actions:
  pass (0), bet (1). Histories: pp showdown +-1; bp seat 1 folds (+1 seat
  0); bb showdown +-2; pbp seat 0 folds (+1 seat 1); pbb showdown +-2.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import chance

NUM_CARDS = 3
MAX_BETS = 3
# Betting-history prefixes at decision nodes, in index order.
HISTORIES = ("", "p", "b", "pb")
NUM_INFO_STATES = NUM_CARDS * len(HISTORIES)  # 12


class KuhnInfo(NamedTuple):
  """What one seat knows: its seat, its own card, the public betting history.

  Attributes:
    seat: i32.
    card: i32 own card 0..2 (J, Q, K).
    bets: i32[3]; 0 pass, 1 bet, -1 not yet played.
    num_bets: i32.
  """

  seat: jax.Array
  card: jax.Array
  bets: jax.Array
  num_bets: jax.Array


def info_index(info: KuhnInfo) -> jax.Array:
  """Index in [0, 12) of a decision-node KuhnInfo: card * 4 + history id.

  Args:
    info: The KuhnInfo of a decision node.

  Returns:
    The i32 index.
  """
  b0 = info.bets[..., 0]
  hist = jnp.where(
      info.num_bets == 0,
      0,
      jnp.where(info.num_bets == 1, jnp.where(b0 == 0, 1, 2), 3),
  )
  return (info.card * len(HISTORIES) + hist).astype(jnp.int32)


class KuhnState(NamedTuple):
  """The full state: both cards and the betting history.

  Attributes:
    p0_card: i32, -1 until dealt.
    p1_card: i32, -1 until dealt.
    bets: i32[3], -1 unused.
    num_bets: i32.
  """

  p0_card: jax.Array
  p1_card: jax.Array
  bets: jax.Array
  num_bets: jax.Array


def _terminal(bets, num_bets):
  """Whether the betting history (bets, num_bets) has ended the hand.

  Args:
    bets: The betting actions.
    num_bets: The number of bets.

  Returns:
    bool[], whether the hand has ended.
  """
  is_pb = (bets[0] == 0) & (bets[1] == 1)
  return ((num_bets == 2) & ~is_pb) | (num_bets >= 3)


class KuhnJaxEnv:
  """types.Env for Kuhn poker (module doc); every method on one state.

  Attributes:
    game: The env itself (it is its own game).
  """

  name = "kuhn_poker_jax"
  num_players = 2
  num_actions = 2
  max_len = MAX_BETS
  max_chance_outcomes = NUM_CARDS
  zero_sum = True

  def __init__(self):
    # The rules live in this module, which registry.rules_sha hashes.
    self.game = self

  # -- raw layer ---------------------------------------------------------------
  def root(self) -> KuhnState:
    """The state before the deal.

    Returns:
      The root state.
    """
    return KuhnState(
        p0_card=jnp.int32(-1),
        p1_card=jnp.int32(-1),
        bets=jnp.full((MAX_BETS,), -1, jnp.int32),
        num_bets=jnp.int32(0),
    )

  def is_chance(self, s: KuhnState) -> jax.Array:
    """Whether a card is still to be dealt.

    Args:
      s: The state.

    Returns:
      bool.
    """
    return (s.p0_card < 0) | (s.p1_card < 0)

  def apply_chance(self, s: KuhnState, outcome) -> KuhnState:
    """Deals `outcome` to the next seat without a card.

    Args:
      s: The state.
      outcome: The card dealt.

    Returns:
      The next state.
    """
    o = jnp.asarray(outcome, jnp.int32)
    p0_first = s.p0_card < 0
    return s._replace(
        p0_card=jnp.where(p0_first, o, s.p0_card),
        p1_card=jnp.where(p0_first, s.p1_card, o),
    )

  def apply(self, s: KuhnState, a) -> KuhnState:
    """Plays action `a` (a no-op at a chance or terminal state).

    Args:
      s: The state.
      a: The action: pass (0) or bet (1).

    Returns:
      The next state.
    """
    a = jnp.asarray(a, jnp.int32)
    live = ~self.terminal(s) & ~self.is_chance(s)
    bets = jnp.where(
        live, s.bets.at[jnp.minimum(s.num_bets, MAX_BETS - 1)].set(a), s.bets
    )
    return s._replace(
        bets=bets, num_bets=jnp.where(live, s.num_bets + 1, s.num_bets)
    )

  # -- resolved layer ----------------------------------------------------------
  def init(self, deal_key) -> KuhnState:
    """The first decision state, the deal sampled with `deal_key`.

    Args:
      deal_key: The deal's PRNG key.

    Returns:
      The state.
    """
    return chance.resolve(self, self.root(), deal_key)[0]

  def step(self, s, a, chance_key) -> KuhnState:
    """Plays `a`, then resolves any chance with `chance_key`.

    Args:
      s: The state.
      a: The action.
      chance_key: The PRNG key of the chance that follows.

    Returns:
      The next state.
    """
    return chance.resolve(self, self.apply(s, a), chance_key)[0]

  # -- queries -----------------------------------------------------------------
  def chance_probs(self, s: KuhnState) -> jax.Array:
    """The probability of each card being dealt next.

    Args:
      s: A chance state.

    Returns:
      f32[3].
    """
    cards = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    p1 = jnp.where(cards == s.p0_card, jnp.float32(0.0), jnp.float32(0.5))
    return jnp.where(
        s.p0_card < 0, jnp.full((NUM_CARDS,), 1.0 / 3.0, jnp.float32), p1
    )

  def terminal(self, s: KuhnState) -> jax.Array:
    """Whether the hand is over.

    Args:
      s: The state.

    Returns:
      bool.
    """
    return ~self.is_chance(s) & _terminal(s.bets, s.num_bets)

  def actor(self, s: KuhnState) -> jax.Array:
    """The seat to act.

    Args:
      s: The state.

    Returns:
      i32 seat, -1 at a terminal state.
    """
    is_pb = (s.bets[0] == 0) & (s.bets[1] == 1)
    seat = jnp.where(
        (s.num_bets == 0) | ((s.num_bets == 2) & is_pb), 0, 1
    ).astype(jnp.int32)
    return jnp.where(self.terminal(s), jnp.int32(-1), seat)

  def legal_mask(self, s: KuhnState) -> jax.Array:
    """The legal actions: both while the hand is being played, else none.

    Args:
      s: The state.

    Returns:
      bool[2].
    """
    playing = ~self.terminal(s) & ~self.is_chance(s)
    return jnp.broadcast_to(playing, (2,))

  def returns(self, s: KuhnState) -> jax.Array:
    """Each seat's return (0 before the end).

    Args:
      s: The state.

    Returns:
      f32[2].
    """
    b, n = s.bets, s.num_bets
    pp = (n == 2) & (b[0] == 0) & (b[1] == 0)
    bb = (n == 2) & (b[0] == 1) & (b[1] == 1)
    bp = (n == 2) & (b[0] == 1) & (b[1] == 0)
    pbp = (n == 3) & (b[0] == 0) & (b[1] == 1) & (b[2] == 0)
    pbb = (n == 3) & (b[0] == 0) & (b[1] == 1) & (b[2] == 1)
    win = jnp.where(s.p0_card > s.p1_card, 1.0, -1.0)
    show = win * jnp.where(pp, 1.0, 2.0)
    fold = jnp.where(bp, 1.0, jnp.where(pbp, -1.0, 0.0))
    r0 = jnp.where(pp | bb | pbb, show, fold)
    r0 = jnp.where(self.terminal(s), r0, 0.0)
    return jnp.stack([r0, -r0]).astype(jnp.float32)

  def info_state(self, s: KuhnState, seat) -> KuhnInfo:
    """What `seat` knows.

    Args:
      s: The state.
      seat: The seat.

    Returns:
      The KuhnInfo.
    """
    seat = jnp.asarray(seat, jnp.int32)
    card = jnp.where(seat == 0, s.p0_card, s.p1_card)
    return KuhnInfo(
        seat=seat,
        card=jnp.asarray(card, jnp.int32),
        bets=jnp.asarray(s.bets, jnp.int32),
        num_bets=jnp.asarray(s.num_bets, jnp.int32),
    )

  def labels(self, s: KuhnState, seat):
    """The hidden-information labels of `seat`.

    Args:
      s: The state.
      seat: The seat.

    Returns:
      {"opp_card": i32}.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return {
        "opp_card": jnp.where(seat == 0, s.p1_card, s.p0_card).astype(jnp.int32)
    }

  def visible_action(self, s, a, seat) -> jax.Array:
    """The action as `seat` sees it (every action is public).

    Args:
      s: The state.
      a: The action.
      seat: The observing seat.

    Returns:
      i32 action.
    """
    del s, seat
    return jnp.asarray(a, jnp.int32)

  def world_from_info(self, info) -> Optional[KuhnState]:
    """No world is built from an info state here.

    Args:
      info: The KuhnInfo.

    Returns:
      None.
    """
    del info
    return None

  def deal(self, p0_card: int, p1_card: int) -> KuhnState:
    """A state with the given cards dealt (for known-answer tests).

    Args:
      p0_card: Seat 0's card.
      p1_card: Seat 1's card.

    Returns:
      The state.
    """
    assert p0_card != p1_card and 0 <= p0_card < 3 and 0 <= p1_card < 3
    return self.apply_chance(self.apply_chance(self.root(), p0_card), p1_card)


def hide_opponent_card(env, s, seat, key):
  """Privacy hook for check_env: swaps seat's opponent card for the undealt one.

  Works on any state with p0_card / p1_card: this env's and the JAX-engine
  adapter's.

  Args:
    env: The env (unused).
    s: The state.
    seat: The seat whose opponent's card is swapped.
    key: A PRNG key (unused).

  Returns:
    The state with the opponent's card replaced.
  """
  del env, key
  seat = jnp.asarray(seat, jnp.int32)
  undealt = 3 - s.p0_card - s.p1_card
  p0 = jnp.where(seat == 1, undealt, s.p0_card)
  p1 = jnp.where(seat == 0, undealt, s.p1_card)
  return s._replace(p0_card=p0.astype(jnp.int32), p1_card=p1.astype(jnp.int32))
