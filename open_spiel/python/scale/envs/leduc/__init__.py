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

"""Leduc poker Env over the JAX leduc_poker game.

Its InfoState is the LeducInfo PyTree.

Cards are OpenSpiel ids 0..5, rank = id // 2 (J J Q Q K K).
Actions: fold 0, call/check 1, raise 2. Chance: the two private cards, then the
public card between rounds (a mid-hand chance node), both through the raw layer
(`root`/`apply_chance`) and `core.chance.resolve`.

The JAX leduc game has no current_player() method (the adapter reads the state
field) and reports -4 at terminal; the adapter normalizes actor() to -1.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import game as game_base
from open_spiel.python.jax.games import leduc_poker
from open_spiel.python.scale.envs import jax_game_env

NUM_CARDS = 6
MAX_BETS_PER_ROUND = 4  # 3 * players - 2
MAX_LEN = 2 * MAX_BETS_PER_ROUND  # player decisions in one hand


class LeducInfo(NamedTuple):
  """What one seat knows: its seat and card, and everything public.

  Attributes:
    seat: i32.
    card: i32 own private card 0..5.
    public_card: i32 0..5, or -1 before round 2.
    round: i32 1 or 2.
    round1: i32[4] round-1 actions (0 fold, 1 call, 2 raise), -1 pad.
    round2: i32[4] round-2 actions, -1 pad.
    ante: i32[2] chips each seat has put in (public).
  """

  seat: jax.Array
  card: jax.Array
  public_card: jax.Array
  round: jax.Array
  round1: jax.Array
  round2: jax.Array
  ante: jax.Array


def _seq(seq, n):
  """The first `n` entries of a bet sequence, -1 after.

  Args:
    seq: The bet sequence.
    n: Its length.

  Returns:
    i32[MAX_BETS_PER_ROUND].
  """
  idx = jnp.arange(MAX_BETS_PER_ROUND, dtype=jnp.int32)
  return jnp.where(idx < n, jnp.asarray(seq, jnp.int32), jnp.int32(-1))


class LeducEnv(jax_game_env.JaxGameEnv):
  """Leduc poker: 2 seats, 3 actions, 6-card deck."""

  name = "leduc_poker"
  max_len = MAX_LEN
  zero_sum = True

  def __init__(self):
    super().__init__(leduc_poker.LeducPoker())
    assert self.max_chance_outcomes == NUM_CARDS, self.max_chance_outcomes
    assert self.num_actions == 3 and self.num_players == 2

  def info_state(self, s: leduc_poker.GameState, seat) -> LeducInfo:
    """What `seat` observes.

    Args:
      s: The state.
      seat: The observing seat.

    Returns:
      The seat's LeducInfo.
    """
    seat = jnp.asarray(seat, jnp.int32)
    card = jnp.asarray(s.private_cards, jnp.int32)[seat]
    card = jnp.where(card == game_base.INVALID_CARD, jnp.int32(-1), card)
    pub = jnp.asarray(s.public_card, jnp.int32)
    pub = jnp.where(pub == game_base.INVALID_CARD, jnp.int32(-1), pub)
    return LeducInfo(
        seat=seat,
        card=card,
        public_card=pub,
        round=jnp.asarray(s.round, jnp.int32),
        round1=_seq(s.round1_sequence, s.round1_seq_len),
        round2=_seq(s.round2_sequence, s.round2_seq_len),
        ante=jnp.asarray(s.ante, jnp.int32),
    )

  def labels(self, s: leduc_poker.GameState, seat):
    """Training labels: the opponent's card (-1 before the deal).

    Args:
      s: The state.
      seat: The seat whose opponent is labelled.

    Returns:
      {"opp_card": i32}.
    """
    seat = jnp.asarray(seat, jnp.int32)
    opp = jnp.asarray(s.private_cards, jnp.int32)[1 - seat]
    return {
        "opp_card": jnp.where(opp == game_base.INVALID_CARD, jnp.int32(-1), opp)
    }


def hide_opponent_card(env: LeducEnv, s: leduc_poker.GameState, seat, key):
  """Privacy hook for check_env: redraws `seat`'s opponent card.

  Uniformly from the cards `seat` cannot rule out (not its own, not the public
  card), keeping deck_mask consistent with the dealt cards.

  Args:
    env: The env (unused).
    s: The state.
    seat: The seat whose view is kept.
    key: The PRNG key.

  Returns:
    The state with the opponent's card redrawn.
  """
  del env
  seat = jnp.asarray(seat, jnp.int32)
  cards = jnp.asarray(s.private_cards, jnp.int32)
  own = cards[seat]
  pub = jnp.asarray(s.public_card, jnp.int32)
  ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
  allowed = (ids != own) & (ids != pub)
  logits = jnp.where(allowed, jnp.float32(0.0), -jnp.inf)
  opp = jax.random.categorical(key, logits).astype(jnp.int32)
  new_cards = cards.at[1 - seat].set(opp)
  dealt = (ids == new_cards[0]) | (ids == new_cards[1]) | (ids == pub)
  return s._replace(private_cards=new_cards, deck_mask=~dealt)
