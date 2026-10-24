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

"""Kuhn poker Env over the JAX engine `kuhn_poker`.

With a KuhnInfo InfoState PyTree.

envs.kuhn_jax ("kuhn_poker_jax") is the same rules as a self-contained
implementation that needs no game package; use it when the JAX games package
is not installed. Both share KuhnInfo, so one set of encoders reads either.
"""

import jax.numpy as jnp

from open_spiel.python.jax.games import kuhn_poker
from open_spiel.python.scale.envs import jax_game_env
from open_spiel.python.scale.envs import kuhn_jax

NUM_CARDS = kuhn_jax.NUM_CARDS
MAX_BETS = kuhn_jax.MAX_BETS
HISTORIES = kuhn_jax.HISTORIES
NUM_INFO_STATES = kuhn_jax.NUM_INFO_STATES
KuhnInfo = kuhn_jax.KuhnInfo
info_index = kuhn_jax.info_index
hide_opponent_card = kuhn_jax.hide_opponent_card


class KuhnEnv(jax_game_env.JaxGameEnv):
  """Kuhn poker: 2 seats, actions pass (0) / bet (1)."""

  name = "kuhn_poker"
  max_len = MAX_BETS
  zero_sum = True

  def __init__(self):
    super().__init__(kuhn_poker.KuhnPoker())

  def info_state(self, s: kuhn_poker.GameState, seat) -> KuhnInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    card = jnp.where(seat == 0, s.p0_card, s.p1_card)
    return KuhnInfo(
        seat=seat,
        card=jnp.asarray(card, jnp.int32),
        bets=jnp.asarray(s.bets, jnp.int32),
        num_bets=jnp.asarray(s.num_bets, jnp.int32),
    )

  def labels(self, s: kuhn_poker.GameState, seat):
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return {
        "opp_card": jnp.where(seat == 0, s.p1_card, s.p0_card).astype(jnp.int32)
    }

  def deal(self, p0_card: int, p1_card: int) -> kuhn_poker.GameState:
    """A state with the given cards dealt (for known-answer tests).

    Args:
      p0_card: Seat 0's card.
      p1_card: Seat 1's card.

    Returns:
      The dealt state.
    """
    assert p0_card != p1_card and 0 <= p0_card < 3 and 0 <= p1_card < 3
    g = self.game
    s = g.apply_action(g.new_initial_state(), jnp.int32(p0_card))
    return g.apply_action(s, jnp.int32(p1_card))
