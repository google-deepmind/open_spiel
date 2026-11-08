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

"""Bargaining Env over the JAX engine `bargaining` (OpenSpiel `bargaining`).

With a BargainingInfo InfoState PyTree.

Deal: one chance node over the 1000 built-in instances (the engine's bundled
bargaining_instances.txt, OpenSpiel's default instances): a public pool of 3
item types and each seat's private values (each summing to 10). Seats
alternate offers, seat 0 first; an offer [a, b, c] (an amount of each item
type, ids in OpenSpiel's order: c outer, b, a inner, a + b + c <= 7) is what
its PROPOSER keeps. AGREE (action 120, legal once an offer exists) accepts
the last offer: the proposer gets values[proposer] . offer, the other seat
values[other] . (pool - offer), both times the discount (applied from the
third decision on, OpenSpiel's move_number >= 3). No agreement within
max_turns offers, or an end drawn at a prob_end chance node (after each offer
from the second on), pays 0, 0. General-sum: zero_sum is False.

Options (OpenSpiel's parameters): max_turns (int >= 1), discount in (0, 1],
prob_end in [0, 1).

Chance ids are pyspiel's, so a pyspiel history replays as it is: the deal
is the instance id; with prob_end > 0 the mid-hand chance node has outcomes
1000 (continue) and 1001 (end), after the 1000 instances.

The engine leaves current_player at chance (-1) at terminal; the adapter's
is_chance excludes terminal states and actor() reports -1.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import bargaining
from open_spiel.python.scale.envs import jax_game_env

NUM_ITEM_TYPES = 3
POOL_MAX_NUM_ITEMS = 7
OFFERS = np.asarray(bargaining.enumerate_offers(), np.int32)
NUM_OFFERS = len(OFFERS)  # 120
AGREE = NUM_OFFERS
NUM_INSTANCES = 1000
DEFAULT_MAX_TURNS = 10


def offer_id(a: int, b: int, c: int) -> int:
  """The action id of the offer keeping (a, b, c) for the proposer.

  Args:
    a: Amount of the first item type.
    b: Amount of the second item type.
    c: Amount of the third item type.

  Returns:
    The action id.

  Raises:
    ValueError: No offer keeps (a, b, c).
  """
  hit = np.flatnonzero((OFFERS == np.asarray([a, b, c], np.int32)).all(axis=1))
  if hit.size != 1:
    raise ValueError(f"no bargaining offer ({a}, {b}, {c})")
  return int(hit[0])


class BargainingInfo(NamedTuple):
  """What one seat knows: its seat and values, and everything public.

  Attributes:
    seat: i32.
    pool: i32[3] public item counts.
    values: i32[3] own private values.
    offers: i32[max_turns, 3] offers in order, -1 rows not yet made.
    num_offers: i32.
  """

  seat: jax.Array
  pool: jax.Array
  values: jax.Array
  offers: jax.Array
  num_offers: jax.Array


class BargainingEnv(jax_game_env.JaxGameEnv):
  """Bargaining: 2 seats, 121 actions (120 offers + AGREE), general-sum.

  Attributes:
    max_turns: The turn limit: the number of offers before the game ends.
    discount: The per-turn discount of the agreed values, in (0, 1].
    prob_end: The chance the game ends after each offer, in [0, 1).
    max_len: The longest game in decisions (max_turns: every decision but AGREE
      is an offer).
  """

  name = "bargaining"
  max_len = DEFAULT_MAX_TURNS
  zero_sum = False

  def __init__(
      self,
      max_turns: int = DEFAULT_MAX_TURNS,
      discount: float = 1.0,
      prob_end: float = 0.0,
  ):
    if (
        isinstance(max_turns, bool)
        or not isinstance(max_turns, int)
        or max_turns < 1
    ):
      raise ValueError(
          f"bargaining max_turns must be an int >= 1, got {max_turns!r}"
      )
    if not 0.0 < float(discount) <= 1.0:
      raise ValueError(f"bargaining discount={discount!r} must be in (0, 1]")
    if not 0.0 <= float(prob_end) < 1.0:
      raise ValueError(f"bargaining prob_end={prob_end!r} must be in [0, 1)")
    self.max_turns = int(max_turns)
    self.discount = float(discount)
    self.prob_end = float(prob_end)
    self.max_len = self.max_turns  # every decision but AGREE is an offer
    super().__init__(
        bargaining.Bargaining(
            max_turns=self.max_turns,
            discount=self.discount,
            prob_end=self.prob_end,
        )
    )
    assert self.num_players == 2 and self.num_actions == NUM_OFFERS + 1
    # The instances, then the mid-hand continue / end outcomes (pyspiel's ids).
    assert (
        self.max_chance_outcomes == NUM_INSTANCES + 2
    ), self.max_chance_outcomes
    assert (self.game.continue_outcome, self.game.end_outcome) == (
        NUM_INSTANCES,
        NUM_INSTANCES + 1,
    )
    assert np.array_equal(self.game.all_offers, OFFERS)

  def info_state(self, s: bargaining.GameState, seat) -> BargainingInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, jnp.int32)
    n = jnp.asarray(s.num_offers, jnp.int32)
    made = jnp.arange(self.max_turns, dtype=jnp.int32)[:, None] < n
    return BargainingInfo(
        seat=seat,
        pool=jnp.asarray(s.pool, jnp.int32),
        values=jnp.asarray(s.values, jnp.int32)[seat],
        offers=jnp.where(made, jnp.asarray(s.offers, jnp.int32), jnp.int32(-1)),
        num_offers=n,
    )

  def labels(self, s: bargaining.GameState, seat):
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return {"opp_values": jnp.asarray(s.values, jnp.int32)[1 - seat]}


def hide_opponent_values(
    env: BargainingEnv, s: bargaining.GameState, seat, key
):
  """Privacy hook for check_env: replace the opponent's values at random.

  Privacy hook for check_env: replace `seat`'s opponent's values with
  random ones in [0, 10] (possibly no instance's: info_state(seat) must be
  blind to them all the same).

  Args:
    env: The env.
    s: The state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The state with `seat`'s opponent's values replaced.
  """
  del env
  seat = jnp.asarray(seat, jnp.int32)
  v = jax.random.randint(key, (NUM_ITEM_TYPES,), 0, 11, dtype=jnp.int32).astype(
      s.values.dtype
  )
  return s._replace(values=s.values.at[1 - seat].set(v))
