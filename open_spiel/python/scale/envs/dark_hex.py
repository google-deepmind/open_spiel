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

"""3x3 Dark Hex Envs over the JAX engines `dark_hex` and `dark_hex_ir`.

  dark_hex     classical (OpenSpiel `dark_hex`, gameversion=cdh,
  reveal-nothing):
               a move onto a cell the opponent holds fails, reveals that cell
               to the mover, and the mover moves again.
  dark_hex_ir  abrupt Dark Hex with imperfect recall (OpenSpiel
               `dark_hex_ir(gameversion=adh)`; pyspiel's default for that name
               is cdh, which serving refuses): a failed move reveals the cell
               and passes the turn, and a seat remembers only its view.

Black (seat 0) joins top to bottom, white (seat 1) left to right; cells and
actions are row-major. Moves are private (visible_action hides the other
seat's moves). The engine states are dicts with a `done` flag and
current_player -4 at terminal; the adapter reports actor -1.

InfoStates follow OpenSpiel's reveal-nothing information-state TENSOR: the
seat's view, and (classical, perfect recall) its own attempts in order. The
information-state STRING of OpenSpiel's dark_hex also carries the total
history length, which reveal-nothing does not give a player; it is not part
of the InfoState.

Stateful policies are refused (Env.admits_stateful_policies False): a policy
carry is updated at every ply, and the plies are not private. In classical
Dark Hex a failed attempt keeps the turn, so the histories [0, 1] and
[0, 0, 1] leave black the same information state (OpenSpiel's reveal-nothing
state omits the opponent's attempts) but three plies to observe instead of
two: a carry could count white's retries. In dark_hex_ir a carry would
restore the memory of past attempts that imperfect recall removes. A
feed-forward policy reads the InfoState alone, which has neither problem.

A hand has at most 17 decisions (each seat attempts each cell at most once,
and both cannot have attempted all 9: the last empty cell would be the first
attempter's); pyspiel's max_game_length is 17 for both games.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games import dark_hex as jdh
from open_spiel.python.jax.games import dark_hex_ir as jdh_ir
from open_spiel.python.scale.envs import hidden_cells
from open_spiel.python.scale.envs import jax_game_env

I32 = jnp.int32
NUM_CELLS = jdh.NUM_CELLS  # 9
MAX_LEN = 2 * NUM_CELLS - 1  # 17
assert jdh_ir.MAX_ATTEMPTS == MAX_LEN and jdh_ir.NUM_CELLS == NUM_CELLS

# Classical view values (the engine's dark_hex == OpenSpiel's cell states).
EMPTY_VIEW = jdh.EMPTY  # 0
BLACK_VIEW = jdh.BLACK_VIEW  # 1 (a black stone)
WHITE_VIEW = jdh.WHITE_VIEW  # -1 (a white stone)
# Board / IR view marks.
EMPTY, BLACK, WHITE = jdh.EMPTY, jdh.BLACK, jdh.WHITE  # 0, 1, 2


class DarkHexInfo(NamedTuple):
  """Classical: what one seat knows, its view and its own attempts in order.

  Attributes:
    seat: i32.
    view: i32[9]: 0 unknown, 1 black, -1 white, 4 / -4 the winning stone.
    own_actions: i32[9]: this seat's attempted cells in order, -1 pad.
    num_own_actions: i32.
  """

  seat: jax.Array
  view: jax.Array
  own_actions: jax.Array
  num_own_actions: jax.Array


class DarkHexIRInfo(NamedTuple):
  """Abrupt, imperfect recall: the seat's view only (no move order).

  Attributes:
    seat: i32.
    view: i32[9]: 0 unknown, 1 black, 2 white.
  """

  seat: jax.Array
  view: jax.Array


class _DarkHexBase(jax_game_env.JaxGameEnv):
  """What both variants share: dict states with `done`, private moves."""

  max_len = MAX_LEN
  zero_sum = True
  admits_stateful_policies = False  # module doc: Stateful policies

  def terminal(self, s) -> jax.Array:
    """Whether the game is over at `s`.

    Args:
      s: The state.

    Returns:
      bool: True at a terminal state.
    """
    return jnp.asarray(s["done"], jnp.bool_)

  def visible_action(self, s, a, seat) -> jax.Array:
    """A seat sees its own moves; the other seat's moves are hidden.

    Args:
      s: The state.
      a: The player action id.
      seat: The seat.

    Returns:
      i32 action as `seat` sees it (NO_ACTION when hidden).
    """
    return hidden_cells.private_visible_action(self, s, a, seat)

  def labels(self, s, seat):
    """Privileged targets for learners (e.g. the opponent's hand).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      A mapping of label name to array.
    """
    opp_mark = jnp.where(jnp.asarray(seat, I32) == 0, I32(WHITE), I32(BLACK))
    return {"opp_stones": jnp.asarray(s["board"], I32).reshape(-1) == opp_mark}


class DarkHexEnv(_DarkHexBase):
  """Classical 3x3 Dark Hex: 2 seats, 9 actions (cells)."""

  name = "dark_hex"

  def __init__(self):
    super().__init__(jdh.DarkHex())
    assert self.max_chance_outcomes == 0 and self.num_actions == NUM_CELLS

  def info_state(self, s, seat) -> DarkHexInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, I32)
    own, n = hidden_cells.own_actions(
        s["action_sequence"],
        s["action_players"],
        s["history_length"],
        seat,
        NUM_CELLS,
    )
    return DarkHexInfo(
        seat=seat,
        view=jnp.asarray(s["views"], I32)[seat],
        own_actions=own,
        num_own_actions=n,
    )


class DarkHexIREnv(_DarkHexBase):
  """Abrupt 3x3 Dark Hex, imperfect recall: 2 seats, 9 actions (cells)."""

  name = "dark_hex_ir"

  def __init__(self):
    super().__init__(jdh_ir.DarkHexIR(gameversion="adh"))
    assert self.max_chance_outcomes == 0 and self.num_actions == NUM_CELLS

  def info_state(self, s, seat) -> DarkHexIRInfo:
    """What `seat` may know at `s` (the privacy boundary).

    Args:
      s: The state.
      seat: The seat (i32).

    Returns:
      The seat's InfoState; hidden information is absent.
    """
    seat = jnp.asarray(seat, I32)
    view = jnp.where(seat == 0, s["p0_view"], s["p1_view"])
    return DarkHexIRInfo(seat=seat, view=jnp.asarray(view, I32).reshape(-1))


# -- privacy hooks for check_env
# ------------------------------------------------


def hide_unseen_cells(env: DarkHexEnv, s: Any, seat, key) -> Any:
  """Classical: permute, at random, every cell `seat` has not seen.

  Moving the opponent's hidden stones with the opponent's view and attempt log.
  The result may be unreachable (e.g. an opponent chain with done False);
  info_state(seat) must be blind to it all the same.

  Args:
    env: The env.
    s: The state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The rewritten state.
  """
  del env
  return hidden_cells.permute_unseen(
      s,
      seat,
      key,
      views="views",
      actions="action_sequence",
      players="action_players",
      empty=EMPTY_VIEW,
  )


def hide_unseen_cells_ir(env: DarkHexIREnv, s: Any, seat, key) -> Any:
  """Abrupt: as hide_unseen_cells, over the IR state's fields.

  Args:
    env: The env.
    s: The state.
    seat: The seat whose InfoState must not change.
    key: PRNG key.

  Returns:
    The rewritten state.
  """
  del env
  seat = jnp.asarray(seat, I32)
  opp = 1 - seat
  my_view = jnp.where(seat == 0, s["p0_view"], s["p1_view"]).reshape(-1)
  perm = hidden_cells.unseen_permutation(
      jnp.asarray(my_view, I32) == EMPTY, key
  )
  out = dict(s)
  out["board"] = hidden_cells.move_cells(s["board"], perm)
  for p, k in ((0, "p0_view"), (1, "p1_view")):
    out[k] = jnp.where(opp == p, hidden_cells.move_cells(s[k], perm), s[k])
  out["action_sequence_actions"] = hidden_cells.move_actions(
      s["action_sequence_actions"], s["action_sequence_players"], opp, perm
  )
  wc = jnp.asarray(s["winning_cell"], I32)
  out["winning_cell"] = jnp.where(wc >= 0, perm[jnp.maximum(wc, 0)], wc).astype(
      s["winning_cell"].dtype
  )
  return out
