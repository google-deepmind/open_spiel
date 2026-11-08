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

"""One-shot N-player matrix games with known equilibria, as sequential Envs.

The unit-test tier of the N-player league: every payoff is a small integer,
every expectation under a product of mixed strategies is a finite sum, so
per-seat, per-side, rotation and sum-invariant bookkeeping can be tested
exactly and in milliseconds.

The simultaneous move is played sequentially: seat 0, then 1, ..., P-1, each
choosing one of A actions. No seat learns another's action
(visible_action hides it, info_state carries only the seat and its own
action), so the extensive form is the normal-form game. The hand ends after
P actions with returns payoffs[a_0, ..., a_{P-1}].

The game is chosen by the option `variant` (registered as "nplayer_matrix"):

  matching_pennies_3p   3 seats, general-sum: OpenSpiel's game of that name
      (Jordan 1993). Seat 0 wants to match seat 1, seat 1 to match seat 2,
      seat 2 to mismatch seat 0; +1 if it does, -1 if not. Unique Nash
      equilibrium: every seat uniform, values (0, 0, 0).
  odd_one_out_3p        3 seats, zero-sum, symmetric: if exactly one seat
      chose differently it gets +2 and the other two -1; all equal, 0 each.
      Equilibria: every seat uniform (values 0), and each of the six pure
      profiles with exactly one odd seat.
  odd_one_out_3p_c3     odd_one_out_3p plus 1 for every seat: constant-sum
      with constant 3 (not zero-sum), the same equilibria (values 1).
  odd_one_out_4p        4 seats, zero-sum, symmetric: if exactly one seat
      chose differently it gets +3 and the others -1; otherwise 0 each.
      Equilibrium: every seat uniform (values 0).
  team_pennies_2v2      4 seats in two partnerships, sides (0, 1, 0, 1),
      zero-sum between them: side 0's bit is a_0 XOR a_2, side 1's is
      a_1 XOR a_3; side 0 scores +1 per seat when the bits are equal, side 1
      when they differ. Team value 0; every profile in which each side's bit
      is uniform is an equilibrium (in particular every seat uniform).
  team_pennies_2v1      3 seats in two uneven sides, sides (0, 1, 0): a pair
      (seats 0 and 2) against one seat (seat 1). The pair's bit is
      a_0 XOR a_2, the lone seat's a_1; when they are equal each pair seat
      scores +1 and the lone seat -1, else the reverse. The side returns
      (mean over a side's seats) sum to 0 (side-level constant-sum, constant
      `side_sum_constant` 0); the seat returns sum to +-1 (not zero-sum).
      Team value 0; equilibria: every seat uniform, and the lone seat
      uniform against any pair whose bit is uniform.
  team_dilemma_2v2      4 seats in two partnerships, sides (0, 1, 0, 1),
      zero-sum; action 0 cooperates (c = 1 - a). Seat i's return is
      2 c_partner - c_i - (c_o + c_o') / 2 (o, o' the other side's seats):
      cooperating costs a seat 1 and gives its partner 2. Partners score
      differently, so seat credit and side credit differ: a seat's own
      return makes defecting dominant (the all-defect profile is the seat
      game's equilibrium, values 0), its side's return (c_i + c_partner) / 2
      - (c_o + c_o') / 2 makes cooperating dominant (the cooperative
      profile, every seat 0, is the team game's equilibrium, values 0).
      The test bed of learners.pg PGConfig.credit="side".
  odd_side_3v1v1        5 seats in three uneven sides, sides
      (0, 1, 0, 2, 0): a team of three (seats 0, 2, 4) and two lone seats
      (seat 1: side 1, seat 3: side 2). The team's bit is a_0 XOR a_2 XOR
      a_4, the lone seats' bits a_1 and a_3. The side whose bit is the odd
      one out of the three scores +2 and the other two -1 (all equal: 0
      each); then side 1 gains a_1 and side 2 loses a_1 (seat 1's bonus,
      paid by seat 3), so the two lone positions differ and a design that
      rotates an entrant over them is visible in its score. The side
      returns sum to 0 (`side_sum_constant` 0); the seat returns do not.
      Equilibrium: the team uniform, seat 1 playing 1, seat 3 playing 0
      (side values -1, 1.5, -0.5).

`KNOWN_EQUILIBRIA[variant]` lists equilibria (one distribution per seat) that
the tests check exactly with core.game_tree (NashConv 0).

Partners: every game but team_dilemma_2v2 gives the seats of one side the
same return (`equal_partners`, checked on the table).

Sides: each game declares its partition (`side_of_seat`) and whether the
side returns have a constant sum (`constant_sum`, the constant in
`sum_constant`), and `rotatable`. `env.sides` is the core.sides.Sides built from
them, so an env whose returns are constant-sum but not zero-sum
(odd_one_out_3p_c3) is never mistaken for general-sum, nor a partnership game
for four individuals.

`seat_index` maps a batched MatrixInfo to its seat, the index_fn of a
policies.tabular.TabularPolicy with one row per seat (a mixed strategy per
seat; a seat decides once, before it knows anything else).
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types


@dataclasses.dataclass(frozen=True)
class MatrixGame:
  """A one-shot game: payoffs f32[A]*P + [P] over joint actions."""

  name: str
  payoffs: np.ndarray
  side_of_seat: tuple[int, ...]
  # The sum of the seat returns when constant, else None.
  sum_constant: Optional[float]
  rotatable: bool
  # The sum of the side returns (core.sides: a side's mean over its seats)
  # when constant and the seat sum is not (uneven sides), else None.
  side_sum_constant: Optional[float] = None
  # Every seat of a side gets the same return (False: team_dilemma_2v2).
  equal_partners: bool = True

  @property
  def num_players(self) -> int:
    """The number of seats."""
    return self.payoffs.ndim - 1

  @property
  def num_actions(self) -> int:
    """The number of actions per seat."""
    return self.payoffs.shape[0]


def _table(
    num_players: int,
    num_actions: int,
    fn: Callable[[tuple[int, ...]], tuple[float, ...]],
) -> np.ndarray:
  """The payoff table of `fn` over every joint action.

  Args:
    num_players: P.
    num_actions: Actions per seat.
    fn: fn(joint action) -> each seat's payoff.

  Returns:
    f32[A, ..., A, P].

  Raises:
    AssertionError: fn returns other than P payoffs.
  """
  out = np.zeros((num_actions,) * num_players + (num_players,), np.float32)
  for joint in itertools.product(range(num_actions), repeat=num_players):
    r = fn(joint)
    if len(r) != num_players:
      raise AssertionError(
          f"payoff of {joint} has {len(r)} entries, want {num_players}"
      )
    out[joint] = r
  return out


def _matching_pennies_3p(a):
  """Three-player matching pennies (each seat matches the next).

  Args:
    a: The joint action.

  Returns:
    Each seat's payoff.
  """
  return (
      1.0 if a[0] == a[1] else -1.0,
      1.0 if a[1] == a[2] else -1.0,
      1.0 if a[2] != a[0] else -1.0,
  )


def _odd_one_out(win: float, shift: float = 0.0):
  """The odd-one-out payoff: the lone deviator wins.

  Args:
    win: The odd seat's payoff (the others get -1).
    shift: Added to every payoff.

  Returns:
    fn(joint action) -> each seat's payoff.
  """

  def fn(a):
    """The payoff of joint action `a`.

    Args:
      a: The joint action.

    Returns:
      Each seat's payoff.
    """
    ones = sum(a)
    odd = None
    if ones == 1:
      odd = a.index(1)
    elif ones == len(a) - 1:
      odd = a.index(0)
    if odd is None:
      return tuple(shift for _ in a)
    return tuple((win if i == odd else -1.0) + shift for i in range(len(a)))

  return fn


def _team_pennies(a):
  """2v2 team pennies: the side whose parities match wins.

  Args:
    a: The joint action.

  Returns:
    Each seat's payoff.
  """
  r = 1.0 if (a[0] ^ a[2]) == (a[1] ^ a[3]) else -1.0
  return (r, -r, r, -r)


def _team_pennies_2v1(a):
  """2v1 team pennies.

  Args:
    a: The joint action.

  Returns:
    Each seat's payoff.
  """
  r = 1.0 if (a[0] ^ a[2]) == a[1] else -1.0
  return (r, -r, r)


def _team_dilemma(a):
  """The 2v2 team dilemma (action 0 cooperates).

  Args:
    a: The joint action.

  Returns:
    Each seat's payoff.
  """
  c = [1.0 - float(x) for x in a]  # action 0 cooperates
  out = []
  for i in range(4):
    partner, opp = (i + 2) % 4, ((i + 1) % 4, (i + 3) % 4)
    out.append(2.0 * c[partner] - c[i] - (c[opp[0]] + c[opp[1]]) / 2.0)
  return tuple(out)


def _odd_side_3v1v1(a):
  """The 3v1v1 odd-side game.

  Args:
    a: The joint action.

  Returns:
    Each seat's payoff.
  """
  bits = (a[0] ^ a[2] ^ a[4], a[1], a[3])
  side = [0.0, 0.0, 0.0]
  for k in range(3):
    others = [bits[j] for j in range(3) if j != k]
    if others[0] == others[1] != bits[k]:
      side = [2.0 if j == k else -1.0 for j in range(3)]
  side[1] += float(a[1])
  side[2] -= float(a[1])
  return (side[0], side[1], side[0], side[2], side[0])


def _game(
    name,
    num_players,
    fn,
    side_of_seat,
    sum_constant,
    rotatable=True,
    side_sum_constant=None,
    equal_partners=True,
) -> MatrixGame:
  """A two-action MatrixGame from its payoff function.

  Args:
    name: The variant.
    num_players: P.
    fn: fn(joint action) -> each seat's payoff.
    side_of_seat: Each seat's side.
    sum_constant: The constant seat sum, or None.
    rotatable: Whether seats may be rotated.
    side_sum_constant: The constant side sum, or None.
    equal_partners: Whether partners always score alike.

  Returns:
    The MatrixGame.
  """
  return MatrixGame(
      name,
      _table(num_players, 2, fn),
      tuple(side_of_seat),
      sum_constant,
      rotatable,
      side_sum_constant,
      equal_partners,
  )


GAMES: dict[str, MatrixGame] = {
    g.name: g
    for g in (
        _game("matching_pennies_3p", 3, _matching_pennies_3p, (0, 1, 2), None),
        _game("odd_one_out_3p", 3, _odd_one_out(2.0), (0, 1, 2), 0.0),
        _game(
            "odd_one_out_3p_c3", 3, _odd_one_out(2.0, shift=1.0), (0, 1, 2), 3.0
        ),
        _game("odd_one_out_4p", 4, _odd_one_out(3.0), (0, 1, 2, 3), 0.0),
        _game("team_pennies_2v2", 4, _team_pennies, (0, 1, 0, 1), 0.0),
        _game(
            "team_pennies_2v1",
            3,
            _team_pennies_2v1,
            (0, 1, 0),
            None,
            side_sum_constant=0.0,
        ),
        _game(
            "team_dilemma_2v2",
            4,
            _team_dilemma,
            (0, 1, 0, 1),
            0.0,
            equal_partners=False,
        ),
        _game(
            "odd_side_3v1v1",
            5,
            _odd_side_3v1v1,
            (0, 1, 0, 2, 0),
            None,
            side_sum_constant=0.0,
        ),
    )
}

_U = (0.5, 0.5)
KNOWN_EQUILIBRIA: dict[str, tuple[tuple[tuple[float, ...], ...], ...]] = {
    "matching_pennies_3p": ((_U,) * 3,),
    "odd_one_out_3p": ((_U,) * 3,) + tuple(
        tuple((1.0, 0.0) if b == 0 else (0.0, 1.0) for b in joint)
        for joint in itertools.product((0, 1), repeat=3)
        if sum(joint) in (1, 2)
    ),
    "odd_one_out_3p_c3": ((_U,) * 3,),
    "odd_one_out_4p": ((_U,) * 4,),
    "team_pennies_2v2": ((_U,) * 4, (_U, (1.0, 0.0), (1.0, 0.0), _U)),
    "team_pennies_2v1": ((_U,) * 3, ((1.0, 0.0), _U, _U)),
    "team_dilemma_2v2": (((0.0, 1.0),) * 4,),
    "odd_side_3v1v1": ((_U, (0.0, 1.0), _U, (1.0, 0.0), _U),),
}


def _check_game(g: MatrixGame) -> None:
  """Fail loud on a table that contradicts its declared structure.

  Args:
    g: The game.

  Raises:
    AssertionError: The table contradicts its declared structure.
  """
  p = g.num_players
  if (
      g.payoffs.shape != (g.num_actions,) * p + (p,)
      or g.payoffs.dtype != np.float32
  ):
    raise AssertionError(
        f"{g.name}: payoffs {g.payoffs.dtype}{g.payoffs.shape}"
    )
  if len(g.side_of_seat) != p:
    raise AssertionError(
        f"{g.name}: side_of_seat {g.side_of_seat} for {p} seats"
    )
  sums = g.payoffs.sum(-1)
  if g.sum_constant is not None and not np.all(sums == g.sum_constant):
    raise AssertionError(
        f"{g.name}: declared constant sum {g.sum_constant}, sums "
        f"{np.unique(sums)}"
    )
  if g.sum_constant is None and np.all(sums == sums.flat[0]):
    raise AssertionError(
        f"{g.name}: declared general-sum, but every sum is {sums.flat[0]}"
    )
  if g.side_sum_constant is not None:
    if g.sum_constant is not None:
      raise AssertionError(
          f"{g.name}: a seat sum constant and a side sum constant; declare "
          "the seat sum (it fixes the side sum for equal sides)"
      )
    side_sums = (
        sides_lib.Sides(g.side_of_seat, True, g.rotatable)
        .side_returns(g.payoffs.astype(np.float64))
        .sum(-1)
    )
    if not np.all(side_sums == g.side_sum_constant):
      raise AssertionError(
          f"{g.name}: declared side sum {g.side_sum_constant}, side sums "
          f"{np.unique(side_sums)}"
      )
  partners_differ = False
  for s in set(g.side_of_seat):
    seats = [i for i, x in enumerate(g.side_of_seat) if x == s]
    r = g.payoffs[..., seats]
    if not np.all(r == r[..., :1]):
      partners_differ = True
      if g.equal_partners:
        raise AssertionError(
            f"{g.name}: partners {seats} of side {s} get different returns"
        )
  if not g.equal_partners and not partners_differ:
    raise AssertionError(
        f"{g.name}: declared unequal partners, but every side's seats score "
        "alike"
    )


for _g in GAMES.values():
  _check_game(_g)


class MatrixInfo(NamedTuple):
  """What one seat knows: its seat and its own action (-1 before it acts)."""

  seat: jax.Array  # i32
  own_action: jax.Array  # i32


class MatrixState(NamedTuple):
  """The raw state: the actions played so far.

  Attributes:
    actions: i32[P]; seat i's action, -1 until it acts.
    num_moves: i32; the actions played.
  """

  actions: jax.Array  # i32[P]; seat i's action, -1 until it acts
  num_moves: jax.Array  # i32


class NPlayerMatrixEnv:
  """types.Env of one game of GAMES (module doc); every method on one state."""

  name = "nplayer_matrix"
  max_chance_outcomes = 0

  def __init__(self, variant: str = "matching_pennies_3p"):
    if variant not in GAMES:
      raise ValueError(
          f"nplayer_matrix variant must be one of {sorted(GAMES)}, got "
          f"{variant!r}"
      )
    self.matrix = GAMES[variant]
    self.variant = variant
    self.num_players = self.matrix.num_players
    self.num_actions = self.matrix.num_actions
    self.max_len = self.num_players
    self.side_of_seat = self.matrix.side_of_seat
    self.sum_constant = self.matrix.sum_constant
    self.side_sum_constant = self.matrix.side_sum_constant
    self.constant_sum = (
        self.sum_constant is not None or self.side_sum_constant is not None
    )
    self.rotatable = self.matrix.rotatable
    self.zero_sum = self.sum_constant == 0.0
    self._flat = jnp.asarray(
        self.matrix.payoffs.reshape(-1, self.num_players), jnp.float32
    )
    # The rules live in this module, which registry.rules_sha hashes.
    self.game = self

  @property
  def sides(self) -> sides_lib.Sides:
    """The core.sides.Sides this game declares.

    side_of_seat, constant_sum and rotatable, so a constant-sum game that is not
    zero-sum (odd_one_out_3p_c3) is never taken for general-sum, nor a
    partnership game (team_pennies_2v2) for four individuals.
    """
    return sides_lib.Sides(
        side_of_seat=self.side_of_seat,
        constant_sum=self.constant_sum,
        rotatable=self.rotatable,
    )

  # -- raw layer (no chance nodes) ---------------------------------------------
  def root(self) -> MatrixState:
    """The state before any seat acts.

    Returns:
      The root state.
    """
    return MatrixState(
        actions=jnp.full((self.num_players,), -1, jnp.int32),
        num_moves=jnp.int32(0),
    )

  def is_chance(self, s: MatrixState) -> jax.Array:
    """Whether `s` is a chance node: never.

    Args:
      s: The state.

    Returns:
      bool False.
    """
    del s
    return jnp.bool_(False)

  def apply_chance(self, s: MatrixState, outcome) -> MatrixState:
    """The identity: the game has no chance node (max_chance_outcomes 0).

    Args:
      s: The state.
      outcome: Unused.

    Returns:
      s.
    """
    del outcome
    return s

  def apply(self, s: MatrixState, a) -> MatrixState:
    """Seat num_moves plays `a` (a terminal state is left as is).

    Args:
      s: The state.
      a: The action.

    Returns:
      The next state.
    """
    a = jnp.asarray(a, jnp.int32)
    live = ~self.terminal(s)
    slot = jnp.minimum(s.num_moves, self.num_players - 1)
    return MatrixState(
        actions=jnp.where(live, s.actions.at[slot].set(a), s.actions),
        num_moves=jnp.where(live, s.num_moves + 1, s.num_moves),
    )

  def init(self, deal_key) -> MatrixState:
    """The first decision state (there is no deal).

    Args:
      deal_key: Unused.

    Returns:
      The root state.
    """
    del deal_key
    return self.root()

  def step(self, s, a, chance_key) -> MatrixState:
    """Plays `a`; there is no chance to resolve.

    Args:
      s: The state.
      a: The action.
      chance_key: Unused.

    Returns:
      The next state.
    """
    del chance_key
    return self.apply(s, a)

  def chance_probs(self, s: MatrixState) -> jax.Array:
    """The chance distribution: empty (no chance node).

    Args:
      s: The state.

    Returns:
      f32[0].
    """
    del s
    return jnp.zeros((0,), jnp.float32)

  # -- queries -----------------------------------------------------------------
  def terminal(self, s: MatrixState) -> jax.Array:
    """Whether every seat has acted.

    Args:
      s: The state.

    Returns:
      bool.
    """
    return s.num_moves >= self.num_players

  def actor(self, s: MatrixState) -> jax.Array:
    """The seat to act, -1 at a terminal state.

    Args:
      s: The state.

    Returns:
      i32.
    """
    return jnp.where(
        self.terminal(s), jnp.int32(-1), s.num_moves.astype(jnp.int32)
    )

  def legal_mask(self, s: MatrixState) -> jax.Array:
    """Every action is legal before the end.

    Args:
      s: The state.

    Returns:
      bool[A].
    """
    return jnp.broadcast_to(~self.terminal(s), (self.num_actions,))

  def returns(self, s: MatrixState) -> jax.Array:
    """Each seat's payoff at a terminal state, zeros before.

    Args:
      s: The state.

    Returns:
      f32[P].
    """
    place = self.num_actions ** jnp.arange(
        self.num_players - 1, -1, -1, dtype=jnp.int32
    )
    idx = jnp.sum(jnp.maximum(s.actions, 0) * place)
    return jnp.where(self.terminal(s), self._flat[idx], 0.0).astype(jnp.float32)

  def info_state(self, s: MatrixState, seat) -> MatrixInfo:
    """What `seat` knows: its seat and its own action.

    Args:
      s: The state.
      seat: The seat.

    Returns:
      The MatrixInfo.
    """
    seat = jnp.asarray(seat, jnp.int32)
    return MatrixInfo(
        seat=seat, own_action=jnp.asarray(s.actions[seat], jnp.int32)
    )

  def labels(self, s: MatrixState, seat):
    """The hidden-information labels of `seat`: none.

    Args:
      s: The state.
      seat: The seat.

    Returns:
      {}.
    """
    del s, seat
    return {}

  def visible_action(self, s: MatrixState, a, seat) -> jax.Array:
    """The actor sees its own action; no other seat sees it.

    Args:
      s: The state before `a`.
      a: The action played.
      seat: The observing seat.

    Returns:
      i32: `a` for the actor, types.NO_ACTION for every other seat.
    """
    mine = self.actor(s) == jnp.asarray(seat, jnp.int32)
    return jnp.where(
        mine, jnp.asarray(a, jnp.int32), jnp.int32(types.NO_ACTION)
    )

  def world_from_info(self, info) -> Optional[MatrixState]:
    """No world is built from an info state here.

    Args:
      info: The MatrixInfo.

    Returns:
      None.
    """
    del info
    return None


def seat_index(info: MatrixInfo) -> jax.Array:
  """The seat of a (batched) MatrixInfo; a tabular policy's row (module doc).

  Args:
    info: The MatrixInfo.

  Returns:
    i32[...].
  """
  return jnp.asarray(info.seat, jnp.int32)


def hide_other_actions(
    env: NPlayerMatrixEnv, s: MatrixState, seat, key
) -> MatrixState:
  """Privacy hook for check_env: every other seat's action redrawn uniformly.

  Args:
    env: The env.
    s: The state.
    seat: The seat whose view is kept.
    key: The PRNG key of the redraw.

  Returns:
    The state with every other seat's played action redrawn.
  """
  seat = jnp.asarray(seat, jnp.int32)
  fresh = jax.random.randint(
      key, (env.num_players,), 0, env.num_actions, jnp.int32
  )
  other = (jnp.arange(env.num_players) != seat) & (s.actions >= 0)
  return s._replace(actions=jnp.where(other, fresh, s.actions))


def exact_returns(variant: str, strategies) -> np.ndarray:
  """Each seat's expected return under independent strategies.

  Seat i plays strategies[i] (a distribution over the actions) independently;
  a finite sum.

  Args:
    variant: A key of GAMES.
    strategies: One distribution over the actions per seat.

  Returns:
    f64[P].

  Raises:
    ValueError: not one distribution over the actions per seat.
  """
  g = GAMES[variant]
  probs = [np.asarray(x, np.float64) for x in strategies]
  if len(probs) != g.num_players or any(
      p.shape != (g.num_actions,) for p in probs
  ):
    raise ValueError(
        f"{variant}: one distribution over {g.num_actions} actions per seat"
    )
  w = probs[0]
  for p in probs[1:]:
    w = np.multiply.outer(w, p)
  return np.tensordot(w, g.payoffs.astype(np.float64), axes=g.num_players)
