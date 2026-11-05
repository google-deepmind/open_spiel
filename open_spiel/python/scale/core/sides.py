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

"""Sides: which seats of a table one entrant controls, for any player count.

A side is the set of seats one entrant (a league member, a learner) holds at
one table. Two-player games have two sides of one seat each; free-for-all
games (kuhn with N players, hearts) have one side per seat; partnership games
have teams (bridge: North-South and East-West).

  sides = sides_of(env)                   # env.sides, else one side per seat
  sides.num_sides, sides.seats(0)         # T, the seats of side 0
  sides.side_returns(returns)             # [..., P] seat returns -> [..., T]

  Sides.individual(3, constant_sum=True)          # kuhn(3): {0} {1} {2}
  Sides.partnerships(4, 2, constant_sum=True)     # bridge: {0, 2} {1, 3}

Side return. A side's return is the mean of its seats' returns, not the sum,
so a partnership whose seats score the same (bridge) has the seat's return as
its side return, and side returns of different side sizes are on one scale.

Identity. Sides are not part of a game's rules identity: an env declares them
as an attribute (`sides`), never in its EnvSpec, so declaring them changes no
rules_sha or artifact. An env without the attribute has one side per seat,
which is every two-player game's case (T = 2, sides {0} and {1}).

Pair scorer. A duplicate pair plays one deal twice with the entrant holding
each side once (two sides). `pair_scorer_of(env)` gives the per-deal score of
an entrant from its side returns on the two rows: by default their mean,
which is what the two-player Evaluator reports per deal pair; a game whose
match score is a nonlinear function of the two tables declares its own
(bridge: IMPs of the point difference, envs.bridge.imps_pair_score).

Lineups. A Lineup says who fills each side of a training table: the learner,
self-play copies, a draw from the learner's opponent mixture, or one fixed
member, for the whole side or seat by seat (mixed partnerships: learner +
frozen partner against a pool pair). It is declarative (a config value); the
league's sampler builds slot tables from it. LeagueConfig.lineup holds one.

This module imports no array framework at import time (numpy only);
side_returns works on numpy and JAX arrays alike.
"""

import dataclasses
from typing import Any, Callable, Optional, Sequence

import numpy as np

from open_spiel.python.scale.core import hooks

PairScorer = Callable[[Any, Any], np.ndarray]


@dataclasses.dataclass(frozen=True)
class Sides:
  """The partition of a table's P seats into T sides (module doc).

  Attributes:
    side_of_seat: len P; side ids 0..T-1, numbered in order of first
      appearance (seat 0 is on side 0).
    constant_sum: the sum of the side returns is the same on every terminal
      (zero-sum games, and constant-sum ones).
    rotatable: sides of the same size are interchangeable seat groups, so a
      design or lineup that rotates entrants over them balances seat
      advantage (`rotation_classes`); False: no side is interchangeable
      (every side keeps its entrant). Uneven sides (2v1, 3v1v1) may be
      rotatable: only equal-size sides are ever permuted.
  """

  side_of_seat: tuple[Any, ...]
  constant_sum: bool
  rotatable: bool = True

  def __post_init__(self):
    sos = tuple(self.side_of_seat)
    if len(sos) < 2:
      raise ValueError(f"Sides: a table has at least 2 seats, got {sos}")
    for s in sos:
      if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
        raise TypeError(f"Sides: side ids are ints, got {s!r} in {sos}")
    sos = tuple(int(s) for s in sos)
    seen = []
    for s in sos:
      if s not in seen:
        if s != len(seen):
          raise ValueError(
              "Sides: side ids must be numbered 0..T-1 in order of first "
              f"appearance, got {sos}"
          )
        seen.append(s)
    if len(seen) < 2:
      raise ValueError(f"Sides: a table has at least 2 sides, got {sos}")
    if not isinstance(self.constant_sum, (bool, np.bool_)):
      raise TypeError(
          f"Sides: constant_sum is a bool, got {self.constant_sum!r}"
      )
    if not isinstance(self.rotatable, (bool, np.bool_)):
      raise TypeError(f"Sides: rotatable is a bool, got {self.rotatable!r}")
    object.__setattr__(self, "side_of_seat", sos)
    object.__setattr__(self, "constant_sum", bool(self.constant_sum))
    object.__setattr__(self, "rotatable", bool(self.rotatable))

  # -- shape -------------------------------------------------------------------
  @property
  def num_players(self) -> int:
    """P, the seats at the table."""
    return len(self.side_of_seat)

  @property
  def num_sides(self) -> int:
    """T, the sides at the table."""
    return max(self.side_of_seat) + 1

  @property
  def side_sizes(self) -> tuple[Any, ...]:
    """Seats per side, by side id."""
    return tuple(self.side_of_seat.count(s) for s in range(self.num_sides))

  def seats(self, side: int) -> tuple[Any, ...]:
    """The seats of `side`, in increasing order.

    Args:
      side: the side id.

    Returns:
      A tuple of seats.

    Raises:
      ValueError: side is out of range.
    """
    side = int(side)
    if not 0 <= side < self.num_sides:
      raise ValueError(f"side {side} not in 0..{self.num_sides - 1}")
    return tuple(p for p, s in enumerate(self.side_of_seat) if s == side)

  def side(self, seat: int) -> int:
    """The side of `seat`.

    Args:
      seat: the seat.

    Returns:
      The side id.

    Raises:
      ValueError: seat is out of range.
    """
    seat = int(seat)
    if not 0 <= seat < self.num_players:
      raise ValueError(f"seat {seat} not in 0..{self.num_players - 1}")
    return self.side_of_seat[seat]

  @property
  def equal_sizes(self) -> bool:
    """Every side holds the same number of seats."""
    return len(set(self.side_sizes)) == 1

  @property
  def fully_rotatable(self) -> bool:
    """Every side is interchangeable with every other.

    Rotatable, equal sizes (the sides every whole-table rotation design, swap2
    and cyclic, needs).
    """
    return self.rotatable and self.equal_sizes

  @property
  def rotation_classes(self) -> tuple[Any, ...]:
    """The interchangeable side classes.

    Tuples of side ids, each sorted, in order of their first side. Rotatable
    sides: one class per side size (all the sides when the sizes are equal); not
    rotatable: one class per side (nothing moves).
    """
    if not self.rotatable:
      return tuple((s,) for s in range(self.num_sides))
    sizes = self.side_sizes
    out, seen = [], set()
    for s in range(self.num_sides):
      if sizes[s] not in seen:
        seen.add(sizes[s])
        out.append(
            tuple(k for k in range(self.num_sides) if sizes[k] == sizes[s])
        )
    return tuple(out)

  @property
  def rotation_period(self) -> int:
    """L, the least common multiple of the rotation class sizes.

    A rotation that shifts every class by r on row r repeats with period L (T
    for fully rotatable sides, 1 when nothing moves).
    """
    return int(np.lcm.reduce([len(c) for c in self.rotation_classes]))

  def rotate_side(self, side: int, shift: int) -> int:
    """The side `side` moves to under a shift within its rotation class.

    Class (c_0, ..., c_{n-1}) sends c_i to c_{(i + shift) % n}.

    Args:
      side: the side id.
      shift: the shift.

    Returns:
      The side id it moves to.

    Raises:
      ValueError: side is out of range.
    """
    side = int(side)
    for c in self.rotation_classes:
      if side in c:
        return c[(c.index(side) + int(shift)) % len(c)]
    raise ValueError(f"side {side} not in 0..{self.num_sides - 1}")

  @property
  def one_seat_per_side(self) -> bool:
    """Every side holds one seat (two-player and free-for-all games)."""
    return self.num_sides == self.num_players

  @property
  def is_two_player(self) -> bool:
    """Two sides of one seat each.

    The two-player case every two-player path (head-to-head Evaluator, payoff
    matrices, the promotion gate) handles.
    """
    return self.num_players == 2

  # -- returns -----------------------------------------------------------------
  def side_returns(self, returns):
    """Side returns [..., T] from seat returns [..., P].

    Each side's mean over its seats, summed in seat order then divided by the
    side's size. Works on numpy arrays (and anything np.asarray takes) and on
    JAX arrays (the result is a JAX array; jit-safe).

    Args:
      returns: seat returns [..., P].

    Returns:
      Side returns [..., T].

    Raises:
      ValueError: the last axis is not one return per seat.
    """
    if _is_jax(returns):
      # Lazy: jax is optional here (numpy returns need no jax).
      import jax.numpy as xp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415
    else:
      xp = np
      returns = np.asarray(returns)
    if returns.ndim < 1 or returns.shape[-1] != self.num_players:
      raise ValueError(
          f"side_returns: want [..., {self.num_players}] seat returns, got "
          f"shape {tuple(returns.shape)}"
      )
    cols = []
    for s in range(self.num_sides):
      seats = self.seats(s)
      total = returns[..., seats[0]]
      for p in seats[1:]:
        total = total + returns[..., p]
      cols.append(total / len(seats) if len(seats) > 1 else total)
    return xp.stack(cols, axis=-1)

  # -- constructors ------------------------------------------------------------
  @staticmethod
  def individual(num_players: int, constant_sum: bool) -> "Sides":
    """One side per seat (two-player and free-for-all games).

    Args:
      num_players: the seats.
      constant_sum: whether the returns sum to a constant.

    Returns:
      The Sides.
    """
    p = _players(num_players)
    return Sides(tuple(range(p)), bool(constant_sum), True)

  @staticmethod
  def partnerships(
      num_players: int, num_sides: int, constant_sum: bool
  ) -> "Sides":
    """num_sides equal partnerships, side = seat % num_sides (bridge: 4, 2).

    Args:
      num_players: the seats.
      num_sides: the partnerships.
      constant_sum: whether the side returns sum to a constant.

    Returns:
      The Sides.

    Raises:
      ValueError: the seats do not split into num_sides >= 2 equal sides.
    """
    p, t = _players(num_players), int(num_sides)
    if t < 2 or p % t:
      raise ValueError(
          f"partnerships: {p} seats do not split into {t} equal sides (t >= 2)"
      )
    return Sides(tuple(s % t for s in range(p)), bool(constant_sum), True)


def _players(num_players: int) -> int:
  """`num_players` as an int, checked.

  Args:
    num_players: The player count.

  Returns:
    The count.

  Raises:
    TypeError: Not an int.
    ValueError: Fewer than two players.
  """
  if isinstance(num_players, bool) or not isinstance(
      num_players, (int, np.integer)
  ):
    raise TypeError(f"num_players is an int, got {num_players!r}")
  if int(num_players) < 2:
    raise ValueError(f"num_players must be >= 2, got {num_players}")
  return int(num_players)


def _is_jax(x) -> bool:
  """Whether `x` is a JAX array.

  Args:
    x: A value.

  Returns:
    True for a jax or jaxlib type.
  """
  return type(x).__module__.startswith(("jax", "jaxlib"))


#: The optional hooks an env declares beyond its EnvSpec (core.hooks): read
#: through sides_of and pair_scorer_of only.
ENV_HOOKS = hooks.Protocol(
    "env",
    [
        hooks.value(
            "sides",
            "a core.sides.Sides of env.num_players seats; absent: one side "
            "per seat (sides_of)",
        ),
        hooks.method(
            "pair_scorer",
            "pair_scorer(r0, r1) -> f64 per-deal score of an entrant's two "
            "duplicate rows; absent: mean_pair_score (pair_scorer_of)",
        ),
    ],
)


def sides_of(env) -> Sides:
  """Env's Sides.

  Its `sides` attribute, else one side per seat, constant-sum when the env is
  zero-sum (`env.zero_sum`).

  Args:
    env: the env.

  Returns:
    The Sides.

  Raises:
    TypeError: env.sides is not a Sides.
    ValueError: env.sides does not have env.num_players seats, or a zero-sum
      env declares sides that are not constant-sum.
  """
  p = int(env.num_players)
  s = ENV_HOOKS.get(env, "sides")
  if s is None:
    return Sides.individual(p, bool(env.zero_sum))
  if not isinstance(s, Sides):
    raise TypeError(
        f"{getattr(env, 'name', env)}: env.sides is a {type(s).__name__}, "
        "not a core.sides.Sides"
    )
  if s.num_players != p:
    raise ValueError(
        f"{getattr(env, 'name', env)}: env.sides has {s.num_players} seats, "
        f"the env {p}"
    )
  if not s.constant_sum and bool(getattr(env, "zero_sum", False)):
    raise ValueError(
        f"{getattr(env, 'name', env)}: a zero-sum env declares sides that are "
        "not constant-sum"
    )
  return s


# -- pair scores ---------------------------------------------------------------
def mean_pair_score(r0, r1) -> np.ndarray:
  """The default pair score: the mean of an entrant's two duplicate rows.

  The side returns on the two rows of a duplicate pair, (r0 + r1) / 2 in f64
  (bit for bit the two-player Evaluator's per-deal-pair mean,
  a_payoff.mean(1)).

  Args:
    r0: the side returns on row 0, per deal.
    r1: the side returns on row 1, per deal.

  Returns:
    f64 per-deal scores.

  Raises:
    ValueError: the rows' shapes differ.
  """
  r0, r1 = np.asarray(r0, np.float64), np.asarray(r1, np.float64)
  if r0.shape != r1.shape:
    raise ValueError(f"pair score: row shapes differ {r0.shape} vs {r1.shape}")
  return np.stack([r0, r1], axis=-1).mean(-1)


def pair_scorer_of(env) -> PairScorer:
  """Env's pair scorer, else mean_pair_score.

  Its `pair_scorer` attribute, a function (r0, r1) -> f64 per-deal score.

  Args:
    env: the env.

  Returns:
    The pair scorer.

  Raises:
    TypeError: env.pair_scorer is not callable.
  """
  fn = ENV_HOOKS.get(env, "pair_scorer")
  return mean_pair_score if fn is None else fn


# -- lineups -------------------------------------------------------------------
#: SideFill kinds: "learner" (the learning slot; its seats are perspective
#: seats), "self" (the learner's current params, not trained on), "pool" (a
#: member drawn from the learner's opponent mixture, independently per row
#: and per side), "fixed" (one named member).
FILL_KINDS = ("learner", "self", "pool", "fixed")
#: Lineup rotations: "cyclic" (row r shifts every fill position by r within
#: its side's rotation class, Sides.rotate_side: the learner's position cycles
#: over the sides interchangeable with its own), "none" (positions as listed).
ROTATIONS = ("cyclic", "none")


@dataclasses.dataclass(frozen=True)
class SideFill:
  """Who fills one side of a table.

  Attributes:
    kind: a FILL_KINDS name.
    learner: the learning slot, for kind "learner" (0 for every other kind).
      Only 0 is seated (league.sampler.table_from_lineup refuses any other,
      on purpose): 0 means "the learner this batch trains", whichever
      learning slot that is, so every learner of a league already plays
      every learner position of the lineup. A second index would name a
      second learner at the same table, and that is not one batch: (1) the
      Arena emits trajectories for one perspective slot (core.arena
      Arena.perspective_slot, the learner's slot 0 in every league kind),
      so a second learner's seats would need a perspective slot per column
      and a learner update that splits one batch between two learner states;
      (2) seating the other learner's current params without training them
      would put live learner params in an Arena position at every update,
      which the league's residency (only pool members and fixed opponents
      hold Arena positions) does not do; (3) learning slots are named and
      spawned and retired by the controller, so an integer has no stable
      referent. Remapping index k to slot 0 would only relabel 0. Seat
      another learner as a "fixed" member (a published snapshot), or train
      it in its own batch, where it is the learner.
    member: the member's name, for kind "fixed" ("" for every other kind).
  """

  kind: str
  learner: int = 0
  member: str = ""

  def __post_init__(self):
    if self.kind not in FILL_KINDS:
      raise ValueError(f"SideFill kind {self.kind!r} not in {FILL_KINDS}")
    if (
        isinstance(self.learner, bool)
        or not isinstance(self.learner, int)
        or self.learner < 0
    ):
      raise ValueError(
          f"SideFill learner must be an int >= 0, got {self.learner!r}"
      )
    if self.kind != "learner" and self.learner != 0:
      raise ValueError(
          f"SideFill {self.kind!r}: learner is only for kind 'learner'"
      )
    if not isinstance(self.member, str):
      raise TypeError(f"SideFill member is a str, got {self.member!r}")
    if (self.kind == "fixed") != bool(self.member):
      raise ValueError(
          f"SideFill {self.kind!r}: a member name is given exactly for kind "
          f"'fixed', got {self.member!r}"
      )


@dataclasses.dataclass(frozen=True)
class Lineup:
  """Who fills each side position of a table, and how the positions rotate.

  A position's fill is one SideFill for the whole side (every seat of it
  holds that one member; a "pool" side draws one member for all its seats),
  or a tuple of SideFill, one per seat of the side in increasing seat order
  (a mixed side: learner + frozen partner, two independently drawn pool
  members, ...). A per-seat fill fits only a side of its size, which a
  rotation preserves (only equal-size sides are interchangeable).

  Attributes:
    fills: len T; each a SideFill or a tuple of SideFill (per seat); at
      least one "learner" seat in all.
    rotate: a ROTATIONS name.
  """

  fills: tuple[Any, ...]
  rotate: str = "cyclic"

  def __post_init__(self):
    fills = []
    for f in tuple(self.fills):
      if isinstance(f, (tuple, list)):
        f = tuple(f)
        if not f:
          raise ValueError("a Lineup per-seat fill names at least one seat")
        for g in f:
          if not isinstance(g, SideFill):
            raise TypeError(
                f"Lineup per-seat fills are SideFill, got {type(g).__name__}"
            )
      elif not isinstance(f, SideFill):
        raise TypeError(
            "Lineup fills are SideFill or tuples of SideFill, got "
            f"{type(f).__name__}"
        )
      fills.append(f)
    fills = tuple(fills)
    if len(fills) < 2:
      raise ValueError(f"a Lineup fills at least 2 sides, got {len(fills)}")
    object.__setattr__(self, "fills", fills)
    if not any(f.kind == "learner" for f in self.units):
      raise ValueError("a Lineup has at least one 'learner' side or seat")
    if self.rotate not in ROTATIONS:
      raise ValueError(f"Lineup rotate {self.rotate!r} not in {ROTATIONS}")

  @property
  def units(self) -> tuple[Any, ...]:
    """Every SideFill of the lineup in position order.

    Per-seat fills flattened.
    """
    out = []
    for f in self.fills:
      out.extend(f if isinstance(f, tuple) else (f,))
    return tuple(out)

  def check(self, sides: Sides) -> None:
    """Raises unless the lineup fits `sides`.

    One fill per side of `sides`, every per-seat fill has one SideFill per seat
    of its side, and a rotating lineup has rotatable sides.

    Args:
      sides: the table's Sides.

    Raises:
      ValueError: the lineup does not fit the sides.
    """
    if len(self.fills) != sides.num_sides:
      raise ValueError(
          f"Lineup has {len(self.fills)} fills for {sides.num_sides} sides"
      )
    for q, f in enumerate(self.fills):
      if isinstance(f, tuple) and len(f) != sides.side_sizes[q]:
        raise ValueError(
            f"Lineup position {q}: a per-seat fill of {len(f)} seats for a "
            f"side of {sides.side_sizes[q]} seats ({sides.seats(q)})"
        )
    if self.rotate == "cyclic" and not sides.rotatable:
      raise ValueError(
          f"Lineup rotate='cyclic' needs rotatable sides, got {sides}"
      )


def lineup(
    kinds: Sequence[Any],
    rotate: str = "cyclic",
    members: Optional[Sequence[str]] = None,
) -> Lineup:
  """A Lineup from fill kinds.

  For example lineup(("learner", "pool", "self", "pool")); an entry that is a
  sequence of kinds is a per-seat fill, e.g.
  lineup((("learner", "fixed"), "pool"), members=("partner@3",)) (learner and
  a frozen partner against a pool pair).

  Args:
    kinds: the fill kind of each side (a str) or of each seat of a side (a
      sequence of str).
    rotate: a ROTATIONS name.
    members: names the member of each "fixed" fill (side or seat), in order.

  Returns:
    The Lineup.

  Raises:
    ValueError: a "fixed" fill without a member name, or member names left
      over.
  """
  members = list(members or ())

  def one(k):
    """The SideFill of one kind.

    Args:
      k: the fill kind.

    Returns:
      The SideFill.

    Raises:
      TypeError: k is not a str.
      ValueError: a "fixed" fill without a member name.
    """
    if not isinstance(k, str):
      raise TypeError(f"lineup: a fill kind is a str, got {k!r}")
    if k == "fixed":
      if not members:
        raise ValueError("lineup: a 'fixed' fill without a member name")
      return SideFill("fixed", member=members.pop(0))
    return SideFill(k)

  fills = []
  for k in kinds:
    fills.append(
        tuple(one(g) for g in k) if isinstance(k, (tuple, list)) else one(k)
    )
  if members:
    raise ValueError(
        f"lineup: member names {members} left over (no 'fixed' fill for them)"
    )
  return Lineup(tuple(fills), rotate)
