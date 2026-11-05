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

"""Duplicate seating designs.

Which seat each entrant holds on each replay of a deal.

A design replays one deal R times, permuting who sits where, so the luck of
the deal and the advantage of a seat cancel within the deal (the deal is the
statistical unit, eval.stats). Chance is keyed per event (core.chance), so
replaying a deal key replays its chance outcomes for any game.

  d = designs.make("cyclic", sides)              # or designs.default(sides)
  rows = designs.table_rows(d, lineups)          # i32[n, R, P] entrant per seat
  plan = designs.key_plan(d, duplicate=True)     # which rows share which keys

Lineups. A lineup lists one entrant per base seat (i32[P]; a side-level
lineup, one entrant per side, expands to its seats with `seat_lineup`). A
design holds B lineup blocks; most designs have one, `substitute` two (the
subject's block and the incumbent's).

Rows. Row r plays block blocks[r] with base seat q at table seat
seat_perms[r][q]. Every permutation maps whole sides onto whole sides (base
side k sits at table side side_perms[r][k]), so a base side position is one
entrant's position throughout: "position 0" is the subject of a design.

  design            rows  for
  swap2             2     two sides (every two-player game; bridge with one
                          entrant per partnership): identity, then the sides
                          swapped
  swap2+partner     2S    two sides of S seats each, entrants mixed within a
                          side (different partners): swap2 times every
                          cyclic shift of the seats within each side
  cyclic            T     T sides: row r puts base side k at side (k + r) % T
  cyclic+reversed   2T    cyclic, then the reflections k -> (r - k) % T, which
                          balance who sits after whom (all 3! = 6 orders at
                          T = 3)
  substitute        2L    the gate: block 0 (subject S plus field F) and block
                          1 (incumbent I plus the same F), each
                          cyclic+classes, so S and I each hold every side
                          position of their class; row r of block 1 shares
                          every policy key with row r of block 0 (L = T for
                          equal sides: two cyclic blocks)
  cyclic+classes    L     any sides, uneven included (2v1, 3v1v1): row r
                          shifts every side by r within its rotation class
                          (core.sides.Sides.rotation_classes: the sides of
                          one size), L = the lcm of the class sizes; equal
                          to cyclic on equal rotatable sides, the identity
                          (1 row) when no two sides are interchangeable

Uneven sides. Only equal-size sides are interchangeable: swap2, cyclic and
cyclic+reversed rotate every side and refuse sides of different sizes (or
sides that are not rotatable); a hand-built Design that maps a side onto a
side of another size is refused when built (_seat_perm, side_perms).

The j-th seat of side k maps to the j-th seat (or, for the within-side
shift of swap2+partner, the (j + shift) % S-th seat) of its image side,
seats of a side in increasing order. At two players every design's rows are
seat permutations of {0, 1}; swap2 is exactly the Evaluator's seat-swapped
duplicate pair (eval.payoff.duplicate_table).

Keys (`key_plan`, the shared-field key helper). With duplicate deals every
row of a deal shares the deal key; rows share policy keys exactly when the
design says so (key_rows): every row draws its own, except substitute's two
blocks, whose row r share all P seats' policy keys, so the field (and the
subject's seat) see the same randomness against S as against I and
self-against-self is exactly 0 even when sampling. duplicate=False gives
every row its own deal and its own policy keys (the unpaired estimator).
core.arena_engine.table_keys turns a plan into keys.

numpy only; no JAX.
"""

import dataclasses
from typing import Any, Sequence

import numpy as np

from open_spiel.python.scale.core import sides as sides_lib

#: Design names `make` builds; "auto" resolves to swap2 at two sides, cyclic
#: otherwise.
NAMES = (
    "swap2",
    "swap2+partner",
    "cyclic",
    "cyclic+reversed",
    "substitute",
    "cyclic+classes",
)


@dataclasses.dataclass(frozen=True)
class Design:
  """A duplicate seating design (module doc).

  Attributes:
    name: a NAMES name.
    sides: the table's core.sides.Sides.
    seat_perms: R tuples of len P; row r puts base seat q at table seat
      seat_perms[r][q].
    blocks: len R; the lineup block row r plays (0..B-1, each used).
    key_rows: len R; rows with equal key_rows share their policy keys
      (0..U-1, each used); rows sharing a key row have equal seat_perms.
  """

  name: str
  sides: sides_lib.Sides
  seat_perms: tuple[Any, ...]
  blocks: tuple[Any, ...]
  key_rows: tuple[Any, ...]

  def __post_init__(self):
    if not isinstance(self.sides, sides_lib.Sides):
      raise TypeError(
          f"design {self.name}: sides is a {type(self.sides).__name__}"
      )
    p = self.sides.num_players
    perms = tuple(tuple(int(s) for s in perm) for perm in self.seat_perms)
    blocks = tuple(int(b) for b in self.blocks)
    key_rows = tuple(int(k) for k in self.key_rows)
    r = len(perms)
    if r < 1 or len(blocks) != r or len(key_rows) != r:
      raise ValueError(
          f"design {self.name}: {r} seat_perms, {len(blocks)} blocks, "
          f"{len(key_rows)} key_rows (one each per row, at least one row)"
      )
    for perm in perms:
      if sorted(perm) != list(range(p)):
        raise ValueError(
            f"design {self.name}: {perm} is not a permutation of {p} seats"
        )
    for name, ids in (("blocks", blocks), ("key_rows", key_rows)):
      if sorted(set(ids)) != list(range(len(set(ids)))):
        raise ValueError(
            f"design {self.name}: {name} {ids} must use every id 0..max"
        )
    for k in set(key_rows):
      shared = {perms[i] for i in range(r) if key_rows[i] == k}
      if len(shared) != 1:
        raise ValueError(
            f"design {self.name}: rows sharing key row {k} have different seat "
            f"permutations {sorted(shared)}"
        )
    object.__setattr__(self, "seat_perms", perms)
    object.__setattr__(self, "blocks", blocks)
    object.__setattr__(self, "key_rows", key_rows)
    # validates that every permutation maps sides onto sides
    _ = self.side_perms

  @property
  def num_rows(self) -> int:
    """R, the rows per deal."""
    return len(self.seat_perms)

  @property
  def num_blocks(self) -> int:
    """B, the lineup blocks."""
    return max(self.blocks) + 1

  @property
  def side_perms(self) -> np.ndarray:
    """i32[R, T]: the table side base side k sits at on row r.

    Returns:
      The side permutations.

    Raises:
      ValueError: a row splits a side over two table sides.
    """
    s = self.sides
    out = np.zeros((self.num_rows, s.num_sides), np.int32)
    for r, perm in enumerate(self.seat_perms):
      for k in range(s.num_sides):
        img = {s.side(perm[q]) for q in s.seats(k)}
        if len(img) != 1:
          raise ValueError(
              f"design {self.name}: row {r} splits side {k} over sides "
              f"{sorted(img)}"
          )
        out[r, k] = img.pop()
      if sorted(out[r].tolist()) != list(range(s.num_sides)):
        raise ValueError(
            f"design {self.name}: row {r} does not permute the sides"
        )
    return out

  def rows_of(self, block: int) -> np.ndarray:
    """The rows of `block`, in order.

    Args:
      block: the lineup block.

    Returns:
      i64[R_b].

    Raises:
      ValueError: the design has no such block.
    """
    rows = np.flatnonzero(np.asarray(self.blocks) == int(block))
    if rows.size == 0:
      raise ValueError(
          f"design {self.name}: no block {block} (blocks "
          f"0..{self.num_blocks - 1})"
      )
    return rows


def _seat_perm(
    sides: sides_lib.Sides, side_perm: Sequence[int], shift: int = 0
) -> tuple[Any, ...]:
  """The seat permutation that sends side k to side side_perm[k].

  The j-th seat of a side goes to the ((j + shift) % size)-th seat of its
  image.

  Args:
    sides: the table's Sides.
    side_perm: side k's image, per side.
    shift: the rotation of the seats within each side.

  Returns:
    The seat permutation, a tuple of P seats.
  """
  perm = [0] * sides.num_players
  for k in range(sides.num_sides):
    src, dst = sides.seats(k), sides.seats(side_perm[k])
    if len(src) != len(dst):
      raise ValueError(
          f"side {k} ({len(src)} seats) cannot sit at side {side_perm[k]} "
          f"({len(dst)} seats)"
      )
    for j, q in enumerate(src):
      perm[q] = dst[(j + shift) % len(dst)]
  return tuple(perm)


def _need_rotatable(name: str, sides: sides_lib.Sides) -> None:
  if not sides.fully_rotatable:
    raise ValueError(
        f"design {name} rotates entrants over every side, which needs "
        f"rotatable sides of one size, got sizes {sides.side_sizes} (rotatable="
        f"{sides.rotatable}); uneven sides take cyclic+classes"
    )


def _class_perms(sides: sides_lib.Sides) -> list[Any]:
  """The side permutations of cyclic+classes.

  Row r sends side k to sides.rotate_side(k, r), r in 0..L-1.

  Args:
    sides: the table's Sides.

  Returns:
    One side permutation (a tuple of T sides) per row.
  """
  return [
      tuple(sides.rotate_side(k, r) for k in range(sides.num_sides))
      for r in range(sides.rotation_period)
  ]


def _single_block(name, sides, side_perms, shifts=None) -> Design:
  """A one-block design of the given side permutations.

  Args:
    name: The design's name.
    sides: The env's Sides.
    side_perms: One side permutation per row.
    shifts: Each row's seat shift within sides (default 0).

  Returns:
    The Design.
  """
  shifts = shifts or [0] * len(side_perms)
  perms = tuple(_seat_perm(sides, sp, sh) for sp, sh in zip(side_perms, shifts))
  r = len(perms)
  return Design(name, sides, perms, (0,) * r, tuple(range(r)))


def swap2(sides: sides_lib.Sides) -> Design:
  """Two sides: the identity, then the sides swapped (module doc).

  Args:
    sides: the table's Sides.

  Returns:
    The Design.

  Raises:
    ValueError: not two equal rotatable sides.
  """
  if sides.num_sides != 2:
    raise ValueError(
        f"design swap2 is for two sides, got {sides.num_sides} ({sides})"
    )
  _need_rotatable("swap2", sides)
  return _single_block("swap2", sides, [(0, 1), (1, 0)])


def swap2_partner(sides: sides_lib.Sides) -> Design:
  """Two sides of S >= 2 seats: swap2 times every within-side shift.

  2S rows.

  Args:
    sides: the table's Sides.

  Returns:
    The Design.

  Raises:
    ValueError: not two equal rotatable sides of at least 2 seats.
  """
  if sides.num_sides != 2:
    raise ValueError(
        f"design swap2+partner is for two sides, got {sides.num_sides}"
    )
  _need_rotatable("swap2+partner", sides)
  size = sides.side_sizes[0]
  if size < 2:
    raise ValueError(
        f"design swap2+partner needs sides of at least 2 seats, got {sides}"
    )
  sps, shifts = [], []
  for shift in range(size):
    for sp in ((0, 1), (1, 0)):
      sps.append(sp)
      shifts.append(shift)
  return _single_block("swap2+partner", sides, sps, shifts)


def _cyclic_perms(t: int) -> list[Any]:
  """The t cyclic rotations of range(t).

  Args:
    t: The number of sides.

  Returns:
    t permutations.
  """
  return [tuple((k + r) % t for k in range(t)) for r in range(t)]


def cyclic(sides: sides_lib.Sides) -> Design:
  """T rows: row r puts base side k at side (k + r) % T.

  Args:
    sides: the table's Sides (rotatable, of one size).

  Returns:
    The Design.
  """
  _need_rotatable("cyclic", sides)
  return _single_block("cyclic", sides, _cyclic_perms(sides.num_sides))


def cyclic_reversed(sides: sides_lib.Sides) -> Design:
  """2T rows: cyclic, then the reflections k -> (r - k) % T.

  Args:
    sides: the table's Sides (rotatable, of one size).

  Returns:
    The Design.
  """
  _need_rotatable("cyclic+reversed", sides)
  t = sides.num_sides
  refl = [tuple((r - k) % t for k in range(t)) for r in range(t)]
  return _single_block("cyclic+reversed", sides, _cyclic_perms(t) + refl)


def cyclic_classes(sides: sides_lib.Sides) -> Design:
  """L rows: row r shifts every side by r within its rotation class.

  Module doc.

  Args:
    sides: the table's Sides.

  Returns:
    The Design.
  """
  return _single_block("cyclic+classes", sides, _class_perms(sides))


def substitute(sides: sides_lib.Sides) -> Design:
  """The gate design: two cyclic+classes blocks whose row r share policy keys.

  Module doc; two cyclic blocks on equal rotatable sides.

  Args:
    sides: the table's Sides.

  Returns:
    The Design.

  Raises:
    ValueError: sides that are not rotatable.
  """
  if not sides.rotatable:
    raise ValueError(
        "design substitute rotates the subject over its side class, which "
        f"needs rotatable sides, got {sides}"
    )
  perms = tuple(_seat_perm(sides, sp) for sp in _class_perms(sides))
  r = len(perms)
  return Design(
      "substitute",
      sides,
      perms + perms,
      (0,) * r + (1,) * r,
      tuple(range(r)) * 2,
  )


_BUILDERS = {
    "swap2": swap2,
    "swap2+partner": swap2_partner,
    "cyclic": cyclic,
    "cyclic+reversed": cyclic_reversed,
    "substitute": substitute,
    "cyclic+classes": cyclic_classes,
}
assert tuple(_BUILDERS) == NAMES


def default(sides: sides_lib.Sides) -> Design:
  """The default design for `sides`.

  swap2 at two equal rotatable sides, cyclic at more, cyclic+classes for
  uneven or non-rotatable sides.

  Args:
    sides: the table's Sides.

  Returns:
    The Design.
  """
  if not sides.fully_rotatable:
    return cyclic_classes(sides)
  return swap2(sides) if sides.num_sides == 2 else cyclic(sides)


def make(name: str, sides: sides_lib.Sides) -> Design:
  """The design `name` (a NAMES name, or "auto" for `default`) for `sides`.

  Args:
    name: the design name.
    sides: the table's Sides.

  Returns:
    The Design.

  Raises:
    ValueError: an unknown name, or a design not valid for these sides.
  """
  if name == "auto":
    return default(sides)
  if name not in _BUILDERS:
    raise ValueError(
        f"unknown design {name!r}; known: auto, {', '.join(NAMES)}"
    )
  return _BUILDERS[name](sides)


# -- lineups and rows ----------------------------------------------------------
def seat_lineup(sides: sides_lib.Sides, side_lineup) -> np.ndarray:
  """A seat-level lineup from a side-level one.

  Every seat of side k holds side k's entrant.

  Args:
    sides: the table's Sides.
    side_lineup: i32[..., T], an entrant per side.

  Returns:
    i32[..., P].

  Raises:
    ValueError: the last axis is not one entrant per side.
  """
  side_lineup = np.asarray(side_lineup, np.int32)
  if side_lineup.ndim < 1 or side_lineup.shape[-1] != sides.num_sides:
    raise ValueError(
        f"seat_lineup: want [..., {sides.num_sides}] entrants per side, got "
        f"{side_lineup.shape}"
    )
  return side_lineup[..., np.asarray(sides.side_of_seat)]


def table_rows(design: Design, lineups) -> np.ndarray:
  """The entrant at each table seat of each row of each deal.

  From base-seat lineups i32[n, B, P] (block b's lineup on deal i).

  Args:
    design: the Design.
    lineups: i32[n, B, P].

  Returns:
    i32[n, R, P].

  Raises:
    ValueError: lineups of the wrong shape.
  """
  lineups = np.asarray(lineups, np.int32)
  n_b_p = (design.num_blocks, design.sides.num_players)
  if lineups.ndim != 3 or lineups.shape[1:] != n_b_p:
    raise ValueError(
        f"design {design.name}: lineups must be [n, {n_b_p[0]}, {n_b_p[1]}], "
        f"got {lineups.shape}"
    )
  n, r, p = lineups.shape[0], design.num_rows, design.sides.num_players
  out = np.zeros((n, r, p), np.int32)
  for row, (perm, b) in enumerate(zip(design.seat_perms, design.blocks)):
    out[:, row, list(perm)] = lineups[:, b, :]
  return out


@dataclasses.dataclass(frozen=True)
class KeyPlan:
  """Which keys each row of a deal uses (module doc, "Keys").

  Attributes:
    deal_index: i32[R]; row r uses deal key deal_index[r] of its deal's
      num_deal_keys.
    policy_index: i32[R]; row r uses policy-key row policy_index[r] (P keys)
      of its deal's num_policy_rows.
  """

  deal_index: tuple[Any, ...]
  policy_index: tuple[Any, ...]

  @property
  def num_deal_keys(self) -> int:
    """The deal keys per deal."""
    return max(self.deal_index) + 1

  @property
  def num_policy_rows(self) -> int:
    """The policy-key rows per deal."""
    return max(self.policy_index) + 1


def key_plan(design: Design, duplicate: bool) -> KeyPlan:
  """The shared-field key helper.

  Duplicate deals share the deal key over every row and policy keys per
  design.key_rows; independent deals share nothing.

  Args:
    design: the Design.
    duplicate: whether the rows of a deal share its deal key.

  Returns:
    The KeyPlan.
  """
  r = design.num_rows
  if duplicate:
    return KeyPlan(deal_index=(0,) * r, policy_index=design.key_rows)
  return KeyPlan(deal_index=tuple(range(r)), policy_index=tuple(range(r)))
