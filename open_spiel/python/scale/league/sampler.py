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

"""Opponent mixes and the slot-table draw: who each learner plays.

A learner's opponent distribution is a `Mix`: a self-play mass (rows where the
learner sits in every seat) plus a mass per member name. Members live in
`Pool`s, one Arena slot each (one architecture, one decision rule) with a
fixed number K of positions; changing who is in a pool, or the weights,
changes values, never shapes, so the Arena compiles once.

    mix = sampler.pfsp(["a@10", "b@20"], winrates=[0.7, 0.4])
    draw = sampler.sample(sampler.LeagueSlots(num_players=2, batch=256,
                                              pools=pools, mix=mix), key)

Rules (pure numpy, each returns a Mix):
  uniform        equal mass over names.
  pfsp           prioritized fictitious self-play (Vinyals et al. 2019): mass
                 f(p) on each member from its win-rate p against the learner
                 (p, p^2 or p(1 - p)), or from a return through a logistic.
  pmix           self-play, lambda over active members by weight, epsilon
                 uniform over audit members, archive_epsilon uniform over
                 archived ones; an empty band folds into self-play.
  lagged_audit   self-play, a lagged snapshot of the learner, and a uniform
                 audit pool; a missing snapshot or empty pool folds into
                 self-play.
  snapshot_pool  self_weight on the learner's lagged snapshot, the rest
                 uniform over the seated pool, no self-play rows.
  meta           a meta-solver strategy over named members (PSRO).
  pipeline       (1 - share) meta-strategy + share uniform over the lower
                 pipeline workers (pipeline PSRO).

`sample(state, key)` returns the Draw (table, pool params, counts).
`table_from_mix` is the two-player row allocation; `entries_table` draws a
weighted (slot, member) pool with it. The learner is slot 0 member 0 in seat
row % P (the perspective seat); an opponent row seats the drawn member in
every other seat. Stratified draws (the default) allocate rows per seat class
by largest remainder over the categories ["self", then names sorted] and
permute them with `key`, so counts are exact and balanced per seat; iid draws
are categorical per row. Pool params pass through untouched.

Sides (any player count). A table's seats split into sides (core.sides): a
member is drawn per side, never per table, so no opponent fills the learner's
partner seat (bridge) and the other players of a free-for-all are drawn
independently (hearts, kuhn with N players).

  table_from_lineup  the table of a core.sides.Lineup: the learner's sides
                     (every seat of them a perspective seat), "self" sides
                     (the learner's params, not trained on), "fixed" sides
                     (one member) and "pool" sides, each pool side drawing
                     its own member from the Mix (stratified per learner
                     position, or iid), independently of the other sides. A
                     position may instead be filled seat by seat (a mixed
                     side: learner + fixed partner, two pool draws, ...).
                     Rotation shifts positions within the sides' rotation
                     classes only (equal-size sides; core.sides), so uneven
                     sides (2v1, 3v1v1) are seated too.
  lineup_from_mix    the Lineup a Mix is played through at P > 2: the learner
                     on one side, rotating over the sides of its size by row,
                     every other side a pool draw. Two-player tables are
                     table_from_mix's.
  draw_table         the one dispatch the league's draw uses: pure self-play
                     on every seat (self_play_seats="all"), else a Lineup,
                     else table_from_mix at P = 2, else lineup_from_mix.

The perspective width M (seats emitted per row) is the same on every row of a
lineup table, and on every table of one Lineup and Sides: the size of the
learner's sides (1 at P = 2 and in free-for-all games, where the table keeps
today's [B] perspective; 2 for a bridge partnership), or P for pure
self-play on every seat. An Arena's trajectory batch keeps one shape.

Keys are `core.rng` keys: a jax key and a numpy key with the same data draw
the same table, so this module imports no array framework.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import slots as slots_lib
from open_spiel.python.scale.core import types

SELF = "self"
F64 = np.float64
MASS_TOL = 1e-9

PFSP_MODES = ("linear", "squared", "variance")
DEFAULT_PFSP_RETURN_SCALE = 5.0


# -- Mix -----------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Mix:
  """self_play + sum(weights.values()) == 1; every mass finite and >= 0.

  Attributes:
    self_play: the self-play mass.
    weights: member name -> mass.
  """

  self_play: float
  weights: Mapping[str, float]

  def __post_init__(self):
    sp = float(self.self_play)
    w = {str(k): float(v) for k, v in dict(self.weights).items()}
    if SELF in w:
      raise ValueError(
          f"no member may be named {SELF!r} (it is the self-play category)"
      )
    vals = [sp] + list(w.values())
    if not all(math.isfinite(v) and v >= 0.0 for v in vals):
      raise ValueError(f"Mix masses must be finite and >= 0: self {sp}, {w}")
    if abs(sum(vals) - 1.0) > MASS_TOL:
      raise ValueError(f"Mix masses sum to {sum(vals)}, not 1: self {sp}, {w}")
    object.__setattr__(self, "self_play", sp)
    object.__setattr__(self, "weights", w)

  def support(self) -> tuple[str, ...]:
    """Names with positive mass, sorted.

    Returns:
      The names.
    """
    return tuple(sorted(n for n, v in self.weights.items() if v > 0.0))

  def categories(self) -> tuple[tuple[str, float], ...]:
    """(name, mass) in draw order: self first, then names sorted.

    Returns:
      The categories.
    """
    return ((SELF, self.self_play),) + tuple(
        (n, self.weights[n]) for n in sorted(self.weights)
    )


def _fold(weights: Mapping[str, float]) -> Mix:
  """A Mix whose self-play mass is whatever the member masses leave.

  Args:
    weights: The member masses.

  Returns:
    The Mix.

  Raises:
    ValueError: The member masses exceed 1.
  """
  total = float(sum(weights.values()))
  if total > 1.0 + MASS_TOL:
    raise ValueError(f"member masses sum to {total} > 1: {dict(weights)}")
  return Mix(max(0.0, 1.0 - total), {k: float(v) for k, v in weights.items()})


def _check_unique(names: Sequence[str]) -> list[str]:
  """Checks the member names are unique.

  Args:
    names: The member names.

  Returns:
    The names as strings.

  Raises:
    ValueError: A name repeats.
  """
  names = [str(n) for n in names]
  if len(set(names)) != len(names):
    raise ValueError(f"repeated member names: {names}")
  return names


# -- rules ---------------------------------------------------------------------
def uniform(names: Sequence[str], self_play: float = 0.0) -> Mix:
  """self_play on self-play, the rest uniform over names.

  Args:
    names: the member names.
    self_play: the self-play mass.

  Returns:
    The Mix.
  """
  names = _check_unique(names)
  assert 0.0 <= self_play <= 1.0, self_play
  if not names:
    return Mix(1.0, {})
  each = (1.0 - self_play) / len(names)
  return _fold({n: each for n in names})


def pfsp_weights(
    winrates: Sequence[float], mode: str = "squared", floor: float = 0.0
) -> np.ndarray:
  """PFSP weights: f(p) = p, p^2 or p(1 - p) by mode, plus floor, normalised.

  Uniform when all are zero.

  Args:
    winrates: probabilities in [0, 1].
    mode: one of PFSP_MODES.
    floor: added to every weight, finite and >= 0.

  Returns:
    The weights, summing to 1.

  Raises:
    ValueError: no win-rates, an unknown mode, a bad floor or a win-rate
      outside [0, 1].
  """
  if not np.size(winrates):
    raise ValueError("pfsp_weights needs at least one win-rate")
  if mode not in PFSP_MODES:
    raise ValueError(
        f"unknown PFSP mode {mode!r}; expected one of {PFSP_MODES}"
    )
  if not (math.isfinite(floor) and floor >= 0.0):
    raise ValueError(f"PFSP floor must be finite and >= 0, got {floor}")
  p = np.asarray(winrates, F64)
  if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
    raise ValueError(
        f"win-rates must be probabilities in [0, 1], got {list(winrates)}"
    )
  mass = {"linear": p, "squared": p**2, "variance": p * (1.0 - p)}[mode] + floor
  total = mass.sum()
  return np.full(p.size, 1.0 / p.size) if total <= 0.0 else mass / total


def returns_to_pfsp_priorities(
    returns: Sequence[float], scale: float = DEFAULT_PFSP_RETURN_SCALE
) -> np.ndarray:
  """Returns as PFSP priorities: logistic(return / scale), in (0, 1).

  Args:
    returns: the members' returns.
    scale: the logistic's scale, finite and > 0.

  Returns:
    The priorities.

  Raises:
    ValueError: no returns, a bad scale or a non-finite return.
  """
  if not np.size(returns):
    raise ValueError("returns_to_pfsp_priorities needs at least one return")
  if not (math.isfinite(scale) and scale > 0.0):
    raise ValueError(f"PFSP return scale must be finite and > 0, got {scale}")
  r = np.asarray(returns, F64)
  if not np.all(np.isfinite(r)):
    raise ValueError(f"returns must be finite, got {list(returns)}")
  return 0.5 * (1.0 + np.tanh(0.5 * r / scale))


def pfsp(
    names: Sequence[str],
    winrates: Optional[Sequence[float]] = None,
    returns: Optional[Sequence[float]] = None,
    mode: str = "squared",
    floor: float = 0.0,
    scale: float = DEFAULT_PFSP_RETURN_SCALE,
    self_play: float = 0.0,
) -> Mix:
  """PFSP over names from each member's win-rate or return vs the learner.

  Members that beat the learner get more mass.

  Args:
    names: the member names.
    winrates: each member's win-rate against the learner.
    returns: each member's return against the learner.
    mode: one of PFSP_MODES.
    floor: added to every weight.
    scale: the logistic's scale for returns.
    self_play: the self-play mass.

  Returns:
    The Mix.

  Raises:
    ValueError: not exactly one of winrates and returns.
  """
  names = _check_unique(names)
  if (winrates is None) == (returns is None):
    raise ValueError("pfsp takes exactly one of winrates or returns")
  pri = (
      np.asarray(winrates, F64)
      if returns is None
      else returns_to_pfsp_priorities(returns, scale)
  )
  assert pri.shape == (len(names),), (pri.shape, names)
  if not names:
    return Mix(1.0, {})
  w = pfsp_weights(pri, mode, floor) * (1.0 - self_play)
  return _fold(dict(zip(names, w.tolist())))


def pmix(
    active: Mapping[str, float],
    lambda_t: float,
    epsilon: float,
    audit: Sequence[str],
    archived: Sequence[str] = (),
    archive_epsilon: float = 0.0,
) -> Mix:
  """Self-play, lambda_t over active, epsilon over audit, the rest archived.

  (1 - lambda_t - epsilon - archive_epsilon) self-play, lambda_t over
  `active` (weights within the active set, summing to 1), epsilon uniform
  over `audit`, archive_epsilon uniform over `archived`.

  Args:
    active: the active members' weights.
    lambda_t: the active mass.
    epsilon: the audit mass.
    audit: the audit members.
    archived: the archived members.
    archive_epsilon: the archived mass.

  Returns:
    The Mix.

  Raises:
    ValueError: a mass outside [0, 1], masses summing to 1 or more, a member
      in two bands, or bad active weights.
  """
  for v in (lambda_t, epsilon, archive_epsilon):
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
      raise ValueError(
          f"pmix masses must be in [0, 1]: {lambda_t}, {epsilon}, "
          f"{archive_epsilon}"
      )
  if lambda_t + epsilon + archive_epsilon >= 1.0:
    raise ValueError(
        "lambda + epsilon + archive_epsilon = "
        f"{lambda_t + epsilon + archive_epsilon} >= 1 leaves no self-play"
    )
  audit, archived = _check_unique(audit), _check_unique(archived)
  bands = [set(active), set(audit), set(archived)]
  for i in range(3):
    if any(bands[i] & bands[j] for j in range(i + 1, 3)):
      raise ValueError(f"a member is in two pmix bands: {bands}")
  w: dict[str, float] = {}
  if active:
    tot = float(sum(active.values()))
    if abs(tot - 1.0) > MASS_TOL or any(v < 0 for v in active.values()):
      raise ValueError(
          f"active weights must be >= 0 and sum to 1, got {dict(active)}"
      )
    w.update({n: lambda_t * float(v) for n, v in active.items()})
  for band, mass in ((audit, epsilon), (archived, archive_epsilon)):
    for n in band:
      w[n] = mass / len(band)
  return _fold(w)


def lagged_audit(
    lagged: Optional[str],
    audit: Sequence[str],
    self_play: float,
    lagged_mass: float,
) -> Mix:
  """Self-play, lagged_mass on the lagged snapshot, the rest on the audit pool.

  Args:
    lagged: the learner's lagged snapshot, or None.
    audit: the audit pool.
    self_play: the self-play mass.
    lagged_mass: the snapshot's mass.

  Returns:
    The Mix.

  Raises:
    ValueError: the masses are negative or exceed 1.
  """
  if not (
      0.0 <= self_play and 0.0 <= lagged_mass and self_play + lagged_mass <= 1.0
  ):
    raise ValueError(
        f"lagged_audit masses: self {self_play} + lagged {lagged_mass} > 1"
    )
  audit = _check_unique(audit)
  if lagged is not None and lagged in audit:
    raise ValueError(f"the lagged snapshot {lagged} is also in the audit pool")
  w: dict[str, float] = {}
  if lagged is not None:
    w[lagged] = lagged_mass
  rest = 1.0 - self_play - lagged_mass
  for n in audit:
    w[n] = rest / len(audit)
  return _fold(w)


def snapshot_pool(
    snapshot: str, seated: Sequence[str], self_weight: float
) -> Mix:
  """The snapshot-pool mix (no self-play rows).

  self_weight on the lagged snapshot, the rest uniform over the seated pool
  members; everything on the snapshot when none is seated. A seated member
  that is the snapshot itself adds its share to the snapshot's mass.

  Args:
    snapshot: the lagged snapshot.
    seated: the seated pool members.
    self_weight: the snapshot's mass, in (0, 1].

  Returns:
    The Mix.

  Raises:
    ValueError: self_weight is outside (0, 1].
  """
  seated = _check_unique(seated)
  if not 0.0 < float(self_weight) <= 1.0:
    raise ValueError(f"self_weight must be in (0, 1], got {self_weight}")
  if not seated:
    return Mix(0.0, {str(snapshot): 1.0})
  w = {str(snapshot): float(self_weight)}
  each = (1.0 - float(self_weight)) / len(seated)
  for n in seated:
    w[n] = w.get(n, 0.0) + each
  return Mix(0.0, w)


def meta(names: Sequence[str], strategy, self_play: float = 0.0) -> Mix:
  """A meta-strategy (f64[n], sums to 1) over names, scaled by 1 - self_play.

  Args:
    names: the member names.
    strategy: the meta-strategy.
    self_play: the self-play mass.

  Returns:
    The Mix.

  Raises:
    ValueError: the strategy does not fit names or does not sum to 1.
  """
  names = _check_unique(names)
  s = np.asarray(strategy, F64)
  if s.shape != (len(names),) or not np.all(np.isfinite(s)) or np.any(s < 0.0):
    raise ValueError(f"meta strategy {s} does not fit names {names}")
  if abs(s.sum() - 1.0) > MASS_TOL:
    raise ValueError(f"meta strategy sums to {s.sum()}")
  return _fold({n: (1.0 - self_play) * float(v) for n, v in zip(names, s)})


def pipeline(
    fixed: Sequence[str], strategy, lower: Sequence[str], active_share: float
) -> Mix:
  """A pipeline-PSRO worker's mix.

  (1 - active_share) of the meta-strategy over `fixed`, and active_share
  uniform over the lower workers.

  Args:
    fixed: the fixed members.
    strategy: the meta-strategy over fixed.
    lower: the lower workers.
    active_share: the lower workers' mass, in (0, 1).

  Returns:
    The Mix.

  Raises:
    ValueError: active_share is outside (0, 1).
  """
  lower = _check_unique(lower)
  if not lower:
    return meta(fixed, strategy)
  if not 0.0 < active_share < 1.0:
    raise ValueError(f"active_share must be in (0, 1), got {active_share}")
  base = meta(fixed, strategy).weights
  w = {n: (1.0 - active_share) * v for n, v in base.items()}
  for n in lower:
    if n in w:
      raise ValueError(f"lower worker {n} is also a fixed member")
    w[n] = active_share / len(lower)
  return _fold(w)


# -- pools and the draw --------------------------------------------------------
@dataclasses.dataclass(frozen=True, eq=False)
class Pool:
  """One opponent Arena slot at fixed K: policy, rule, K positions.

  Attributes:
    policy: the slot's policy.
    rule: the slot's decision rule.
    names: K entries, each a member name or None (empty).
    params: JAX: stacked [K, ...]; HOST_DRIVEN: a list of K host objects.
  """

  policy: types.Policy
  rule: types.DecisionRule
  # K entries, each a member name or None (empty)
  names: tuple[Optional[str], ...]
  params: Any  # JAX: stacked [K, ...]; HOST_DRIVEN: a list of K host objects

  def __post_init__(self):
    names = tuple(self.names)
    object.__setattr__(self, "names", names)
    assert len(names) >= 1, "a pool has K >= 1 positions"
    held = [n for n in names if n is not None]
    if len(set(held)) != len(held):
      raise ValueError(f"a pool holds a name twice: {names}")
    if SELF in held:
      raise ValueError(f"{SELF!r} is reserved")
    assert isinstance(self.rule, types.DecisionRule), self.rule

  @property
  def k(self) -> int:
    """The number of positions K."""
    return len(self.names)

  def slot(self) -> slots_lib.Slot:
    """The pool's Arena slot.

    Returns:
      The Slot.
    """
    return slots_lib.Slot(self.policy, self.k, self.rule)

  def index(self, name: str) -> int:
    """The position of member `name`.

    Args:
      name: the member name.

    Returns:
      The position.
    """
    return self.names.index(name)


@dataclasses.dataclass(frozen=True, eq=False)
class LeagueSlots:
  """What the sampler reads: the opponent pools, the batch and one Mix.

  Slot 0 is the learner (K = 1); the pools are Arena slots 1...

  Attributes:
    num_players: the game's seat count.
    batch: the rows per batch.
    pools: the opponent pools.
    mix: the learner's Mix.
    stratified: stratified (True) or iid draws.
    sides: the table's core.sides.Sides; None: table_from_mix, the seating
      before sides (at P > 2 one drawn member fills every other seat;
      sample's doc).
    lineup: draw_table's lineup (needs sides).
    self_play_seats: draw_table's self_play_seats (needs sides).
  """

  num_players: int
  batch: int
  pools: tuple[Any, ...]
  mix: Mix
  stratified: bool = True
  sides: Optional[sides_lib.Sides] = None
  lineup: Optional[sides_lib.Lineup] = None
  self_play_seats: str = "one"

  def __post_init__(self):
    object.__setattr__(self, "pools", tuple(self.pools))
    assert self.num_players >= 2 and self.batch >= 1
    assert isinstance(self.mix, Mix), type(self.mix)
    if self.sides is None:
      if self.lineup is not None or self.self_play_seats != "one":
        raise ValueError(
            "LeagueSlots: a lineup or self_play_seats other than 'one' needs "
            "sides"
        )
    else:
      if not isinstance(self.sides, sides_lib.Sides):
        raise TypeError(
            "LeagueSlots.sides is a core.sides.Sides, got "
            f"{type(self.sides).__name__}"
        )
      if self.sides.num_players != self.num_players:
        raise ValueError(
            f"LeagueSlots: sides have {self.sides.num_players} seats, the "
            f"table {self.num_players}"
        )
    seen: set[str] = set()
    for pool in self.pools:
      assert isinstance(pool, Pool), type(pool)
      for n in pool.names:
        if n is not None:
          if n in seen:
            raise ValueError(f"member {n} is in two pools")
          seen.add(n)

  def locate(self, name: str) -> tuple[int, int]:
    """(Arena slot, member) of a member name.

    Args:
      name: the member name.

    Returns:
      (slot, member).

    Raises:
      KeyError: no pool holds it.
    """
    for j, pool in enumerate(self.pools):
      if name in pool.names:
        return j + 1, pool.index(name)
    raise KeyError(
        f"no pool holds member {name!r} (pools: "
        f"{[p.names for p in self.pools]})"
    )


class Draw(NamedTuple):
  """A drawn slot table.

  Attributes:
    table: the SlotTable.
    params: the pools' params.
    counts: rows per category.
  """

  # i32 [B, P] slots/members, i32 [B] perspective seat
  table: slots_lib.SlotTable
  params: list[Any]  # per pool: its params, untouched (fixed K)
  counts: dict[Any, int]  # category -> rows (positive counts only)


def _largest_remainder(masses: np.ndarray, rows: int) -> np.ndarray:
  """Splits `rows` by `masses` with the largest-remainder rule.

  Args:
    masses: The masses, summing to 1.
    rows: The rows.

  Returns:
    int64 rows per mass.
  """
  quota = masses * rows
  base = np.floor(quota + 1e-12).astype(np.int64)
  left = rows - int(base.sum())
  assert 0 <= left <= masses.size, (left, masses, rows)
  frac = quota - base
  order = sorted(range(masses.size), key=lambda i: (-round(frac[i], 12), i))
  for i in order[:left]:
    base[i] += 1
  assert int(base.sum()) == rows
  return base


def table_from_mix(
    mix: Mix,
    where: Mapping[str, tuple[int, int]],
    batch: int,
    num_players: int,
    key,
    stratified: bool = True,
    learner: tuple[int, int] = (0, 0),
) -> tuple[slots_lib.SlotTable, dict[str, Any]]:
  """The row allocation (module doc): (SlotTable, counts {category: rows}).

  The learner (slot, member) sits in seat row % P (the perspective seat); an
  opponent row seats where[name] in every other seat. Every name with
  positive mass must be in `where`. Pure in its arguments.

  Args:
    mix: the Mix.
    where: (slot, member) of every name with positive mass.
    batch: the rows.
    num_players: the seat count P.
    key: a core.rng key.
    stratified: stratified (True) or iid draws.
    learner: the learner's (slot, member).

  Returns:
    (SlotTable, counts).

  Raises:
    KeyError: a name with positive mass is not in `where`.
  """
  p, b = int(num_players), int(batch)
  assert p >= 2 and b >= 1, (p, b)
  cats = mix.categories()
  names = [n for n, _ in cats]
  masses = np.asarray([m for _, m in cats], F64)
  loc = [tuple(learner)]
  for n, m in cats[1:]:
    if m > 0.0 and n not in where:
      raise KeyError(f"no slot holds member {n!r}")
    loc.append(tuple(where[n]) if m > 0.0 else (-1, -1))
  seat = (np.arange(b) % p).astype(np.int32)
  cat_of_row = np.zeros(b, np.int64)
  if stratified:
    perm_keys = rng.split(key, p)
    for s in range(p):
      rows = np.flatnonzero(seat == s)
      counts = _largest_remainder(masses, rows.size)
      cats_s = np.repeat(np.arange(masses.size), counts)
      perm = np.asarray(rng.permutation(perm_keys[s], rows.size))
      cat_of_row[rows] = cats_s[perm]
  else:
    cat_of_row = np.asarray(
        rng.choice(key, masses.size, (b,), p=masses), np.int64
    )
  slot = np.zeros((b, p), np.int32)
  member = np.zeros((b, p), np.int32)
  for r in range(b):
    j, m = loc[int(cat_of_row[r])]
    assert j >= 0, (r, names[int(cat_of_row[r])])
    slot[r, :], member[r, :] = j, m
    slot[r, seat[r]], member[r, seat[r]] = learner
  counts = {}
  for c in range(masses.size):
    n = int((cat_of_row == c).sum())
    if n:
      counts[names[c]] = n
  table = slots_lib.SlotTable(
      seat_slot=slot, seat_member=member, perspective_seat=seat
  )
  return table, counts


def sample(state: LeagueSlots, key) -> Draw:
  """The slot table for one learner batch; pure in (state, key).

  With state.sides, draw_table (module doc). Without, table_from_mix, which
  seats one drawn member on every other seat: at P > 2 that is the table
  before sides (the opponent also fills the learner's partner seats, and one
  opponent fills every seat of a free-for-all table), kept as is for callers
  that do not pass sides yet; a caller seating by sides passes them.

  Args:
    state: the LeagueSlots.
    key: a core.rng key.

  Returns:
    The Draw.
  """
  where = {}
  for n, m in state.mix.categories()[1:]:
    if m > 0.0:
      where[n] = state.locate(n)
  if state.sides is None:
    table, counts = table_from_mix(
        state.mix, where, state.batch, state.num_players, key, state.stratified
    )
  else:
    if state.lineup is not None:
      for f in state.lineup.units:
        if f.kind == "fixed":
          where[f.member] = state.locate(f.member)
    table, counts = draw_table(
        state.mix,
        where,
        state.batch,
        state.sides,
        key,
        stratified=state.stratified,
        lineup=state.lineup,
        self_play_seats=state.self_play_seats,
    )
  return Draw(
      table=table, params=[pool.params for pool in state.pools], counts=counts
  )


# -- sides: lineups and the draw dispatch --------------------------------------
#: draw_table's self_play_seats: "one" (the learner's sides' seats), "all"
#: (every seat of a pure self-play row).
SELF_PLAY_SEATS = ("one", "all")
# The salt of the keys of a lineup's second and later pool sides: the first
# pool side draws with the table key itself, exactly as table_from_mix draws
# its opponent, and rng.split(key, n)[i] == rng.fold_in(key, i), so later
# sides fold their position into a salted key, never into the table key.
_SIDE_KEY_SALT = 0x51DE5
# The fold-in id of a per-seat pool unit (position q, seat j of the side):
# above every side position id q < T, so a per-seat unit never shares a key
# with a whole-side unit.
_SEAT_UNIT_BASE = 0x10000


def _check_learner(learner) -> tuple[int, int]:
  """A learner (slot, member) pair, checked.

  Args:
    learner: (slot, member).

  Returns:
    (slot, member) as ints.

  Raises:
    ValueError: A negative index.
  """
  j, m = (int(x) for x in learner)
  if j < 0 or m < 0:
    raise ValueError(f"learner (slot, member) must be >= 0, got {learner}")
  return j, m


def lineup_from_mix(
    sides: sides_lib.Sides, learner_side: int = 0
) -> sides_lib.Lineup:
  """The Lineup a learner's Mix is played through at P > 2 (module doc).

  The learner at position `learner_side`, cyclic by row over the sides of its
  rotation class (every side when the sides are fully rotatable; the sides of
  its size when they are uneven), every other side drawing its own member from
  the Mix (table_from_lineup takes the Mix at draw time; a Mix changes at run
  time, a Lineup is a config value).

  Uneven sides: the learner's perspective width is the size of its side on
  every row, so one table seats it on sides of one size; a learner that
  should play every role takes one lineup per side size (learner_side).
  Sides that are not rotatable keep the learner at learner_side ("none").

  Args:
    sides: the table's core.sides.Sides.
    learner_side: the learner's side position.

  Returns:
    The Lineup.

  Raises:
    TypeError: sides is not a core.sides.Sides.
    ValueError: learner_side is out of range.
  """
  if not isinstance(sides, sides_lib.Sides):
    raise TypeError(
        f"lineup_from_mix takes a core.sides.Sides, got {type(sides).__name__}"
    )
  ls = int(learner_side)
  if not 0 <= ls < sides.num_sides:
    raise ValueError(
        f"lineup_from_mix: learner_side {learner_side} not in "
        f"0..{sides.num_sides - 1}"
    )
  fills = [sides_lib.SideFill("pool")] * sides.num_sides
  fills[ls] = sides_lib.SideFill("learner")
  return sides_lib.Lineup(tuple(fills), "cyclic" if sides.rotatable else "none")


def table_from_lineup(
    lineup: sides_lib.Lineup,
    mix: Mix,
    where: Mapping[str, tuple[int, int]],
    batch: int,
    sides: sides_lib.Sides,
    key,
    stratified: bool = True,
    learner: tuple[int, int] = (0, 0),
) -> tuple[slots_lib.SlotTable, dict[str, Any]]:
  """The table of a Lineup (module doc): (SlotTable, counts).

  Row r shifts every fill position by r % L within its rotation class under
  rotate="cyclic" (core.sides.Sides.rotate_side: fill position q sits at
  side rotate_side(q, r); L = sides.rotation_period, T for fully rotatable
  sides), by 0 under "none". A whole-side fill seats every seat of its side;
  a per-seat fill seats the j-th seat of its side with its j-th SideFill.
  A "learner" unit seats `learner` (slot, member), every seat of it a
  perspective seat; a "self" unit seats `learner` with no perspective seat;
  a "fixed" unit seats where[member]; a "pool" unit draws a category of
  `mix` ("self": the learner's params, not trained on; a name: where[name]).
  Pool units draw independently of each other: the first pool unit's draw is
  keyed by `key`, the others by fold_in(fold_in(key, salt), id) (id = q for
  a whole side at position q, a salted id for a seat of a per-seat fill),
  and stratified draws allocate each shift class's rows by largest remainder
  over the categories and permute them (exact marginals per unit and
  shift); iid draws are categorical per row and unit. With one pool side at
  P = 2 this is table_from_mix bit for bit.

  perspective_seat is [B] when the learner holds one seat in all, else
  [B, M] in increasing seat order (M = the learner's seats, the same on every
  row: rotation keeps every fill on a side of its size). counts: category ->
  pool-unit draws (rows at P = 2), plus "self" for self units and each fixed
  member's rows (positive counts only).

  Args:
    lineup: the core.sides.Lineup.
    mix: the learner's Mix.
    where: member name -> (slot, member).
    batch: the rows.
    sides: the table's core.sides.Sides.
    key: the draw key.
    stratified: stratified (True) or iid (False) pool draws.
    learner: the learner's (slot, member).

  Returns:
    (SlotTable, counts).

  Raises:
    TypeError: lineup, sides or mix of the wrong type.
    KeyError: a positive-mass name or a fixed member is not in `where`.
    ValueError: the lineup does not fit the sides, or batch < 1.
    NotImplementedError: a "learner" fill names a learning slot other than 0:
      a table is one learner's batch (perspective seats of one slot); seat a
      second learning slot's current params as a "fixed" member instead
      (why a second index is refused rather than remapped: core.sides
      SideFill.learner).
  """
  if not isinstance(lineup, sides_lib.Lineup):
    raise TypeError(
        "table_from_lineup takes a core.sides.Lineup, got "
        f"{type(lineup).__name__}"
    )
  if not isinstance(sides, sides_lib.Sides):
    raise TypeError(
        "table_from_lineup takes a core.sides.Sides, got "
        f"{type(sides).__name__}"
    )
  if not isinstance(mix, Mix):
    raise TypeError(f"table_from_lineup takes a Mix, got {type(mix).__name__}")
  lineup.check(sides)
  b, p, t = int(batch), sides.num_players, sides.num_sides
  if b < 1:
    raise ValueError(f"batch must be >= 1, got {batch}")
  learner = _check_learner(learner)
  fills = lineup.fills
  for f in lineup.units:
    if f.kind == "learner" and f.learner != 0:
      raise NotImplementedError(
          f"a Lineup with learning slot {f.learner}: a table is one learner's "
          "batch, whose perspective seats are learning slot 0's (the `learner` "
          "argument); seat another learning slot's current params as a 'fixed' "
          "member, or draw that slot's own batch with it as the learner"
      )
    if f.kind == "fixed" and f.member not in where:
      raise KeyError(f"no slot holds the fixed member {f.member!r}")
  # Units: (position q, seat index j or None for a whole side, SideFill).
  units = []
  for q, f in enumerate(fills):
    if isinstance(f, tuple):
      units.extend((q, j, g) for j, g in enumerate(f))
    else:
      units.append((q, None, f))
  pool_units = [(q, j) for q, j, f in units if f.kind == "pool"]
  cats = mix.categories()
  names = [n for n, _ in cats]
  masses = np.asarray([m for _, m in cats], F64)
  loc = [learner]
  for n, m in cats[1:]:
    if m > 0.0 and pool_units and n not in where:
      raise KeyError(f"no slot holds member {n!r}")
    loc.append(tuple(where[n]) if (m > 0.0 and pool_units) else (-1, -1))
  cyclic = lineup.rotate == "cyclic"
  period = sides.rotation_period if cyclic else 1
  shift = (np.arange(b) % period).astype(np.int64)
  cat = {}  # (q, j) -> i64[B] category per row
  for i, (q, j) in enumerate(pool_units):
    if i == 0:
      k_q = key
    else:
      uid = q if j is None else _SEAT_UNIT_BASE + q * 256 + j
      k_q = rng.fold_in(rng.fold_in(key, _SIDE_KEY_SALT), uid)
    col = np.full(b, -1, np.int64)
    if stratified:
      perm_keys = rng.split(k_q, period)
      for c in range(period):
        rows = np.flatnonzero(shift == c)
        counts_c = _largest_remainder(masses, rows.size)
        cats_c = np.repeat(np.arange(masses.size), counts_c)
        perm = np.asarray(rng.permutation(perm_keys[c], rows.size))
        col[rows] = cats_c[perm]
    else:
      col[:] = np.asarray(
          rng.choice(k_q, masses.size, (b,), p=masses), np.int64
      )
    cat[(q, j)] = col
  side_seats = [sides.seats(s) for s in range(t)]
  # Where fill position q sits on each shift: rotation keeps sizes, so a
  # per-seat fill always meets a side of its size (Lineup.check sized it).
  side_at = [[sides.rotate_side(q, c) for q in range(t)] for c in range(period)]
  for c in range(period):
    assert sorted(side_at[c]) == list(range(t)), side_at[c]
    for q in range(t):
      assert sides.side_sizes[side_at[c][q]] == sides.side_sizes[q], (c, q)
  m_width = sum(
      len(side_seats[q]) if j is None else 1
      for q, j, f in units
      if f.kind == "learner"
  )
  slot = np.zeros((b, p), np.int32)
  member = np.zeros((b, p), np.int32)
  persp = np.zeros((b, m_width), np.int32)
  for r in range(b):
    held = []
    for q, j, f in units:
      s = side_at[int(shift[r])][q]
      seats = side_seats[s] if j is None else (side_seats[s][j],)
      if f.kind in ("learner", "self"):
        jm = learner
        if f.kind == "learner":
          held.extend(seats)
      elif f.kind == "fixed":
        jm = tuple(where[f.member])
      else:
        jm = loc[int(cat[(q, j)][r])]
        assert jm[0] >= 0, (r, q, j, names[int(cat[(q, j)][r])])
      for seat in seats:
        slot[r, seat], member[r, seat] = jm
    assert len(held) == m_width, (r, held, m_width)
    persp[r] = sorted(held)
  counts: dict[str, Any] = {}
  for c in range(masses.size):
    n = int(sum(int((col == c).sum()) for col in cat.values()))
    if n:
      counts[names[c]] = n
  n_self = b * sum(1 for _, _, f in units if f.kind == "self")
  if n_self:
    counts[SELF] = counts.get(SELF, 0) + n_self
  for _, _, f in units:
    if f.kind == "fixed":
      counts[f.member] = counts.get(f.member, 0) + b
  table = slots_lib.SlotTable(
      seat_slot=slot,
      seat_member=member,
      perspective_seat=persp[:, 0].copy() if m_width == 1 else persp,
  )
  return table, counts


def draw_table(
    mix: Mix,
    where: Mapping[str, tuple[int, int]],
    batch: int,
    sides: sides_lib.Sides,
    key,
    *,
    stratified: bool = True,
    learner: tuple[int, int] = (0, 0),
    lineup: Optional[sides_lib.Lineup] = None,
    self_play_seats: str = "one",
) -> tuple[slots_lib.SlotTable, dict[str, Any]]:
  """The league's table for one learner batch (module doc).

  (SlotTable, counts).

  self_play_seats="all"  core.arena.self_play_table(all_seats=True); the mix
                         must be pure self-play and there is no lineup.
  lineup                 table_from_lineup(lineup, mix, ...).
  P = 2                  table_from_mix (the two-player seating, unchanged).
  otherwise              table_from_lineup(lineup_from_mix(sides), mix, ...).

  Args:
    mix: the learner's Mix.
    where: member name -> (slot, member).
    batch: the rows.
    sides: the table's core.sides.Sides.
    key: the draw key.
    stratified: stratified (True) or iid (False) pool draws.
    learner: the learner's (slot, member).
    lineup: a core.sides.Lineup, or None.
    self_play_seats: a SELF_PLAY_SEATS name.

  Returns:
    (SlotTable, counts).

  Raises:
    TypeError: sides is not a core.sides.Sides.
    ValueError: an unknown self_play_seats, or "all" with a lineup or a mix
      that is not pure self-play.
  """
  if not isinstance(sides, sides_lib.Sides):
    raise TypeError(
        f"draw_table takes a core.sides.Sides, got {type(sides).__name__}"
    )
  if self_play_seats not in SELF_PLAY_SEATS:
    raise ValueError(
        f"self_play_seats {self_play_seats!r} not in {SELF_PLAY_SEATS}"
    )
  if self_play_seats == "all":
    if lineup is not None:
      raise ValueError(
          "self_play_seats='all' seats the learner on every side; a lineup "
          "says who fills each side: give one or the other"
      )
    if mix.self_play != 1.0:
      raise ValueError(
          "self_play_seats='all' emits every seat of a self-play row and "
          f"needs a pure self-play mix, got {mix}"
      )
    # core.arena.self_play_table(all_seats=True) with the learner's (slot,
    # member); built here so this module imports no array framework.
    j, m = _check_learner(learner)
    p, b = sides.num_players, int(batch)
    table = slots_lib.SlotTable(
        seat_slot=np.full((b, p), j, np.int32),
        seat_member=np.full((b, p), m, np.int32),
        perspective_seat=np.broadcast_to(
            np.arange(p, dtype=np.int32), (b, p)
        ).copy(),
    )
    return table, {SELF: b}
  if lineup is not None:
    return table_from_lineup(
        lineup, mix, where, batch, sides, key, stratified, learner
    )
  if sides.num_players == 2:
    return table_from_mix(mix, where, batch, 2, key, stratified, learner)
  return table_from_lineup(
      lineup_from_mix(sides), mix, where, batch, sides, key, stratified, learner
  )


def entries_table(
    batch: int,
    num_players: int,
    entries: Sequence[tuple[int, int, float]],
    key,
    learner: tuple[int, int] = (0, 0),
) -> slots_lib.SlotTable:
  """Learner vs a weighted pool, from (slot, member, weight) entries.

  Weights > 0 summing to 1, no self-play: table_from_mix over the entries,
  stratified (each entry meets the learner in every seat when its rows
  allow).

  Args:
    batch: the rows.
    num_players: the seat count.
    entries: (slot, member, weight) per entry.
    key: a core.rng key.
    learner: the learner's (slot, member).

  Returns:
    The SlotTable.

  Raises:
    ValueError: a weight is not > 0, or the weights do not sum to 1.
  """
  assert entries, "an empty pool"
  w = {}
  where = {}
  for i, (j, m, wt) in enumerate(entries):
    # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
    # pylint: disable-next=unnecessary-negation
    if not float(wt) > 0.0:
      raise ValueError(f"pool weights must be > 0, got {wt} for entry {i}")
    name = f"entry{i:06d}"
    w[name], where[name] = float(wt), (int(j), int(m))
  total = sum(w.values())
  if abs(total - 1.0) > MASS_TOL:
    raise ValueError(f"pool weights must sum to 1, got {total}")
  table, _ = table_from_mix(
      Mix(0.0, w), where, batch, num_players, key, True, learner
  )
  return table
