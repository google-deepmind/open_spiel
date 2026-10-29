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
`table_from_mix` is the one row allocation `sample` uses; `entries_table`
draws a weighted (slot, member) pool with it. The learner is slot 0 member 0
in seat row % P (the perspective seat); an opponent row seats the drawn
member in every other seat. Stratified draws (the default) allocate rows per
seat class by largest remainder over the categories ["self", then names
sorted] and permute them with `key`, so counts are exact and balanced per
seat; iid draws are categorical per row. Pool params pass through untouched.

Keys are `core.rng` keys: a jax key and a numpy key with the same data draw
the same table, so this module imports no array framework.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from open_spiel.python.scale.core import rng
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
  """

  num_players: int
  batch: int
  pools: tuple[Any, ...]
  mix: Mix
  stratified: bool = True

  def __post_init__(self):
    object.__setattr__(self, "pools", tuple(self.pools))
    assert self.num_players >= 2 and self.batch >= 1
    assert isinstance(self.mix, Mix), type(self.mix)
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
  table, counts = table_from_mix(
      state.mix, where, state.batch, state.num_players, key, state.stratified
  )
  return Draw(
      table=table, params=[pool.params for pool in state.pools], counts=counts
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
