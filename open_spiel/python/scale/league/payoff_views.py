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

"""Payoff views: the payoffs a league's controllers read.

Reduced from table results.

The evaluation primitive is a table record (eval.tables.TableResult). A
league's controllers read a K x K PayoffMatrix over its entrants; this module
says what a cell of that matrix means for the table's sides (core.sides) and
computes it from table records.

  view_kind(sides)           "pair" (two interchangeable sides) or "field"
                             (more than two, or two uneven ones)

  Two interchangeable sides ("pair"; every two-player game, and bridge's
    partnerships).
    The matrix is the side-against-side payoff of a two-player zero-sum game
    over the entrants. Two-player games use the Evaluator's head-to-head
    cell directly (league.loop). Two sides of several seats
    build the cell from paired differences on a swap2 table:

      side_cell(res, scorer)   (mean_ab, se_ab, mean_ba, se_ba) with
                               mean_ab = mean over deals of (s_a - s_b) / 2,
                               s the pair score of each position (the env's
                               pair scorer, e.g. bridge IMPs, else the mean of
                               the two rows), and mean_ba = -mean_ab.

    The cell is exactly antisymmetric (the LP meta-solvers assume the column
    player gets -P) and centred: a constant sum c of the side returns cancels
    in the difference. Drawing one member per side makes a member a team
    strategy correlated only through its shared policy, so the matrix is the
    zero-sum game over that strategy class (mixed partnerships are lineups,
    outside the K x K matrix).

  More than two sides, or two uneven ones ("field"; kuhn with N players,
    hearts, a pair against one seat). A pairwise payoff is not defined (two
    uneven sides are two roles). Cell (i, j) is entrant i's side return
    against a field of j: i holds one side, every other side is j, under a
    rotation-balanced design (cyclic, or cyclic+reversed; at uneven sides
    cyclic+classes, the league averaging i over every side position), so
    the diagonal is the self-play return (c / T in expectation for
    constant-sum sides). The
    matrix is K x K evaluations, is neither antisymmetric nor zero-sum, and
    every number read from it is relative to the field it names; it is not an
    equilibrium claim. Controllers that solve it as a two-player game are
    refused at more than two sides (league.loop's capability check).

      field_cell(res)          (mean, se) of position 0's score per deal

  PFSP input at any sides.
      pfsp_winrates(res)       per entrant m at positions other than 0: the
                               probability position 0's side return beats
                               m's, over the rows and positions where m sits,
                               ties split; at two sides this is the subject's
                               win rate against m.

  Reports.
      field_scores(...)        per entrant, vs_field against a mixture of the
                               entrants (every other side drawn independently
                               per deal): mean, SE and placement.
      league_table(...)        printable rows of a payoff matrix (and field
                               scores).

numpy only; no JAX.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.eval import designs as designs_lib
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.eval import stats
from open_spiel.python.scale.eval import tables

VIEWS = ("pair", "field")
#: The designs a field view's cells may use (LeagueConfig.eval_design):
#: rotation-balanced over the sides, one lineup block.
FIELD_DESIGNS = ("cyclic", "cyclic+reversed", "cyclic+classes")
#: Antisymmetry tolerance of check_antisymmetric (absolute, f64).
ANTISYM_ATOL = 1e-9


def view_kind(sides: sides_lib.Sides) -> str:
  """The view of `sides`: "pair" or "field".

  "pair" for two interchangeable sides (a side-against-side matrix), "field"
  for more, or for two uneven (or non-rotatable) sides.

  Args:
    sides: the table's core.sides.Sides.

  Returns:
    "pair" or "field".

  Raises:
    TypeError: sides is not a core.sides.Sides.
  """
  if not isinstance(sides, sides_lib.Sides):
    raise TypeError(
        f"view_kind takes a core.sides.Sides, got {type(sides).__name__}"
    )
  return "pair" if sides.num_sides == 2 and sides.fully_rotatable else "field"


def resolve_design(name: str, sides: sides_lib.Sides) -> str:
  """The design name of a league's payoff cells (LeagueConfig.eval_design).

  "auto": swap2 at two interchangeable sides, cyclic at more, cyclic+classes
  at uneven (or non-rotatable) sides, two of them included. A pair view
  takes only swap2; a field view only FIELD_DESIGNS.

  Args:
    name: the configured design name, or "auto".
    sides: the table's core.sides.Sides.

  Returns:
    The design name.

  Raises:
    ValueError: a design not valid for these sides.
  """
  kind = view_kind(sides)
  if name == "auto":
    if kind == "pair":
      return "swap2"
    return "cyclic" if sides.fully_rotatable else "cyclic+classes"
  if kind == "pair" and name != "swap2":
    raise ValueError(
        f"eval_design {name!r}: a two-sided league's payoff cell is the swap2 "
        "duplicate pair ('auto' or 'swap2')"
    )
  if kind == "field" and name not in FIELD_DESIGNS:
    raise ValueError(
        f"eval_design {name!r}: a field cell at {sides.num_sides} sides takes "
        f"one of {FIELD_DESIGNS} (or 'auto')"
    )
  if name in ("cyclic", "cyclic+reversed") and not sides.fully_rotatable:
    raise ValueError(
        f"eval_design {name!r} rotates every side; sides of sizes "
        f"{sides.side_sizes} (rotatable={sides.rotatable}) take cyclic+classes"
    )
  return name


# -- cells ---------------------------------------------------------------------
def side_cell(
    res: tables.TableResult, scorer: Optional[Callable[..., Any]] = None
) -> tuple[Any, ...]:
  """(mean_ab, se_ab, mean_ba, se_ba) of a two-sided swap2 table (module doc).

  Position 0 is a, position 1 is b; s_k is position k's pair score per deal
  (scorer(row 0, row 1), or the mean of the two rows without one). The cell
  is the mean of (s_a - s_b) / 2 with its SE; (b, a) is its negative.

  Args:
    res: the TableResult.
    scorer: a pair scorer, or None for the mean.

  Returns:
    (mean_ab, se_ab, mean_ba, se_ba).

  Raises:
    ValueError: not a swap2 table of two interchangeable sides.
  """
  _need(res, two_sides=True)
  if res.design.name != "swap2":
    raise ValueError(
        f"side_cell reads a swap2 table, got design {res.design.name}"
    )
  sa = res.deal_scores(0, 0, scorer)
  sb = res.deal_scores(1, 0, scorer)
  e = stats.estimate(0.5 * (sa - sb))
  return (e.mean, e.se, -e.mean, e.se)


def field_cell(res: tables.TableResult) -> tuple[Any, ...]:
  """(mean, se) of position 0's side return per deal on a one-block table.

  The subject against its field.

  Args:
    res: the TableResult.

  Returns:
    (mean, se).

  Raises:
    ValueError: not a one-block table of a field view.
  """
  _need(res, two_sides=False)
  if res.design.num_blocks != 1:
    raise ValueError(
        f"field_cell reads a one-block table, got design {res.design.name}"
    )
  e = res.position_score(0)
  return (e.mean, e.se)


def pfsp_winrates(res: tables.TableResult, block: int = 0) -> dict[str, Any]:
  """Position 0's win rate against each entrant it met.

  {entrant name: (win rate, rows)}: for each entrant m holding a position
  other than 0, P(position 0's side return > m's side return) over the rows of
  `block` and the positions where m sits, a tie counting 1/2 (module doc).
  rows counts the (row, position) samples behind it.

  Args:
    res: the TableResult.
    block: the design block.

  Returns:
    The dict.

  Raises:
    TypeError: res is not a TableResult.
  """
  if not isinstance(res, tables.TableResult):
    raise TypeError(
        f"pfsp_winrates takes a TableResult, got {type(res).__name__}"
    )
  t = res.sides.num_sides
  mine = res.position_returns(0, block)  # [n, R_b]
  wins: dict[str, Any] = {}
  for k in range(1, t):
    # [n]: position k's entrant per deal
    who = res.lineup[:, block, res.sides.seats(k)[0]]
    theirs = res.position_returns(k, block)
    score = (mine > theirs).astype(np.float64) + 0.5 * (mine == theirs)
    for e in np.unique(who).tolist():
      sel = who == e
      w, c = wins.get(e, (0.0, 0))
      wins[e] = (w + float(score[sel].sum()), c + int(score[sel].size))
  return {res.entrants[e]: (w / c, c) for e, (w, c) in sorted(wins.items())}


def check_antisymmetric(
    pm: payoff.PayoffMatrix, atol: float = ANTISYM_ATOL
) -> None:
  """Raises unless pm.mean is antisymmetric off the diagonal.

  A pair view built from differences.

  Args:
    pm: the PayoffMatrix.
    atol: the tolerance.

  Raises:
    AssertionError: pm.mean is not antisymmetric.
  """
  m = np.asarray(pm.mean, np.float64)
  off = m + m.T
  np.fill_diagonal(off, 0.0)
  bad = np.argwhere(np.abs(off) > atol)
  if bad.size:
    i, j = (int(x) for x in bad[0])
    raise AssertionError(
        f"payoff matrix not antisymmetric: ({pm.names[i]}, {pm.names[j]}) "
        f"{m[i, j]} vs {m[j, i]}"
    )


# -- reports -------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FieldScore:
  """One entrant's vs_field score against a mixture of entrants.

  Attributes:
    name: the entrant.
    score: its side-return Estimate per deal (position 0).
    placement: f64[T], P(1st), ..., P(T-th), ties split.
    winrates: pfsp_winrates of its table.
  """

  name: str
  score: stats.Estimate
  placement: np.ndarray
  winrates: Mapping[str, tuple[Any, ...]]


def field_scores(
    evaluator,
    entrants: Sequence[Any],
    weights,
    n_deals: int,
    key,
    design: str = "auto",
) -> tuple[Any, ...]:
  """FieldScore per entrant against the field of every entrant.

  evaluator.vs_field(entrant, entrants, weights) under fold_in(key, i) for
  entrant i (every other side drawn independently per deal from the weights).
  Relative to that field; not an equilibrium claim.

  Args:
    evaluator: the eval.match Evaluator.
    entrants: the entrants.
    weights: the field's weights over the entrants.
    n_deals: deals per entrant.
    key: the key.
    design: the design name, or "auto".

  Returns:
    A tuple of FieldScore.
  """
  # Lazy: core.rng imports jax only when called, and this module is part of the
  # JAX-free control plane.
  from open_spiel.python.scale.core import rng  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  entrants = list(entrants)
  d = resolve_design(design, evaluator.sides)
  out = []
  for i, e in enumerate(entrants):
    res = evaluator.vs_field(
        e, entrants, weights, n_deals, rng.fold_in(key, i), design=d
    )
    out.append(
        FieldScore(
            e.name, res.position_score(0), res.placement(0), pfsp_winrates(res)
        )
    )
  return tuple(out)


def league_table(
    pm: payoff.PayoffMatrix,
    kind: str,
    field: Optional[Sequence[FieldScore]] = None,
) -> str:
  """A printable table of a league's payoff matrix.

  Per entrant its row mean (the mean of its cells against every entrant: a
  pair view's score against the uniform population, a field view's against
  uniform single-member fields), the matrix, and its FieldScore when given.

  Args:
    pm: the PayoffMatrix.
    kind: a VIEWS name.
    field: the FieldScores, or None.

  Returns:
    The table as text.

  Raises:
    ValueError: an unknown kind.
  """
  if kind not in VIEWS:
    raise ValueError(f"league_table kind {kind!r} not in {VIEWS}")
  names = list(pm.names)
  w = max(8, max(len(n) for n in names))
  head = "cell (i, j): " + (
      "i against j, side against side"
      if kind == "pair"
      else "i against a field of j (relative to j)"
  )
  lines = [
      head,
      " " * w + "  row-mean " + " ".join(f"{n[:9]:>9}" for n in names),
  ]
  for i, n in enumerate(names):
    cells = " ".join(f"{pm.mean[i, j]:+9.4f}" for j in range(len(names)))
    lines.append(f"{n:<{w}}  {pm.mean[i].mean():+9.4f} {cells}")
  if field:
    lines.append(
        "vs_field against the uniform mixture of the entrants (per-side draws):"
    )
    for f in field:
      lines.append(
          f"{f.name:<{w}}  {f.score.mean:+.4f} +- {f.score.se:.4f} "
          f"(n={f.score.n})  placement "
          f"{np.array2string(np.asarray(f.placement), precision=3)}"
      )
  return "\n".join(lines)


def _need(res, two_sides: bool) -> None:
  if not isinstance(res, tables.TableResult):
    raise TypeError(f"want a TableResult, got {type(res).__name__}")
  kind = view_kind(res.sides)
  if two_sides and kind != "pair":
    raise ValueError(
        "a pair cell needs two interchangeable sides, the table has sides "
        f"{res.sides.side_of_seat}"
    )
  if not two_sides and kind == "pair":
    raise ValueError(
        "a field cell is for more than two sides (or two uneven ones); two "
        "interchangeable sides use side_cell"
    )


def design_names() -> tuple[Any, ...]:
  """Every design name a league's eval_design may take ('auto' first).

  Returns:
    A tuple of names.
  """
  return ("auto",) + tuple(
      n for n in designs_lib.NAMES if n in ("swap2",) + FIELD_DESIGNS
  )
