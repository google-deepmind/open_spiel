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

"""Table results: what every seat scored on every replay of every deal.

For any sides.

A pairwise payoff "a against b" is not defined once a table has more than
two sides (hearts, kuhn with three players), so the primitive record of an
evaluation is the table itself: for n deals, each replayed R times by a
seating design (eval.designs), who sat where and what each seat and side
returned. Every number is an estimator over that record, with the deal as
the statistical unit (eval.stats):

  res = ev.vs_field(subject, field, weights, n_deals, key)    # a TableResult
  res.position_score(0)        # the subject's side return per deal, + its SE
  res.by_side(0)               # the same by the table side it held
  res.diff(0, 1)               # paired: position 0 minus position 1, per deal
  res.block_diff(0)            # substitute's gate statistic: S - I, per deal
  res.placement(0)             # P(1st), ..., P(T-th), ties split
  res.entrant_score("name")    # an entrant's seats, wherever it sat
  res.check_sum()              # the constant-sum invariant (or logged)
  res.as_head_to_head()        # two sides, swap2: the two-player MatchResult

Seats. lineup and seats record the entrant of every seat, so a mixed side
(partners of different entrants) is recorded per seat: res.seat_score(q) is
base seat q's own return, res.position_score(k) the side's, which belongs to
the lineup position (not to either partner).

Positions. A design maps whole base sides onto table sides, so "position k"
(the entrant of base side k, design.side_perms) is one entrant's role on
every row of a block: position 0 is a design's subject by default
(Evaluator.vs_field and substitute take another `position`, e.g. a lone
side's at 3v1v1; read that position). Position estimators need no entrant
to be fixed (a field draws per deal).

Pair scores. A two-row block (swap2) can be scored by a pair scorer
(core.sides.pair_scorer_of: the mean of the two rows by default, IMPs for
bridge) instead of the mean of its rows; pass `scorer=`.

Constant sum. When the sides are constant-sum the per-row sum of side
returns must be one constant: `check_sum` raises otherwise (the Evaluator
calls it on every table). A general-sum game has no such invariant; its
sums are logged, never asserted.

numpy only; no JAX.
"""

import dataclasses
from typing import Any, Callable, Optional

from absl import logging
import numpy as np

from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import designs as designs_lib
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.eval import stats

#: Relative tolerance of check_sum: sums of f32 rewards accumulated in f64.
SUM_RTOL = 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class TableResult:
  """n deals of one design (module doc).

  Attributes:
    game: the env's name.
    config: the EvaluatorConfig's name.
    design: the eval.designs.Design played.
    entrants: entrant names; lineups index into them.
    rules: each entrant's DecisionRule, as played.
    lineup: i32[n, B, P]; block b's entrant per base seat on deal i.
    seats: i32[n, R, P]; the entrant at each table seat of each row
      (designs.table_rows of lineup).
    returns: f64[n, R, P]; each seat's payoff over the match's hands.
    side_returns: f64[n, R, T]; each table side's return (mean of its
      seats', core.sides.Sides.side_returns).
    hands_per_match: hands per row.
    duplicate: rows of a deal share its deal key (False: independent deals).
  """

  game: str
  config: str
  design: designs_lib.Design
  entrants: tuple[Any, ...]
  rules: tuple[Any, ...]
  lineup: np.ndarray
  seats: np.ndarray
  returns: np.ndarray
  side_returns: np.ndarray
  hands_per_match: int
  duplicate: bool

  def __post_init__(self):
    d = self.design
    n = self.n_deals
    r, p, t = d.num_rows, d.sides.num_players, d.sides.num_sides
    want = {
        "lineup": (n, d.num_blocks, p),
        "seats": (n, r, p),
        "returns": (n, r, p),
        "side_returns": (n, r, t),
    }
    for name, shape in want.items():
      if getattr(self, name).shape != shape:
        raise ValueError(
            f"TableResult.{name}: shape {getattr(self, name).shape}, want "
            f"{shape}"
        )
    if len(self.rules) != len(self.entrants):
      raise ValueError(
          f"TableResult: {len(self.rules)} rules for {len(self.entrants)} "
          "entrants"
      )
    k = len(self.entrants)
    if self.lineup.min() < 0 or self.lineup.max() >= k:
      raise ValueError(f"TableResult: lineup indexes outside {k} entrants")
    if not np.array_equal(self.seats, designs_lib.table_rows(d, self.lineup)):
      raise ValueError(
          "TableResult: seats are not the design's rows of the lineup"
      )
    if not np.all(np.isfinite(self.returns)):
      raise ValueError("TableResult: non-finite returns")

  @property
  def n_deals(self) -> int:
    """The number of deals."""
    return int(self.lineup.shape[0])

  @property
  def sides(self) -> sides_lib.Sides:
    """The design's sides."""
    return self.design.sides

  # -- positions ---------------------------------------------------------------
  def position_returns(self, k: int = 0, block: int = 0) -> np.ndarray:
    """Base side k's side return on each row of `block`.

    Args:
      k: the position (base side).
      block: the design block.

    Returns:
      f64[n, R_b].
    """
    k = self._position(k)
    rows = self.design.rows_of(block)
    cols = self.design.side_perms[rows, k]
    return self.side_returns[:, rows, cols]

  def deal_scores(
      self,
      k: int = 0,
      block: int = 0,
      scorer: Optional[Callable[..., Any]] = None,
  ) -> np.ndarray:
    """Position k's score per deal.

    The mean over the block's rows, or scorer(row 0, row 1) of a two-row
    block.

    Args:
      k: the position.
      block: the design block.
      scorer: a pair scorer, or None for the mean.

    Returns:
      f64[n].

    Raises:
      ValueError: a scorer on a block that is not two rows, or a scorer that
        returns the wrong shape.
    """
    x = self.position_returns(k, block)
    if scorer is None:
      return x.mean(1)
    if x.shape[1] != 2:
      raise ValueError(
          f"a pair scorer scores two-row blocks; design {self.design.name} "
          f"block {block} has {x.shape[1]} rows"
      )
    out = np.asarray(scorer(x[:, 0], x[:, 1]), np.float64)
    if out.shape != (self.n_deals,):
      raise ValueError(
          f"pair scorer returned shape {out.shape}, want ({self.n_deals},)"
      )
    return out

  def position_score(
      self,
      k: int = 0,
      block: int = 0,
      scorer: Optional[Callable[..., Any]] = None,
  ) -> stats.Estimate:
    """Position k's mean score per deal (deal_scores) with its SE and 95% CI.

    Args:
      k: the position.
      block: the design block.
      scorer: a pair scorer, or None for the mean.

    Returns:
      The Estimate.
    """
    return stats.estimate(self.deal_scores(k, block, scorer))

  def seat_returns(self, q: int, block: int = 0) -> np.ndarray:
    """Base seat q's own return on each row of `block`.

    The seat the design moved it to. On a mixed side (partners of different
    entrants, lineup[:, block, q] per deal) this is that entrant's share; the
    side's result, which belongs to the lineup position, is position_returns.

    Args:
      q: the base seat.
      block: the design block.

    Returns:
      f64[n, R_b].

    Raises:
      ValueError: q is not a seat.
    """
    q = int(q)
    if not 0 <= q < self.sides.num_players:
      raise ValueError(f"seat {q} not in 0..{self.sides.num_players - 1}")
    rows = self.design.rows_of(block)
    cols = np.asarray([self.design.seat_perms[r][q] for r in rows], np.int64)
    return self.returns[:, rows, cols]

  def seat_score(self, q: int, block: int = 0) -> stats.Estimate:
    """Base seat q's mean return per deal over the block's rows, with its SE.

    Args:
      q: the base seat.
      block: the design block.

    Returns:
      The Estimate.
    """
    return stats.estimate(self.seat_returns(q, block).mean(1))

  def by_side(self, k: int = 0, block: int = 0) -> tuple[Any, ...]:
    """Position k's return by the table side it held.

    Per table side s, the Estimate of position k's return on the block's rows
    where it held side s (their mean per deal), or None if it never did.

    Args:
      k: the position.
      block: the design block.

    Returns:
      A tuple of Estimate or None, one per table side.
    """
    k = self._position(k)
    rows = self.design.rows_of(block)
    out = []
    for s in range(self.sides.num_sides):
      sel = rows[self.design.side_perms[rows, k] == s]
      out.append(
          None
          if sel.size == 0
          else stats.estimate(self.side_returns[:, sel, s].mean(1))
      )
    return tuple(out)

  def diff(self, k: int = 0, l: int = 1, block: int = 0) -> stats.Estimate:
    """Paired per deal: position k's return minus position l's.

    The mean over the block's rows.

    Args:
      k: the first position.
      l: the second position.
      block: the design block.

    Returns:
      The Estimate.
    """
    return stats.estimate(
        (
            self.position_returns(k, block) - self.position_returns(l, block)
        ).mean(1)
    )

  def block_diff(
      self,
      k: int = 0,
      b0: int = 0,
      b1: int = 1,
      scorer: Optional[Callable[..., Any]] = None,
  ) -> stats.Estimate:
    """Paired per deal: position k's score in block b0 minus in block b1.

    substitute: the subject's minus the incumbent's against one field.

    Args:
      k: the position.
      b0: the first block.
      b1: the second block.
      scorer: a pair scorer, or None for the mean.

    Returns:
      The Estimate.
    """
    return stats.estimate(
        self.deal_scores(k, b0, scorer) - self.deal_scores(k, b1, scorer)
    )

  def placement(self, k: int = 0, block: int = 0) -> np.ndarray:
    """The probability position k's side finishes 1st, ..., T-th.

    By side return (1st highest) over the block's rows and the deals; a tie
    over places i..j gives each 1 / (j - i + 1).

    Args:
      k: the position.
      block: the design block.

    Returns:
      f64[T].
    """
    k = self._position(k)
    rows = self.design.rows_of(block)
    t = self.sides.num_sides
    out = np.zeros((t,))
    for r in rows:
      mine = self.side_returns[:, r, self.design.side_perms[r, k]][:, None]
      sr = self.side_returns[:, r, :]
      above = (sr > mine).sum(1)
      tied = (sr == mine).sum(1)  # includes itself
      for a, tcount in zip(above, tied):
        out[a : a + tcount] += 1.0 / tcount
    return out / (len(rows) * self.n_deals)

  # -- entrants ----------------------------------------------------------------
  def entrant_score(self, e) -> stats.Estimate:
    """An entrant's mean seat return over the seats it held, per deal.

    Over the deals where it sat at all (Estimate.n counts them). A row where
    it holds several sides counts the mean of all its seats.

    Args:
      e: the entrant's name or index.

    Returns:
      The Estimate.

    Raises:
      ValueError: the entrant sat in fewer than 2 deals.
    """
    i = self._entrant(e)
    held = self.seats == i
    rows = held.any(2)
    per_row = np.where(
        rows, (self.returns * held).sum(2) / np.maximum(held.sum(2), 1), 0.0
    )
    deals = rows.any(1)
    if deals.sum() < 2:
      raise ValueError(
          f"entrant {self.entrants[i]!r} sat in {int(deals.sum())} deals; an "
          "SE needs at least 2"
      )
    return stats.estimate(per_row[deals].sum(1) / rows[deals].sum(1))

  # -- invariants --------------------------------------------------------------
  def check_sum(self) -> Optional[float]:
    """The constant sum of the side returns.

    Checked on every row when the sides are constant-sum; None for
    general-sum sides (sums logged).

    Returns:
      The constant sum, or None for general-sum sides.

    Raises:
      AssertionError: constant-sum sides whose rows sum differently.
    """
    sums = self.side_returns.sum(-1).reshape(-1)
    if not self.sides.constant_sum:
      logging.info(
          "%s/%s: general-sum sides, side-return sums mean %.6g min %.6g max "
          "%.6g over %d rows (no invariant)",
          self.game,
          self.config,
          float(sums.mean()),
          float(sums.min()),
          float(sums.max()),
          sums.size,
      )
      return None
    c = float(sums[0])
    tol = SUM_RTOL * max(1.0, float(np.abs(self.side_returns).max()))
    bad = np.flatnonzero(np.abs(sums - c) > tol)
    if bad.size:
      raise AssertionError(
          f"{self.game}: constant-sum sides, but row sums vary: row 0 {c}, "
          f"row {int(bad[0])} {float(sums[bad[0]])} ({bad.size} rows off)"
      )
    return c

  # -- the two-player view -----------------------------------------------------
  def as_head_to_head(
      self,
      a_name: Optional[str] = None,
      b_name: Optional[str] = None,
      scorer: Optional[Callable[..., Any]] = None,
  ) -> payoff.MatchResult:
    """Two sides, design swap2: position 0 (a) against position 1 (b).

    The MatchResult, column r of a_payoff = a at side r (row r of each deal).
    Without a scorer this is the two-player Evaluator's accounting
    (eval.payoff.match_result).

    Args:
      a_name: names position 0 (default: its entrant, which must then be the
        same on every deal).
      b_name: names position 1, as a_name.
      scorer: a pair scorer (module doc); then `mean` and `diff` are pair
        scores, and se_unpaired and variance_reduction, which compare with
        unpaired rows, are NaN.

    Returns:
      The MatchResult.

    Raises:
      ValueError: not two sides under swap2, an unnamed position held by
        several entrants, or a position playing several rules.
    """
    d = self.design
    if self.sides.num_sides != 2 or d.name != "swap2":
      raise ValueError(
          "as_head_to_head is for two sides under swap2, got "
          f"{self.sides.num_sides} sides, design {d.name}"
      )
    assert d.side_perms.tolist() == [[0, 1], [1, 0]], d.side_perms
    names, rules = [], []
    for k, given in ((0, a_name), (1, b_name)):
      who = np.unique(self.lineup[:, 0, self.sides.seats(k)])
      if given is None and who.size != 1:
        raise ValueError(
            f"as_head_to_head: position {k} holds {who.size} entrants over the "
            "deals; name it (a_name / b_name)"
        )
      rs = {self.rules[i] for i in who.tolist()}
      if len(rs) != 1:
        raise ValueError(
            f"as_head_to_head: position {k} plays several rules "
            f"{sorted(map(str, rs))}"
        )
      names.append(given if given is not None else self.entrants[int(who[0])])
      rules.append(rs.pop())
    a_pay, b_pay = self.position_returns(0), self.position_returns(1)
    res = payoff.match_result(
        self.game,
        self.config,
        self.hands_per_match,
        self.duplicate,
        names[0],
        names[1],
        rules[0],
        rules[1],
        a_pay,
        b_pay,
    )
    if scorer is None:
      return res
    sa, sb = self.deal_scores(0, 0, scorer), self.deal_scores(1, 0, scorer)
    return res._replace(
        mean=stats.estimate(sa),
        diff=stats.estimate(sa - sb),
        se_unpaired=float("nan"),
        variance_reduction=float("nan"),
    )

  # -- helpers -----------------------------------------------------------------
  def _position(self, k) -> int:
    """A side position, checked.

    Args:
      k: The position.

    Returns:
      It, an int.

    Raises:
      ValueError: Out of range.
    """
    k = int(k)
    if not 0 <= k < self.sides.num_sides:
      raise ValueError(f"position {k} not in 0..{self.sides.num_sides - 1}")
    return k

  def _entrant(self, e) -> int:
    """An entrant's index, by name or index.

    Args:
      e: A name or an index.

    Returns:
      The index.

    Raises:
      ValueError: No such entrant.
    """
    if isinstance(e, str):
      if e not in self.entrants:
        raise ValueError(f"no entrant {e!r} in {self.entrants}")
      return self.entrants.index(e)
    i = int(e)
    if not 0 <= i < len(self.entrants):
      raise ValueError(f"entrant {i} not in 0..{len(self.entrants) - 1}")
    return i


def table_result(
    game: str,
    config: str,
    design: designs_lib.Design,
    entrants,
    rules,
    lineup,
    returns,
    hands_per_match: int,
    duplicate: bool,
) -> TableResult:
  """The TableResult of per-seat returns of the design's rows of `lineup`.

  Args:
    game: the env's name.
    config: the EvaluatorConfig's name.
    design: the eval.designs.Design played.
    entrants: entrant names; lineups index into them.
    rules: each entrant's DecisionRule.
    lineup: i32[n, B, P].
    returns: f64[n, R, P], each seat's payoff.
    hands_per_match: hands per row.
    duplicate: rows of a deal share its deal key.

  Returns:
    The TableResult.

  Raises:
    TypeError: a rule is not a DecisionRule.
  """
  lineup = np.asarray(lineup, np.int32)
  returns = np.asarray(returns, np.float64)
  for rule in rules:
    if not isinstance(rule, types.DecisionRule):
      raise TypeError(f"table_result: {rule!r} is not a DecisionRule")
  return TableResult(
      game=game,
      config=config,
      design=design,
      entrants=tuple(entrants),
      rules=tuple(rules),
      lineup=lineup,
      seats=designs_lib.table_rows(design, lineup),
      returns=returns,
      side_returns=np.asarray(design.sides.side_returns(returns), np.float64),
      hands_per_match=int(hands_per_match),
      duplicate=bool(duplicate),
  )
