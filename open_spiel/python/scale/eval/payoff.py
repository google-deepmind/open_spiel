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

"""The Evaluator's payoff accounting and its play-engine protocol, in numpy.

Code that only reads results (the league controller, the meta-solvers) must
not depend on how matches are played. This module holds the result types and
the accounting, and does not import JAX (numpy only).

Accounting: a row's payoff is its summed rewards over its first `hands`
hands (`payoffs`); a seat-swapped duplicate match of n deal pairs is a
MatchResult (`match_result`; the deal pair is the unit, eval.match's module
doc); a payoff matrix is assembled from the pair results, cell (j, i) from
the same games as (i, j) (`matrix_from_results`).

MatchEngine: how the Evaluator plays. eval.match.Evaluator groups entrants
into slots, builds the duplicate slot tables, chunks rows and does the
accounting; an engine stacks a slot's members, derives the per-deal keys and
plays rows:

  row_unit() -> int                 rows per chunk must be a multiple of it
  stack(slot, members) -> params    one slot's params from its members'
  job_keys(keys, n, duplicate) -> (deal_key [J*2n], policy_key [J*2n, 2])
                                    rows (i, r) of each job's n deal pairs
                                    (eval.match's module doc: duplicate
                                    deals share a deal key)
  play(slots, params, table, deal_key, policy_key, num_steps, *,
       perspective, cache_key) -> (rewards f64[B, T, P], done bool[B, T])
                                    T <= num_steps plies of every row
                                    from a fresh match (an engine may stop
                                    once every row has finished its
                                    hands); perspective: the slot of
                                    the rows' perspective seat; cache_key:
                                    hashable, equal for calls of one slot
                                    layout (an engine may cache per key)

core.arena_engine.ArenaEngine, over the Arena / HostArena, is the default
engine.
"""

from typing import Any, NamedTuple, Protocol, Sequence

import numpy as np

from open_spiel.python.scale.core import slots
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import stats


class MatchResult(NamedTuple):
  """One duplicate match of a against b (eval.match's module doc).

  Attributes:
    game: the game's name.
    config: EvaluatorConfig.name.
    a: entrant a's name.
    b: entrant b's name.
    a_rule: a's decision rule.
    b_rule: b's decision rule.
    n_deals: deal pairs; 2 * n_deals matches were played.
    hands_per_match: hands per match.
    duplicate: whether the two rows of a pair shared the deal.
    a_payoff: f64[n, 2]: a's payoff per match, column r = a in seat r.
    b_payoff: f64[n, 2]: b's payoff in the same matches.
    mean: a's payoff per match (pair means).
    seat: (Estimate a in seat 0, Estimate a in seat 1).
    diff: a - b per match (pair means).
    se_unpaired: SE of the mean if the 2n matches were independent.
    variance_reduction: (se_unpaired / mean.se) ** 2; when mean.se == 0: inf
      if se_unpaired > 0, NaN if both are 0.
  """

  game: str
  config: str  # EvaluatorConfig.name
  a: str
  b: str
  a_rule: types.DecisionRule
  b_rule: types.DecisionRule
  n_deals: int  # deal pairs; 2 * n_deals matches were played
  hands_per_match: int
  duplicate: bool
  # f64[n, 2]: a's payoff per match, column r = a in seat r
  a_payoff: np.ndarray
  b_payoff: np.ndarray  # f64[n, 2]: b's payoff in the same matches
  mean: stats.Estimate  # a's payoff per match (pair means)
  # (Estimate a in seat 0, Estimate a in seat 1)
  seat: tuple[stats.Estimate, stats.Estimate]
  diff: stats.Estimate  # a - b per match (pair means)
  se_unpaired: float  # SE of the mean if the 2n matches were independent
  # (se_unpaired / mean.se) ** 2; when mean.se == 0: inf if se_unpaired > 0, NaN
  # if both are 0.
  variance_reduction: float


class MixtureMatch(NamedTuple):
  """A mixture match result.

  A mixture match: the MatchResult of population a against b and the
  member each deal pair drew.

  Attributes:
    result: the MatchResult.
    a_members: i32[n]: the a member of deal pair i (both seat-swapped rows).
    b_members: i32[n]: the b member of deal pair i.
    a_weights: f64[|a|]: population a's meta-strategy.
    b_weights: f64[|b|]: population b's meta-strategy.
  """

  result: MatchResult
  # i32[n]: the a member of deal pair i (both seat-swapped rows)
  a_members: np.ndarray
  b_members: np.ndarray  # i32[n]
  a_weights: np.ndarray  # f64[|a|]
  b_weights: np.ndarray  # f64[|b|]


class PayoffMatrix(NamedTuple):
  """The payoff matrix of a set of entrants, with SEs and 95% CIs.

  mean[i, j]: entrant i's mean payoff per match against entrant j, with SE
  and 95% CI; cell (j, i) comes from the same games as (i, j). The diagonal
  is a real match of each entrant against itself.

  Attributes:
    names: the entrants' names.
    mean: f64[K, K]: mean[i, j] is entrant i's mean payoff against j.
    se: f64[K, K]: the SE of each mean.
    lo: f64[K, K]: the lower end of each 95% CI.
    hi: f64[K, K]: the upper end of each 95% CI.
    n_deals: deal pairs per match.
    results: (i, j), i <= j -> MatchResult of entrant i (a) vs entrant j (b).
  """

  names: tuple[str, ...]
  mean: np.ndarray  # f64[K, K]
  se: np.ndarray
  lo: np.ndarray
  hi: np.ndarray
  n_deals: int
  # (i, j), i <= j -> MatchResult of entrant i (a) vs entrant j (b)
  results: dict[tuple[int, int], MatchResult]


class MatchEngine(Protocol):
  """How the Evaluator plays rows (module doc)."""

  def row_unit(self) -> int:
    """Rows are dealt in multiples of this."""
    ...

  def stack(self, slot: slots.Slot, members: Sequence[Any]) -> Any:
    """A slot's members' params as the engine plays them.

    Args:
      slot: the slot.
      members: its members' params.

    Returns:
      the stacked params.
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def job_keys(
      self, keys: Sequence[Any], n: int, duplicate: bool
  ) -> tuple[Any, Any]:
    """The (deal keys, policy keys) of the rows of every job, concatenated.

    Args:
      keys: one key per job.
      n: deal pairs per job.
      duplicate: whether the two rows of a pair share the deal.

    Returns:
      (deal keys, policy keys).
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def play(
      self,
      slot_list: Sequence[slots.Slot],
      params: Sequence[Any],
      table: slots.SlotTable,
      deal_key: Any,
      policy_key: Any,
      num_steps: int,
      *,
      perspective: int,
      cache_key: Any,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Plays the rows of `table`: (rewards [B, T, P], done [B, T]) on host.

    Args:
      slot_list: the slots.
      params: each slot's stacked params.
      table: the slot table of the rows.
      deal_key: per-row deal keys.
      policy_key: per-row policy keys.
      num_steps: plies to play.
      perspective: the seat the batch is emitted for.
      cache_key: the key of the compiled program.

    Returns:
      (rewards [B, T, P], done [B, T]) on host.
    """
    ...


def payoffs(rewards, done, hands: int) -> np.ndarray:
  """f64[B, P]: each row's summed rewards over its first `hands` hands.

  f64[B, P]: each row's summed rewards over its first `hands` hands, from
  per-ply rewards [B, T, P] and done [B, T].

  Args:
    rewards: per-ply rewards [B, T, P].
    done: per-ply hand ends [B, T].
    hands: hands per row.

  Returns:
    f64[B, P] payoffs.

  Raises:
    RuntimeError: a row did not finish `hands` hands.
  """
  rewards = np.asarray(rewards, np.float64)
  done = np.asarray(done)
  n_done = done.sum(1)
  if not (n_done >= hands).all():
    bad = int(np.flatnonzero(n_done < hands)[0])
    raise RuntimeError(
        f"row {bad} finished {int(n_done[bad])} < {hands} hands in "
        f"{done.shape[1]} plies "
        "(Env.max_len understates the longest hand?)"
    )
  counted = done & (np.cumsum(done, axis=1) <= hands)
  return (rewards * counted[..., None]).sum(1)


def match_result(
    game: str,
    config: str,
    hands_per_match: int,
    duplicate: bool,
    a: str,
    b: str,
    a_rule: types.DecisionRule,
    b_rule: types.DecisionRule,
    a_pay: np.ndarray,
    b_pay: np.ndarray,
) -> MatchResult:
  """The MatchResult of two sides' payoffs over n deal pairs.

  The MatchResult of a's payoffs a_pay f64[n, 2] and b's b_pay (column r:
  a in seat r) over n deal pairs.

  Args:
    game: the game name.
    config: the evaluator config name.
    hands_per_match: hands per match.
    duplicate: whether deals were duplicated across the seat swap.
    a: a's name.
    b: b's name.
    a_rule: a's decision rule.
    b_rule: b's decision rule.
    a_pay: a's payoffs f64[n, 2].
    b_pay: b's payoffs f64[n, 2].

  Returns:
    the MatchResult.
  """
  n = a_pay.shape[0]
  mean = stats.estimate(a_pay.mean(1))
  se_un = float(a_pay.reshape(-1).std(ddof=1) / np.sqrt(2 * n))
  if mean.se > 0:
    vr = (se_un / mean.se) ** 2
  else:
    vr = np.inf if se_un > 0 else np.nan
  return MatchResult(
      game=game,
      config=config,
      a=a,
      b=b,
      a_rule=a_rule,
      b_rule=b_rule,
      n_deals=n,
      hands_per_match=hands_per_match,
      duplicate=duplicate,
      a_payoff=a_pay,
      b_payoff=b_pay,
      mean=mean,
      seat=(stats.estimate(a_pay[:, 0]), stats.estimate(a_pay[:, 1])),
      diff=stats.estimate((a_pay - b_pay).mean(1)),
      se_unpaired=se_un,
      variance_reduction=float(vr),
  )


def matrix_from_results(
    names: Sequence[str],
    results: dict[tuple[int, int], MatchResult],
    n_deals: int,
) -> PayoffMatrix:
  """The PayoffMatrix of pairwise match results.

  The PayoffMatrix of pair results {(i, j), i <= j: MatchResult of i vs j}:
  mean[i, j] from a's payoffs, mean[j, i] from b's in the same games.

  Args:
    names: the entrants' names.
    results: {(i, j), i <= j: MatchResult of i vs j}.
    n_deals: deal pairs per match.

  Returns:
    the PayoffMatrix.
  """
  k = len(names)
  mean, se = np.zeros((k, k)), np.zeros((k, k))
  for (i, j), r in results.items():
    mean[i, j], se[i, j] = r.mean.mean, r.mean.se
    if i != j:
      e = stats.estimate(r.b_payoff.mean(1))
      mean[j, i], se[j, i] = e.mean, e.se
  return PayoffMatrix(
      names=tuple(names),
      mean=mean,
      se=se,
      lo=mean - stats.Z95 * se,
      hi=mean + stats.Z95 * se,
      n_deals=n_deals,
      results=results,
  )


def duplicate_table(sa, ma, sb, mb) -> slots.SlotTable:
  """The duplicate-deal slot table of two players.

  arena.rotated_table(n, [sa, sb], [ma, mb], perspective=0) for two
  players, with per-deal slots / members (i32[n] each): row 2i has a in seat
  0, row 2i + 1 has a in seat 1; the batch is emitted for a's seat.

  Args:
    sa: a's slot per deal, i32[n].
    ma: a's member per deal, i32[n].
    sb: b's slot per deal, i32[n].
    mb: b's member per deal, i32[n].

  Returns:
    the SlotTable of 2n rows.
  """
  sa, ma, sb, mb = (np.asarray(x, np.int32) for x in (sa, ma, sb, mb))
  n = sa.shape[0]
  slot = np.zeros((2 * n, 2), np.int32)
  member = np.zeros((2 * n, 2), np.int32)
  slot[0::2, 0], slot[0::2, 1], member[0::2, 0], member[0::2, 1] = (
      sa,
      sb,
      ma,
      mb,
  )
  slot[1::2, 0], slot[1::2, 1], member[1::2, 0], member[1::2, 1] = (
      sb,
      sa,
      mb,
      ma,
  )
  persp = np.tile(np.array([0, 1], np.int32), n)
  return slots.SlotTable(slot, member, persp)
