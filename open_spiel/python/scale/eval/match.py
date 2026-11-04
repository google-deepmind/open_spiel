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

"""Head-to-head evaluation of two-player policies on duplicate deals.

A win rate between two policies is only as useful as its error bar. The
Evaluator plays matches in seat-swapped duplicate pairs, so luck of the deal
cancels within each pair, and reports every mean with its standard error and
95% interval (eval.stats).

  ev = match.Evaluator(env, match.EvaluatorConfig(hands_per_match=1))
  a = match.Entrant.from_artifact(ckpt.artifact.load_artifact(run_dir, step))
  b = match.Entrant("uniform", uniform_policy, {}, types.SAMPLE)
  res = ev.head_to_head(a, b, n_deals=4096, key=jax.random.key(0))
  res.mean.mean, res.mean.lo, res.mean.hi       # a's payoff per match, 95% CI

Duplicate deals. Deal i is played twice with the seats swapped: row (i, 0)
puts a in seat 0, row (i, 1) puts a in seat 1. With k_i = fold_in64(pair_key,
i), both rows of deal i share the deal key and draw separate policy keys
(core.arena.duplicate_keys), so they see the same cards, board and stock
order while stochastic play stays independent. `duplicate=False` gives every
row its own deal (core.arena.independent_keys): the same estimator without
pairing, useful to measure the variance pairing removes. Keys are per deal,
so results do not depend on chunking or on which other pairs share the
batch: a payoff_matrix cell equals head_to_head under that pair's key.

Statistics (f64). The unit is the deal pair. a_payoff f64[n, 2] holds a's
payoff per match with a in seat r in column r. `mean` is the pair mean with
its SE and 95% CI; `seat[r]` the per-seat estimates; `diff` the paired
difference a - b per match (twice the mean in a zero-sum game);
`se_unpaired` the SE the same 2n games would have if independent, and
`variance_reduction` = (se_unpaired / se)**2.

Population mixtures (`mixture_head_to_head`, the population-vs-population
metric of PSRO). Population a under meta-strategy wa plays population b under
wb: deal i draws members I_i ~ wa and J_i ~ wb and plays the duplicate pair
a[I_i] vs b[J_i] on deal i exactly as head_to_head would. The mean estimates
sum_ij wa_i wb_j E[payoff of a_i vs b_j], the per-game meta-mixture, which is
realization-equivalent to the mixed strategy by Kuhn's theorem
(league.mixture). Each population plays one decision rule.

Slots. Every match runs through one Arena (core.arena_engine): `Arena` when
every entrant is in-graph, `HostArena` when one is HOST_DRIVEN. Entrants that
can share compiled code share a slot with their params stacked [K, ...]: two
entrants share a slot when their policies return the same non-None
`stack_key()` and play the same rule. A policy without `stack_key` shares by
architecture and arch_kwargs when it is NATIVE_JAX, and by object identity
otherwise.

A match is `hands_per_match` hands. Each row is rolled for hands_per_match *
(env.max_len + 1) plies; a row that does not finish its hands raises
(payoff.payoffs).

This module does not import JAX (numpy only). Payoff accounting lives in
eval.payoff and play goes through a MatchEngine; host-side draws (mixture
members, pair keys) use core.rng, which accepts jax or numpy keys.
"""

import dataclasses
import json
from typing import Any, Callable, Optional, Sequence

import numpy as np

from open_spiel.python.scale.core import deadline as deadline_lib
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import slots as slots_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import payoff


@dataclasses.dataclass(frozen=True, eq=False)
class Entrant:
  """One side of a match: a name, a policy, one member's params and a rule.

  Attributes:
    name: the entrant's name (non-empty).
    policy: the policy.
    params: an unstacked JAX pytree for an in-graph policy, or the host
      object a HOST_DRIVEN policy takes per member.
    rule: the decision rule.
  """

  name: str
  policy: types.Policy
  params: Any
  rule: types.DecisionRule

  def __post_init__(self):
    if not (isinstance(self.name, str) and self.name):
      raise ValueError(
          f"an Entrant's name is a non-empty string, got {self.name!r}"
      )
    if not isinstance(self.rule, types.DecisionRule):
      raise TypeError(
          f"Entrant {self.name}: rule {self.rule!r} is not a DecisionRule"
      )
    if not isinstance(getattr(self.policy, "spec", None), types.PolicySpec):
      raise TypeError(
          f"Entrant {self.name}: {type(self.policy).__name__} has no PolicySpec"
      )

  @classmethod
  def from_artifact(
      cls,
      art: types.Artifact,
      rule: Optional[types.DecisionRule] = None,
      name: Optional[str] = None,
  ) -> "Entrant":
    """An Artifact as an entrant.

    The default name is "<run_id>@<step>", or "<path>@<step>" when the
    artifact records no run id; the default rule is the manifest's
    default_rule.

    Args:
      art: the artifact.
      rule: the decision rule (default: the manifest's default_rule).
      name: the entrant's name (default: "<run_id>@<step>").

    Returns:
      the Entrant.
    """
    m = art.manifest
    if name is None:
      run_id = _run_id(m)
      name = (
          f"{run_id}@{m.step}" if run_id is not None else f"{art.path}@{m.step}"
      )
    return cls(
        name, art.policy, art.params, m.default_rule if rule is None else rule
    )


def _run_id(manifest) -> Optional[str]:
  """The manifest's run id as a string, or None.

  Args:
    manifest: The manifest.

  Returns:
    The run id, or None.
  """
  rid = getattr(manifest, "run_id", None)
  return None if rid is None else str(rid)


@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
  """How an Evaluator plays.

  Attributes:
    name: what the evaluation is called in reports.
    hands_per_match: hands per match, the unit of one payoff.
    duplicate: seat-swapped duplicate deals (True) or independent deals.
    rule: if set, overrides every entrant's decision rule (for example ARGMAX).
    wrap: if set, applied to every entrant before play (for example to route
      a game's final phase to a scripted policy).
    reference: if set, builds the fixed opponent of Evaluator.vs_reference.
    max_rows: Arena rows per rollout, a memory bound; results do not depend
      on it. It is rounded down to a multiple of the engine's row unit (two
      rows per device), and a max_rows below one unit runs one unit.
  """

  name: str = "h2h"
  hands_per_match: int = 1
  duplicate: bool = True
  rule: Optional[types.DecisionRule] = None
  wrap: Optional[Callable[[Entrant], Entrant]] = None
  reference: Optional[Callable[[], Entrant]] = None
  max_rows: int = 4096

  def __post_init__(self):
    if self.hands_per_match < 1:
      raise ValueError(
          f"hands_per_match must be >= 1, got {self.hands_per_match}"
      )
    if not (self.rule is None or isinstance(self.rule, types.DecisionRule)):
      raise TypeError(f"rule {self.rule!r} is not a DecisionRule or None")
    if self.max_rows < 2 or self.max_rows % 2:
      raise ValueError(
          f"max_rows must be an even number >= 2, got {self.max_rows}"
      )


def pair_key(key, i: int, j: int):
  """The key payoff_matrix gives the match of entrant i (a) vs entrant j (b).

  Args:
    key: payoff_matrix's key.
    i: entrant a's index.
    j: entrant b's index.

  Returns:
    the pair's key.
  """
  return rng.fold_in(rng.fold_in(key, i), j)


def _slot_key(e: Entrant):
  """Entrants with equal keys share a slot (module doc).

  Args:
    e: The entrant.

  Returns:
    The slot key.
  """
  stack_key = getattr(e.policy, "stack_key", None)
  if stack_key is not None:
    k = stack_key()
    if k is None:
      return ("object", id(e.policy), e.rule)
    return ("stack", type(e.policy), k, e.rule)
  spec = e.policy.spec
  if spec.capability == types.Capability.NATIVE_JAX and spec.framework == "jax":
    # A spec that names an injected function records only its qualname, so two
    # different functions can share a spec; such policies share by identity.
    if types.names_injected(spec.arch_kwargs):
      return ("object", id(e.policy), e.rule)
    kw = json.dumps(dict(spec.arch_kwargs), sort_keys=True)
    return ("spec", type(e.policy), spec.arch_name, kw, e.rule)
  # A host policy may keep state on the object, so it never shares a slot.
  return ("object", id(e.policy), e.rule)


def _weights(w, k: int, what: str) -> np.ndarray:
  """Checks and normalises population weights.

  Args:
    w: The weights.
    k: The members.
    what: The population, for errors.

  Returns:
    f64[k], the weights summing to 1.

  Raises:
    ValueError: The weights do not fit, or do not sum to 1.
  """
  w = np.asarray(w, np.float64)
  if w.shape != (k,) or not np.all(np.isfinite(w)) or np.any(w < 0):
    raise ValueError(
        f"population {what}: weights {w.tolist()} do not fit {k} members "
        "(finite, >= 0)"
    )
  if abs(float(w.sum()) - 1.0) > 1e-6:
    raise ValueError(
        f"population {what}: weights sum to {float(w.sum())}, not 1"
    )
  return w / w.sum()


class Evaluator:
  """Head-to-head matches, payoff matrices and reference matches over one Env.

  Supports 2-player games: duplicate deals swap two seats and a payoff is a's
  result against b. The constructor raises for any other player count.

  Attributes:
    env: the Env.
    config: the EvaluatorConfig.
    mesh: the device mesh of the default ArenaEngine, or None.
    engine: the MatchEngine every match plays through.
  """

  def __init__(
      self,
      env: types.Env,
      config: EvaluatorConfig = EvaluatorConfig(),
      mesh=None,
      engine: Optional[payoff.MatchEngine] = None,
  ):
    if env.num_players != 2:
      raise ValueError(
          f"{env.name}: head-to-head is defined for 2 players (got "
          f"{env.num_players})"
      )
    if not isinstance(config, EvaluatorConfig):
      raise TypeError(
          f"config is a {type(config).__name__}, not an EvaluatorConfig"
      )
    self.env, self.config, self.mesh = env, config, mesh
    if engine is None:
      # Lazy: an optional dependency (arena_engine imports JAX; this module is
      # numpy only, module doc).
      from open_spiel.python.scale.core import arena_engine  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      engine = arena_engine.ArenaEngine(env, config.hands_per_match, mesh)
    else:
      if mesh is not None:
        raise ValueError(
            "a mesh configures the default ArenaEngine; pass it to the engine"
        )
    self.engine = engine

  # -- public ------------------------------------------------------------------
  def head_to_head(
      self,
      a: Entrant,
      b: Entrant,
      n_deals: int,
      key,
      progress: Optional[Callable[[np.ndarray], None]] = None,
  ) -> payoff.MatchResult:
    """A against b over n_deals duplicate deal pairs.

    Args:
      a: the first entrant.
      b: the second entrant.
      n_deals: deal pairs (at least 2, for an SE).
      key: the match key.
      progress: called after every row chunk with a's payoffs f64[d, 2] of
        the d deal pairs played so far, deals 0..d-1 in order (for a running
        estimate). The result does not depend on it.

    Returns:
      the MatchResult of a against b.
    """
    return self._play([a, b], [(0, 1)], n_deals, [key], progress=progress)[
        (0, 1)
    ]

  def vs_reference(
      self, subject: Entrant, n_deals: int, key
  ) -> payoff.MatchResult:
    """Subject (a) against the config's reference opponent (b).

    Args:
      subject: the entrant under test.
      n_deals: deal pairs.
      key: the match key.

    Returns:
      the MatchResult of subject against the reference.

    Raises:
      ValueError: the config has no reference.
    """
    if self.config.reference is None:
      raise ValueError(f"{self.config.name}: no reference configured")
    return self.head_to_head(subject, self.config.reference(), n_deals, key)

  def mixture_head_to_head(
      self,
      a: Sequence[Entrant],
      wa,
      b: Sequence[Entrant],
      wb,
      n_deals: int,
      key,
      a_name: str = "a",
      b_name: str = "b",
  ) -> payoff.MixtureMatch:
    """Population a under meta-strategy wa against population b under wb.

    Args:
      a: population a's entrants.
      wa: a's meta-strategy, f64 weights that sum to 1.
      b: population b's entrants.
      wb: b's meta-strategy, f64 weights that sum to 1.
      n_deals: as in head_to_head.
      key: as in head_to_head.
      a_name: the name the result reports for a.
      b_name: the name the result reports for b.

    Returns:
      A MixtureMatch: the MatchResult of a_name against b_name and the member
      each deal pair drew.

    Raises:
      ValueError: an empty population, weights that do not fit, or a
        population whose members play different decision rules.
    """
    a, b = list(a), list(b)
    if not a or not b:
      raise ValueError("mixture_head_to_head: empty population")
    wa, wb = _weights(wa, len(a), a_name), _weights(wb, len(b), b_name)
    ready = [
        self._prepare(e) for e in a + b
    ]  # once per member: wrap may be costly
    for name, members in ((a_name, ready[: len(a)]), (b_name, ready[len(a) :])):
      rules = {e.rule for e in members}
      if len(rules) != 1:
        raise ValueError(
            f"population {name}: members play different decision rules "
            f"{sorted(map(str, rules))} (one rule per population)"
        )
    k_play, k_draw = rng.split(key)
    ka, kb = rng.split(k_draw)
    ia = np.asarray(rng.choice(ka, len(a), (n_deals,), p=wa), np.int32)
    ib = np.asarray(rng.choice(kb, len(b), (n_deals,), p=wb), np.int32)
    assert not np.any(wa[ia] == 0.0) and not np.any(
        wb[ib] == 0.0
    ), "drew a zero-weight member"
    entrants = a + b
    prepared, a_pay, b_pay = self._play_jobs(
        entrants, [(ia, ib + len(a))], n_deals, [k_play], prepared=ready
    )
    pa, pb = prepared[0], prepared[len(a)]
    res = self._result(
        dataclasses.replace(pa, name=a_name),
        dataclasses.replace(pb, name=b_name),
        a_pay[0],
        b_pay[0],
    )
    return payoff.MixtureMatch(
        result=res, a_members=ia, b_members=ib, a_weights=wa, b_weights=wb
    )

  def payoff_matrix(
      self, entrants: Sequence[Entrant], n_deals: int, key
  ) -> payoff.PayoffMatrix:
    """The payoff matrix of every pair of entrants.

    Every pair (i <= j) of entrants, the match of i against j under
    pair_key(key, i, j); cell (j, i) comes from the same games.

    Args:
      entrants: the entrants.
      n_deals: deal pairs per pair of entrants.
      key: the matrix key.

    Returns:
      the PayoffMatrix.
    """
    k = len(entrants)
    assert k >= 1
    pairs = []
    for i in range(k):
      for j in range(i, k):
        pairs.append((i, j))
    res = self._play(
        list(entrants), pairs, n_deals, [pair_key(key, i, j) for i, j in pairs]
    )
    return payoff.matrix_from_results(
        tuple(e.name for e in entrants), res, n_deals
    )

  # -- internals ---------------------------------------------------------------
  def _prepare(self, e: Entrant) -> Entrant:
    """An entrant as this evaluator plays it: wrapped, rule set and checked.

    Args:
      e: the entrant.

    Returns:
      the entrant after config.wrap and config.rule.

    Raises:
      TypeError: e, or what config.wrap returns, is not an Entrant.
      ValueError: its policy's action or player count differs from the game's.
    """
    if not isinstance(e, Entrant):
      raise TypeError(f"want an Entrant, got {type(e).__name__}")
    if self.config.wrap is not None:
      e = self.config.wrap(e)
      if not isinstance(e, Entrant):
        raise TypeError(
            f"config.wrap returned a {type(e).__name__}, not an Entrant"
        )
    if self.config.rule is not None:
      e = dataclasses.replace(e, rule=self.config.rule)
    spec, env = e.policy.spec, self.env
    if (spec.num_actions, spec.num_players) != (
        env.num_actions,
        env.num_players,
    ):
      raise ValueError(
          f"entrant {e.name!r}: its policy has {spec.num_actions} actions and "
          f"{spec.num_players} players, the game {env.name!r} has "
          f"{env.num_actions} and {env.num_players}"
      )
    return e

  def _play(
      self, entrants, pairs, n: int, keys, progress=None
  ) -> dict[tuple[int, int], payoff.MatchResult]:
    """Plays each pair's match of n hands.

    Args:
      entrants: The entrants.
      pairs: (i, j) pairs.
      n: Hands per match.
      keys: The keys.
      progress: An optional progress callback.

    Returns:
      {(i, j): MatchResult}.
    """
    jobs = [
        (np.full((n,), i, np.int32), np.full((n,), j, np.int32))
        for i, j in pairs
    ]
    prepared, a_pay, b_pay = self._play_jobs(
        entrants, jobs, n, keys, progress=progress
    )
    return {
        (i, j): self._result(prepared[i], prepared[j], a_pay[q], b_pay[q])
        for q, (i, j) in enumerate(pairs)
    }

  def _play_jobs(
      self, entrants, jobs, n: int, keys, progress=None, prepared=None
  ):
    """(prepared entrants, a_pay f64[n, 2] per job, b_pay per job).

    A job is (ia i32[n], ib i32[n]): deal pair d plays entrant ia[d] (a)
    against entrant ib[d] (b), seat-swapped, under the job's key. progress
    is head_to_head's and needs exactly one job. prepared: the entrants
    already through _prepare (one per entrant), so none is prepared twice.

    Args:
      entrants: the entrants.
      jobs: the jobs, each (ia i32[n], ib i32[n]).
      n: deal pairs per job.
      keys: one key per job.
      progress: head_to_head's progress callback, or None.
      prepared: the entrants already through _prepare, or None.

    Returns:
      (prepared entrants, a_pay f64[n, 2] per job, b_pay f64[n, 2] per job).

    Raises:
      ValueError: n < 2, or progress with more than one job.
    """
    if n < 2:
      raise ValueError(f"n_deals={n}: an SE needs at least 2 deal pairs")
    assert len(jobs) == len(keys), (len(jobs), len(keys))
    if progress is not None and len(jobs) != 1:
      raise ValueError(
          f"progress reports one job's running payoffs, got {len(jobs)} jobs"
      )
    cfg = self.config
    if prepared is None:
      prepared = [self._prepare(e) for e in entrants]
    else:
      prepared = list(prepared)
      assert len(prepared) == len(entrants), (len(prepared), len(entrants))
    # One slot per slot key (_slot_key), its members stacked.
    group_of, members, slot_rule, slot_policy = {}, [], [], []
    seat_of = []  # entrant -> (slot, member)
    for e in prepared:
      g = _slot_key(e)
      if g not in group_of:
        group_of[g] = len(members)
        members.append([])
        slot_rule.append(e.rule)
        slot_policy.append(e.policy)
      s = group_of[g]
      members[s].append(e.params)
      seat_of.append((s, len(members[s]) - 1))
    slots = [
        slots_lib.Slot(p, len(m), r)
        for p, m, r in zip(slot_policy, members, slot_rule)
    ]
    params = [self.engine.stack(sl, m) for sl, m in zip(slots, members)]
    sig = tuple((g, len(members[s])) for g, s in group_of.items())
    slot_arr = np.array([so[0] for so in seat_of], np.int32)
    member_arr = np.array([so[1] for so in seat_of], np.int32)

    unit = int(self.engine.row_unit())
    assert unit >= 2 and unit % 2 == 0, unit
    hands = cfg.hands_per_match
    num_steps = hands * (self.env.max_len + 1)
    a_out, b_out = [None] * len(jobs), [None] * len(jobs)
    by_slot = {}
    for q, (ia, ib) in enumerate(jobs):
      ia, ib = np.asarray(ia, np.int32), np.asarray(ib, np.int32)
      assert ia.shape == ib.shape == (n,), (ia.shape, ib.shape, n)
      persp = set(slot_arr[ia].tolist())
      assert (
          len(persp) == 1
      ), f"job {q}: a's entrants sit in several slots {persp}"
      by_slot.setdefault(persp.pop(), []).append(q)
    for persp, qs in by_slot.items():
      tabs = []
      for q in qs:
        ia, ib = (np.asarray(x, np.int32) for x in jobs[q])
        tabs.append(
            payoff.duplicate_table(
                slot_arr[ia], member_arr[ia], slot_arr[ib], member_arr[ib]
            )
        )
      deal_key, policy_key = self.engine.job_keys(
          [keys[q] for q in qs], n, cfg.duplicate
      )
      table = slots_lib.SlotTable(*(np.concatenate(x) for x in zip(*tabs)))
      rows = table.seat_slot.shape[0]
      # max_rows rounded down to a multiple of the engine's row unit, at
      # least one unit, and no more than the rows rounded up to a unit.
      bound = max(cfg.max_rows - cfg.max_rows % unit, unit)
      chunk = min(bound, -(-rows // unit) * unit)
      pay = np.zeros((rows, 2))
      for start in range(0, rows, chunk):
        deadline_lib.check(f"{cfg.name}: rows {start}.. of {rows}")
        idx = np.arange(start, min(start + chunk, rows))
        full = np.concatenate([idx, np.full(chunk - idx.size, idx[0])])
        sub = slots_lib.SlotTable(*(np.asarray(x)[full] for x in table))
        rewards, done = self.engine.play(
            slots,
            params,
            sub,
            deal_key[full],
            policy_key[full],
            num_steps,
            perspective=persp,
            cache_key=sig,
        )
        pay[idx] = payoff.payoffs(rewards, done, hands)[: idx.size]
        if progress is not None:
          # Rows are (deal, seat) in order and a chunk is a multiple of 2 rows:
          # the first `end` rows are end / 2 whole deal pairs.
          end = int(idx[-1]) + 1
          assert end % 2 == 0 or end == rows, (end, rows)
          seat = np.asarray(table.perspective_seat)[:end]
          progress(pay[np.arange(end), seat].reshape(end // 2, 2))
      a_seat = np.asarray(table.perspective_seat)
      r_all = np.arange(rows)
      a_rows, b_rows = pay[r_all, a_seat], pay[r_all, 1 - a_seat]
      for k, q in enumerate(qs):
        sl = slice(k * 2 * n, (k + 1) * 2 * n)
        a_out[q], b_out[q] = a_rows[sl].reshape(n, 2), b_rows[sl].reshape(n, 2)
    return prepared, a_out, b_out

  def _result(
      self, a: Entrant, b: Entrant, a_pay: np.ndarray, b_pay: np.ndarray
  ):
    """The MatchResult of a against b.

    Args:
      a: The first entrant.
      b: The second entrant.
      a_pay: a's payoffs.
      b_pay: b's payoffs.

    Returns:
      The MatchResult.
    """
    return payoff.match_result(
        self.env.name,
        self.config.name,
        self.config.hands_per_match,
        self.config.duplicate,
        a.name,
        b.name,
        a.rule,
        b.rule,
        a_pay,
        b_pay,
    )


def head_to_head(
    env: types.Env,
    a: Entrant,
    b: Entrant,
    n_deals: int,
    key,
    config: EvaluatorConfig = EvaluatorConfig(),
) -> payoff.MatchResult:
  """Evaluator(env, config).head_to_head(a, b, n_deals, key).

  Args:
    env: the game's Env.
    a: the first entrant.
    b: the second entrant.
    n_deals: deal pairs.
    key: the match key.
    config: the evaluator config.

  Returns:
    the MatchResult of a against b.
  """
  return Evaluator(env, config).head_to_head(a, b, n_deals, key)


def payoff_matrix(
    env: types.Env,
    entrants: Sequence[Entrant],
    n_deals: int,
    key,
    config: EvaluatorConfig = EvaluatorConfig(),
) -> payoff.PayoffMatrix:
  """Evaluator(env, config).payoff_matrix(entrants, n_deals, key).

  Args:
    env: the game's Env.
    entrants: the entrants.
    n_deals: deal pairs per pair of entrants.
    key: the matrix key.
    config: the evaluator config.

  Returns:
    the PayoffMatrix.
  """
  return Evaluator(env, config).payoff_matrix(entrants, n_deals, key)
