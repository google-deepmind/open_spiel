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

"""Head-to-head and table evaluation of policies on duplicate deals.

For any sides.

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

Tables (any sides: core.sides). A table of T sides is played under a
seating design (eval.designs) and recorded as an eval.tables.TableResult,
the primitive every N-player number is estimated from:

  # fixed entrants, design "auto"
  ev.table([e0, e1, e2], n_deals, key)
  # every other side drawn per deal
  ev.vs_field(subject, field, weights, n, key)
  # the N-player gate
  ev.substitute(subject, incumbent, field, weights, n, key)
  # any lineups
  ev.play_table(entrants, lineups, design, n, key)

vs_field seats the subject at a side position (`position`, default 0) and
draws every other side's entrant independently per deal from the field
mixture (side k from fold_in(draw key, k)); its value is
res.position_score(position). substitute plays the subject's block and the
incumbent's against the same field draws, with shared policy keys
(eval.designs), so res.block_diff(position) is "S does better than I in I's
place against the field"; self against self is 0 deal by deal. Either way
the design moves the subject over every side of its position's rotation
class (core.sides.Sides.rotation_classes: the sides of its size), and a
design that does not is refused: at uneven sides (2v1, 3v1v1) the position
picks the role (in 3v1v1 position 0 is the team of three, which no design
moves; a lone side's position rotates over both lone sides).
Every table's constant-sum invariant is checked when the sides are
constant-sum (TableResult.check_sum).

Dispatch on sides. Two seats (every two-player game): head_to_head,
payoff_matrix and mixture_head_to_head are the code above, unchanged. Two
sides of several seats (bridge's partnerships): the same methods play swap2
tables and report TableResult.as_head_to_head, scored by the env's pair
scorer when it declares one (core.sides.pair_scorer_of; bridge: IMPs), so a
payoff cell is side against side. More than two sides: those methods raise
(a pairwise payoff is not defined); use the table methods.

This module does not import JAX (numpy only). Payoff accounting lives in
eval.payoff and play goes through a MatchEngine; host-side draws (mixture
members, pair keys) use core.rng, which accepts jax or numpy keys.
"""

import dataclasses
import functools
import json
from typing import Any, Callable, Optional, Sequence

import numpy as np

from open_spiel.python.scale.core import deadline as deadline_lib
from open_spiel.python.scale.core import hooks
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import slots as slots_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import designs as designs_lib
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.eval import stats
from open_spiel.python.scale.eval import tables


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


#: The optional hooks of a payoff evaluator (core.hooks): the League's gate
#: evaluator (League.gate_cell) and a run's in-loop evaluator
#: (league.runner.eval_estimate) are read through it.
EVALUATOR_HOOKS = hooks.Protocol(
    "evaluator",
    [
        hooks.value(
            "sides",
            "the core.sides.Sides it plays; absent: two sides (the "
            "head-to-head against the reference)",
            check=lambda s: isinstance(s, sides_lib.Sides),
        ),
        hooks.method(
            "gate_estimate",
            "gate_estimate(subject, incumbent, field_entrants, weights, "
            "n_deals, key) -> stats.Estimate: replaces the League's gate "
            "estimate (League.gate_cell)",
        ),
    ],
)

#: The hooks of a match engine (eval.payoff.MatchEngine) beyond head to head.
ENGINE_HOOKS = hooks.Protocol(
    "match engine",
    [
        hooks.method(
            "table_keys",
            "table_keys(keys, n, plan, num_players): the per-table keys of "
            "a table match (Evaluator.table); an engine without it plays "
            "two-player head-to-head only",
        ),
    ],
)


class Evaluator:
  """Head-to-head matches, payoff matrices, reference matches and tables.

  Over one Env. The pairwise methods need two sides (module doc, "Dispatch on
  sides"); the table methods take any sides.

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

  @functools.cached_property
  def sides(self) -> sides_lib.Sides:
    """The env's core.sides.Sides.

    Read on first use: the two-seat paths never need it.

    Returns:
      The Sides.
    """
    return sides_lib.sides_of(self.env)

  @functools.cached_property
  def pair_scorer(self) -> Optional[Callable[..., Any]]:
    """The env's declared pair scorer, or None.

    The mean of the two rows, exactly the two-player accounting.

    Returns:
      The pair scorer, or None.
    """
    if sides_lib.ENV_HOOKS.get(self.env, "pair_scorer") is None:
      return None
    return sides_lib.pair_scorer_of(self.env)

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
        estimate). The result does not depend on it. Two seats only.

    Returns:
      the MatchResult of a against b.

    Raises:
      ValueError: more than two sides (module doc, "Dispatch on sides").
    """
    if self.env.num_players == 2:
      return self._play([a, b], [(0, 1)], n_deals, [key], progress=progress)[
          (0, 1)
      ]
    self._need_two_sides("head_to_head")
    if progress is not None:
      raise ValueError(
          "head_to_head: progress reports two-seat deal pairs; this table has "
          f"{self.env.num_players} seats"
      )
    return self.table([a, b], n_deals, key, "swap2").as_head_to_head(
        scorer=self.pair_scorer
    )

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
    self._need_two_sides("mixture_head_to_head")
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
    if self.env.num_players != 2:
      lineup = np.stack([ia, ib + len(a)], axis=1)[:, None, :]
      tab = self._play_table(
          ready, lineup, designs_lib.swap2(self.sides), n_deals, k_play
      )
      res = tab.as_head_to_head(a_name, b_name, scorer=self.pair_scorer)
      return payoff.MixtureMatch(
          result=res, a_members=ia, b_members=ib, a_weights=wa, b_weights=wb
      )
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
    self._need_two_sides("payoff_matrix")
    k = len(entrants)
    assert k >= 1
    pairs = []
    for i in range(k):
      for j in range(i, k):
        pairs.append((i, j))
    if self.env.num_players != 2:
      return self._side_matrix(list(entrants), pairs, n_deals, key)
    res = self._play(
        list(entrants), pairs, n_deals, [pair_key(key, i, j) for i, j in pairs]
    )
    return payoff.matrix_from_results(
        tuple(e.name for e in entrants), res, n_deals
    )

  # -- tables (any sides) ------------------------------------------------------
  def table(
      self, entrants: Sequence[Entrant], n_deals: int, key, design="auto"
  ) -> tables.TableResult:
    """One fill per side position (len T, fixed over the deals).

    Under `design` (a name, "auto", or an eval.designs.Design with one block).
    A fill is an Entrant (every seat of the side) or a sequence of Entrants, one
    per seat of the side in increasing seat order (a mixed partnership). The
    result records the entrant of every seat (TableResult.lineup and seats);
    res.position_score(k) is side k's return, res.seat_score(q) base seat q's
    own return.

    Args:
      entrants: one fill per side.
      n_deals: deals (at least 2).
      key: the job key.
      design: a design name, "auto", or a one-block Design.

    Returns:
      The TableResult.

    Raises:
      ValueError: not one fill per side, a design of several blocks, or a
        per-seat fill of the wrong size.
      TypeError: a per-seat fill holds something other than an Entrant.
    """
    d = self._design(design)
    sides = self.sides
    t = sides.num_sides
    fills = list(entrants)
    if len(fills) != t or d.num_blocks != 1:
      raise ValueError(
          f"table: {len(fills)} fills for {t} sides, design {d.name} with "
          f"{d.num_blocks} blocks (want one entrant per side, or per seat of a "
          "side, and one block)"
      )
    if all(isinstance(f, Entrant) for f in fills):
      lineup = np.broadcast_to(np.arange(t, dtype=np.int32), (n_deals, 1, t))
      return self.play_table(fills, lineup, d, n_deals, key)
    flat, seat_entrant = [], np.zeros(sides.num_players, np.int32)
    for k, f in enumerate(fills):
      seats = sides.seats(k)
      per_seat = [f] * len(seats) if isinstance(f, Entrant) else list(f)
      if len(per_seat) != len(seats):
        raise ValueError(
            f"table: side {k} has {len(seats)} seats {seats}, its fill names "
            f"{len(per_seat)} entrants"
        )
      for q, e in zip(seats, per_seat):
        if not isinstance(e, Entrant):
          raise TypeError(
              f"table: side {k} fill holds a {type(e).__name__}, not an Entrant"
          )
        same = [i for i, x in enumerate(flat) if x is e]
        if not same:
          flat.append(e)
        seat_entrant[q] = same[0] if same else len(flat) - 1
    lineup = np.broadcast_to(seat_entrant, (n_deals, 1, sides.num_players))
    return self.play_table(flat, lineup, d, n_deals, key)

  def vs_field(
      self,
      subject: Entrant,
      field: Sequence[Entrant],
      weights,
      n_deals: int,
      key,
      design="auto",
      position: int = 0,
  ) -> tables.TableResult:
    """Subject at `position`, every other side drawn per deal from the field.

    The field mixture (module doc); res.position_score(position) is its value
    against the field, over every side of the position's rotation class (the
    design moves it there, and a design that does not is refused).

    Args:
      subject: the entrant under test.
      field: the field's entrants.
      weights: the field's weights.
      n_deals: deals (at least 2).
      key: the job key.
      design: a design name, "auto", or a one-block Design.
      position: the subject's base side (0..T-1); at uneven sides it picks
        the subject's role (module doc).

    Returns:
      The TableResult.

    Raises:
      ValueError: a design of several blocks, a position out of range, or a
        design that does not rotate the position over its class.
    """
    d = self._design(design)
    if d.num_blocks != 1:
      raise ValueError(
          f"vs_field plays one lineup block, design {d.name} has {d.num_blocks}"
      )
    position = self._subject_position(d, position, "vs_field")
    field = list(field)
    w = _weights(weights, len(field), "field")
    k_play, k_draw = rng.split(key)
    t = self.sides.num_sides
    others = [k for k in range(t) if k != position]
    lineup = np.zeros((n_deals, 1, t), np.int32)
    lineup[:, 0, others] = 1 + self._draw_field(k_draw, w, n_deals, others)
    return self.play_table([subject] + field, lineup, d, n_deals, k_play)

  def substitute(
      self,
      subject: Entrant,
      incumbent: Entrant,
      field: Sequence[Entrant],
      weights,
      n_deals: int,
      key,
      position: int = 0,
  ) -> tables.TableResult:
    """The subject's block and the incumbent's against the same field draws.

    Each cyclic+classes over the sides (cyclic at equal sides), with shared
    policy keys (eval.designs "substitute"); the subject (block 0) and the
    incumbent (block 1) hold base side `position`, which the design moves
    over every side of its rotation class; res.block_diff(position) is the
    gate statistic.

    Args:
      subject: the entrant under test.
      incumbent: the entrant it would replace.
      field: the field's entrants.
      weights: the field's weights.
      n_deals: deals (at least 2).
      key: the job key.
      position: the base side the subject and incumbent hold (0..T-1); at
        uneven sides it picks their role (module doc).

    Returns:
      The TableResult.

    Raises:
      ValueError: a position out of range.
    """
    d = designs_lib.substitute(self.sides)
    position = self._subject_position(d, position, "substitute")
    field = list(field)
    w = _weights(weights, len(field), "field")
    k_play, k_draw = rng.split(key)
    t = self.sides.num_sides
    others = [k for k in range(t) if k != position]
    lineup = np.zeros((n_deals, 2, t), np.int32)
    lineup[:, 1, position] = 1
    lineup[:, :, others] = (2 + self._draw_field(k_draw, w, n_deals, others))[
        :, None, :
    ]
    return self.play_table(
        [subject, incumbent] + field, lineup, d, n_deals, k_play
    )

  def play_table(
      self,
      entrants: Sequence[Entrant],
      lineups,
      design: designs_lib.Design,
      n_deals: int,
      key,
  ) -> tables.TableResult:
    """n_deals deals of `design` under `key`.

    One job: deal i keyed fold_in64(key, i), core.arena_engine.table_keys.

    Args:
      entrants: the entrants lineups index.
      lineups: i32[n, B, T] (one entrant per side) or i32[n, B, P] (per
        seat, partners may differ): block b's lineup on deal i.
      design: an eval.designs.Design for this env's sides.
      n_deals: deals (at least 2).
      key: the job key.

    Returns:
      The TableResult.
    """
    entrants = list(entrants)
    return self._play_table(
        [self._prepare(e) for e in entrants], lineups, design, n_deals, key
    )

  # -- internals ---------------------------------------------------------------
  def _need_two_sides(self, what: str) -> None:
    if self.env.num_players == 2:
      return
    if self.sides.num_sides != 2:
      raise ValueError(
          f"{self.env.name}: {what} is pairwise (swap2), a design not valid "
          f"for these sides ({self.sides.num_sides} sides, {self.sides}); use "
          "Evaluator.table, vs_field or substitute"
      )

  def _design(self, design) -> designs_lib.Design:
    """A design by name or object, checked against the env's sides.

    Args:
      design: A design name or an eval.designs.Design.

    Returns:
      The Design.

    Raises:
      TypeError: Neither a name nor a Design.
      ValueError: The design is for other sides.
    """
    d = (
        designs_lib.make(design, self.sides)
        if isinstance(design, str)
        else design
    )
    if not isinstance(d, designs_lib.Design):
      raise TypeError(
          f"design is a name or an eval.designs.Design, got {type(d).__name__}"
      )
    if d.sides != self.sides:
      raise ValueError(
          f"design {d.name} is for {d.sides}, the env's sides are {self.sides}"
      )
    return d

  def _subject_position(self, d, position, what: str) -> int:
    """The subject's base side, checked against the design (module doc).

    Args:
      d: The Design.
      position: The subject's base side.
      what: The caller, for messages.

    Returns:
      The position, an int.

    Raises:
      TypeError: position is not an int.
      ValueError: position is out of range or not held over its rotation
        class.
    """
    if isinstance(position, bool) or not isinstance(
        position, (int, np.integer)
    ):
      raise TypeError(f"{what}: position is an int, got {position!r}")
    position = int(position)
    t = self.sides.num_sides
    if not 0 <= position < t:
      raise ValueError(f"{what}: position {position} not in 0..{t - 1}")
    cls = next(c for c in self.sides.rotation_classes if position in c)
    held = sorted(set(d.side_perms[:, position].tolist()))
    if held != sorted(cls):
      raise ValueError(
          f"{what}: design {d.name} seats position {position} at sides "
          f"{held}, not over its rotation class {cls} (sizes "
          f"{self.sides.side_sizes})"
      )
    return position

  def _draw_field(
      self, key, w: np.ndarray, n: int, positions=None
  ) -> np.ndarray:
    """i32[n, len(positions)]: field member indices of those side positions.

    Default sides 1..T-1; side k drawn from fold_in(key, k).

    Args:
      key: the PRNG key.
      w: f64[K] the field's member weights.
      n: rows to draw.
      positions: the side positions to draw (default 1..T-1).

    Returns:
      i32[n, len(positions)] member indices.
    """
    t = self.sides.num_sides
    positions = range(1, t) if positions is None else positions
    out = np.stack(
        [
            np.asarray(
                rng.choice(rng.fold_in(key, k), w.size, (n,), p=w), np.int32
            )
            for k in positions
        ],
        axis=1,
    )
    assert not np.any(w[out] == 0.0), "drew a zero-weight field member"
    return out

  def _side_matrix(self, entrants, pairs, n: int, key) -> payoff.PayoffMatrix:
    """payoff_matrix at two sides of several seats.

    Swap2 tables, cells side against side, both cells of a pair from one table
    and one scorer.

    Args:
      entrants: the entrants.
      pairs: the (i, j) cells to play, i <= j.
      n: deals per cell.
      key: the PRNG key.

    Returns:
      The PayoffMatrix.
    """
    ready = [self._prepare(e) for e in entrants]
    d = designs_lib.swap2(self.sides)
    k = len(entrants)
    mean, se = np.zeros((k, k)), np.zeros((k, k))
    results = {}
    for i, j in pairs:
      lineup = np.broadcast_to(np.array([i, j], np.int32), (n, 1, 2))
      tab = self._play_table(ready, lineup, d, n, pair_key(key, i, j))
      r = tab.as_head_to_head(scorer=self.pair_scorer)
      results[(i, j)] = r
      mean[i, j], se[i, j] = r.mean.mean, r.mean.se
      if i != j:
        e = stats.estimate(tab.deal_scores(1, 0, self.pair_scorer))
        mean[j, i], se[j, i] = e.mean, e.se
    return payoff.PayoffMatrix(
        names=tuple(e.name for e in ready),
        mean=mean,
        se=se,
        lo=mean - stats.Z95 * se,
        hi=mean + stats.Z95 * se,
        n_deals=n,
        results=results,
    )

  def _slot_layout(self, prepared):
    """The slot layout of the prepared entrants.

    (slots, stacked params, cache signature, slot per entrant, member per
    entrant): one slot per slot key (_slot_key), its members stacked.

    Args:
      prepared: the prepared entrants.

    Returns:
      (slots, stacked params, cache signature, slot per entrant, member per
      entrant).
    """
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
    return slots, params, sig, slot_arr, member_arr

  def _play_table(
      self, prepared, lineups, design: designs_lib.Design, n: int, key
  ) -> tables.TableResult:
    """play_table over entrants already through _prepare.

    Args:
      prepared: The prepared entrants.
      lineups: The lineups, indices into prepared.
      design: The design (name or Design).
      n: Deals per lineup.
      key: The PRNG key.

    Returns:
      The TableResult.

    Raises:
      ValueError: Fewer than 2 deals, or lineups of the wrong shape or range.
      TypeError: The engine plays no tables.
    """
    d = self._design(design)
    sides, cfg = self.sides, self.config
    p, t = sides.num_players, sides.num_sides
    if n < 2:
      raise ValueError(f"n_deals={n}: an SE needs at least 2 deals")
    ENGINE_HOOKS.require(self.engine, "table_keys")
    lineups = np.asarray(lineups, np.int32)
    if lineups.ndim == 3 and lineups.shape[2] == t and t != p:
      lineups = designs_lib.seat_lineup(sides, lineups)
    if lineups.shape != (n, d.num_blocks, p):
      raise ValueError(
          f"lineups shape {lineups.shape}, want ({n}, {d.num_blocks}, {t} "
          f"sides or {p} seats)"
      )
    if lineups.min() < 0 or lineups.max() >= len(prepared):
      raise ValueError(f"lineups index outside {len(prepared)} entrants")
    slots, params, sig, slot_arr, member_arr = self._slot_layout(prepared)
    r = d.num_rows
    rows = n * r
    flat = designs_lib.table_rows(d, lineups).reshape(rows, p)
    seat_slot, seat_member = slot_arr[flat], member_arr[flat]
    # The perspective seat is position 0's first seat, so a row is emitted for
    # its subject's slot; rows are played grouped by that slot.
    persp_seat = np.tile(
        np.array([perm[0] for perm in d.seat_perms], np.int32), n
    )
    persp_slot = seat_slot[np.arange(rows), persp_seat]
    deal_key, policy_key = self.engine.table_keys(
        [key], n, designs_lib.key_plan(d, cfg.duplicate), p
    )
    unit = int(self.engine.row_unit())
    assert unit >= 1, unit
    hands = cfg.hands_per_match
    num_steps = hands * (self.env.max_len + 1)
    bound = max(cfg.max_rows - cfg.max_rows % unit, unit)
    pay = np.zeros((rows, p))
    for persp in sorted(set(persp_slot.tolist())):
      group = np.flatnonzero(persp_slot == persp)
      chunk = min(bound, -(-group.size // unit) * unit)
      for start in range(0, group.size, chunk):
        deadline_lib.check(
            f"{cfg.name}: table rows {start}.. of {group.size} (slot {persp})"
        )
        idx = group[start : start + chunk]
        full = np.concatenate([idx, np.full(chunk - idx.size, idx[0])])
        sub = slots_lib.SlotTable(
            seat_slot[full], seat_member[full], persp_seat[full]
        )
        rewards, done = self.engine.play(
            slots,
            params,
            sub,
            deal_key[full],
            policy_key[full],
            num_steps,
            perspective=int(persp),
            cache_key=sig,
        )
        pay[idx] = payoff.payoffs(rewards, done, hands)[: idx.size]
    res = tables.table_result(
        self.env.name,
        cfg.name,
        d,
        [e.name for e in prepared],
        [e.rule for e in prepared],
        lineups,
        pay.reshape((n, r, p)),
        hands,
        cfg.duplicate,
    )
    res.check_sum()
    return res

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
    slots, params, sig, slot_arr, member_arr = self._slot_layout(prepared)

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
