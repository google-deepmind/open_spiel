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

"""League controllers: host-side policies that grow and steer a population.

A controller reads a `LeagueView` (the members, the learners and the
Evaluator's payoff matrix over them) and returns actions the League applies
in order. It is deterministic, keeps only small bookkeeping, and updates that
bookkeeping at most once per `view.iteration`, so the League may call `step`
again in the same iteration after re-evaluating new entrants without double
counting.

    ctl = controller.PSRO(meta_solver="nash", iterations=10, br_updates=200)
    actions = ctl.step(view)          # League.tick applies them

Actions:
  SpawnLearner(name, role, init_from)    a new learning slot (fresh, or warm
                                         from a member or another learner)
  RetireLearner(name)                    stop training it
  Admit(learner, member, role, lifecycle)
                                         freeze the learner's params as member
                                         "<base>@<step>": an artifact and a
                                         pool entry
  SetLifecycle(member, lifecycle)        candidate / active / inactive /
                                         archived / evicted (leaves the pool) /
                                         fixed (outside every gate)
  SetMix(learner, mix)                   the learner's opponent distribution
  CopyMutate(src, dst, shrink, perturb)  dst params <- shrink * src + perturb *
                                         a fresh init (PBT)
  Stop(reason)

Controllers:
  PSRO                 best responses to a meta-strategy over the population
                       (Lanctot et al. 2017); with pipeline_depth > 1 it is
                       pipeline PSRO, several best responses at staggered
                       levels (McAleer et al. 2020).
  GatedPool            one main learner against a pool of its snapshots whose
                       members are gated in and out by their payoff
                       intervals, with an adaptive mixing weight.
  ContinualExploiter   an always-on best response to refreshed snapshots of
                       the main learner.
  BROracle             periodic best-response launches against the main
                       learner, admitted when they beat it.
  PBT                  population-based training (Jaderberg et al. 2017).
  LaggedAudit          self-play, a lagged snapshot and a uniform audit pool.
  ScheduledPromotion   self-play against a lagged snapshot plus a pool of
                       harvested snapshots on a fixed schedule.
  FixedMix             one learner with one constant mix (a plain run).
  Composite            several controllers, actions concatenated in order.

This module imports no array framework: payoffs are `eval.payoff`
PayoffMatrix values and keys are `core.rng` keys.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Mapping, Optional, Protocol, Sequence
import zlib

from absl import logging
import numpy as np

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.eval import stats
from open_spiel.python.scale.league import meta_solvers
from open_spiel.python.scale.league import sampler

LIFECYCLES = ("candidate", "active", "inactive", "archived", "evicted", "fixed")
GATED = ("candidate", "active", "inactive", "archived")
_ID_RE = re.compile(r"^([A-Za-z0-9_.\-]+)@(\d+)$")


# -- views ---------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MemberInfo:
  """A population member as a controller sees it.

  Attributes:
    name: the member's artifact id <base>@<step>.
    role: init, psro, exploiter, oracle, target, lagged, snapshot, anchor,
      pbt, ...
    lifecycle: one of LIFECYCLES.
    origin: the learner it was frozen from.
    admitted_at: the league iteration it was admitted at.
  """

  name: str  # artifact id <base>@<step>
  # init, psro, exploiter, oracle, target, lagged, snapshot, anchor, pbt, ...
  role: str
  lifecycle: str
  origin: Optional[str] = None  # the learner it was frozen from
  admitted_at: int = 0  # league iteration

  def __post_init__(self):
    if self.lifecycle not in LIFECYCLES:
      raise ValueError(f"lifecycle {self.lifecycle!r} not in {LIFECYCLES}")
    if not _ID_RE.match(self.name):
      raise ValueError(
          f"member names are artifact ids '<base>@<step>', got {self.name!r}"
      )


@dataclasses.dataclass(frozen=True)
class LearnerInfo:
  """A live learner as a controller sees it.

  Attributes:
    name: the learner name.
    role: the learner's role.
    updates: its update count.
    started_at: the league iteration it was spawned at.
    mix: its opponent Mix.
    scores: one per iteration: expected payoff vs its mix (payoff matrix).
    parent: the member or learner it was initialised from.
  """

  name: str
  role: str
  updates: int
  started_at: int = 0
  mix: sampler.Mix = sampler.Mix(1.0, {})
  # one per iteration: expected payoff vs its mix (payoff matrix)
  scores: tuple[float, ...] = ()
  parent: Optional[str] = None


def payoff_matrix(names: Sequence[str], mean, se=None) -> payoff.PayoffMatrix:
  """A PayoffMatrix from arrays (synthetic payoffs; exact payoffs: se 0).

  Args:
    names: the entrants.
    mean: the mean payoffs, [n, n].
    se: their standard errors; zeros when None.

  Returns:
    The PayoffMatrix.

  Raises:
    ValueError: the arrays are not finite or se is negative.
  """
  names = tuple(names)
  mean = np.asarray(mean, np.float64).reshape(len(names), len(names))
  se = (
      np.zeros_like(mean)
      if se is None
      else np.asarray(se, np.float64).reshape(mean.shape)
  )
  if not (
      np.all(np.isfinite(mean)) and np.all(np.isfinite(se)) and np.all(se >= 0)
  ):
    raise ValueError("payoff mean/se must be finite and se >= 0")
  return payoff.PayoffMatrix(
      names=names,
      mean=mean,
      se=se,
      lo=mean - stats.Z95 * se,
      hi=mean + stats.Z95 * se,
      n_deals=0,
      results={},
  )


@dataclasses.dataclass(frozen=True)
class GateRequest:
  """A controller's request for a promotion gate cell.

  `subject` (a live learner, as its plain network) against `opponent` (a
  member, unwrapped the same way) for `generation`, on `deals` duplicate deal
  pairs keyed by `seed`. The League plays it and puts the GateCell on the
  next view.

  Attributes:
    subject: the learner name.
    opponent: the member name.
    generation: the round index.
    seed: the deal seed.
    deals: the number of duplicate deal pairs.
  """

  subject: str
  opponent: str
  generation: int
  seed: int
  deals: int


@dataclasses.dataclass(frozen=True)
class GateCell:
  """A played gate cell.

  Subject's mean payoff per deal pair against the opponent, with its SE and
  95% interval (eval.stats), on n_deals duplicate deal pairs keyed by seed.

  Attributes:
    generation: the round index.
    subject: the learner name.
    opponent: the member name.
    seed: the deal seed.
    n_deals: the number of duplicate deal pairs.
    mean: the mean payoff per pair.
    se: its standard error.
    lo: the 95% interval's lower end.
    hi: the 95% interval's upper end.
  """

  generation: int
  subject: str
  opponent: str
  seed: int
  n_deals: int
  mean: float
  se: float
  lo: float
  hi: float


@dataclasses.dataclass(frozen=True)
class LeagueView:
  """What a controller reads: members, learners and the payoff matrix.

  payoff is None when the league computes no payoffs (LeagueConfig
  payoff="none", for controllers that read none, such as FixedMix or
  ScheduledPromotion); reading a cell of such a view raises. gate is the
  gate cell the controller requested (GateRequest), else None.

  Attributes:
    iteration: the league iteration.
    members: MemberInfo per member, in admission order.
    learners: LearnerInfo per live learner, in spawn order.
    payoff: the payoff matrix, or None (see above).
    gate: the requested gate cell, or None.
  """

  iteration: int
  members: tuple[MemberInfo, ...]  # MemberInfo, admission order
  learners: tuple[LearnerInfo, ...]  # LearnerInfo, spawn order
  payoff: Optional[payoff.PayoffMatrix]
  gate: Optional[GateCell] = None

  def __post_init__(self):
    if self.payoff is None:
      object.__setattr__(self, "_index", None)
      return
    names = set(self.payoff.names)
    need = [m.name for m in self.members if m.lifecycle != "evicted"]
    need += [l.name for l in self.learners]
    missing = [n for n in need if n not in names]
    if missing:
      raise ValueError(f"the payoff matrix misses {missing}")
    object.__setattr__(
        self, "_index", {n: i for i, n in enumerate(self.payoff.names)}
    )

  def _need_payoff(self) -> None:
    if self.payoff is None:
      raise ValueError(
          "this league computes no payoffs (LeagueConfig payoff='none'); a "
          "controller that reads payoffs needs payoff='evaluator' or 'exact'"
      )

  def member(self, name: str) -> MemberInfo:
    """The member named `name`.

    Args:
      name: the member name.

    Returns:
      The MemberInfo.

    Raises:
      KeyError: no such member.
    """
    for m in self.members:
      if m.name == name:
        return m
    raise KeyError(f"no member {name!r}")

  def has_member(self, name: str) -> bool:
    """Whether a member is named `name`.

    Args:
      name: the member name.

    Returns:
      True when there is one.
    """
    return any(m.name == name for m in self.members)

  def learner(self, name: str) -> LearnerInfo:
    """The learner named `name`.

    Args:
      name: the learner name.

    Returns:
      The LearnerInfo.

    Raises:
      KeyError: no such learner.
    """
    for l in self.learners:
      if l.name == name:
        return l
    raise KeyError(f"no learner {name!r}")

  def has_learner(self, name: str) -> bool:
    """Whether a learner is named `name`.

    Args:
      name: the learner name.

    Returns:
      True when there is one.
    """
    return any(l.name == name for l in self.learners)

  def cell(self, a: str, b: str) -> tuple[float, float]:
    """The payoff of entrant a against b.

    Args:
      a: an entrant name.
      b: another entrant name.

    Returns:
      (mean, se).
    """
    self._need_payoff()
    i, j = self._index[a], self._index[b]
    return float(self.payoff.mean[i, j]), float(self.payoff.se[i, j])

  def submatrix(self, names: Sequence[str]) -> np.ndarray:
    """The mean payoffs among `names`, in that order.

    Args:
      names: the entrant names.

    Returns:
      A copy of the [n, n] submatrix.
    """
    self._need_payoff()
    idx = [self._index[n] for n in names]
    return self.payoff.mean[np.ix_(idx, idx)].copy()


# -- actions -------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SpawnLearner:
  """Spawns a learner.

  Attributes:
    name: the learner name.
    role: its role.
    init_from: a member id or a learner name; None = fresh init.
  """

  name: str
  role: str
  # a member id or a learner name; None = fresh init
  init_from: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class RetireLearner:
  """Retires a learner.

  Attributes:
    name: the learner name.
  """

  name: str


@dataclasses.dataclass(frozen=True)
class Admit:
  """Admits a learner's current params as a member.

  Attributes:
    learner: the learner name.
    member: the member's artifact id <base>@<step>.
    role: the member's role.
    lifecycle: its lifecycle (not "evicted").
  """

  learner: str
  member: str
  role: str
  lifecycle: str = "fixed"

  def __post_init__(self):
    if not _ID_RE.match(self.member):
      raise ValueError(
          "Admit: member must be an artifact id '<base>@<step>', got "
          f"{self.member!r}"
      )
    if self.lifecycle not in LIFECYCLES or self.lifecycle == "evicted":
      raise ValueError(f"Admit: bad lifecycle {self.lifecycle!r}")


@dataclasses.dataclass(frozen=True)
class SetLifecycle:
  """Moves a member to another lifecycle.

  Attributes:
    member: the member name.
    lifecycle: one of LIFECYCLES.
    reason: why, for the logs.
  """

  member: str
  lifecycle: str
  reason: str = ""

  def __post_init__(self):
    if self.lifecycle not in LIFECYCLES:
      raise ValueError(f"SetLifecycle: {self.lifecycle!r} not in {LIFECYCLES}")


@dataclasses.dataclass(frozen=True)
class SetMix:
  """Sets a learner's opponent mix.

  Attributes:
    learner: the learner name.
    mix: the Mix.
  """

  learner: str
  mix: sampler.Mix

  def __post_init__(self):
    assert isinstance(self.mix, sampler.Mix), type(self.mix)


@dataclasses.dataclass(frozen=True)
class CopyMutate:
  """PBT: dst's params become shrink * src's + perturb * fresh ones.

  Attributes:
    src: the source learner.
    dst: the destination learner.
    shrink: the weight of src's params.
    perturb: the weight of freshly initialised params.
  """

  src: str
  dst: str
  shrink: float = 1.0
  perturb: float = 0.0

  def __post_init__(self):
    for v in (self.shrink, self.perturb):
      if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(
            "CopyMutate: shrink/perturb must be finite and >= 0, got "
            f"{self.shrink}, {self.perturb}"
        )


@dataclasses.dataclass(frozen=True)
class Stop:
  """Ends the run.

  Attributes:
    reason: why, for the logs and the commit.
  """

  reason: str


STRUCTURAL = (SpawnLearner, Admit)  # actions that add entrants (re-evaluate)


def _plateau(scores: Sequence[float], patience: int, threshold: float) -> bool:
  """Whether the recent scores have plateaued.

  Whether the best of the last `patience` scores improves on the score
  before them by less than `threshold`.

  Args:
    scores: the scores, oldest first.
    patience: how many recent scores to look at.
    threshold: the least improvement that is not a plateau.

  Returns:
    Whether the scores have plateaued.
  """
  if len(scores) < patience + 1:
    return False
  recent = scores[-patience:]
  return max(recent) - scores[-(patience + 1)] < threshold


def _lcb(view: LeagueView, a: str, b: str, z: float) -> float:
  """The lower confidence bound of a against b.

  Args:
    view: The league view.
    a: The row entrant.
    b: The column entrant.
    z: The z multiplier.

  Returns:
    mean - z * se.
  """
  mean, se = view.cell(a, b)
  return mean - z * se


# -- PSRO / pipeline-PSRO ------------------------------------------------------
#: Meta-solver keyword arguments PSRO uses when solver_kwargs is not given.
#: alpha-rank with population size m = 2 makes each fixation probability a
#: function of one pairwise payoff difference, so the ranking is defined for
#: any matrix the league evaluates; alpha = 5 is the selection intensity.
DEFAULT_SOLVER_KWARGS = {"alpha_rank": {"population_size": 2, "alpha": 5.0}}


class PSRO:
  """Policy-Space Response Oracles as a controller (Lanctot et al. 2017).

  Each iteration trains a best response (BR) against the meta-strategy the
  meta-solver computes over the population's payoff matrix, admits it as a
  member, re-evaluates and re-solves. With pipeline_depth > 1 this is
  pipeline PSRO (McAleer et al. 2020): pipeline_depth concurrent BR slots at
  staggered levels; level j trains against (1 - active_weight_share) of the
  meta-strategy plus active_weight_share uniform over the live lower levels,
  and only the lowest is promoted.

  A ready BR is promoted once it is the lowest level, at most one promotion
  per iteration.

  Attributes:
    meta_solver: the `meta_solvers.SOLVERS` name.
    solver_kwargs: the solver's keyword arguments.
    iterations: BRs to admit before stopping.
    br_updates: a BR's update budget, counted from its own spawn.
    min_br_updates: None, or the updates after which a plateaued BR is
      promoted early.
    plateau_patience: None, or the scores a plateau spans.
    plateau_threshold: the score change below which a BR has plateaued.
    depth: concurrent BR levels (pipeline_depth).
    share: the lower levels' share of a pipeline BR's mix
      (active_weight_share).
    pipeline_warm_start: a new top-level BR starts from the highest live BR.
    warm_start: a new BR starts from the member just admitted.
    time_averaged: train against the running average of the meta-strategies.
    population_roles: member roles the meta-strategy covers.
    last_meta: None, or the last (names, float64 strategy) meta-strategy.
  """

  BR_ROLE = "psro-br"
  MEMBER_ROLE = "psro"

  def __init__(
      self,
      meta_solver: str = "nash",
      iterations: int = 10,
      br_updates: int = 100,
      solver_kwargs: Optional[Mapping[str, Any]] = None,
      min_br_updates: Optional[int] = None,
      plateau_patience: Optional[int] = None,
      plateau_threshold: float = 0.01,
      pipeline_depth: int = 1,
      active_weight_share: float = 0.5,
      pipeline_warm_start: bool = False,
      warm_start: bool = False,
      time_averaged: bool = False,
      population_roles: Sequence[str] = ("init", "psro"),
  ):
    """Initializes the controller.

    Args:
      meta_solver: a `meta_solvers.SOLVERS` name.
      iterations: BRs to admit before stopping.
      br_updates: a BR's update budget, counted from its own spawn.
      solver_kwargs: the solver's keyword arguments (default
        DEFAULT_SOLVER_KWARGS[meta_solver]).
      min_br_updates: with plateau_patience, promote earlier once a BR has
        min_br_updates updates and its score has plateaued.
      plateau_patience: the scores a plateau spans (with min_br_updates).
      plateau_threshold: the score change below which a BR has plateaued.
      pipeline_depth: concurrent BR levels.
      active_weight_share: the lower levels' share of a pipeline BR's mix.
      pipeline_warm_start: a new top-level BR starts from the highest live
        BR.
      warm_start: a new BR starts from the member just admitted.
      time_averaged: train against the running average of the
        meta-strategies.
      population_roles: member roles the meta-strategy covers.

    Raises:
      KeyError: meta_solver is not a known meta-solver.
      ValueError: only one of min_br_updates and plateau_patience is set.
    """
    if meta_solver not in meta_solvers.SOLVERS:
      raise KeyError(f"unknown meta-solver {meta_solver!r}")
    assert iterations >= 1 and br_updates >= 1 and pipeline_depth >= 1
    if (min_br_updates is None) != (plateau_patience is None):
      raise ValueError(
          "min_br_updates and plateau_patience go together (plateau promotion)"
      )
    self.meta_solver = meta_solver
    if solver_kwargs is None:
      solver_kwargs = dict(DEFAULT_SOLVER_KWARGS.get(meta_solver, {}))
    self.solver_kwargs = dict(solver_kwargs)
    self.iterations, self.br_updates = int(iterations), int(br_updates)
    self.min_br_updates, self.plateau_patience = (
        min_br_updates,
        plateau_patience,
    )
    self.plateau_threshold = float(plateau_threshold)
    self.depth = int(pipeline_depth)
    self.share = float(active_weight_share)
    self.pipeline_warm_start, self.warm_start = pipeline_warm_start, warm_start
    self.time_averaged = time_averaged
    self.population_roles = tuple(population_roles)
    self._levels: dict[str, int] = {}
    self._next_br = 0
    self._promotions = 0
    # view.iteration of the last promotion
    self._promoted_at: Optional[int] = None
    self._started = False
    self._avg: Optional[np.ndarray] = None
    self._avg_count = 0
    self._avg_pop: tuple[Any, ...] = ()
    self.last_meta: Optional[tuple[Any, ...]] = None  # (names, f64 strategy)

  def levels(self) -> dict[str, int]:
    """The pipeline level of each live BR.

    Returns:
      {learner name: level}.
    """
    return dict(self._levels)

  def _spawn(self, level: int, init_from: Optional[str]) -> SpawnLearner:
    """Spawns the next best-response learner.

    Args:
      level: The best response's level.
      init_from: The member to start from, or None.

    Returns:
      The SpawnLearner.
    """
    name = f"br{self._next_br}"
    self._next_br += 1
    self._levels[name] = level
    return SpawnLearner(name, self.BR_ROLE, init_from=init_from)

  def _ready(self, info: LearnerInfo) -> bool:
    """Whether the best response is done training.

    Args:
      info: The learner's info.

    Returns:
      True after br_updates, or after min_br_updates on a plateau.
    """
    if info.updates >= self.br_updates:
      return True
    return (
        self.min_br_updates is not None
        and info.updates >= self.min_br_updates
        and _plateau(info.scores, self.plateau_patience, self.plateau_threshold)
    )

  def _meta(self, view: LeagueView, pop: list[str]) -> np.ndarray:
    """The opponent mixture over `pop` from the meta-solver.

    Args:
      view: the league view.
      pop: the population.

    Returns:
      The column player's mixture, averaged over populations when
      `time_averaged`.
    """
    sol = meta_solvers.solve(
        self.meta_solver, view.submatrix(pop), **self.solver_kwargs
    )
    sigma = sol.col  # a learner's opponents play the column player's mixture
    if self.time_averaged:
      if tuple(pop) != self._avg_pop:
        if self._avg is None:
          self._avg, self._avg_count = sigma.copy(), 1
        else:
          assert pop[: len(self._avg_pop)] == list(
              self._avg_pop
          ), "population reordered"
          prev = np.pad(self._avg, (0, len(pop) - len(self._avg)))
          self._avg = (self._avg_count * prev + sigma) / (self._avg_count + 1)
          self._avg /= self._avg.sum()
          self._avg_count += 1
        self._avg_pop = tuple(pop)
      sigma = self._avg
    self.last_meta = (tuple(pop), sigma.copy())
    return sigma

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.

    Raises:
      ValueError: the view has no population member.
    """
    pop = [
        m.name
        for m in view.members
        if m.role in self.population_roles and m.lifecycle != "evicted"
    ]
    if not pop:
      raise ValueError(
          f"PSRO needs a population member with role in {self.population_roles}"
      )
    acts: list[Any] = []
    if not self._started:
      self._started = True
      for level in range(self.depth):
        acts.append(self._spawn(level, None))
    elif (
        self._promotions < self.iterations
        and self._promoted_at != view.iteration
    ):
      # At most one promotion per iteration: League.tick calls step() again in
      # the same iteration after the Admit, and the next-lowest worker first
      # trains as the lowest.
      workers = sorted(self._levels, key=self._levels.get)
      lowest = workers[0]
      if view.has_learner(lowest) and self._ready(view.learner(lowest)):
        info = view.learner(lowest)
        self._promotions += 1
        self._promoted_at = int(view.iteration)
        member = f"psro{self._promotions}@{info.updates}"
        acts += [
            Admit(lowest, member, self.MEMBER_ROLE, "fixed"),
            RetireLearner(lowest),
        ]
        del self._levels[lowest]
        for w in self._levels:
          self._levels[w] -= 1
        logging.info(
            "PSRO: promotion %d/%d: %s -> %s",
            self._promotions,
            self.iterations,
            lowest,
            member,
        )
        if self._promotions < self.iterations:
          shifted = sorted(self._levels, key=self._levels.get)
          init = None
          if self.pipeline_warm_start and shifted:
            init = shifted[-1]
          elif self.warm_start:
            init = member
          acts.append(self._spawn(self.depth - 1, init))
    sigma = self._meta(view, pop)
    by_level = sorted(self._levels, key=self._levels.get)
    for w in by_level:
      lower = [x for x in by_level if self._levels[x] < self._levels[w]]
      acts.append(SetMix(w, sampler.pipeline(pop, sigma, lower, self.share)))
    if self._promotions >= self.iterations:
      acts.append(Stop(f"psro: {self.iterations} iterations"))
    return acts


# -- GatedPool -----------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GateConfig:
  """GatedPool's membership gate and mixing weights.

  A gated member becomes active when the lower bound of its payoff interval
  against the main learner exceeds tau_on, and inactive when the upper bound
  falls below tau_off (hysteresis: tau_off < tau_on). The active members'
  share of the main's rows is lambda = clip(lambda_min + lambda_slope *
  excess, lambda_min, lambda_max), where excess is the best active member's
  mean payoff above tau_on; within it a member's weight is weight_floor +
  max(0, mean - tau_off) ** weight_alpha, and an active member whose weight
  is <= 0 is demoted to inactive. epsilon is the audit share spread over
  inactive members and anchors; z the interval's width in SEs.
  active_set_max caps the active set (strongest by mean, then lower bound).

  The defaults keep the main learner on self-play for most of its rows: at
  the cap the gated members take lambda_max = 0.28 and the audit epsilon =
  0.05, so self-play keeps 1 - 0.28 - 0.05 = 0.67 (two thirds) of the rows.
  They are a starting point, not a tuned optimum for any game.

  Attributes:
    tau_on: the lower bound above which a member becomes active.
    tau_off: the upper bound below which a member becomes inactive.
    lambda_min: the active members' least share of the main's rows.
    lambda_max: the active members' greatest share of the main's rows.
    lambda_slope: lambda's growth per unit of excess.
    epsilon: the audit share.
    z: the payoff interval's width in standard errors.
    weight_alpha: the exponent of an active member's weight.
    weight_floor: the floor of an active member's weight.
    active_set_max: None, or the active set's cap.
  """

  tau_on: float = 0.5
  tau_off: float = 0.2
  lambda_min: float = 0.05
  lambda_max: float = 0.28
  lambda_slope: float = 0.1
  epsilon: float = 0.05
  z: float = stats.Z95
  weight_alpha: float = 1.0
  weight_floor: float = 0.0
  active_set_max: Optional[int] = None

  def __post_init__(self):
    if not self.tau_off < self.tau_on:
      raise ValueError(
          f"tau_off {self.tau_off} must be < tau_on {self.tau_on} (hysteresis)"
      )
    if not 0.0 <= self.lambda_min <= self.lambda_max:
      raise ValueError("need 0 <= lambda_min <= lambda_max")
    if not 0.0 <= self.epsilon < 1.0 or self.lambda_max + self.epsilon >= 1.0:
      raise ValueError("need 0 <= epsilon and lambda_max + epsilon < 1")
    if (
        self.z < 0
        or self.lambda_slope < 0
        or self.weight_alpha < 0
        or self.weight_floor < 0
    ):
      raise ValueError(
          "z, lambda_slope, weight_alpha, weight_floor must be >= 0"
      )
    if self.active_set_max is not None and self.active_set_max < 1:
      raise ValueError("active_set_max must be None or >= 1")


@dataclasses.dataclass(frozen=True)
class RetirementConfig:
  """GatedPool's retirement settings.

  GatedPool's retirement: an inactive member whose payoff upper bound is
  below -retire_threshold on retire_after of its last retire_window scored
  iterations is archived; archived members keep archive_epsilon of the mix.
  pool_cap bounds the sampled non-anchor members (archived evicted first,
  oldest first, then inactive by lowest score).

  Attributes:
    retire_threshold: the payoff upper bound below -retire_threshold counts
      toward retirement.
    retire_after: the low scores that archive a member.
    retire_window: the scored iterations the low scores are counted over.
    archive_epsilon: the archived members' share of the mix.
    pool_cap: None, or the cap on sampled non-anchor members.
  """

  retire_threshold: float = 0.5
  retire_after: int = 3
  retire_window: int = 4
  archive_epsilon: float = 0.0
  pool_cap: Optional[int] = None

  def __post_init__(self):
    if not self.retire_threshold >= 0.0:
      raise ValueError(
          f"retire_threshold must be >= 0, got {self.retire_threshold}"
      )
    if not 1 <= self.retire_after <= self.retire_window:
      raise ValueError(
          "need 1 <= retire_after <= retire_window, got retire_after "
          f"{self.retire_after}, retire_window {self.retire_window}"
      )
    if not 0.0 <= self.archive_epsilon < 1.0:
      raise ValueError(
          f"archive_epsilon must be in [0, 1), got {self.archive_epsilon}"
      )
    if not (self.pool_cap is None or self.pool_cap >= 1):
      raise ValueError(f"pool_cap must be None or >= 1, got {self.pool_cap}")


def gate_lifecycle(prev: str, lcb: float, ucb: float, cfg: GateConfig) -> str:
  """The gated lifecycle after one scoring.

  Active above tau_on, inactive below tau_off, otherwise unchanged; archived
  members stay archived until they clear tau_on.

  Args:
    prev: the gated lifecycle before the scoring.
    lcb: the payoff's lower confidence bound.
    ucb: the payoff's upper confidence bound.
    cfg: the GateConfig.

  Returns:
    The lifecycle.

  Raises:
    ValueError: prev is not a gated lifecycle.
  """
  if prev not in GATED:
    raise ValueError(f"gate_lifecycle: {prev!r} is not a gated lifecycle")
  if lcb > cfg.tau_on:
    return "active"
  if prev == "archived":
    return "archived"
  if ucb < cfg.tau_off:
    return "inactive"
  return prev


class GatedPool:
  """One main learner against a gated pool of members (GateConfig).

  Every iteration each gated member is scored by its payoff against the main
  learner; the gate moves it between candidate, active, inactive and
  archived (RetirementConfig) and the pool cap evicts the excess. The main's
  mix is `sampler.pmix`: self-play, lambda over the active members by
  weight, epsilon over the audit set (inactive members and anchors), and
  archive_epsilon over the archived. Anchors are never gated. `events`
  records retirements and evictions.

  Attributes:
    main: the main learner's name.
    gate: the GateConfig.
    retirement: the RetirementConfig.
    anchors: the members never gated.
    events: retirements and evictions, in order.
  """

  def __init__(
      self,
      main: str = "main",
      gate: GateConfig = GateConfig(),
      retirement: Optional[RetirementConfig] = None,
      anchors: Sequence[str] = (),
  ):
    self.main, self.gate, self.retirement = main, gate, retirement
    self.anchors = tuple(anchors)
    if (
        retirement is not None
        and retirement.pool_cap is not None
        and (
            gate.active_set_max is not None
            and retirement.pool_cap < gate.active_set_max
        )
    ):
      raise ValueError("pool_cap must be >= active_set_max")
    self._beaten: dict[str, list[str]] = {}
    self._archived_at: dict[str, int] = {}
    self._last_it: Optional[int] = None
    self.events: list[Any] = []

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.
    """
    if not view.has_learner(self.main):
      return [
          SpawnLearner(self.main, "main"),
          SetMix(self.main, sampler.Mix(1.0, {})),
      ]
    cfg = self.gate
    fresh_tick = view.iteration != self._last_it
    self._last_it = view.iteration
    gated = [
        m
        for m in view.members
        if m.lifecycle in GATED and m.name not in self.anchors
    ]
    score = {}
    for m in gated:
      r, se = view.cell(m.name, self.main)
      score[m.name] = (r, r - cfg.z * se, r + cfg.z * se)
    new = {
        m.name: gate_lifecycle(
            m.lifecycle, score[m.name][1], score[m.name][2], cfg
        )
        for m in gated
    }
    # Active-set cap: keep the strongest N (by mean, then lower bound, then
    # name).
    active = [n for n in new if new[n] == "active"]
    if cfg.active_set_max is not None and len(active) > cfg.active_set_max:
      ranked = sorted(active, key=lambda n: (-score[n][0], -score[n][1], n))
      for n in ranked[cfg.active_set_max :]:
        new[n] = "inactive"
      active = ranked[: cfg.active_set_max]
    raw = {
        n: (
            cfg.weight_floor
            + max(0.0, score[n][0] - cfg.tau_off) ** cfg.weight_alpha
        )
        for n in active
    }
    for n in [n for n in active if raw[n] <= 0.0]:
      new[n] = "inactive"  # a member with no weight is not active
      del raw[n]
    active = list(raw)
    ret = self.retirement
    if ret is not None:
      for m in gated:
        if fresh_tick:
          h = self._beaten.setdefault(m.name, [])
          h.append(score[m.name][2] < -ret.retire_threshold)
          del h[: -ret.retire_window]
        h = self._beaten.get(m.name, [])
        if (
            new[m.name] == "inactive"
            and m.lifecycle == "inactive"
            and sum(h) >= ret.retire_after
        ):
          new[m.name] = "archived"
          self._archived_at[m.name] = view.iteration
          self.events.append(
              dict(event="retired", member=m.name, iteration=view.iteration)
          )
      if ret.pool_cap is not None:
        sampled = [
            n for n in new if new[n] in ("active", "inactive", "archived")
        ]
        excess = len(sampled) - ret.pool_cap
        oldest_archived = sorted(
            (n for n in sampled if new[n] == "archived"),
            key=lambda n: (self._archived_at.get(n, -1), score[n][0], n),
        )
        inact = sorted(
            (n for n in sampled if new[n] == "inactive"),
            key=lambda n: (score[n][0], n),
        )
        for n in (oldest_archived + inact)[: max(0, excess)]:
          new[n] = "evicted"
          self.events.append(
              dict(event="evicted", member=n, iteration=view.iteration)
          )
    acts: list[Any] = [
        SetLifecycle(m.name, new[m.name], "pool gate")
        for m in gated
        if new[m.name] != m.lifecycle
    ]
    if active:
      excess = max(max(0.0, score[n][0] - cfg.tau_on) for n in active)
      lam = min(
          max(cfg.lambda_min + cfg.lambda_slope * excess, cfg.lambda_min),
          cfg.lambda_max,
      )
      tot = sum(raw.values())
      weights = {n: raw[n] / tot for n in active}
    else:
      lam, weights = 0.0, {}
    audit = [n for n in new if new[n] == "inactive"]
    audit += [
        a
        for a in self.anchors
        if view.has_member(a) and view.member(a).lifecycle != "evicted"
    ]
    archived = [n for n in new if new[n] == "archived"]
    arch_eps = ret.archive_epsilon if ret is not None else 0.0
    mix = sampler.pmix(
        weights,
        lam,
        cfg.epsilon,
        audit,
        archived=archived,
        archive_epsilon=arch_eps,
    )
    return acts + [SetMix(self.main, mix)]


# -- continual exploiter, best-response oracle ---------------------------------
class ContinualExploiter:
  """An always-on best-response learner against snapshots of `target`.

  Every refresh_every iterations a new snapshot of the target is admitted and
  the previous one evicted; the exploiter keeps training from where it was
  (it is not reset). Before a refresh replaces the snapshot, the exploiter is
  admitted as a member when the lower bound of its payoff against that
  snapshot exceeds admit_lcb.

  Attributes:
    target: the learner whose snapshots are exploited.
    name: the exploiter learner's name.
    refresh_every: iterations between snapshot refreshes.
    admit_lcb: the payoff lower bound above which the exploiter is admitted.
    z: the payoff interval's width in standard errors.
    admit_lifecycle: the lifecycle an admitted exploiter starts in.
  """

  def __init__(
      self,
      target: str = "main",
      name: str = "exploiter",
      refresh_every: int = 10,
      admit_lcb: float = 0.0,
      z: float = stats.Z95,
      admit_lifecycle: str = "candidate",
  ):
    assert refresh_every >= 1
    self.target, self.name, self.refresh_every = target, name, refresh_every
    self.admit_lcb, self.z, self.admit_lifecycle = admit_lcb, z, admit_lifecycle
    self._snapshot: Optional[str] = None
    self._admitted_for: set[str] = set()
    self._count = 0
    self._last_refresh: Optional[int] = None

  def _admission(self, view: LeagueView) -> list[Any]:
    """Admits the exploiter if its lower bound beats its target snapshot.

    Args:
      view: The league view.

    Returns:
      [Admit], or [] when not admitted.
    """
    if (
        self._snapshot is not None
        and view.has_learner(self.name)
        and view.has_member(self._snapshot)
        and self._snapshot not in self._admitted_for
        and _lcb(view, self.name, self._snapshot, self.z) > self.admit_lcb
    ):
      self._count += 1
      self._admitted_for.add(self._snapshot)
      upd = view.learner(self.name).updates
      return [
          Admit(
              self.name,
              f"{self.name}-x{self._count}@{upd}",
              "exploiter",
              self.admit_lifecycle,
          )
      ]
    return []

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.
    """
    if not view.has_learner(self.target):
      return []
    acts: list[Any] = []
    if not view.has_learner(self.name):
      acts.append(SpawnLearner(self.name, "exploiter"))
    if (
        view.iteration % self.refresh_every == 0
        and self._last_refresh != view.iteration
    ):
      # The exploiter is judged against the snapshot it trained on before the
      # refresh evicts that snapshot.
      acts += self._admission(view)
      self._last_refresh = view.iteration
      snap = f"{self.target}-target@{view.learner(self.target).updates}"
      if snap != self._snapshot:
        if not view.has_member(snap):
          acts.append(Admit(self.target, snap, "target", "fixed"))
        if self._snapshot is not None:
          acts.append(
              SetLifecycle(self._snapshot, "evicted", "target refreshed")
          )
        self._snapshot = snap
    elif self._last_refresh != view.iteration:
      # Not in a later round of a refresh iteration, whose new snapshot has
      # not been trained against yet.
      acts += self._admission(view)
    if self._snapshot is not None:
      acts.append(SetMix(self.name, sampler.Mix(0.0, {self._snapshot: 1.0})))
    return acts


class BROracle:
  """Periodic best responses against the main learner.

  Every `every` iterations a snapshot of the target is admitted and num_brs
  fresh BR learners start against it (at most max_concurrent running). A BR
  finishes at max_updates, or after min_updates once its score plateaus; it
  is admitted as a candidate when the lower bound of its payoff against the
  snapshot exceeds admission_lcb, and retired either way. A snapshot no
  running BR targets is evicted. `events` records admissions and refusals.

  Attributes:
    target: the learner the BRs play against.
    every: iterations between launches.
    num_brs: BR learners started per launch.
    max_updates: a BR's update budget.
    min_updates: None, or the updates after which a plateaued BR finishes.
    patience: None, or the scores a plateau spans.
    threshold: the score change below which a BR has plateaued.
    bar: the payoff lower bound a BR needs to be admitted (admission_lcb).
    z: the payoff interval's width in standard errors.
    max_concurrent: the most BRs running at once.
    events: admissions and refusals, in order.
  """

  def __init__(
      self,
      target: str = "main",
      every: int = 10,
      num_brs: int = 1,
      max_updates: int = 1000,
      min_updates: Optional[int] = None,
      plateau_patience: Optional[int] = None,
      plateau_threshold: float = 0.01,
      admission_lcb: float = 0.0,
      z: float = stats.Z95,
      max_concurrent: int = 4,
  ):
    assert (
        every >= 1
        and num_brs >= 1
        and max_updates >= 1
        and max_concurrent >= num_brs
    )
    if (min_updates is None) != (plateau_patience is None):
      raise ValueError("min_updates and plateau_patience go together")
    self.target, self.every, self.num_brs = target, every, num_brs
    self.max_updates, self.min_updates = max_updates, min_updates
    self.patience, self.threshold = plateau_patience, plateau_threshold
    self.bar, self.z, self.max_concurrent = admission_lcb, z, max_concurrent
    self._jobs: dict[str, str] = {}  # running BR -> its target snapshot
    # target snapshots this controller admitted, still pooled
    self._snaps: set[str] = set()
    self._next = 0
    self._last_launch: Optional[int] = None
    self.events: list[Any] = []

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.
    """
    if not view.has_learner(self.target):
      return []
    acts: list[Any] = []
    for br, snap in list(self._jobs.items()):
      if not view.has_learner(br):
        continue
      info = view.learner(br)
      done = info.updates >= self.max_updates or (
          self.min_updates is not None
          and info.updates >= self.min_updates
          and _plateau(info.scores, self.patience, self.threshold)
      )
      if not done:
        continue
      lcb = _lcb(view, br, snap, self.z)
      if lcb > self.bar:
        acts.append(Admit(br, f"{br}@{info.updates}", "oracle", "candidate"))
        self.events.append(
            dict(
                event="admitted", learner=br, lcb=lcb, iteration=view.iteration
            )
        )
      else:
        self.events.append(
            dict(event="refused", learner=br, lcb=lcb, iteration=view.iteration)
        )
      acts.append(RetireLearner(br))
      del self._jobs[br]
    if (
        view.iteration % self.every == 0
        and self._last_launch != view.iteration
        and len(self._jobs) + self.num_brs <= self.max_concurrent
    ):
      self._last_launch = view.iteration
      snap = f"{self.target}-oracle@{view.learner(self.target).updates}"
      if not view.has_member(snap):
        acts.append(Admit(self.target, snap, "target", "fixed"))
        self._snaps.add(snap)
      for _ in range(self.num_brs):
        br = f"oracle{self._next}"
        self._next += 1
        self._jobs[br] = snap
        acts.append(SpawnLearner(br, "oracle"))
    # A snapshot no running BR targets would cost a payoff row and column every
    # iteration: it leaves the pool.
    for snap in sorted(self._snaps - set(self._jobs.values())):
      acts.append(SetLifecycle(snap, "evicted"))
      self._snaps.discard(snap)
    for br, snap in self._jobs.items():
      acts.append(SetMix(br, sampler.Mix(0.0, {snap: 1.0})))
    return acts


# -- PBT, lagged audit ---------------------------------------------------------
class PBT:
  """Truncation population-based training (Jaderberg et al. 2017).

  Every `interval` iterations the learners are ranked by their mean payoff
  against the others; each of the bottom `fraction` copies a top learner
  (exploit; with bank_exploit the source is first admitted as a member) and
  perturbs it, dst <- shrink * src + perturb * fresh init (explore). The
  optimizer state is not touched. Each learner's mix is uniform over the
  others, with a self_play share.

  Attributes:
    learners: the learners' names.
    interval: iterations between exploit/explore steps.
    fraction: the bottom fraction that copies a top learner.
    shrink: the weight of the copied params.
    perturb: the weight of the fresh init.
    self_play: each learner's self-play share.
    bank_exploit: admit an exploit's source as a member first.
  """

  def __init__(
      self,
      learners: Sequence[str],
      interval: int = 10,
      fraction: float = 0.25,
      shrink: float = 1.0,
      perturb: float = 0.0,
      self_play: float = 0.0,
      bank_exploit: bool = True,
  ):
    assert len(learners) >= 2 and interval >= 1 and 0.0 < fraction <= 0.5
    self.learners, self.interval, self.fraction = (
        tuple(learners),
        interval,
        fraction,
    )
    self.shrink, self.perturb, self.self_play = shrink, perturb, self_play
    self.bank_exploit = bank_exploit
    self._last: Optional[int] = None

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.
    """
    acts: list[Any] = [
        SpawnLearner(n, "pbt") for n in self.learners if not view.has_learner(n)
    ]
    live = [n for n in self.learners if view.has_learner(n)]
    if (
        len(live) == len(self.learners)
        and view.iteration > 0
        and view.iteration % self.interval == 0
        and self._last != view.iteration
    ):
      self._last = view.iteration
      sub = view.submatrix(live)
      fitness = {
          n: float(np.delete(sub[i], i).mean()) for i, n in enumerate(live)
      }
      ranked = sorted(live, key=lambda n: (-fitness[n], n))
      k = max(1, int(math.floor(self.fraction * len(live))))
      top, bottom = ranked[:k], ranked[-k:]
      for i, dst in enumerate(bottom):
        src = top[i % len(top)]
        if self.bank_exploit:
          acts.append(
              Admit(
                  src, f"{src}-pbt@{view.learner(src).updates}", "pbt", "fixed"
              )
          )
        acts.append(CopyMutate(src, dst, self.shrink, self.perturb))
    for n in self.learners:
      acts.append(
          SetMix(
              n,
              sampler.uniform(
                  [x for x in self.learners if x != n], self.self_play
              ),
          )
      )
    return acts


class LaggedAudit:
  """Self-play, a lagged snapshot of the main learner, and an audit pool.

  Self-play, a lagged snapshot of the main learner refreshed every
  refresh_every iterations, and a uniform audit pool (`sampler.lagged_audit`).

  Attributes:
    main: the main learner's name.
    audit: the audit pool's members.
    refresh_every: iterations between lagged-snapshot refreshes.
    self_play: the self-play share.
    lagged_mass: the lagged snapshot's share.
  """

  def __init__(
      self,
      main: str = "main",
      audit: Sequence[str] = (),
      refresh_every: int = 10,
      self_play: float = 0.5,
      lagged_mass: float = 0.25,
  ):
    assert refresh_every >= 1
    self.main, self.audit, self.refresh_every = (
        main,
        tuple(audit),
        refresh_every,
    )
    self.self_play, self.lagged_mass = self_play, lagged_mass
    self._lagged: Optional[str] = None
    self._last: Optional[int] = None

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.
    """
    if not view.has_learner(self.main):
      return [
          SpawnLearner(self.main, "main"),
          SetMix(
              self.main,
              sampler.lagged_audit(
                  None, self.audit, self.self_play, self.lagged_mass
              ),
          ),
      ]
    acts: list[Any] = []
    if (
        view.iteration > 0
        and view.iteration % self.refresh_every == 0
        and self._last != view.iteration
    ):
      self._last = view.iteration
      snap = f"{self.main}-lag@{view.learner(self.main).updates}"
      if snap != self._lagged:
        acts.append(Admit(self.main, snap, "lagged", "fixed"))
        if self._lagged is not None:
          acts.append(
              SetLifecycle(self._lagged, "evicted", "lagged snapshot refreshed")
          )
        self._lagged = snap
    acts.append(
        SetMix(
            self.main,
            sampler.lagged_audit(
                self._lagged, self.audit, self.self_play, self.lagged_mass
            ),
        )
    )
    return acts


# -- scheduled promotion -------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SnapshotSchedule:
  """When ScheduledPromotion promotes, harvests and seats snapshots.

  A round is snapshot_every updates; its lagged snapshot is the learner at
  the round's start, frozen for the round. Snapshots at multiples of
  harvest_every (a multiple of snapshot_every) form the pool, newest
  max_pool kept. Each round seats pool_slots pool members (Arena slots). The
  snapshot takes self_weight of the rows while a pool exists, every row
  before. Every field except pool_slots (default 1) is required.

  The league splits rows per round, not per hand: every Arena slot computes
  the full row count, so seating every pool member per hand would cost one
  network forward per member per ply. Over rounds each pool member's
  expected share is still (1 - self_weight) / len(pool).

  Attributes:
    self_weight: the snapshot's share of the rows while a pool exists.
    snapshot_every: updates per round.
    harvest_every: updates between pool harvests (a multiple of snapshot_every).
    max_pool: the newest harvested snapshots kept.
    pool_slots: pool members seated per round.
  """

  self_weight: float
  snapshot_every: int
  harvest_every: int
  max_pool: int
  pool_slots: int = 1

  def __post_init__(self):
    if not 0.0 < self.self_weight <= 1.0:
      raise ValueError(f"self_weight must be in (0, 1], got {self.self_weight}")
    for name in ("snapshot_every", "harvest_every", "max_pool", "pool_slots"):
      if int(getattr(self, name)) < 1:
        raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
    if int(self.harvest_every) % int(self.snapshot_every):
      raise ValueError(
          f"harvest_every {self.harvest_every} must be a multiple of "
          "snapshot_every "
          f"{self.snapshot_every} (a harvested member is a round-start "
          "snapshot)"
      )

  def round_index(self, update: int) -> int:
    """The round of update `update`.

    Args:
      update: the learner's update.

    Returns:
      update // snapshot_every.
    """
    return int(update) // int(self.snapshot_every)

  def snapshot_step(self, update: int) -> int:
    """The learner step whose params are the snapshot during update `update`.

    Args:
      update: the learner's update.

    Returns:
      The step.
    """
    return self.round_index(update) * int(self.snapshot_every)

  def pool_steps(self, update: int) -> tuple[int, ...]:
    """The harvested steps pooled during update `update`, oldest first.

    Multiples of harvest_every in (0, snapshot_step], the newest max_pool.

    Args:
      update: the learner's update.

    Returns:
      The steps.
    """
    top = self.snapshot_step(update)
    h = int(self.harvest_every)
    steps = tuple(range(h, top + 1, h))
    return steps[-int(self.max_pool) :] if steps else ()

  def active_pool(self, pool: Sequence[int], key) -> tuple[int, ...]:
    """The seated members of `pool`.

    min(pool_slots, len(pool)) members drawn uniformly without replacement,
    sorted.

    Args:
      pool: the pooled steps.
      key: a core.rng key (numpy or jax, the same draw).

    Returns:
      The seated steps.
    """
    pool = tuple(int(p) for p in pool)
    m = min(int(self.pool_slots), len(pool))
    if m == 0:
      return ()
    if m == len(pool):
      return tuple(sorted(pool))
    idx = np.asarray(rng.choice(key, len(pool), (m,), replace=False))
    return tuple(sorted(pool[int(i)] for i in idx))


#: Promotion predicates: pure functions of a GateCell. "ci_above_zero": the
#: whole 95% interval is above 0; "mean_above_zero": the mean is.
PROMOTION_PREDICATES = {
    "ci_above_zero": lambda g: g.lo > 0.0,
    "mean_above_zero": lambda g: g.mean > 0.0,
}


class PromotionGate(Protocol):
  """Decides whether a new snapshot may join ScheduledPromotion's pool.

  At each round start after the first, the controller requests a gate cell:
  the main learner (the candidate) against the previous round's snapshot (the
  incumbent), both as their plain networks, on duplicate deals keyed by
  `seed(generation)`. `passes` decides from the played cell.

  Attributes:
    name: a label for logs and the controller's records.
  """

  name: str

  def seed(self, generation: int) -> int:
    """The deal seed of the gate cell of `generation` (a round index).

    Args:
      generation: the round index.

    Returns:
      The seed.
    """

  def passes(self, candidate: str, incumbent: str, result: GateCell) -> bool:
    """Whether the candidate's snapshot is harvested into the pool.

    Args:
      candidate: the candidate snapshot.
      incumbent: the snapshot it played.
      result: the played gate cell.

    Returns:
      True when the gate passes.
    """


class SeededGate:
  """A PromotionGate with seeds derived from one integer.

  A PromotionGate with seeds derived from one integer and a predicate from
  PROMOTION_PREDICATES.

  Attributes:
    base_seed: the integer the gate's seeds derive from.
    name: the PROMOTION_PREDICATES name.
  """

  def __init__(self, seed: int, predicate: str = "mean_above_zero"):
    if predicate not in PROMOTION_PREDICATES:
      raise ValueError(
          f"promotion {predicate!r} not in {sorted(PROMOTION_PREDICATES)}"
      )
    self.base_seed, self.name = int(seed), predicate

  def seed(self, generation: int) -> int:
    """A crc32 of the base seed and the generation.

    Args:
      generation: the round index.

    Returns:
      The seed.
    """
    return zlib.crc32(
        f"promotion-gate|{self.base_seed}|{int(generation)}".encode()
    )

  def passes(self, candidate: str, incumbent: str, result: GateCell) -> bool:
    """Whether the PROMOTION_PREDICATES predicate passes on `result`.

    Args:
      candidate: the candidate snapshot (unused).
      incumbent: the snapshot it played (unused).
      result: the played gate cell.

    Returns:
      True when the gate passes.
    """
    del candidate, incumbent
    return bool(PROMOTION_PREDICATES[self.name](result))


class ScheduledPromotion:
  """Self-play against a lagged snapshot plus a pool of harvested ones.

  Self-play against a lagged snapshot plus a pool of harvested snapshots,
  for one main learner, on a SnapshotSchedule.

    * At every multiple u of snapshot_every of the main's updates (u = 0
      included: the snapshot of round 0 is the initial params) the main is
      admitted as member "<main>@<u>", the round's lagged snapshot.
    * Admitted steps that are multiples of harvest_every (u > 0) are the pool,
      the newest max_pool of them; a superseded snapshot outside the pool and
      a pool member past max_pool are evicted.
    * Each round seats min(pool_slots, len(pool)) pool members uniformly
      without replacement (key fold_in64(key(seed), round)); the main's mix is
      `sampler.snapshot_pool(snapshot, seated, self_weight)`.

  Every round changes the main's mix (a new snapshot name), so a League with
  redraw="on_change" deals fresh hands at each round start and keeps them for
  the round. Promotions happen at league iterations, so snapshot_every must be
  a multiple of updates_per_iteration. Reads no payoffs. `last_info` holds the
  round's snapshot step, pool size and seated count.

  Gating. With a `gate` (a PromotionGate, or `promotion`: a
  PROMOTION_PREDICATES name for a SeededGate on `seed`) each round start
  u > 0 requests a gate cell of gate_deals deal pairs (`gate_request`) and
  the new snapshot is harvested into the pool only if the gate passes. The
  lagged-snapshot admission never waits on the gate. Each generation gets
  exactly one attempt, recorded in `gate_attempts`. Two-player games only
  (the Evaluator's).

  Attributes:
    gate: None, or the PromotionGate a snapshot must pass.
    gate_deals: duplicate deal pairs per gate cell.
    promotion: None, or the gate's predicate name.
    schedule: the SnapshotSchedule.
    main: the main learner's name.
    seed: the seed of the seating draws.
    role: the role of admitted snapshots.
    gate_attempts: update -> the gate's record for that snapshot.
    last_info: the round's snapshot step, pool size and seated count.
  """

  def __init__(
      self,
      main: str = "main",
      *,
      snapshot_every: int,
      harvest_every: int,
      max_pool: int,
      self_weight: float,
      pool_slots: int = 1,
      seed: int = 0,
      role: str = "main",
      promotion: Optional[str] = None,
      gate: Optional[PromotionGate] = None,
      gate_deals: int = 1024,
  ):
    if promotion is not None and promotion not in PROMOTION_PREDICATES:
      raise ValueError(
          f"promotion {promotion!r} not in {sorted(PROMOTION_PREDICATES)}"
      )
    if int(gate_deals) < 2:
      raise ValueError(f"gate_deals must be >= 2, got {gate_deals}")
    if gate is None and promotion is not None:
      gate = SeededGate(seed, promotion)
    self.gate, self.gate_deals = gate, int(gate_deals)
    self.promotion = None if gate is None else gate.name
    self.schedule = SnapshotSchedule(
        self_weight=self_weight,
        snapshot_every=snapshot_every,
        harvest_every=harvest_every,
        max_pool=max_pool,
        pool_slots=pool_slots,
    )
    self.main, self.seed, self.role = main, int(seed), role
    # Replaced (never mutated) by the methods below.
    self.gate_attempts: Mapping[int, Mapping[str, Any]] = {}
    self._passed: frozenset[int] = frozenset()
    self._round: Optional[int] = None
    self._snapshot: Optional[str] = None
    self._pool: tuple[Any, ...] = ()
    self._seated: tuple[Any, ...] = ()
    self.last_info: dict[str, Any] = {}

  def draw_guards(self) -> tuple[Any, ...]:
    """The draw guards the League runs on every drawn opponent table.

    The gate's own, if it defines draw_guards(); otherwise none.

    Returns:
      The guards.
    """
    return tuple(getattr(self.gate, "draw_guards", lambda: ())())

  def _name(self, step: int) -> str:
    """The member name of the main learner at `step`.

    Args:
      step: The step.

    Returns:
      The name.
    """
    return f"{self.main}@{int(step)}"

  def gate_request(self, view: LeagueView) -> Optional[GateRequest]:
    """The gate cell this iteration needs (class doc), or None.

    Args:
      view: the league view.

    Returns:
      The GateRequest, or None.
    """
    if self.gate is None or not view.has_learner(self.main):
      return None
    u = int(view.learner(self.main).updates)
    r = self.schedule.round_index(u)
    if r == self._round or u == 0 or self._snapshot is None:
      return None
    return GateRequest(
        subject=self.main,
        opponent=self._snapshot,
        generation=r,
        seed=int(self.gate.seed(r)),
        deals=self.gate_deals,
    )

  def _gate(self, view: LeagueView, r: int, u: int) -> bool:
    """Runs generation `r`'s promotion gate and records the attempt.

    Args:
      view: the league view.
      r: the generation.
      u: the main learner's update count.

    Returns:
      Whether the gate passed.

    Raises:
      ValueError: the view has no gate cell for generation `r`.
      AssertionError: generation `r` already had a gate attempt.
    """
    g = view.gate
    if g is None or g.generation != r or g.subject != self.main:
      raise ValueError(
          f"ScheduledPromotion: round {r} needs its gate cell on the view "
          f"(gate {self.promotion!r}), got {g}"
      )
    if r in self.gate_attempts:
      raise AssertionError(
          f"ScheduledPromotion: a second gate attempt for generation {r}"
      )
    passed = bool(self.gate.passes(g.subject, g.opponent, g))
    attempt = dict(
        dataclasses.asdict(g),
        snapshot=self._name(u),
        predicate=self.promotion,
        passed=passed,
    )
    self.gate_attempts = {**self.gate_attempts, r: attempt}
    if passed:
      self._passed = self._passed | {int(u)}
    logging.info(
        "ScheduledPromotion: gate generation %d %s vs %s seed %d: %+.4f +- "
        "%.4f [%+.4f, "
        "%+.4f] -> %s (%s); harvest %s",
        r,
        g.subject,
        g.opponent,
        g.seed,
        g.mean,
        g.se,
        g.lo,
        g.hi,
        "PASS" if passed else "FAIL",
        self.promotion,
        "allowed" if passed else "refused",
    )
    return passed

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.

    Raises:
      ValueError: the main learner is gone after its first round, or a round
        starts off an iteration boundary.
    """
    acts: list[Any] = []
    if view.has_learner(self.main):
      u = int(view.learner(self.main).updates)
    else:
      if self._round is not None:
        raise ValueError(
            f"ScheduledPromotion: the main learner {self.main!r} is gone"
        )
      acts.append(SpawnLearner(self.main, self.role))
      u = 0
    sch = self.schedule
    r = sch.round_index(u)
    if r != self._round:
      snap_step = sch.snapshot_step(u)
      if snap_step != u:
        raise ValueError(
            f"ScheduledPromotion: round {r} starts at update {snap_step} but "
            "the "
            f"first tick of it saw update {u}: snapshot_every "
            f"{sch.snapshot_every} must be a multiple of updates_per_iteration"
        )
      snap = self._name(u)
      # never waits on the gate
      acts.append(Admit(self.main, snap, "snapshot", "fixed"))
      steps = sch.pool_steps(u)
      if self.gate is not None:
        if u > 0:
          self._gate(view, r, u)
        steps = tuple(st for st in steps if st in self._passed)
      pool = tuple(self._name(s) for s in steps)
      keep = set(pool) | {snap}
      gone = [
          n
          for n in ((self._snapshot,) + self._pool)
          if n is not None and n not in keep
      ]
      for n in sorted(set(gone)):
        acts.append(
            SetLifecycle(n, "evicted", "superseded snapshot / past max_pool")
        )
      seated_steps = sch.active_pool(
          steps, rng.fold_in64(rng.key(self.seed), np.int64(r))
      )
      self._round, self._snapshot, self._pool = r, snap, pool
      self._seated = tuple(self._name(s) for s in seated_steps)
      self.last_info = dict(
          snapshot_step=int(u),
          pool_size=len(pool),
          pool_seated=len(self._seated),
      )
      if self.gate is not None and r in self.gate_attempts:
        a = self.gate_attempts[r]
        self.last_info.update(
            gate_mean=float(a["mean"]),
            gate_lo=float(a["lo"]),
            gate_passed=int(a["passed"]),
        )
      logging.info(
          "ScheduledPromotion: round %d at update %d: snapshot %s, pool %d, "
          "seated %s",
          r,
          u,
          snap,
          len(pool),
          self._seated,
      )
    acts.append(
        SetMix(
            self.main,
            sampler.snapshot_pool(
                self._snapshot, self._seated, sch.self_weight
            ),
        )
    )
    return acts


# -- fixed mix, composite ------------------------------------------------------
class FixedMix:
  """One learner with one constant mix (self-play: Mix(1.0, {})).

  Attributes:
    learner: the learner's name.
    mix: its constant Mix.
    role: its role.
  """

  def __init__(self, learner: str, mix: sampler.Mix, role: str = "main"):
    self.learner, self.mix, self.role = learner, mix, role

  def step(self, view: LeagueView) -> list[Any]:
    """The controller's actions for this view.

    Args:
      view: the league view.

    Returns:
      The actions.
    """
    acts: list[Any] = []
    if not view.has_learner(self.learner):
      acts.append(SpawnLearner(self.learner, self.role))
    return acts + [SetMix(self.learner, self.mix)]


class Composite:
  """Several controllers; their actions concatenated in order.

  What the League reads from its controller besides step() is forwarded from
  the children: the gate request of the one gated child, every child's draw
  guards, the last_meta of the one child that records a meta-strategy, and
  the children's last_info merged in order.

  Raises:
    ValueError: no children, more than one gated child, more than one child
      that records a meta-strategy, or two children's last_info share a key.

  Attributes:
    controllers: the child controllers, in order.
  """

  def __init__(self, controllers: Sequence[Any]):
    if not controllers:
      raise ValueError("a Composite needs at least one controller")
    self.controllers = tuple(controllers)
    gated = [
        c
        for c in self.controllers
        if hasattr(c, "gate_request") and getattr(c, "gate", None) is not None
    ]
    if len(gated) > 1:
      raise ValueError(
          f"a Composite holds at most one gated controller, got {len(gated)}"
      )
    metas = [c for c in self.controllers if hasattr(c, "last_meta")]
    if len(metas) > 1:
      raise ValueError(
          "a Composite holds at most one controller that records a "
          f"meta-strategy (last_meta), got {len(metas)}"
      )

  def step(self, view: LeagueView) -> list[Any]:
    """Every child's actions, concatenated in order.

    Args:
      view: the league view.

    Returns:
      The actions.
    """
    out: list[Any] = []
    for c in self.controllers:
      out += list(c.step(view))
    return out

  def gate_request(self, view: LeagueView) -> Optional[GateRequest]:
    """The gated child's gate request, or None.

    Args:
      view: the league view.

    Returns:
      The GateRequest, or None.

    Raises:
      ValueError: two children request a gate cell.
    """
    reqs = [
        r
        for r in (
            c.gate_request(view)
            for c in self.controllers
            if hasattr(c, "gate_request")
        )
        if r is not None
    ]
    if len(reqs) > 1:
      raise ValueError(
          f"two children of a Composite request a gate cell: {reqs}"
      )
    return reqs[0] if reqs else None

  def draw_guards(self) -> tuple[Any, ...]:
    """Every child's draw guards, in order.

    Returns:
      The guards.
    """
    guards = []
    for c in self.controllers:
      guards.extend(getattr(c, "draw_guards", lambda: ())())
    return tuple(guards)

  @property
  def last_meta(self) -> Optional[tuple[Any, ...]]:
    """The last_meta of the child that records one, or None."""
    for c in self.controllers:
      if hasattr(c, "last_meta"):
        return c.last_meta
    return None

  @property
  def last_info(self) -> dict[str, Any]:
    """The children's last_info, merged in order.

    Raises:
      ValueError: two children report the same key.
    """
    out: dict[str, Any] = {}
    for c in self.controllers:
      info = dict(getattr(c, "last_info", {}) or {})
      clash = sorted(set(out) & set(info))
      if clash:
        raise ValueError(
            f"two children of a Composite report last_info {clash}"
        )
      out.update(info)
    return out
