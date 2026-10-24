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

"""The run config and driver: every training run is a league.

`LeagueRunConfig` is what the `train` entry point runs: a game, a learner-kind
spec (`league.kinds`), the population's network, a `LeagueConfig`, fixed
opponents, a named controller with its keyword arguments, the run length in
league iterations, and optional in-loop evaluations. `run` builds the League,
resumes it when the checkpoint dir holds a commit, and trains it one
iteration at a time.

    cfg = runner.one_slot("ppo_kuhn", "kuhn_poker", kinds.PG(learner),
                          num_envs=256, num_steps=64, num_updates=1000,
                          policy_kwargs=network_kwargs)
    result = runner.run(cfg, checkpoint_dir="/tmp/ppo_kuhn")

Recipes: `one_slot` (one learner under a fixed mix: self-play, or fixed
opponents), `snapshot_pool` (self-play against a lagged snapshot plus a pool
of harvested snapshots) and `psro` (PSRO and pipeline PSRO).

Metrics. Every update of every learner writes one "train" row (keys
"<learner>/<metric>", step = the run's update count, int64 frames and
learner decisions, update_seconds, frames_per_s, decisions_per_s); the
artifact learner's rows also carry the in-loop evals ("eval/<key>/..." at
update 0 and every `every` updates) and BC's held-out metrics. One "league"
row per iteration holds members, learners, the controller's info and PSRO's
meta weights. Values that only change when the league's shape changes go to
the run's params until they move (`metrics.MetricsWriter` held). Kuhn and
leduc PG runs also track exact exploitability in-run (GameSpec.exact_eval).
In-run evals are low-sample monitoring estimates, not final evaluations.

Checkpoints. Every keep_every iterations, at a stop, at stop_after and at the
end: League.save, then train_state.json naming it (the commit,
`league.commit`), then the artifact learner's artifact at <ckpt> (step = the
kind's artifact step), published from the committed state, then the
registered on-commit hooks. A dir holding a train_state.json resumes (same
config by `league.config_common.resume_hash`; a run whose step size does not
depend on its length may extend num_iterations). Between keeps a resume
checkpoint (`league.resume`) is taken every resume_every_s seconds and when
the preemption notice is up; once the notice is up, evals still due are
skipped and the run ends after the iteration in flight (result.preempted).
Without a checkpoint_dir the members go to a fresh temporary directory.

Wall budget. With wall_budget_s, the run ends at the first iteration boundary
at or past that many seconds of clock() since `run` started: the artifact
learner's evals not yet run at that update run, the iteration is committed
with train_state "wall_budget" {budget_s, reached_at, of}, and the result's
`wall_budget` holds the same record (the run is incomplete). Every eval of
the call runs under an eval deadline (`core.deadline`) wall_budget_s +
eval_grace_s after the start, so an eval still running then raises
EvalOverBudgetError instead of running on.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

from absl import logging
import jax
import numpy as np

from open_spiel.python.scale import metrics as metrics_lib
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import layout
from open_spiel.python.scale.ckpt import records
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import deadline as deadline_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import exploitability
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import mcts_baseline
from open_spiel.python.scale.eval import registry as eval_registry
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller as controller_lib
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import resume
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.league import siblings as siblings_lib
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import uniform

KIND = "league"
STORE_DIR = layout.STORE_DIR


# -- controllers ---------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ControllerConfig:
  """What a controller factory receives.

  Attributes:
    name: the registered controller name.
    kwargs: LeagueRunConfig.controller_kwargs.
    run: the LeagueRunConfig.
  """

  name: str
  kwargs: Mapping[str, Any]
  run: Any


class _ControllerEntry(NamedTuple):
  """A registered controller.

  Attributes:
    factory: builds the controller from its ControllerConfig.
    single_learner: None, or the keyword argument naming its one learner.
  """

  factory: Callable[[ControllerConfig], Any]
  single_learner: Optional[str]


_CONTROLLERS: dict[str, _ControllerEntry] = {}


def register_controller(
    name: str,
    factory: Callable[[ControllerConfig], Any],
    *,
    single_learner: Optional[str] = None,
) -> None:
  """Registers a controller a LeagueRunConfig can name.

  Args:
    name: the name.
    factory: ControllerConfig -> the controller.
    single_learner: for a controller that runs exactly one learner, spawned
      at iteration 0 and never retired: the controller_kwargs key naming that
      learner (its schedule horizon is then the run's length).

  Raises:
    ValueError: the name is registered with another factory.
  """
  entry = _ControllerEntry(factory, single_learner)
  old = _CONTROLLERS.get(name)
  if old is not None and old != entry:
    raise ValueError(f"controller {name!r} is already registered")
  _CONTROLLERS[name] = entry


def controller_names() -> tuple[str, ...]:
  """The registered controller names.

  Returns:
    The names, in registration order.
  """
  return tuple(_CONTROLLERS)


def single_learner_key(name: str) -> Optional[str]:
  """The controller_kwargs key naming a single-learner controller's learner.

  Args:
    name: the controller name.

  Returns:
    The key, or None for a multi-learner controller.
  """
  return _CONTROLLERS[name].single_learner


def _fixed_mix(c: ControllerConfig):
  """The FixedMix controller of a config.

  Args:
    c: The controller config.

  Returns:
    The controller.
  """
  kw = dict(c.kwargs)
  return controller_lib.FixedMix(
      kw.get("learner", "main"),
      sampler.Mix(
          float(kw.get("self_play", 1.0)), dict(kw.get("weights") or {})
      ),
      role=kw.get("role", "main"),
  )


def _kwargs_factory(cls):
  """A controller factory that passes the config's kwargs to `cls`.

  Args:
    cls: the controller class.

  Returns:
    ControllerConfig -> the controller.
  """

  def make(c: ControllerConfig):
    """Builds the controller from the config's kwargs.

    Args:
      c: the controller config.

    Returns:
      The controller.
    """
    return cls(**dict(c.kwargs))

  make.__qualname__ = f"make_{cls.__name__}"
  return make


register_controller("psro", _kwargs_factory(controller_lib.PSRO))
register_controller("fixed_mix", _fixed_mix, single_learner="learner")
register_controller(
    "scheduled_promotion",
    _kwargs_factory(controller_lib.ScheduledPromotion),
    single_learner="main",
)
register_controller(
    "gated_pool",
    _kwargs_factory(controller_lib.GatedPool),
    single_learner="main",
)
register_controller(
    "continual_exploiter", _kwargs_factory(controller_lib.ContinualExploiter)
)
register_controller("br_oracle", _kwargs_factory(controller_lib.BROracle))
register_controller("pbt", _kwargs_factory(controller_lib.PBT))
register_controller(
    "lagged_audit",
    _kwargs_factory(controller_lib.LaggedAudit),
    single_learner="main",
)


# -- on-commit hooks -----------------------------------------------------------
OnCommit = Callable[[str, int, str], None]  # (ckpt_dir, step, ref)
_ON_COMMIT: dict[str, OnCommit] = {}
_ALWAYS_ON_COMMIT: list[Any] = []


def register_on_commit(
    name: str, fn: OnCommit, *, always: bool = False
) -> None:
  """Registers a hook run after each keep's commit and publish.

  A run's LeagueRunConfig.on_commit names the hooks it runs; a hook
  registered with always=True runs for every run. fn(ckpt_dir, step, ref)
  reads train_state.json for the committed record.

  Args:
    name: the hook name.
    fn: the hook.
    always: run the hook for every run.

  Raises:
    ValueError: another hook is registered under `name`.
  """
  if _ON_COMMIT.get(name, fn) is not fn:
    raise ValueError(f"on-commit hook {name!r} is already registered")
  _ON_COMMIT[name] = fn
  if always and name not in _ALWAYS_ON_COMMIT:
    _ALWAYS_ON_COMMIT.append(name)


# -- in-loop evals -------------------------------------------------------------
# Game-agnostic in-loop evaluators, defined on every 2-player game:
#   vs_uniform  against the uniform-random policy (SAMPLE)
#   vs_mcts     against the random-rollout MCTS bot with InLoopEval.simulations
#               simulations (perfect-information games only)
GENERIC_EVALS = ("vs_uniform", "vs_mcts")


class _EvalEntry(NamedTuple):
  """A registered reference eval.

  Attributes:
    env: the game it plays.
    reference_type: None, or the type its reference must have.
    build: None, or builds the evaluator.
  """

  env: str
  reference_type: Optional[type[Any]]
  build: Optional[Callable[..., Any]]


_REFERENCE_EVALS: dict[str, _EvalEntry] = {}


def register_reference_eval(
    name: str,
    env: str,
    reference_type: Optional[type[Any]],
    # The keyword is part of the public signature.
    build: Callable[..., Any],  # pylint: disable=redefined-outer-name
) -> None:
  """Registers an in-loop evaluator defined on game `env`.

  build(ev, env, mesh, cache_dir) -> its eval.match.Evaluator; cache_dir is
  <run dir>/refs, where a reference may be materialized for the run. A
  reference_type means the eval plays a configured reference
  (InLoopEval.reference, an instance of it).

  Args:
    name: the eval name.
    env: the game the eval is defined on.
    reference_type: the reference's type, or None for no configured
      reference.
    build: builds the Evaluator.

  Raises:
    ValueError: name is a generic eval, or is already defined otherwise.
  """
  if name in GENERIC_EVALS:
    raise ValueError(f"{name!r} is a generic in-loop eval")
  entry = _EvalEntry(env, reference_type, build)
  old = _REFERENCE_EVALS.get(name)
  if old is not None and old != entry:
    raise ValueError(f"in-loop eval {name!r} is already defined")
  _REFERENCE_EVALS[name] = entry


def _game_evals() -> dict[str, str]:
  """{eval name: game} of the evaluators the game table declares.

  {eval name: game} of the evaluators the game table declares
  (GameSpec.reference_evals, built by eval.registry).

  Returns:
    The {eval name: game} mapping.
  """
  return dict(env_registry.by_field("reference_evals"))


def eval_names() -> dict[str, str]:
  """{eval name: the game it is defined on} of every non-generic in-loop eval.

  Returns:
    The mapping.
  """
  return {**_game_evals(), **{n: e.env for n, e in _REFERENCE_EVALS.items()}}


@dataclasses.dataclass(frozen=True)
class InLoopEval:
  """An in-loop evaluation of the run's artifact learner.

  The evaluator `name` plays Entrant("<learner>@<u>", the run's artifact
  policy, ARGMAX) against its reference at update 0, every `every` updates
  and at the last update, on `deals` duplicate deal pairs keyed by
  fold_in64(key(seed), u). Metrics are "eval/<key>/{mean, se, lo, hi,
  n_deals, variance_reduction}" with key = label, or "h2h_<reference label>"
  for a reference eval, "vs_mcts<simulations>" for vs_mcts, else the name.

  Attributes:
    name: a GENERIC_EVALS name or a registered eval (`eval_names`).
    every: updates between evaluations.
    deals: duplicate deal pairs.
    seed: the deals' seed.
    reference: a reference eval's reference.
    label: the metric key.
    simulations: vs_mcts's simulations per move (0 for every other eval).
  """

  name: str
  every: int
  deals: int
  seed: int = 0
  reference: Optional[Any] = None
  label: str = ""
  simulations: int = 0

  def __post_init__(self):
    known = eval_names()
    assert self.name in known or self.name in GENERIC_EVALS, (
        self.name,
        sorted(known),
        GENERIC_EVALS,
    )
    if isinstance(self.simulations, bool) or not isinstance(
        self.simulations, int
    ):
      raise TypeError(
          f"eval {self.name!r}: simulations must be an int, got "
          f"{self.simulations!r}"
      )
    if (self.name == "vs_mcts") != (self.simulations > 0):
      raise ValueError(
          f"eval {self.name!r}: simulations {self.simulations} (vs_mcts takes "
          "N >= 1; every other eval 0)"
      )
    assert (
        self.every >= 1
    ), "an in-loop eval runs every >= 1 updates (omit it to turn it off)"
    if self.deals < 2:
      raise ValueError(f"deals {self.deals}: an SE needs at least 2 deal pairs")
    takes_reference = [
        n for n, e in _REFERENCE_EVALS.items() if e.reference_type is not None
    ]
    if (self.name in takes_reference) != (self.reference is not None):
      raise ValueError(
          f"eval {self.name!r}: a reference is given exactly for "
          f"{takes_reference}"
      )
    if self.reference is not None:
      want = _REFERENCE_EVALS[self.name].reference_type
      if not isinstance(self.reference, want):
        raise TypeError(
            f"eval {self.name!r}: reference must be a {want.__name__}, "
            f"got {type(self.reference).__name__}"
        )
    if self.label and not all(c.isalnum() or c in "_.-" for c in self.label):
      raise ValueError(
          f"eval label {self.label!r}: letters, digits, '_', '.', '-' only"
      )

  @property
  def key(self) -> str:
    """The metric name stem of this eval (eval/<key>/...)."""
    if self.label:
      return self.label
    if self.reference is not None:
      return f"h2h_{self.reference.label}"
    if self.name == "vs_mcts":
      return f"vs_mcts{self.simulations}"
    return self.name


# -- the run config ------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LeagueRunConfig:
  """A training run (module doc).

  Attributes:
    name: the run's name.
    env: the game.
    env_kwargs: the game's options.
    learner: a learner-kind spec (`league.kinds`), every learning slot's.
    league: the LeagueConfig.
    controller: a registered controller name.
    controller_kwargs: the controller's keyword arguments (JSON).
    num_iterations: league iterations, the run length (a Stop ends it earlier).
    policy_kwargs: the PG / BC / AZ network's keyword arguments for the
      policy registry (AZ: empty for the AZConfig's own network).
    policy_arch: the network's registered architecture.
    fixed_opponents: refs.FixedOpponent tuple.
    num_initial: random-init members "init<i>@0".
    keep_every: iterations between keeps.
    seed: the run's seed (equal to league.seed).
    artifact_learner: the learner whose params are the run's artifact ("":
      none).
    eval: an InLoopEval, a tuple of them, or None.
    metadata: free-form string labels recorded with the config; the framework
      never reads them.
    on_commit: registered on-commit hook names (`register_on_commit`).
    notes: free text.
  """

  name: str
  env: str
  learner: Any
  league: loop.LeagueConfig
  controller: str
  controller_kwargs: Mapping[str, Any]
  num_iterations: int
  policy_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  policy_arch: str = "network"
  fixed_opponents: tuple[Any, ...] = ()
  num_initial: int = 0
  keep_every: int = 1
  seed: int = 0
  env_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  artifact_learner: str = "main"
  eval: Any = None
  metadata: Mapping[str, str] = dataclasses.field(default_factory=dict)
  on_commit: tuple[Any, ...] = ()
  notes: str = ""

  def __post_init__(self):
    assert self.env in common.env_names(), self.env
    # unknown / non-JSON / out-of-range env options raise here
    common.env_spec(self)
    kind = kinds.of(self.learner)
    if self.controller not in _CONTROLLERS:
      raise KeyError(
          f"unknown controller {self.controller!r} (registered "
          f"{sorted(_CONTROLLERS)})"
      )
    assert isinstance(self.league, loop.LeagueConfig), type(self.league)
    assert (
        self.num_iterations >= 1
        and self.num_initial >= 0
        and self.keep_every >= 1
    )
    if self.league.seed != self.seed:
      raise ValueError(f"league.seed {self.league.seed} != seed {self.seed}")
    json.dumps(dict(self.policy_kwargs), sort_keys=True)
    json.dumps(dict(self.controller_kwargs), sort_keys=True)
    json.dumps(dict(self.env_kwargs), sort_keys=True)
    if not all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in self.metadata.items()
    ):
      raise TypeError(
          f"metadata maps strings to strings, got {dict(self.metadata)!r}"
      )
    unknown = [n for n in self.on_commit if n not in _ON_COMMIT]
    if unknown:
      raise KeyError(
          f"on_commit hooks {unknown} are not registered ({sorted(_ON_COMMIT)})"
      )
    names = [o.name for o in self.fixed_opponents]
    for o in self.fixed_opponents:
      assert isinstance(o, refs.FixedOpponent), type(o)
    if len(set(names)) != len(names):
      raise ValueError(f"repeated fixed opponent names {names}")
    total = self.run_updates()
    single = single_learner_key(self.controller)
    if not getattr(kind, "multi_learner", False) and single is None:
      raise ValueError(
          f"a {kind.name} league trains one learner: use a single-learner "
          "controller ("
          f"{sorted(n for n in _CONTROLLERS if single_learner_key(n))})"
      )
    if single is not None:
      who = self.controller_kwargs.get(single, "main")
      if self.artifact_learner not in ("", who):
        raise ValueError(
            f"artifact_learner {self.artifact_learner!r} is not the "
            "controller's "
            f"learner {who!r}"
        )
      h = self.learner.fixed_horizon()
      if h is not None and h != total:
        raise ValueError(
            f"the learner's schedule horizon {h} != the run's {total} updates "
            "(num_iterations x updates_per_iteration): set both together with "
            "league.config_common.with_overrides(cfg, num_updates=...)"
        )
    elif not self.learner.constant_lr():
      raise ValueError(
          "learners of a multi-learner league train from scratch per slot: "
          "they "
          "need a constant LR"
      )
    self.learner.validate_run(self)
    evals = self.evals()
    games = eval_names()
    for ev in evals:
      if ev.name in GENERIC_EVALS:
        continue  # any 2-player game (eval.match.Evaluator checks at build)
      if games[ev.name] != self.env:
        raise ValueError(
            f"eval {ev.name!r} is defined on {games[ev.name]!r}, "
            f"not env {self.env!r}"
        )
    if evals and not self.artifact_learner:
      raise ValueError(
          "an in-loop eval scores the artifact learner: set artifact_learner"
      )
    keys = [ev.key for ev in evals]
    if len(set(keys)) != len(keys):
      raise ValueError(
          f"in-loop evals share a metric key: {keys} (give each a label)"
      )

  def run_updates(self) -> int:
    """The run length in updates: iterations x updates per iteration.

    Returns:
      The number of updates.
    """
    return int(self.num_iterations) * int(self.league.updates_per_iteration)

  def evals(self) -> tuple[Any, ...]:
    """The in-loop evals.

    () without one, (eval,) for one InLoopEval, the tuple as given.

    Returns:
      The evals.

    Raises:
      TypeError: eval is neither an InLoopEval nor a non-empty tuple of them.
    """
    if self.eval is None:
      return ()
    evs = (self.eval,) if isinstance(self.eval, InLoopEval) else self.eval
    if not isinstance(evs, tuple) or not evs:
      raise TypeError(
          "eval must be an InLoopEval or a non-empty tuple of them, got "
          f"{type(self.eval).__name__}"
      )
    for ev in evs:
      if not isinstance(ev, InLoopEval):
        raise TypeError(
            f"eval entries must be InLoopEval, got {type(ev).__name__}"
        )
    return evs

  def build_controller(self):
    """The run's controller, built by its registered factory.

    Returns:
      The controller.
    """
    return _CONTROLLERS[self.controller].factory(
        ControllerConfig(self.controller, dict(self.controller_kwargs), self)
    )


class LeagueRunResult(NamedTuple):
  """What `run` returns.

  Attributes:
    config: the config.
    league: the League object.
    metrics: the artifact learner's update rows of this call.
    rows: one "league" row per iteration of this call.
    heldout: BC's last held-out metrics.
    preempted: the preemption notice ended this call (after a resume
      checkpoint when the run has a checkpoint_dir); the run is not finished.
    wall_budget: None, or {budget_s, reached_at, of} when the wall budget
      ended this call (module doc).
  """

  config: LeagueRunConfig
  league: loop.League
  metrics: list[Any]
  rows: list[Any]
  heldout: Optional[dict[str, Any]]
  preempted: bool = False
  wall_budget: Optional[dict[str, Any]] = None

  @property
  def learner(self):
    """The league's Learner."""
    return self.league.learner

  @property
  def state(self):
    """The artifact learner's state (None without one).

    The artifact learner's state (None without one). A retired learner's
    state is released at the first save after its retirement, so after a
    checkpointed run this is None for it; its committed state is in the run's
    checkpoint.
    """
    lg, name = self.league, self.config.artifact_learner
    if not name:
      return None
    ls = lg.learners.get(name) or lg.retired.get(name)
    return None if ls is None else ls.state


def scratch_dir_for(
    run_dir: str, kind: str, workdir: Optional[str] = None
) -> str:
  """A local dir for a learner kind that reads only local files.

  A spec class with `needs_scratch_dir = True` (LearnerKind.build_driver):
  <run_dir>/<kind>_resume for a local run dir, a fresh temporary dir under
  workdir (default: the working dir) for a gs:// one.

  Args:
    run_dir: the run dir.
    kind: the learner kind.
    workdir: the parent of a temporary dir.

  Returns:
    The dir.
  """
  name = f"{kind}_resume"
  if not run_dir.startswith(artifact.GCS_PREFIX):
    return os.path.join(run_dir, name)
  return tempfile.mkdtemp(
      prefix=f"{name}-", dir=workdir if workdir is not None else os.getcwd()
  )


def build(
    cfg: LeagueRunConfig,
    run_dir: str,
    *,
    mesh=None,
    git_commit: Optional[str] = None,
    data_dir: Optional[str] = None,
    scratch_dir: Optional[str] = None,
    on_update=None,
) -> loop.League:
  """The League of cfg, with its members under run_dir.

  Args:
    cfg: the run config.
    run_dir: the run dir.
    mesh: the device mesh.
    git_commit: recorded in artifacts.
    data_dir: BC from recorded shards.
    scratch_dir: a local dir for kinds that need one.
    on_update: called after each update.

  Returns:
    The League.

  Raises:
    ValueError: a fixed opponent plays another game.
  """
  env = common.build_env_by_name(cfg.env, **dict(cfg.env_kwargs))
  if scratch_dir is None and getattr(cfg.learner, "needs_scratch_dir", False):
    scratch_dir = scratch_dir_for(run_dir, cfg.learner.kind)
  driver = kinds.make_driver(
      cfg.learner,
      env,
      cfg.policy_kwargs,
      data_dir=data_dir,
      scratch_dir=scratch_dir,
      policy_arch=cfg.policy_arch,
  )
  root = jax.random.fold_in(jax.random.key(cfg.seed), 2)
  initial = {
      f"init{i}@0": driver.policy.init_params(jax.random.fold_in(root, i))
      for i in range(cfg.num_initial)
  }
  fixed = {}
  for o in cfg.fixed_opponents:
    pol, params = o.ref.build()
    if pol.spec.game != env.name:
      raise ValueError(
          f"fixed opponent {o.name} plays {pol.spec.game}, not {env.name}"
      )
    fixed[o.name] = (pol, params, o.rule)
  store = artifact_store.ArtifactStore(os.path.join(run_dir, STORE_DIR))
  return loop.League(
      env,
      driver.policy,
      driver.learner,
      cfg.build_controller(),
      store,
      run_dir,
      cfg.league,
      initial_members=initial,
      fixed_opponents=fixed,
      mesh=mesh,
      git_commit=git_commit,
      driver=driver,
      on_update=on_update,
  )


# League-row keys that describe the league's shape rather than measure it: a
# held setting until they move (metrics.MetricsWriter held), then a series.
LEAGUE_HELD = ("members", "learners", "retired", "stopped")
LEAGUE_HELD_PREFIXES = ("controller.",)


def is_timing_key(key: str) -> bool:
  """Whether a learner metric is a wall-clock timing.

  "seconds" or "*_seconds": timings go to the metrics stream, never to
  LeagueRunResult rows.

  Args:
    key: the metric name.

  Returns:
    True for a timing.
  """
  return key == "seconds" or key.endswith("_seconds")


def league_held(row: Mapping[str, Any]) -> tuple[Any, ...]:
  """The keys of a league row that are held settings.

  Args:
    row: a league row.

  Returns:
    The keys.
  """
  return tuple(
      k for k in row if k in LEAGUE_HELD or k.startswith(LEAGUE_HELD_PREFIXES)
  )


# Games small enough for the exact exploitability (eval.exploitability.GameTree)
# of the artifact learner's sampling policy to be tracked in-run, as
# eval/exploitability and eval/nash_conv, about EXACT_EVAL_POINTS times per run
# (and at the first and last update), for a PG run of an in-graph JAX policy.
# Exact values; it reads the params and draws no training key.
EXACT_EVAL_POINTS = 100
# (env name, env kwargs) -> GameTree, built once per process
_EXACT_TREES: dict[Any, Any] = {}


def exact_eval_game(name: str) -> bool:
  """Whether game `name` declares an exact in-run eval (GameSpec.exact_eval).

  Args:
    name: the game name.

  Returns:
    True when it does.
  """
  return name in env_registry.names() and env_registry.game(name).exact_eval


def exact_eval_every(total: Optional[int]) -> int:
  """Updates between exact evals: EXACT_EVAL_POINTS over the run.

  Args:
    total: the run's updates, or None.

  Returns:
    The interval, at least 1.
  """
  return max(1, -(-int(total) // EXACT_EVAL_POINTS)) if total else 1


def build_exact_eval(
    cfg: "LeagueRunConfig", env: types.Env, policy: types.Policy
):
  """The GameTree of cfg's env when the run qualifies, else None.

  A run qualifies with GameSpec.exact_eval, a PG learner and an in-graph JAX
  slot policy. A search policy is stochastic per history and has no plain
  table, so AZ runs do not qualify.

  Args:
    cfg: the run config.
    env: the run's env.
    policy: the slot policy.

  Returns:
    The GameTree, or None.
  """
  if not exact_eval_game(cfg.env) or not isinstance(cfg.learner, kinds.PG):
    return None
  spec = policy.spec
  if spec.framework != "jax" or spec.capability not in arena_lib.IN_GRAPH:
    return None

  key = (cfg.env, json.dumps(dict(cfg.env_kwargs), sort_keys=True))
  if key not in _EXACT_TREES:
    _EXACT_TREES[key] = exploitability.GameTree(env)
  return _EXACT_TREES[key]


def _exact_row(tree, lg: loop.League, ls) -> dict[str, Any]:
  """{exploitability, nash_conv} of the learner's policy (SAMPLE rule).

  Args:
    tree: The game tree.
    lg: The league.
    ls: The learner slot.

  Returns:
    {exploitability, nash_conv}.
  """
  r = exploitability.exploitability(
      lg.env,
      lg.slot_policy,
      lg.driver.wrap(lg.learner.params(ls.state)),
      types.SAMPLE,
      tree=tree,
  )
  return {
      "exploitability": float(r.exploitability),
      "nash_conv": float(r.nash_conv),
  }


def _summary(lg: loop.League, rec: Mapping[str, Any]) -> dict[str, Any]:
  """The summary row of a league iteration.

  Args:
    lg: the league.
    rec: the iteration's history record.

  Returns:
    The flat row of counts, meta weights and controller info.
  """
  row: dict[str, Any] = dict(
      members=int(rec["members"]),
      learners=len(lg.learners),
      retired=len(lg.retired),
      stopped=int(lg.stopped),
      iteration=int(rec["iteration"]),
  )
  for name, d in rec["learners"].items():
    row[f"{name}.updates"] = d["updates"]
    row[f"{name}.learner_decisions"] = d["decisions"]
  meta = getattr(lg.controller, "last_meta", None)
  if meta is not None:
    for n, w in zip(meta[0], np.asarray(meta[1], np.float64)):
      row[f"meta.{n}"] = float(w)
  for k, v in dict(getattr(lg.controller, "last_info", {}) or {}).items():
    row[f"controller.{k}"] = v
  return row


EVAL_SERIES = "eval"  # an in-loop eval's series mean: eval/<key>


def eval_series_mean(key: str, est) -> dict[str, Any]:
  """An in-loop eval's estimate as the series mean of its n samples.

  The estimate is an eval.stats.Estimate (mean, se, n); the series mean
  {name, n, sum, sumsq} (metrics.emit_series) is chosen so a consumer's band
  (sample variance with n - 1, mean +- 1.96 SE) is the Evaluator's own
  lo / hi: sum = n mean, sumsq = (n - 1) n se^2 + n mean^2.

  Args:
    key: the eval's metric key.
    est: the estimate.

  Returns:
    The series mean.

  Raises:
    ValueError: n < 2.
  """
  n = int(est.n)
  if n < 2:
    raise ValueError(
        f"eval {key}: a series mean band needs n >= 2 samples, got {n}"
    )
  mean, se = float(est.mean), float(est.se)
  return {
      "name": f"{EVAL_SERIES}/{key}",
      "n": n,
      "sum": n * mean,
      "sumsq": (n - 1) * n * se * se + n * mean * mean,
  }


def _eval_row(
    ev: InLoopEval,
    evaluator,
    lg: loop.League,
    ls,
    u: int,
    series_step: Optional[int] = None,
) -> dict[str, Any]:
  """One in-loop eval's metrics row.

  One in-loop eval's metrics row ("<key>/mean", se, lo, hi, n_deals, the
  variance ratio). series_step: also emit its series mean at that step.

  Args:
    ev: the InLoopEval.
    evaluator: its built evaluator.
    lg: the League.
    ls: the artifact learner's LearnerSlot.
    u: the learner's update.
    series_step: None, or the step to also emit the series mean at.

  Returns:
    The metrics row.
  """
  pol, params = lg.driver.artifact(ls)  # what the run publishes
  ent = match.Entrant(f"{ls.name}@{u}", pol, params, types.ARGMAX)
  key = x64.fold_in64(jax.random.key(ev.seed), np.int64(u))
  r = evaluator.vs_reference(ent, ev.deals, key)
  name = ev.key
  if series_step is not None:
    metrics_lib.emit_series(
        int(series_step),
        {},
        step_unit="updates",
        mean=eval_series_mean(name, r.mean),
    )
  row = {
      f"{name}/mean": r.mean.mean,
      f"{name}/se": r.mean.se,
      f"{name}/lo": r.mean.lo,
      f"{name}/hi": r.mean.hi,
      f"{name}/n_deals": int(r.n_deals),
  }
  # The duplicate-deal variance ratio is undefined when the pair SE is 0 (inf:
  # every pair sums to zero; nan: every game scored the same). The metrics
  # writer refuses non-finite values, so the row says which case it is.
  vr = float(r.variance_reduction)
  row[f"{name}/variance_reduction_defined"] = int(np.isfinite(vr))
  if np.isfinite(vr):
    row[f"{name}/variance_reduction"] = vr
  else:
    logging.info(
        "eval %s at update %d: variance_reduction %s (pair SE 0); not written",
        name,
        u,
        vr,
    )
  return row


REFS_DIR = "refs"  # <run dir>/refs: references materialized for this run


def build_eval(
    ev: InLoopEval, env: types.Env, mesh=None, cache_dir: Optional[str] = None
):
  """The Evaluator of one in-loop eval (its reference resolved now).

  Args:
    ev: the InLoopEval.
    env: the run's env.
    mesh: the device mesh.
    cache_dir: where a reference may be materialized.

  Returns:
    The Evaluator.

  Raises:
    KeyError: an unknown eval.
  """
  assert isinstance(ev, InLoopEval), type(ev)
  if ev.name in GENERIC_EVALS:
    return build_generic_eval(ev, env, mesh=mesh)
  entry = _REFERENCE_EVALS.get(ev.name)
  if entry is not None:
    return entry.build(ev, env, mesh, cache_dir)
  if ev.name in _game_evals():
    return eval_registry.build(ev.name, env, mesh=mesh)
  raise KeyError(f"in-loop eval {ev.name!r} has no registered builder")


def build_generic_eval(ev: InLoopEval, env: types.Env, mesh=None):
  """The Evaluator of a GENERIC_EVALS eval on `env`.

  The reference plays the run's env options, so a non-default board is the
  one scored.

  Args:
    ev: the InLoopEval.
    env: the run's env.
    mesh: the device mesh.

  Returns:
    The Evaluator.

  Raises:
    KeyError: not a generic eval.
  """
  if ev.name == "vs_uniform":

    ref = match.Entrant(
        "uniform",
        uniform.UniformPolicy(
            env.name, int(env.num_actions), int(env.num_players)
        ),
        {},
        types.SAMPLE,
    )
    return match.Evaluator(
        env,
        match.EvaluatorConfig(name=ev.key, reference=lambda: ref),
        mesh=mesh,
    )
  if ev.name == "vs_mcts":

    kw = dict(env_registry.spec_of(env).kwargs)
    default = dict(env_registry.spec(env.name).kwargs)
    return match.Evaluator(
        env,
        mcts_baseline.vs_mcts_config(
            env.name, ev.simulations, env_kwargs=None if kw == default else kw
        ),
        mesh=mesh,
    )
  raise KeyError(ev.name)


def build_evals(
    cfg: LeagueRunConfig,
    env: types.Env,
    mesh=None,
    run_dir: Optional[str] = None,
) -> list[Any]:
  """[(InLoopEval, Evaluator)] for every in-loop eval of cfg.

  Args:
    cfg: the run config.
    env: the run's env.
    mesh: the device mesh.
    run_dir: the run dir; references are cached under it.

  Returns:
    The pairs.
  """
  cache = os.path.join(run_dir, REFS_DIR) if run_dir else None
  return [(ev, build_eval(ev, env, mesh, cache)) for ev in cfg.evals()]


# -- publishing ----------------------------------------------------------------
def _publish_targets(lg: loop.League, ls) -> list[Any]:
  """[(mode or None, (policy, params))] the run publishes for ls.

  [(mode or None, (policy, params))] the run publishes for ls, in write
  order: the plain sibling first, then the operator, else one artifact.

  Args:
    lg: the League.
    ls: the artifact learner's LearnerSlot.

  Returns:
    The (mode, (policy, params)) pairs, in write order.
  """
  if lg.driver.publishes_siblings:
    return [
        ("plain", lg.driver.plain_artifact(ls)),
        ("search", lg.driver.artifact(ls)),
    ]
  return [(None, lg.driver.artifact(ls))]


def _check_buildable(policy: types.Policy) -> None:
  """A published policy must be rebuilt by the policy registry.

  A published policy must be rebuilt by the policy registry from its
  recorded identity; a training-only wrapper is not.

  Args:
    policy: the policy to publish.
  """
  sp = policy.spec
  try:
    built = registry.build(sp.arch_name, dict(sp.arch_kwargs))
  except (KeyError, TypeError, ValueError, AssertionError) as e:
    raise ValueError(
        f"the run's artifact policy {sp.arch_name} is not registry-buildable "
        f"({type(e).__name__}: {e}): it cannot be published"
    ) from e
  if built.spec != sp:
    raise ValueError(
        f"the registry rebuilds {sp.arch_name} as {built.spec}, not {sp}"
    )


def _publish_run_artifact(
    ckpt: str,
    cfg: LeagueRunConfig,
    lg: loop.League,
    run_id: Optional[str | int],
    previous_step: Optional[int],
) -> Optional[int]:
  """Writes the artifact learner's artifact at <ckpt> from the committed state.

  previous_step is the artifact step of the commit before: an unchanged step
  was published then and is not written again. Params a crashed publish
  left without their step metadata are moved aside (never deleted), then the
  step is written. Sibling runs write <ckpt>/plain first, then
  <ckpt>/search, each the same way.

  Args:
    ckpt: the checkpoint dir.
    cfg: the run config.
    lg: the League.
    run_id: the run's id.
    previous_step: the artifact step of the commit before, or None.

  Returns:
    the artifact step, or None without an artifact learner.
  """
  name = cfg.artifact_learner
  ls = lg.learners.get(name) if name else None
  if ls is None:
    return None
  step = lg.driver.artifact_step(ls)
  for mode, (policy, params) in _publish_targets(lg, ls):
    _check_buildable(policy)
    d = ckpt if mode is None else siblings_lib.mode_dir(ckpt, mode)
    if artifact.exists(d, step):
      if previous_step is not None and int(previous_step) == int(step):
        continue  # published with the previous commit (no update since)
      raise ValueError(
          f"{d}: artifact step {step} is already published, but the previous "
          f"commit's artifact step is {previous_step}: not this commit's"
      )
    aside = artifact.clear_partial(d, step, str(lg.attempt))
    if aside is not None:
      logging.warning(
          "league run: %s@%d had params but no manifest (a crashed publish); "
          "set aside in %s",
          d,
          step,
          aside,
      )
    extra = dict(lg.driver.artifact_extra())
    if mode is not None:
      extra["extensions"] = dict(
          extra.get("extensions", {}),
          **{
              siblings_lib.RECORD_KEY: siblings_lib.sibling_record(
                  mode, policy, params, lg.git_commit
              )
          },
      )
    artifact.write_artifact(
        d,
        step,
        policy,
        params,
        rules_sha=common.rules_sha(lg.env),
        config_hash=common.config_hash(cfg),
        algo=lg.driver.algo,
        env_name=lg.env.name,
        default_rule=lg.driver.default_rule,
        run_id=None if run_id is None else str(run_id),
        git_commit=lg.git_commit,
        env_spec=env_registry.spec_of(lg.env).to_json(),
        **extra,
    )
  return step


# -- run identity and commits --------------------------------------------------
# train_state.json and resume_state.json keys of the run's identity
# (stored under ckpt.records' names).
_RUN_KEY, _ATTEMPT_KEY = "run_id", "attempt_id"


def run_identity(
    ts: Optional[Mapping[str, Any]],
    run_id: Optional[str | int],
    attempt_id: Optional[int],
) -> tuple[Optional[int], Optional[int], list[Any]]:
  """(run id, attempt id, attempts) of a process.

  The run's id is the one its dir belongs to: on a resume the dir's recorded
  id (an explicit run_id naming another run raises), else run_id (the
  launcher's), else this attempt's id (a fresh run's first attempt is the
  run). attempt_id is this process's (`ckpt.provenance.resolve_run_ids`); a
  relaunch is a new attempt of the same run. attempts: every attempt that
  committed to the run, in order, plus this one once.

  Args:
    ts: the train_state.json record, or None.
    run_id: the launcher's run id, or None.
    attempt_id: this process's attempt id, or None.

  Returns:
    (run id, attempt id, attempts).

  Raises:
    ValueError: run_id names another run than the dir's.
  """
  ts = None if ts is None else records.decode(ts)
  ts_run = None if ts is None else ts.get(_RUN_KEY)
  if ts_run is not None:
    if run_id is not None and str(run_id) != str(ts_run):
      raise ValueError(
          f"run id {run_id} but the checkpoint dir is run {ts_run}'s (its "
          f"{run_state.TRAIN_STATE}); a resume names the run it continues"
      )
    the_run = ts_run
  else:
    the_run = run_id if run_id is not None else attempt_id
  attempts = list((ts or {}).get("attempts") or [])
  if attempt_id is not None and str(attempt_id) not in {
      str(a) for a in attempts
  }:
    attempts.append(attempt_id)
  return the_run, attempt_id, attempts


def _commit(
    ckpt: str,
    cfg: LeagueRunConfig,
    lg: loop.League,
    run_id: Optional[str | int],
    attempt_id: Optional[int] = None,
    attempts: tuple[Any, ...] = (),
    wall_budget: Optional[dict[str, Any]] = None,
) -> None:
  """A keep: League.save, the commit, the artifact, the hooks.

  A keep: League.save, then train_state.json naming it (the commit), then
  the run's artifact from the committed state, then the on-commit hooks.

  Args:
    ckpt: the checkpoint dir.
    cfg: the run config.
    lg: the League.
    run_id: the run's id.
    attempt_id: this process's attempt id.
    attempts: the run's attempts so far.
    wall_budget: None, or the wall budget record.
  """
  step = lg.iteration
  previous = run_state.read_train_state(ckpt)
  r = lg.save(ckpt, step)
  name = cfg.artifact_learner
  ls = lg.learners.get(name) if name else None
  art_step = None if ls is None else int(lg.driver.artifact_step(ls))
  rec = dict(
      _run_record(cfg, lg, run_id, attempt_id, attempts),
      kind=KIND,
      artifact_step=art_step,
      config=common.config_dict(cfg),
  )
  if ls is not None and lg.driver.publishes_siblings:
    rec.update(artifact_modes=list(siblings_lib.MODES), plain_step=art_step)
  if wall_budget is not None:
    # the run ended at its wall budget
    rec.update(wall_budget=dict(wall_budget))
  committed = commit.with_commit(rec, previous, step, r)
  run_state.write_train_state(ckpt, committed)
  logging.info("league run: committed iteration %d (%s)", step, r)
  # The keep supersedes any resume checkpoint: record first, then its dir.
  resume.drop(ckpt, committed["league_commits"].values())
  got = _publish_run_artifact(
      ckpt,
      cfg,
      lg,
      run_id,
      None if previous is None else previous.get("artifact_step"),
  )
  assert got == art_step, (got, art_step)
  for hook in dict.fromkeys((*_ALWAYS_ON_COMMIT, *cfg.on_commit)):
    _ON_COMMIT[hook](ckpt, step, r)
  logging.info(
      "league run: checkpoint at iteration %d (artifact step %s) -> %s",
      step,
      art_step,
      ckpt,
  )


def _run_record(
    cfg: LeagueRunConfig,
    lg: loop.League,
    run_id: Optional[str | int],
    attempt_id: Optional[int],
    attempts: tuple[Any, ...],
) -> dict[str, Any]:
  """The fields a keep's train_state.json and a resume record share.

  Args:
    cfg: The run config.
    lg: The league.
    run_id: The run id.
    attempt_id: The attempt id.
    attempts: The attempts so far.

  Returns:
    The shared fields.
  """
  every = list(lg.learners.values()) + list(lg.retired.values())
  return {
      "learner_kind": cfg.learner.kind,
      "step": int(lg.iteration),
      "updates": int(lg.updates_total),
      "frames": sum(int(l.frames) for l in every),
      "decisions": sum(int(l.decisions) for l in every),
      "stopped": bool(lg.stopped),
      "stop_reason": lg.stop_reason,
      "members": sorted(lg.members),
      _RUN_KEY: run_id,
      _ATTEMPT_KEY: attempt_id,
      "attempts": list(attempts),
      "resume_hash": common.resume_hash(cfg),
      "config_hash": common.config_hash(cfg),
  }


def _save_resume(
    ckpt: str,
    cfg: LeagueRunConfig,
    lg: loop.League,
    run_id: Optional[str | int],
    attempt_id: Optional[int],
    attempts: tuple[Any, ...],
    reason: str,
) -> str:
  """A resume checkpoint at lg.iteration.

  A resume checkpoint at lg.iteration: League.save without releasing
  retired states, the record over the previous one, then the previous one's
  dir removed. Returns its ref.

  Args:
    ckpt: the checkpoint dir.
    cfg: the run config.
    lg: the League.
    run_id: the run's id.
    attempt_id: this process's attempt id.
    attempts: the run's attempts so far.
    reason: why the checkpoint is saved.

  Returns:
    The resume checkpoint's ref.
  """
  ts = run_state.read_train_state(ckpt)
  old = resume.read(ckpt)
  r = lg.save(ckpt, lg.iteration, release_retired=False)
  rec = dict(
      _run_record(cfg, lg, run_id, attempt_id, attempts),
      kind=resume.KIND,
      league_ref=r,
      keep_step=None if ts is None else int(ts["step"]),
      reason=reason,
      attempt=lg.attempt,
      written_at=resume.utc_now(),
  )
  resume.write(ckpt, rec)
  if old is not None and old["league_ref"] != r:
    resume.prune(
        ckpt,
        old["league_ref"],
        ((ts or {}).get("league_commits") or {}).values(),
    )
  logging.info(
      "league run: resume checkpoint (%s) at iteration %d -> %s/%s",
      reason,
      lg.iteration,
      ckpt,
      r,
  )
  return r


def _final_evals(
    lg: loop.League,
    cfg: LeagueRunConfig,
    evaluate,
    total: int,
    exact_every: int,
) -> None:
  """The artifact learner's evals not yet run at its update.

  The artifact learner's evals not yet run at its current update (the
  wall budget ends the run here: its last update gets every eval).

  Args:
    lg: the League.
    cfg: the run config.
    evaluate: runs the evals of one update.
    total: the learner's last update, where every eval is due.
    exact_every: updates between exact evals.
  """
  name = cfg.artifact_learner
  ls = lg.learners.get(name) if name else None
  if ls is None:
    return
  u = int(ls.updates)
  evaluate(
      lg,
      ls,
      u,
      lambda ev: not (u % ev.every == 0 or u == total),
      exact_due=not (u % exact_every == 0 or u == total),
  )


def run(
    cfg: LeagueRunConfig,
    checkpoint_dir: Optional[str] = None,
    *,
    tensorboard: bool = False,
    run_id: Optional[str | int] = None,
    git_commit: Optional[str] = None,
    mesh=None,
    data_dir: Optional[str] = None,
    stop_after: Optional[int] = None,
    heldout: bool = True,
    scratch_dir: Optional[str] = None,
    attempt_id: Optional[int] = None,
    resume_every_s: Optional[float] = None,
    preempted: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = time.monotonic,
    sinks: Sequence[Any] = (),
    wall_budget_s: Optional[float] = None,
    eval_grace_s: float = 0.0,
) -> LeagueRunResult:
  """Trains cfg to num_iterations.

  Resumes from checkpoint_dir if it holds a train_state.json (module doc).

  Args:
    cfg: the run config.
    checkpoint_dir: where keeps, resume checkpoints, metrics and members go.
    tensorboard: also write TensorBoard event files.
    run_id: the run's id if the launcher declares one (`run_identity` settles
      it; the run's params record it).
    git_commit: recorded in artifacts.
    mesh: the device mesh.
    data_dir: BC from recorded shards.
    stop_after: end this process after that iteration (checkpointed), with the
      run length and every schedule horizon unchanged; resuming continues.
    heldout: compute BC's held-out metrics.
    scratch_dir: a local dir for kinds that need one.
    attempt_id: this process's attempt id (`run_identity`; the run's params
      record it).
    resume_every_s: take a resume checkpoint once that many seconds of clock()
      passed since the last checkpoint (None: no timer).
    preempted: the preemption notice, read after every iteration and before
      every eval.
    clock: the timer's clock.
    sinks: extra metric sinks (objects or registered names, metrics) beside the
      run's metrics.jsonl.
    wall_budget_s: end the run at the first iteration boundary this many seconds
      of clock() after the call started (module doc, "Wall budget").
    eval_grace_s: with wall_budget_s, an eval still running this long past the
      budget raises core.deadline.EvalOverBudgetError.

  Returns:
    the LeagueRunResult of this call.

  Raises:
    ValueError: an invalid timer, budget or grace setting.
  """
  x64.enable_x64()  # an entry point (package doc)
  assert isinstance(cfg, LeagueRunConfig), type(cfg)
  # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
  # pylint: disable-next=unnecessary-negation
  if resume_every_s is not None and not float(resume_every_s) >= 0.0:
    raise ValueError(
        f"resume_every_s must be >= 0 (or None), got {resume_every_s}"
    )
  notice = preempted if preempted is not None else (lambda: False)
  t_start = clock()
  # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
  # pylint: disable-next=unnecessary-negation
  if wall_budget_s is not None and not float(wall_budget_s) > 0.0:
    raise ValueError(
        f"wall_budget_s must be > 0 (or None), got {wall_budget_s}"
    )
  # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
  # pylint: disable-next=unnecessary-negation
  if not float(eval_grace_s) >= 0.0:
    raise ValueError(f"eval_grace_s must be >= 0, got {eval_grace_s}")
  end = cfg.num_iterations
  if stop_after is not None:
    if int(stop_after) < 1:
      raise ValueError(f"stop_after must be >= 1, got {stop_after}")
    end = min(int(stop_after), cfg.num_iterations)
  run_dir = checkpoint_dir or tempfile.mkdtemp(prefix="league-")
  if not checkpoint_dir:
    logging.info("league run: no checkpoint_dir; members go to %s", run_dir)
  ts = run_state.read_train_state(checkpoint_dir) if checkpoint_dir else None
  if checkpoint_dir and ts is None and artifact.exists(checkpoint_dir):
    raise ValueError(
        f"{checkpoint_dir} holds an artifact but no {run_state.TRAIN_STATE}: "
        "not "
        "a resumable league run (use a fresh checkpoint_dir)"
    )
  rs = resume.read(checkpoint_dir) if checkpoint_dir else None
  from_rs = resume.choose(ts, rs, common.resume_hash(cfg), cfg.num_iterations)
  # stale: a keep at or past it whose drop never ran
  if rs is not None and from_rs is None:
    resume.drop(
        checkpoint_dir, ((ts or {}).get("league_commits") or {}).values()
    )
  run_id, attempt_id, attempts = run_identity(from_rs or ts, run_id, attempt_id)
  total = cfg.run_updates()
  rows, upd_rows = [], []
  last_heldout: list[Any] = [None]
  state = dict(writer=None, evaluators=[], exact=None, clock={})

  exact_every = exact_eval_every(total)

  def evaluate(
      lg: loop.League, ls, u: int, due, exact_due: bool = False
  ) -> None:
    if notice():
      logging.warning(
          "league run: preemption notice up: the evals due at update %d are "
          "skipped (the resume checkpoint goes first)",
          u,
      )
      return
    row: dict[str, Any] = {}
    for ev, evaluator in state["evaluators"]:
      if due(ev):
        row.update(
            _eval_row(
                ev, evaluator, lg, ls, u, series_step=int(lg.updates_total)
            )
        )
    if exact_due and state["exact"] is not None:
      row.update(_exact_row(state["exact"], lg, ls))
    if row:
      state["writer"].write(int(lg.updates_total), row, prefix="eval")

  def before(lg: loop.League, ls) -> None:
    if ls.name == cfg.artifact_learner and int(ls.updates) == 0:
      evaluate(lg, ls, 0, lambda ev: True, exact_due=True)
    # Throughput clock: this learner's rollout + update, evals excluded.
    state["clock"][ls.name] = (
        time.perf_counter(),
        int(ls.frames),
        int(ls.decisions),
    )

  def on_update(lg: loop.League, ls, m: dict[str, Any]) -> None:
    step = int(lg.updates_total)
    u = int(ls.updates)
    row = dict(m)
    # Throughput and wall-clock timings (a driver's "seconds" / "*_seconds"
    # metrics) go to the metrics stream only: the result rows stay
    # deterministic.
    timing: dict[str, float] = {
        k: row.pop(k) for k in list(row) if is_timing_key(k)
    }
    clock_start = state["clock"].pop(ls.name, None)
    if clock_start is not None:
      t0, f0, d0 = clock_start
      dt = time.perf_counter() - t0
      timing["update_seconds"] = dt
      if dt > 0.0:
        timing["frames_per_s"] = (int(ls.frames) - f0) / dt
        timing["decisions_per_s"] = (int(ls.decisions) - d0) / dt
    main = ls.name == cfg.artifact_learner
    if main and heldout and isinstance(cfg.learner, kinds.BC):
      every = cfg.learner.heldout_every
      if (every and u % every == 0) or u == total:
        last_heldout[0] = lg.driver.heldout_metrics(ls)
        row.update(last_heldout[0])
    held = getattr(lg.learner, "held_metrics", lambda: ())()
    state["writer"].write(
        step,
        {f"{ls.name}/{k}": v for k, v in {**row, **timing}.items()},
        prefix="train",
        held=[f"{ls.name}/{k}" for k in held],
    )
    if main:
      upd_rows.append(row)
      evaluate(
          lg,
          ls,
          u,
          lambda ev: u % ev.every == 0 or u == total,
          exact_due=u % exact_every == 0 or u == total,
      )

  lg = build(
      cfg,
      run_dir,
      mesh=mesh,
      git_commit=git_commit,
      data_dir=data_dir,
      scratch_dir=scratch_dir,
      on_update=on_update,
  )
  lg.before_update = before
  start = 0
  if from_rs is not None:
    start = int(from_rs["step"])
    lg.restore(checkpoint_dir, start, from_rs["league_ref"])
    logging.info(
        "league run: resuming %s from its resume checkpoint (%s, %s) at "
        "iteration %d "
        "(update %d; keep %s)",
        cfg.name,
        from_rs["reason"],
        from_rs["league_ref"],
        start,
        int(lg.updates_total),
        from_rs["keep_step"],
    )
  elif ts is not None:
    start = run_state.check_resume(
        checkpoint_dir, ts, KIND, common.resume_hash(cfg), cfg.num_iterations
    )
    lg.restore(checkpoint_dir, start)
    # A process that died between its commit and its publish: publish the
    # committed artifact now (a no-op when it is there).
    if ts.get("artifact_step") is not None:
      got = _publish_run_artifact(
          checkpoint_dir, cfg, lg, run_id, ts["artifact_step"]
      )
      if got != int(ts["artifact_step"]):
        raise ValueError(
            f"{checkpoint_dir}: the restored artifact step {got} != the "
            "committed "
            f"{ts['artifact_step']}"
        )
    logging.info(
        "league run: resuming %s at iteration %d (update %d)",
        cfg.name,
        start,
        int(lg.updates_total),
    )
  state["evaluators"] = build_evals(cfg, lg.env, mesh, run_dir)
  state["exact"] = build_exact_eval(cfg, lg.env, lg.slot_policy)
  state["writer"] = metrics_lib.MetricsWriter(
      os.path.join(checkpoint_dir, run_state.METRICS_DIR)
      if checkpoint_dir
      else None,
      tensorboard=tensorboard,
      resume_step=int(lg.updates_total) if start else None,
      sinks=sinks,
  )
  writer = state["writer"]
  if run_id is not None or attempt_id is not None:
    # The run's identity is a setting of this attempt, not a series.
    ids = records.encode({
        k: v
        for k, v in ((_RUN_KEY, run_id), (_ATTEMPT_KEY, attempt_id))
        if v is not None
    })
    writer.params({
        **{
            f"run/{k}": np.int64(v) if isinstance(v, int) else str(v)
            for k, v in ids.items()
        },
        "run/attempt": np.int64(len(attempts)),
        "run/attempt_start_update": np.int64(lg.updates_total),
    })
  logging.info(
      "league run: %s (run %s, attempt %s) (%s, controller %s) on %s, "
      "iterations %d..%d of %d",
      cfg.name,
      run_id,
      attempt_id,
      cfg.learner.kind,
      cfg.controller,
      cfg.env,
      start,
      end,
      cfg.num_iterations,
  )
  last_save = clock()
  stopped_by_notice = False
  wall = None
  eval_at = (
      None
      if wall_budget_s is None
      else t_start + float(wall_budget_s) + float(eval_grace_s)
  )
  try:
    with deadline_lib.deadline(
        eval_at,
        f"wall budget {wall_budget_s} s + grace {eval_grace_s} s",
        clock=clock,
    ):
      while lg.iteration < end and not lg.stopped:
        lg.run(1)
        row = _summary(lg, lg.history[-1])
        writer.write(
            int(lg.updates_total), row, prefix="league", held=league_held(row)
        )
        rows.append(row)
        done = lg.iteration
        now = clock()
        up = bool(notice())
        if (
            wall_budget_s is not None
            and not up
            and done < end
            and not lg.stopped
            and now - t_start >= float(wall_budget_s)
        ):
          wall = {
              "budget_s": float(wall_budget_s),
              "reached_at": int(done),
              "of": int(end),
          }
          _final_evals(lg, cfg, evaluate, total, exact_every)
        if checkpoint_dir and (
            done % cfg.keep_every == 0
            or lg.stopped
            or done == end
            or wall is not None
        ):
          writer.flush()
          _commit(
              checkpoint_dir,
              cfg,
              lg,
              run_id,
              attempt_id,
              tuple(attempts),
              wall_budget=wall,
          )
          last_save = now
        elif checkpoint_dir and (
            up
            or (
                resume_every_s is not None
                and now - last_save >= float(resume_every_s)
            )
        ):
          writer.flush()
          _save_resume(
              checkpoint_dir,
              cfg,
              lg,
              run_id,
              attempt_id,
              tuple(attempts),
              "preemption" if up else "timer",
          )
          last_save = now
        if wall is not None:
          logging.warning(
              "league run: wall budget reached at %d of %d (%.0f s of %.0f s): "
              "final evals run, %s",
              done,
              end,
              now - t_start,
              float(wall_budget_s),
              "checkpointed"
              if checkpoint_dir
              else "no checkpoint_dir: nothing saved",
          )
          break
        if up:
          logging.warning(
              "league run: preemption notice up: ending after iteration %d of "
              "%d "
              "(%s)",
              done,
              end,
              "checkpointed"
              if checkpoint_dir
              else "no checkpoint_dir: nothing saved",
          )
          # else this segment is complete
          stopped_by_notice = done < end and not lg.stopped
          break
  finally:
    writer.close()
  return LeagueRunResult(
      cfg, lg, upd_rows, rows, last_heldout[0], stopped_by_notice, wall
  )


# -- probes --------------------------------------------------------------------
class OneSlotParts(NamedTuple):
  """The pieces of a single-learner league's update, from the League itself.

  For benchmarks and probes that time or shard one rollout and one update.

  Attributes:
    env: the env.
    policy: the learner's policy.
    learner: the Learner.
    arena: the driver's Arena / HostArena.
    table: the learner's first table.
    slot_params: learner params -> the Arena params list (opponents
      included).
    num_steps: plies per update.
    league: the League.
  """

  env: types.Env
  policy: types.Policy  # the learner's
  learner: Any
  arena: Any  # the driver's Arena / HostArena
  table: Any  # the learner's first table
  slot_params: Callable[..., Any]
  num_steps: int
  league: loop.League


def one_slot_parts(cfg: LeagueRunConfig, mesh=None) -> OneSlotParts:
  """The OneSlotParts of a single-learner run config.

  Args:
    cfg: the run config.
    mesh: the device mesh.

  Returns:
    The OneSlotParts.

  Raises:
    ValueError: the controller is not single-learner.
  """
  if single_learner_key(cfg.controller) is None:
    raise ValueError(
        f"{cfg.name}: one_slot_parts needs a single-learner controller"
    )
  lg = build(cfg, tempfile.mkdtemp(prefix="league-parts-"), mesh=mesh)
  lg.tick(0)
  ls = lg.learners[cfg.artifact_learner]
  # Uses private _draw of a module in this package.
  table, opp, _, _ = lg._draw(ls, lg.updates_total)  # pylint: disable=protected-access

  def stack(p):
    """Adds a leading member axis of 1.

    Args:
      p: The params.

    Returns:
      The stacked params.
    """
    return jax.tree_util.tree_map(lambda x: jax.numpy.asarray(x)[None], p)

  def slot_params(p):
    """The slot params: the learner's, then the opponents'.

    Args:
      p: The learner's params.

    Returns:
      The slot params.
    """
    return [stack(lg.driver.wrap(p)), *opp]

  return OneSlotParts(
      lg.env,
      lg.policy,
      lg.learner,
      lg.arena,
      table,
      slot_params,
      cfg.league.num_steps,
      lg,
  )


def bc_heldout_set(cfg: LeagueRunConfig):
  """A BC config's fixed held-out state set (kinds.BCDriver.heldout_set).

  Args:
    cfg: a BC run config.

  Returns:
    The HeldoutSet.

  Raises:
    ValueError: cfg is not a BC config.
  """
  if not isinstance(cfg.learner, kinds.BC):
    raise ValueError(f"{cfg.name} is a {cfg.learner.kind} config, not a BC one")
  lg = build(cfg, tempfile.mkdtemp(prefix="league-heldout-"))
  lg.tick(0)
  return lg.driver.heldout_set(lg.learners[cfg.artifact_learner])


def record(cfg: LeagueRunConfig, out_dir: str, num_batches: int) -> str:
  """BC: writes the teacher's games of updates 0..num_batches-1 to shards.

  kinds.BCDriver.record over the run's first table and update keys, so a
  data="shards" run over out_dir replays the online run's batches.

  Args:
    cfg: a BC run config.
    out_dir: the shards' directory.
    num_batches: the number of updates recorded.

  Returns:
    the teacher's id, which the shards record as their producer.

  Raises:
    ValueError: cfg is not a BC config.
  """
  if not isinstance(cfg.learner, kinds.BC):
    raise ValueError(f"record is a BC option (got a {cfg.learner.kind} config)")
  lg = build(cfg, tempfile.mkdtemp(prefix="league-record-"))
  lg.tick(0)
  ls = lg.learners[cfg.artifact_learner]
  # Uses private _draw of a module in this package.
  table, opp, _, _ = lg._draw(ls, lg.updates_total)  # pylint: disable=protected-access

  def key_of(g):
    """The loop key of update g.

    Args:
      g: The global update index.

    Returns:
      The key.
    """
    # Uses private _k_loop of a module in this package.
    return x64.fold_in64(lg._k_loop, np.int64(g))  # pylint: disable=protected-access

  return lg.driver.record(ls, table, opp, out_dir, num_batches, key_of)


# -- recipes -------------------------------------------------------------------
def one_slot(
    name: str,
    env: str,
    learner: Any,
    *,
    num_envs: int,
    num_steps: int,
    num_updates: int,
    policy_kwargs: Optional[Mapping[str, Any]] = None,
    learner_rule: types.DecisionRule = types.SAMPLE,
    self_play: bool = True,
    opponents: tuple[Any, ...] = (),
    weights: Optional[Mapping[str, float]] = None,
    keep_every: int = 1,
    seed: int = 0,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    # `eval` is the keyword the callers pass.
    eval: Any = None,  # pylint: disable=redefined-builtin
    metadata: Optional[Mapping[str, str]] = None,
    notes: str = "",
) -> LeagueRunConfig:
  """A plain run: learner "main" under controller fixed_mix.

  One update per iteration (iterations are updates), no population, no
  payoffs.

  Args:
    name: the run name.
    env: a registered env name.
    learner: the learner kind spec.
    num_envs: Arena rows per update.
    num_steps: plies per update.
    num_updates: the run length in updates.
    policy_kwargs: the network's arch_kwargs.
    learner_rule: the learner's decision rule.
    self_play: every seat is the learner and every seat a perspective (B * P
      rows); else the learner sits at seat row % P against `opponents`
      (refs.FixedOpponent) mixed by `weights` (default uniform), with the
      table drawn once.
    opponents: the fixed opponents.
    weights: the opponents' mix.
    keep_every: iterations between keeps.
    seed: the run's seed.
    env_kwargs: env options.
    eval: an InLoopEval, a tuple of them, or None.
    metadata: free-form string labels recorded with the run.
    notes: free text recorded with the run.

  Returns:
    The LeagueRunConfig.
  """
  if self_play:
    assert not opponents and not weights, "self-play has no opponents"
    ctl = dict(learner="main", self_play=1.0, weights={})
  else:
    assert opponents, "a non-self-play run needs opponents"
    w = dict(weights or {o.name: 1.0 / len(opponents) for o in opponents})
    assert set(w) == {o.name for o in opponents}, (
        sorted(w),
        [o.name for o in opponents],
    )
    ctl = dict(learner="main", self_play=0.0, weights=w)
  league = loop.LeagueConfig(
      num_envs=int(num_envs),
      num_steps=int(num_steps),
      updates_per_iteration=1,
      pool_capacity=0,
      learner_rule=learner_rule,
      payoff="none",
      seed=int(seed),
      redraw="on_change",
      self_play_seats="all" if self_play else "one",
      algo=learner.kind,
  )
  return LeagueRunConfig(
      name=name,
      env=env,
      learner=learner,
      league=league,
      controller="fixed_mix",
      controller_kwargs=ctl,
      num_iterations=int(num_updates),
      policy_kwargs=dict(policy_kwargs or {}),
      fixed_opponents=tuple(opponents),
      num_initial=0,
      keep_every=int(keep_every),
      seed=int(seed),
      env_kwargs=dict(env_kwargs or {}),
      artifact_learner="main",
      eval=eval,
      metadata=dict(metadata or {}),
      notes=notes,
  )


def snapshot_pool(
    name: str,
    env: str,
    learner: kinds.PG,
    policy_kwargs: Mapping[str, Any],
    *,
    num_envs: int,
    num_steps: int,
    num_updates: int,
    snapshot_every: int,
    harvest_every: int,
    max_pool: int,
    self_weight: float,
    pool_slots: int = 1,
    opponent_rule: types.DecisionRule = types.ARGMAX,
    learner_rule: types.DecisionRule = types.SAMPLE,
    keep_every: int = 200,
    seed: int = 0,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    # `eval` is the keyword the callers pass.
    eval: Any = None,  # pylint: disable=redefined-builtin
    metadata: Optional[Mapping[str, str]] = None,
    notes: str = "",
) -> LeagueRunConfig:
  """Self-play against a lagged snapshot plus a pool of harvested snapshots.

  One main learner (one update per iteration) under controller
  scheduled_promotion: a lagged snapshot promoted every snapshot_every
  updates, a pool of snapshots harvested every harvest_every (newest
  max_pool), pool_slots of them seated per round, self_weight of the rows on
  the snapshot; opponents play opponent_rule. The schedule values have no
  defaults. The pool is 1 + pool_slots single-member Arena slots
  (pool_layout "slots": no per-row gather of a large network); the rest of
  the harvest waits off the Arena (bench). The table is drawn per round and
  the round's hands continue across its updates.

  Args:
    name: the run name.
    env: a registered env name.
    learner: the PG spec.
    policy_kwargs: the network's arch_kwargs.
    num_envs: Arena rows per update.
    num_steps: plies per update.
    num_updates: the run length in updates.
    snapshot_every: updates between lagged-snapshot promotions.
    harvest_every: updates between pool harvests.
    max_pool: the pool's size (the newest harvests).
    self_weight: the rows' share on the lagged snapshot.
    pool_slots: pool members seated per round.
    opponent_rule: the opponents' decision rule.
    learner_rule: the learner's decision rule.
    keep_every: iterations between keeps.
    seed: the run's seed.
    env_kwargs: env options.
    eval: an InLoopEval, a tuple of them, or None.
    metadata: free-form string labels recorded with the run.
    notes: free text recorded with the run.

  Returns:
    The LeagueRunConfig.
  """
  league = loop.LeagueConfig(
      num_envs=int(num_envs),
      num_steps=int(num_steps),
      updates_per_iteration=1,
      pool_capacity=1 + int(pool_slots),
      learner_rule=learner_rule,
      pool_rule=opponent_rule,
      payoff="none",
      seed=int(seed),
      redraw="on_change",
      self_play_seats="one",
      pool_layout="slots",
      bench=True,
      algo="pg",
  )
  ctl = dict(
      main="main",
      snapshot_every=int(snapshot_every),
      harvest_every=int(harvest_every),
      max_pool=int(max_pool),
      pool_slots=int(pool_slots),
      self_weight=float(self_weight),
      seed=int(seed),
  )
  return LeagueRunConfig(
      name=name,
      env=env,
      learner=learner,
      league=league,
      controller="scheduled_promotion",
      controller_kwargs=ctl,
      num_iterations=int(num_updates),
      policy_kwargs=dict(policy_kwargs),
      num_initial=0,
      keep_every=int(keep_every),
      seed=int(seed),
      env_kwargs=dict(env_kwargs or {}),
      artifact_learner="main",
      eval=eval,
      metadata=dict(metadata or {}),
      notes=notes,
  )


def psro(
    name: str,
    env: str,
    learner: Any,
    policy_kwargs: Mapping[str, Any],
    *,
    num_envs: int,
    num_steps: int,
    psro_iterations: int,
    br_updates: int,
    meta_solver: str,
    eval_deals: int,
    warm_start: bool = False,
    pipeline_depth: int = 1,
    updates_per_iteration: Optional[int] = None,
    min_br_updates: Optional[int] = None,
    plateau_patience: Optional[int] = None,
    plateau_threshold: float = 0.01,
    pipeline_warm_start: bool = False,
    active_weight_share: float = 0.5,
    reset_optimizer_each_iteration: bool = False,
    seed: int = 0,
    metadata: Optional[Mapping[str, str]] = None,
    notes: str = "",
) -> LeagueRunConfig:
  """PSRO, or pipeline PSRO with pipeline_depth > 1, as a league run.

  The population starts from one random-init member. Best responses are the
  `learner` spec's PG learner (a constant step size), br_updates updates
  each; payoffs are eval_deals duplicate deal pairs per pair of entrants.

  Batch PSRO (pipeline_depth 1): one league iteration trains a BR for its
  whole budget, so num_iterations = psro_iterations + 1 (the last iteration
  admits the last BR and stops). Pipeline PSRO: an iteration is
  updates_per_iteration updates of every worker; the lowest worker is
  promoted at the BR budget, or on a plateau after min_br_updates updates.

  Args:
    name: the run name.
    env: a registered env name.
    learner: the PG spec of the best responses.
    policy_kwargs: the network's arch_kwargs.
    num_envs: Arena rows per update.
    num_steps: plies per update.
    psro_iterations: the number of best responses.
    br_updates: updates per best response.
    meta_solver: the meta-solver name (league.meta_solvers).
    eval_deals: duplicate deal pairs per payoff entry.
    warm_start: a new BR starts from the member just admitted.
    pipeline_depth: the number of concurrent BR levels.
    updates_per_iteration: pipeline PSRO's updates per iteration.
    min_br_updates: the fewest updates before a plateau promotion.
    plateau_patience: iterations a score must stay flat to plateau.
    plateau_threshold: the score change that counts as flat.
    pipeline_warm_start: a new top-level BR starts from the highest live BR.
    active_weight_share: the lower levels' share of a pipeline BR's mix.
    reset_optimizer_each_iteration: a fresh optimizer every iteration.
    seed: the run's seed.
    metadata: free-form string labels recorded with the run.
    notes: free text recorded with the run.

  Returns:
    The LeagueRunConfig.

  Raises:
    ValueError: pipeline PSRO without updates_per_iteration, min_br_updates
      and plateau_patience.
  """
  kw: dict[str, Any] = dict(
      meta_solver=meta_solver,
      iterations=int(psro_iterations),
      br_updates=int(br_updates),
      warm_start=bool(warm_start),
  )
  if pipeline_depth == 1:
    per_iteration, iterations = int(br_updates), int(psro_iterations) + 1
  else:
    if (
        updates_per_iteration is None
        or plateau_patience is None
        or min_br_updates is None
    ):
      raise ValueError(
          "pipeline PSRO needs updates_per_iteration, min_br_updates and "
          "plateau_patience"
      )
    per_iteration = int(updates_per_iteration)
    kw.update(
        pipeline_depth=int(pipeline_depth),
        pipeline_warm_start=bool(pipeline_warm_start),
        active_weight_share=float(active_weight_share),
        min_br_updates=int(min_br_updates),
        plateau_patience=int(plateau_patience),
        plateau_threshold=float(plateau_threshold),
    )
    ticks_per_br = -(-int(br_updates) // per_iteration)
    iterations = int(psro_iterations) * ticks_per_br + int(pipeline_depth) + 1
  league = loop.LeagueConfig(
      num_envs=int(num_envs),
      num_steps=int(num_steps),
      updates_per_iteration=per_iteration,
      reset_optimizer_each_iteration=bool(reset_optimizer_each_iteration),
      pool_capacity=1 + int(psro_iterations) + int(pipeline_depth),
      learner_rule=types.SAMPLE,
      pool_rule=types.SAMPLE,
      eval_deals=int(eval_deals),
      payoff="evaluator",
      seed=int(seed),
  )
  return LeagueRunConfig(
      name=name,
      env=env,
      policy_kwargs=dict(policy_kwargs),
      learner=learner,
      league=league,
      controller="psro",
      controller_kwargs=kw,
      num_iterations=iterations,
      num_initial=1,
      keep_every=1,
      seed=int(seed),
      artifact_learner="",
      metadata=dict(metadata or {}),
      notes=notes,
  )
