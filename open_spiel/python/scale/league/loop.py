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

"""The League: one loop that trains learners against a managed population.

Every training run is a league. A plain self-play run is a league with one
learner slot and a fixed opponent mix; PSRO, snapshot pools, exploiters and
PBT are the same loop with another controller. The pieces are pluggable:

  driver      the learner kind (`league.kinds`): the Learner, the Arena its
              slots play in and one update per call
  sampler     each learner's opponent mix -> an Arena slot table (`sampler`)
  pool        admitted members at K fixed positions (`pool_params`), so the
              rollout and the update compile once
  evaluator   the payoff matrix over members and learners (`eval.match`), exact
              payoffs on small games, or none
  store       member artifacts and their ids (`artifact_store`)
  controller  reads the view and returns actions (`controller`)

    lg = League(env, policy, learner,
                controller.FixedMix("main", S.Mix(1.0, {})), store, run_dir,
                LeagueConfig(num_envs=64, num_steps=32,
                             updates_per_iteration=1, pool_capacity=0))
    lg.run(100)
    ref = lg.save(ckpt_dir, lg.iteration)     # league.runner commits the ref

One iteration: evaluate, view, controller.step, apply (repeated while the
actions add entrants, up to max_rounds), then every live learner trains
updates_per_iteration updates against the others' start-of-iteration params.
Keys follow one schedule from key(seed): update g uses fold_in64(k_loop, g),
a table drawn at update g uses fold_in64(k_draw, g), and a payoff cell uses
fold_in(k_eval, crc32 of its entrants and versions), so a resumed league
replays the iterations of an uninterrupted one. Counters are int64.

Residency: only the Arena's positions hold member weights on the device. A
benched member (bench=True, no free position) keeps its weights in host
memory; an evicted member keeps only its artifact, loaded on demand.

Payoffs, meta-solvers and PSRO are two-player zero-sum; the sampler and the
Arena are P-player.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import pickle
from typing import Any, Callable, Mapping, Optional, Sequence
import zlib

from absl import logging
import jax
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import exploitability as expl
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import stats
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller as controller_lib
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import mixture
from open_spiel.python.scale.league import pool_params
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.policies import members as members_lib

PAYOFFS = ("evaluator", "exact", "none")
REDRAWS = ("update", "on_change")
SELF_PLAY_SEATS = ("one", "all")
POOL_LAYOUTS = ("stacked", "slots")
#: The newest League.history records and LearnerSlot.draw_counts entries kept
#: (and saved): a checkpoint's size does not grow with the length of the run.
#: The metrics stream has every iteration and update.
RECORD_CAP = 1024

#: The schema version of the state dict League.save writes ("state_version").
STATE_VERSION = 1
_STATE_TRANSLATIONS: list[Callable[[dict[str, Any]], dict[str, Any]]] = []


def register_state_translation(
    fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
  """Registers a migration of saved league states between schema versions.

  Every reader of a saved state (`read_state`: League.restore and the
  population readers) applies fn(state) -> state to it, in registration
  order, and then requires state_version == STATE_VERSION. A migration brings
  a state of another state_version to this one and sets its state_version; a
  state no migration brings to STATE_VERSION is refused. With no migration
  registered, a state of another version is refused with a ValueError that
  names both versions.

  Raises:
    TypeError: fn is not callable.

  Args:
    fn: state -> state.
  """
  if not callable(fn):
    raise TypeError(f"state translation {fn!r} is not callable")
  if fn not in _STATE_TRANSLATIONS:
    _STATE_TRANSLATIONS.append(fn)


def read_state(path: Any) -> dict[str, Any]:
  """The league state saved at `path`, after every registered translation.

  Args:
    path: an epath.Path to a state.pkl.

  Returns:
    The state dict (migrate_state).
  """
  # The run's own state file, written by this league's commit; not untrusted
  # input.
  # pylint: disable-next=g-unsafe-pickle-load
  return migrate_state(pickle.loads(path.read_bytes()), path)  # noqa: S301


def migrate_state(blob: dict[str, Any], where: Any) -> dict[str, Any]:
  """A saved state dict after every registered migration, at STATE_VERSION.

  Args:
    blob: the saved state dict.
    where: names the state in errors.

  Returns:
    The migrated state.

  Raises:
    ValueError: the state is not at STATE_VERSION after the migrations.
  """
  for fn in _STATE_TRANSLATIONS:
    blob = fn(blob)
  got = blob.get("state_version")
  if got != STATE_VERSION:
    raise ValueError(
        f"{where}: league state version {got!r}, this library reads "
        f"{STATE_VERSION} (register a state translation to migrate it)"
    )
  return blob


@dataclasses.dataclass(frozen=True)
class LeagueConfig:
  """The shape and policies of a League.

  Attributes:
    num_envs: Arena rows per update.
    num_steps: plies per update.
    updates_per_iteration: updates every live learner trains per iteration.
    pool_capacity: K, the population pool's positions (0: no pool slot).
    learner_rule: the learners' decision rule.
    pool_rule: the decision rule of pool entrants (members and live learners
      as opponents).
    eval_deals: duplicate deal pairs per payoff cell.
    eval_max_rows: the Evaluator's row cap.
    payoff: "evaluator" (sampled payoffs), "exact" (GameTree tables) or "none".
    max_rounds: evaluate / step / apply rounds allowed per iteration.
    stratified: exact per-seat row allocation (else iid per row).
    seed: the league's root seed.
    algo: the `algo` recorded in member artifacts.
    redraw: "update" draws a fresh table and deals fresh hands every update;
      "on_change" redraws only when the learner's mix or the pool's seating
      changes, and otherwise continues the same hands across updates.
    self_play_seats: "all" emits every seat of a pure self-play row (B * P
      rows); "one" emits the perspective seat.
    pool_layout: "stacked" is one Arena slot of K stacked members (one params
      gather per row); "slots" is K slots of one member each (no gather, for
      large networks).
    bench: members beyond the pool's positions wait off the Arena, host-side,
      and are seated when a mix names them.
    reset_optimizer_each_iteration: give every live learner a fresh optimizer
      state at the start of each training phase (params, value EMA and
      counters carry on), so each best-response phase starts its adaptive
      moments from scratch.
  """

  num_envs: int
  num_steps: int
  updates_per_iteration: int
  pool_capacity: int
  learner_rule: types.DecisionRule = types.SAMPLE
  pool_rule: types.DecisionRule = types.SAMPLE
  eval_deals: int = 1024
  eval_max_rows: int = 4096
  payoff: str = "evaluator"
  max_rounds: int = 4
  stratified: bool = True
  seed: int = 0
  algo: str = "league"
  redraw: str = "update"
  self_play_seats: str = "one"
  pool_layout: str = "stacked"
  bench: bool = False
  reset_optimizer_each_iteration: bool = False

  def __post_init__(self):
    for name, low in (
        ("num_envs", 1),
        ("num_steps", 1),
        ("updates_per_iteration", 1),
        ("pool_capacity", 0),
        ("eval_deals", 2),
        ("max_rounds", 1),
    ):
      if getattr(self, name) < low:
        raise ValueError(
            f"LeagueConfig.{name} must be >= {low}, got {getattr(self, name)}"
        )
    for name, allowed in (
        ("payoff", PAYOFFS),
        ("redraw", REDRAWS),
        ("self_play_seats", SELF_PLAY_SEATS),
        ("pool_layout", POOL_LAYOUTS),
    ):
      if getattr(self, name) not in allowed:
        raise ValueError(
            f"LeagueConfig.{name} must be one of {allowed}, got "
            f"{getattr(self, name)!r}"
        )
    for name in ("learner_rule", "pool_rule"):
      if not isinstance(getattr(self, name), types.DecisionRule):
        raise TypeError(
            f"LeagueConfig.{name} must be a DecisionRule, got "
            f"{getattr(self, name)!r}"
        )


@dataclasses.dataclass
class LearnerSlot:
  """One learning slot's host-side record.

  Attributes:
    name: the learner's name.
    role: its role.
    state: its learner state.
    spawn_index: its position in spawn order.
    started_at: the iteration it was spawned at.
    mix: its opponent Mix.
    parent: None, or the member it was spawned from.
    updates: updates trained.
    frames: frames played.
    decisions: decisions trained on.
    scores: per iteration, its expected payoff against its mix.
    draw_counts: per update, the drawn table's opponent row counts.
    last_metrics: the last update's metrics.
    draw: None, or the current draw (redraw="on_change").
    draw_sig: the signature of the mix and seating the draw was made for.
  """

  name: str
  role: str
  state: Any
  spawn_index: int
  started_at: int
  mix: sampler.Mix = sampler.Mix(1.0, {})
  parent: Optional[str] = None
  updates: np.int64 = np.int64(0)
  frames: np.int64 = np.int64(0)
  decisions: np.int64 = np.int64(0)
  scores: list[Any] = dataclasses.field(default_factory=list)
  # newest RECORD_CAP
  draw_counts: list[Any] = dataclasses.field(default_factory=list)
  last_metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
  # (SlotTable, counts) of the standing table (redraw on_change)
  draw: Optional[tuple[Any, ...]] = None
  # the (mix, pool seating) the standing table was drawn for
  draw_sig: Any = None


def _group_key(policy: types.Policy, rule: types.DecisionRule) -> str:
  """The key of policies that may share one pool group.

  Args:
    policy: The policy.
    rule: Its decision rule.

  Returns:
    The JSON key.
  """
  sp = policy.spec
  return json.dumps(
      [sp.arch_name, dict(sp.arch_kwargs), rule.kind, float(rule.temperature)],
      sort_keys=True,
      default=str,
  )


# The learner-state rewrites the league performs:
# each needs named entries in Learner.state_dict, and a learner without them
# is refused with refusal_message.
# SpawnLearner(init_from=...), CopyMutate(shrink 1, perturb 0).
REPLACE_PARAMS = "replace_params"
RESET_OPT = "reset_opt"  # reset_optimizer_each_iteration
PERTURB = "perturb"  # CopyMutate with shrink != 1 or perturb != 0
CAPABILITIES = (REPLACE_PARAMS, RESET_OPT, PERTURB)


def refusal_message(
    action: str,
    learner: str,
    learner_cls: str,
    capability: str,
    missing: Sequence[str],
    have: Sequence[str],
) -> str:
  """The refusal of a learner-state rewrite.

  It names the learner slot, its Learner class and the missing capability.

  Args:
    action: the refused action.
    learner: the learner slot's name.
    learner_cls: the Learner's class name.
    capability: one of CAPABILITIES.
    missing: the missing state_dict entries.
    have: the state_dict's entries.

  Returns:
    The message.
  """
  assert capability in CAPABILITIES, capability
  return (
      f"{action}: refused: learner {learner!r} ({learner_cls}) lacks the "
      f"{capability!r} "
      f"capability: its state_dict has no {list(missing)} (it has "
      f"{list(have)}); the "
      "league rewrites learner states only through those entries"
  )


class League:
  """A league over one Env and one population architecture (module doc).

  Every learning slot is a state of the one `learner`: slots differ in name,
  role, init, mix and state, never in kind or network. Actions that rewrite a
  learner state's params or optimizer (CopyMutate, SpawnLearner(init_from),
  the optimizer reset) are refused for a learner whose state_dict has no
  "params" / "opt_state" entry. Payoffs need a 2-player game.

  Attributes:
    env: the game.
    policy: the learner's policy.
    learner: the Learner.
    controller: the LeagueController.
    driver: the learner kind's driver.
    slot_policy: the policy a learner slot plays.
    member_policy: the policy a pool member plays.
    store: the ArtifactStore.
    run_dir: the dir member artifacts go under.
    config: the LeagueConfig.
    git_commit: recorded in member artifacts.
    on_update: None, or called after every update.
    before_update: None, or called before every update.
    draw_guards: the guards run on every drawn opponent table.
    pools: the population pool slots.
    fixed: the fixed-opponent slots.
    evaluator: the payoff Evaluator.
    attempt: this process's attempt id.
    members: member name -> MemberInfo.
    learners: live learner name -> LearnerSlot.
    retired: retired learner name -> LearnerSlot.
    updates_total: updates trained over every learner and attempt.
    payoff: None, or the last payoff matrix.
    stopped: a Stop action ended the run.
    stop_reason: the Stop's reason.
    history: the newest RECORD_CAP iteration records.
    iteration: the league iteration.
  """

  def __init__(
      self,
      env: types.Env,
      policy: types.Policy,
      learner: Any,
      controller: Any,
      store: artifact_store.ArtifactStore,
      run_dir: str,
      config: LeagueConfig,
      initial_members: Optional[Mapping[str, Any]] = None,
      initial_ids: Sequence[str] = (),
      fixed_opponents: Optional[Mapping[str, tuple[Any, ...]]] = None,
      evaluator: Optional[match.Evaluator] = None,
      mesh=None,
      git_commit: Optional[str] = None,
      driver: Any = None,
      on_update: Optional[
          Callable[["League", LearnerSlot, dict[str, Any]], None]
      ] = None,
      draw_guards: Optional[
          Sequence[Callable[[Mapping[str, int]], None]]
      ] = None,
  ):
    """Builds the league and admits its initial members.

    Args:
      env: the game.
      policy: the learner's policy.
      learner: the Learner.
      controller: the LeagueController.
      store: the ArtifactStore.
      run_dir: the dir member artifacts go under.
      config: the LeagueConfig.
      initial_members: name -> learner params, admitted at construction.
      initial_ids: artifact ids in the store, admitted at construction.
      fixed_opponents: name -> (policy, params, rule), opponents outside the
        population; those with the same spec and rule share one Arena slot.
      evaluator: the payoff Evaluator (default: one over env).
      mesh: the device mesh of the Arena.
      git_commit: recorded in member artifacts.
      driver: the learner kind's driver (`league.kinds`); default
        kinds.PGDriver(policy, learner).
      on_update: called as on_update(league, learner slot, metrics) after
        every update.
      draw_guards: called with every drawn table's {opponent: rows} counts;
        a guard raises to refuse an opponent (default: the controller's
        draw_guards(), if it has one).
    """
    x64.assert_x64("League")
    assert isinstance(config, LeagueConfig), type(config)
    if config.payoff != "none":
      assert (
          env.num_players == 2
      ), "the league's payoffs are 2-player (Evaluator)"
    if driver is None:
      driver = kinds.PGDriver(policy, learner)
    if driver.policy is not policy or driver.learner is not learner:
      raise ValueError(
          "the driver's policy and learner must be the league's objects"
      )
    if learner.policy() is not policy:
      raise ValueError("learner.policy() must be the league's policy object")
    self.env, self.policy, self.learner, self.controller = (
        env,
        policy,
        learner,
        controller,
    )
    self.driver = driver
    self.slot_policy = driver.slot_policy
    # What the pool holds and an Admit writes: the slot policy, or the
    # driver's member policy when the acting slot only labels (a mixture of
    # base and operator) or pools the plain network.
    self.member_policy = driver.member_policy
    self.store, self.run_dir, self.config = store, run_dir, config
    self.git_commit = git_commit
    self.on_update = on_update
    self.before_update: Optional[Callable[["League", LearnerSlot], None]] = None
    if draw_guards is None:
      draw_guards = getattr(controller, "draw_guards", lambda: ())()
    self.draw_guards = tuple(draw_guards)
    self._template = driver.wrap_member(policy.init_params(jax.random.key(0)))
    self._pool_params = pool_params.for_policy(self.member_policy)
    # The population pool: K positions over one stacked slot or K slots.
    k = config.pool_capacity
    if k == 0:
      shapes = []
    elif config.pool_layout == "stacked":
      shapes = [k]
    else:
      shapes = [1] * k
    self.pools: list[sampler.Pool] = [
        sampler.Pool(
            policy=self.member_policy,
            rule=config.pool_rule,
            names=(None,) * n,
            params=self._empty(n),
        )
        for n in shapes
    ]
    self._positions = []
    for i, n in enumerate(shapes):
      self._positions.extend((i, m) for m in range(n))
    # Fixed opponents: same (spec, rule) -> one slot, members stacked in order.
    groups: dict[str, list[Any]] = {}
    for name, (pol, params, rule) in dict(fixed_opponents or {}).items():
      pol, params = driver.wrap_opponent(pol, params)
      groups.setdefault(_group_key(pol, rule), []).append(
          (name, pol, params, rule)
      )
    self.fixed: list[sampler.Pool] = []
    for members in groups.values():
      pol, rule = members[0][1], members[0][3]
      stacked = members_lib.stack(pol, [(m[1], m[2]) for m in members])
      self.fixed.append(
          sampler.Pool(
              policy=pol,
              rule=rule,
              names=tuple(m[0] for m in members),
              params=stacked,
          )
      )
    self.driver.attach(self, mesh)
    if config.payoff == "evaluator":
      self.evaluator = evaluator or match.Evaluator(
          env,
          match.EvaluatorConfig(name="league", max_rows=config.eval_max_rows),
          mesh=mesh,
      )
      self._tree = None
    else:
      assert evaluator is None, f"payoff={config.payoff!r} uses no Evaluator"
      self.evaluator = None
      self._tree = expl.GameTree(env) if config.payoff == "exact" else None
    self._rules_sha = common.rules_sha(env)
    self._env_spec = env_registry.spec_of(env).to_json()
    spec = self.slot_policy.spec
    self._config_hash = hashlib.sha256(
        json.dumps(
            dict(
                config=dataclasses.asdict(config),
                controller=type(controller).__name__,
                arch=spec.arch_name,
                arch_kwargs=dict(spec.arch_kwargs),
                kind=driver.kind,
            ),
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    root = jax.random.key(config.seed)
    self._k_init, self._k_loop = rng.split(root)
    self._k_spawn, self._k_draw, self._k_eval, self._k_mutate = rng.split(
        rng.fold_in(root, 1), 4
    )
    # this League object's attempt (league.commit)
    self.attempt = commit.new_attempt_id()
    self._members_root = os.path.join(run_dir, commit.MEMBERS_DIR)
    self.members: dict[str, controller_lib.MemberInfo] = {}
    # Benched members' slot params, host numpy (Residency); evicted: none.
    self._off_arena: dict[str, Any] = {}
    self.learners: dict[str, LearnerSlot] = {}
    self.retired: dict[str, LearnerSlot] = {}
    self._spawned = 0
    self._mutations = 0
    # g: the run's update counter
    self.updates_total = np.int64(driver.update_offset)
    self._cells: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    self._tables: dict[tuple[Any, ...], np.ndarray] = {}
    self.payoff: Optional[match.PayoffMatrix] = None
    self.stopped, self.stop_reason = False, ""
    self.history: list[Any] = []  # the newest RECORD_CAP iteration records
    self.iteration = 0
    self._gate_evaluator = None
    self._gate_cells: dict[int, Any] = {}
    for name, params in dict(initial_members or {}).items():
      self._add_member(
          name,
          params,
          role="init",
          lifecycle="fixed",
          origin=None,
          at=0,
          parent=None,
          initial=True,
      )
    for aid in initial_ids:
      art = store.load(aid, cache=False)
      m = art.manifest
      ms = self.member_policy.spec
      if (m.arch_name, dict(m.arch_kwargs)) != (
          ms.arch_name,
          dict(ms.arch_kwargs),
      ):
        raise ValueError(
            f"{aid} is {m.arch_name}; the league's pool is {ms.arch_name}"
        )
      self._store_member(aid, self._load_member(aid))
      self.members[aid] = controller_lib.MemberInfo(
          aid, "init", "fixed", None, 0
      )

  def _empty(self, n: int) -> Any:
    """Empty pool params for n members.

    Args:
      n: The members.

    Returns:
      The params.
    """
    return self._pool_params.empty(self._template, n)

  @property
  def arena(self):
    """The driver's Arena."""
    return self.driver.arena

  @property
  def pool(self) -> sampler.Pool:
    """The population pool of a stacked layout (one slot).

    Raises:
      AttributeError: the layout is not one stacked pool slot.
    """
    if len(self.pools) != 1 or self.config.pool_layout != "stacked":
      raise AttributeError(
          "League.pool is the one stacked pool slot; use League.pools"
      )
    return self.pools[0]

  def opponent_slots(self) -> list[Any]:
    """Arena slots 1..: the population pool slot(s), then the fixed opponents.

    Returns:
      The slots.
    """
    return [p.slot() for p in self.pools] + [p.slot() for p in self.fixed]

  # -- pool positions ----------------------------------------------------------
  def placed(self) -> list[Any]:
    """The names in the Arena's positions, in position order.

    Returns:
      The names.
    """
    return [self.pools[i].names[m] for i, m in self._positions]

  def _position_of(self, name: str) -> Optional[int]:
    """The pool position of member `name`.

    Args:
      name: The member name.

    Returns:
      The position, or None.
    """
    for p, (i, m) in enumerate(self._positions):
      if self.pools[i].names[m] == name:
        return p
    return None

  def _set_position(self, p: int, name: Optional[str], params: Any) -> None:
    i, m = self._positions[p]
    pool = self.pools[i]
    names = list(pool.names)
    names[m] = name
    stacked = pool.params
    if params is not None:
      self._pool_params.check(params, self._template)
      stacked = self._pool_params.set(pool.params, m, params)
    self.pools[i] = dataclasses.replace(
        pool, names=tuple(names), params=stacked
    )

  def _free_position(self) -> Optional[int]:
    """The first free pool position.

    Returns:
      The position, or None when the pool is full.
    """
    for p, (i, m) in enumerate(self._positions):
      if self.pools[i].names[m] is None:
        return p
    return None

  def _pool_full(self) -> RuntimeError:
    """The error for a full pool.

    Returns:
      The RuntimeError.
    """
    return RuntimeError(
        "the population pool is full (pool_capacity="
        f"{self.config.pool_capacity}: "
        f"{self.placed()}); evict members, raise pool_capacity or set "
        "bench=True"
    )

  def _place(self, name: str, params: Any) -> None:
    """Puts slot params into name's position (or a free one); raises if full."""
    p = self._position_of(name)
    if p is None:
      p = self._free_position()
    if p is None:
      raise self._pool_full()
    self._set_position(p, name, params)
    self._off_arena.pop(name, None)

  def _unplace(self, name: str) -> None:
    """Takes name out of its position.

    Takes name out of its position; a member's params move off the Arena
    (to host memory).

    Args:
      name: the member or learner name.
    """
    p = self._position_of(name)
    if p is None:
      return
    if name in self.members:
      self._off_arena[name] = self._pool_params.to_host(
          self._position_params(p)
      )
    self._set_position(p, None, None)

  def _evict(self, name: str) -> None:
    """Name leaves the Arena and keeps no weights.

    Name leaves the Arena and keeps no weights (its artifact has them);
    host-held weights are released for reuse (`policies.members.release`).

    Args:
      name: the member name.
    """
    p = self._position_of(name)
    held = None
    if p is not None:
      held = self._position_params(p)
      self._set_position(p, None, None)
    elif name in self._off_arena:
      held = self._off_arena[name]
    self._off_arena.pop(name, None)
    if held is not None and name in self.members:
      members_lib.release(self.member_policy, held)

  def _position_params(self, p: int) -> Any:
    """The member params at pool position p.

    Args:
      p: The pool position.

    Returns:
      The params.
    """
    i, m = self._positions[p]
    return self._pool_params.get(self.pools[i].params, m)

  def _store_member(self, name: str, slot_params: Any) -> None:
    """A member's slot params into a position, else (bench) off the Arena."""
    self._pool_params.check(slot_params, self._template)
    if self._position_of(name) is not None or self._free_position() is not None:
      self._place(name, slot_params)
    elif self.config.bench:
      self._off_arena[name] = self._pool_params.to_host(slot_params)
    else:
      raise self._pool_full()

  def member_params(self, name: str) -> Any:
    """A member's slot params (the slot policy's: what plays and is stored).

    A seated member's from its position, a benched one's from host memory, an
    evicted one's loaded from its artifact (host numpy, not kept).

    Args:
      name: the member name.

    Returns:
      The slot params.

    Raises:
      KeyError: no such member.
      AssertionError: the member is neither seated, benched nor evicted.
    """
    if name not in self.members:
      raise KeyError(f"no member {name!r}")
    p = self._position_of(name)
    if p is not None:
      return self._position_params(p)
    if name in self._off_arena:
      return self._off_arena[name]
    if self.members[name].lifecycle != "evicted":
      raise AssertionError(
          f"member {name} is neither seated, benched nor evicted"
      )
    return self._load_member(name)

  def _load_member(self, name: str) -> Any:
    """Name's slot params from its artifact, as host numpy.

    Name's slot params from its artifact as host numpy (uncached), adopted
    into the member policy (`policies.members.adopt`).

    Args:
      name: the member name.

    Returns:
      The slot params.
    """
    art = self.store.load(name, cache=False)
    params = self._pool_params.to_host(
        members_lib.adopt(self.member_policy, art.policy, art.params)
    )
    self._pool_params.check(params, self._template)
    return params

  def resident_members(self) -> dict[str, str]:
    """Where each member's weights are.

    "arena" (a position, on device), "host" (benched) or "artifact" (evicted:
    none in memory).

    Returns:
      {member: place}.
    """
    out = {}
    for name in self.members:
      if self._position_of(name) is not None:
        out[name] = "arena"
      elif name in self._off_arena:
        out[name] = "host"
      else:
        out[name] = "artifact"
    return out

  def learner_params(self, name: str) -> Any:
    """A learner's params (the learner's policy's).

    Args:
      name: the learner name.

    Returns:
      The params.
    """
    return self.learner.params(self.learners[name].state)

  def _registered_here(self, name: str, strict: bool = True) -> bool:
    """Whether name is registered under this run's members dir.

    Name is in the store index, registered under this run's members dir
    (registered elsewhere: raises when strict, else False).

    Args:
      name: the member name.
      strict: raise when name is registered elsewhere.

    Returns:
      Whether name is registered under this run's members dir.
    """
    if name not in self.store.ids():
      return False
    path = self.store.get(name).path
    root = artifact.norm_path(self._members_root).rstrip("/") + "/"
    if not (path.rstrip("/") + "/").startswith(root):
      if strict:
        raise ValueError(
            f"member id {name!r} is registered to {path}, outside this run's "
            f"members dir {root}"
        )
      return False
    return True

  def _add_member(
      self, name, params, role, lifecycle, origin, at, parent, initial=False
  ):
    """params: the learner's params (wrapped into the slot policy's here).

    The artifact goes under <run_dir>/members/<base>/<attempt>/. A name the
    store already holds under this run's members dir was registered by an
    attempt whose iteration was never committed (a committed member is in
    self.members); its registration is replaced, its files left alone. An
    initial member (construction, a pure function of the seed) a previous
    attempt of this run registered is reused as is.

    Args:
      name: the member's name.
      params: the learner's params (wrapped into the slot policy's here).
      role: the member's role.
      lifecycle: the member's lifecycle.
      origin: where the member came from.
      at: the iteration it is admitted at.
      parent: None, or the member it descends from.
      initial: an initial member (construction, a pure function of the seed).

    Raises:
      ValueError: the member id is already used in this league.
      RuntimeError: the population pool is full.
    """
    if name in self.members or name in self.learners or name in self.retired:
      raise ValueError(f"member id {name!r} is already used in this league")
    if not self.config.bench and self._position_of(name) is None:
      if self._free_position() is None:
        raise self._pool_full()  # before anything is frozen or written
    base, step = artifact_store.parse_artifact_id(name)
    registered = self._registered_here(name)
    if initial and registered:
      sp = self._load_member(name)
      logging.info(
          "league: initial member %s reused from %s",
          name,
          self.store.get(name).path,
      )
    else:
      self.driver.check_admissible()
      sp = self.driver.freeze(params)
      if registered:
        old = self.store.unregister(name, under=self._members_root)
        logging.info(
            "league: %s replaces the uncommitted registration at %s",
            name,
            old.path,
        )
      path = os.path.join(self._members_root, base, self.attempt)
      parents, lineage = (), None
      if parent is not None:
        _, p_step = artifact_store.parse_artifact_id(parent)
        parents, lineage = [parent], dict(
            source=parent, parent_run_id=None, step=p_step
        )
      artifact.write_artifact(
          path,
          step,
          self.member_policy,
          sp,
          rules_sha=self._rules_sha,
          config_hash=self._config_hash,
          algo=self.config.algo,
          env_name=self.env.name,
          default_rule=self.config.pool_rule,
          git_commit=self.git_commit,
          parents=parents,
          lineage=lineage,
          env_spec=self._env_spec,
      )
      got = self.store.register(path, step, artifact_id=name)
      assert got == name, (got, name)
    self._store_member(name, sp)
    self.members[name] = controller_lib.MemberInfo(
        name, role, lifecycle, origin, at
    )
    logging.info(
        "league: admitted %s (role %s, %s) from %s",
        name,
        role,
        lifecycle,
        origin,
    )

  # -- views and payoffs -------------------------------------------------------
  def _entrants(self) -> list[tuple[str, int]]:
    """The live entrants: members, then learners with their updates.

    Returns:
      [(name, version)].
    """
    out = [(n, 0) for n, m in self.members.items() if m.lifecycle != "evicted"]
    out += [(n, int(ls.updates)) for n, ls in self.learners.items()]
    return out

  def _params_of(self, name: str) -> Any:
    """Slot params of an entrant (a live learner's wrapped, or a member's).

    Args:
      name: The entrant name.

    Returns:
      The slot params.
    """
    if name in self.learners:
      return self.driver.wrap_member(self.learner_params(name))
    return self.member_params(name)

  def _table(self, name: str, ver: int) -> np.ndarray:
    """The policy table of entrant `name` at version `ver`, cached.

    Args:
      name: The entrant.
      ver: Its version.

    Returns:
      The policy table.
    """
    key = (name, ver)
    if key not in self._tables:
      self._tables[key] = self._tree.policy_table(
          self.member_policy, self._params_of(name), self.config.pool_rule
      )
    return self._tables[key]

  def _cell(self, a: tuple[Any, ...], b: tuple[Any, ...]) -> tuple[Any, ...]:
    """The payoff cell of entrant `a` against `b`, cached.

    Args:
      a: the row entrant, (name, version).
      b: the column entrant, (name, version).

    Returns:
      (a's mean, its se, b's mean, its se).
    """
    key = (a, b)
    if key not in self._cells:
      if self._tree is not None:
        v = mixture.symmetric_payoff(
            self._tree, self._table(*a), self._table(*b)
        )
        self._cells[key] = (v, 0.0, -v, 0.0)
      else:
        tag = f"{a[0]}|{a[1]}|{b[0]}|{b[1]}".encode()
        k = rng.fold_in(self._k_eval, zlib.crc32(tag))
        rule = self.config.pool_rule
        ea = match.Entrant(
            a[0], self.member_policy, self._params_of(a[0]), rule
        )
        eb = match.Entrant(
            b[0], self.member_policy, self._params_of(b[0]), rule
        )
        r = self.evaluator.head_to_head(ea, eb, self.config.eval_deals, k)
        eb_est = stats.estimate(r.b_payoff.mean(1))
        self._cells[key] = (r.mean.mean, r.mean.se, eb_est.mean, eb_est.se)
    return self._cells[key]

  def evaluate(self) -> Optional[match.PayoffMatrix]:
    """Computes the payoff matrix over the entrants (None when payoff="none").

    Returns:
      The PayoffMatrix, or None.
    """
    if self.config.payoff == "none":
      self.payoff = None
      return None
    ents = self._entrants()
    n = len(ents)
    mean, se = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
      for j in range(i + 1, n):
        ma, sa, mb, sb = self._cell(ents[i], ents[j])
        mean[i, j], se[i, j], mean[j, i], se[j, i] = ma, sa, mb, sb
    self.payoff = controller_lib.payoff_matrix([e[0] for e in ents], mean, se)
    return self.payoff

  def view(self) -> controller_lib.LeagueView:
    """The LeagueView the controller steps on.

    Returns:
      The view.
    """
    assert (
        self.payoff is not None or self.config.payoff == "none"
    ), "evaluate() first"
    learners = tuple(
        controller_lib.LearnerInfo(
            l.name,
            l.role,
            int(l.updates),
            l.started_at,
            l.mix,
            tuple(l.scores),
            l.parent,
        )
        for l in self.learners.values()
    )
    v = controller_lib.LeagueView(
        iteration=self.iteration,
        members=tuple(self.members.values()),
        learners=learners,
        payoff=self.payoff,
    )
    req_fn = getattr(self.controller, "gate_request", None)
    req = req_fn(v) if req_fn is not None else None
    if req is None:
      return v
    return dataclasses.replace(v, gate=self.gate_cell(req))

  def gate_cell(
      self, req: controller_lib.GateRequest
  ) -> controller_lib.GateCell:
    """Plays a promotion gate cell.

    `req.subject` (the learner's params as the plain policy) against
    `req.opponent` (the member's, unwrapped), ARGMAX both sides, on req.deals
    duplicate deal pairs keyed by req.seed. A repeated request for a generation
    gets the same cell. Two-player only.

    Args:
      req: the GateRequest.

    Returns:
      The GateCell.

    Raises:
      ValueError: the game is not two-player.
    """
    if req.generation in self._gate_cells:
      cell = self._gate_cells[req.generation]
      assert (cell.subject, cell.opponent) == (req.subject, req.opponent), (
          cell,
          req,
      )
      return cell
    if self.env.num_players != 2:
      raise ValueError("the promotion gate is two-player (the Evaluator's)")
    seed, n = int(req.seed), int(req.deals)
    if self._gate_evaluator is None:
      self._gate_evaluator = self.evaluator or match.Evaluator(
          self.env,
          match.EvaluatorConfig(
              name="gate", max_rows=self.config.eval_max_rows
          ),
      )
    # The plain network (the driver's plain_policy: the learner's policy, or
    # the slot factory's plain_member, e.g. a base routed to a scripted phase).
    pol, plain = self.driver.plain_policy, self.driver.plain_params
    a = match.Entrant(
        req.subject, pol, plain(self.learner_params(req.subject)), types.ARGMAX
    )
    b = match.Entrant(
        req.opponent,
        pol,
        plain(self.driver.unwrap_member(self.member_params(req.opponent))),
        types.ARGMAX,
    )
    r = self._gate_evaluator.head_to_head(a, b, n, jax.random.key(seed))
    est = stats.estimate(np.asarray(r.a_payoff, np.float64).mean(1))
    cell = controller_lib.GateCell(
        generation=int(req.generation),
        subject=req.subject,
        opponent=req.opponent,
        seed=seed,
        n_deals=n,
        mean=float(est.mean),
        se=float(est.se),
        lo=float(est.lo),
        hi=float(est.hi),
    )
    self._gate_cells[req.generation] = cell
    logging.info("league: gate cell generation %d: %s", req.generation, cell)
    return cell

  def _score(self, ls: LearnerSlot, pm: match.PayoffMatrix) -> float:
    """The learner's mix-weighted mean payoff.

    Args:
      ls: The learner slot.
      pm: The payoff matrix.

    Returns:
      The score.
    """
    idx = {n: i for i, n in enumerate(pm.names)}
    i = idx[ls.name]
    return float(
        sum(
            w * pm.mean[i, idx[n]]
            for n, w in ls.mix.weights.items()
            if w > 0.0 and n in idx
        )
    )

  # -- actions -----------------------------------------------------------------
  def _fixed_names(self) -> set[str]:
    """The names in the fixed pools.

    Returns:
      The names.
    """
    names = set()
    for p in self.fixed:
      names.update(p.names)
    return names

  def _known(self, name: str) -> bool:
    """Whether `name` is a live learner, member or fixed member.

    Args:
      name: The name.

    Returns:
      True when known.
    """
    if self.config.bench:  # seated after the actions (_seat)
      live = name in self.learners or (
          name in self.members and self.members[name].lifecycle != "evicted"
      )
      return live or name in self._fixed_names()
    return name in self.placed() or name in self._fixed_names()

  def _check_mix(self, learner: str, mix: sampler.Mix) -> None:
    for n in mix.support():
      if not self._known(n):
        raise KeyError(
            f"SetMix({learner}): {n!r} is not in the pool (members: "
            f"{[x for x in self.placed() if x]}; fixed: "
            f"{sorted(self._fixed_names())})"
        )

  def _seat(self) -> None:
    """Bench: seats every population name a live mix references.

    bench: every population name a live mix references sits in a position
    (unreferenced occupants make room; their params move off the Arena).

    Raises:
      RuntimeError: the live mixes name more entrants than the pool has
        positions.
    """
    fixed = self._fixed_names()
    wanted: list[Any] = []
    for ls in self.learners.values():
      for n in ls.mix.support():
        if n not in fixed and n not in wanted:
          wanted.append(n)
    if len(wanted) > len(self._positions):
      raise RuntimeError(
          f"the live mixes name {len(wanted)} population entrants {wanted} but "
          f"the pool has {len(self._positions)} positions (pool_capacity)"
      )
    for n in wanted:
      if self._position_of(n) is not None:
        continue
      p = self._free_position()
      if p is None:
        for q in range(len(self._positions)):
          occ = self.placed()[q]
          if occ not in wanted:
            self._unplace(occ)
            p = q
            break
      assert p is not None, (n, wanted, self.placed())
      params = (
          self.driver.wrap_member(self.learner_params(n))
          if n in self.learners
          else self._off_arena[n]
      )
      self._set_position(p, n, params)
      self._off_arena.pop(n, None)

  def _spawn(self, a: controller_lib.SpawnLearner, it: int) -> None:
    """Spawns a learner, fresh or initialized from a learner or member.

    Args:
      a: the SpawnLearner action.
      it: the iteration.

    Raises:
      ValueError: the name is already used or has an '@'.
    """
    if (
        a.name in self.learners
        or a.name in self.members
        or a.name in self.retired
    ):
      raise ValueError(f"learner name {a.name!r} is already used")
    if "@" in a.name:
      raise ValueError(f"learner names have no '@' (member ids do): {a.name!r}")
    idx = self._spawned
    self._spawned += 1
    key = self._k_init if idx == 0 else rng.fold_in64(self._k_spawn, idx)
    parent = None
    if a.init_from is None:
      state = self.driver.init_state(key, self.env)
    else:
      if a.init_from in self.learners:
        params = self.learner_params(a.init_from)
      else:
        params = self.driver.unwrap_member(self.member_params(a.init_from))
        parent = a.init_from
      state = self._state_from_params(
          key,
          params,
          f"SpawnLearner({a.name}, init_from={a.init_from})",
          a.name,
          REPLACE_PARAMS,
      )
    if self.config.reset_optimizer_each_iteration:
      self._refuse_state_rewrite(
          "reset_optimizer_each_iteration",
          a.name,
          RESET_OPT,
          state,
          ("opt_state",),
          from_params=True,
      )
    state = self._commit_state(state)
    self.learners[a.name] = LearnerSlot(
        a.name, a.role, state, idx, it, parent=parent
    )
    # a live pool entry (no pool: none)
    if self._positions and not self.config.bench:
      self._place(a.name, self.driver.wrap_member(self.learner.params(state)))

  def _commit_state(self, state: Any) -> Any:
    """The driver's commit form of a learner state.

    Args:
      state: The learner state.

    Returns:
      The committed state.
    """
    return self.driver.commit_state(state)

  def _refuse_state_rewrite(
      self,
      action: str,
      learner: str,
      capability: str,
      state: Any,
      entries: Sequence[str],
      from_params: bool = False,
  ) -> None:
    """Refuses a learner-state rewrite the learner cannot take.

    Refuses, before anything changes, an action that rewrites state_dict
    entries (params / opt_state) a learner does not have. The message names
    the learner slot, its Learner class and the missing capability
    (refusal_message). from_params: the action builds a state from params
    (learner.init_from_params, else the state_dict's "params" entry).

    Args:
      action: the action's name.
      learner: the learner slot's name.
      capability: the capability the action needs.
      state: the learner state.
      entries: the state_dict entries the action rewrites.
      from_params: the action builds a state from params.
    """
    need = list(entries)
    if from_params and getattr(self.learner, "init_from_params", None) is None:
      need.append("params")
    if not need:
      return
    have = sorted(self.learner.state_dict(state))
    missing = [k for k in dict.fromkeys(need) if k not in have]
    if missing:
      raise ValueError(
          refusal_message(
              action,
              learner,
              type(self.learner).__name__,
              capability,
              missing,
              have,
          )
      )

  def _state_from_params(
      self, key, params, action: str, learner: str, capability: str
  ) -> Any:
    """A learner state holding `params`.

    Args:
      key: the init PRNG key.
      params: the params.
      action: names the action in refusals.
      learner: the learner slot's name.
      capability: the capability the action needs.

    Returns:
      The learner state.
    """
    init_from_params = getattr(self.learner, "init_from_params", None)
    if init_from_params is not None:
      return init_from_params(params)
    # Refuse before init_state when a learner state exists to read the
    # state_dict's entries from: a host learner's init may replace its live
    # trainer. The first spawn has nothing to change yet.
    existing = next(
        (ls.state for ls in self.learners.values() if ls.state is not None),
        None,
    )
    if existing is not None:
      self._refuse_state_rewrite(
          action, learner, capability, existing, (), from_params=True
      )
    state = self.driver.init_state(key, self.env)
    self._refuse_state_rewrite(
        action, learner, capability, state, (), from_params=True
    )
    blob = dict(self.learner.state_dict(state))
    blob["params"] = params
    return self.learner.load_state(blob)

  def _reset_optimizer(self, ls: LearnerSlot) -> None:
    """Resets ls's optimizer state around its current params.

    Ls's optimizer state <- a fresh one around its current params; every
    other entry of the learner's state_dict is kept (LeagueConfig doc).

    Args:
      ls: the LearnerSlot.
    """
    self._refuse_state_rewrite(
        "reset_optimizer_each_iteration",
        ls.name,
        RESET_OPT,
        ls.state,
        ("opt_state",),
    )
    blob = dict(self.learner.state_dict(ls.state))
    key = rng.fold_in64(self._k_spawn, ls.spawn_index)
    fresh = self.learner.state_dict(
        self._state_from_params(
            key,
            self.learner.params(ls.state),
            "reset_optimizer_each_iteration",
            ls.name,
            RESET_OPT,
        )
    )
    blob["opt_state"] = fresh["opt_state"]
    ls.state = self._commit_state(self.learner.load_state(blob))

  def _copy_mutate(self, a: controller_lib.CopyMutate) -> None:
    """Applies a CopyMutate: dst <- shrink * src + perturb * fresh.

    Args:
      a: the CopyMutate action.
    """
    src, dst = self.learners[a.src], self.learners[a.dst]
    cap = REPLACE_PARAMS if (a.shrink, a.perturb) == (1.0, 0.0) else PERTURB
    self._refuse_state_rewrite(
        f"CopyMutate({a.src} -> {a.dst})", a.dst, cap, dst.state, ("params",)
    )
    if a.src == a.dst and a.shrink == 1.0 and a.perturb == 0.0:
      return  # copying a learner onto itself keeps the same state object
    key = rng.fold_in64(self._k_mutate, self._mutations)
    fresh = self.policy.init_params(key)
    new = self.driver.mutate(
        self.learner.params(src.state), fresh, a.shrink, a.perturb
    )
    self._mutations += 1
    blob = dict(self.learner.state_dict(dst.state))
    blob["params"] = new
    dst.state = self._commit_state(self.learner.load_state(blob))
    logging.info(
        "league: %s <- %.3g * %s + %.3g * fresh",
        a.dst,
        a.shrink,
        a.src,
        a.perturb,
    )

  def _apply(self, actions: Sequence[Any], it: int) -> bool:
    """Applies the controller's actions.

    Args:
      actions: the actions.
      it: the iteration.

    Returns:
      Whether an action spawned a learner or admitted a member.

    Raises:
      ValueError: an evicted member re-enters.
      TypeError: an action is unknown.
    """
    structural = False
    for a in actions:
      if isinstance(a, controller_lib.SpawnLearner):
        self._spawn(a, it)
        structural = True
      elif isinstance(a, controller_lib.RetireLearner):
        ls = self.learners.pop(a.name)
        # off the device; released at the next save
        ls.state = self.driver.retire_state(ls.state)
        self.retired[a.name] = ls
        self.driver.forget(a.name)
        p = self._position_of(a.name)
        if p is not None:
          self._set_position(p, None, None)
      elif isinstance(a, controller_lib.Admit):
        ls = self.learners[a.learner]
        # A learner warm-started from a member descends from that artifact.
        self._add_member(
            a.member,
            self.learner.params(ls.state),
            a.role,
            a.lifecycle,
            a.learner,
            it,
            ls.parent,
        )
        structural = True
      elif isinstance(a, controller_lib.SetLifecycle):
        m = self.members[a.member]
        if a.lifecycle == "evicted" and m.lifecycle != "evicted":
          self._evict(a.member)
        elif m.lifecycle == "evicted" and a.lifecycle != "evicted":
          raise ValueError(
              f"{a.member} was evicted; re-entry is a new admission"
          )
        self.members[a.member] = dataclasses.replace(m, lifecycle=a.lifecycle)
      elif isinstance(a, controller_lib.SetMix):
        self._check_mix(a.learner, a.mix)
        self.learners[a.learner].mix = a.mix
      elif isinstance(a, controller_lib.CopyMutate):
        self._copy_mutate(a)
      elif isinstance(a, controller_lib.Stop):
        if not self.stopped:
          logging.info("league: stop at iteration %d: %s", it, a.reason)
        self.stopped, self.stop_reason = True, a.reason
      else:
        raise TypeError(f"unknown action {a!r}")
    if self.config.bench:
      self._seat()
    for ls in self.learners.values():
      self._check_mix(ls.name, ls.mix)
      if self.config.bench:
        for n in ls.mix.support():
          assert n in self.placed() or n in self._fixed_names(), (
              n,
              self.placed(),
          )
    return structural

  # -- the loop ----------------------------------------------------------------
  def tick(self, it: int) -> list[Any]:
    """One league iteration: evaluate, step the controller, apply, train.

    Args:
      it: the iteration.

    Returns:
      The applied actions.

    Raises:
      AssertionError: the controller still adds entrants after max_rounds.
    """
    applied = []
    for rnd in range(self.config.max_rounds):
      pm = self.evaluate()
      if rnd == 0 and pm is not None:
        for ls in self.learners.values():
          if ls.updates > 0:
            ls.scores.append(self._score(ls, pm))
      acts = list(self.controller.step(self.view()))
      applied += acts
      if not self._apply(acts, it):
        return applied
    raise AssertionError(
        "the controller still added entrants after max_rounds="
        f"{self.config.max_rounds} rounds at iteration {it}"
    )

  def all_pools(self) -> tuple[Any, ...]:
    """The population pool slots, then the fixed-opponent slots.

    Returns:
      The pools.
    """
    return (*self.pools, *self.fixed)

  def _draw(self, ls: LearnerSlot, g: np.int64) -> tuple[Any, ...]:
    """(table, opponent params, counts, fresh) for learner ls's update g.

    Args:
      ls: The learner slot.
      g: The global update index.

    Returns:
      (table, opponent params, counts, fresh).

    Raises:
      ValueError: self_play_seats='all' with a mix that is not pure self-play.
    """
    cfg = self.config
    pools = self.all_pools()
    sig = (ls.mix, tuple(tuple(p.names) for p in pools))
    fresh = cfg.redraw == "update" or ls.draw is None or ls.draw_sig != sig
    if fresh:
      p = self.env.num_players
      if cfg.self_play_seats == "all":
        if ls.mix.self_play != 1.0:
          raise ValueError(
              "self_play_seats='all' emits every seat of a self-play row and "
              f"needs a pure self-play mix; {ls.name}'s mix is {ls.mix}"
          )
        table = arena_lib.self_play_table(cfg.num_envs, p, all_seats=True)
        counts = {sampler.SELF: cfg.num_envs}
      else:
        d = sampler.sample(
            sampler.LeagueSlots(
                num_players=p,
                batch=cfg.num_envs,
                pools=pools,
                mix=ls.mix,
                stratified=cfg.stratified,
            ),
            rng.fold_in64(self._k_draw, g),
        )
        table, counts = d.table, d.counts
      for guard in self.draw_guards:
        guard({str(k): int(v) for k, v in counts.items()})
      ls.draw, ls.draw_sig = (table, counts), sig
    table, counts = ls.draw
    return table, [pool.params for pool in pools], counts, fresh

  def _train(self, ls: LearnerSlot) -> None:
    """Runs one learner's updates of the iteration.

    Args:
      ls: the learner slot.
    """
    for _ in range(self.config.updates_per_iteration):
      g = np.int64(self.updates_total)
      key = rng.fold_in64(self._k_loop, g)
      table, opp, counts, fresh = self._draw(ls, g)
      if self.before_update is not None:
        self.before_update(self, ls)
      ls.state, m, d_frames, d_decisions = self.driver.train_step(
          ls, table, opp, key, fresh
      )
      ls.updates = np.int64(ls.updates + np.int64(1))
      ls.frames = np.int64(ls.frames + np.int64(d_frames))
      ls.decisions = np.int64(ls.decisions + np.int64(d_decisions))
      self.updates_total = np.int64(g + np.int64(1))
      m = dict(m, frames=int(ls.frames), learner_decisions=int(ls.decisions))
      ls.draw_counts.append(counts)
      del ls.draw_counts[:-RECORD_CAP]
      ls.last_metrics = m
      if self.on_update is not None:
        self.on_update(self, ls, m)

  def run(
      self,
      iterations: int,
      on_iteration: Optional[Callable[["League", int], None]] = None,
  ) -> list[Any]:
    """Runs up to `iterations` iterations (fewer if the controller stops).

    Args:
      iterations: the most iterations to run.
      on_iteration: called with (league, iteration) after each one.

    Returns:
      The history (its newest RECORD_CAP records).
    """
    for _ in range(int(iterations)):
      it = self.iteration
      acts = self.tick(it)
      rec = dict(
          iteration=it,
          actions=[repr(a) for a in acts],
          members=len(
              [m for m in self.members.values() if m.lifecycle != "evicted"]
          ),
          learners={},
      )
      if not self.stopped:
        order = sorted(
            self.learners, key=lambda n: self.learners[n].spawn_index
        )
        for n in order:  # placed live entries: start-of-iteration params
          if self._position_of(n) is not None:
            self._place(n, self.driver.wrap_member(self.learner_params(n)))
        for n in order:
          ls = self.learners[n]
          if self.config.reset_optimizer_each_iteration:
            self._reset_optimizer(ls)
          self._train(ls)
          rec["learners"][n] = dict(
              updates=int(ls.updates),
              frames=int(ls.frames),
              decisions=int(ls.decisions),
              last=ls.last_metrics,
          )
      self.history.append(rec)
      del self.history[:-RECORD_CAP]
      self.iteration += 1
      if on_iteration is not None:
        on_iteration(self, it)
      if self.stopped:
        break
    return self.history

  # -- checkpoint --------------------------------------------------------------
  def save(self, ckpt_dir: str, step: int, release_retired: bool = True) -> str:
    """Writes the league's state after iteration step - 1.

    self.iteration == step.

    Under this attempt's <ckpt_dir>/league/<step>-<attempt>/: every live
    learner's state, then state.pkl (iteration, counters, members, pool
    seating, learner records, the controller, the driver's host state and
    every ArenaSource snapshot). The caller commits the ref
    (`league.commit.with_commit`).

    Retired learners' states are released once this checkpoint is written,
    unless release_retired=False (a resume checkpoint, `league.resume`, which
    the next one replaces).

    Args:
      ckpt_dir: the run's checkpoint dir.
      step: the step, equal to self.iteration.
      release_retired: release retired learners' states.

    Returns:
      The ref.

    Raises:
      FileExistsError: this attempt already saved `step`.
    """
    assert int(step) == self.iteration, (step, self.iteration)
    r = commit.ref(step, self.attempt)
    for name, ls in self.learners.items():
      run_state.save(
          commit.learner_dir(ckpt_dir, name, r=r), step, self.learner, ls.state
      )

    def strip(ls):
      """The slot without its learner state.

      Args:
        ls: The learner slot.

      Returns:
        The slot.
      """
      return dataclasses.replace(ls, state=None)

    sources = self.driver.sources_state()
    assert set(sources) <= set(self.learners), (
        sorted(sources),
        sorted(self.learners),
    )
    blob = dict(
        iteration=self.iteration,
        updates_total=int(self.updates_total),
        controller=self.controller,
        members=dict(self.members),
        pool_names=tuple(self.placed()),
        learners={n: strip(l) for n, l in self.learners.items()},
        retired={n: strip(l) for n, l in self.retired.items()},
        spawned=self._spawned,
        mutations=self._mutations,
        stopped=self.stopped,
        stop_reason=self.stop_reason,
        history=list(self.history),
        config_hash=self._config_hash,
        driver_state=self.driver.host_state(),
        sources=sources,
        attempt=self.attempt,
        state_version=STATE_VERSION,
    )
    buf = io.BytesIO()
    pickle.dump(blob, buf, protocol=pickle.HIGHEST_PROTOCOL)
    path = commit.state_path(ckpt_dir, step, r)
    if path.exists():
      raise FileExistsError(
          f"{path} exists: attempt {self.attempt} saved step {step} twice"
      )
    artifact.atomic_write(path, buf.getvalue())
    for ls in self.retired.values() if release_retired else ():
      ls.state = None
    logging.info(
        "league: state after iteration %d -> %s/%s", step - 1, ckpt_dir, r
    )
    return r

  def restore(
      self, ckpt_dir: str, step: int, ref: Optional[str] = None
  ) -> None:
    """Loads the checkpoint of `step` into this freshly built League.

    The League must be built from the same config (env, policy, learner,
    store); its controller is replaced by the saved one, which must be of the
    same class. ref: what save returned (default: the ref train_state.json
    commits for step). Store entries under this run's members dir that the
    checkpoint does not name (an uncommitted attempt's admissions) are
    unregistered. Payoff caches are recomputed from deterministic keys.

    Args:
      ckpt_dir: the run's checkpoint dir.
      step: the step.
      ref: what save returned (default: the ref train_state.json commits for
        step).

    Raises:
      RuntimeError: the League is not freshly built.
      ValueError: the checkpoint is of another config, or a learner's source
        table is not its standing table.
      TypeError: the saved controller is of another class.
    """
    if self.iteration or self.learners or self.retired:
      raise RuntimeError("restore() needs a freshly built League")
    loc = commit.locate(ckpt_dir, step, ref)
    path = loc.state_path
    blob = read_state(path)
    if blob["config_hash"] != self._config_hash:
      raise ValueError(f"{path}: a league of a different config")
    if type(blob["controller"]) is not type(self.controller):
      raise TypeError(
          f"{path}: controller {type(blob['controller']).__name__}, this "
          "league's "
          f"is {type(self.controller).__name__}"
      )
    assert int(blob["iteration"]) == int(step), (blob["iteration"], step)
    self.controller = blob["controller"]
    self.iteration = int(blob["iteration"])
    self.updates_total = np.int64(blob["updates_total"])
    self._spawned, self._mutations = int(blob["spawned"]), int(
        blob["mutations"]
    )
    self.stopped, self.stop_reason = bool(blob["stopped"]), blob["stop_reason"]
    self.history = list(blob["history"])
    self.retired = dict(blob["retired"])
    for name, ls in blob["learners"].items():
      d = os.path.join(loc.learners_root, name)
      state = run_state.restore(d, step, self.learner)
      self.learners[name] = dataclasses.replace(
          ls, state=self._commit_state(state)
      )
    self.members = dict(blob["members"])
    for aid in self.store.ids():  # an uncommitted attempt's registrations
      if aid not in self.members and self._registered_here(aid, strict=False):
        old = self.store.unregister(aid, under=self._members_root)
        logging.info(
            "league: dropped the uncommitted registration %s (%s)",
            aid,
            old.path,
        )
    names = tuple(blob["pool_names"])
    assert len(names) == len(self._positions), (names, self._positions)
    for p in range(len(self._positions)):
      self._set_position(p, None, None)
    self._off_arena = {}
    for p, name in enumerate(names):
      if name is None:
        continue
      params = (
          self.driver.wrap_member(self.learner_params(name))
          if name in self.learners
          else self._load_member(name)
      )
      self._set_position(p, name, params)
    # benched: host memory; evicted: nothing
    for (
        name,
        m,
    ) in self.members.items():
      if self._position_of(name) is None and m.lifecycle != "evicted":
        self._off_arena[name] = self._load_member(name)
    self.driver.load_host_state(blob["driver_state"])
    sources = dict(blob["sources"])
    for name, snap in sources.items():
      ls = self.learners[name]
      if ls.draw is None or not commit.same_table(
          commit.device_tree(snap["table"]), ls.draw[0]
      ):
        raise ValueError(
            f"{path}: {name}'s ArenaSource table is not its standing table"
        )
    self.driver.load_sources_state(sources)
    logging.info(
        "league: restored iteration %d from %s (%s; %d members, %d learners, "
        "%d "
        "sources)",
        self.iteration,
        ckpt_dir,
        loc.label,
        len(self.members),
        len(self.learners),
        len(sources),
    )
