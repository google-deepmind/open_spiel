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

"""The behaviour-cloning learner kind: a student network imitates a teacher.

    spec = kinds.BC(teacher=refs.PolicyRef(artifact="/runs/teacher@4000"),
                    learner=bc.BCConfig(...),
                    beta=dagger.beta_schedule(1.0, 0.0, decay_iters=100))

Data comes from a DAgger rollout (`data="online"`: `sources.dagger`; the
teacher acts with probability beta and the student with 1 - beta, and the
teacher labels every decision) or from recorded shards (`data="shards"`: the
run's `data_dir`, written by `record`). The student is the run's `policy_kwargs`
network. Every `heldout_every` updates and at the end the student is scored
against the teacher on a fixed teacher-driven held-out state set (top-1
agreement, KL; `eval.heldout`).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import heldout as heldout_lib
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.league.kinds import base
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.parallel import data_parallel
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources import shards

BC_DATA = ("online", "shards")
#: ShardWriter info_schema_version of the InfoStates BC records. Bump it when
#: an Env's InfoState layout changes, so stale shards are refused.
INFO_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class BC:
  """Behaviour cloning / distillation.

  Attributes:
    teacher: the policy imitated.
    learner: the BCConfig of the one BCLearner.
    beta: the probability the teacher acts, a Schedule over updates.
    student_init: None, or a PolicyRef whose params warm-start the student
      (fresh optimizer). Its policy must have the student's spec.
    endgame: a final phase played by every seat, with endgame_route
      (`sources.dagger.EndgameRoute`); both or neither. Unlike PG's
      scripted_endgame, which names the game's own scripted endgame
      (`GameSpec.endgame`), BC takes any policy and route: a PolicyRef to
      that architecture with the game's route is the same final phase.
    endgame_route: the route of the endgame phase.
    opening_plies: plies played uniformly at random before recording.
    first_hand_only: record only each row's first hand.
    gamma: the value targets' discount (`sources.targets`).
    value_target: "return" (Monte Carlo returns, completed with the next
      segment when return_lookahead) or "teacher" (the teacher's value).
    return_lookahead: complete returns with the next segment.
    heldout_states: the held-out set's size.
    heldout_seed: the held-out set's seed.
    heldout_every: the updates between held-out evaluations (0: only at the
      end).
    heldout_num_envs: the held-out evaluation's rows (0: the league's
      num_envs).
    data: "online" (DAgger) or "shards" (the run's data_dir).
    kind: the kind's registered name.
  """

  teacher: refs.PolicyRef
  learner: bc.BCConfig
  beta: schedules.Schedule = schedules.constant(1.0)
  student_init: Optional[refs.PolicyRef] = None
  endgame: Optional[refs.PolicyRef] = None
  endgame_route: str = ""
  opening_plies: int = 0
  first_hand_only: bool = False
  gamma: float = 1.0
  value_target: str = "return"
  return_lookahead: bool = True
  heldout_states: int = 10_000
  heldout_seed: int = 999
  heldout_every: int = 0
  heldout_num_envs: int = 0
  data: str = "online"
  kind: str = "bc"

  def __post_init__(self):
    if self.kind != "bc":
      raise ValueError(f"BC.kind must be 'bc', got {self.kind!r}")
    for name, typ in (
        ("learner", bc.BCConfig),
        ("teacher", refs.PolicyRef),
        ("beta", schedules.Schedule),
    ):
      if not isinstance(getattr(self, name), typ):
        raise TypeError(
            f"BC.{name} must be a {typ.__name__}, got "
            f"{type(getattr(self, name)).__name__}"
        )
    if not (0.0 <= self.beta.start <= 1.0 and 0.0 <= self.beta.end <= 1.0):
      raise ValueError(f"BC.beta must stay in [0, 1], got {self.beta}")
    if self.data not in BC_DATA:
      raise ValueError(f"BC.data must be one of {BC_DATA}, got {self.data!r}")
    if self.heldout_states < 1:
      raise ValueError(
          f"BC.heldout_states must be >= 1, got {self.heldout_states}"
      )
    if self.heldout_every < 0 or self.heldout_num_envs < 0:
      raise ValueError(
          "BC.heldout_every and heldout_num_envs must be >= 0, got "
          f"{self.heldout_every}, {self.heldout_num_envs}"
      )
    if (self.endgame is None) != (not self.endgame_route):
      raise ValueError("BC.endgame and BC.endgame_route go together")
    if not (
        self.student_init is None
        or isinstance(self.student_init, refs.PolicyRef)
    ):
      raise TypeError(
          "BC.student_init must be None or a PolicyRef, got "
          f"{type(self.student_init).__name__}"
      )

  def constant_lr(self) -> bool:
    """Whether the step size is independent of the run length.

    Returns:
      True when the learner's schedule is constant.
    """
    return self.learner.schedule().is_constant()

  def horizon(self) -> int:
    """The run length in updates: total_steps over steps per update.

    Returns:
      The run length.
    """
    return int(self.learner.total_steps) // (
        self.learner.num_epochs * self.learner.num_minibatches
    )

  def fixed_horizon(self) -> Optional[int]:
    """The run length the step-size schedule is built for (None: constant).

    Returns:
      The run length, or None.
    """
    return None if self.constant_lr() else self.horizon()

  def with_run_updates(self, n: int) -> "BC":
    """This spec with the run length set to n updates.

    Args:
      n: the run length in updates.

    Returns:
      The extended spec (unchanged when the schedule is constant).
    """
    if self.constant_lr():
      return self
    total = n * self.learner.num_epochs * self.learner.num_minibatches
    return dataclasses.replace(
        self, learner=dataclasses.replace(self.learner, total_steps=total)
    )

  def validate_run(self, run: Any) -> None:
    """Checks this spec against its LeagueRunConfig.

    Args:
      run: the run config.

    Raises:
      ValueError: the run names no network in policy_kwargs.
    """
    if not run.policy_kwargs:
      raise ValueError("a bc run names its network in policy_kwargs")


class BCKind(base.KindBase):
  """The behaviour-cloning learner kind."""

  name = "bc"
  spec_cls = BC

  def run_length_paths(self) -> tuple[str, ...]:
    """The run-length field: learner.total_steps.

    Returns:
      The paths.
    """
    return ("learner.total_steps",)

  def add_flags(self, flag_values: Any) -> None:
    """Defines the BC flags (data_dir, record_shards, record_batches).

    Args:
      flag_values: the absl FlagValues to define the flags on.
    """
    flags.DEFINE_string(
        "data_dir",
        None,
        "BC configs: learn from these shards (recorded with --record_shards).",
        flag_values=flag_values,
    )
    flags.DEFINE_string(
        "record_shards",
        None,
        "BC configs: record the teacher's games of the "
        "run's first --record_batches updates to this dir instead of training.",
        flag_values=flag_values,
    )
    flags.DEFINE_integer(
        "record_batches",
        None,
        "Batches to record (--record_shards).",
        flag_values=flag_values,
    )

  def prepare(
      self, run_cfg: Any, flag_values: Any
  ) -> Optional[tuple[Any, Mapping[str, Any]]]:
    """Applies the BC flags to a run.

    --record_shards records the teacher's games and trains nothing;
    --data_dir trains from those shards (the spec's data becomes "shards").

    Args:
      run_cfg: the run config.
      flag_values: the parsed absl FlagValues.

    Returns:
      (the run config to train, extra keyword arguments of
      `league.runner.run`), or None after recording shards.

    Raises:
      ValueError: --record_shards without --record_batches.
    """
    if flag_values["record_shards"].value:
      if not flag_values["record_batches"].value:
        raise ValueError("--record_shards needs --record_batches")
      # Lazy: cycle: runner imports league.kinds.
      from open_spiel.python.scale.league import runner  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      runner.record(
          run_cfg,
          flag_values["record_shards"].value,
          flag_values["record_batches"].value,
      )
      return None
    data_dir = flag_values["data_dir"].value
    if data_dir is None:
      return run_cfg, {}
    if run_cfg.learner.data != "shards":
      run_cfg = dataclasses.replace(
          run_cfg, learner=dataclasses.replace(run_cfg.learner, data="shards")
      )
    return run_cfg, {"data_dir": data_dir}

  def build_driver(
      self,
      cfg: BC,
      env: types.Env,
      *,
      policy_arch: str = "network",
      policy_kwargs: Optional[Mapping[str, Any]] = None,
      data_dir: Optional[str] = None,
      scratch_dir: Optional[str] = None,
  ) -> "BCDriver":
    """The BCDriver of a spec over `env`.

    Args:
      cfg: the BC spec.
      env: the league's env.
      policy_arch: the registered architecture of the student's policy.
      policy_kwargs: the architecture's kwargs.
      data_dir: recorded shards to train from, or None for online BC.
      scratch_dir: unused.

    Returns:
      The BCDriver.
    """
    del scratch_dir
    return BCDriver(
        cfg,
        env,
        dict(policy_kwargs or {}),
        data_dir=data_dir,
        policy_arch=policy_arch,
    )


class BCDriver(base.Driver):
  """The BC kind's driver (`sources.dagger` or `.shards`, and `learners.bc`).

  Attributes:
    spec: the BC kind spec.
    env: the game's Env.
    data_dir: the run's data directory (shards).
    policy: the student network.
    slot_policy: the policy the learner slot plays (the student).
    teacher: the teacher policy.
    teacher_params: the teacher's params.
    learner: the BCLearner.
    endgame: None, or the endgame route (sources.dagger.EndgameRoute).
    league: the League (set by attach).
    mesh: the device mesh (set by attach).
    dp: the DataParallel over the mesh (set by attach).
    arena: None: each DaggerSource owns its Arena.
  """

  kind = "bc"
  algo = "bc"

  def __init__(
      self,
      spec: BC,
      env: types.Env,
      student_kwargs: Mapping[str, Any],
      data_dir: Optional[str] = None,
      policy_arch: str = "network",
  ):
    assert isinstance(spec, BC), type(spec)
    if (spec.data == "shards") != (data_dir is not None):
      raise ValueError(f"BC data={spec.data!r} but data_dir={data_dir!r}")
    self.spec, self.env, self.data_dir = spec, env, data_dir
    self.policy = registry.build(policy_arch, dict(student_kwargs))
    self.slot_policy = self.policy
    self.teacher, self.teacher_params = spec.teacher.build()
    assert self.teacher.spec.game == self.policy.spec.game == env.name, (
        self.teacher.spec.game,
        self.policy.spec.game,
        env.name,
    )
    self.learner = bc.BCLearner(self.policy, spec.learner)
    self.endgame = None
    if spec.endgame is not None:
      pol, params = spec.endgame.build()
      self.endgame = dagger.EndgameRoute(
          policy=pol, params=params, route=spec.endgame_route
      )
    self._sources: dict[str, Any] = {}
    self._owner: Optional[str] = None
    self._iteration: dict[str, int] = {}  # restored DAgger segment counters
    # restored DAgger source snapshots (sources_state)
    self._restored: dict[str, Any] = {}
    self._heldout = None

  def wrap_opponent(self, policy, params):
    """How a fixed opponent plays (routed with the scripted endgame).

    Args:
      policy: the opponent's policy.
      params: the opponent's params.

    Returns:
      (policy, params) as played.
    """
    if self.endgame is None:
      return policy, params

    r = routed.RoutedPolicy(policy, self.endgame.policy, self.endgame.route)
    return r, r.make_params(params, self.endgame.params)

  def attach(self, league, mesh) -> None:
    """Keeps the league and its mesh; each DaggerSource owns its Arena.

    Args:
      league: the League.
      mesh: the device mesh, or None for the batch mesh.
    """
    self.league = league
    self.mesh = data_parallel.batch_mesh() if mesh is None else mesh
    self.dp = data_parallel.DataParallel(self.mesh)
    self.arena = None  # each DaggerSource owns its Arena

  def init_state(self, key, env):
    """Fresh params from key, or student_init's (warm start, fresh optimizer).

    Args:
      key: the init key.
      env: the league's env.

    Returns:
      The learner state.
    """
    if self.spec.student_init is None:
      return self.learner.init(key, env)
    pol, params = self.spec.student_init.build()
    assert pol.spec == self.policy.spec, (pol.spec, self.policy.spec)
    return self.learner.init_from_params(params)

  def artifact_extra(self) -> dict[str, Any]:
    """The warm start's lineage, if any.

    Returns:
      The extra artifact fields.
    """
    si = self.spec.student_init
    return {} if si is None else si.lineage()

  def dagger_source(
      self,
      table,
      opponent_params,
      student_params,
      *,
      beta=None,
      value_target=None,
  ):
    """The DaggerSource over the league's opponent slots.

    The training one, or with beta 1 and value_target "teacher" the held-out
    set's.

    Args:
      table: the slot table.
      opponent_params: the opponent slots' params.
      student_params: the student's params.
      beta: the teacher-mixing schedule; the spec's when None.
      value_target: the value target; the spec's when None.

    Returns:
      The DaggerSource.
    """
    s = self.spec
    vt = value_target or s.value_target
    return dagger.DaggerSource(
        self.env,
        self.policy,
        self.teacher,
        student_params=student_params,
        teacher_params=self.teacher_params,
        table=table,
        num_steps=self.league.config.num_steps,
        beta=s.beta if beta is None else beta,
        opponents=self.league.opponent_slots(),
        opponent_params=list(opponent_params),
        rule=self.league.config.learner_rule,
        gamma=s.gamma,
        value_target=vt,
        return_lookahead=s.return_lookahead and vt == "return",
        opening_plies=s.opening_plies,
        first_hand_only=s.first_hand_only,
        endgame=self.endgame,
        mesh=self.mesh,
    )

  def _shards(self, ls):
    """The learner's shard source, skipped past the updates it has made.

    Args:
      ls: The learner slot.

    Returns:
      The SequentialSource.
    """
    carry_like = None
    if self.policy.spec.stateful:
      carry_like = self.policy.init_carry(
          self.learner.params(ls.state),
          1,
          jnp.zeros((1,), jnp.int32),
          jax.random.key(0),
      )
    # A stored carry is reused only when it is this student network's own
    # (recorded from its DAgger run); otherwise the rows must start at hand
    # boundaries and the student starts from its own init_carry.
    reader = shards.ShardReader(
        self.data_dir,
        self.env,
        rules_sha=common.rules_sha(self.env),
        info_schema_version=INFO_SCHEMA_VERSION,
        init_carry_like=carry_like,
        student=self.policy.spec,
    )
    src = shards.SequentialSource(reader)
    src.skip(int(ls.updates))
    return src

  def train_step(self, ls, table, opponent_params, key, fresh):
    """One update of learner slot `ls`.

    Args:
      ls: the learner slot.
      table: the drawn slot table.
      opponent_params: the opponent slots' params, in slot order.
      key: the update's key.
      fresh: whether to start a fresh trajectory source.

    Returns:
      (state, metrics, frames, decisions).

    Raises:
      NotImplementedError: a fresh table after the first update (a BC slot's
        opponent table is fixed for the run).
    """
    self._single(ls)
    k_roll, k_upd = jax.random.split(key)
    src = self._sources.get(ls.name)
    params = self.learner.params(ls.state)
    if src is None:
      if self.data_dir is not None:
        src = self._shards(ls)
      else:
        src = self.dagger_source(table, opponent_params, params)
        if ls.name in self._iteration:
          src.iteration = np.int64(self._iteration.pop(ls.name))
        if ls.name in self._restored:
          self._restore_dagger(src, self._restored.pop(ls.name))
      self._sources[ls.name] = src
    elif fresh:
      raise NotImplementedError(
          "a BC slot's opponent table is fixed for the run (a fixed "
          "sampler rule with redraw='on_change')"
      )
    if self.data_dir is None:
      src.opponent_params = list(opponent_params)
      src.set_student_params(params)
    f0, d0 = src.frames, src.decisions
    batch = src.next(k_roll)
    state, m = self.dp.update(self.learner, ls.state, batch, k_upd)
    m = {k: float(v) for k, v in m.items()}
    if self.data_dir is None:
      m["beta"] = float(self.spec.beta.value(np.int64(ls.updates)))
    return state, m, src.frames - f0, src.decisions - d0

  def heldout_metrics(self, ls) -> dict[str, Any]:
    """Held-out metrics of ls's student on heldout_set(ls).

    Args:
      ls: the learner slot.

    Returns:
      heldout/{top1, kl, ce, teacher_entropy, num_states}.
    """
    out = heldout_lib.score(
        self.policy, self.learner.params(ls.state), self.heldout_set(ls)
    )
    return {f"heldout/{k}": v for k, v in out.items()}

  def heldout_set(self, ls):
    """The fixed teacher-driven held-out state set (`eval.heldout.HeldoutSet`).

    Built once: beta 1, the run's opening, episodes and opponents (ls's mix),
    heldout_num_envs rows, keys from key(heldout_seed). Independent of the
    training seed and of the student.

    Args:
      ls: the learner slot.

    Returns:
      The HeldoutSet.
    """
    if self._heldout is None:
      lg, s = self.league, self.spec
      n = s.heldout_num_envs or lg.config.num_envs
      k_held = jax.random.key(s.heldout_seed)
      p = self.env.num_players
      if lg.config.self_play_seats == "all":
        assert ls.mix.self_play == 1.0, ls.mix
        table = arena_lib.self_play_table(n, p, all_seats=True)
      else:
        table = sampler.sample(
            sampler.LeagueSlots(
                num_players=p,
                batch=n,
                pools=lg.all_pools(),
                mix=ls.mix,
                stratified=lg.config.stratified,
            ),
            jax.random.fold_in(k_held, 1),
        ).table
      src = self.dagger_source(
          table,
          [pool.params for pool in lg.all_pools()],
          self.learner.params(ls.state),
          beta=schedules.constant(1.0),
          value_target="teacher",
      )
      self._heldout = heldout_lib.collect(src, k_held, s.heldout_states)
    return self._heldout

  def host_state(self) -> dict[str, Any]:
    """Each DaggerSource's iteration counter.

    Returns:
      {"dagger_iteration": {learner: iteration}}, or {} when there is none.
    """
    its = {
        n: int(src.iteration)
        for n, src in self._sources.items()
        if hasattr(src, "iteration")
    }
    return {"dagger_iteration": its} if its else {}

  def sources_state(self) -> dict[str, Any]:
    """Online BC: each DaggerSource's snapshot.

    The inner ArenaSource (the hands in flight), its decision counter and its
    lookahead segment, so a resumed run rolls the same segments as an
    uninterrupted one. Shard-fed sources resume by position
    (SequentialSource.skip) and are not snapshotted.

    Returns:
      The snapshots by learner name.
    """
    out = {}
    for name, src in self._sources.items():
      if isinstance(src, dagger.DaggerSource):
        snap = commit.snapshot_source(src.inner)
        snap["dagger"] = dict(
            decisions=np.int64(src.decisions),
            pending=None
            if src.pending is None
            else commit.host_tree(src.pending),
        )
        out[name] = snap
    return out

  def load_sources_state(self, d: Mapping[str, Any]) -> None:
    """Restores the snapshots `sources_state()` returned.

    Args:
      d: the snapshots by learner name.

    Raises:
      ValueError: a learner already has a source, or a snapshot lacks its
        DAgger state.
    """
    for name, snap in d.items():
      if name in self._sources or name in self._restored:
        raise ValueError(f"learner {name} already has a source")
      if "dagger" not in snap:
        raise ValueError(
            f"learner {name}: a BC source snapshot carries its DAgger state"
        )
      self._restored[name] = snap

  def _restore_dagger(self, src, snap) -> None:
    inner = commit.restore_source(src.arena, snap)
    if inner.num_steps != src.inner.num_steps:
      raise ValueError(
          f"restored DAgger source has {inner.num_steps} plies per segment, "
          f"the run {src.inner.num_steps}"
      )
    src.inner = inner
    src.decisions = np.int64(snap["dagger"]["decisions"])
    pending = snap["dagger"]["pending"]
    src.pending = None if pending is None else commit.device_tree(pending)

  def load_host_state(self, d: Mapping[str, Any]) -> None:
    """Restores the DaggerSource iteration counters.

    Args:
      d: the saved host state.
    """
    self._iteration = dict(d.get("dagger_iteration", {}))

  def record(
      self, ls, table, opponent_params, out_dir: str, num_batches: int, key_of
  ) -> str:
    """Writes the teacher's games of updates 0..num_batches-1 to shards.

    Uses the training source and the run's rollout keys (key_of(g): the
    league's key of update g), so a data="shards" run over out_dir replays
    the online run's batches. Only a teacher-driven spec (beta 1 throughout)
    records.

    Returns:
      the teacher's id, which the shards record as their producer.

    Args:
      ls: the learner slot.
      table: the slot table.
      opponent_params: the opponent slots' params.
      out_dir: the shards' directory.
      num_batches: the number of updates recorded.
      key_of: g -> the league's key of update g.

    Raises:
      ValueError: the spec is not teacher-driven.
    """
    assert num_batches >= 1, num_batches
    b = self.spec.beta
    if not (b.is_constant() and b.start == 1.0):
      raise ValueError(
          f"record needs a teacher-driven config (beta 1), got {b}"
      )
    src = self.dagger_source(
        table, opponent_params, self.learner.params(ls.state)
    )
    # The DAgger carry is the student's.
    with shards.ShardWriter(
        out_dir,
        self.env,
        rules_sha=common.rules_sha(self.env),
        info_schema_version=INFO_SCHEMA_VERSION,
        producer=self.spec.teacher.id,
        carry_owner=self.policy.spec,
    ) as w:
      for g in range(num_batches):
        k_roll, _ = jax.random.split(key_of(g))
        w.write(src.next(k_roll))
    logging.info(
        "bc: recorded %d batches of %s to %s",
        num_batches,
        self.spec.teacher.id,
        out_dir,
    )
    return self.spec.teacher.id


KIND = BCKind()
base.register_kind(KIND)
