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

"""The policy-gradient learner kind.

The policy-gradient learner kind: on-policy rollouts in the Arena, one
`learners.pg.PGLearner` update per batch.

    spec = kinds.PG(learner=pg.PGConfig(loss="ppo", num_updates=1000, ...))

The learner slot plays its network (or the network with the game's scripted
final phase, `scripted_endgame`) against the opponents the league's sampler
draws. The PG kind is the only kind that trains several learners at once
(PSRO best responses, PBT populations).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Mapping, Optional

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import hash_compat
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league.kinds import base
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.parallel import data_parallel
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.sources import decisions


def endgames() -> dict[str, Any]:
  """{scripted final-phase architecture: the game it plays}.

  Read from the game table.

  Returns:
    The mapping.
  """
  return env_registry.by_field("endgame")


def scripted_routes(endgame: str) -> tuple[str, ...]:
  """The routes the scripted final phase `endgame` may be played on.

  Args:
    endgame: a scripted final-phase architecture name.

  Returns:
    Its declared "module:attr" routes (policies.routed).
  """
  return routed.phase_routes(registry.build(endgame, {}))


@dataclasses.dataclass(frozen=True)
class PG:
  """The policy-gradient learner (loss "ppo", "mmd" or a registered variant).

  Attributes:
    learner: the PGConfig.
    scripted_endgame: a scripted final-phase architecture of the run's game
      (`GameSpec.endgame`). Every slot (learner, pool, fixed opponents) plays
      RoutedPolicy(policy, endgame), the published artifact is that routed
      policy, and the learner trains on its own network's decisions only.
    scripted_route: the "module:attr" route the scripted endgame is played on;
      "" (the default) is the endgame's own route. Any other value must be a
      route the endgame declares (its class's ROUTES), e.g. gin's
      AGENT_KNOCK_ROUTE, which leaves the knocker's discard to the network
      (and so to the learner). Left out of the config hash at "".
    init: None (fresh params from the run's key) or a PolicyRef whose params
      warm-start the learner with a fresh optimizer. Its policy (the base of a
      routed artifact) must have the learner policy's spec.
    parents: the artifact ids the run's artifacts name as parents.
    kind: the kind's registered name.
    teacher: None (the default; left out of the config hash) or a PolicyRef
      of a frozen distillation teacher (learners.pg module doc "Teacher
      distillation"; its policy, the base of a routed artifact, plays the
      learner's game with its action and player counts, its spec its own).
      It needs learner.teacher_coef. It is never the warm start, a pool
      member, an opponent or the magnet.
  """

  learner: pg.PGConfig
  scripted_endgame: str = ""
  init: Optional[refs.PolicyRef] = None
  parents: tuple[Any, ...] = ()
  kind: str = "pg"
  scripted_route: str = hash_compat.added("")
  teacher: Optional[refs.PolicyRef] = hash_compat.added(None)

  def __post_init__(self):
    assert self.kind == "pg", self.kind
    assert isinstance(self.learner, pg.PGConfig), type(self.learner)
    assert self.init is None or isinstance(self.init, refs.PolicyRef), type(
        self.init
    )
    assert all(
        isinstance(p, str) and "@" in p for p in self.parents
    ), self.parents
    if self.parents and self.init is None:
      raise ValueError("parents name the warm start: set init")
    if self.teacher is not None:
      if not isinstance(self.teacher, refs.PolicyRef):
        raise ValueError(f"teacher is a PolicyRef, got {type(self.teacher)}")
      if not self.learner.teacher_on():
        raise ValueError(
            "a teacher needs learner.teacher_coef (not constant 0)"
        )
      if self.init is not None and self.init.id == self.teacher.id:
        raise ValueError(
            f"teacher {self.teacher.id} is also the warm start (init): a "
            "kickstarted run starts from fresh params"
        )
    elif self.learner.teacher_on():
      raise ValueError(
          "learner.teacher_coef distils from a teacher: set teacher"
      )
    if self.scripted_endgame:
      table = endgames()
      assert self.scripted_endgame in table, (
          self.scripted_endgame,
          sorted(table),
      )
    if self.scripted_route:
      if not self.scripted_endgame:
        raise ValueError("scripted_route needs a scripted_endgame")
      declared = scripted_routes(self.scripted_endgame)
      if self.scripted_route not in declared:
        raise ValueError(
            f"scripted_route {self.scripted_route!r} is not a route of "
            f"{self.scripted_endgame!r}: {list(declared)}"
        )

  def constant_lr(self) -> bool:
    """Whether the step size is independent of num_updates.

    A constant schedule, or an lr_sched that is a function of the train step
    alone, so the run may be extended.

    Returns:
      True when it is.
    """
    return (
        self.learner.lr_sched is not None
        or self.learner.lr_schedule().is_constant()
    )

  def horizon(self) -> int:
    """The run length in updates.

    Returns:
      learner.num_updates.

    Raises:
      ValueError: num_updates < 1.
    """
    if int(self.learner.num_updates) < 1:
      raise ValueError(
          "a PG run needs learner.num_updates >= 1 (its run length)"
      )
    return int(self.learner.num_updates)

  def fixed_horizon(self) -> int:
    """The run length the learner's schedules are built for.

    Returns:
      `horizon()`.
    """
    return self.horizon()

  def with_run_updates(self, n: int) -> "PG":
    """This spec with the run length set to n updates.

    Args:
      n: the run length in updates.

    Returns:
      The extended spec.
    """
    return dataclasses.replace(
        self, learner=dataclasses.replace(self.learner, num_updates=n)
    )

  def validate_run(self, run: Any) -> None:
    """Checks this spec against its LeagueRunConfig.

    Args:
      run: the run config.

    Raises:
      ValueError: the spec and the run disagree.
    """
    if not run.policy_kwargs:
      raise ValueError("a pg run names its network in policy_kwargs")
    if self.scripted_endgame and endgames()[self.scripted_endgame] != run.env:
      raise ValueError(
          f"scripted_endgame {self.scripted_endgame!r} plays "
          f"{endgames()[self.scripted_endgame]!r}, not {run.env!r}"
      )
    if (
        pg.extension(self.learner.loss) is not None
        and run.league.learner_rule.temperature != 1.0
    ):
      raise ValueError(
          f"{self.learner.loss} recomputes the behaviour distribution as "
          "softmax(behavior_logits) (T = 1); a learner_rule temperature of "
          f"{run.league.learner_rule.temperature} would make its ratios wrong"
      )


class PGKind(base.KindBase):
  """The policy-gradient learner kind."""

  name = "pg"
  spec_cls = PG
  multi_learner = True

  def run_length_paths(self) -> tuple[str, ...]:
    """The run-length field: learner.num_updates.

    Returns:
      The paths.
    """
    return ("learner.num_updates",)

  def build_driver(
      self,
      cfg: PG,
      env: types.Env,
      *,
      policy_arch: str = "network",
      policy_kwargs: Optional[Mapping[str, Any]] = None,
      data_dir: Optional[str] = None,
      scratch_dir: Optional[str] = None,
  ) -> "PGDriver":
    """The PGDriver of a spec over `env`.

    Args:
      cfg: the PG spec.
      env: the league's env.
      policy_arch: the registered architecture of the learner's policy.
      policy_kwargs: the architecture's kwargs.
      data_dir: must be None (a BC option).
      scratch_dir: unused.

    Returns:
      The PGDriver.

    Raises:
      ValueError: data_dir is given.
    """
    del scratch_dir
    if data_dir is not None:
      raise ValueError("data_dir is a BC option")
    policy = registry.build(policy_arch, dict(policy_kwargs or {}))
    return PGDriver(
        policy,
        pg.PGLearner(policy, cfg.learner, env_max_len=env.max_len),
        cfg.scripted_endgame,
        init=cfg.init,
        parents=cfg.parents,
        env=env,
        scripted_route=cfg.scripted_route,
        teacher=cfg.teacher,
    )


class PGDriver(base.Driver):
  """The PG kind's driver; also what a League built from a bare learner uses.

  Attributes:
    policy: the learner's network.
    learner: the PG learner.
    slot_policy: the policy the learner slot plays (routed when scripted).
    algo: the `algo` recorded in member artifacts.
    num_steps: plies per update (set by attach).
    arena: the Arena (set by attach).
    mesh: the Arena's device mesh (set by attach).
    dp: the DataParallel over the mesh (set by attach).
  """

  kind = "pg"

  def __init__(
      self,
      policy: types.Policy,
      learner: Any,
      scripted_endgame: str = "",
      init: Optional[refs.PolicyRef] = None,
      parents: tuple[Any, ...] = (),
      env: Optional[types.Env] = None,
      scripted_route: str = "",
      teacher: Optional[refs.PolicyRef] = None,
  ):
    if scripted_route and not scripted_endgame:
      raise ValueError("scripted_route needs a scripted_endgame")
    self.policy, self.learner = policy, learner
    # The distillation teacher: built on first use (_ensure_teacher), so a
    # driver that never trains never reads its artifact.
    self._teacher_ref = teacher
    self._teacher_built = False
    self._route = scripted_route or None
    self._init, self._parents = init, tuple(parents)
    self._routed = None
    self._default_routed = None
    self._game = None
    if scripted_endgame:
      self._game = env_registry.game(
          policy.spec.game if env is None else env.name
      )
      if self._game.endgame != scripted_endgame:
        raise ValueError(
            f"scripted_endgame {scripted_endgame!r} is not "
            f"{self._game.name}'s ({self._game.endgame!r})"
        )
      self._routed = routed.with_scripted_phase(
          policy, self._game.endgame, self._route
      )
      if self._route is not None:
        # The endgame on its own route: the rows it would script that this
        # run's route leaves to the learner are counted per update.
        self._default_routed = routed.with_scripted_phase(
            policy, self._game.endgame
        )
      assert not jax.tree_util.tree_leaves(
          self._routed.override.init_params()
      ), "the scripted endgame has params"
    self.slot_policy = self._routed or policy
    loss = getattr(getattr(learner, "cfg", None), "loss", "")
    self.algo = f"pg.{loss}" if loss else "pg"
    self._sources: dict[str, arena_lib.ArenaSource] = {}

  def wrap(self, params):
    """Learner params -> slot params (routed with the scripted endgame).

    Args:
      params: the learner params.

    Returns:
      The slot params.
    """
    return (
        params if self._routed is None else self._routed.make_params(params, {})
    )

  def unwrap(self, slot_params):
    """Slot params -> learner params.

    Args:
      slot_params: the slot params.

    Returns:
      The learner params.
    """
    return slot_params if self._routed is None else slot_params["base"]

  def init_state(self, key, env):
    """Fresh params from key, or the spec's init (warm start, fresh optimizer).

    Args:
      key: the init key.
      env: the league's env.

    Returns:
      The learner state.

    Raises:
      ValueError: the init's policy is not the learner's.
    """
    self._ensure_teacher(check_only=True)
    if self._init is None:
      return self.learner.init(key, env)
    pol, params = routed.unwrap_routed(*self._init.build())
    if pol.spec != self.policy.spec:
      raise ValueError(
          f"init {self._init.id}: its policy {pol.spec.arch_name} "
          f"{dict(pol.spec.arch_kwargs)} is not the learner's "
          f"{self.policy.spec.arch_name} {dict(self.policy.spec.arch_kwargs)}"
      )
    logging.info(
        "PGDriver: warm start from %s (fresh optimizer)", self._init.id
    )
    return self.learner.init_from_params(params)

  def artifact_extra(self) -> dict[str, Any]:
    """The warm start's lineage and the value head's units.

    The value is points / PGConfig.value_scale.

    Returns:
      The extra artifact fields.
    """
    extra = {}
    vs = getattr(getattr(self.learner, "cfg", None), "value_scale", None)
    if vs is not None:
      extra["extensions"] = {"value_scale": float(vs)}
    if self._init is None:
      return extra
    if self._parents:
      return dict(extra, **base.lineage_extra(self._init.id, self._parents))
    return dict(extra, **self._init.lineage())

  def host_state(self) -> dict[str, Any]:
    """The teacher's id when the run distils from one (else nothing).

    The checkpoint records it (the league's driver state); member artifacts do
    not (the run config's metadata names it), so their format is unchanged.

    Returns:
      {"teacher": id} or {}.
    """
    if self._teacher_ref is None:
      return {}
    return {"teacher": self._teacher_ref.id}

  def load_host_state(self, d: Mapping[str, Any]) -> None:
    """Checks a resumed run's teacher is this config's.

    Args:
      d: the saved host state.

    Raises:
      ValueError: the checkpoint's teacher is not the config's.
    """
    want = self.host_state()
    if dict(d) != want:
      raise ValueError(
          f"resume: the checkpoint's driver state {dict(d)} is not this "
          f"config's {want} (a run's teacher cannot change)"
      )

  def _ensure_teacher(self, check_only: bool = False) -> None:
    """Builds the teacher once and hands it to the learner (set_teacher).

    The teacher is the base of a routed artifact (its own scripted endgame
    dropped: the learner trains its network decisions only), frozen, outside
    the pool and the magnet. Its params are replicated on the mesh.

    Args:
      check_only: build and check it, but do not set it (before attach).

    Raises:
      ValueError: the teacher does not fit the learner, or attach has not run.
    """
    if self._teacher_ref is None or self._teacher_built:
      return
    pol, params = routed.unwrap_routed(*self._teacher_ref.build())
    self._check_teacher(pol)
    if check_only:
      return
    if getattr(self, "dp", None) is None:
      raise ValueError("the teacher is placed on the mesh: attach first")
    self.learner.set_teacher(
        pol, self.dp.replicate(params), name=self._teacher_ref.id
    )
    self._teacher_built = True
    logging.info(
        "PGDriver: teacher %s (frozen; not in the pool, not the magnet)",
        self._teacher_ref.id,
    )

  def _check_teacher(self, pol) -> None:
    """The teacher's game, action and player counts are the learner's.

    Args:
      pol: the teacher policy.

    Raises:
      ValueError: they differ.
    """
    ts, ls = pol.spec, self.policy.spec
    if (ts.game, ts.num_actions, ts.num_players) != (
        ls.game,
        ls.num_actions,
        ls.num_players,
    ):
      raise ValueError(
          f"teacher {self._teacher_ref.id}: {ts.game} / {ts.num_actions} "
          f"actions / {ts.num_players} players, the learner {ls.game} / "
          f"{ls.num_actions} / {ls.num_players}"
      )

  def wrap_opponent(self, policy, params):
    """How a fixed opponent plays (routed with the scripted endgame).

    Args:
      policy: the opponent's policy.
      params: the opponent's params.

    Returns:
      (policy, params) as played.
    """
    if self._routed is None:
      return policy, params
    if routed.is_scripted_phase(policy, self._game.endgame):
      return policy, params  # already routed: plays the route it has
    return routed.routed_pair(policy, params, self._game.endgame, self._route)

  def attach(self, league, mesh) -> None:
    """Builds the Arena over league.opponent_slots().

    The Arena emits the reward credit of the learner's PGConfig.credit
    (core.arena "Credit"; seat credit for a learner without one).

    Args:
      league: the League.
      mesh: the device mesh.

    Raises:
      ValueError: the policy is not an in-graph JAX policy.
    """
    spec = self.policy.spec
    if spec.capability not in arena_lib.IN_GRAPH or spec.framework != "jax":
      raise ValueError(
          "the PG kind trains in-graph JAX policies (a host learner is a "
          "registered host kind)"
      )
    self.num_steps = league.config.num_steps
    self.arena = self._arena(
        league,
        mesh,
        arena_lib.Slot(self.slot_policy, 1, league.config.learner_rule),
        credit=getattr(getattr(self.learner, "cfg", None), "credit", "seat"),
    )
    self.mesh = self.arena.mesh
    self.dp = data_parallel.DataParallel(self.mesh)

  def _source(self, ls, params, table, fresh):
    """The learner's arena source, rebuilt when `fresh`.

    Args:
      ls: The learner slot.
      params: The slot params.
      table: The SlotTable.
      fresh: Whether the draw changed.

    Returns:
      The ArenaSource.
    """
    src = self._sources.get(ls.name)
    if src is None or fresh:
      src = arena_lib.ArenaSource(self.arena, params, table, self.num_steps)
      self._sources[ls.name] = src
    else:
      src.set_params(params)
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
    """
    self.check_table(table)
    self._ensure_teacher()
    k_roll, k_upd = jax.random.split(key)
    params = [
        base.stack1(self.wrap(self.learner.params(ls.state))),
        *opponent_params,
    ]
    src = self._source(ls, params, table, fresh)
    f0, d0 = src.frames, src.decisions
    # Phase timers (wall clock, metrics stream only: runner.is_timing_key):
    # rollout_seconds is the batch's play and preparation, learn_seconds the
    # update; each phase is synced before its clock stops.
    t0 = time.perf_counter()
    batch = src.next(k_roll)
    extra: dict[str, Any] = {}
    if self._routed is not None:
      n_before = int(jnp.sum(batch.valid, dtype=jnp.int64))
      batch, n_routed = decisions.exclude_override(self._routed, batch)
      # The source counts the decisions the learner trains on.
      src.decisions = np.int64(src.decisions - n_routed)
      extra = dict(routed_excluded=int(n_routed), batch_decisions=n_before)
      if self._default_routed is not None:
        # Learner decisions the endgame's own route would have scripted
        # (gin's AGENT_KNOCK_ROUTE: the knocker's discards).
        own = np.asarray(
            routed.override_rows(self._default_routed, batch.steps.info)
        )
        extra["agent_endgame_decisions"] = int(
            (np.asarray(batch.valid) & own).sum(dtype=np.int64)
        )
    jax.block_until_ready(batch)
    t1 = time.perf_counter()
    state, m = self.dp.update(self.learner, ls.state, batch, k_upd)
    jax.block_until_ready(m)
    extra.update(
        rollout_seconds=t1 - t0, learn_seconds=time.perf_counter() - t1
    )
    return state, dict(m, **extra), src.frames - f0, src.decisions - d0


KIND = PGKind()
base.register_kind(KIND)
