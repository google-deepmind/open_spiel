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

"""The AlphaZero learner kind, and the driver it shares with expert iteration.

    spec = kinds.AZ(az=az.AZConfig(env_name="connect_four", ...))

One update is one AZ iteration: every Arena row plays one fresh game with a
`SearchPolicy` over the network, the plies get the AZ value and policy
targets (`learners.az.outcome_targets`), they enter the `PlyReplay`, and the
`BCLearner` trains one replay epoch. The AZConfig's run-level fields must
agree with the league run's (env_name = env, batch_size = league.num_envs,
seed; `expert_iteration.check_az_run` raises otherwise), and its
num_iterations is the run's length in updates.

`AZDriver` also drives `kinds.ExIt`: AZ is ExIt with the default operator and
targets, and the driver reads its options from an ExIt spec (`options`).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Mapping, Optional

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import slot_factories
from open_spiel.python.scale.league.kinds import base
from open_spiel.python.scale.league.kinds import expert_iteration
from open_spiel.python.scale.learners import az as az_lib
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.parallel import data_parallel
from open_spiel.python.scale.policies import operators
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.sources import decisions
from open_spiel.python.scale.sources import targets as targets_lib

tree_map = jax.tree_util.tree_map


@dataclasses.dataclass(frozen=True)
class AZ:
  """AlphaZero over `az` (module doc); checkpoints follow keep_every.

  Attributes:
    az: the learners.az.AZConfig.
    kind: the kind's registered name.
  """

  az: Any  # learners.az.AZConfig
  kind: str = "az"

  def __post_init__(self):
    assert self.kind == "az", self.kind
    assert isinstance(self.az, az_lib.AZConfig), type(self.az)

  def constant_lr(self) -> bool:
    """Whether the step size is independent of the run length.

    Returns:
      True when the AZConfig's learning rate does not anneal.
    """
    return self.az.lr_end == self.az.lr

  def horizon(self) -> int:
    """The run length in updates.

    Returns:
      The AZConfig's num_iterations.
    """
    return int(self.az.num_iterations)

  def fixed_horizon(self) -> int:
    """The run length the schedule is fixed to.

    Returns:
      `horizon()`.
    """
    return self.horizon()

  def with_run_updates(self, n: int) -> "AZ":
    """This spec with the run length set to n updates.

    Args:
      n: the run length in updates.

    Returns:
      The extended spec.
    """
    return dataclasses.replace(
        self, az=dataclasses.replace(self.az, num_iterations=n)
    )

  def validate_run(self, run: Any) -> None:
    """Checks this spec against the run config.

    Args:
      run: the run config.

    Raises:
      ValueError: the spec and the run disagree.
    """
    expert_iteration.check_az_run(self.az, run)
    if run.league.self_play_seats != "all":
      raise ValueError(
          "AZ self-play stores every seat's decisions: self_play_seats='all'"
      )

  def as_expert_iteration(self) -> expert_iteration.ExIt:
    """The equal ExIt spec: the default operator and targets.

    Returns:
      The ExIt spec.
    """
    return expert_iteration.ExIt(az=self.az)


class AZKind(expert_iteration.AZFamilyKind):
  """The AlphaZero learner kind."""

  name = "az"
  spec_cls = AZ


def plain_network(policy: types.Policy, params: Any) -> tuple[Any, ...]:
  """(network, params) inside a routed and / or search policy.

  The base of every RoutedPolicy and SearchPolicy wrapper, recursively.

  Args:
    policy: the policy.
    params: its params.

  Returns:
    (network, params).

  Raises:
    AssertionError: the wrappers nest too deep.
  """
  for _ in range(8):
    if isinstance(policy, (routed.RoutedPolicy, search.SearchPolicy)):
      policy, params = policy.base, params["base"]
    else:
      return policy, params
  raise AssertionError("routed / search wrappers nest too deep")


class AZDriver(base.Driver):
  """The AZ and ExIt kinds' driver.

  The AZ and ExIt kinds' driver (`learners.az` pieces: build, the value
  targets, PlyReplay, BCLearner).

  Attributes:
    spec: the AZ or ExIt kind spec.
    options: the spec, or an AZ spec as its ExIt equivalent.
    az_lib: the learners.az module.
    cfg: the AZConfig.
    factory: the slot factory that builds the operator.
    operator: the operator over the base network.
    slot_policy: the policy the learner slot plays.
    policy: the base network.
    learner: the learner.
    replay: the replay buffer.
    publishes_siblings: every checkpoint writes the plain network first.
    seg: the plies of a whole game and its terminal ply.
    arena: the Arena (set by attach).
    mesh: the Arena's device mesh (set by attach).
    dp: the DataParallel over the mesh (set by attach).
  """

  kind = "az"
  algo = "az"

  def __init__(
      self,
      spec: Any,
      env: types.Env,
      policy_kwargs: Optional[Mapping[str, Any]] = None,
      policy_arch: str = "network",
  ):
    """policy_arch / policy_kwargs: the base network (empty: the AZConfig's)."""
    if isinstance(spec, AZ):
      opts = spec.as_expert_iteration()
    elif isinstance(spec, expert_iteration.ExIt):
      opts = spec
    else:
      raise TypeError(
          f"AZDriver drives AZ and ExIt specs, got {type(spec).__name__}"
      )
    self.spec, self.options = spec, opts
    self.az_lib, self.cfg = az_lib, opts.az
    default_factory = not opts.factory
    built_env, base_net, sp, _, _, learner, replay = az_lib.build(
        opts.az,
        env=env,
        policy_arch=policy_arch,
        policy_kwargs=dict(policy_kwargs or {}),
        with_search=default_factory,
    )
    assert built_env is env, (built_env.name, env.name)
    if opts.policy_mask != learner.cfg.policy_mask:
      learner = bc.BCLearner(
          base_net,
          dataclasses.replace(learner.cfg, policy_mask=opts.policy_mask),
      )
    self.factory = slot_factories.get(opts.factory, opts.factory_kwargs)
    # Built once (attach builds the Arena around it); generations swap params
    # only.
    self.operator = (
        sp if default_factory else self.factory.build(base_net, env, opts)
    )
    operators.check_publishable(self.operator)
    self.slot_policy = operators.with_behavior(
        base_net, self.operator, opts.behavior
    )
    # The plain network as a pool member, gate entrant and plain sibling: the
    # factory's plain_member(base): the base, or the base routed to a
    # scripted phase it cannot play.
    self._plain_policy = self.factory.plain_member(base_net)
    if not (
        self._plain_policy is base_net
        or isinstance(self._plain_policy, routed.RoutedPolicy)
    ):
      raise TypeError(
          f"{self.factory.name}.plain_member returned "
          f"{type(self._plain_policy).__name__}: want the base or a "
          "RoutedPolicy over "
          "it (the factory's plain_params / unplain_params wrap the base's "
          "params)"
      )
    if (
        self._plain_policy is not base_net
        and self._plain_policy.base is not base_net
    ):
      raise TypeError(
          f"{self.factory.name}.plain_member must route the base itself"
      )
    self._member_policy = (
        self._plain_policy if opts.plain_members else self.operator
    )
    self.policy, self.learner, self.replay = base_net, learner, replay
    self.publishes_siblings = bool(opts.publish_plain)
    self._host_slot = (
        self.slot_policy.spec.capability == types.Capability.HOST_DRIVEN
    )
    if self._host_slot and not opts.host:
      raise ValueError(
          f"the operator {self.operator.spec.arch_name} is HOST_DRIVEN: set "
          "host=True "
          "(the HostArena)"
      )
    # Its override rows are not search decisions: a RoutedPolicy operator, or a
    # HOST_DRIVEN one that routes internally (operators.routed_view).
    self._routed_operator = operators.routed_view(self.operator)
    # Uses private _core of a module in this package.
    op_core = operators._core(self.operator)  # pylint: disable=protected-access
    op_vs = getattr(getattr(op_core, "cfg", None), "value_scale", None)
    if (
        opts.value_scale is not None
        and op_vs is not None
        and float(op_vs) != float(opts.value_scale)
    ):
      raise ValueError(
          f"the operator's value_scale {op_vs} != the spec's "
          f"{opts.value_scale} "
          "(the base value head's units)"
      )
    if (
        opts.value_lambda > 0.0
        and operators.value_bias_of(self.operator) == "fused_upper_bound"
    ):
      raise ValueError(
          f"value_lambda {opts.value_lambda} > 0 from a fused_upper_bound "
          "operator: "
          "its root values are a clairvoyant's upper bound (PIMC strategy "
          "fusion)"
      )
    self.seg = env.max_len + 1  # a whole game and its terminal ply
    self._sources: dict[str, Any] = {}
    self._owner: Optional[str] = None
    self._pending_replay: Optional[bytes] = None
    self._replay_generation: Optional[int] = None
    # (snapshot member name, stacked slot params)
    self._gen: Optional[tuple[Any, ...]] = None
    self._league = None

  # -- params ------------------------------------------------------------------
  @property
  def member_policy(self) -> types.Policy:
    """The policy the population pool stores."""
    return self._member_policy

  def wrap(self, params):
    """Learner params -> the acting slot's params.

    The operator's, or a labelling slot's {"base", "operator"}.

    Args:
      params: the learner params.

    Returns:
      The slot params.
    """
    op = self.factory.params(self.operator, params)
    if self.slot_policy is self.operator:
      return op
    return operators.BehaviorMixture.make_params(params, op)

  def unwrap(self, slot_params):
    """Slot params -> learner params.

    Args:
      slot_params: the slot params.

    Returns:
      The learner params.
    """
    if self.slot_policy is not self.operator:
      return slot_params["base"]
    return self.factory.unwrap(self.operator, slot_params)

  @property
  def plain_policy(self) -> types.Policy:
    """The plain network's policy (the factory's plain_member)."""
    return self._plain_policy

  def plain_params(self, params):
    """Learner params -> plain_policy's params (the factory's plain_params).

    Args:
      params: the learner params.

    Returns:
      The plain params.
    """
    return self.factory.plain_params(self._plain_policy, params)

  def unplain_params(self, plain_params):
    """Plain params -> learner params (the factory's unplain_params).

    Args:
      plain_params: the plain params.

    Returns:
      The learner params.
    """
    return self.factory.unplain_params(self._plain_policy, plain_params)

  def wrap_member(self, params):
    """Learner params -> pool member params (plain or operator).

    Args:
      params: the learner params.

    Returns:
      The member params.
    """
    if self.options.plain_members:
      return self.plain_params(params)
    return self.factory.params(self.operator, params)

  def unwrap_member(self, member_params):
    """Pool member params -> learner params.

    Args:
      member_params: the member params.

    Returns:
      The learner params.
    """
    if self.options.plain_members:
      return self.unplain_params(member_params)
    return self.factory.unwrap(self.operator, member_params)

  def _stack_slot(self, slot_params):
    """Arena slot params of one member.

    Arena slot params of one member: stacked [1] (in-graph) or a list of one
    host object (the HostArena's HOST_DRIVEN form).

    Args:
      slot_params: one member's slot params.

    Returns:
      The Arena slot params.
    """
    return [slot_params] if self._host_slot else base.stack1(slot_params)

  def init_state(self, key, env):
    """Fresh params from key, or generation 0 warm-started from the spec's init.

    The init's plain network, with a fresh optimizer.

    Args:
      key: the init key.
      env: the league's env.

    Returns:
      The learner state.

    Raises:
      ValueError: the init's network is not the base, or its value_scale
        differs from the spec's.
    """
    opts = self.options
    if opts.init is None:
      return self.learner.init(key, env)

    pol, params = plain_network(*opts.init.build())
    if pol.spec != self.policy.spec:
      raise ValueError(
          f"init {opts.init.id}: its network {pol.spec.arch_name} "
          f"{dict(pol.spec.arch_kwargs)} is not the base "
          f"{self.policy.spec.arch_name} {dict(self.policy.spec.arch_kwargs)}"
      )
    if opts.init.artifact and opts.value_scale is not None:
      path, step = opts.init.artifact.rsplit("@", 1)
      got = artifact_lib.value_scale(path, int(step))
      if got is None or float(got) != float(opts.value_scale):
        raise ValueError(
            f"init {opts.init.id} records value_scale {got}, the spec says "
            f"{opts.value_scale}"
        )
    logging.info(
        "AZDriver: generation 0 warm start from %s (fresh optimizer)",
        opts.init.id,
    )
    return self.learner.init_from_params(params)

  def artifact(self, ls) -> tuple[Any, ...]:
    """(operator, its params): the operator is published, not a labelling slot.

    Args:
      ls: the learner slot.

    Returns:
      (operator, params).
    """
    return self.operator, self.factory.params(
        self.operator, self.learner.params(ls.state)
    )

  def plain_artifact(self, ls) -> tuple[Any, ...]:
    """(plain network, its params): the plain sibling (publish_plain).

    The base network, or the factory's plain_member over it.

    Args:
      ls: the learner slot.

    Returns:
      (plain policy, params).
    """
    return self.plain_policy, self.plain_params(self.learner.params(ls.state))

  def artifact_extra(self) -> dict[str, Any]:
    """The value head's units and link, and the warm start's lineage.

    The units and link come from this config, never from the parent.

    Returns:
      The extra artifact fields.
    """
    opts = self.options
    extra: dict[str, Any] = {}
    ext: dict[str, Any] = {}
    if opts.value_scale is not None:
      ext["value_scale"] = float(opts.value_scale)
    if opts.value_link is not None:
      ext["value_link"] = opts.value_link
    if ext:
      extra["extensions"] = ext
    if opts.init is not None:
      extra.update(
          base.lineage_extra(opts.init.id, opts.parents)
          if opts.parents
          else opts.init.lineage()
      )
    return extra

  # -- attach / train ----------------------------------------------------------
  def attach(self, league, mesh) -> None:
    """Builds the Arena over league.opponent_slots().

    Args:
      league: the League.
      mesh: the device mesh.

    Raises:
      ValueError: num_steps is not one whole game per row.
    """
    cfg = league.config
    if cfg.num_steps != self.seg:
      raise ValueError(
          "an AZ league plays one whole game per row per update: num_steps "
          "must "
          f"be max_len + 1 = {self.seg}, got {cfg.num_steps}"
      )
    rule = self.cfg.slot(self.slot_policy)
    if cfg.learner_rule != rule.rule:
      raise ValueError(
          f"learner_rule {cfg.learner_rule} != the AZ self-play rule "
          f"{rule.rule}"
      )
    n = self.options.updates_per_generation
    if n is not None:
      sps = [
          c
          for c in base.controllers_of(league.controller)
          if isinstance(c, controller.ScheduledPromotion)
      ]
      if len(sps) != 1 or int(sps[0].schedule.snapshot_every) != int(n):
        raise ValueError(
            f"updates_per_generation={n}: a generation is one "
            "ScheduledPromotion "
            "round, so the controller must be a ScheduledPromotion with "
            f"snapshot_every == {n} (got {type(league.controller).__name__}"
            + (
                f", snapshot_every {sps[0].schedule.snapshot_every}"
                if sps
                else ""
            )
            + ")"
        )
    if (
        self._host_slot
        and not self.options.plain_members
        and cfg.pool_capacity > 0
        and not hasattr(self.factory, "freeze")
    ):
      raise ValueError(
          "a HOST_DRIVEN operator as a pool member needs its factory to supply "
          "freeze: run with plain_members=True or pool_capacity=0"
      )
    self.arena = self._arena(league, mesh, rule, host=self.options.host)
    self.mesh = self.arena.mesh
    self.dp = data_parallel.DataParallel(self.mesh)
    if self.cfg.minibatch_size % self.dp.num_devices:
      raise ValueError(
          f"minibatch_size {self.cfg.minibatch_size} is not divisible by "
          f"{self.dp.num_devices} devices"
      )
    learner = self.learner
    self._league = league

    def train_epoch(s, mbs, key):
      """One epoch of learner updates over the minibatches.

      Args:
        s: the learner state.
        mbs: the stacked minibatches.
        key: the epoch's key.

      Returns:
        (state, metrics).
      """

      def body(c, xs):
        """One minibatch update.

        Args:
          c: the learner state.
          xs: (minibatch, index).

        Returns:
          (state, metrics).
        """
        mb, i = xs
        return learner.update(c, mb, x64.fold_in64(key, i))

      n = mbs.valid.shape[0]
      s, ms = jax.lax.scan(body, s, (mbs, jnp.arange(n, dtype=jnp.int64)))
      return s, tree_map(jnp.mean, ms)

    self._train_epoch = jax.jit(train_epoch)

  def generation(self, ls) -> Optional[int]:
    """Ls's generation (updates // updates_per_generation), None without one.

    Args:
      ls: the learner slot.

    Returns:
      The generation, or None.
    """
    n = self.options.updates_per_generation
    return None if n is None else int(ls.updates) // int(n)

  def _slot0(self, ls):
    """Slot 0's params.

    Slot 0's params: the learner's current params, or per generation the
    frozen params of the admitted snapshot "<learner>@<u>".

    Args:
      ls: the LearnerSlot.

    Returns:
      Slot 0's params.

    Raises:
      KeyError: the generation snapshot is not an admitted member.
    """
    n = self.options.updates_per_generation
    if n is None:
      return self._stack_slot(self.wrap(self.learner.params(ls.state)))
    snap = f"{ls.name}@{self.generation(ls) * int(n)}"
    if self._gen is None or self._gen[0] != snap:
      if self._league is None or snap not in self._league.members:
        raise KeyError(
            f"generation snapshot {snap} is not an admitted member "
            "(ScheduledPromotion "
            "admits it at the round start)"
        )
      base_params = self.unwrap_member(self._league.member_params(snap))
      base_params = tree_map(jnp.asarray, base_params)
      self._gen = (snap, self._stack_slot(self.wrap(base_params)))
      logging.info(
          "AZDriver: generation %d plays the frozen %s",
          self.generation(ls),
          snap,
      )
    return self._gen[1]

  def _targets(self, batch: types.TrajectoryBatch):
    """The value (and policy) targets of a self-play batch (ExIt doc).

    Args:
      batch: The self-play batch.

    Returns:
      (batch with targets, the excluded rows or None).
    """
    opts = self.options
    excluded = None
    if self._routed_operator is not None:
      batch, excluded = decisions.exclude_override(self._routed_operator, batch)
    if opts.value_lambda > 0.0:
      batch = self.az_lib.search_value_targets(
          batch,
          lam=opts.value_lambda,
          value_scale=opts.value_scale,
          first_hand_only=True,
      )
    else:
      batch = self.az_lib.outcome_targets(
          batch, first_hand_only=True, value_scale=opts.value_scale
      )
    if opts.q_softmax_temperature is not None:
      batch = targets_lib.with_q_softmax_targets(
          batch, temperature=opts.q_softmax_temperature
      )
    return batch, excluded

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
      AssertionError: a first hand did not finish in the segment.
    """
    del fresh  # every update plays one fresh game per row
    self._single(ls)
    t0 = time.time()
    k_play, k_epoch, k_train = jax.random.split(key, 3)
    params = [self._slot0(ls), *opponent_params]
    src = self._sources.get(ls.name)
    if src is None:
      src = arena_lib.ArenaSource(self.arena, params, table, self.seg)
      self._sources[ls.name] = src
    src.table = table
    src.set_params(params)
    src.restart()
    f0, d0 = src.frames, src.decisions
    batch = src.next(k_play)
    if self.slot_policy is self.operator:
      base_carry = self.factory.base_carry(batch.init_carry)
    else:  # a labelling slot's {"base", "operator"}
      base_carry = batch.init_carry["base"]
    batch, excluded = self._targets(batch._replace(init_carry=base_carry))
    complete = np.asarray(batch.labels[targets_lib.TARGET_VALUE_VALID])
    if not np.all(complete | ~np.asarray(batch.valid)):
      raise AssertionError(
          f"a first hand did not finish in {self.seg} plies: its value "
          "targets would be missing"
      )
    if self._pending_replay is not None:
      self.replay.load_state_bytes(self._pending_replay, batch)
      self._pending_replay = None
    gen = self.generation(ls)
    if gen is not None:  # the replay holds the current generation's plies only
      if self._replay_generation is not None and self._replay_generation != gen:
        self.replay.clear()
      self._replay_generation = gen
    self.replay.add(batch)
    t_selfplay = time.time()  # the completeness check above synced the batch
    dp = self.dp
    # The epoch's BC steps run data parallel over each minibatch's rows.
    with dp.scope():
      state, m = self._train_epoch(
          dp.replicate(ls.state),
          dp.shard_rows(self.replay.epoch(k_epoch), axis=1),
          dp.replicate(k_train),
      )
    m = {k: float(v) for k, v in m.items()}
    m["avg_game_length"] = (
        float(jnp.sum(batch.valid, dtype=jnp.int64)) / self.cfg.batch_size
    )
    m["seconds"] = time.time() - t0
    m["selfplay_seconds"] = t_selfplay - t0
    m["train_seconds"] = m["seconds"] - m["selfplay_seconds"]
    if excluded is not None:
      m["routed_excluded"] = float(excluded)
    if gen is not None:
      m["generation"] = float(gen)
    return state, m, src.frames - f0, src.decisions - d0

  def sources_state(self) -> dict[str, Any]:
    """No snapshots: every update restarts every row.

    Returns:
      {}.
    """
    return {}  # every update restarts every row

  def host_state(self) -> dict[str, Any]:
    """The replay not yet trained on, and its generation.

    Returns:
      The state, or {} when there is nothing to save.
    """
    if self._owner is None and self._pending_replay is None:
      return {}
    # A restored replay not yet trained on is saved as restored.
    out = (
        {"replay": self._pending_replay}
        if self._pending_replay is not None
        else {"replay": self.replay.state_bytes()}
    )
    if self.options.updates_per_generation is not None:
      out["replay_generation"] = self._replay_generation
    return out

  def load_host_state(self, d: Mapping[str, Any]) -> None:
    """Restores `host_state()`.

    Args:
      d: the saved host state.
    """
    self._pending_replay = d.get("replay")
    if self.options.updates_per_generation is not None:
      self._replay_generation = d.get("replay_generation")


KIND = AZKind()
base.register_kind(KIND)
