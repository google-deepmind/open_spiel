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

"""The learner-kind protocol, the kind registry and the driver base class.

A league trains one learner kind. The kind is a JSON-ready spec in the run
config (`LeagueRunConfig.learner`, for example
`kinds.PG(learner=PGConfig(...))`) and a `LearnerKind` object, registered
under the spec's `kind` name, that knows how to resize, reseed and extend that
spec and how to build the driver the
`League` calls once per update. Code that handles run configs generically
(config overrides, resume hashing, the command line) goes through the kind
object and never branches on the kind name.

    kind = kinds.of(cfg.learner)                  # the LearnerKind of a spec
    spec = kind.with_run_updates(cfg.learner, 500)
    driver = kind.build_driver(spec, env, policy_arch="network",
                               policy_kwargs=cfg.policy_kwargs)

A driver owns the one `types.Learner` every learning slot of the league
shares, the Arena (or HostArena) its slots play in, one trajectory source per
learner and the update. The League uses only the members of `Driver`:

  policy, learner, slot_policy   the learner's policy, the Learner, and what
                                 plays in Arena slot 0 and is published
  wrap(params) / unwrap(p)       learner params <-> slot params
  member_policy, wrap_member     what the population pool stores
  wrap_opponent(policy, params)  how a fixed opponent plays
  attach(league, mesh)           builds the Arena over league.opponent_slots()
  init_state(key, env)           a fresh learner state
  train_step(ls, table, opponent_params, key, fresh)
                                 one update of learner slot ls; returns
                                 (state, metrics, frames, decisions); a
                                 wall-clock metric is named "*_seconds"
                                 (runner.is_timing_key) and stays out of
                                 the run's result rows
  freeze(params)                 slot params of a pool member that keeps the
                                 learner's current params
  commit_state / retire_state    where a learner state lives between updates
  mutate(src, fresh, s, p)       the PBT parameter edit s * src + p * fresh
  host_state / load_host_state   what a resume needs beyond learner states
  sources_state / load_sources_state
                                 every learner's ArenaSource snapshot

Learner-state rewrites: the League rewrites a learner state only through the
named entries "params" and "opt_state" of `Learner.state_dict` (warm starts,
PBT, the per-iteration optimizer reset), or through
`Learner.init_from_params` when the learner has it. A learner whose
state_dict lacks those entries is refused when such an action is applied.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import host_arena
from open_spiel.python.scale.core import types
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import pool_params
from open_spiel.python.scale.policies import members as members_lib

tree_map = jax.tree_util.tree_map


@runtime_checkable
class LearnerKind(Protocol):
  """How generic code handles one learner kind's spec.

  Attributes:
    name: the spec's `kind` value, the registry key.
    spec_cls: the spec dataclass.
  """

  name: str
  spec_cls: type[Any]

  def with_batch(self, cfg: Any, num_envs: int) -> Any:
    """Cfg with every field that must equal the league's batch rows set.

    Args:
      cfg: the kind's spec.
      num_envs: the league's batch rows.

    Returns:
      The resized spec.
    """

  def with_seed(self, cfg: Any, seed: int) -> Any:
    """Cfg with every field that must equal the run's seed set.

    Args:
      cfg: the kind's spec.
      seed: the run's seed.

    Returns:
      The reseeded spec.
    """

  def with_run_updates(self, cfg: Any, n: int) -> Any:
    """Cfg with its schedule horizon set to n updates.

    Args:
      cfg: the kind's spec.
      n: the run length in updates.

    Returns:
      The extended spec.
    """

  def run_length_paths(self) -> tuple[str, ...]:
    """Dotted paths, relative to the spec, of the fields that set run length.

    Returns:
      The paths.
    """

  def extendable_hash_fields(self) -> tuple[str, ...]:
    """Dotted spec paths the resume hash may ignore so the run may grow.

    The paths are ignored when the spec's step size does not depend on the run
    length (`cfg.constant_lr()`).

    Returns:
      The paths.
    """

  def build_driver(
      self,
      cfg: Any,
      env: types.Env,
      *,
      policy_arch: str = "network",
      policy_kwargs: Optional[Mapping[str, Any]] = None,
      data_dir: Optional[str] = None,
      scratch_dir: Optional[str] = None,
  ) -> "Driver":
    """The driver a League over `env` calls for each update.

    Args:
      cfg: the kind's spec.
      env: the league's env.
      policy_arch: the registered architecture of the learner's policy.
      policy_kwargs: the architecture's kwargs.
      data_dir: recorded shards to train from, for a kind that reads them.
      scratch_dir: a local directory for a kind whose learner can only read
        and write local files (for example a host framework that resumes from a
        local path even when the run directory is on gs://). A spec class sets
        the class attribute `needs_scratch_dir = True` to get one
        (`league.runner.scratch_dir_for`); every other kind receives None.

    Returns:
      The Driver.
    """


class KindCLI(Protocol):
  """The optional command-line hook of a learner kind.

  The optional command-line hook of a learner kind, used only by the train
  entry point (`train.py`). A kind without these methods has no flags of its
  own and trains the run config as given.
  """

  def add_flags(self, flag_values: Any) -> None:
    """Defines this kind's command-line flags on an absl FlagValues.

    Args:
      flag_values: the absl FlagValues to define the flags on.
    """

  def prepare(
      self, run_cfg: Any, flag_values: Any
  ) -> Optional[tuple[Any, Mapping[str, Any]]]:
    """Applies this kind's flags to a run.

    Args:
      run_cfg: the run config.
      flag_values: the parsed absl FlagValues.

    Returns:
      (the run config to train, extra keyword arguments of
      `league.runner.run`), or None when the flags asked for another action
      instead of training, which `prepare` performed.
    """


class KindBase:
  """Defaults for the optional parts of LearnerKind and KindCLI.

  Defaults for the optional parts of the LearnerKind protocol and the
  KindCLI hook (no flags; the run config trains as given).

  Attributes:
    spec_cls: the kind spec class the kind's CLI builds.
  """

  name = ""
  spec_cls: type[Any] = object
  multi_learner = False  # may train several learner slots at once (PSRO, PBT)
  fixed_num_steps = False  # the kind sets the plies per update itself

  def with_batch(self, cfg: Any, num_envs: int) -> Any:
    """Returns cfg unchanged: no field follows the batch rows.

    Args:
      cfg: the kind's spec.
      num_envs: the league's batch rows (unused).

    Returns:
      cfg.
    """
    del num_envs
    return cfg

  def with_seed(self, cfg: Any, seed: int) -> Any:
    """Returns cfg unchanged: no field follows the run's seed.

    Args:
      cfg: the kind's spec.
      seed: the run's seed (unused).

    Returns:
      cfg.
    """
    del seed
    return cfg

  def with_run_updates(self, cfg: Any, n: int) -> Any:
    """Cfg extended to n updates by its own with_run_updates.

    Args:
      cfg: the kind's spec.
      n: the run length in updates.

    Returns:
      The extended spec.
    """
    return cfg.with_run_updates(n)

  def run_length_paths(self) -> tuple[str, ...]:
    """No run-length fields by default.

    Returns:
      An empty tuple.
    """
    return ()

  def extendable_hash_fields(self) -> tuple[str, ...]:
    """The run-length paths.

    Returns:
      `run_length_paths()`.
    """
    return self.run_length_paths()

  def add_flags(self, flag_values: Any) -> None:
    """Defines no flags.

    Args:
      flag_values: the absl FlagValues (unused).
    """
    del flag_values

  def prepare(
      self, run_cfg: Any, flag_values: Any
  ) -> Optional[tuple[Any, Mapping[str, Any]]]:
    """Trains the run config as given.

    Args:
      run_cfg: the run config.
      flag_values: the parsed absl FlagValues (unused).

    Returns:
      (run_cfg, {}).
    """
    del flag_values
    return run_cfg, {}

  @staticmethod
  def refuse_network(
      cfg: Any,
      policy_arch: str,
      policy_kwargs: Optional[Mapping[str, Any]],
      data_dir: Optional[str],
  ) -> None:
    """Raises for the network and data options of a kind that builds its own.

    Args:
      cfg: the kind's spec.
      policy_arch: the requested architecture; only "network" is allowed.
      policy_kwargs: the requested architecture kwargs; must be empty.
      data_dir: the requested data directory; must be None.

    Raises:
      ValueError: a network or data option was given.
    """
    if policy_arch != "network" or policy_kwargs:
      raise ValueError(
          "policy_arch / policy_kwargs name the pg / bc / az network, not a "
          f"{cfg.kind} learner's"
      )
    if data_dir is not None:
      raise ValueError("data_dir is a BC option")


_KINDS: dict[str, LearnerKind] = {}


def register_kind(kind: LearnerKind, *, replace: bool = False) -> None:
  """Registers a learner kind under `kind.name`.

  Args:
    kind: the LearnerKind.
    replace: whether a different kind object may replace a registered one.

  Raises:
    ValueError: the name is taken by another kind object (unless replace).
    TypeError: `kind` does not implement LearnerKind.
  """
  if not isinstance(kind, LearnerKind):
    raise TypeError(
        f"{type(kind).__name__} does not implement league.kinds.LearnerKind"
    )
  name = kind.name
  if not isinstance(name, str) or not name:
    raise ValueError(f"learner kind name {name!r}")
  old = _KINDS.get(name)
  if old is not None and old is not kind and not replace:
    raise ValueError(f"learner kind {name!r} is already registered")
  _KINDS[name] = kind


def get(name: str) -> LearnerKind:
  """The registered learner kind named `name`.

  Args:
    name: the kind name.

  Returns:
    The LearnerKind.

  Raises:
    KeyError: no kind is registered under `name`.
  """
  if name not in _KINDS:
    raise KeyError(f"no learner kind {name!r}; registered: {names()}")
  return _KINDS[name]


def of(spec: Any) -> LearnerKind:
  """The LearnerKind of a spec (by its `kind` field), checking its class.

  Args:
    spec: a kind spec.

  Returns:
    The LearnerKind.

  Raises:
    TypeError: spec is not a registered kind's spec.
  """
  name = getattr(spec, "kind", None)
  if not isinstance(name, str) or name not in _KINDS:
    raise TypeError(
        "learner must be a league.kinds spec of a registered kind ("
        f"{names()}), "
        f"got {type(spec).__name__}"
    )
  kind = _KINDS[name]
  if not isinstance(spec, kind.spec_cls):
    raise TypeError(
        f"{type(spec).__name__} is not the {kind.name!r} kind's spec "
        f"{kind.spec_cls.__name__}"
    )
  return kind


def names() -> tuple[str, ...]:
  """The registered kind names.

  Returns:
    The names in registration order.
  """
  return tuple(_KINDS)


def spec_classes() -> tuple[type[Any], ...]:
  """The registered kinds' spec classes.

  Returns:
    The classes in registration order.
  """
  return tuple(k.spec_cls for k in _KINDS.values())


def make_driver(
    spec: Any,
    env: types.Env,
    policy_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    data_dir: Optional[str] = None,
    scratch_dir: Optional[str] = None,
    policy_arch: str = "network",
) -> "Driver":
  """The driver of a kind spec over `env` (`LearnerKind.build_driver`).

  Args:
    spec: a kind spec.
    env: the league's env.
    policy_kwargs: the policy architecture's kwargs.
    data_dir: recorded shards to train from.
    scratch_dir: a local scratch directory.
    policy_arch: the registered architecture of the learner's policy.

  Returns:
    The Driver.
  """
  return of(spec).build_driver(
      spec,
      env,
      policy_arch=policy_arch,
      policy_kwargs=dict(policy_kwargs or {}),
      data_dir=data_dir,
      scratch_dir=scratch_dir,
  )


def stack1(tree: Any) -> Any:
  """Tree with a leading member axis of size 1 (Arena slot params of K = 1).

  Args:
    tree: a params tree.

  Returns:
    The stacked tree.
  """
  return tree_map(lambda x: jnp.asarray(x)[None], tree)


def lineage_extra(source: str, parents: tuple[Any, ...]) -> dict[str, Any]:
  """The `parents` and `lineage` artifact fields of a warm-started run.

  Args:
    source: the warm-start source.
    parents: the parent artifact ids; [source] when empty.

  Returns:
    {"parents": ..., "lineage": ...}.
  """
  parents = list(parents) or [source]
  m = re.match(r"^([A-Za-z0-9_.\-]+)@(\d+)$", parents[0])  # "<run id>@<step>"
  return dict(
      parents=parents,
      lineage=dict(
          source=source,
          parent_run_id=m.group(1) if m else None,
          step=int(m.group(2)) if m else None,
      ),
  )


def controllers_of(c: Any) -> list[Any]:
  """C and, for a composite controller, every controller inside it.

  Args:
    c: a controller.

  Returns:
    The controllers, depth first.
  """
  inner = getattr(c, "controllers", None)
  if inner is None:
    return [c]
  out = []
  for i in inner:
    out.extend(controllers_of(i))
  return out


class Driver:
  """What every driver shares (module doc)."""

  kind = ""
  # the first update index g (a continued host learner's train step)
  update_offset = 0
  in_graph_learner = True
  default_rule = types.ARGMAX
  # writes plain and search artifacts beside each other
  publishes_siblings = False

  def wrap(self, params):
    """Learner params -> slot params (identity by default).

    Args:
      params: the learner params.

    Returns:
      The slot params.
    """
    return params

  def unwrap(self, slot_params):
    """Slot params -> learner params (identity by default).

    Args:
      slot_params: the slot params.

    Returns:
      The learner params.
    """
    return slot_params

  @property
  def member_policy(self) -> types.Policy:
    """The policy the population pool stores: the slot policy."""
    return self.slot_policy

  def wrap_member(self, params):
    """Learner params -> pool member params.

    Args:
      params: the learner params.

    Returns:
      The member params.
    """
    return self.wrap(params)

  def unwrap_member(self, member_params):
    """Pool member params -> learner params.

    Args:
      member_params: the member params.

    Returns:
      The learner params.
    """
    return self.unwrap(member_params)

  def wrap_opponent(self, policy, params):
    """How a fixed opponent plays (unchanged by default).

    Args:
      policy: the opponent's policy.
      params: the opponent's params.

    Returns:
      (policy, params) as played.
    """
    return policy, params

  # The plain network: promotion-gate entrants, a plain sibling artifact and
  # plain pool members. The learner's policy by default.
  @property
  def plain_policy(self) -> types.Policy:
    """The plain network's policy: the learner's policy."""
    return self.policy

  def plain_params(self, params):
    """Learner params -> plain_policy's params.

    Args:
      params: the learner params.

    Returns:
      The plain policy's params.
    """
    return params

  def check_admissible(self) -> None:
    """Raises if the kind cannot admit pool members (never by default)."""
    pass

  def freeze(self, params):
    """Slot params of a pool member that keeps playing `params`.

    `params` are the learner's current params. JAX params are values; a policy
    with host-held weights copies them into another member
    (`policies.members.freeze`).

    Args:
      params: the learner params.

    Returns:
      The member's slot params.
    """
    return self.wrap_member(members_lib.freeze(self.policy, params))

  def commit_state(self, state):
    """An in-graph learner state, replicated on the Arena's mesh.

    That is the placement the update returns, so a spawned or mutated state
    reuses the compiled update of a trained one.

    Args:
      state: the learner state.

    Returns:
      The placed state; unchanged for a host learner.
    """
    if not self.in_graph_learner:
      return state

    return jax.device_put(state, NamedSharding(self.mesh, PartitionSpec()))

  def retire_state(self, state):
    """A retired learner's state in host memory (released at the next save).

    Args:
      state: the learner state.

    Returns:
      The host-resident state; unchanged for a host learner.
    """
    if not self.in_graph_learner:
      return state

    return pool_params.JaxStacked().to_host(state)

  def mutate(self, src, fresh, shrink: float, perturb: float):
    """Shrink * src + perturb * fresh, leafwise in src's dtypes.

    Args:
      src: the source params.
      fresh: freshly initialised params.
      shrink: the weight of src.
      perturb: the weight of fresh.

    Returns:
      The edited params.
    """
    sh, pt = jnp.float32(shrink), jnp.float32(perturb)
    return tree_map(
        lambda s_, f: (sh * s_ + pt * f).astype(s_.dtype), src, fresh
    )

  def init_state(self, key, env):
    """A fresh learner state.

    Args:
      key: the init key.
      env: the league's env.

    Returns:
      The learner state.
    """
    return self.learner.init(key, env)

  def artifact_step(self, ls) -> int:
    """The artifact step of learner slot `ls`: its update count.

    Args:
      ls: the learner slot.

    Returns:
      The step.
    """
    return int(ls.updates)

  def artifact(self, ls) -> tuple[Any, ...]:
    """(policy, params) of learner ls's published artifact: the slot policy.

    Args:
      ls: the learner slot.

    Returns:
      (policy, params).
    """
    return self.slot_policy, self.wrap(self.learner.params(ls.state))

  def artifact_extra(self) -> dict[str, Any]:
    """Extra artifact fields (none by default).

    Returns:
      The fields.
    """
    return {}

  def host_state(self) -> dict[str, Any]:
    """What a resume needs beyond learner states (nothing by default).

    Returns:
      The state, JSON-ready.
    """
    return {}

  def load_host_state(self, d: Mapping[str, Any]) -> None:
    """Restores `host_state()` (which must be empty by default).

    Args:
      d: the saved host state.
    """
    assert not d, sorted(d)

  def forget(self, name: str) -> None:
    """Drops learner `name`'s trajectory source.

    Args:
      name: the learner name.
    """
    self._sources.pop(name, None)

  def sources_state(self) -> dict[str, Any]:
    """Every learner's ArenaSource snapshot.

    Returns:
      The snapshots by learner name.
    """
    return {
        n: commit.snapshot_source(s)
        for n, s in self._sources.items()
        if isinstance(s, arena_lib.ArenaSource)
    }

  def load_sources_state(self, d: Mapping[str, Any]) -> None:
    """Restores the snapshots `sources_state()` returned.

    Args:
      d: the snapshots by learner name.

    Raises:
      ValueError: there is no Arena to restore into, or a learner already has a
        source.
    """
    if d and getattr(self, "arena", None) is None:
      raise ValueError(
          f"the {self.kind} kind has no Arena to restore sources {sorted(d)} "
          "into"
      )
    for name, snap in d.items():
      if name in self._sources:
        raise ValueError(f"learner {name} already has a source")
      self._sources[name] = commit.restore_source(self.arena, snap)

  def _arena(
      self, league, mesh, learner_slot: arena_lib.Slot, host: bool = False
  ):
    """The learner's arena against the league's opponents.

    Args:
      league: The league.
      mesh: The mesh.
      learner_slot: The learner's slot.
      host: Whether to build a HostArena.

    Returns:
      The arena.
    """
    cls = host_arena.HostArena if host else arena_lib.Arena
    return cls(
        league.env,
        [learner_slot, *league.opponent_slots()],
        perspective_slot=0,
        mesh=mesh,
    )

  def _single(self, ls) -> None:
    """Refuses a second learner slot for a single-learner kind."""
    # The kinds that call this set _owner in __init__ (kinds/host.py).
    # pylint: disable-next=access-member-before-definition
    if self._owner is None:
      self._owner = ls.name
    elif self._owner != ls.name:
      raise NotImplementedError(
          f"a {self.kind} league trains one learner ({self._owner}); "
          f"{ls.name} would be a second"
      )
