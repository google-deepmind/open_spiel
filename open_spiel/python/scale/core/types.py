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

"""The contracts every component implements.

Env, Policy, Learner and the records that pass between them.

Environments, policies, belief samplers, learners and trajectory sources are
written against these Protocols and records only, so any component that
honours a contract composes with every other: the rollout drivers
(core.arena), the decision rule (core.decide), checkpoints (ckpt) and serving
(serve) never special-case one.

The module is stdlib and typing only. Importing it loads no array framework,
so a manifest reader, a serving shell or a host-side component can depend on
the contracts alone; core/tests/types_hygiene_test.py enforces this.

Shapes. Array-valued fields are typed `Any`; their shapes are documented per
field with B = batch rows, T = time (plies), A = number of actions and
P = number of players.

Dtypes. The package runs with jax_enable_x64 on (the package's enable_x64(),
which the entry points call; components assert it), so every array states its
dtype:
  * actions, seats, slots, members, to_move, last_action, last_actor: i32;
  * masks and flags: bool;
  * logits, values, rewards, behaviour log-probabilities and targets: f32;
  * params: f32 unless a network declares otherwise (bf16 compute where
    configured, outputs f32);
  * every counter (frames, updates, env steps, plies, hands, chance events,
    RNG fold-in indices): i64, folded into keys with core.x64.fold_in64.
`decide` computes in f64 and returns (action i32[B], probs f64[B, A]).
"""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import json
import math
from typing import (
    Any,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TypeVar,
)

Array = (
    Any  # an array of the component's framework (jax.Array, np.ndarray, ...)
)
PyTree = Any
# PRNG keys. NATIVE_JAX components get typed jax keys (jax.random.key). A host
# component never sees a typed key: the CALLBACK_BATCHED and HOST_DRIVEN
# drivers pass raw uint32[..., 2] key data instead. The per-ply derivation is
# public (core.plies), so a host replay reproduces in-graph sampling exactly.
Key = Any
Metrics = Mapping[str, Any]

E = TypeVar("E")  # env state
C = TypeVar("C")  # policy carry
Bc = TypeVar("Bc")  # belief-sampler carry
S = TypeVar("S")  # learner state

# SeatStep.last_actor sentinel.
NO_ACTOR = -1  # nothing has acted yet this hand (first ply after a reset)
# SeatStep.last_action sentinel: no last action, or the action is hidden from
# this seat (Env.visible_action).
NO_ACTION = -1

# PolicyOutput.aux keys the Arena copies into TrajectoryBatch.target_policy /
# target_value when the perspective slot emits them (e.g. a SearchPolicy's
# visit distribution f32[B, A] and root value f32[B, P]).
AUX_TARGET_POLICY = "target_policy"
AUX_TARGET_VALUE = "target_value"

# labels key, bool[B, T]: target_value holds a real target at this ply. Every
# writer of TrajectoryBatch.target_value sets it and refuses a batch that
# already carries it, so a batch has exactly one value-target writer: the
# Arena when it copies AUX_TARGET_VALUE (a teacher's value), or a
# sources.targets builder (Monte Carlo returns, search values). The check is
# on pytree structure: no device sync, and it works under jit.
TARGET_VALUE_VALID = "target_value_valid"


class Capability(enum.Enum):
  """How a component can be driven.

  NATIVE_JAX: pure jittable JAX; runs inside the Arena's jit and scan.
  CALLBACK_BATCHED: a host implementation (numpy, or another framework)
    reached from inside a JAX graph through
    `jax.pure_callback(vmap_method="broadcast_all")`.
  HOST_DRIVEN: a Python loop owns the step (core.host_arena.HostArena), for
    components that cannot be called from inside a JAX graph.
  """

  NATIVE_JAX = "native_jax"
  CALLBACK_BATCHED = "callback_batched"
  HOST_DRIVEN = "host_driven"


@runtime_checkable
class Env(Protocol[E]):
  """One game's dynamics: no opponents, no encoders.

  All methods operate on a single (unbatched) state; callers vmap.

  Two layers. The raw layer (`root`, `apply`, `is_chance`, `apply_chance`)
  exposes chance nodes so a recorded game (human/engine logs, an app history,
  another trainer's deals) can be replayed with its actual chance outcomes. The
  resolved layer (`init`, `step`) is a thin wrapper that samples every chance
  node with `core.chance.resolve` (chance event n of the call uses
  core.x64.fold_in64(key, n)); its states are never at a chance node.

  An env may set the class attribute `admits_stateful_policies = False`
  (absent means True): its SeatStep event stream says more than its InfoState
  does, so a policy carry, which observe updates at every ply, would learn
  what the seat may not know. check_policy_admitted refuses such a pairing.
  The hidden-move board games (phantom_ttt, dark_hex and their imperfect-recall
  variants) set it: a failed attempt onto a held cell is private, yet it is one
  more ply every seat observes, so a carry could count the opponent's retries.

  Attributes:
    name: the game's registered name.
    num_players: P, the number of seats.
    num_actions: A, the number of distinct player actions.
    max_len: the most player decisions in one hand.
    max_chance_outcomes: C, the most outcomes of one chance node.
    zero_sum: whether returns sum to 0 at terminal (checked by check_env).
  """

  name: str  # the game's registered name
  num_players: int
  num_actions: int
  max_len: int  # max player decisions in one hand
  max_chance_outcomes: int
  zero_sum: bool  # returns sum to 0 at terminal (checked by check_env)

  def root(self) -> E:
    """The game's initial state, chance unresolved (may be a chance node)."""
    ...

  def apply(self, s: E, a: Array) -> E:
    """Applies player action a; chance unresolved.

    Args:
      s: the state.
      a: the player action.

    Returns:
      the state after a, chance unresolved.
    """
    ...

  def is_chance(self, s: E) -> Array:
    """Whether s is a chance node (bool; False at terminal).

    Args:
      s: the state.

    Returns:
      whether s is a chance node (bool; False at terminal).
    """
    ...

  def apply_chance(self, s: E, outcome: Array) -> E:
    """Applies a known chance outcome at a chance node.

    Args:
      s: the chance node.
      outcome: the chance outcome.

    Returns:
      the state after the outcome.
    """
    ...

  def init(self, deal_key: Key) -> E:
    """Root() with every leading chance node resolved from deal_key.

    Args:
      deal_key: the deal key.

    Returns:
      the root with every leading chance node resolved.
    """
    ...

  def step(self, s: E, a: Array, chance_key: Key) -> E:
    """Apply(s, a), then every chance node resolved from chance_key.

    Args:
      s: the state.
      a: the player action.
      chance_key: the key the chance outcomes are drawn from.

    Returns:
      the state after a with every chance node resolved.
    """
    ...

  def chance_probs(self, s: E) -> Array:
    """The outcome probabilities at a chance node, f32[max_chance_outcomes].

    Args:
      s: the chance node.

    Returns:
      the outcome probabilities, f32[max_chance_outcomes].
    """
    ...

  def actor(self, s: E) -> Array:
    """The i32 seat to move in [0, P); -1 at a terminal state.

    Args:
      s: the state.

    Returns:
      the i32 seat to move in [0, P); -1 at a terminal state.
    """
    ...

  def legal_mask(self, s: E) -> Array:
    """The actor's legal actions, bool[A]; all False at terminal.

    Args:
      s: the state.

    Returns:
      the actor's legal actions, bool[A]; all False at terminal.
    """
    ...

  def terminal(self, s: E) -> Array:
    """Whether s is terminal (bool).

    Args:
      s: the state.

    Returns:
      whether s is terminal (bool).
    """
    ...

  def returns(self, s: E) -> Array:
    """The seats' returns, f32[P]; meaningful at terminal.

    Args:
      s: the state.

    Returns:
      the seats' returns, f32[P].
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def info_state(self, s: E, seat: Array) -> PyTree:
    """The privacy boundary: what `seat` may know.

    Hidden information is absent.

    Args:
      s: the state.
      seat: the seat whose view is taken.

    Returns:
      the InfoState of `seat`: what it may know.
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def labels(self, s: E, seat: Array) -> Mapping[str, Array]:
    """Privileged targets (e.g. the opponent's hand). Learners only.

    May be empty; a learner asserts the keys it needs.

    Args:
      s: the state.
      seat: the seat the targets are for.

    Returns:
      the privileged targets by name (may be empty).
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def visible_action(self, s: E, a: Array, seat: Array) -> Array:
    """What `seat` learns of action a taken at state s (i32).

    s is the state before a. The result is a itself for a public action,
    NO_ACTION for a private one (phantom tic-tac-toe, dark hex). It feeds
    SeatStep.last_action per seat, so the event stream obeys the same privacy
    boundary as info_state.

    Args:
      s: the state before a.
      a: the action.
      seat: the seat that observes.

    Returns:
      what `seat` learns of a (i32): a itself, or NO_ACTION.
    """
    ...

  def world_from_info(self, info: PyTree) -> Optional[E]:
    """The one state consistent with `info` when info determines it, else None.

    info determines the state under perfect information.

    Args:
      info: an InfoState.

    Returns:
      the one consistent state, or None.
    """
    ...


class SeatStep(NamedTuple):
  """What one seat sees at one ply. Batched: every field has leading [B].

  A hand's plies: decision plies (to_move >= 0), then one terminal ply
  (to_move = -1, legal_mask all False, info = the terminal InfoState, e.g. a
  showdown; last_action/last_actor = the hand-ending action). The terminal ply
  is observe-only: no evaluate, no action, valid False. The next hand's first
  ply has new_hand True and last_action/last_actor NO_ACTION/NO_ACTOR.

  Attributes:
    info: the seat's InfoState PyTree.
    legal_mask: bool[B, A]; the legal actions if this seat is to move, else all
      False (another seat's legal set can leak its private information).
    seat: i32[B], the observing seat.
    to_move: i32[B], the seat to act at this ply; -1 on a terminal ply.
    new_hand: bool[B], first ply of a hand (policy resets per-hand state).
    new_match: bool[B], first ply of a match (policy resets all state; implies
      new_hand).
    last_action: i32[B], the previous player action as this seat sees it
      (Env.visible_action), or NO_ACTION.
    last_actor: i32[B], the previous player action's actor, or NO_ACTOR.
      Chance outcomes are folded into the player step that triggered them
      (Env.step, core.chance.resolve), so last_action/last_actor only ever
      describe player actions; a seat learns chance outcomes through its info.
  """

  info: PyTree
  legal_mask: Array
  seat: Array
  to_move: Array
  new_hand: Array
  new_match: Array
  last_action: Array
  last_actor: Array


class PolicyOutput(NamedTuple):
  """A policy's output.

  logits f32[B, A], value f32[B, P] (one value per seat) and aux, a mapping of
  extra heads [B, ...].

  Attributes:
    logits: f32[B, A].
    value: f32[B, P], one value per seat.
    aux: a mapping of extra heads [B, ...].
  """

  logits: Array
  value: Array
  aux: Mapping[str, Array]


#: Who chose an action: a policy's decider, or "forced" (one legal action,
#: so no policy was asked).
DECIDERS = ("network", "search", "scripted", "forced")
#: Who chooses a policy's actions (PolicySpec.decider, decider_of): DECIDERS
#: without "forced", which is an outcome of play, not a property of a policy.
POLICY_DECIDERS = ("network", "search", "scripted")

#: The prefix of an arch_kwargs string that names a function injected as an
#: object ("injected:<qualname>"): such a spec cannot be rebuilt, and two
#: different functions can share it.
INJECTED_PREFIX = "injected:"


def names_injected(arch_kwargs: Any) -> bool:
  """Whether arch_kwargs holds a string naming an injected function.

  Nested mappings and sequences are walked; an injected name starts with
  INJECTED_PREFIX.

  Args:
    arch_kwargs: the architecture kwargs (or any value within them).

  Returns:
    True if a string within starts with INJECTED_PREFIX.
  """
  if isinstance(arch_kwargs, str):
    return arch_kwargs.startswith(INJECTED_PREFIX)
  if isinstance(arch_kwargs, Mapping):
    return any(names_injected(v) for v in arch_kwargs.values())
  if isinstance(arch_kwargs, (list, tuple)):
    return any(names_injected(v) for v in arch_kwargs)
  return False


@dataclasses.dataclass(frozen=True)
class PolicySpec:
  """A policy's identity.

  (arch_name, arch_kwargs) rebuild the policy through the architecture
  registry (policies.registry.build), so a network is defined once and chosen
  by config; arch_kwargs must be JSON-serialisable, because artifacts record
  it.

  Attributes:
    game: the env's registered name.
    obs_format: the encoder name the policy reads.
    num_actions: the env's number of distinct actions.
    num_players: the env's number of seats.
    stateful: whether the policy's carry holds state across plies.
    capability: how the policy can be driven (Capability).
    framework: the array framework of its params ("jax", "numpy", or a name
      a plugin registers in core.frameworks); codecs are keyed by it.
    arch_name: the architecture's registered name.
    arch_kwargs: the architecture's arguments.
    decider: who chooses the action, one of POLICY_DECIDERS. A policy class
      may instead fix it with a `decider` attribute (decider_of).
    publishable: False for a policy that must not be written as an artifact
      (ckpt refuses it), for example one that only labels data.
  """

  game: str
  obs_format: str
  num_actions: int
  num_players: int
  stateful: bool
  capability: Capability
  framework: str
  arch_name: str
  arch_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  decider: str = "network"
  publishable: bool = True

  def __post_init__(self):
    assert isinstance(self.capability, Capability), self.capability
    assert isinstance(self.framework, str) and self.framework, self.framework
    assert self.decider in POLICY_DECIDERS, self.decider
    assert isinstance(self.publishable, bool), self.publishable
    assert self.num_actions > 0, self.num_actions
    assert self.num_players > 0, self.num_players
    assert isinstance(self.arch_name, str) and self.arch_name, self.arch_name
    assert isinstance(self.arch_kwargs, Mapping), type(self.arch_kwargs)
    json.dumps(dict(self.arch_kwargs), sort_keys=True)  # raises if not JSON


class Policy(Protocol[C]):
  """A policy over InfoState. It never sees env state.

  The carry changes only in `observe`; `evaluate` returns no carry, so a
  stateless replay of `observe` over a history equals live play. JAX
  implementations subclass this Protocol explicitly to inherit the scan
  defaults of observe_seq and evaluate_seq (core.sequence); a policy of
  another framework implements both itself (the defaults raise).

  Params. The Arena stacks a slot's params [K, ...] and gathers each row's
  member. A host-side slot whose weights live outside the params pytree puts
  an i32 member-id leaf in its params (for example {"member": arange(K)}), so
  it receives the member per row and indexes its own weights
  (HostWeightsPolicy).

  Serving equals evaluation: a stateless replay of a match (one observe_seq
  over the history, evaluate on the last ply, decide) gives what the Arena
  gave at that decision:
    * tabular, uniform and host-computed (callback, host-driven) policies,
      and routed policies over them: bitwise, with the same logits, the same
      f64 decide distribution and the same action;
    * every other NATIVE_JAX policy (networks, feed-forward or recurrent,
      search over a network): the same action, with logits within 1e-6
      absolute plus 1e-6 relative and the decide distribution within the
      bound that implies (twice the largest logit change). The Arena
      evaluates inside its rollout scan and a replay is a different XLA
      program, whose fusion choices can round the float arithmetic
      differently in the last bits.
  serve/tests/serve_eval_test asserts this per policy kind.

  Attributes:
    spec: the policy's PolicySpec.
  """

  spec: PolicySpec

  def init_params(self, key: Key) -> PyTree:
    """Fresh params for this architecture (one member, unstacked).

    Args:
      key: the init key.

    Returns:
      the params of one member, unstacked.
    """
    ...

  def encode(self, info: PyTree) -> Array:
    """Features [B, ...] of a batched InfoState (the policy owns its encoding).

    Args:
      info: a batched InfoState.

    Returns:
      Array [B, ...] of features.
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def init_carry(self, params: PyTree, batch: int, seat: Array, key: Key) -> C:
    """The carry of `batch` rows before any observe.

    Args:
      params: one member's params.
      batch: the row count.
      seat: i32[batch], each row's seat.
      key: the carry's key.

    Returns:
      the carry of `batch` rows.
    """
    ...

  def observe(self, params: PyTree, carry: C, step: SeatStep) -> C:
    """Called for every seat at every ply. Resets on step.new_hand/new_match.

    Args:
      params: one member's params.
      carry: the carry.
      step: the SeatStep of this ply.

    Returns:
      the carry after the step.
    """
    ...

  def observe_seq(self, params: PyTree, carry: C, steps: SeatStep) -> C:
    """The carry after observing steps [B, T, ...] in order.

    The default (JAX only) is a scan of `observe`; an override must equal it
    (testing.check_policy).

    Args:
      params: one member's params.
      carry: the carry before steps.
      steps: SeatSteps [B, T, ...].

    Returns:
      the carry after the last step.
    """
    _require_jax_default(self, "observe_seq")
    # Lazy: core.sequence imports jax, and this module loads no array framework.
    from open_spiel.python.scale.core import sequence  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return sequence.scan_observe(self, params, carry, steps)

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def evaluate(
      self, params: PyTree, carry: C, step: SeatStep, key: Key
  ) -> PolicyOutput:
    """Acting rows only in the Arena. Returns no carry.

    Args:
      params: one member's params.
      carry: the carry after observing step.
      step: the SeatStep of the acting rows.
      key: the evaluate key.

    Returns:
      the PolicyOutput of the acting rows.
    """
    ...

  def evaluate_seq(
      self, params: PyTree, carry: C, steps: SeatStep, keys: Optional[Key]
  ) -> tuple[C, PolicyOutput]:
    """(final carry, PolicyOutput [B, T, ...]).

    observe then evaluate at every ply of steps [B, T]. keys: what evaluate
    receives at each ply, stacked on a time axis (JAX: [T] keys; host: raw
    uint32[B, T, 2]), or None. The default (JAX only) is a scan; an override
    must equal it (testing.check_policy).

    Args:
      params: one member's params.
      carry: the carry before steps.
      steps: SeatSteps [B, T, ...].
      keys: the per-ply evaluate keys stacked on a time axis, or None.

    Returns:
      (final carry, PolicyOutput [B, T, ...]).
    """
    _require_jax_default(self, "evaluate_seq")
    # Lazy: core.sequence imports jax, and this module loads no array framework.
    from open_spiel.python.scale.core import sequence  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return sequence.scan_evaluate(self, params, carry, steps, keys)


def decider_of(policy: Any) -> str:
  """Who chooses `policy`'s actions, one of POLICY_DECIDERS.

  A policy's `decider` attribute, when set, wins over its PolicySpec.decider:
  a class whose decider follows from its code (a search policy is "search",
  a scripted bot "scripted") or from a policy it wraps (a routed policy
  reports its base's) declares it there. Otherwise the spec's.

  Raises:
    ValueError: the decider is not in POLICY_DECIDERS.

  Args:
    policy: the policy.

  Returns:
    the decider, one of POLICY_DECIDERS.
  """
  d = getattr(policy, "decider", None) or policy.spec.decider
  if d not in POLICY_DECIDERS:
    raise ValueError(
        f"{policy.spec.arch_name}: decider {d!r} not in {POLICY_DECIDERS}"
    )
  return d


def check_policy_admitted(env: Any, spec: PolicySpec) -> None:
  """Refuses a stateful policy on an env that does not admit one.

  spec.stateful policies need Env.admits_stateful_policies. Every driver that
  feeds a policy SeatSteps calls this: the Arena and HostArena, serving
  (HostAgent, over serve.replay) and the shard reader of a BC student.

  Raises:
    ValueError: spec is stateful and env refuses stateful policies.
    TypeError: env.admits_stateful_policies is not a bool.

  Args:
    env: the env.
    spec: the policy's spec.
  """
  admits = getattr(env, "admits_stateful_policies", True)
  if not isinstance(admits, bool):
    raise TypeError(
        f"{type(env).__name__}.admits_stateful_policies must be a bool, got "
        f"{admits!r}"
    )
  if spec.stateful and not admits:
    raise ValueError(
        f"{env.name} refuses stateful policies: a policy carry updated at "
        "every ply would "
        "learn what the seat may not know (Env.admits_stateful_policies); "
        f"{spec.arch_name} is stateful. Use a feed-forward policy: the "
        "InfoState carries "
        "everything the seat may use."
    )


def _require_jax_default(policy: Any, name: str) -> None:
  if policy.spec.framework != "jax":
    raise NotImplementedError(
        f"{type(policy).__name__}.{name}: the default is a jax scan; a "
        f"{policy.spec.framework} policy must implement {name} itself"
    )


@runtime_checkable
class RowMaskedPolicy(Protocol):
  """A policy that can evaluate a batch in which some rows are padding.

  The Arena runs a slot's rows in fixed-size chunks and pads a chunk with
  copies of a real row. A policy with batch-dependent behaviour (a host
  module with running statistics, a component whose cost depends on how many
  rows take a branch) implements evaluate_rows, and the Arena calls it with
  the padding marked instead of evaluate.
  """

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def evaluate_rows(
      self, params: PyTree, carry: Any, step: "SeatStep", key: Key, rows: Array
  ) -> "PolicyOutput":
    """Evaluate with the batch's real rows marked.

    Padding rows change neither the outputs of real rows nor the policy's state.

    Args:
      params: one member's params.
      carry: the carry.
      step: the SeatStep.
      key: the evaluate key.
      rows: bool[B], True for real rows.

    Returns:
      the PolicyOutput of every row (padding rows' outputs unused).
    """
    ...


class HostWeightsPolicy(Protocol):
  """A policy whose weights live outside the params pytree, as host objects.

  Its params hold member ids; stacking, freezing and releasing members go
  through these methods instead of tree operations (policies.members).
  """

  def export_member(self, params: PyTree, member: int) -> Any:
    """A copy of member's weights, detached from the live policy.

    Args:
      params: the live params.
      member: the member index.

    Returns:
      the member's weights, detached from the live policy.
    """
    ...

  def import_member(self, params: PyTree, member: int, blob: Any) -> PyTree:
    """Params with member's weights set from an exported blob.

    Args:
      params: the live params.
      member: the member index.
      blob: an export_member blob.

    Returns:
      the params with the member's weights set from blob.
    """
    ...

  def release_member(self, member: int) -> None:
    """Frees member's host weights.

    Args:
      member: the member index.
    """
    ...


class SearchOperator(Protocol):
  """A policy whose outputs are search targets (policies.operators).

  Attributes:
    decider: who chooses its action, one of POLICY_DECIDERS.
    prior_dependence: how its targets depend on the network it improves, a
      key of policies.operator_vocab.PRIOR_DEPENDENCE.
    emits_target_policy: whether it emits AUX_TARGET_POLICY.
  """

  decider: str
  prior_dependence: str
  emits_target_policy: bool

  def value_bias(self) -> str:
    """The bias of its values: "unbiased" or "fused_upper_bound".

    "fused_upper_bound" when its values come from searching worlds the acting
    seat cannot tell apart (strategy fusion).

    Returns:
      "unbiased" or "fused_upper_bound".
    """
    ...


@dataclasses.dataclass(frozen=True)
class DecisionRule:
  """How `decide` turns logits into an action.

  Attributes:
    kind: "argmax" (first maximal legal index) or "sample" (categorical).
    temperature: > 0; logits are divided by it before masking/softmax.
  """

  kind: str = "argmax"
  temperature: float = 1.0

  def __post_init__(self):
    assert self.kind in ("argmax", "sample"), self.kind
    t = float(self.temperature)
    assert t > 0.0 and math.isfinite(t), self.temperature


ARGMAX = DecisionRule("argmax", 1.0)
SAMPLE = DecisionRule("sample", 1.0)


class BeliefSampler(Protocol[Bc]):
  """Samples complete game states consistent with what one seat has observed.

  Search in an imperfect-information game needs concrete worlds to expand: a
  sampler turns a seat's InfoState history into K env states and log-weights.

    sampler = belief.registry.build("enumeration", env)
    params = sampler.table_for(opponent, opponent_params, core.types.SAMPLE)
    carry = sampler.init_carry(params, batch, seat, key)
    carry = sampler.observe(params, carry, step)          # once per SeatStep
    worlds, logw = sampler.sample_worlds(params, carry, step.info, k=32,
                                         key=key)

  Every method takes `params`, so a learned sampler (a belief network, for
  example) has the same shape as a Policy. A sampler without parameters
  returns {} from init_params.

  Attributes:
    capability: how the sampler can be driven (search runs NATIVE_JAX ones).
    is_perfect_information: True if `info` determines the world, in which
      case search values carry no strategy-fusion bias.
    params_source: where `params` come from: "none", "own" (the searching
      policy's params) or "opponent" (an opponent policy or its table).
  """

  capability: Capability
  is_perfect_information: bool
  params_source: str

  def init_params(self, key: Key) -> PyTree:
    """Fresh sampler params.

    Args:
      key: the init key.

    Returns:
      the sampler params.
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def init_carry(self, params: PyTree, batch: int, seat: Array, key: Key) -> Bc:
    """The sampler carry of `batch` rows before any observe.

    Args:
      params: the sampler params.
      batch: the row count.
      seat: i32[batch], each row's seat.
      key: the carry's key.

    Returns:
      the sampler carry of `batch` rows.
    """
    ...

  def observe(self, params: PyTree, carry: Bc, step: SeatStep) -> Bc:
    """The carry after one SeatStep.

    Args:
      params: the sampler params.
      carry: the carry.
      step: the SeatStep of this ply.

    Returns:
      the carry after the step.
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def sample_worlds(
      self, params: PyTree, carry: Bc, info: PyTree, k: int, key: Key
  ) -> Any:
    """(env states [B, K, ...], log-weights f32[B, K]).

    A finite log-weight is the world's importance weight (0 for an exact draw
    from the belief). -inf marks a world the sampler could not draw (no world
    is consistent with the row's InfoState, or a dead particle); its state is
    a placeholder that a consumer must not use. Search averages a row over
    its finite worlds and refuses (NaN, a bad ply) a row with none.

    Args:
      params: the sampler params.
      carry: the carry.
      info: the rows' InfoStates.
      k: K, worlds per row.
      key: the sampling key.

    Returns:
      (env states [B, K, ...], log-weights f32[B, K]).
    """
    ...


class TrajectoryBatch(NamedTuple):
  """A batch of trajectories [B, T, ...]; each row is one seat's perspective.

  Attributes:
    steps: SeatStep [B, T] of the perspective seat at every ply (acting or not,
      terminal plies included, so observe_seq can rebuild the carry).
    init_carry: the perspective policy's carry before steps[:, 0]. A sequence
      need not start at a new_hand ply (online sources continue hands).
    action: i32[B, T], the action taken at the ply by whoever acted; -1 on a
      terminal ply.
    behavior_logits: f32[B, T, A], the perspective seat's raw logits where it
      acted, else 0.
    behavior_logp: f32[B, T], log-probability of `action` under the behaviour
      policy actually used (decide with the slot's rule: the masked tempered
      softmax for "sample", 0 for the deterministic "argmax") where the
      perspective seat acted, else 0. The importance-ratio denominator.
    value_pred: f32[B, T, P], its value head where it acted, else 0.
    target_policy: f32[B, T, A], filled by the source (search visits,
      teacher, outcome); from a plain Arena, the perspective policy's
      aux[AUX_TARGET_POLICY] where it acted, else zeros.
    target_value: f32[B, T, P], filled the same way; from a plain Arena,
      aux[AUX_TARGET_VALUE] where it acted, else zeros.
    reward: f32[B, T], the perspective seat's reward for the transition out of
      ply t. Convention: the terminal return lands on the ply whose action ended
      the hand (done[t]), which is often an opponent ply (valid False); returns
      and GAE credit it to the perspective seat's last valid ply of the hand.
    done: bool[B, T], ply t's action ended a hand (the next ply is terminal).
    valid: bool[B, T], the perspective seat acted at this ply (loss mask).
    labels: privileged targets at each ply (learners only).
    aux: the perspective policy's other aux heads [B, T, ...] where it acted.
    bootstrap_step: SeatStep [B], the perspective seat's step at ply T (the
      first ply after the segment), for bootstrapping a truncated segment.
  """

  steps: SeatStep
  init_carry: PyTree
  action: Array
  behavior_logits: Array
  behavior_logp: Array
  value_pred: Array
  target_policy: Array
  target_value: Array
  reward: Array
  done: Array
  valid: Array
  labels: Mapping[str, Array]
  aux: Mapping[str, Array]
  bootstrap_step: SeatStep


class TrajectorySource(Protocol):
  """A producer of TrajectoryBatches.

  the Arena (online), a shard reader (offline) or a replay buffer.
  """

  def next(self, key: Key) -> TrajectoryBatch:
    """The next TrajectoryBatch.

    Args:
      key: this batch's key.

    Returns:
      the next TrajectoryBatch.
    """
    ...


class Learner(Protocol[S]):
  """Trains one policy from TrajectoryBatches.

  A learner of another framework receives numpy batches through interop.host.

  Its state is saved by the StateCodec of its framework (ckpt.codecs): the
  learner's `framework` attribute (a str) when it declares one, else
  policy().spec.framework.
  """

  def init(self, key: Key, env: Env) -> S:
    """A fresh learner state for `env` (its config is the learner's own).

    Args:
      key: the init key.
      env: the env.

    Returns:
      a fresh learner state.
    """
    ...

  def policy(self) -> Policy:
    """The Policy the learner trains."""
    ...

  def params(self, s: S) -> PyTree:
    """The policy params in learner state s.

    Args:
      s: the learner state.

    Returns:
      the policy params in s.
    """
    ...

  def update(self, s: S, batch: TrajectoryBatch, key: Key) -> tuple[S, Metrics]:
    """One update on a batch: (new state, metrics).

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      key: the update key.

    Returns:
      (new state, metrics).
    """
    ...

  def state_dict(self, s: S) -> Mapping[str, Any]:
    """Everything needed to resume (params, optimizer state, step).

    A mapping ckpt can write (a host framework's state need not be a pytree).

    Args:
      s: the learner state.

    Returns:
      a mapping ckpt can write.
    """
    ...

  def load_state(self, blob: Mapping[str, Any]) -> S:
    """Inverse of state_dict.

    Args:
      blob: a state_dict mapping.

    Returns:
      the learner state the mapping holds.
    """
    ...

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def init_from_params(self, params: PyTree) -> S:
    """A fresh learner state around `params` (a new optimizer state, step 0).

    A JAX learner resumed by ckpt.codecs.JaxStateCodec must implement it: the
    codec builds the restore template from init_from_params(init_params(key)).
    The league also uses it, when present, to warm-start a learner from
    published params. A learner of another framework, whose state codec
    does not need a template, may omit it.

    Args:
      params: the policy params.

    Returns:
      a fresh learner state around params.
    """
    ...


class _EmptyMapping(collections.abc.Mapping):
  """An immutable, picklable empty mapping.

  the default of Manifest's mapping fields, so no two manifests share a
  mutable default.
  """

  def __getitem__(self, key):
    raise KeyError(key)

  def __iter__(self):
    return iter(())

  def __len__(self) -> int:
    return 0

  def __repr__(self) -> str:
    return "{}"


_EMPTY = _EmptyMapping()


class Manifest(NamedTuple):
  """The identity of one artifact step.

  what it plays, how to rebuild its policy, and where it came from.

  Every artifact format (ckpt.formats) reads into this one type, so
  code downstream of the loader never depends on a file layout.

  The native format (ckpt.artifact) stores each field under its own
  name in manifest.json.

  Attributes:
    schema_version: the version of the format the step was read from.
    game: the env registry name of the game the policy plays.
    rules_sha: the rules hash of that game (GameSpec.rules_sha), so a policy
      is never served under rules it was not trained on.
    obs_format: the encoder name the policy reads (PolicySpec.obs_format).
    arch_name: the architecture name the policy registry rebuilds it under.
    arch_kwargs: the JSON kwargs of that architecture.
    framework: the framework that owns the params (core.frameworks).
    capability: how the policy runs (Capability).
    default_rule: the decision rule a server applies by default.
    params_sha256: the params digest, as defined by the params codec.
    step: the step this manifest describes.
    parents: artifact ids ("<run>@<step>") of a warm start's parents.
    env_spec: the EnvSpec JSON {name, kwargs} of the training env, or None
      when the artifact does not record it.
    run_id: the id of the run that wrote the step, or None.
    provenance: JSON provenance (git_commit, config_hash, algo, env_name,
      code_repo, a warm start's lineage, ...); `git_commit` reads the commit
      from it.
    extensions: registered extension records (ckpt.artifact.
      register_meta_extension).
  """

  schema_version: int
  game: str
  rules_sha: str
  obs_format: str
  arch_name: str
  arch_kwargs: Mapping[str, Any]
  framework: str
  capability: Capability
  default_rule: DecisionRule
  params_sha256: str
  step: int
  parents: Sequence[str]
  env_spec: Optional[Mapping[str, Any]] = None
  run_id: Optional[str] = None
  provenance: Mapping[str, Any] = _EMPTY
  extensions: Mapping[str, Any] = _EMPTY

  @property
  def git_commit(self) -> str:
    """The code commit of the run that wrote the step ("" if unknown)."""
    return str(self.provenance.get("git_commit") or "")


class Artifact(NamedTuple):
  """A loaded artifact: its manifest, its params and the Policy built on it.

  Attributes:
    manifest: the artifact's Manifest.
    params: the loaded params.
    policy: the Policy built on the params.
    path: where the artifact was loaded from.
  """

  manifest: Manifest
  params: PyTree
  policy: Policy
  path: str
