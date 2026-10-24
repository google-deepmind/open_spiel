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

"""Monte Carlo tree search (mctx) over Env dynamics, packaged as a Policy.

SearchPolicy turns any NATIVE_JAX base policy (priors and values) and an Env
(the dynamics) into a stronger policy: at each decision it runs MCTS from
the current position and plays from the root visit distribution. It is the
operator of AlphaZero-style training and a test-time search for any trained
network. Like every Policy it sees only the seat's SeatSteps: the roots
come from a belief sampler (`belief`), never from the true state.

  sampler = belief.registry.build("perfect_information", env)
  policy = SearchPolicy(network, env, sampler, MCTSConfig(num_simulations=64))
  params = policy.make_params(network_params)  # {"base": ..., "sampler": ...}
  out = policy.evaluate(params, carry, step, key)  # logits = log(visits)

Worlds. The sampler draws K root worlds per row (`MCTSConfig.num_worlds`).
With K > 1 the B x K worlds run as one mctx batch and the root visit
distributions and values are averaged uniformly over the K worlds
(perfect-information Monte Carlo, PIMC). With a perfect-information sampler
the search is exact AlphaZero-style MCTS; with an imperfect-information
sampler each tree knows its world below the root (strategy fusion), so its
values are biased upwards (`value_bias()`).

Tree nodes. A decision node (`Node`) holds the env state, every seat's base
carry (each seat observes every transition in the tree, so a recurrent base
sees the same SeatStep stream as in the Arena), the searching seat, the
seat to move and a terminal flag. Priors and values come from
base.evaluate on the mover's SeatStep; illegal actions get prior logit
-1e9.

Modes (`MCTSConfig.mode`):
  zero_sum      two players. Node values are from the mover's view; a
                transition's discount is -1 when the seat to move changes,
                +1 when the same seat moves again, 0 into a terminal.
  single_agent  the other seats are folded into the dynamics: after the
                searching seat acts they play the base policy under
                `opponent_rule` until the searching seat is to move again or
                the hand ends. Values and rewards are from the searching
                seat's view.

Chance. In zero_sum mode on an env with chance nodes the tree is
mctx.stochastic_muzero_policy: every action leads to an afterstate whose
outcomes are the env's chance_probs (one outcome with probability 1 where no
chance node follows). `chance_value="exact"` values an afterstate by the
exact expectation over its outcomes (one network forward per outcome);
`"sampled"` uses one outcome drawn per simulation (one forward, unbiased at
chance afterstates). In single_agent mode chance is sampled open-loop per
simulation. `chance="determinized"` instead fixes each world's chance
outcomes up front (`Node.chance_scores`: every chance node takes the
argmax over positive-probability outcomes of log p + score), so the tree
has no chance nodes; a sampler with `sample_determinized` (envs.gin.belief)
supplies the scores with its worlds, any other gets i.i.d. Gumbel scores.

Scripted phases (`fold_routed`). When the base is a RoutedPolicy whose
override is a scripted phase (for example gin's endgame), every node the
route sends to the override is folded into the dynamics: the override's
action is applied and the transition continues until a node the base
decides. A root the route sends to the override is not searched; evaluate
returns the base's output there. This needs a tree without stochastic
chance nodes.

Outputs. evaluate returns logits log(visit distribution + 1e-8)
(`act="weights"`) or 0 at mctx's selected action and -inf elsewhere
(`act="mctx_action"`, which for gumbel search is the sequential-halving
winner). The policy has no temperature of its own: the acting slot's
DecisionRule and core.arena.TemperatureDrop set it. value is the root value
as per-seat values f32[B, P] (zero_sum: +v for the seat, -v for the other;
single_agent: the seat only). aux carries AUX_TARGET_POLICY (the visit
distribution f32[B, A], the policy target of expert iteration),
AUX_SEARCH_VALUE (root value f32[B]) and AUX_SEARCH_Q (root Q per action
f32[B, A], 0 where unvisited). Rows whose seat is not to move get zeros;
in a B > 1 call they are not searched, while a one-row call (the Arena's
per-row program, or a single request) still searches them and zeroes the
result.

Requires jax_enable_x64 (search asserts it).

Determinism. In a B > 1 call each row searches alone with its own key (the
row index folded in), so a row's result never depends on the other rows or
on the batch size.

Value units. Env returns are divided by `value_scale` inside the tree, so
rewards match the units of the base's value head (for example a value head
trained on points / 50 uses value_scale 50).

Identity. arch_kwargs record the base (arch_name, arch_kwargs), the env as
its EnvSpec JSON, the sampler as its registered name and kwargs, and the full
MCTSConfig, so the registry rebuilds the policy from an artifact.

Params: {"base": base params, "sampler": sampler params}; the sampler entry
is {} when the sampler reads the base params (`params_source == "own"`).

Privacy. evaluate reads the seat's SeatStep and carry only. A stateful
base's other seats start each search from a fresh carry that observes the
root, because their real carries depend on their private information; for a
stateless base this is exact.

Limitations: zero_sum mode is two-player; single_agent mode folds any
number of other seats into the dynamics. Stochastic chance trees are PUCT
only (gumbel raises).
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import mctx

from open_spiel.python.scale.belief import registry as belief_registry
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import chance as chance_lib
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import routed as routed_lib

ARCH = "search"
F32 = jnp.float32
NEG = -1e9  # prior logit of an illegal action
AUX_SEARCH_VALUE = operator_vocab.AUX_SEARCH_VALUE
AUX_SEARCH_Q = operator_vocab.AUX_SEARCH_Q
SEARCH_KINDS = ("muzero", "gumbel")
ACTS = ("weights", "mctx_action")
MODES = ("zero_sum", "single_agent")
CHANCES = ("tree", "determinized")
CHANCE_VALUES = ("exact", "sampled")

tree_map = jax.tree_util.tree_map


@dataclasses.dataclass(frozen=True)
class MCTSConfig:
  """The search's settings (module doc for the concepts).

  Attributes:
    num_simulations: simulations per tree.
    search: "muzero" (PUCT with Dirichlet root noise,
      mctx.muzero_policy) or "gumbel" (mctx.gumbel_muzero_policy).
    mode: "zero_sum" (two players, alternating perspective) or
      "single_agent" (the other seats are part of the dynamics).
    num_worlds: K, root worlds drawn from the sampler per row; PIMC when > 1.
    dirichlet_alpha: the PUCT root noise's Dirichlet concentration.
    dirichlet_fraction: the PUCT root noise's mixing fraction; 0 disables the
      noise.
    pb_c_init: the PUCT exploration constant c_init.
    pb_c_base: the PUCT exploration constant c_base.
    gumbel_scale: the Gumbel noise scale of gumbel search; 0 makes it
      deterministic.
    max_depth: the deepest simulated path; None is unbounded.
    opponent_rule: how folded seats act in single_agent mode, "sample" or
      "argmax" under the base policy.
    act: "weights" (logits = log visit distribution) or "mctx_action" (play
      mctx's selected action; needs num_worlds 1).
    chance: "tree" (chance nodes in the tree, or sampled open-loop in
      single_agent mode) or "determinized" (each world's chance outcomes are
      fixed).
    fold_routed: fold a RoutedPolicy base's override plies into the dynamics.
    value_scale: env returns are divided by it inside the tree.
    chance_value: afterstate value of a chance-node tree, "exact" (expectation
      over outcomes) or "sampled" (one outcome per simulation).
  """

  num_simulations: int = 32
  search: str = "muzero"
  mode: str = "zero_sum"
  num_worlds: int = 1
  dirichlet_alpha: float = 0.3
  dirichlet_fraction: float = 0.25
  pb_c_init: float = 1.25
  pb_c_base: float = 19652.0
  gumbel_scale: float = 1.0
  max_depth: Optional[int] = None
  opponent_rule: str = "sample"
  act: str = "weights"
  chance: str = "tree"
  fold_routed: bool = False
  value_scale: float = 1.0
  chance_value: str = "exact"

  def __post_init__(self):
    checks = (
        (self.num_simulations >= 1, "num_simulations >= 1"),
        (self.search in SEARCH_KINDS, f"search in {SEARCH_KINDS}"),
        (self.mode in MODES, f"mode in {MODES}"),
        (self.num_worlds >= 1, "num_worlds >= 1"),
        (self.dirichlet_alpha > 0, "dirichlet_alpha > 0"),
        (0.0 <= self.dirichlet_fraction <= 1.0, "dirichlet_fraction in [0, 1]"),
        (
            self.opponent_rule in ("sample", "argmax"),
            "opponent_rule in (sample, argmax)",
        ),
        (
            self.max_depth is None or self.max_depth >= 1,
            "max_depth None or >= 1",
        ),
        (self.act in ACTS, f"act in {ACTS}"),
        (
            self.act == "weights" or self.num_worlds == 1,
            (
                "act='mctx_action' plays one tree's action, so num_worlds must"
                " be 1"
            ),
        ),
        (self.chance in CHANCES, f"chance in {CHANCES}"),
        (isinstance(self.fold_routed, bool), "fold_routed is a bool"),
        (float(self.value_scale) > 0.0, "value_scale > 0"),
        (
            self.chance_value in CHANCE_VALUES,
            f"chance_value in {CHANCE_VALUES}",
        ),
    )
    for ok, what in checks:
      if not ok:
        raise ValueError(f"MCTSConfig: want {what}; got {self}")

  def to_json(self) -> dict[str, Any]:
    """Every field, JSON-ready (the `mcts` entry of SearchPolicy's arch_kwargs).

    Returns:
      The fields as a dict.
    """
    return dataclasses.asdict(self)


class Node(NamedTuple):
  """A decision-node embedding, leaves [N, ...].

  Attributes:
    state: the env state.
    carries: base carries, leaves [N, P, ...].
    seat: i32, the searching seat.
    actor: i32, the seat to move (the last mover at a terminal node).
    terminal: bool.
    chance_scores: f32[C], the world's fixed chance scores; C = 0 unless
      determinized.
  """

  state: Any
  carries: Any  # base carries, leaves [N, P, ...]
  seat: Any  # i32 searching seat
  actor: Any  # i32 seat to move (the last mover at a terminal node)
  terminal: Any  # bool
  # f32[C] the world's fixed chance scores; C = 0 unless determinized
  chance_scores: Any


class After(NamedTuple):
  """An afterstate (chance-node) embedding: the action applied, chance not.

  Attributes:
    state: the env state after the action.
    carries: base carries, as Node.
    seat: the searching seat.
    actor: the seat that took the action.
    visible: i32[N, P], the action as each seat sees it.
    was_terminal: bool, the action was taken at a terminal node (no-op).
    chance_scores: f32[C], as Node.
  """

  state: Any
  carries: Any
  seat: Any
  actor: Any  # the seat that took the action
  visible: Any  # i32[N, P] the action as each seat sees it
  was_terminal: Any  # bool: the action was taken at a terminal node (no-op)
  chance_scores: Any  # f32[C], as Node


class SearchResult(NamedTuple):
  """Per-world search outputs, [B, K, ...] (evaluate averages over K).

  Attributes:
    worlds: env states [B, K].
    weights: f32[B, K, A], the root visit distribution.
    values: f32[B, K], the root value (the mover's view).
    q: f32[B, K, A], the root Q (the mover's view).
    actions: i32[B, K], the search's own action (mctx PolicyOutput.action).
    logw: f32[B, K], the sampler's log-weight of each world (-inf: no world).
  """

  worlds: Any  # env states [B, K]
  weights: Any  # f32[B, K, A] root visit distribution
  values: Any  # f32[B, K] root value (the mover's view)
  q: Any  # f32[B, K, A] root Q (the mover's view)
  actions: Any  # i32[B, K] the search's own action (mctx PolicyOutput.action)
  logw: Any  # f32[B, K] the sampler's log-weight of each world (-inf: no world)


def _take(x, idx):
  """Each row's entry at `idx`.

  Args:
    x: [B, N, ...] values.
    idx: i32[B] indices.

  Returns:
    The entries.
  """
  return x[jnp.arange(x.shape[0], dtype=jnp.int32), idx]


def _where(cond, a, b):
  """Selects pytree `a` where `cond` holds, else `b`.

  Args:
    cond: the bool mask, broadcast over each leaf's trailing axes.
    a: the pytree taken where `cond` holds.
    b: the pytree taken elsewhere.

  Returns:
    The selected pytree.
  """

  def sel(x, y):
    """Selects `x` where `cond` holds, broadcasting `cond` over trailing axes.

    Args:
      x: the leaf taken where `cond` holds.
      y: the leaf taken elsewhere.

    Returns:
      The selected leaf.
    """
    c = cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim))
    return jnp.where(c, x, y)

  return tree_map(sel, a, b)


def search_chunks(rows: int) -> tuple[int, ...]:
  """Chunk sizes the acting rows of a search are gathered into.

  See core.arena.owned_only: ceil(B/2) (a batch where every seat's rows are
  present, e.g. Policy.evaluate_seq over a two-seat batch) or B.

  Args:
    rows: the batch size B.

  Returns:
    The sorted chunk sizes.
  """
  r = int(rows)
  return tuple(sorted({max(1, -(-r // 2)), r}))


def _check_own_params_model(base: types.Policy, sampler: Any) -> None:
  """Checks that an "own"-params sampler's model takes the base's params.

  A sampler fed the base's params (params_source "own") whose model is a
  policy (`opponent`) must take those params: the two param trees must have
  the same structure, shapes and dtypes.

  Args:
    base: the searched policy.
    sampler: the belief sampler.

  Raises:
    ValueError: the model's params differ from the base's.
  """
  model = getattr(sampler, "opponent", None)
  if sampler.params_source != "own" or model is None:
    return
  key = jax.random.key(0)
  want = jax.eval_shape(base.init_params, key)
  got = jax.eval_shape(model.init_params, key)
  same = jax.tree_util.tree_structure(want) == jax.tree_util.tree_structure(
      got
  ) and all(
      (a.shape, a.dtype) == (b.shape, b.dtype)
      for a, b in zip(
          jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)
      )
  )
  if not same:
    raise ValueError(
        f"the sampler's model {model.spec.arch_name} is fed the searched "
        "base's "
        "params (params_source 'own'), but its params differ from the base's "
        f"({base.spec.arch_name})"
    )


class SearchPolicy(types.Policy):
  """MCTS over `env` with `base` priors and values and `sampler` roots.

  Attributes:
    base: the NATIVE_JAX JAX policy giving priors and values.
    env: the dynamics.
    sampler: the belief sampler drawing the root worlds.
    cfg: the search settings.
    determinized: each world's chance outcomes are fixed.
    stochastic: the tree has chance nodes (zero_sum, not determinized).
    num_actions: the game's number of actions.
    num_players: the game's number of players.
    spec: the PolicySpec.
  """

  decider = "search"
  prior_dependence = "puct_prior+value"
  emits_target_policy = True

  def value_bias(self) -> str:
    """The bias of the search's values.

    "unbiased" for a perfect-information tree with the env's own chance
    nodes, "fused_upper_bound" otherwise: below the root a PIMC tree
    (determinized chance, or worlds drawn from a belief over hidden
    information) knows its world, so its values are optimistic.

    Returns:
      "unbiased" or "fused_upper_bound".
    """
    exact = bool(self.sampler.is_perfect_information) and not self.determinized
    return "unbiased" if exact else "fused_upper_bound"

  def __init__(
      self,
      base: types.Policy,
      env: types.Env,
      sampler: Any,
      mcts_cfg: MCTSConfig,
  ):
    """Builds the search policy.

    Args:
      base: a NATIVE_JAX JAX policy for env's game.
      env: the dynamics; must be a registered game (its EnvSpec is recorded).
      sampler: a registered belief sampler for env.
      mcts_cfg: the search settings.

    Raises:
      TypeError: mcts_cfg is not an MCTSConfig.
      ValueError: a base that is not NATIVE_JAX JAX or does not match env, a
        non-NATIVE_JAX sampler, zero_sum mode on a game that is not
        two-player, gumbel search on a chance-node tree, or fold_routed
        without a RoutedPolicy base or with chance nodes in the tree.
      KeyError: an unregistered env or sampler.
    """
    if not isinstance(mcts_cfg, MCTSConfig):
      raise TypeError(
          f"mcts_cfg must be an MCTSConfig, got {type(mcts_cfg).__name__}"
      )
    belief_registry.check(sampler)
    bs = base.spec
    if bs.capability != types.Capability.NATIVE_JAX or bs.framework != "jax":
      raise ValueError(
          "SearchPolicy runs the base inside mctx: it must be NATIVE_JAX, got "
          f"{bs.capability}/{bs.framework}"
      )
    if (
        bs.game != env.name
        or bs.num_actions != env.num_actions
        or bs.num_players != env.num_players
    ):
      raise ValueError(f"base spec {bs} does not match env {env.name}")
    if sampler.capability != types.Capability.NATIVE_JAX:
      raise ValueError(
          f"the sampler must be NATIVE_JAX, got {sampler.capability}"
      )
    _check_own_params_model(base, sampler)
    self.determinized = mcts_cfg.chance == "determinized"
    self.stochastic = (
        mcts_cfg.mode == "zero_sum"
        and env.max_chance_outcomes > 0
        and not self.determinized
    )
    self._num_scores = int(env.max_chance_outcomes) if self.determinized else 0
    self._route = None
    if mcts_cfg.fold_routed:
      if not isinstance(base, routed_lib.RoutedPolicy):
        raise ValueError(
            "fold_routed folds a RoutedPolicy's override plies into the "
            "dynamics; "
            f"the base is {type(base).__name__}"
        )
      if self.stochastic:
        raise ValueError(
            "fold_routed needs a tree without stochastic chance nodes: "
            f"{env.name} in zero_sum mode needs chance='determinized'"
        )
      self._route = base.route
    if self.stochastic and mcts_cfg.search == "gumbel":
      raise ValueError(
          "a chance-node tree runs mctx.stochastic_muzero_policy (PUCT); "
          "gumbel "
          f"is not available for {env.name} in zero_sum mode"
      )
    if mcts_cfg.mode == "zero_sum" and env.num_players != 2:
      raise ValueError(
          f"zero_sum search is two-player; {env.name} has {env.num_players}"
      )
    env_spec = env_registry.spec_of(env)  # raises for an unregistered env
    self.base, self.env, self.sampler, self.cfg = base, env, sampler, mcts_cfg
    self.num_actions, self.num_players = env.num_actions, env.num_players
    stateful = bool(bs.stateful or sampler.stateful)
    self.spec = types.PolicySpec(
        game=bs.game,
        obs_format=bs.obs_format,
        num_actions=bs.num_actions,
        num_players=bs.num_players,
        stateful=stateful,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=ARCH,
        arch_kwargs=dict(
            base=dict(arch_name=bs.arch_name, arch_kwargs=dict(bs.arch_kwargs)),
            env=env_spec.to_json(),
            sampler=dict(
                name=belief_registry.name_of(sampler),
                kwargs=dict(sampler.kwargs),
            ),
            mcts=mcts_cfg.to_json(),
        ),
    )

  # -- params / carry ----------------------------------------------------------
  @property
  def shares_base_params(self) -> bool:
    """The sampler reads the base params (its params_source is "own")."""
    return self.sampler.params_source == "own"

  def init_params(self, key):
    """The base's and (unless it shares them) the sampler's initial params.

    Args:
      key: a PRNG key, split between the parts.

    Returns:
      {"base", "sampler"} params.
    """
    kb, ks = jax.random.split(key)
    sp = {} if self.shares_base_params else self.sampler.init_params(ks)
    return {"base": self.base.init_params(kb), "sampler": sp}

  def make_params(self, base_params, sampler_params=None) -> dict[str, Any]:
    """The params of this policy from the base's params and the sampler's.

    Args:
      base_params: the base's params.
      sampler_params: the sampler's params, unless the sampler shares the
        base's; None is sampler.init_params().

    Returns:
      {"base", "sampler"} params.

    Raises:
      ValueError: sampler_params for a sampler that reads the base params.
    """
    if self.shares_base_params:
      if sampler_params is not None:
        raise ValueError(
            "the sampler reads the base params; pass no sampler_params"
        )
      return {"base": base_params, "sampler": {}}
    if sampler_params is None:
      sampler_params = self.sampler.init_params(jax.random.key(0))
    return {"base": base_params, "sampler": sampler_params}

  def _sampler_params(self, params):
    """The sampler's params.

    Args:
      params: The search params.

    Returns:
      The params.
    """
    return params["base"] if self.shares_base_params else params["sampler"]

  def encode(self, info):
    """The base's encoding.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      The base's features.
    """
    return self.base.encode(info)

  def init_carry(self, params, batch, seat, key):
    """The base's and the sampler's initial carries.

    Args:
      params: the params.
      batch: the number of rows B.
      seat: the seat of each row.
      key: a PRNG key, split between the parts.

    Returns:
      {"base", "belief"} carries.
    """
    kb, ks = jax.random.split(key)
    return {
        "base": self.base.init_carry(params["base"], batch, seat, kb),
        "belief": self.sampler.init_carry(
            self._sampler_params(params), batch, seat, ks
        ),
    }

  def observe(self, params, carry, step):
    """The base and the sampler observe the step.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].

    Returns:
      The carry after observing `step`.
    """
    return {
        "base": self.base.observe(params["base"], carry["base"], step),
        "belief": self.sampler.observe(
            self._sampler_params(params), carry["belief"], step
        ),
    }

  # -- tree pieces -------------------------------------------------------------
  def _observe_all(self, pb, carries, states, visible, last_actor):
    """Every seat observes the transition into `states`: (carries, steps).

    Args:
      pb: The base params.
      carries: The carries [N, P, ...].
      states: The states [N].
      visible: The visible actions.
      last_actor: The last actors.

    Returns:
      (carries, steps).
    """
    n = jnp.shape(last_actor)[0]
    p = self.num_players
    no = jnp.zeros((n,), jnp.bool_)
    steps = plies.seat_steps(self.env, states, no, no, visible, last_actor)

    def flat(x):
      """Merges the node and seat axes.

      Args:
        x: A leaf, [n, p, ...].

      Returns:
        The leaf, [n * p, ...].
      """
      return x.reshape((n * p,) + x.shape[2:])

    def unflat(x):
      """Splits the node and seat axes.

      Args:
        x: A leaf, [n * p, ...].

      Returns:
        The leaf, [n, p, ...].
      """
      return x.reshape((n, p) + x.shape[1:])

    c = self.base.observe(pb, tree_map(flat, carries), tree_map(flat, steps))
    return tree_map(unflat, c), steps

  def _leaf(self, pb, node: Node, steps, key):
    """The base's prior and value at a node, and whether it is routed.

    Args:
      pb: the base's params.
      node: the Node.
      steps: the SeatSteps the node's carries observed.
      key: a PRNG key.

    Returns:
      (prior logits f32[N, A] with -1e9 on illegal actions, value f32[N],
      routed bool[N]: a live node the base's route hands to its override).
    """
    a_safe = jnp.maximum(node.actor, 0)
    step_a = tree_map(lambda x: _take(x, a_safe), steps)
    c_a = tree_map(lambda x: _take(x, a_safe), node.carries)
    out = self.base.evaluate(pb, c_a, step_a, key)
    logits = jnp.where(step_a.legal_mask, out.logits.astype(F32), F32(NEG))
    persp = node.actor if self.cfg.mode == "zero_sum" else node.seat
    value = _take(out.value.astype(F32), jnp.maximum(persp, 0))
    if self._route is None:
      routed = jnp.zeros(node.terminal.shape, jnp.bool_)
    else:
      routed = jnp.asarray(self._route(step_a.info), jnp.bool_) & ~node.terminal
    return logits, jnp.where(node.terminal, F32(0.0), value), routed

  def _after(self, node: Node, action) -> After:
    """The afterstate of `action` at `node`.

    Args:
      node: The node.
      action: The action.

    Returns:
      The After.
    """
    # mctx's stochastic tree also calls this at chance nodes with an index in
    # [A, A + C); that output is discarded, the clip keeps it a real action.
    a = jnp.clip(jnp.asarray(action).astype(jnp.int32), 0, self.num_actions - 1)
    applied = jax.vmap(self.env.apply)(node.state, a)
    visible = plies.visible_actions(self.env, node.state, a)
    return After(
        state=_where(node.terminal, node.state, applied),
        carries=node.carries,
        seat=node.seat,
        actor=node.actor,
        visible=visible,
        was_terminal=node.terminal,
        chance_scores=node.chance_scores,
    )

  def _resolve_fixed(self, s, scores):
    """One state's chance nodes resolved by the world's fixed scores.

    Determinized: argmax over p > 0 of log p + score.

    Args:
      s: the state.
      scores: f32[C], the world's fixed chance scores.

    Returns:
      The state at its next non-chance node.
    """
    env = self.env

    def cond(st):
      """Whether the state is at a live chance node.

      Args:
        st: the state.

      Returns:
        A scalar bool.
      """
      return env.is_chance(st) & ~env.terminal(st)

    def body(st):
      """Applies the chance outcome the scores pick.

      Args:
        st: the state.

      Returns:
        The next state.
      """
      p = jnp.asarray(env.chance_probs(st), F32)
      logp = jnp.log(jnp.where(p > 0, p, F32(1.0)))
      o = jnp.argmax(jnp.where(p > 0, logp + scores, F32(-jnp.inf))).astype(
          jnp.int32
      )
      return env.apply_chance(st, o)

    return jax.lax.while_loop(cond, body, s)

  def _resolve_rest(self, states, key, scores=None):
    """Resolves any remaining chance nodes.

    By the worlds' fixed scores when determinized, else sampled (open-loop)
    with per-row keys.

    Args:
      states: the states [N].
      key: per-row PRNG keys [N].
      scores: f32[N, C], the worlds' fixed chance scores when determinized.

    Returns:
      The states at their next non-chance nodes.
    """
    if self.env.max_chance_outcomes == 0:
      return states
    if self.determinized:
      assert scores is not None
      return jax.vmap(self._resolve_fixed)(states, scores)
    n = jax.tree_util.tree_leaves(states)[0].shape[0]
    keys = jax.random.split(key, n)
    return jax.vmap(lambda s, k: chance_lib.resolve(self.env, s, k, 0)[0])(
        states, keys
    )

  def _transition(self, pb, after: After, s_new, key):
    """The transition into s_new: (Node, reward, discount, logits, value).

    Args:
      pb: The base params.
      after: The afterstate.
      s_new: The new state.
      key: The key.

    Returns:
      (Node, reward, discount, logits, value, routed).
    """
    env = self.env
    was = after.was_terminal
    s_new = _where(was, after.state, s_new)
    carries, steps = self._observe_all(
        pb, after.carries, s_new, after.visible, after.actor
    )
    carries = _where(was, after.carries, carries)
    term = jax.vmap(env.terminal)(s_new)
    nxt = jax.vmap(env.actor)(s_new).astype(jnp.int32)
    node = Node(
        state=s_new,
        carries=carries,
        seat=after.seat,
        actor=jnp.where(term, after.actor, nxt).astype(jnp.int32),
        terminal=term,
        chance_scores=after.chance_scores,
    )
    rets = jax.vmap(env.returns)(s_new).astype(F32)
    if self.cfg.value_scale != 1.0:
      rets = (rets / F32(self.cfg.value_scale)).astype(F32)
    persp = after.actor if self.cfg.mode == "zero_sum" else after.seat
    reward = jnp.where(
        term & ~was, _take(rets, jnp.maximum(persp, 0)), F32(0.0)
    )
    if self.cfg.mode == "zero_sum":
      disc = jnp.where(nxt == after.actor, F32(1.0), F32(-1.0))
    else:
      disc = jnp.ones_like(reward)
    disc = jnp.where(term, F32(0.0), disc)
    logits, value, routed = self._leaf(pb, node, steps, key)
    return node, reward, disc, logits, value, routed

  def _fold(self, pb, node, reward, disc, logits, value, routed, key):
    """Folds plies into the dynamics until a node the search decides.

    Or until the end of the hand: single_agent, the other seats act (base
    policy, opponent_rule); fold_routed, a routed node plays the override's
    action (argmax of the base's logits, the override's one-hot).

    Args:
      pb: the base's params.
      node: the Node.
      reward: f32[N], the reward so far.
      disc: f32[N], the discount so far.
      logits: f32[N, A], the base's prior logits at the node.
      value: f32[N], the base's value at the node.
      routed: bool[N], the rows the base's route hands to its override.
      key: a PRNG key.

    Returns:
      (node, reward, disc, logits, value) at the first node the search
      decides.
    """
    rule = types.DecisionRule(self.cfg.opponent_rule, 1.0)
    greedy = types.DecisionRule("argmax", 1.0)
    single = self.cfg.mode == "single_agent"

    def active(nd, rt):
      """The rows whose plies are folded: live, and routed or another seat's.

      Args:
        nd: the Node.
        rt: bool[N], the routed rows.

      Returns:
        bool[N].
      """
      act = rt | (nd.actor != nd.seat) if single else rt
      return ~nd.terminal & act

    def cond(c):
      """Whether a ply is left and some row is still folding.

      Args:
        c: the loop state.

      Returns:
        A scalar bool.
      """
      i, nd, rt = c[0], c[1], c[6]
      return (i < self.env.max_len) & jnp.any(active(nd, rt))

    def body(c):
      """Plays one folded ply on the active rows.

      Args:
        c: (ply, node, reward, discount, logits, value, routed).

      Returns:
        The next loop state.
      """
      i, nd, r, d, lg, v, rt = c
      on = active(nd, rt)
      k = x64.fold_in64(key, i)
      kd, kc, kl = jax.random.split(k, 3)
      legal = lg > F32(NEG / 2)
      if single:
        a, _ = decide_lib.decide(lg, legal, rule, kd)
        if self._route is not None:
          a_rt, _ = decide_lib.decide(lg, legal, greedy, kd)
          a = jnp.where(rt, a_rt, a)
      else:
        a, _ = decide_lib.decide(lg, legal, greedy, kd)
      after = self._after(nd, jnp.maximum(a, 0))
      s_new = self._resolve_rest(after.state, kc, after.chance_scores)
      nd_next, r_next, d_next, lg_next, v_next, rt_next = self._transition(
          pb, after, s_new, kl
      )
      return (
          i + jnp.int64(1),
          _where(on, nd_next, nd),
          jnp.where(on, r + d * r_next, r),
          jnp.where(on, d * d_next, d),
          jnp.where(on[:, None], lg_next, lg),
          jnp.where(on, v_next, v),
          jnp.where(on, rt_next, rt),
      )

    _, node, reward, disc, logits, value, _ = jax.lax.while_loop(
        cond, body, (jnp.int64(0), node, reward, disc, logits, value, routed)
    )
    return node, reward, disc, logits, value

  def _recurrent_fn(self, pb, rng, action, node: Node):
    """The mctx recurrent function: one action and its chance resolution.

    Args:
      pb: The base params.
      rng: The key.
      action: The actions.
      node: The nodes.

    Returns:
      (RecurrentFnOutput, the next Node).
    """
    kc, kl, kf = jax.random.split(rng, 3)
    after = self._after(node, action)
    s_new = self._resolve_rest(after.state, kc, after.chance_scores)
    node2, r, d, logits, v, routed = self._transition(pb, after, s_new, kl)
    if self.cfg.mode == "single_agent" or self._route is not None:
      node2, r, d, logits, v = self._fold(
          pb, node2, r, d, logits, v, routed, kf
      )
    return (
        mctx.RecurrentFnOutput(
            reward=r, discount=d, prior_logits=logits, value=v
        ),
        node2,
    )

  def _chance_state(self, after: After, outcome, key):
    """Applies a chance outcome at an afterstate, then resolves the rest.

    Args:
      after: The afterstate.
      outcome: The chance outcome.
      key: The key.

    Returns:
      The state.
    """
    env = self.env
    at_chance = jax.vmap(env.is_chance)(after.state) & ~after.was_terminal
    o = jnp.maximum(jnp.asarray(outcome).astype(jnp.int32), 0)
    dealt = jax.vmap(env.apply_chance)(after.state, o)
    return self._resolve_rest(_where(at_chance, dealt, after.state), key)

  def _decision_fn(self, pb, rng, action, node: Node):
    """The mctx decision function: the afterstate, chance logits and value.

    Args:
      pb: The base params.
      rng: The key.
      action: The actions.
      node: The nodes.

    Returns:
      (DecisionRecurrentFnOutput, the After).
    """
    env = self.env
    kl, kr = jax.random.split(rng)
    after = self._after(node, action)
    c = env.max_chance_outcomes
    at_chance = jax.vmap(env.is_chance)(after.state) & ~after.was_terminal
    probs = jnp.where(
        at_chance[:, None],
        jax.vmap(env.chance_probs)(after.state).astype(F32),
        jax.nn.one_hot(0, c, dtype=F32)[None],
    )
    chance_logits = jnp.where(
        probs > 0, jnp.log(jnp.where(probs > 0, probs, F32(1.0))), -jnp.inf
    ).astype(F32)
    if self.cfg.chance_value == "sampled":
      n = jnp.shape(after.actor)[0]
      ks, kc = jax.random.split(kr)
      o = jax.vmap(jax.random.categorical)(
          jax.random.split(ks, n), chance_logits
      )
      s_o = self._chance_state(after, o.astype(jnp.int32), kc)
      _, r, d, _, v, _ = self._transition(pb, after, s_o, kl)
      return (
          mctx.DecisionRecurrentFnOutput(
              chance_logits=chance_logits,
              afterstate_value=(r + d * v).astype(F32),
          ),
          after,
      )

    def per_outcome(o):
      """The afterstate value of chance outcome `o`.

      Args:
        o: the chance outcome.

      Returns:
        f32[N], reward + discount * value.
      """
      n = jnp.shape(after.actor)[0]
      s_o = self._chance_state(
          after, jnp.full((n,), o, jnp.int32), jax.random.fold_in(kr, o)
      )
      _, r, d, _, v, _ = self._transition(pb, after, s_o, kl)
      return r + d * v

    # [N, C]
    q = jax.vmap(per_outcome, out_axes=1)(jnp.arange(c, dtype=jnp.int32))
    value = jnp.sum(jnp.where(probs > 0, probs * q, F32(0.0)), axis=-1).astype(
        F32
    )
    return (
        mctx.DecisionRecurrentFnOutput(
            chance_logits=chance_logits, afterstate_value=value
        ),
        after,
    )

  def _chance_fn(self, pb, rng, outcome, after: After):
    """The mctx chance function: the chance outcome's transition.

    Args:
      pb: The base params.
      rng: The key.
      outcome: The chance outcomes.
      after: The afterstates.

    Returns:
      (ChanceRecurrentFnOutput, the next Node).
    """
    kl, kr = jax.random.split(rng)
    s_o = self._chance_state(after, outcome, kr)
    node, r, d, logits, v, _ = self._transition(pb, after, s_o, kl)
    return (
        mctx.ChanceRecurrentFnOutput(
            action_logits=logits, value=v, reward=r, discount=d
        ),
        node,
    )

  # -- roots and the search ----------------------------------------------------
  def _root_carries(self, pb, worlds, own_carry, seat, key):
    """[N, P] carries: the seat's own (observed) carry; fresh for the others.

    Args:
      pb: The base params.
      worlds: The root worlds [N].
      own_carry: The seat's observed carry.
      seat: The seat of each world.
      key: The key.

    Returns:
      The carries, leaves [N, P, ...].
    """
    n = seat.shape[0]
    p = self.num_players
    seats = jnp.broadcast_to(jnp.arange(p, dtype=jnp.int32), (n, p)).reshape(-1)
    fresh = self.base.init_carry(pb, n * p, seats, key)
    yes = jnp.ones((n,), jnp.bool_)
    steps = plies.seat_steps(
        self.env,
        worlds,
        yes,
        yes,
        jnp.full((n,), types.NO_ACTION, jnp.int32),
        jnp.full((n,), types.NO_ACTOR, jnp.int32),
    )
    fresh = self.base.observe(
        pb, fresh, tree_map(lambda x: x.reshape((n * p,) + x.shape[2:]), steps)
    )
    fresh = tree_map(lambda x: x.reshape((n, p) + x.shape[1:]), fresh)
    own = tree_map(
        lambda x: jnp.broadcast_to(x[:, None], (n, p) + x.shape[1:]), own_carry
    )
    mine = jnp.arange(p, dtype=jnp.int32)[None, :] == seat[:, None]
    return _where(mine, own, fresh)

  def _run(self, pb, root: mctx.RootFnOutput, legal, key):
    """Runs the configured mctx search from `root`.

    Args:
      pb: The base params.
      root: The root.
      legal: The legal mask.
      key: The key.

    Returns:
      mctx's PolicyOutput.
    """
    cfg = self.cfg
    invalid = ~legal
    if self.stochastic:
      return mctx.stochastic_muzero_policy(
          params=pb,
          rng_key=key,
          root=root,
          decision_recurrent_fn=self._decision_fn,
          chance_recurrent_fn=self._chance_fn,
          num_simulations=cfg.num_simulations,
          invalid_actions=invalid,
          max_depth=cfg.max_depth,
          dirichlet_fraction=cfg.dirichlet_fraction,
          dirichlet_alpha=cfg.dirichlet_alpha,
          pb_c_init=cfg.pb_c_init,
          pb_c_base=cfg.pb_c_base,
      )
    if cfg.search == "muzero":
      return mctx.muzero_policy(
          params=pb,
          rng_key=key,
          root=root,
          recurrent_fn=self._recurrent_fn,
          num_simulations=cfg.num_simulations,
          invalid_actions=invalid,
          max_depth=cfg.max_depth,
          dirichlet_fraction=cfg.dirichlet_fraction,
          dirichlet_alpha=cfg.dirichlet_alpha,
          pb_c_init=cfg.pb_c_init,
          pb_c_base=cfg.pb_c_base,
      )
    return mctx.gumbel_muzero_policy(
        params=pb,
        rng_key=key,
        root=root,
        recurrent_fn=self._recurrent_fn,
        num_simulations=cfg.num_simulations,
        invalid_actions=invalid,
        max_depth=cfg.max_depth,
        gumbel_scale=cfg.gumbel_scale,
    )

  def _search_flat(
      self,
      pb,
      worlds,
      carries,
      seat,
      root_logits,
      root_value,
      legal,
      key,
      scores,
  ):
    """Searches flat worlds from their root priors and values.

    Args:
      pb: The base params.
      worlds: The worlds.
      carries: Their carries.
      seat: Their seats.
      root_logits: The root priors.
      root_value: The root values.
      legal: The legal mask.
      key: The key.
      scores: The chance scores.

    Returns:
      (mctx output, action weights, root value, root Q-values).
    """
    root = mctx.RootFnOutput(
        prior_logits=root_logits.astype(F32),
        value=root_value.astype(F32),
        embedding=Node(
            state=worlds,
            carries=carries,
            seat=seat,
            actor=seat,
            terminal=jnp.zeros(seat.shape, jnp.bool_),
            chance_scores=scores.astype(F32),
        ),
    )
    out = self._run(pb, root, legal, key)
    summary = out.search_tree.summary()
    return (
        out,
        out.action_weights.astype(F32),
        summary.value.astype(F32),
        summary.qvalues.astype(F32),
    )

  def search(self, params, carry, step, key, rows=None) -> SearchResult:
    """The per-world search at step (the seat's SeatStep, already observed).

    Rows whose seat is not to move (or, given `rows` bool[B], outside rows)
    get zeros; with B > 1 they are not searched, with B == 1 the row is
    searched anyway and its result zeroed.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.
      rows: bool[B], the rows to search; None is every row.

    Returns:
      The SearchResult.
    """
    x64.assert_x64("SearchPolicy.search")
    seat = jnp.asarray(step.seat).astype(jnp.int32)
    acting = jnp.asarray(step.to_move) == seat
    if rows is not None:
      acting = acting & jnp.asarray(rows, jnp.bool_)
    if self._route is not None:  # a routed root is not searched (module doc)
      acting = acting & ~jnp.asarray(self._route(step.info), jnp.bool_)
    b = seat.shape[0]
    if b == 1:
      # One row: the Arena's per-row program (vmapped over the acting rows,
      # core.arena.slot_rows) or a single request. Gating cannot save work
      # under vmap (a cond becomes a select running both branches), so the
      # row searches unconditionally; a non-acting row searches over every
      # action and evaluate zeroes it.
      return self._search_rows(params, carry, step, key, acting)
    # owned_only needs a per-row program, because a row's result must not
    # depend on the chunk it is gathered into: each row searches alone (vmap)
    # with its own key, the row's index folded in.
    keys = jax.vmap(lambda i: x64.fold_in64(key, i))(
        jnp.arange(b, dtype=jnp.int64)
    )

    def one(c, s, k):
      """The search of one row.

      Args:
        c: the row's carry.
        s: the row's SeatStep.
        k: the row's PRNG key.

      Returns:
        The row's SearchResult.
      """
      res = self._search_rows(
          params,
          tree_map(lambda x: x[None], c),
          tree_map(lambda x: x[None], s),
          k,
          None,
      )
      return tree_map(lambda x: x[0], res)

    def fn(xs):
      """The vmapped search over the owned rows.

      Args:
        xs: (carry, step, keys) of the owned rows.

      Returns:
        The per-row results.
      """
      return jax.vmap(one)(*xs)

    return arena_lib.owned_only(
        acting, fn, (carry, step, keys), search_chunks(b)
    )

  def _search_rows(self, params, carry, step, key, acting) -> SearchResult:
    """The search of `step`'s rows.

    acting None: every row's seat is to move; else a non-acting row searches
    over every action (finite garbage).

    Args:
      params: the policy's params.
      carry: the policy's carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.
      acting: bool[B], the rows whose seat is to move, or None.

    Returns:
      The SearchResult.
    """
    k = self.cfg.num_worlds
    k_world, k_search, k_eval, k_init = jax.random.split(key, 4)
    pb = params["base"]
    seat = jnp.asarray(step.seat).astype(jnp.int32)
    b = seat.shape[0]
    sp = self._sampler_params(params)
    draw = getattr(self.sampler, "sample_determinized", None)
    if self.determinized and draw is not None:
      det = draw(sp, carry["belief"], step.info, k, k_world)
      worlds, scores = det.states, det.chance_scores
      logw = jnp.zeros((b, k), F32)
    else:
      # The K worlds are draws from the belief (log-weight 0). A sampler marks
      # a world it could not draw (no consistent world, a dead particle) with
      # log-weight -inf; evaluate averages over the finite worlds only.
      worlds, logw = self.sampler.sample_worlds(
          sp, carry["belief"], step.info, k, k_world
      )
      logw = jnp.asarray(logw).astype(F32)
      scores = jax.random.gumbel(
          jax.random.fold_in(k_world, 1), (b, k, self._num_scores), F32
      )
    assert scores.shape == (b, k, self._num_scores), (
        scores.shape,
        self._num_scores,
    )
    legal = step.legal_mask
    if acting is not None:
      legal = jnp.where(acting[:, None], legal, True)
    out = self.base.evaluate(pb, carry["base"], step, k_eval)
    root_logits = jnp.where(legal, out.logits.astype(F32), F32(NEG))
    # an override's -inf logits (routed roots are not used)
    if self._route is not None:
      root_logits = jnp.maximum(root_logits, F32(NEG))
    root_value = _take(out.value.astype(F32), seat)

    def rep(x):
      """Repeats each row k times.

      Args:
        x: A leaf, [B, ...].

      Returns:
        The leaf, [B * k, ...].
      """
      return jnp.repeat(x, k, axis=0)

    flat_w = tree_map(lambda x: x.reshape((b * k,) + x.shape[2:]), worlds)
    carries = self._root_carries(
        pb, flat_w, tree_map(rep, carry["base"]), rep(seat), k_init
    )
    out, w, v, q = self._search_flat(
        pb,
        flat_w,
        carries,
        rep(seat),
        rep(root_logits),
        rep(root_value),
        rep(legal),
        k_search,
        scores.reshape((b * k, self._num_scores)),
    )
    a = self.num_actions
    return SearchResult(
        worlds=worlds,
        weights=w.reshape(b, k, a),
        values=v.reshape(b, k),
        q=q.reshape(b, k, a),
        actions=jnp.asarray(out.action).astype(jnp.int32).reshape(b, k),
        logw=logw,
    )

  def search_states(self, params, states, key):
    """Searches from known states instead of sampled worlds, for analysis.

    The mover of each state searches from a fresh base carry. evaluate, not
    this, is how the policy acts: its roots come from the sampler.

      out = policy.search_states(params, states, key)  # states: leaves [N, ...]
      visits = out.search_tree.summary().visit_counts  # i32[N, A]
      q = out.search_tree.summary().qvalues            # f32[N, A], mover's view

    Args:
      params: the params.
      states: env states, leaves [N, ...].
      key: a PRNG key.

    Returns:
      The mctx PolicyOutput of N independent searches.
    """
    x64.assert_x64("SearchPolicy.search_states")
    pb = params["base"]
    env = self.env
    seat = jnp.maximum(jax.vmap(env.actor)(states), 0).astype(jnp.int32)
    n = seat.shape[0]
    # One stream per use: the tree search draws from `key`, the chance
    # scores from fold_in(key, 1), the base's carries and leaf from
    # fold_in(key, 2).
    k_init, k_root, k_leaf = jax.random.split(jax.random.fold_in(key, 2), 3)
    carries = self._root_carries(
        pb, states, self.base.init_carry(pb, n, seat, k_init), seat, k_root
    )
    steps = plies.seat_steps(
        env,
        states,
        jnp.ones((n,), jnp.bool_),
        jnp.ones((n,), jnp.bool_),
        jnp.full((n,), types.NO_ACTION, jnp.int32),
        jnp.full((n,), types.NO_ACTOR, jnp.int32),
    )
    scores = jax.random.gumbel(
        jax.random.fold_in(key, 1), (n, self._num_scores), F32
    )
    node = Node(
        state=states,
        carries=carries,
        seat=seat,
        actor=seat,
        terminal=jnp.zeros((n,), jnp.bool_),
        chance_scores=scores,
    )
    logits, value, _ = self._leaf(pb, node, steps, k_leaf)
    legal = jax.vmap(env.legal_mask)(states)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=node)
    return self._run(pb, root, legal, key)

  # -- Policy ------------------------------------------------------------------
  def evaluate(self, params, carry, step, key):
    """The search's action distribution (logits = log visit weights).

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.

    Returns:
      The PolicyOutput with the search aux.
    """
    return self._evaluate(params, carry, step, key, None)

  def evaluate_rows(self, params, carry, step, key, rows):
    """Evaluate for the rows where `rows`, the RoutedPolicy gating hook.

    See policies.routed: only those rows are searched, the others' outputs
    are zeros (unused by the caller). On its rows it equals evaluate (each
    row's search is its own program). A RoutedPolicy over a SearchPolicy
    calls this instead of wrapping the search in a lax.cond (a cond under the
    Arena's per-row vmap becomes a select over both branches, and would put
    a routed base's custom_vmap gate inside a batched cond).

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.
      rows: bool[B], the rows to evaluate.

    Returns:
      The PolicyOutput.
    """
    return self._evaluate(
        params, carry, step, key, jnp.asarray(rows, jnp.bool_)
    )

  def _evaluate(self, params, carry, step, key, rows):
    """The search's PolicyOutput: world means on acting rows, else zeros.

    Args:
      params: The params.
      carry: The carry.
      step: The SeatStep.
      key: The key.
      rows: The rows to evaluate, or None for all.

    Returns:
      The PolicyOutput, the search targets in aux.
    """
    res = self.search(params, carry, step, key, rows)
    seat = jnp.asarray(step.seat).astype(jnp.int32)
    acting = jnp.asarray(step.to_move) == seat
    if rows is not None:
      acting = acting & rows
    # Worlds with a non-finite log-weight are not draws from the belief: a row
    # averages its finite worlds, and a row with none gets NaN (a bad ply).
    ok = jnp.isfinite(res.logw)
    all_ok = jnp.all(ok, axis=1)
    n_ok = jnp.sum(ok, axis=1).astype(F32)

    def world_mean(x):  # [B, K, ...] -> [B, ...]
      """The mean over the worlds; over the finite ones when some are not.

      A row with no finite world gets NaN.

      Args:
        x: [B, K, ...].

      Returns:
        f32[B, ...].
      """
      tail = (1,) * (x.ndim - 2)
      live = jnp.sum(
          jnp.where(ok.reshape(ok.shape + tail), x, jnp.zeros((), x.dtype)),
          axis=1,
      ) / n_ok.reshape(n_ok.shape + tail)
      return jnp.where(
          all_ok.reshape(all_ok.shape + tail), jnp.mean(x, axis=1), live
      ).astype(F32)

    w = world_mean(res.weights)
    v = world_mean(res.values)
    q = world_mean(res.q)
    if self.cfg.act == "mctx_action":
      hot = jax.nn.one_hot(res.actions[:, 0], self.num_actions, dtype=F32) > 0
      logits = jnp.where(hot, F32(0.0), F32(-jnp.inf))
      logits = jnp.where(all_ok[:, None], logits, F32(jnp.nan))
    else:
      logits = jnp.log(w + F32(1e-8)).astype(F32)
    oh = jax.nn.one_hot(seat, self.num_players, dtype=F32)
    sign = (F32(2.0) * oh - F32(1.0)) if self.cfg.mode == "zero_sum" else oh
    value = sign * v[:, None]
    if self._route is not None:  # routed roots: the base's (override's) output
      use = acting & jnp.asarray(self._route(step.info), jnp.bool_)
      ob = self.base.evaluate(
          params["base"], carry["base"], step, jax.random.fold_in(key, 3)
      )
      ob_logits = ob.logits.astype(F32)
      ob_w = jax.nn.softmax(
          jnp.where(step.legal_mask, ob_logits, F32(-jnp.inf)), axis=-1
      )
      col = use[:, None]
      logits = jnp.where(col, ob_logits, logits)
      w = jnp.where(col, ob_w.astype(F32), w)
      value = jnp.where(col, ob.value.astype(F32), value)
      v = jnp.where(use, _take(ob.value.astype(F32), seat), v)
      q = jnp.where(col, F32(0.0), q)

    def z(x):
      """Zeros the rows that are not acting.

      Args:
        x: A leaf, [B, ...].

      Returns:
        The masked leaf.
      """
      return jnp.where(
          acting.reshape(acting.shape + (1,) * (x.ndim - 1)),
          x,
          jnp.zeros((), x.dtype),
      )

    return types.PolicyOutput(
        logits=z(logits),
        value=z(value.astype(F32)),
        aux={
            types.AUX_TARGET_POLICY: z(w),
            AUX_SEARCH_VALUE: z(v),
            AUX_SEARCH_Q: z(q),
        },
    )


def build(base, env, sampler, mcts) -> SearchPolicy:
  """The registry constructor: a SearchPolicy from its JSON arch_kwargs.

  Args:
    base: the base's {"arch_name", "arch_kwargs"} record.
    env: the EnvSpec JSON.
    sampler: the sampler's {"name", "kwargs"} record.
    mcts: the MCTSConfig fields.

  Returns:
    The SearchPolicy.
  """
  base_pol = registry.build(base["arch_name"], base["arch_kwargs"])
  env_obj = env_registry.make(env_registry.EnvSpec.from_json(env))
  sampler_obj = belief_registry.build(
      sampler["name"], env_obj, **sampler["kwargs"]
  )
  return SearchPolicy(base_pol, env_obj, sampler_obj, MCTSConfig(**mcts))


build.policy_class = SearchPolicy
registry.register(ARCH, build)
