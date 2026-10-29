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

"""One policy per row, chosen by an InfoState predicate.

Some games split naturally into phases played by different policies: a
network for the strategic decisions and a script for a mechanical endgame,
for example. RoutedPolicy composes two policies by a route predicate
`route(info) -> bool[B]`: the override plays the rows where it is True, the
base plays the others. Both parts observe every ply (each keeps its own
carry), so either can take over mid-hand with an up-to-date state.

  def late(info):                              # a "module:attr" route
    return info.num_moves >= 40
  policy = RoutedPolicy(network, endgame_script, "my_game.routes:late")
  params = policy.make_params(network_params, script_params)

The parts are used as written. logits, value and shared aux keys are
selected per row; aux keys of one part only are passed through. Params
{"base", "override"}; carry {"base", "override"}. Capability is
CALLBACK_BATCHED if either part is, else NATIVE_JAX; a HOST_DRIVEN part is
not supported (it cannot run inside the in-graph select). arch_kwargs nest
each part's (arch_name, arch_kwargs), so the registry rebuilds the whole
policy, and each part's params are serialised by its own framework's codec.

Gating. Each part is evaluated only on its own rows. A part with an
`evaluate_rows(params, carry, step, key, rows)` method (a nested
RoutedPolicy, a SearchPolicy) receives the row mask. Any other part runs
under lax.cond(any(rows)), so a batch with no row for it skips it. Under the
Arena's per-row vmap a cond would become a select that computes both
branches on every row, so there the override runs through a custom vmap
rule on the rows routed to it only, gathered into core.arena.on_rows'
chunks, and the base runs through a rule that keeps its params unbatched
(one copy of the network). A part's outputs on rows that are not its own are
zeros and never used.

Scripted phases. `with_scripted_phase(policy, endgame)` routes a game's
scripted final phase (an architecture with a `phase_route`) around any
policy; `routed_pair` does the same for (policy, params); `unwrap_routed`
returns the base of such a composition. `override_rows` gives, for any
leading batch shape, the rows a RoutedPolicy hands to its override.
"""

from typing import Any, Callable, Mapping, Optional, Union
import weakref

import jax
from jax import custom_batching
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import sequence
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry

ARCH = "routed"
PartSpec = Mapping[str, Any]  # {"arch_name": str, "arch_kwargs": {...}}


def _select(cond, a, b):
  """Selects `a` where `cond`, broadcast over trailing axes.

  Args:
    cond: The condition.
    a: The values where it holds.
    b: The values elsewhere.

  Returns:
    The selection.
  """
  c = cond.reshape(cond.shape + (1,) * (a.ndim - cond.ndim))
  return jnp.where(c, a, b)


class RoutedPolicy(types.Policy):
  """`override` on the rows where route(info), `base` elsewhere (module doc).

  Attributes:
    base: the in-graph policy on the rows the route does not take.
    override: the in-graph policy on the rows where route(info).
    route: route(info) -> bool[B], resolved from its "module:attr".
    spec: the PolicySpec of the composite.

  Raises:
    ValueError: route is not a "module:attr" string, or the parts play
      different games.
    NotImplementedError: a HOST_DRIVEN part.
  """

  #: The attributes holding the two parts (policies.members recurses into them).
  routed_parts = ("base", "override")

  def __init__(
      self,
      base: Union[types.Policy, PartSpec],
      override: Union[types.Policy, PartSpec],
      route: str,
  ):
    """Builds the composite.

    Args:
      base: an in-graph policy, or its {"arch_name", "arch_kwargs"} record.
      override: an in-graph policy, or its {"arch_name", "arch_kwargs"}
        record.
      route: "module:attr" of route(info) -> bool[B].

    Raises:
      ValueError: route is not a "module:attr" string, or the parts play
        different games.
      NotImplementedError: a HOST_DRIVEN part.
    """
    if not (isinstance(route, str) and ":" in route):
      raise ValueError(f"route is a 'module:attr' string, got {route!r}")
    self.base, self.override = registry.build_part(base), registry.build_part(
        override
    )
    self.route = registry.resolve_callable(route)
    sb, so = self.base.spec, self.override.spec
    for s in (sb, so):
      if s.capability == types.Capability.HOST_DRIVEN:
        raise NotImplementedError(
            f"RoutedPolicy composes in-graph policies; {s.arch_name} is "
            f"{s.capability.value}/{s.framework}"
        )
    for f in ("game", "num_actions", "num_players"):
      if getattr(sb, f) != getattr(so, f):
        raise ValueError(
            f"RoutedPolicy parts disagree on {f}: {getattr(sb, f)!r} vs "
            f"{getattr(so, f)!r}"
        )
    cap = (
        types.Capability.CALLBACK_BATCHED
        if types.Capability.CALLBACK_BATCHED in (sb.capability, so.capability)
        else types.Capability.NATIVE_JAX
    )
    self.spec = types.PolicySpec(
        game=sb.game,
        obs_format=sb.obs_format,
        num_actions=sb.num_actions,
        num_players=sb.num_players,
        stateful=sb.stateful or so.stateful,
        capability=cap,
        # A composite is in its non-JAX part's framework, if it has one.
        framework=so.framework if sb.framework == "jax" else sb.framework,
        arch_name=ARCH,
        arch_kwargs=dict(
            base=dict(arch_name=sb.arch_name, arch_kwargs=dict(sb.arch_kwargs)),
            override=dict(
                arch_name=so.arch_name, arch_kwargs=dict(so.arch_kwargs)
            ),
            route=route,
        ),
    )

  @property
  def decider(self) -> str:
    """The base's decider: who chooses on rows the route does not take."""
    return types.decider_of(self.base)

  def decider_for(self, info) -> str:
    """The decider of one request whose last-ply InfoState is `info` [1, ...].

    It is the override's where route(info) holds, else the base's (nested
    routes recurse).

    Args:
      info: the request's last-ply InfoState, leaves [1, ...].

    Returns:
      The decider.
    """
    use = np.asarray(self.route(info), np.bool_)
    assert use.shape == (1,), use.shape
    part = self.override if bool(use[0]) else self.base
    f = getattr(part, "decider_for", None)
    return f(info) if f is not None else types.decider_of(part)

  def to(self, device: str) -> "RoutedPolicy":
    """Places every part that has a `to(device)` method on `device`.

    Args:
      device: the device name.

    Returns:
      self.

    Raises:
      TypeError: no part can be placed.
    """
    parts = []
    for p in (self.base, self.override):
      if callable(getattr(p, "to", None)):
        parts.append(p)
    if not parts:
      raise TypeError(
          f"RoutedPolicy({type(self.base).__name__}, "
          f"{type(self.override).__name__}): no part can be placed on a device"
      )
    for p in parts:
      p.to(device)
    return self

  def make_params(self, base_params, override_params):
    """The params of the two parts.

    Args:
      base_params: the base's params.
      override_params: the override's params.

    Returns:
      {"base": base_params, "override": override_params}.
    """
    return {"base": base_params, "override": override_params}

  def init_params(self, key=None):
    """Both parts' initial params.

    Args:
      key: a PRNG key, split between the parts; None uses key 0.

    Returns:
      {"base", "override"} params.
    """
    kb, ko = jax.random.split(jax.random.key(0) if key is None else key)
    return self.make_params(
        self.base.init_params(kb), self.override.init_params(ko)
    )

  def encode(self, info):
    """The base's encoding.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      The base's features.
    """
    return self.base.encode(info)

  def init_carry(self, params, batch, seat, key):
    """Both parts' initial carries.

    Args:
      params: the params.
      batch: the number of rows B.
      seat: the seat of each row.
      key: a PRNG key, or None.

    Returns:
      {"base", "override"} carries.
    """
    kb = ko = key
    if key is not None:
      kb, ko = jax.random.fold_in(key, 0), jax.random.fold_in(key, 1)
    return {
        "base": self.base.init_carry(params["base"], batch, seat, kb),
        "override": self.override.init_carry(
            params["override"], batch, seat, ko
        ),
    }

  def observe(self, params, carry, step):
    """Both parts observe the step.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].

    Returns:
      The carry after observing `step`.
    """
    return {
        "base": self.base.observe(params["base"], carry["base"], step),
        "override": self.override.observe(
            params["override"], carry["override"], step
        ),
    }

  # Both parts are in-graph (JAX, or a host part behind jax.pure_callback), so
  # the sequence forms are the core scans whatever the framework label says.
  def observe_seq(self, params, carry, steps):
    """Observes a sequence of steps by a scan of observe.

    Args:
      params: the params.
      carry: the carry.
      steps: SeatSteps, leaves [B, T, ...].

    Returns:
      The carry after observing every step.
    """
    return sequence.scan_observe(self, params, carry, steps)

  def evaluate_seq(self, params, carry, steps, keys):
    """Evaluates a sequence of steps by a scan of observe and evaluate.

    Args:
      params: the params.
      carry: the carry.
      steps: SeatSteps, leaves [B, T, ...].
      keys: [T] keys (ply t's evaluate key), or None.

    Returns:
      (final carry, PolicyOutput [B, T, ...]).
    """
    return sequence.scan_evaluate(self, params, carry, steps, keys)

  def evaluate(self, params, carry, step, key):
    """Evaluate on every row.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.

    Returns:
      The PolicyOutput.
    """
    rows = jnp.ones(jnp.shape(step.legal_mask)[:1], jnp.bool_)
    return self.evaluate_rows(params, carry, step, key, rows)

  def evaluate_rows(self, params, carry, step, key, rows):
    """Evaluate for the rows where `rows`.

    Each part runs only on its own rows (module doc); other rows' outputs are
    unused.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.
      rows: bool[B], the rows to evaluate.

    Returns:
      The PolicyOutput: the override's where route() holds, the base's
      elsewhere.

    Raises:
      ValueError: route() returns the wrong shape, or the parts disagree on an
        aux key's shape or dtype.
    """
    kb = ko = key
    if key is not None:
      kb, ko = jax.random.fold_in(key, 0), jax.random.fold_in(key, 1)
    use = jnp.asarray(self.route(step.info), jnp.bool_)
    if use.shape != step.legal_mask.shape[:1]:
      raise ValueError(
          f"route returned shape {use.shape}; want {step.legal_mask.shape[:1]}"
      )
    rows = jnp.asarray(rows, jnp.bool_)
    ob = _part_evaluate(
        self.base,
        params["base"],
        carry["base"],
        step,
        kb,
        rows & ~use,
        vmap_gate=False,
    )
    oo = _part_evaluate(
        self.override,
        params["override"],
        carry["override"],
        step,
        ko,
        rows & use,
    )
    aux = dict(ob.aux)
    for k, v in oo.aux.items():
      if k in aux and aux[k].shape == v.shape and aux[k].dtype == v.dtype:
        aux[k] = _select(use, v, aux[k])
      elif k not in aux:
        aux[k] = v
      else:
        raise ValueError(
            f"aux {k!r}: parts disagree on shape/dtype "
            f"{aux[k].shape}/{aux[k].dtype} vs {v.shape}/{v.dtype}"
        )
    return types.PolicyOutput(
        logits=_select(
            use, oo.logits.astype(jnp.float32), ob.logits.astype(jnp.float32)
        ),
        value=_select(
            use, oo.value.astype(jnp.float32), ob.value.astype(jnp.float32)
        ),
        aux=aux,
    )


def _part_evaluate(
    part, params, carry, step, key, rows, vmap_gate: bool = True
) -> types.PolicyOutput:
  """Part's PolicyOutput where `rows`.

  The part is not evaluated on a batch without a row for it (module doc,
  gating).

  vmap_gate: under the Arena's per-row vmap, run a JAX part only on its own
  rows, gathered into on_rows' chunks. The base is called with False: it
  usually owns most rows, so gathering buys nothing and would compile one
  copy of the network per chunk size. It still runs through a vmap rule,
  never through a vmapped lax.cond: a cond with a batched predicate becomes a
  select over both branches with every operand batched, closed-over params
  included, so each row would get its own copy of the weights and every
  matmul would become a batched one.

  Args:
    part: the part policy.
    params: the part's params.
    carry: the part's carry.
    step: the SeatStep, leaves [B, ...].
    key: a PRNG key.
    rows: bool[B], the rows the part plays.
    vmap_gate: see above.

  Returns:
    The part's PolicyOutput.
  """
  fn = getattr(part, "evaluate_rows", None)
  if fn is not None:
    return fn(params, carry, step, key, rows)
  return _gated(part, chunked=vmap_gate)(params, carry, step, key, rows)


# (id(part), chunked) -> gated fn. The functions reach the part only through
# a weak reference and a finalizer drops the part's entries when it is freed,
# so the cache never keeps a part (or the weights it closes over) alive.
_GATED: dict[tuple[int, bool], Callable[..., Any]] = {}


def _gated(part, chunked: bool = True):
  """_cond_evaluate of `part` with a vmap rule (jax.custom_batching).

  Vmapped one row per call, as the Arena calls a slot policy, the part runs
  as a vmap of part.evaluate with its params unbatched; rows without a row
  for the part get zeros. chunked: only the vmapped rows that have a row for
  the part run, gathered into core.arena.on_rows' chunks (one compiled copy
  of the part per chunk size); otherwise every vmapped row runs (one copy)
  and the others are zeroed. Unbatched (one call on a batch) it is
  _cond_evaluate itself.

  Args:
    part: the part policy.
    chunked: see above.

  Returns:
    gated(params, carry, step, key, rows) -> the part's PolicyOutput.
  """
  cache_key = (id(part), bool(chunked))
  hit = _GATED.get(cache_key)
  if hit is not None:
    return hit
  part_ref = weakref.ref(part)
  del part  # the functions below reach it through part_ref only

  @custom_batching.custom_vmap
  def gated(params, carry, step, key, rows):
    """_cond_evaluate of the part, unbatched.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.
      rows: bool[B], the rows to evaluate.

    Returns:
      The part's PolicyOutput, zeros when no row is for it.
    """
    return _cond_evaluate(part_ref(), params, carry, step, key, rows)

  @gated.def_vmap
  def _rule(axis_size, in_batched, params, carry, step, key, rows):
    """The batching rule: the part on the rows routed to it.

    Args:
      axis_size: The batch size.
      in_batched: Which arguments are batched.
      params: The params.
      carry: The carry.
      step: The SeatStep.
      key: The key.
      rows: The rows routed to the part.

    Returns:
      (the outputs, their batched flags).
    """
    args = [params, carry, step, key, rows]

    def is_none(x):
      """Whether `x` is None.

      Args:
        x: A value.

      Returns:
        True for None.
      """
      return x is None

    axes = jax.tree_util.tree_map(lambda b: 0 if b else None, list(in_batched))
    # The gathered rows all have a row for the part (padding past the count
    # is dropped by on_rows): evaluate it directly, no cond.
    part = part_ref()

    def one(p, c, st, k, row_mask):
      """The part's evaluation, rows ignored.

      Args:
        p: The part's params.
        c: The carry.
        st: The SeatStep.
        k: The key.
        row_mask: Unused: every gathered row is the part's.

      Returns:
        The PolicyOutput.
      """
      del row_mask
      return part.evaluate(p, c, st, k)

    r = (
        rows
        if in_batched[4]
        else jnp.broadcast_to(rows, (axis_size,) + jnp.shape(rows))
    )
    need = jnp.any(jnp.reshape(r, (axis_size, -1)), axis=1)

    def compute(idx):
      """The part's outputs on the vmapped rows `idx` (all of them for None).

      Args:
        idx: i32 row indices, or None.

      Returns:
        The vmapped PolicyOutput.
      """
      if idx is None:
        sub, n = args, axis_size
      else:
        sub = jax.tree_util.tree_map(
            lambda ax, x: x if ax is None else x[idx],
            axes,
            args,
            is_leaf=is_none,
        )
        n = idx.shape[0]
      return jax.vmap(one, in_axes=tuple(axes), axis_size=n)(*sub)

    if chunked:
      shapes = jax.eval_shape(compute, None)
      zeros = jax.tree_util.tree_map(
          lambda sd: jnp.zeros(sd.shape, sd.dtype), shapes
      )
      out = arena_lib.on_rows(need, compute, zeros)
    else:
      full = compute(None)
      out = jax.tree_util.tree_map(
          lambda x: _select(need, x, jnp.zeros_like(x)), full
      )
    return out, jax.tree_util.tree_map(lambda _: True, out)

  _GATED[cache_key] = gated
  weakref.finalize(part_ref(), _GATED.pop, cache_key, None)
  return gated


def _cond_evaluate(part, params, carry, step, key, rows) -> types.PolicyOutput:
  """part.evaluate under lax.cond(any(rows)), zeros when no row is for it.

  Args:
    part: The policy part.
    params: Its params.
    carry: The carry.
    step: The SeatStep.
    key: The key.
    rows: The rows routed to the part.

  Returns:
    The PolicyOutput, zeros when no row is the part's.
  """

  def run(_):
    """The part's evaluation.

    Args:
      _: the unused cond operand.

    Returns:
      The PolicyOutput.
    """
    return part.evaluate(params, carry, step, key)

  shapes = jax.eval_shape(run, None)

  def skip(_):
    """Zeros of the evaluation's shapes.

    Args:
      _: the unused cond operand.

    Returns:
      A zero PolicyOutput.
    """
    return jax.tree_util.tree_map(
        lambda sd: jnp.zeros(sd.shape, sd.dtype), shapes
    )

  return jax.lax.cond(jnp.any(rows), run, skip, None)


registry.register(ARCH, RoutedPolicy)


def override_rows(policy: RoutedPolicy, info) -> jax.Array:
  """Where `policy` hands the decision to its override.

  Works for an InfoState with any leading shape ([B], [B, T], ...): the
  leading shape is that of the InfoState's lowest-rank leaf, which every leaf
  must start with, and route() runs on the flattened rows.

  Args:
    policy: the RoutedPolicy.
    info: the InfoState, leaves [*lead, ...].

  Returns:
    bool[*lead].

  Raises:
    TypeError: `policy` is not a RoutedPolicy.
    ValueError: an empty InfoState or inconsistent leading shapes.
  """
  if not isinstance(policy, RoutedPolicy):
    raise TypeError(
        f"override_rows needs a RoutedPolicy, got {type(policy).__name__}"
    )
  leaves = jax.tree_util.tree_leaves(info)
  if not leaves:
    raise ValueError("empty InfoState")
  lead = min((jnp.shape(x) for x in leaves), key=len)
  if not lead or any(jnp.shape(x)[: len(lead)] != lead for x in leaves):
    raise ValueError(
        "InfoState leaves need a common leading batch shape; got "
        f"{[jnp.shape(x) for x in leaves]}"
    )
  n = int(np.prod(lead))
  flat = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, (n,) + jnp.shape(x)[len(lead) :]), info
  )
  use = jnp.asarray(policy.route(flat), jnp.bool_)
  return use.reshape(lead)


# -- scripted phases -----------------------------------------------------------
def _phase_route(override: types.Policy) -> Optional[str]:
  """The "module:attr" route an endgame policy declares, or None.

  That is its `phase_route`, or its class's ROUTE.

  Args:
    override: the endgame policy.

  Returns:
    The route, or None.
  """
  return getattr(override, "phase_route", None) or getattr(
      type(override), "ROUTE", None
  )


def is_scripted_phase(
    policy: types.Policy,
    endgame: Optional[str] = None,
    route: Optional[str] = None,
) -> bool:
  """Whether `policy` routes a scripted phase on its own route.

  That is a RoutedPolicy whose override is a scripted phase played on its
  own route.

  Args:
    policy: any policy.
    endgame: the scripted phase's architecture name (a game's
      GameSpec.endgame); None accepts any scripted phase.
    route: the route to require; default the override's declared route.

  Returns:
    bool.
  """
  if not isinstance(policy, RoutedPolicy):
    return False
  if endgame is not None and policy.override.spec.arch_name != endgame:
    return False
  want = route or _phase_route(policy.override)
  return want is not None and registry.resolve_callable(want) is policy.route


def with_scripted_phase(
    policy: types.Policy, endgame: Optional[str], route: Optional[str] = None
) -> types.Policy:
  """`policy` with a scripted phase routed around it.

  Args:
    policy: any in-graph policy.
    endgame: the scripted phase's architecture name (a game's
      GameSpec.endgame); None (a game without one) returns `policy`.
    route: the phase's "module:attr" route; default the endgame policy's
      declared route (`phase_route`, or its class's ROUTE).

  Returns:
    RoutedPolicy(policy, endgame policy, route); `policy` itself when it
    already routes that phase.

  Raises:
    ValueError: no route given and the endgame policy declares none.
  """
  if endgame is None or is_scripted_phase(policy, endgame, route):
    return policy
  override = registry.build(endgame, {})
  route = route or _phase_route(override)
  if route is None:
    raise ValueError(
        f"scripted phase {endgame!r} declares no route; pass route"
    )
  return RoutedPolicy(policy, override, route)


def routed_pair(
    policy: types.Policy,
    params: Any,
    endgame: Optional[str],
    route: Optional[str] = None,
) -> tuple[types.Policy, Any]:
  """(policy, params) with the scripted phase `endgame` routed.

  See with_scripted_phase; a pair that already routes it is returned as is.

  Args:
    policy: any in-graph policy.
    params: its params.
    endgame: the scripted phase's architecture name, or None.
    route: the phase's "module:attr" route; default the endgame policy's.

  Returns:
    (policy, params), routed.
  """
  pol = with_scripted_phase(policy, endgame, route)
  if pol is policy:
    return policy, params
  return pol, pol.make_params(
      params, pol.override.init_params(jax.random.key(0))
  )


def unwrap_routed(
    policy: types.Policy, params: Any
) -> tuple[types.Policy, Any]:
  """(base, base params) of a RoutedPolicy; any other pair as is.

  Args:
    policy: any policy.
    params: its params.

  Returns:
    (policy, params).
  """
  if isinstance(policy, RoutedPolicy):
    return policy.base, params["base"]
  return policy, params
