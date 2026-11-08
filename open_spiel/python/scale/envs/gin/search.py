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

"""Search for gin rummy: PIMC over determinized deals, on top of a gin network.

A trained gin network plays one decision at a time from its own InfoState.
This module composes it with SearchPolicy into a policy that looks ahead:
each decision draws K complete deals consistent with what the seat has seen
(envs.gin.belief) and searches each as a perfect-information game, averaging the
root visit distributions and values over the K deals (perfect-information
Monte Carlo, PIMC; Long et al. 2010). It builds nothing new: the search, the
sampler and the scripted endgame are the shared components.

  pol = gin_search.make(net, env, gin_search.mcts_config(num_simulations=64,
  num_worlds=8,
                                                         value_scale=50.0))
  params = gin_search.make_params(pol, net_params)
  pol, params = gin_search.from_artifact(artifact, env)   # value units from the
  artifact

The composition:

  RoutedPolicy(
      base=SearchPolicy(
          base=RoutedPolicy(net, GinScriptedEndgamePolicy, gin_endgame.ROUTE),
          env=GinEnv, sampler=GinBeliefSampler,
          mcts=MCTSConfig(mode="zero_sum", chance="determinized",
          fold_routed=True, ...)),
      override=GinScriptedEndgamePolicy, route=gin_endgame.ROUTE)

The scripted endgame inside the tree. KNOCK, LAYOFF and WALL plies are
played by the scripted endgame in every gin policy, so inside the tree they
are deterministic dynamics, not search nodes (`fold_routed`): a KNOCK leads
straight to the scored end of the hand. Stock draws follow each world's
sampled stock order (`chance="determinized"`), so every tree is a
deterministic two-player game. At the root the outer RoutedPolicy plays the
scripted endgame on KNOCK, LAYOFF and WALL decisions, so the decider a
server reports is "scripted" there and "search" elsewhere.

Limitations of PIMC, which make it a policy-improvement operator to measure
rather than a solver:

  * Strategy fusion. Inside each world the tree knows the hidden cards, so
    below the root it may act differently in worlds the seat cannot tell
    apart. The averaged root value is that of a clairvoyant: an upper bound
    on any real strategy's value, biased toward actions whose payoff depends
    on hidden cards and against information-gathering or hedging moves.
  * Non-locality. Each world is searched on its own with worlds drawn once at
    the root, so neither side reasons about what the other's moves reveal.
  * The belief is uniform-consistent: the opponent's moves count as no
    information about its cards. Against a hand-dependent opponent the
    pooled marginals stay calibrated, but cards of ranks it shed or took are
    mis-weighted.
  * Only the root decision is played. Root values carry Monte Carlo error in
    K as well as in the simulations per world.

Value units. Tree rewards are env points / value_scale, the units the base
value head was trained in (a network trained on points / 50 has
value_scale 50). `from_artifact` reads them from the artifact and refuses
one that does not record them; `make` takes the value_scale it is given.
"""

from __future__ import annotations

from typing import Any, Optional

from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.gin import belief as gin_belief
from open_spiel.python.scale.league import slot_factories
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import search


def mcts_config(
    *,
    num_simulations: int = 64,
    num_worlds: int = 8,
    search_kind: str = "gumbel",
    value_scale: float = 1.0,
    **kw: Any,
) -> search.MCTSConfig:
  """The gin PIMC MCTSConfig (module doc); other MCTSConfig fields via kw.

  search_kind: "gumbel" (no root noise) or "muzero" (PUCT; pass
  dirichlet_fraction=0.0 for evaluation).

  Args:
    num_simulations: Simulations per search.
    num_worlds: Determinized worlds per search.
    search_kind: "gumbel" (no root noise) or "muzero" (PUCT; pass
      dirichlet_fraction=0.0 for evaluation).
    value_scale: The base value head's units.
    **kw: Other MCTSConfig fields.

  Returns:
    The MCTSConfig.

  Raises:
    ValueError: kw sets mode, chance, fold_routed or search, which gin search
      fixes.
  """
  for fixed in ("mode", "chance", "fold_routed", "search"):
    if fixed in kw:
      raise ValueError(f"gin search fixes MCTSConfig.{fixed}")
  return search.MCTSConfig(
      num_simulations=num_simulations,
      num_worlds=num_worlds,
      search=search_kind,
      mode="zero_sum",
      chance="determinized",
      fold_routed=True,
      value_scale=float(value_scale),
      **kw,
  )


def routed_net(net: types.Policy, env: Any = None) -> routed.RoutedPolicy:
  """The network `net` routed to the game's scripted endgame.

  (GameSpec.endgame of gin_rummy); an already routed net is kept.

  Args:
    net: The network policy.
    env: The env (None: gin_rummy).

  Returns:
    The routed policy.

  Raises:
    ValueError: net is routed on something other than a scripted endgame.
  """
  if isinstance(net, routed.RoutedPolicy):
    if not routed.is_scripted_phase(net):
      raise ValueError(
          f"the base's route {net.spec.arch_kwargs['route']!r} is not a "
          "scripted endgame's"
      )
    return net
  return routed.with_scripted_phase(
      net, env_registry.game("gin_rummy" if env is None else env.name).endgame
  )


def make(
    net: types.Policy, env: Any, cfg: search.MCTSConfig
) -> routed.RoutedPolicy:
  """The servable gin search policy (module doc) over `net` in `env` (a GinEnv).

  Args:
    net: The network policy.
    env: The GinEnv.
    cfg: The search config.

  Returns:
    The routed search policy.

  Raises:
    ValueError: cfg is not a gin search config (mcts_config builds one).
  """
  if not (
      cfg.mode == "zero_sum"
      and cfg.chance == "determinized"
      and cfg.fold_routed
  ):
    raise ValueError(
        "gin search needs mode='zero_sum', chance='determinized', "
        f"fold_routed=True; got {cfg}"
    )
  base = routed_net(net, env)
  sp = search.SearchPolicy(base, env, gin_belief.GinBeliefSampler(env), cfg)
  return routed.RoutedPolicy(
      sp, type(base.override)(), base.spec.arch_kwargs["route"]
  )


def make_params(policy: routed.RoutedPolicy, net_params: Any) -> dict[str, Any]:
  """The composed params of `make(net, ...)` from the network's own params.

  (for an already routed net: the params of its base).

  Args:
    policy: The policy `make` built.
    net_params: The network's params.

  Returns:
    The composed params.
  """
  inner = search_policy(policy).base
  return policy.make_params(
      {"base": inner.make_params(net_params, {}), "sampler": {}}, {}
  )


def from_artifact(
    art: Any, env: Any, cfg: Optional[search.MCTSConfig] = None, **mcts_kw: Any
) -> tuple[routed.RoutedPolicy, dict[str, Any]]:
  """(policy, params): `make` over a loaded gin artifact's network.

  A routed gin training artifact keeps its route, and its base params are the
  network's. The search runs in the value units the artifact records.

  Args:
    art: a loaded artifact of a NATIVE_JAX gin network.
    env: the GinEnv to search in.
    cfg: the search config; None builds mcts_config(value_scale=<recorded>,
      **mcts_kw). A given cfg must carry the recorded value_scale.
    **mcts_kw: MCTSConfig keywords, when cfg is None.

  Returns:
    (policy, params).

  Raises:
    ValueError: a non-JAX artifact, an artifact without recorded value units,
      a cfg in other units, or both cfg and mcts_kw.
  """
  m = art.manifest
  if m.framework != "jax":
    raise ValueError(
        f"{art.path}@{m.step}: a {m.framework} artifact; the search needs a "
        "NATIVE_JAX network"
    )
  vs = artifact_lib.value_scale(art.path, m.step)
  if vs is None:
    raise ValueError(
        f"{art.path}@{m.step}: the artifact records no value_scale, so the "
        "search's rewards would be in unknown units"
    )
  if cfg is None:
    cfg = mcts_config(value_scale=vs, **mcts_kw)
  else:
    if mcts_kw:
      raise ValueError(
          f"pass either cfg or MCTSConfig keywords, not both: {sorted(mcts_kw)}"
      )
    if float(cfg.value_scale) != vs:
      raise ValueError(
          f"cfg.value_scale {cfg.value_scale} but {art.path}@{m.step} records "
          f"value_scale {vs} (its value head's units)"
      )
  net, net_params = art.policy, art.params
  if isinstance(net, routed.RoutedPolicy):
    routed_net(net)  # refuses a foreign route
    net_params = art.params["base"]
  pol = make(net, env, cfg)
  return pol, make_params(pol, net_params)


def search_policy(policy: routed.RoutedPolicy) -> search.SearchPolicy:
  """The SearchPolicy inside `make`'s composition.

  Args:
    policy: The policy `make` built.

  Returns:
    The SearchPolicy.

  Raises:
    TypeError: The policy was not built by `make`.
  """
  if not (
      isinstance(policy, routed.RoutedPolicy)
      and isinstance(policy.base, search.SearchPolicy)
  ):
    raise TypeError(
        f"want a policy built by gin_search.make, got {type(policy).__name__}"
    )
  return policy.base


class GinMctxPimc:
  """The "gin_mctx_pimc" slot factory of expert iteration.

  It is registered in league.slot_factories: `make` over the base network.
  factory_kwargs are
  mcts_config's (num_simulations, num_worlds, search_kind, ...); value_scale is
  the AlphaZero learner config's (the expert-iteration spec's value_scale), the
  base value head's units.

  Attributes:
    kwargs: The factory_kwargs, a copy.

  Raises:
    ValueError: factory_kwargs set value_scale, or the learner config has none.
  """

  name = "gin_mctx_pimc"

  def __init__(self, kwargs: Any):
    self.kwargs = dict(kwargs)
    if "value_scale" in self.kwargs:
      raise ValueError(
          "gin_mctx_pimc takes value_scale from the AlphaZero learner config "
          "(the base value head's units), not factory_kwargs"
      )

  def build(self, base, env, spec):
    """The search policy over the base network.

    Args:
      base: The base network policy.
      env: The GinEnv.
      spec: The expert-iteration spec (its value_scale).

    Returns:
      The routed search policy.

    Raises:
      ValueError: The spec has no value_scale.
    """
    if spec.value_scale is None:
      raise ValueError(
          "gin_mctx_pimc needs the AlphaZero learner config's value_scale (the "
          "base value head's units, e.g. 50 for a network trained on points / "
          "50)"
      )
    return make(
        base,
        env,
        mcts_config(value_scale=float(spec.value_scale), **self.kwargs),
    )

  def params(self, operator, base_params):
    """The operator's params from the base's.

    Args:
      operator: The search policy.
      base_params: The base network's params.

    Returns:
      The composed params.
    """
    return make_params(operator, base_params)

  # make(net) nests RoutedPolicy(SearchPolicy(RoutedPolicy(net, endgame)),
  # endgame), and its
  # params and carry nest the same way.
  def unwrap(self, operator, op_params):
    """The base network's params inside the operator's.

    Args:
      operator: Unused.
      op_params: The composed params.

    Returns:
      The base params.
    """
    del operator
    return op_params["base"]["base"]["base"]

  def base_carry(self, init_carry):
    """The base network's carry inside the operator's.

    Args:
      init_carry: The operator's carry.

    Returns:
      The base carry.
    """
    return init_carry["base"]["base"]["base"]

  # The plain member is the base routed to the scripted endgame at KNOCK /
  # LAYOFF / WALL, as every gin entrant plays (a network whose encoder is
  # defined only away from those phases is NaN there).
  def plain_member(self, base):
    """The plain (search-free) member: the base routed to the scripted endgame.

    Args:
      base: The base network policy.

    Returns:
      The routed base (routed_net).

    Raises:
      ValueError: The base is already routed (plain_params would wrap its
        params twice).
    """
    if isinstance(base, routed.RoutedPolicy):
      raise ValueError(
          "gin_mctx_pimc's base is already a RoutedPolicy; want the network"
      )
    return routed_net(base)

  def plain_params(self, plain, base_params):
    """The plain member's params: the routed base's, over the base's.

    Args:
      plain: The plain member (plain_member).
      base_params: The base params.

    Returns:
      The plain member's params.
    """
    return plain.make_params(base_params, {})

  def unplain_params(self, plain, plain_params):
    """The base params from the plain member's params.

    Args:
      plain: Unused.
      plain_params: The plain member's params.

    Returns:
      The base params inside them.
    """
    del plain
    return plain_params["base"]


def _register_slot_factory() -> None:
  if GinMctxPimc.name not in slot_factories.names():
    slot_factories.register(GinMctxPimc.name, GinMctxPimc)


_register_slot_factory()
