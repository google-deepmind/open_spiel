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

"""Search operators: policies whose outputs carry improvement targets.

Expert iteration trains a network toward the output of an operator that
improves on it (a tree search over the network, for example). Here an
operator is an ordinary Policy whose PolicyOutput.aux carries its targets
under the reserved keys (AUX_TARGET_POLICY, AUX_SEARCH_VALUE, AUX_SEARCH_Q);
there is no parallel protocol. This module reads what a training loop needs
to know about an operator from its class and config, and provides the
behaviour wrapper that lets the network act while the operator labels.

  op = SearchPolicy(network, env, sampler, MCTSConfig())
  # {"decider": "search", "prior_dependence": ..., "value_bias": ...}
  describe(op)
  # acts: network, or op on 25% of rows
  slot = with_behavior(network, op, "mixture(0.25)")

Facts of an operator (SearchOperator, core.types):

  decider            who chooses the action ("network", "search", "scripted").
  prior_dependence   how the operator's targets depend on the network it
                     improves (a key of
                     policies.operator_vocab.PRIOR_DEPENDENCE).
  value_bias()       "unbiased", or "fused_upper_bound" for a PIMC tree whose
                     values are optimistic under strategy fusion; value
                     targets from such an operator need care.
  operator_id        sha256 of the canonical JSON of its arch_kwargs.

Wrappers (RoutedPolicy, BehaviorMixture and any class with
`wraps_operator = True`) report the facts of the operator inside them.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.interop import host
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.policies import routed

# training-only: never registered, never written
ARCH_BEHAVIOR_MIXTURE = "behavior_mixture"
# bool[B]: the operator's logits acted
AUX_OPERATOR_ACTED = "behavior/operator_acted"
BEHAVIORS = ("search", "plain", "mixture")
_MIXTURE = re.compile(r"^mixture\(([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)$")
F32 = jnp.float32


def parse_behavior(behavior: str) -> tuple[str, float]:
  """The (kind, beta) of a behaviour string.

  "search" -> ("search", 1.0): the operator acts. "plain" -> ("plain", 0.0):
  the base network acts. "mixture(0.25)" -> ("mixture", 0.25): the operator
  acts on a random 25% of rows.

  Args:
    behavior: "search", "plain" or "mixture(<beta>)".

  Returns:
    ("search" | "plain" | "mixture", beta).

  Raises:
    ValueError: an unknown string, or a mixture beta outside (0, 1).
  """
  if behavior == "search":
    return "search", 1.0
  if behavior == "plain":
    return "plain", 0.0
  m = _MIXTURE.match(str(behavior))
  if m is None:
    raise ValueError(
        f"behavior {behavior!r}: want 'search', 'plain' or 'mixture(<beta>)'"
    )
  beta = float(m.group(1))
  if not 0.0 < beta < 1.0:
    raise ValueError(
        f"behavior {behavior!r}: mixture beta must be in (0, 1) (0 is 'plain', "
        "1 "
        "is 'search')"
    )
  return "mixture", beta


def operator_of(policy: types.Policy) -> types.Policy:
  """The operator inside behaviour or serving wrappers; the policy otherwise.

  Args:
    policy: the policy.

  Returns:
    The innermost wrapped operator.

  Raises:
    ValueError: wrappers nest more than 16 deep.
  """
  for _ in range(16):
    if not getattr(type(policy), "wraps_operator", False):
      return policy
    policy = policy.operator
  raise ValueError("operator wrappers nest more than 16 deep")


def routed_view(policy: types.Policy):
  """The RoutedPolicy whose route marks the rows handed to a scripted override.

  That is a RoutedPolicy itself, or a HOST_DRIVEN operator that routes
  internally (a RoutedPolicy part cannot be HOST_DRIVEN) through its
  routed_view(). Wrapper classes with wraps_operator=True are seen through.
  Learners drop those rows (sources.decisions.exclude_override); HostAgent
  reports their decider (per request: the view is built once per operator
  object and kept, weakly keyed).

  Args:
    policy: the policy.

  Returns:
    The RoutedPolicy, or None.

  Raises:
    TypeError: an operator's routed_view() returns something else.
  """
  if isinstance(policy, routed.RoutedPolicy):
    return policy
  op = operator_of(policy)
  fn = getattr(op, "routed_view", None)
  if not callable(fn):
    return None
  view = _ROUTED_VIEWS.get(op)
  if view is None:
    view = fn()
    if not isinstance(view, routed.RoutedPolicy):
      raise TypeError(
          f"{type(op).__name__}.routed_view() returned "
          f"{type(view).__name__}, not a RoutedPolicy"
      )
    _ROUTED_VIEWS[op] = view
  return view


#: operator -> its routed_view() (routed_view; the operator's config is fixed
#: at construction, so its view is too).
_ROUTED_VIEWS: "weakref.WeakKeyDictionary[types.Policy, Any]" = (
    weakref.WeakKeyDictionary()
)


def _core(policy: types.Policy) -> types.Policy:
  """The innermost operator under any routing.

  Args:
    policy: The policy.

  Returns:
    The policy.
  """
  policy = operator_of(policy)
  while isinstance(policy, routed.RoutedPolicy):
    policy = operator_of(policy.base)
  return policy


def decider_of(policy: types.Policy) -> str:
  """Who chooses the action of `policy` (core.types.decider_of).

  Wrappers and RoutedPolicy report their operator's or base's.

  Args:
    policy: the policy.

  Returns:
    The decider.

  Raises:
    ValueError: a declared decider outside core.types.POLICY_DECIDERS.
  """
  return types.decider_of(_core(policy))


def prior_dependence_of(policy: types.Policy) -> str:
  """The operator class's prior_dependence.

  Args:
    policy: the policy.

  Returns:
    A policies.operator_vocab.PRIOR_DEPENDENCE key.

  Raises:
    ValueError: the policy declares none (it is not a search operator).
  """
  core = _core(policy)
  v = getattr(type(core), "prior_dependence", None)
  if v is None:
    raise ValueError(
        f"{type(core).__name__} ({core.spec.arch_name}) declares no "
        "prior_dependence: not a search operator"
    )
  return operator_vocab.check_prior_dependence(v)


def value_bias_of(policy: types.Policy) -> str:
  """The operator's value bias, computed from its config by value_bias().

  Args:
    policy: the policy.

  Returns:
    One of policies.operator_vocab.VALUE_BIASES.

  Raises:
    ValueError: the policy has no value_bias() (not a search operator) or it
      returns a value outside policies.operator_vocab.VALUE_BIASES.
  """
  core = _core(policy)
  fn = getattr(core, "value_bias", None)
  if not callable(fn):
    raise ValueError(
        f"{type(core).__name__} ({core.spec.arch_name}) has no value_bias(): "
        "not a "
        "search operator"
    )
  v = fn()
  if v not in operator_vocab.VALUE_BIASES:
    raise ValueError(
        f"{type(core).__name__}.value_bias() = {v!r} not in "
        f"{operator_vocab.VALUE_BIASES}"
    )
  return v


def canonical_json(obj: Any) -> bytes:
  """JSON with sorted keys and fixed separators: the bytes operator_id hashes.

  Args:
    obj: a JSON-able object.

  Returns:
    The ASCII bytes.
  """
  return json.dumps(
      obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
  ).encode()


def operator_id(spec_or_kwargs: Any) -> str:
  """The sha256 hex of canonical_json of an operator's arch_kwargs.

  Args:
    spec_or_kwargs: a PolicySpec, a Manifest or an arch_kwargs mapping.

  Returns:
    The hex digest.
  """
  kw = getattr(spec_or_kwargs, "arch_kwargs", spec_or_kwargs)
  return hashlib.sha256(canonical_json(dict(kw))).hexdigest()


class BehaviorMixture(types.Policy):
  """The base network acts while the operator labels; training only.

  Behaviour "plain" plays the base's logits on every row; "mixture(beta)"
  plays the operator's logits on a per-row coin (uniform < beta) and the
  base's elsewhere. The operator runs on every row, so its aux (its targets,
  AUX_TARGET_POLICY included) labels every decision, and
  aux[AUX_OPERATOR_ACTED] (bool[B]) records which rows the operator played.

  Params {"base", "operator"}; carry {"base", "operator"}; both parts observe
  every ply. For an in-graph search the operator's params hold the same base
  arrays, so nothing is copied.

  A BehaviorMixture is never written as an artifact or pooled: its spec says
  publishable=False, it is not in the architecture registry and
  `check_publishable` refuses it. Publish the
  operator or the plain network.

  Attributes:
    base: the base network policy.
    operator: the operator policy.
    behavior: the behaviour string, "plain" or "mixture(beta)".
    kind: the parsed kind, "plain" or "mixture".
    beta: the probability that the operator acts on a row.
    spec: the PolicySpec (publishable=False).

  Raises:
    ValueError: behaviour "search" (use the operator itself), parts that play
      different games, or a HOST_DRIVEN operator over a base that is not
      NATIVE_JAX.
  """

  emits_target_policy = True
  wraps_operator = True

  def __init__(self, base: types.Policy, operator: types.Policy, behavior: str):
    kind, beta = parse_behavior(behavior)
    if kind == "search":
      raise ValueError(
          "BehaviorMixture is the plain or mixture behaviour; 'search' plays "
          "the "
          "operator itself"
      )
    bs, os_ = base.spec, operator.spec
    for f in ("game", "num_actions", "num_players"):
      if getattr(bs, f) != getattr(os_, f):
        raise ValueError(
            f"BehaviorMixture: base {f} {getattr(bs, f)!r} != operator's "
            f"{getattr(os_, f)!r}"
        )
    caps = {bs.capability, os_.capability}
    if types.Capability.HOST_DRIVEN in caps:
      if bs.capability != types.Capability.NATIVE_JAX:
        raise ValueError(
            "BehaviorMixture over a HOST_DRIVEN operator needs a NATIVE_JAX "
            "base"
        )
      cap = types.Capability.HOST_DRIVEN
    elif types.Capability.CALLBACK_BATCHED in caps:
      cap = types.Capability.CALLBACK_BATCHED
    else:
      cap = types.Capability.NATIVE_JAX
    self.base, self.operator = base, operator
    self.behavior, self.kind, self.beta = str(behavior), kind, float(beta)
    framework = os_.framework if bs.framework == "jax" else bs.framework
    self.spec = types.PolicySpec(
        game=bs.game,
        obs_format=bs.obs_format,
        num_actions=bs.num_actions,
        num_players=bs.num_players,
        stateful=bool(bs.stateful or os_.stateful),
        capability=cap,
        framework=framework,
        arch_name=ARCH_BEHAVIOR_MIXTURE,
        arch_kwargs=dict(
            base=dict(arch_name=bs.arch_name, arch_kwargs=dict(bs.arch_kwargs)),
            operator=dict(
                arch_name=os_.arch_name, arch_kwargs=dict(os_.arch_kwargs)
            ),
            behavior=self.behavior,
        ),
        publishable=False,
    )

  @staticmethod
  def make_params(base_params, operator_params) -> dict[str, Any]:
    """The params of the two parts.

    Args:
      base_params: the base's params.
      operator_params: the operator's params.

    Returns:
      {"base": base_params, "operator": operator_params}.
    """
    return {"base": base_params, "operator": operator_params}

  def init_params(self, key):
    """Both parts' initial params.

    Args:
      key: a PRNG key, split between the parts.

    Returns:
      {"base", "operator"} params.
    """
    kb, ko = jax.random.split(key)
    return self.make_params(
        self.base.init_params(kb), self.operator.init_params(ko)
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
      key: a PRNG key (raw uint32[B, 2] row keys for a HOST_DRIVEN operator).

    Returns:
      {"base", "operator"} carries.
    """
    if self.spec.capability == types.Capability.HOST_DRIVEN:
      # A HOST_DRIVEN carry receives raw uint32[B, 2] row keys (host contract).
      typed = jax.random.wrap_key_data(jnp.asarray(np.asarray(key))[0])
      return {
          "base": self.base.init_carry(
              params["base"], batch, jnp.asarray(seat), typed
          ),
          "operator": self.operator.init_carry(
              params["operator"], batch, seat, key
          ),
      }
    kb, ko = jax.random.split(key)
    return {
        "base": self.base.init_carry(params["base"], batch, seat, kb),
        "operator": self.operator.init_carry(
            params["operator"], batch, seat, ko
        ),
    }

  def observe_seq(self, params, carry, steps):
    """Observes a sequence of steps; per step for a HOST_DRIVEN operator.

    Args:
      params: the params.
      carry: the carry.
      steps: SeatSteps, leaves [B, T, ...].

    Returns:
      The carry after observing every step.
    """
    if self.spec.capability != types.Capability.HOST_DRIVEN:
      return super().observe_seq(params, carry, steps)
    t = int(np.asarray(steps.seat).shape[1])
    for i in range(t):
      carry = self.observe(
          params, carry, jax.tree_util.tree_map(lambda x, i=i: x[:, i], steps)
      )
    return carry

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
        "operator": self.operator.observe(
            params["operator"], carry["operator"], step
        ),
    }

  def evaluate(self, params, carry, step, key):
    """The operator's logits on the coin's rows, the base's elsewhere.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key (the operator's, the base's and the coin's).

    Returns:
      The PolicyOutput with the operator's aux and AUX_OPERATOR_ACTED.

    Raises:
      ValueError: `key` is None.
    """
    if key is None:
      raise ValueError(
          "BehaviorMixture.evaluate needs a key (the operator's and the coin's)"
      )
    if self.spec.capability == types.Capability.HOST_DRIVEN:
      return self._evaluate_host(params, carry, step, key)
    kc, kb, ko = jax.random.split(key, 3)
    op = self.operator.evaluate(params["operator"], carry["operator"], step, ko)
    bo = self.base.evaluate(params["base"], carry["base"], step, kb)
    b = jnp.shape(bo.logits)[0]
    if self.kind == "plain":
      coin = jnp.zeros((b,), jnp.bool_)
    else:
      coin = jax.random.uniform(kc, (b,), F32) < F32(self.beta)
    return self._combine(op, bo, coin)

  def _evaluate_host(self, params, carry, step, key):
    """Evaluate for a HOST_DRIVEN operator.

    It gets raw uint32[B, 2] row keys: the coin and the operator's keys derive
    from each row's key (fold_in 1 and 2), and the NATIVE_JAX base runs on the
    host arrays.

    Args:
      params: {"base", "operator"} params.
      carry: {"base", "operator"} carries.
      step: the SeatStep, host arrays.
      key: raw uint32[B, 2] row keys.

    Returns:
      The combined PolicyOutput, as numpy arrays.
    """
    raw = jnp.asarray(np.asarray(key))
    typed = jax.vmap(jax.random.wrap_key_data)(raw)
    k_op = host.raw_keys(jax.vmap(lambda k: jax.random.fold_in(k, 2))(typed))
    op = host.to_numpy(
        self.operator.evaluate(
            params["operator"], carry["operator"], step, k_op
        )
    )
    jstep = jax.tree_util.tree_map(jnp.asarray, step)
    bo = self.base.evaluate(params["base"], carry["base"], jstep, typed[0])
    b = jnp.shape(bo.logits)[0]
    if self.kind == "plain":
      coin = jnp.zeros((b,), jnp.bool_)
    else:
      coin = jax.vmap(
          lambda k: jax.random.uniform(jax.random.fold_in(k, 1), (), F32)
      )(typed) < F32(self.beta)
    out = self._combine(
        types.PolicyOutput(*jax.tree_util.tree_map(jnp.asarray, tuple(op))),
        bo,
        coin,
    )
    return host.to_numpy(out)

  def _combine(self, op, bo, coin):
    """The operator's output where `coin`, else the behaviour's.

    Args:
      op: The operator's PolicyOutput.
      bo: The behaviour's PolicyOutput.
      coin: bool[B], the operator's rows.

    Returns:
      The combined PolicyOutput.

    Raises:
      ValueError: The operator emits a reserved aux key.
    """
    logits = jnp.where(
        coin[:, None], jnp.asarray(op.logits, F32), jnp.asarray(bo.logits, F32)
    )
    value = jnp.where(
        coin[:, None], jnp.asarray(op.value, F32), jnp.asarray(bo.value, F32)
    )
    aux = dict(op.aux)
    if AUX_OPERATOR_ACTED in aux:
      raise ValueError(
          f"the operator emits {AUX_OPERATOR_ACTED!r}, which is reserved for "
          "BehaviorMixture"
      )
    if types.AUX_TARGET_VALUE in aux:
      raise ValueError(
          "the operator emits AUX_TARGET_VALUE, a teacher's key that a search "
          "operator never emits (policies.operator_vocab)"
      )
    aux[AUX_OPERATOR_ACTED] = coin
    return types.PolicyOutput(logits=logits, value=value, aux=aux)


def check_publishable(policy: types.Policy) -> None:
  """Refuses a training-only wrapper (BehaviorMixture).

  What is written as an artifact or pooled is the operator or the plain
  network.

  Args:
    policy: the policy.

  Raises:
    ValueError: the policy is not publishable.
  """
  if (
      not policy.spec.publishable
      or policy.spec.arch_name == ARCH_BEHAVIOR_MIXTURE
  ):
    raise ValueError(
        f"{type(policy).__name__} ({policy.spec.arch_name}) is a training-only "
        "behaviour wrapper and is never published or pooled; write the "
        "operator "
        "or the plain network"
    )


def with_behavior(
    base: types.Policy, operator: types.Policy, behavior: str
) -> types.Policy:
  """The acting policy of a behaviour.

  Args:
    base: the base network policy.
    operator: the operator.
    behavior: "search", "plain" or "mixture(<beta>)".

  Returns:
    The operator itself for "search", BehaviorMixture(base, operator,
    behavior) otherwise.
  """
  kind, _ = parse_behavior(behavior)
  return (
      operator
      if kind == "search"
      else BehaviorMixture(base, operator, behavior)
  )


def describe(policy: types.Policy) -> dict[str, Any]:
  """The operator facts of a policy, for logs and records.

  prior_dependence and value_bias are None when it does not declare them (not
  an operator).

  Args:
    policy: the policy.

  Returns:
    {"decider", "operator_id", "prior_dependence", "value_bias"}.

  Raises:
    ValueError: a declared prior_dependence or value_bias is invalid.
  """
  out: dict[str, Optional[str]] = dict(
      decider=decider_of(policy),
      operator_id=operator_id(operator_of(policy).spec),
  )
  core = _core(policy)
  # Only an absent declaration means "not an operator"; an operator that
  # declares an invalid value raises from the *_of check.
  declares = (
      getattr(type(core), "prior_dependence", None) is not None,
      callable(getattr(core, "value_bias", None)),
  )
  for (k, fn), present in zip(
      (
          ("prior_dependence", prior_dependence_of),
          ("value_bias", value_bias_of),
      ),
      declares,
  ):
    out[k] = fn(policy) if present else None
  return out
