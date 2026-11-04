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

"""A host-driven search operator for tests, and its AlphaZero slot factory.

Expert iteration with a host-side operator (the AlphaZero kind league.kinds.AZ,
host=True) needs an operator the HostArena can drive. HostEchoOperator
("testing.host_echo") is the smallest one: HOST_DRIVEN (numpy SeatSteps, raw
uint32 keys) over a NATIVE_JAX base network whose params it receives per
generation. Its "search" is the base's masked softmax:

  AUX_TARGET_POLICY   softmax(base logits over the legal actions)
  AUX_SEARCH_VALUE    the base's value for the seat
  AUX_SEARCH_DECIDED  True
  logits              log(target), so decide plays the target

Class facts: decider "search", prior_dependence "puct_prior+value",
value_bias "unbiased". It has an infoset-invariance fixture
(testing/tests/infoset_invariance_fixtures.py).

  with host_operator_fixtures.registered():
    op = registry.build("testing.host_echo", {"base": base_arch})

The slot factory "testing.host_echo" is registered on import; the
architecture is registered only inside `registered()`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.interop import host
from open_spiel.python.scale.league import slot_factories
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.policies import registry

F32 = jnp.float32


class HostEchoOperator(types.Policy):
  """The echo operator (module doc).

  arch_kwargs: {"base": {arch_name, arch_kwargs}}.

  Attributes:
    base: The policy the operator echoes.
    spec: The PolicySpec.
  """

  ARCH = "testing.host_echo"
  decider = "search"
  prior_dependence = "puct_prior+value"
  emits_target_policy = True

  def __init__(self, base: Mapping[str, Any]):
    self.base = registry.build(base["arch_name"], base["arch_kwargs"])
    bs = self.base.spec
    if bs.capability != types.Capability.NATIVE_JAX or bs.stateful:
      raise ValueError("HostEchoOperator wraps a stateless NATIVE_JAX base")
    self.spec = types.PolicySpec(
        game=bs.game,
        obs_format=bs.obs_format,
        num_actions=bs.num_actions,
        num_players=bs.num_players,
        stateful=False,
        capability=types.Capability.HOST_DRIVEN,
        framework="jax",
        arch_name=self.ARCH,
        arch_kwargs=dict(
            base=dict(arch_name=bs.arch_name, arch_kwargs=dict(bs.arch_kwargs))
        ),
    )
    self._eval = jax.jit(self._eval_impl)

  def value_bias(self) -> str:
    """Returns the bias of its value target.

    Returns:
      "unbiased".
    """
    return "unbiased"

  def init_params(self, key):
    """Returns fresh params for this architecture (one member, unstacked).

    Args:
      key: The PRNG key.

    Returns:
      The params.
    """
    return self.base.init_params(key)

  def encode(self, info):
    """Returns the features [B, ...] of a batched InfoState.

    The policy owns its encoding.

    Args:
      info: The batched InfoState.

    Returns:
      The features.
    """
    return self.base.encode(info)

  def init_carry(self, params, batch, seat, key):
    """Returns the initial carry of `batch` rows.

    Args:
      params: The params.
      batch: The number of rows.
      seat: i32[batch], each row's seat.
      key: The PRNG key.

    Returns:
      The carry.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """Observes one step; called for every seat at every ply.

    Resets on step.new_hand/new_match.

    Args:
      params: The params.
      carry: The carry.
      step: The SeatStep.

    Returns:
      The carry after the step.
    """
    del params, step
    return carry

  def observe_seq(self, params, carry, steps):
    """Returns the carry after observing steps [B, T, ...] in order.

    The default (JAX only) is a scan of `observe`; an override must equal it
    (testing.check_policy).

    Args:
      params: The params.
      carry: The carry.
      steps: The SeatSteps [B, T, ...].

    Returns:
      The carry after the last step.
    """
    del params, steps
    return carry

  def _eval_impl(self, params, step):
    """The base's softmax over the legal actions and the seat's value.

    Args:
      params: The base params.
      step: The SeatStep.

    Returns:
      (logits, value, probabilities, the seat's value).
    """
    out = self.base.evaluate(params, (), step, jax.random.key(0))
    legal = step.legal_mask
    p = jax.nn.softmax(
        jnp.where(legal, out.logits.astype(F32), -jnp.inf), axis=-1
    )
    p = jnp.where(jnp.any(legal, -1, keepdims=True), p, F32(0.0)).astype(F32)
    seat = step.seat.astype(jnp.int32)
    v = jnp.take_along_axis(out.value.astype(F32), seat[:, None], axis=-1)[:, 0]
    logits = jnp.where(
        p > 0, jnp.log(jnp.where(p > 0, p, F32(1.0))), -jnp.inf
    ).astype(F32)
    return logits, out.value.astype(F32), p, v

  def evaluate(self, params, carry, step, key):
    """Evaluates the acting rows (only those, in the Arena).

    Args:
      params: The params.
      carry: The carry; not updated.
      step: The SeatStep.
      key: The PRNG key.

    Returns:
      The PolicyOutput.
    """
    del carry
    keys = np.asarray(key)
    b = int(np.asarray(step.seat).shape[0])
    # Raw per-row keys: the host contract.
    assert keys.shape[0] == b, (keys.shape, b)
    logits, value, p, v = self._eval(
        params, jax.tree_util.tree_map(jnp.asarray, step)
    )
    out = types.PolicyOutput(
        logits=logits,
        value=value,
        aux={
            types.AUX_TARGET_POLICY: p,
            operator_vocab.AUX_SEARCH_VALUE: v,
            operator_vocab.AUX_SEARCH_DECIDED: jnp.ones((b,), jnp.bool_),
        },
    )
    return host.to_numpy(out)

  def evaluate_seq(self, params, carry, steps, keys):
    """Observes then evaluates at every ply of steps [B, T].

    The default (JAX only) is a scan; an override must equal it
    (testing.check_policy).

    Args:
      params: The params.
      carry: The carry.
      steps: The SeatSteps [B, T, ...].
      keys: What evaluate receives at each ply, stacked on a time axis (JAX:
        [T] keys; host: raw uint32[B, T, 2]), or None.

    Returns:
      (final carry, PolicyOutput [B, T, ...]).
    """
    t = int(np.asarray(steps.seat).shape[1])
    outs = [
        self.evaluate(
            params,
            carry,
            jax.tree_util.tree_map(lambda x, i=i: x[:, i], steps),
            np.asarray(keys)[:, i],
        )
        for i in range(t)
    ]
    return carry, types.PolicyOutput(
        *jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=1), *outs)
    )


class HostEchoFactory(slot_factories.PlainBase):
  """The AlphaZero slot factory that builds HostEchoOperator over the base."""

  name = HostEchoOperator.ARCH

  def __init__(self, kwargs: Mapping[str, Any]):
    if kwargs:
      raise ValueError(f"{self.name} takes no factory_kwargs: {sorted(kwargs)}")

  def build(self, base, env, spec):
    """Builds the echo operator over `base`.

    Args:
      base: The base policy.
      env: Unused.
      spec: Unused.

    Returns:
      The HostEchoOperator.
    """
    del env, spec
    bs = base.spec
    return HostEchoOperator(
        dict(arch_name=bs.arch_name, arch_kwargs=dict(bs.arch_kwargs))
    )

  def params(self, operator, base_params):
    """Returns the operator's params: the base's.

    Args:
      operator: Unused.
      base_params: The base params.

    Returns:
      `base_params`.
    """
    del operator
    return base_params

  def unwrap(self, operator, op_params):
    """Returns the base params of the operator's params: the same.

    Args:
      operator: Unused.
      op_params: The operator params.

    Returns:
      `op_params`.
    """
    del operator
    return op_params

  def base_carry(self, init_carry):
    """Returns the base's carry of the operator's carry: the same.

    Args:
      init_carry: The operator carry.

    Returns:
      `init_carry`.
    """
    return init_carry


@contextlib.contextmanager
def registered():
  """Registers "testing.host_echo" as an architecture for the block.

  Artifacts of the operator can then be rebuilt.

  Yields:
    None, inside the block.
  """
  with registry.temporary(HostEchoOperator.ARCH, HostEchoOperator):
    yield


if HostEchoFactory.name not in slot_factories.names():
  slot_factories.register(HostEchoFactory.name, HostEchoFactory)
