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

"""check_policy: the Policy contract, for every policy author.

Given SeatSteps of one hand (step1 opens a match: new_hand and new_match
True; step2 continues it: both False; step3 opens a LATER hand of the same
match: new_hand True, new_match False; step3 defaults to step1 with new_match
set to False) and two param sets of the same shapes:
  0. spec is a PolicySpec; init_params(key) gives params of params_a's shapes;
  1. init_carry -> observe keeps the carry's tree structure, shapes and dtypes;
  2. evaluate returns a PolicyOutput: logits f32[B, A], value f32[B, P], aux a
     Mapping; deterministic under the same key;
  3. evaluate never mutates the carry (snapshot before == after);
  4. a new_match step restores the fresh-start carry: observe(c, step1)
     equals observe(init_carry, step1) for any prior carry c;
  5. a new_hand step (new_match False) resets per-hand state:
     observe(c, step3) equals observe(init_carry, step3), unless
     cross_hand_state=True (the policy keeps state across the hands of a
     match by design; check 4 still requires new_match to reset it);
  6. stateful: evaluate after observe(step1), observe(step2) differs from
     evaluate after reset-then-observe(step2) only; a stateless policy
     (spec.stateful False) must give equal outputs. This is a property of
     the steps as well as the policy: a policy whose carry changes its
     output only later in a hand fails it on these two plies, so its author
     passes a step1 and step2 whose histories its output does separate;
  7. observe_seq over [step1, step2] equals two observe calls, and
     evaluate_seq equals observe+evaluate per ply; a host policy must
     override both (the jax-scan defaults raise for it);
  8. JAX policies: jitted observe and evaluate trace once across the two
     param sets.

Host policies (HOST_DRIVEN, or a framework other than JAX) are driven as the
host adapters drive them: numpy SeatSteps and raw uint32[..., 2] keys; their
outputs are compared as numpy.

  step1, step2 = steps.two_ply_steps(env, batch=8, seat=0)
  check_policy.check_policy(policy, params_a, params_b, step1, step2)
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.interop import host
from open_spiel.python.scale.testing.errors import ContractError
from open_spiel.python.scale.testing.errors import require

tree_map = jax.tree_util.tree_map


def _sig(tree):
  """A tree's (treedef, per-leaf shape and dtype).

  Args:
    tree: A pytree.

  Returns:
    The signature.
  """
  leaves, treedef = jax.tree_util.tree_flatten(host.to_numpy(tree))
  return treedef, [(tuple(x.shape), np.dtype(x.dtype)) for x in leaves]


def _equal(a, b, atol=0.0) -> bool:
  """Whether two pytrees are equal within `atol`, NaNs equal.

  Args:
    a: A pytree.
    b: The other pytree.
    atol: The absolute tolerance.

  Returns:
    True when equal.
  """
  la, ta = jax.tree_util.tree_flatten(host.to_numpy(a))
  lb, tb = jax.tree_util.tree_flatten(host.to_numpy(b))
  if ta != tb:
    return False
  return all(
      x.shape == y.shape
      and np.allclose(x, y, atol=atol, rtol=0.0, equal_nan=True)
      for x, y in zip(la, lb)
  )


def check_policy(
    policy: types.Policy,
    params_a: Any,
    params_b: Any,
    step1: types.SeatStep,
    step2: types.SeatStep,
    step3: Optional[types.SeatStep] = None,
    cross_hand_state: bool = False,
    seed: int = 0,
) -> None:
  """Checks the Policy contract (module doc).

  Args:
    policy: The policy.
    params_a: One param set.
    params_b: Another param set of the same structure.
    step1: The SeatStep that opens a match.
    step2: The SeatStep after one action.
    step3: A SeatStep that opens a later hand of the match, or None.
    cross_hand_state: Whether the policy keeps state across hands.
    seed: The seed.

  Raises:
    ContractError: on the first broken property.
  """
  spec = policy.spec
  require(
      isinstance(spec, types.PolicySpec), "policy.spec must be a PolicySpec"
  )
  is_jax = (
      spec.framework == "jax"
      and spec.capability != types.Capability.HOST_DRIVEN
  )
  if step3 is None:
    step3 = step1._replace(
        new_match=jnp.zeros_like(jnp.asarray(step1.new_match))
    )
  require(
      np.asarray(step1.new_hand).all() and np.asarray(step1.new_match).all(),
      "step1 must open a match (new_hand and new_match True)",
  )
  require(
      not np.asarray(step2.new_hand).any()
      and not np.asarray(step2.new_match).any(),
      "step2 must continue the hand (new_hand and new_match False)",
  )
  require(
      np.asarray(step3.new_hand).all()
      and not np.asarray(step3.new_match).any(),
      "step3 must open a later hand of the match (new_hand True, new_match "
      "False)",
  )
  if cross_hand_state:
    require(spec.stateful, "cross_hand_state requires a stateful policy")
  b = np.asarray(step1.seat).shape[0]
  a, p = spec.num_actions, spec.num_players
  key = jax.random.key(seed)
  k_init, k_eval, k_params = jax.random.split(key, 3)
  if not is_jax:
    step1, step2, step3 = host.to_numpy((step1, step2, step3))
    k_init_rows = np.broadcast_to(host.raw_keys(k_init), (b, 2)).copy()
    k_eval = np.broadcast_to(host.raw_keys(k_eval), (b, 2)).copy()
  else:
    k_init_rows = k_init

  # 0. init_params.
  require(
      _sig(policy.init_params(k_params if is_jax else host.raw_keys(k_params)))
      == _sig(params_a),
      "init_params(key) does not match params_a's shapes",
  )
  require(
      _sig(params_a) == _sig(params_b), "the two param sets differ in shape"
  )

  def init(seat):
    """Returns a fresh carry for `seat`.

    Args:
      seat: i32[B], each row's seat.

    Returns:
      The carry.
    """
    return policy.init_carry(params_a, b, seat, k_init_rows)

  # 1. Carry signature.
  c0 = init(step1.seat)
  c1 = policy.observe(params_a, c0, step1)
  c2 = policy.observe(params_a, c1, step2)
  require(
      _sig(c0) == _sig(c1) == _sig(c2),
      f"carry signature changes under observe: {_sig(c0)} vs {_sig(c2)}",
  )

  # 2. Output contract and determinism.
  out = policy.evaluate(params_a, c2, step2, k_eval)
  require(isinstance(out, types.PolicyOutput), f"evaluate returned {type(out)}")
  o = host.to_numpy(out)
  require(
      tuple(o.logits.shape) == (b, a) and o.logits.dtype == np.float32,
      f"logits must be f32[{b},{a}], got {o.logits.dtype}{o.logits.shape}",
  )
  require(
      tuple(o.value.shape) == (b, p) and o.value.dtype == np.float32,
      f"value must be f32[{b},{p}], got {o.value.dtype}{o.value.shape}",
  )
  require(hasattr(out.aux, "items"), "aux must be a Mapping")
  require(
      _equal(out, policy.evaluate(params_a, c2, step2, k_eval)),
      "evaluate is not deterministic under the same key",
  )

  # 3. evaluate never mutates the carry.
  before = host.to_numpy(c2)
  policy.evaluate(params_a, c2, step2, k_eval)
  require(_equal(before, c2), "evaluate mutated the carry")

  # 4. A new_match step resets to the fresh-start carry.
  fresh = policy.observe(params_a, init(step1.seat), step1)
  require(
      _equal(policy.observe(params_a, c2, step1), fresh, atol=1e-6),
      "observe(new_match step) does not reset the carry",
  )

  # 5. A new_hand step resets per-hand state.
  if not cross_hand_state:
    fresh3 = policy.observe(params_a, init(step3.seat), step3)
    require(
        _equal(policy.observe(params_a, c2, step3), fresh3, atol=1e-6),
        "observe(new_hand step, new_match False) does not reset the carry; a "
        "policy "
        "that keeps state across hands must say so (cross_hand_state=True)",
    )

  # 6. Stateful vs stateless.
  only2 = policy.observe(params_a, init(step2.seat), step2)
  out_only2 = policy.evaluate(params_a, only2, step2, k_eval)
  differs = not _equal(out.logits, out_only2.logits, atol=1e-6)
  if spec.stateful:
    require(
        differs,
        "stateful policy: evaluate after (step1, step2) equals "
        "evaluate after step2 alone; the carry carries nothing",
    )
  else:
    require(not differs, "stateless policy depends on history")

  # 7. observe_seq / evaluate_seq == the per-ply loop.
  stack = (
      (lambda x, y: jnp.stack([jnp.asarray(x), jnp.asarray(y)], axis=1))
      if is_jax
      else (lambda x, y: np.stack([np.asarray(x), np.asarray(y)], axis=1))
  )
  seq = tree_map(stack, step1, step2)
  if not is_jax:
    for name in ("observe_seq", "evaluate_seq"):
      require(
          getattr(type(policy), name) is not getattr(types.Policy, name, None),
          f"a host policy must implement {name} (the default is a jax scan)",
      )
  c_seq = policy.observe_seq(params_a, c0, seq)
  require(
      _equal(c_seq, c2, atol=1e-6), "observe_seq differs from repeated observe"
  )
  keys_seq = (
      jnp.broadcast_to(k_eval, (2,))
      if is_jax
      else np.broadcast_to(k_eval[:, None], (b, 2, 2)).copy()
  )
  c_ev, out_seq = policy.evaluate_seq(params_a, c0, seq, keys_seq)
  require(
      _equal(c_ev, c2, atol=1e-6),
      "evaluate_seq's final carry differs from observe",
  )
  out1 = policy.evaluate(params_a, c1, step1, k_eval)
  o_seq = host.to_numpy(out_seq)
  for t, ref in enumerate((out1, out)):
    r = host.to_numpy(ref)
    require(
        np.allclose(o_seq.logits[:, t], r.logits, atol=1e-6, equal_nan=True)
        and np.allclose(o_seq.value[:, t], r.value, atol=1e-6, equal_nan=True),
        f"evaluate_seq differs from observe+evaluate at ply {t}",
    )

  # 8. Trace once across param sets (JAX).
  if not is_jax:
    return
  traces = {"observe": 0, "evaluate": 0}

  def obs_fn(pp, c, s):
    """Counts a trace of observe and observes.

    Args:
      pp: The params.
      c: The carry.
      s: The SeatStep.

    Returns:
      The carry after the step.
    """
    traces["observe"] += 1
    return policy.observe(pp, c, s)

  def eval_fn(pp, c, s, k):
    """Counts a trace of evaluate and evaluates.

    Args:
      pp: The params.
      c: The carry.
      s: The SeatStep.
      k: The PRNG key.

    Returns:
      The PolicyOutput.
    """
    traces["evaluate"] += 1
    return policy.evaluate(pp, c, s, k)

  obs_j, eval_j = jax.jit(obs_fn), jax.jit(eval_fn)
  for pp in (params_a, params_b):
    cc = obs_j(pp, c1, step2)
    jax.block_until_ready(eval_j(pp, cc, step2, k_eval))
  if traces["observe"] != 1 or traces["evaluate"] != 1:
    raise ContractError(f"jit retraced across param sets: {traces}")
