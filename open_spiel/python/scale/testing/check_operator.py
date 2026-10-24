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

"""check_operator: the conformance test of a search operator.

A search operator is a policy whose outputs are training targets (the
expert of expert iteration). Its targets must be well-formed and must
depend on each row's own inputs only, or training on them is silently
wrong. On a batch of SeatSteps [B] with acting and non-acting rows, the
check requires:

  * class facts in the operator vocabularies: decider "search",
    prior_dependence one of PRIOR_DEPENDENCE, value_bias one of VALUE_BIASES
    (policies.operator_vocab, read through policies.operators);
  * every reserved aux key with its declared dtype and shape, finite on the
    acting rows; AUX_TARGET_POLICY a distribution over the legal actions on
    acting rows; no AUX_TARGET_VALUE (the Arena reads that key as a
    teacher's value);
  * determinism: the same call twice is bitwise equal;
  * row independence at a fixed padded shape: replacing the other rows by
    copies of one row leaves a subset's outputs bitwise unchanged. For a
    HOST_DRIVEN operator, whose keys are raw per-row uint32[B, 2], permuting
    rows and keys permutes the outputs bitwise. (A JAX operator takes one key
    per call and derives row keys from the row index, so a permutation is
    not a property of a row.)

Nothing is required across batch sizes. An operator receives SeatSteps
only, so no label can reach it.

  report = check_operator.check_operator(policy, params, steps)
  report["decider"], report["acting"]
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.interop import host
from open_spiel.python.scale.policies import operators
from open_spiel.python.scale.testing.errors import ContractError
from open_spiel.python.scale.testing.errors import require

tree_map = jax.tree_util.tree_map


def _host(policy) -> bool:
  """Whether `policy` is host-driven.

  Args:
    policy: The policy.

  Returns:
    True for a host policy.
  """
  return policy.spec.capability == types.Capability.HOST_DRIVEN


def _call(policy, params, step, key, carry_key):
  """One observe + evaluate from a fresh carry.

  Args:
    policy: The operator.
    params: Its params.
    step: The SeatStep observed and evaluated.
    key: The evaluate key.
    carry_key: A typed key (JAX operator) or raw per-row keys (host
      operator), for the fresh carry.

  Returns:
    The PolicyOutput, as numpy.
  """
  b = int(np.asarray(step.seat).shape[0])
  if _host(policy):
    c = policy.observe(
        params,
        policy.init_carry(
            params, b, np.asarray(step.seat), np.asarray(carry_key)
        ),
        step,
    )
    return host.to_numpy(policy.evaluate(params, c, step, np.asarray(key)))

  def run(p, st, k, ck):
    """Observes then evaluates from a fresh carry.

    Args:
      p: The params.
      st: The SeatStep.
      k: The evaluate key.
      ck: The init_carry key.

    Returns:
      The PolicyOutput.
    """
    c = policy.observe(p, policy.init_carry(p, b, st.seat, ck), st)
    return policy.evaluate(p, c, st, k)

  return host.to_numpy(jax.jit(run)(params, step, key, carry_key))


def _eq(a, b) -> bool:
  """Whether two pytrees are byte-identical, dtypes and shapes included.

  Args:
    a: A pytree.
    b: The other pytree.

  Returns:
    True when equal.
  """
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  return len(la) == len(lb) and all(
      np.asarray(x).dtype == np.asarray(y).dtype
      and np.asarray(x).shape == np.asarray(y).shape
      and np.asarray(x).tobytes() == np.asarray(y).tobytes()
      for x, y in zip(la, lb)
  )


def _rows(tree, idx):
  """Rows `idx` of every leaf.

  Args:
    tree: A pytree.
    idx: The row indices.

  Returns:
    The rows.
  """
  return tree_map(lambda x: np.asarray(x)[idx], tree)


def check_operator(
    policy: types.Policy,
    params: Any,
    step: types.SeatStep,
    *,
    seed: int = 0,
    subset: Optional[np.ndarray] = None,
) -> dict[str, Any]:
  """Runs the conformance test (module doc).

  Args:
    policy: the operator.
    params: its params (one member).
    step: SeatStep [B], numpy or jax, with acting and non-acting rows.
    seed: the key seed.
    subset: the rows whose outputs must survive padding (default every
      other row).

  Returns:
    {decider, prior_dependence, value_bias, rows, acting}.

  Raises:
    ContractError: on the first property that does not hold.
  """
  facts = dict(
      decider=operators.decider_of(policy),
      prior_dependence=operators.prior_dependence_of(policy),
      value_bias=operators.value_bias_of(policy),
  )
  require(
      facts["decider"] in types.DECIDERS and facts["decider"] == "search",
      f"decider {facts['decider']!r}: an operator decides by search",
  )
  step = host.to_numpy(step)
  b = int(step.seat.shape[0])
  a_dim = policy.spec.num_actions
  k_eval, k_carry = jax.random.split(jax.random.key(seed))
  key = host.raw_keys(jax.random.split(k_eval, b)) if _host(policy) else k_eval
  if _host(policy):
    k_carry = host.raw_keys(jax.random.split(k_carry, b))
  rk = (lambda idx: _rows(key, idx)) if _host(policy) else (lambda idx: key)
  rc = (
      (lambda idx: _rows(k_carry, idx))
      if _host(policy)
      else (lambda idx: k_carry)
  )
  out = _call(policy, params, step, key, k_carry)
  aux = dict(out.aux)
  require(
      types.AUX_TARGET_VALUE not in aux,
      "an operator emits AUX_TARGET_VALUE (the Arena pops it as a DAgger "
      "teacher's value)",
  )
  require(types.AUX_TARGET_POLICY in aux, "an operator emits AUX_TARGET_POLICY")
  try:
    arena_lib.check_reserved_aux(
        {k: jnp.asarray(v) for k, v in aux.items()}, b, a_dim
    )
  except (TypeError, ValueError) as e:
    raise ContractError(f"reserved aux: {e}") from e
  acting = (np.asarray(step.to_move) == np.asarray(step.seat)) & np.asarray(
      step.legal_mask
  ).any(-1)
  fin = np.asarray(
      arena_lib.reserved_aux_finite(
          {k: jnp.asarray(v) for k, v in aux.items()}, b
      )
  )
  require(
      fin[acting].all(),
      f"non-finite reserved aux on acting rows {np.flatnonzero(~fin & acting)}",
  )
  tp = np.asarray(aux[types.AUX_TARGET_POLICY])
  legal = np.asarray(step.legal_mask)
  require(
      np.allclose(tp[acting].sum(-1), 1.0, atol=1e-5),
      "AUX_TARGET_POLICY is not a distribution on acting rows",
  )
  require(
      not (tp[acting] * ~legal[acting]).any(),
      "AUX_TARGET_POLICY puts mass on illegal actions",
  )
  again = _call(policy, params, step, key, k_carry)
  require(_eq(out, again), "the same call twice is not bitwise equal")
  # Each padding row copies a subset row of the same acting status, so the
  # number of acting rows, and with it the shape of any chunk an operator
  # gathers its acting rows into, stays fixed.
  sub = np.arange(0, b, 2) if subset is None else np.asarray(subset)
  pad_idx = np.arange(b)
  for i in range(b):
    if i in set(sub.tolist()):
      continue
    same = [j for j in sub if acting[j] == acting[i]]
    if same:
      pad_idx[i] = int(same[0])
  padded = _call(policy, params, _rows(step, pad_idx), rk(pad_idx), rc(pad_idx))
  require(
      _eq(_rows(out, sub), _rows(padded, sub)),
      "a row's output changed when the OTHER rows were replaced by padding "
      "(fixed shape)",
  )
  if _host(policy):
    perm = np.random.default_rng(seed).permutation(b)
    permuted = _call(policy, params, _rows(step, perm), rk(perm), rc(perm))
    require(
        _eq(_rows(out, perm), permuted),
        "permuting rows and their keys did not permute the outputs bitwise "
        "(fixed shape)",
    )
  return dict(facts, rows=b, acting=int(acting.sum()))
