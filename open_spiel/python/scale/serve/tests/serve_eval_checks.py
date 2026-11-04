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

"""Serve-equals-eval checks: the per-policy tolerance and the case checker."""


import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.serve import replay

# The NUMERIC kinds' logit bound, |dlogit| <= ATOL + RTOL |logit|: a last-bit
# f32 difference is relative (1 ULP of a logit near 20 is 1.9e-6).
ATOL_NUMERIC = 1e-6
RTOL_NUMERIC = 1e-6


def probs_atol(logits, legal, temperature) -> float:
  """The decide-distribution bound the NUMERIC logit bound implies.

  softmax(z / T) moves each probability by at most twice the largest change of
  z / T (to first order: dp_i = p_i (dz_i - sum_j p_j dz_j) / T), over the
  finite legal logits.

  Args:
    logits: the Arena's logits [A].
    legal: the legal mask [A].
    temperature: the rule's temperature.

  Returns:
    the absolute bound on |dprob|.
  """
  z = np.abs(np.asarray(logits, np.float64)[np.asarray(legal, np.bool_)])
  z = z[np.isfinite(z)]  # a search's unvisited action is -inf: probability 0
  dz = ATOL_NUMERIC + RTOL_NUMERIC * (float(z.max()) if z.size else 0.0)
  return 2.0 * dz / float(temperature)


def leaves_equal(a, b, what):
  """Asserts two pytrees have equal leaves, dtypes included."""
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  assert len(la) == len(lb), what
  for x, y in zip(la, lb):
    x, y = np.asarray(x), np.asarray(y)
    assert x.dtype == y.dtype, (what, x.dtype, y.dtype)
    np.testing.assert_array_equal(x, y, err_msg=what)


BITWISE, NUMERIC = "bitwise", "numeric"


def serve_eval_rule(policy) -> str:
  """The serve == eval tolerance of a policy kind.

  BITWISE when no XLA program computes the logits with float arithmetic: a
  table read (TabularPolicy), constant logits (UniformPolicy), or a policy the
  host computes (CALLBACK_BATCHED, HOST_DRIVEN). NUMERIC for every other
  NATIVE_JAX policy (networks, feed-forward or recurrent, search over a
  network, the history fixture): the Arena's rollout and the stateless replay
  are different XLA programs, and XLA:CPU may fuse and round them differently
  in the last bits (jax 0.12's YNNPACK dot fusions do, for a feed-forward MLP
  too). A RoutedPolicy takes its base's kind.

  Args:
    policy: the policy.

  Returns:
    BITWISE or NUMERIC.
  """
  if isinstance(policy, routed.RoutedPolicy):
    return serve_eval_rule(policy.base)
  if isinstance(policy, (tabular.TabularPolicy, uniform.UniformPolicy)):
    return BITWISE
  if policy.spec.capability is not types.Capability.NATIVE_JAX:
    return BITWISE
  return NUMERIC


def check_cases(env, agents, cases, rule, exact=True):
  """Checks every case against the Arena.

  replay == Arena steps; the decision == Arena logits / probs / action (bitwise;
  to ATOL_NUMERIC when exact is False, the action always exact).

  Args:
    env: the env.
    agents: {seat: HostAgent}.
    cases: the arena_ref Cases.
    rule: the DecisionRule.
    exact: compare bitwise.

  Returns:
    (forced, bitwise) counts.
  """
  forced = bitwise = 0
  for i, c in enumerate(cases):
    rep = replay.replay_seat_steps(env, c.hands, c.seat)
    leaves_equal(
        jax.tree_util.tree_map(lambda x: x[0], rep.steps),
        c.steps,
        f"case {i} steps",
    )
    reply = agents[c.seat].act(
        c.hands, seat_key=c.seat_key, first_ply=c.first_ply
    )
    _, want = decide_lib.decide(
        jnp.asarray(c.logits)[None],
        jnp.asarray(c.legal)[None],
        rule,
        jax.random.key(0),
    )
    want = np.asarray(want[0])
    assert reply.probs.dtype == np.float64 and want.dtype == np.float64
    if exact:
      np.testing.assert_array_equal(
          reply.probs, want, err_msg=f"case {i} probs"
      )
    else:
      np.testing.assert_allclose(
          reply.probs,
          want,
          rtol=0,
          atol=probs_atol(c.logits, c.legal, rule.temperature),
          err_msg=f"case {i} probs",
      )
    assert reply.action == c.action, (i, reply.action, c.action)
    np.testing.assert_array_equal(reply.legal, c.legal)
    if reply.decider == "forced":
      forced += 1
      assert reply.logits is None
    else:
      assert reply.logits.dtype == np.float32
      if exact:
        np.testing.assert_array_equal(
            reply.logits, c.logits, err_msg=f"case {i} logits"
        )
      else:
        np.testing.assert_allclose(
            reply.logits,
            c.logits,
            rtol=RTOL_NUMERIC,
            atol=ATOL_NUMERIC,
            err_msg=f"case {i} logits",
        )
    bitwise += int(np.array_equal(reply.probs, want))
  return forced, bitwise
