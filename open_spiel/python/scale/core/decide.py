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

"""The decision rule: logits and a legal mask in, an action out.

Training, evaluation and serving all decide through this one function, so a
served decision is the decision the policy made in training. It computes in
f64 and is jittable:
  logits = f64(logits)                     (any input dtype: bf16, f32, f64)
  scaled = clip(logits / T, f64 min, f64 max)
  masked = where(legal_mask, scaled, -inf)
  probs  = softmax(masked)  -> f64[B, A]   (a zero row if nothing is legal)
  action = argmax(masked) (first maximal index)  | categorical(key, masked)
           -> i32[B], and -1 if nothing is legal.

Returned dtypes: action int32, probs float64. Callers that store probs or
log-probs in a batch cast them (TrajectoryBatch.behavior_logp is f32).

probs is the temperature softmax under every rule. Under "sample" it is the
distribution the action was drawn from; under "argmax" it is not the played
distribution (that is one-hot at the action), so a caller that needs the
probabilities of the played action under argmax builds the one-hot itself.
It asserts x64 at trace time: with x64 off, f64 would silently be f32.

The clip keeps a legal action above every illegal one (a legal -inf becomes
the f64 minimum) and keeps probs finite when a legal logit is +inf or
logits / T overflows. NaN logits are not cleaned: a NaN on a legal action
gives an all-NaN probs row, which the Arena counts as a bad ply and raises
on. testing.check_decide pins both.

  action, probs = decide.decide(logits, legal_mask, types.SAMPLE, key)
"""

from typing import Any

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64

# Every equation decide traces runs under this name scope (scope()), so a
# jaxpr audit (testing.check_f64) tells decide's f64 (by contract) from any
# other f64 without guessing from source paths. Metadata only: no numerics
# change.
DEFAULT_SCOPE = "policy_decide"
_SCOPE = {"name": DEFAULT_SCOPE}


def set_scope(name: str) -> None:
  """Sets the jax.named_scope decide traces under (default DEFAULT_SCOPE).

  For a deployment whose recorded traces and profiles use another name. Set it
  once, at import, before anything is traced.

  Args:
    name: an identifier.

  Raises:
    ValueError: name is not an identifier.
  """
  if not isinstance(name, str) or not name.isidentifier():
    raise ValueError(f"a decide scope is an identifier, got {name!r}")
  _SCOPE["name"] = name


def scope() -> str:
  """The named scope of decide's equations (set_scope).

  Reading it has no side effect: a plugin that sets it is loaded by the entry
  point (plugins.load_env), before anything is traced.

  Returns:
    the scope name.
  """
  return _SCOPE["name"]


def _logits_of(out: Any) -> jax.Array:
  """The logits of a PolicyOutput, or `out` itself.

  Args:
    out: A PolicyOutput or logits.

  Returns:
    The logits.
  """
  if isinstance(out, types.PolicyOutput):
    return out.logits
  return out


def decide(
    out: Any,
    legal_mask: jax.Array,
    rule: types.DecisionRule,
    key: Any,
) -> tuple[jax.Array, jax.Array]:
  """Decides an action per row.

  probs is the temperature softmax whatever the rule (module doc: under "argmax"
  it is not the played distribution).

  Args:
    out: a PolicyOutput or raw logits [B, A].
    legal_mask: bool[B, A].
    rule: a static DecisionRule (hashable; part of the trace).
    key: a PRNG key; consumed only by rule.kind == "sample".

  Returns:
    (action i32[B], probs f64[B, A]).
  """
  assert isinstance(rule, types.DecisionRule), rule
  x64.assert_x64("decide")
  with jax.named_scope(scope()):
    return _decide(out, legal_mask, rule, key)


def _decide(out, legal_mask, rule, key):
  """The body of decide, without its rule check and named scope.

  Args:
    out: a PolicyOutput or raw logits [B, A].
    legal_mask: bool[B, A].
    rule: a static DecisionRule.
    key: a PRNG key; consumed only by rule.kind == "sample".

  Returns:
    (action i32[B], -1 on rows with nothing legal; probs f64[B, A]).
  """
  logits = jnp.asarray(_logits_of(out)).astype(jnp.float64)
  legal_mask = jnp.asarray(legal_mask)
  assert logits.ndim == 2, logits.shape
  assert legal_mask.shape == logits.shape, (legal_mask.shape, logits.shape)
  assert legal_mask.dtype == jnp.bool_, legal_mask.dtype

  any_legal = jnp.any(legal_mask, axis=-1)
  scaled = logits / jnp.float64(rule.temperature)
  finfo = jnp.finfo(jnp.float64)
  scaled = jnp.clip(scaled, finfo.min, finfo.max)
  masked = jnp.where(legal_mask, scaled, -jnp.inf)
  # Rows with nothing legal: softmax over a finite dummy row, zeroed below.
  safe = jnp.where(any_legal[:, None], masked, jnp.float64(0.0))
  probs = jax.nn.softmax(safe, axis=-1)
  probs = jnp.where(any_legal[:, None], probs, jnp.float64(0.0)).astype(
      jnp.float64
  )

  if rule.kind == "argmax":
    action = jnp.argmax(masked, axis=-1).astype(jnp.int32)  # first max index
  else:
    action = jax.random.categorical(key, safe, axis=-1).astype(jnp.int32)
  action = jnp.where(any_legal, action, jnp.int32(-1))
  return action, probs
