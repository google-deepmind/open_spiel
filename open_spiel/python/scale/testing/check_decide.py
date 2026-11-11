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

"""check_decide: known-answer contract for a decide implementation.

decide_fn(logits, legal_mask, rule, key) -> (action i32[B], probs f64[B, A]),
with logits f32[B, A] or f64[B, A] (the checker passes raw logits only;
core.decide also accepts a PolicyOutput). probs is the temperature softmax
under every rule, argmax included (core.decide).

decide computes in f64 (the package runs with x64 on), whatever the logits
dtype; one known answer below is only reachable in f64.
"""

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.testing.errors import require


def check_decide(decide_fn, num_samples: int = 20000, seed: int = 0) -> None:
  """Raises ContractError on the first broken contract.

  Args:
    decide_fn: decide_fn(logits, legal_mask, rule, key) -> (action, probs).
    num_samples: The samples drawn to check the sampling rule.
    seed: The seed.
  """
  key = jax.random.key(seed)
  argmax = types.DecisionRule("argmax", 1.0)
  sample = types.DecisionRule("sample", 1.0)

  logits = jnp.array(
      [
          [1.0, 3.0, 3.0, 0.0],  # tie: first max index (1)
          [5.0, 1.0, 2.0, 0.0],  # best is illegal (0) -> 2
          [0.0, 0.0, 0.0, 0.0],  # nothing legal
          [-jnp.inf, 0.0, 0.0, 0.0],
      ],
      jnp.float32,
  )  # legal -inf only
  mask = jnp.array([
      [True, True, True, True],
      [False, True, True, True],
      [False, False, False, False],
      [True, False, False, False],
  ])
  act, probs = jax.jit(decide_fn, static_argnums=2)(logits, mask, argmax, key)
  act, probs = np.asarray(act), np.asarray(probs)
  require(
      act.dtype == np.int32 and probs.dtype == np.float64,
      f"dtypes must be i32/f64, got {act.dtype}/{probs.dtype}",
  )
  require(act.tolist() == [1, 2, -1, 0], f"argmax known answer: {act.tolist()}")
  e = np.exp([1.0, 3.0, 3.0, 0.0])
  e /= e.sum()
  require(
      np.allclose(probs[0], e, rtol=1e-12, atol=0), f"softmax row 0: {probs[0]}"
  )
  e1 = np.exp([1.0, 2.0, 0.0])
  e1 /= e1.sum()
  require(
      np.allclose(probs[1], [0.0, *e1], rtol=1e-12, atol=0),
      f"masked softmax row 1: {probs[1]}",
  )
  require((probs[2] == 0).all(), "a row with nothing legal must be all zeros")
  require(
      np.allclose(probs[3], [1, 0, 0, 0]),
      f"single legal action row: {probs[3]}",
  )

  # Temperature: T=2 halves the logits.
  _, p_t = decide_fn(
      logits[:2], mask[:2], types.DecisionRule("argmax", 2.0), key
  )
  e2 = np.exp(np.array([1.0, 3.0, 3.0, 0.0]) / 2)
  e2 /= e2.sum()
  require(
      np.allclose(np.asarray(p_t)[0], e2, rtol=1e-12, atol=0),
      "temperature not applied",
  )

  # f64 arithmetic: logits [0, 1e-9] give p1 - p0 = tanh(5e-10) (f32 gives 0),
  # [0, 40] gives p0 = 1 / (1 + e^40) ~ 4.25e-18 (f32 loses it next to 1).
  _, p64 = decide_fn(
      jnp.array([[0.0, 1e-9], [0.0, 40.0]], jnp.float64),
      jnp.ones((2, 2), bool),
      argmax,
      key,
  )
  p64 = np.asarray(p64)
  require(p64.dtype == np.float64, f"probs must be f64, got {p64.dtype}")
  require(
      abs((p64[0, 1] - p64[0, 0]) / np.tanh(5e-10) - 1.0) < 1e-6,
      f"f64 known answer: p1 - p0 = {p64[0, 1] - p64[0, 0]}, want "
      f"{np.tanh(5e-10)}",
  )
  require(
      abs(p64[1, 0] * (1.0 + np.exp(40.0)) - 1.0) < 1e-9,
      f"f64 known answer: p0 = {p64[1, 0]}, want {1.0 / (1.0 + np.exp(40.0))}",
  )

  # Sampling: frequencies match probs, never illegal, determinism.
  lg = jnp.broadcast_to(
      jnp.array([0.0, jnp.log(3.0), 9.0], jnp.float32), (num_samples, 3)
  )
  mk = jnp.broadcast_to(jnp.array([True, True, False]), (num_samples, 3))
  a1, p1 = decide_fn(lg, mk, sample, key)
  a2, _ = decide_fn(lg, mk, sample, key)
  a1 = np.asarray(a1)
  require(
      np.array_equal(a1, np.asarray(a2)),
      "sampling not deterministic under the same key",
  )
  require((a1 != 2).all(), "sampled an illegal action")
  freq1 = (a1 == 1).mean()
  require(
      abs(freq1 - 0.75) < 4 * np.sqrt(0.75 * 0.25 / num_samples) + 1e-3,
      f"sample frequency {freq1} vs 0.75",
  )
  require(
      np.allclose(np.asarray(p1)[0], [0.25, 0.75, 0.0], atol=1e-6),
      "sample probs",
  )
  a_none, _ = decide_fn(logits[2:3], mask[2:3], sample, key)
  require(
      int(np.asarray(a_none)[0]) == -1, "sample with nothing legal must be -1"
  )

  # Overflow: a legal +inf logit, or logits / T overflowing f64, keeps probs
  # finite (scaled logits are clipped to the f64 range).
  for lg_row, rule in (
      ([jnp.inf, 0.0], argmax),
      ([jnp.inf, jnp.inf], argmax),
      ([1.0, 1.0], types.DecisionRule("sample", 1e-30)),
      ([1e10, 1e10], types.DecisionRule("sample", 1e-300)),
  ):
    a_o, p_o = decide_fn(
        jnp.array([lg_row], jnp.float32), jnp.ones((1, 2), bool), rule, key
    )
    p_o = np.asarray(p_o)
    require(
        np.isfinite(p_o).all() and abs(p_o.sum() - 1.0) < 1e-6,
        f"probs not finite/normalised on overflow {lg_row} "
        f"T={rule.temperature}: {p_o}",
    )
    require(
        int(np.asarray(a_o)[0]) in (0, 1), "overflow row chose no legal action"
    )

  # NaN: a NaN on a legal action is not cleaned; probs are all NaN (so the
  # Arena's bad-ply check raises). A NaN on an illegal action is masked.
  _, p_nan = decide_fn(
      jnp.array([[jnp.nan, 0.0], [0.0, jnp.nan]], jnp.float32),
      jnp.array([[True, True], [True, False]]),
      argmax,
      key,
  )
  p_nan = np.asarray(p_nan)
  require(
      np.isnan(p_nan[0]).all(),
      f"legal NaN logit must give NaN probs: {p_nan[0]}",
  )
  require(
      np.allclose(p_nan[1], [1.0, 0.0]),
      f"illegal NaN logit must be masked: {p_nan[1]}",
  )
