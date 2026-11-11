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

"""Decide: known answers on hand-computed logits, by check_decide and direct."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.testing import check_decide


def test_check_decide_passes():
  """Decide passes check_decide."""
  check_decide.check_decide(decide_lib.decide)


def test_policy_output_input_and_hand_computed_probs():
  """Decide takes a PolicyOutput; probs equal hand-computed ones."""
  out = types.PolicyOutput(
      logits=jnp.array([[0.0, jnp.log(2.0), jnp.log(5.0)]]),
      value=jnp.zeros((1, 2)),
      aux={},
  )
  a, p = decide_lib.decide(
      out, jnp.array([[True, True, False]]), types.ARGMAX, jax.random.key(0)
  )
  # legal logits [0, log 2] -> probs [1/3, 2/3, 0]; argmax index 1.
  np.testing.assert_allclose(np.asarray(p), [[1 / 3, 2 / 3, 0.0]], atol=1e-6)
  assert int(a[0]) == 1 and a.dtype == jnp.int32


def test_f64_probs_from_any_input_dtype():
  """Decide returns f64 probs from any input dtype."""
  for dt in (jnp.bfloat16, jnp.float32, jnp.float64):
    a, p = decide_lib.decide(
        jnp.zeros((2, 3), dt),
        jnp.ones((2, 3), bool),
        types.SAMPLE,
        jax.random.key(1),
    )
    assert p.dtype == jnp.float64 and a.dtype == jnp.int32, (
        dt,
        p.dtype,
        a.dtype,
    )
  assert jax.config.jax_enable_x64


def test_f64_known_answer_finer_than_f32():
  """Known answer that only f64 arithmetic gets.

  logits [0, 1e-9] give p1 - p0 = tanh(5e-10) = 5e-10 (f32 rounds exp(1e-9) to
  1 and returns [0.5, 0.5]); [0, 40] gives p0 = 1 / (1 + e^40) = 4.248354e-18,
  below f32's resolution next to 1.
  """
  _, p = decide_lib.decide(
      jnp.array([[0.0, 1e-9], [0.0, 40.0]], jnp.float64),
      jnp.ones((2, 2), bool),
      types.ARGMAX,
      jax.random.key(0),
  )
  p = np.asarray(p)
  assert p.dtype == np.float64
  np.testing.assert_allclose(p[0, 1] - p[0, 0], np.tanh(5e-10), rtol=1e-6)
  np.testing.assert_allclose(p[1, 0], 1.0 / (1.0 + np.exp(40.0)), rtol=1e-12)
  np.testing.assert_allclose(p.sum(-1), [1.0, 1.0], rtol=0, atol=1e-15)


def test_logp_matches_f64_probs():
  """decide_rows' behaviour logp is log of decide's f64 prob (stored f32)."""
  lg = jnp.array([[0.0, jnp.log(3.0), 5.0]], jnp.float32)
  m = jnp.array([[True, True, False]])
  a, lp, fin = arena_lib.decide_rows(
      types.SAMPLE, lg, m, jax.random.split(jax.random.key(4), 1)
  )
  assert lp.dtype == jnp.float32 and bool(fin[0])
  want = {0: np.log(0.25), 1: np.log(0.75)}[int(a[0])]
  np.testing.assert_allclose(float(lp[0]), want, rtol=1e-6)


def test_shape_and_dtype_errors_fail_loud():
  """Shape and dtype errors fail loud."""
  with pytest.raises(AssertionError):
    decide_lib.decide(
        jnp.zeros((2, 3)), jnp.ones((2, 4), bool), types.ARGMAX, None
    )
  with pytest.raises(AssertionError):
    decide_lib.decide(
        jnp.zeros((2, 3)), jnp.ones((2, 3), jnp.int32), types.ARGMAX, None
    )
  with pytest.raises(AssertionError):
    decide_lib.decide(jnp.zeros((3,)), jnp.ones((3,), bool), types.ARGMAX, None)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
