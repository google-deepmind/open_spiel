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

"""core.advantages: GAE and TD(lambda) on hand-built batches.

TrajectoryBatch conventions (core.types): each row is one seat's
perspective; valid[t] = the seat acted at ply t; done[t] = ply t's action ended
the hand (often an opponent ply); reward[t] = the seat's reward for the
transition out of ply t; one observe-only terminal ply follows every done ply;
bootstrap_value = the seat's value at bootstrap_step (ply T).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import advantages as adv_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.parallel import compiles

F = np.float32


def _row(**kw):
  """[1, T] arrays from python lists.

  Args:
    **kw: {name: python list of length T}.

  Returns:
    {name: [1, T] array}.
  """
  out = {}
  for k, v in kw.items():
    dt = {"valid": bool, "done": bool}.get(k, F)
    out[k] = jnp.asarray(np.asarray(v, dt)[None])
  return out


# --------------------------------------------------------------------------
# Known answers, hand computed (the numbers are in the comments).
# --------------------------------------------------------------------------


def test_known_answer_episode_boundary_and_bootstrap():
  # t0 seat V=.5 | t1 opp | t2 seat V=.25 | t3 opp ends hand, seat gets +1 |
  # t4 terminal ply | t5 seat (new hand) V=.1 | bootstrap V=.2
  # gamma=.9 lam=.5:
  #   t5: d = .9*.2 - .1 = .08                      A=.08
  #   t2: d = 1 - .25 = .75 (hand over, no bootstrap) A=.75
  #   t0: d = .9*.25 - .5 = -.275; A = -.275 + .45*.75 = .0625
  """Known answer episode boundary and bootstrap."""
  r = _row(
      reward=[0, 0, 0, 1, 0, 0],
      value=[0.5, 0, 0.25, 0, 0, 0.1],
      valid=[1, 0, 1, 0, 0, 1],
      done=[0, 0, 0, 1, 0, 0],
  )
  out = adv_lib.gae(
      r["reward"],
      r["value"],
      r["valid"],
      r["done"],
      jnp.asarray([0.2], F),
      gamma=0.9,
      lam=0.5,
  )
  np.testing.assert_allclose(
      out.advantages[0], [0.0625, 0, 0.75, 0, 0, 0.08], rtol=1e-6, atol=1e-7
  )
  np.testing.assert_allclose(
      out.returns[0], [0.5625, 0, 1.0, 0, 0, 0.18], rtol=1e-6, atol=1e-7
  )


def test_known_answer_consecutive_own_plies_and_midhand_reward():
  # Seat acts twice in a row (gin draw+discard), a non-terminal reward .5 lands
  # on an opponent ply, gamma applies once per own decision.
  # gamma=.5 lam=1, V=[1,2,_,_,3], bootstrap 4:
  #   t4: d = .5*4 - 3 = -1                         A=-1
  #   t1: d = .5 + .5*3 - 2 = 0;  A = 0 + .5*(-1) = -.5
  #   t0: d = .5*2 - 1 = 0;       A = .5*(-.5) = -.25
  """Known answer consecutive own plies and midhand reward."""
  r = _row(
      reward=[0, 0, 0.5, 0, 0],
      value=[1, 2, 0, 0, 3],
      valid=[1, 1, 0, 0, 1],
      done=[0, 0, 0, 0, 0],
  )
  out = adv_lib.gae(
      r["reward"],
      r["value"],
      r["valid"],
      r["done"],
      jnp.asarray([4.0], F),
      gamma=0.5,
      lam=1.0,
  )
  np.testing.assert_allclose(
      out.advantages[0], [-0.25, -0.5, 0, 0, -1], rtol=1e-6
  )
  np.testing.assert_allclose(out.returns[0], [0.75, 1.5, 0, 0, 2], rtol=1e-6)


def test_known_answer_segment_starts_midhand_and_own_terminal_action():
  # t0: an opponent ply ends the PREVIOUS segment's hand (-1): no own ply in
  # this segment to credit, dropped. t1 terminal ply. t2: own action ends the
  # hand (+2), so the bootstrap (100) must not leak in.
  """Known answer segment starts midhand and own terminal action."""
  r = _row(
      reward=[-1, 0, 2], value=[0, 0, 0.5], valid=[0, 0, 1], done=[1, 0, 1]
  )
  out = adv_lib.gae(
      r["reward"],
      r["value"],
      r["valid"],
      r["done"],
      jnp.asarray([100.0], F),
      gamma=0.99,
      lam=0.95,
  )
  np.testing.assert_allclose(out.advantages[0], [0, 0, 1.5], rtol=1e-6)
  np.testing.assert_allclose(out.returns[0], [0, 0, 2.0], rtol=1e-6)


def test_lambda_one_gamma_one_is_monte_carlo_and_lambda_zero_is_td0():
  # One complete hand of 3 own plies; return +3 at the end.
  """Lambda one gamma one is monte carlo and lambda zero is td0."""
  r = _row(
      reward=[0, 0, 0, 0, 3, 0],
      value=[0.1, 0, 0.2, 0, 0.3, 0],
      valid=[1, 0, 1, 0, 1, 0],
      done=[0, 0, 0, 0, 1, 0],
  )
  mc = adv_lib.gae(
      r["reward"],
      r["value"],
      r["valid"],
      r["done"],
      jnp.asarray([9.0], F),
      gamma=1.0,
      lam=1.0,
  )
  np.testing.assert_allclose(mc.returns[0], [3, 0, 3, 0, 3, 0], rtol=1e-6)
  td0 = adv_lib.gae(
      r["reward"],
      r["value"],
      r["valid"],
      r["done"],
      jnp.asarray([9.0], F),
      gamma=1.0,
      lam=0.0,
  )
  np.testing.assert_allclose(td0.returns[0], [0.2, 0, 0.3, 0, 3, 0], rtol=1e-6)


def test_rows_are_independent_and_batched():
  """Rows are independent and batched."""
  a = _row(
      reward=[0, 0, 0, 1, 0, 0],
      value=[0.5, 0, 0.25, 0, 0, 0.1],
      valid=[1, 0, 1, 0, 0, 1],
      done=[0, 0, 0, 1, 0, 0],
  )
  b = _row(
      reward=[0, 1, 0, 0, 0, 0],
      value=[0, 0.3, 0, 0.7, 0, 0.9],
      valid=[0, 1, 0, 1, 0, 1],
      done=[0, 0, 0, 0, 0, 0],
  )
  cat = {k: jnp.concatenate([v, b[k]]) for k, v in a.items()}
  boot = jnp.asarray([0.2, -0.4], F)
  both = adv_lib.gae(
      cat["reward"],
      cat["value"],
      cat["valid"],
      cat["done"],
      boot,
      gamma=0.9,
      lam=0.7,
  )
  for i, r in enumerate((a, b)):
    one = adv_lib.gae(
        r["reward"],
        r["value"],
        r["valid"],
        r["done"],
        boot[i : i + 1],
        gamma=0.9,
        lam=0.7,
    )
    np.testing.assert_array_equal(both.advantages[i], one.advantages[0])
    np.testing.assert_array_equal(both.returns[i], one.returns[0])


def test_invalid_plies_are_zero_and_ignore_their_values():
  """Invalid plies are zero and ignore their values."""
  rng = np.random.default_rng(0)
  b, t = 4, 20
  valid = rng.random((b, t)) < 0.5
  done = rng.random((b, t)) < 0.1
  reward = (rng.normal(size=(b, t)) * done).astype(F)
  value = rng.normal(size=(b, t)).astype(F)
  boot = rng.normal(size=(b,)).astype(F)
  out = adv_lib.gae(reward, value, valid, done, boot, gamma=0.95, lam=0.9)
  assert not np.any(np.asarray(out.advantages)[~valid])
  assert not np.any(np.asarray(out.returns)[~valid])
  junk = np.where(valid, value, rng.normal(size=(b, t)) * 1e6).astype(F)
  out2 = adv_lib.gae(reward, junk, valid, done, boot, gamma=0.95, lam=0.9)
  np.testing.assert_array_equal(out.advantages, out2.advantages)


def test_dtypes_under_x64_and_input_validation():
  """Dtypes under x64 and input validation."""
  r = _row(reward=[0, 1], value=[0.5, 0.5], valid=[1, 1], done=[0, 1])
  out = adv_lib.gae(
      r["reward"],
      r["value"],
      r["valid"],
      r["done"],
      jnp.asarray([0.0], F),
      gamma=1.0,
      lam=1.0,
  )
  assert jax.config.jax_enable_x64
  assert out.advantages.dtype == jnp.float32
  assert out.returns.dtype == jnp.float32
  with pytest.raises(AssertionError):  # valid must be bool
    adv_lib.gae(
        r["reward"],
        r["value"],
        r["value"],
        r["done"],
        jnp.asarray([0.0], F),
        gamma=1.0,
        lam=1.0,
    )
  with pytest.raises(AssertionError):  # bootstrap shape [B]
    adv_lib.gae(
        r["reward"],
        r["value"],
        r["valid"],
        r["done"],
        jnp.zeros((2,), F),
        gamma=1.0,
        lam=1.0,
    )
  with pytest.raises(AssertionError):  # f64 value would silently promote
    adv_lib.gae(
        r["reward"],
        r["value"].astype(jnp.float64),
        r["valid"],
        r["done"],
        jnp.asarray([0.0], F),
        gamma=1.0,
        lam=1.0,
    )
  with pytest.raises(AssertionError):
    adv_lib.gae(
        r["reward"],
        r["value"],
        r["valid"],
        r["done"],
        jnp.asarray([0.0], F),
        gamma=1.0,
        lam=1.5,
    )


def test_jit_and_seat_value():
  """Jit and seat value."""
  vp = jnp.asarray(np.arange(2 * 3 * 2, dtype=F).reshape(2, 3, 2))
  seat = jnp.asarray([[0, 0, 0], [1, 1, 1]], jnp.int32)
  np.testing.assert_array_equal(
      adv_lib.seat_value(vp, seat), [[0, 2, 4], [7, 9, 11]]
  )
  r = _row(reward=[0, 1], value=[0.5, 0.5], valid=[1, 1], done=[0, 1])
  f = jax.jit(lambda *a: adv_lib.gae(*a, gamma=0.9, lam=0.8))
  out = f(r["reward"], r["value"], r["valid"], r["done"], jnp.asarray([0.0], F))
  ref = adv_lib.gae(
      r["reward"],
      r["value"],
      r["valid"],
      r["done"],
      jnp.asarray([0.0], F),
      gamma=0.9,
      lam=0.8,
  )
  np.testing.assert_array_equal(out.advantages, ref.advantages)


def test_from_batch_uses_perspective_seat_values():
  """batch_gae reads reward/valid/done and value_pred[..., seat] off a batch."""
  b, t, p = 1, 3, 2
  steps = types.SeatStep(
      info=None,
      legal_mask=None,
      seat=jnp.ones((b, t), jnp.int32),
      to_move=None,
      new_hand=None,
      new_match=None,
      last_action=None,
      last_actor=None,
  )
  vp = np.zeros((b, t, p), F)
  vp[0, :, 1] = [0.5, 0, 0.25]
  vp[0, :, 0] = [99, 99, 99]  # the other seat's head: must not be read
  batch = types.TrajectoryBatch(
      steps=steps,
      init_carry=None,
      action=None,
      behavior_logits=None,
      behavior_logp=None,
      value_pred=jnp.asarray(vp),
      target_policy=None,
      target_value=None,
      reward=jnp.asarray([[0, 1, 0]], F),
      done=jnp.asarray([[0, 1, 0]], bool),
      valid=jnp.asarray([[1, 0, 1]], bool),
      labels={},
      aux={},
      bootstrap_step=None,
  )
  out = adv_lib.batch_gae(batch, jnp.asarray([0.0], F), gamma=1.0, lam=1.0)
  np.testing.assert_allclose(out.advantages[0], [0.5, 0, -0.25], rtol=1e-6)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


def _random_trace(seed, b=16, t=48):
  """A 2-seat trace from seat 0's view, hands of random length.

  A 2-seat trace from seat 0's view, with hands of random length, one
  terminal ply after every hand, random seat order (so a seat may act twice).

  Args:
    seed: the numpy seed.
    b: rows.
    t: plies per row.

  Returns:
    (reward, value, valid, done, boot) numpy arrays, [b, t] and boot [b].
  """
  rng = np.random.default_rng(seed)
  valid = np.zeros((b, t), bool)
  done = np.zeros((b, t), bool)
  reward = np.zeros((b, t), F)
  for i in range(b):
    k = 0
    left = rng.integers(1, 9)
    while k < t:
      if left == 0:  # the terminal ply
        k += 1
        left = rng.integers(1, 9)
        continue
      valid[i, k] = rng.random() < 0.55
      left -= 1
      if left == 0:
        done[i, k] = True
        reward[i, k] = rng.choice([-2.0, -1.0, 1.0, 2.0])
      k += 1
  value = np.where(valid, rng.normal(size=(b, t)), 0.0).astype(F)
  boot = rng.normal(size=(b,)).astype(F)
  return reward, value, valid, done, boot


def test_eager_gae_compiles_once_per_shape():
  """Eager gae compiles once per shape.

  Called outside a jit (AZ / BC targets), gae must not recompile its scan on
  every call (it did: a fresh scan closure per call, one compile per AZ
  iteration).
  """
  b, t = 3, 5
  args = (
      jnp.ones((b, t), jnp.float32),
      jnp.zeros((b, t), jnp.float32),
      jnp.ones((b, t), jnp.bool_),
      jnp.zeros((b, t), jnp.bool_),
      jnp.zeros((b,), jnp.float32),
  )
  first = adv_lib.gae(*args, gamma=0.9, lam=0.8)
  with compiles.assert_no_compiles("second eager gae call"):
    again = adv_lib.gae(*args, gamma=0.9, lam=0.8)
  np.testing.assert_array_equal(
      np.asarray(first.advantages), np.asarray(again.advantages)
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
