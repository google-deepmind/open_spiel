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

"""A recurrent policy's carry through the PG learner.

A recurrent policy's carry through the PG learner: the carry flows only
through observe, so the learner cannot drop it.

The learner's forward over a TrajectoryBatch (evaluate_seq from the batch's
init_carry) reproduces exactly the logits the Arena acted with live, on
segments that start mid-hand with a non-zero carry; a forward that drops
the carry every ply does not; step-2 logits depend on the step-1 input; and
the first update's importance ratios are exactly 1.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx

tree_map = jax.tree_util.tree_map


def _learner(network, shuffle="env_axis"):
  """A Leduc PG learner with `network`.

  Args:
    network: The network config.
    shuffle: The shuffle mode.

  Returns:
    (policy, learner, state).
  """
  pol = fx.make_policy("leduc_poker", network)
  cfg = pg.PGConfig(
      loss="mmd",
      lr=1e-3,
      num_updates=10,
      update_epochs=1,
      num_minibatches=2,
      shuffle=shuffle,
      kl_coef=schedules.Schedule("constant", 0.2),
  )
  learner = pg.PGLearner(pol, cfg)
  return (
      pol,
      learner,
      learner.init(jax.random.key(0), fx.make_env("leduc_poker")),
  )


def _dropped_carry_logits(pol, params, steps):
  """A forward that drops the carry: every ply starts from a fresh carry.

  Args:
    pol: The policy.
    params: Its params.
    steps: The SeatSteps [B, T].

  Returns:
    The logits [B, T, A].
  """
  b, t = steps.seat.shape
  outs = []
  for i in range(t):
    st = tree_map(lambda x, i=i: x[:, i], steps)
    c = pol.init_carry(params, b, st.seat, None)
    outs.append(
        pol.evaluate(params, pol.observe(params, c, st), st, None).logits
    )
  return jnp.stack(outs, axis=1)


def _valid_diff(a, b, valid):
  """The largest difference over valid plies.

  Args:
    a: Logits.
    b: Other logits.
    valid: bool[B, T], the valid plies.

  Returns:
    The difference.
  """
  return float(jnp.max(jnp.where(valid[..., None], jnp.abs(a - b), 0.0)))


def test_learner_forward_equals_live_logits_across_segments():
  """Learner forward equals live logits across segments."""
  pol, learner, s = _learner(fx.rnn())
  src = fx.source(pol, s.params, "leduc_poker", b=4, t=5)
  saw_mid_hand_carry = False
  for i in range(4):
    batch = src.next(jax.random.key(10 + i))
    h0 = np.asarray(batch.init_carry["h"])
    if i > 0 and np.abs(h0).max() > 0:
      saw_mid_hand_carry = True
    logits, _, _ = learner.forward(s.params, batch.init_carry, batch.steps)
    assert _valid_diff(logits, batch.behavior_logits, batch.valid) <= 1e-6
    lp = loss_lib.action_log_prob(
        logits, batch.steps.legal_mask, jnp.maximum(batch.action, 0)
    )
    np.testing.assert_allclose(
        np.asarray(jnp.where(batch.valid, lp, 0.0)),
        np.asarray(batch.behavior_logp),
        atol=1e-5,
    )
    bug = _dropped_carry_logits(pol, s.params, batch.steps)
    assert (
        _valid_diff(bug, batch.behavior_logits, batch.valid) > 1e-3
    ), "the planted dropped-carry policy must be distinguishable"
  assert (
      saw_mid_hand_carry
  ), "segments 2+ must start mid-hand with a non-zero carry"


def test_step2_logits_depend_on_step1_input():
  """Step2 logits depend on step1 input."""
  pol, learner, s = _learner(fx.rnn())
  batch = fx.self_play_batch(pol, s.params, game="leduc_poker", b=4, t=6)
  info = batch.steps.info
  pert = batch.steps._replace(
      info=info._replace(card=info.card.at[:, 0].set((info.card[:, 0] + 1) % 6))
  )
  a, _, _ = learner.forward(s.params, batch.init_carry, batch.steps)
  b, _, _ = learner.forward(s.params, batch.init_carry, pert)
  assert float(jnp.max(jnp.abs(a[:, 1] - b[:, 1]))) > 1e-4
  # A feed-forward policy does not carry the perturbation to ply 1.
  pol_ff, learner_ff, s_ff = _learner(fx.mlp(), shuffle="flat")
  batch_ff = fx.self_play_batch(
      pol_ff, s_ff.params, game="leduc_poker", b=4, t=6
  )
  info = batch_ff.steps.info
  pert = batch_ff.steps._replace(
      info=info._replace(card=info.card.at[:, 0].set((info.card[:, 0] + 1) % 6))
  )

  def flat(st):
    """Merges the row and time axes.

    Args:
      st: SeatSteps, leaves [B, T, ...].

    Returns:
      The SeatSteps.
    """
    return tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), st)

  a, _, _ = learner_ff.forward(s_ff.params, (), flat(batch_ff.steps))
  b, _, _ = learner_ff.forward(s_ff.params, (), flat(pert))
  rows, t = batch_ff.valid.shape
  a, b = a.reshape(rows, t, -1), b.reshape(rows, t, -1)
  np.testing.assert_array_equal(np.asarray(a[:, 1:]), np.asarray(b[:, 1:]))


def test_first_minibatch_ratios_are_one():
  """First minibatch ratios are one."""
  pol, learner, s = _learner(fx.rnn())
  src = fx.source(pol, s.params, "leduc_poker", b=4, t=5)
  src.next(jax.random.key(1))
  batch = src.next(jax.random.key(2))  # mid-hand segment
  prep = learner.prepare(s, batch)
  _, per_mb = learner.update_prepared(s, batch, prep, jax.random.key(3))
  assert float(per_mb["clip_frac_policy"][0, 0]) == 0.0
  assert abs(float(per_mb["approx_kl"][0, 0])) <= 1e-6


def test_recurrent_policy_rejects_flat_shuffle():
  """Recurrent policy rejects flat shuffle."""
  with pytest.raises(ValueError, match="env_axis"):
    _learner(fx.rnn(), shuffle="flat")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
