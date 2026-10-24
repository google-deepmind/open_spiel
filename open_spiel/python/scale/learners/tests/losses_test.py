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

"""learners.losses: known answers and masking invariance.

learners.losses: known answers worked by hand, and masking invariance
(illegal logits and invalid rows cannot move a loss or receive gradient).
"""

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.learners import losses as loss_lib

F = np.float32
RTOL = 1e-5


def _close(a, b, rtol=RTOL, atol=1e-7):
  np.testing.assert_allclose(
      np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=rtol, atol=atol
  )


# --------------------------------------------------------------------------
# Primitives: known answers.
# --------------------------------------------------------------------------


def test_masked_log_softmax_and_action_log_prob():
  """Masked log softmax and action log prob."""
  logits = jnp.asarray([[1.0, 2.0, 3.0]], F)
  legal = jnp.asarray([[True, False, True]])
  lse = math.log(math.exp(1) + math.exp(3))
  lp = loss_lib.masked_log_softmax(logits, legal)
  _close(lp[0, [0, 2]], [1 - lse, 3 - lse])
  assert float(lp[0, 1]) < -1e8
  _close(
      loss_lib.action_log_prob(logits, legal, jnp.asarray([2], jnp.int32)),
      [3 - lse],
  )
  assert lp.dtype == jnp.float32


def test_entropy_uniform_over_legal_is_log_k():
  """Entropy uniform over legal is log k."""
  logits = jnp.asarray([[5.0, -3.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]], F)
  legal = jnp.asarray([[True, False, True, True], [True, True, True, True]])
  _close(loss_lib.entropy(logits, legal), [math.log(3), math.log(4)])


def test_kl_to_reference_known_answer():
  # p = (.5, .5), ref = (.25, .75): KL = .5 ln 2 + .5 ln(2/3)
  """KL to reference known answer."""
  logits = jnp.asarray([[0.0, 0.0, 7.0]], F)
  ref = jnp.asarray([[0.0, math.log(3.0), -7.0]], F)
  legal = jnp.asarray([[True, True, False]])
  _close(
      loss_lib.kl_to_reference(logits, ref, legal),
      [0.5 * math.log(2) + 0.5 * math.log(2 / 3)],
  )
  _close(loss_lib.kl_to_reference(logits, logits, legal), [0.0], atol=1e-7)


def test_kl_to_uniform_known_answer():
  # p = (.25, .75) over 2 legal: KL(p || U) = ln 2 - H(p)
  """KL to uniform known answer."""
  logits = jnp.asarray([[0.0, math.log(3.0), 4.0]], F)
  legal = jnp.asarray([[True, True, False]])
  h = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
  _close(loss_lib.kl_to_uniform(logits, legal), [math.log(2) - h])


def test_ppo_clip_objective_known_answer():
  """PPO clip objective known answer."""
  ratio = jnp.asarray([0.5, 1.0, 1.5], F)
  adv = jnp.asarray([1.0, 2.0, -1.0], F)
  # min(r A, clip(r) A) = .5, 2, -1.5 -> loss (negated) = -.5, -2, 1.5
  _close(
      loss_lib.ppo_clip_objective(ratio, adv, clip_eps=0.2), [-0.5, -2.0, 1.5]
  )
  _close(loss_lib.clip_fraction(ratio, clip_eps=0.2), [1.0, 0.0, 1.0])


def test_normalize_advantages_excludes_masked_rows():
  """Normalize advantages excludes masked rows."""
  adv = jnp.asarray([1.0, 2.0, 100.0], F)
  out = loss_lib.normalize_advantages(adv, jnp.asarray([True, True, False]))
  _close(out[:2], [-1.0, 1.0], rtol=1e-6)


def test_masked_mean_empty_is_zero_with_zero_grad():
  """Masked mean empty is zero with zero grad."""
  x = jnp.asarray([1.0, 2.0], F)
  m = jnp.asarray([False, False])
  assert float(loss_lib.masked_mean(x, m)) == 0.0
  g = jax.grad(lambda x: loss_lib.masked_mean(x, m))(x)
  np.testing.assert_array_equal(g, [0.0, 0.0])
  _close(loss_lib.masked_mean(x, jnp.asarray([True, False])), 1.0)


def test_value_errors_known_answer():
  """Value errors known answer."""
  v = jnp.asarray([1.0, 1.0], F)
  _close(
      loss_lib.value_squared_error(v, jnp.asarray([3.0, 0.0], F)), [4.0, 1.0]
  )
  # clipped: old .5, eps .2: v_clip = .7 -> max((1-3)^2, (.7-3)^2) = 5.29
  _close(
      loss_lib.clipped_value_squared_error(
          v,
          jnp.asarray([0.5, 0.5], F),
          jnp.asarray([3.0, 0.0], F),
          clip_eps=0.2,
      ),
      [5.29, 1.0],
  )


def test_hl_gauss_known_answers():
  """HL-Gauss known answers."""
  s = loss_lib.hl_gauss_support(-1.0, 1.0, 2, 0.75)
  probs = loss_lib.hl_gauss_target_probs(jnp.asarray([0.0], F), s)
  _close(probs, [[0.5, 0.5]], rtol=1e-6)
  _close(
      loss_lib.hl_gauss_cross_entropy(
          jnp.zeros((1, 2), F), jnp.asarray([0.0], F), s
      ),
      [math.log(2)],
      rtol=1e-6,
  )
  s101 = loss_lib.hl_gauss_support(-110.0, 110.0, 101, 0.75)
  y = jnp.asarray([-37.3, 0.0, 12.5, 80.0], F)
  p = loss_lib.hl_gauss_target_probs(y, s101)
  _close(p.sum(-1), np.ones(4), rtol=1e-6)
  _close(
      jnp.sum(p * s101.centers, -1), y, rtol=1e-3, atol=1e-3
  )  # mean-preserving interior
  _close(
      loss_lib.hl_gauss_logits_to_scalar(jnp.log(p + 1e-30), s101),
      y,
      rtol=1e-3,
      atol=1e-3,
  )
  with pytest.raises(ValueError):
    loss_lib.hl_gauss_support(1.0, -1.0, 10)
  with pytest.raises(ValueError):
    loss_lib.hl_gauss_support(-1.0, 1.0, 1)


def test_z_loss_known_answer():
  """Z loss known answer."""
  logits = jnp.asarray([[0.0, 0.0, 0.0, 9.0]], F)
  legal = jnp.asarray([[True, True, True, False]])
  _close(loss_lib.z_loss(logits, legal), [math.log(3) ** 2])


def test_distill_primitives_known_answers():
  """Distill primitives known answers."""
  logits = jnp.asarray([[0.0, math.log(3.0), 5.0]], F)
  legal = jnp.asarray([[True, True, False]])
  onehot = jnp.asarray([[0.0, 1.0, 0.0]], F)
  _close(
      loss_lib.cross_entropy_to_target(logits, onehot, legal), [-math.log(0.75)]
  )
  _close(
      loss_lib.cross_entropy_to_action(
          logits, legal, jnp.asarray([1], jnp.int32)
      ),
      [-math.log(0.75)],
  )
  target = jnp.asarray([[0.25, 0.75, 0.0]], F)
  _close(loss_lib.kl_from_target(target, logits, legal), [0.0], atol=1e-6)
  # KL(t || q) with t=(.5,.5), q=(.25,.75): .5 ln2 + .5 ln(2/3)
  _close(
      loss_lib.kl_from_target(jnp.asarray([[0.5, 0.5, 0.0]], F), logits, legal),
      [0.5 * math.log(2) + 0.5 * math.log(2 / 3)],
  )


def test_aux_losses_known_answers():
  """Aux losses known answers."""
  _close(
      loss_lib.sigmoid_cross_entropy(
          jnp.zeros((2, 4), F), jnp.asarray([[0, 1, 0, 1]] * 2, F)
      ),
      [math.log(2)] * 2,
  )
  _close(
      loss_lib.aux_loss(
          "sigmoid_ce", jnp.zeros((1, 3), F), jnp.ones((1, 3), F)
      ),
      [math.log(2)],
  )
  _close(
      loss_lib.aux_loss(
          "mse", jnp.asarray([[1.0, 2.0]], F), jnp.asarray([[0.0, 0.0]], F)
      ),
      [2.5],
  )
  _close(
      loss_lib.aux_loss(
          "softmax_ce", jnp.zeros((1, 4), F), jnp.asarray([2], jnp.int32)
      ),
      [math.log(4)],
  )
  with pytest.raises(ValueError):
    loss_lib.aux_loss("nope", jnp.zeros((1, 3), F), jnp.ones((1, 3), F))


# --------------------------------------------------------------------------
# Schedules, magnet refresh, counters (int64, x64 on).
# --------------------------------------------------------------------------


def test_magnet_refresh_every_n_int64():
  """Magnet refresh every n int64."""
  assert bool(loss_lib.magnet_due(jnp.asarray(500, jnp.int64), 500))
  assert not bool(loss_lib.magnet_due(jnp.asarray(499, jnp.int64), 500))
  step = jnp.asarray(
      2**31 + 1500 - (2**31 % 500), jnp.int64
  )  # multiple of 500 past 2**31
  assert int(step) % 500 == 0 and int(step) > 2**31
  assert bool(loss_lib.magnet_due(step, 500))
  p = {"w": jnp.ones((2,), F)}
  m = {"w": jnp.zeros((2,), F)}
  np.testing.assert_array_equal(
      loss_lib.refresh_magnet(p, m, jnp.asarray(500, jnp.int64), 500)["w"],
      [1.0, 1.0],
  )
  np.testing.assert_array_equal(
      loss_lib.refresh_magnet(p, m, jnp.asarray(501, jnp.int64), 500)["w"],
      [0.0, 0.0],
  )
  with pytest.raises(AssertionError):
    loss_lib.magnet_due(jnp.asarray(5, jnp.int32), 500)  # counters are int64


# --------------------------------------------------------------------------
# Masking invariance: illegal logits and invalid rows do not affect any loss.
# --------------------------------------------------------------------------


def _batch(seed, n=64, a=9, all_legal_rows=True, hl_bins=0):
  """A random flat batch for the loss tests.

  Args:
    seed: The seed.
    n: The rows.
    a: The actions.
    all_legal_rows: Whether every row has a legal action.
    hl_bins: HL-Gauss bins, 0 for none.

  Returns:
    {name: array}.
  """
  rng = np.random.default_rng(seed)
  legal = rng.random((n, a)) < 0.6
  legal[np.arange(n), rng.integers(0, a, n)] = True
  if not all_legal_rows:
    legal[: n // 8] = False
  logits = (rng.normal(size=(n, a)) * 2).astype(F)
  beh = (logits + rng.normal(size=(n, a)) * 0.4).astype(F)
  lp_beh = np.asarray(
      loss_lib.masked_log_softmax(jnp.asarray(beh), jnp.asarray(legal))
  )
  action = np.array(
      [rng.choice(np.flatnonzero(l)) if l.any() else 0 for l in legal], np.int32
  )
  b = dict(
      logits=logits,
      legal=legal,
      action=action,
      behavior_logp=np.where(
          legal.any(-1), lp_beh[np.arange(n), action], 0.0
      ).astype(F),
      behavior_log_probs=lp_beh.astype(F),
      reference_logits=(logits + rng.normal(size=(n, a))).astype(F),
      advantages=rng.normal(size=n).astype(F),
      targets=(rng.normal(size=n) * 2).astype(F),
      value=rng.normal(size=n).astype(F),
      value_old=rng.normal(size=n).astype(F),
      mask=legal.any(-1),
      target_policy=None,
  )
  tp = np.where(legal, rng.random((n, a)), 0.0)
  b["target_policy"] = (
      tp / np.maximum(tp.sum(-1, keepdims=True), 1e-9)
  ).astype(F)
  if hl_bins:
    b["value_logits"] = rng.normal(size=(n, hl_bins)).astype(F)
  return {k: jnp.asarray(v) for k, v in b.items()}


PG_CFGS = {
    "ppo": loss_lib.PGLossConfig(clip_eps=0.2, vf_coef=0.5),
    "mmd_clipv": loss_lib.PGLossConfig(
        clip_eps=0.1, vf_coef=0.5, clip_value=True
    ),
    "mmd_filter": loss_lib.PGLossConfig(
        clip_eps=0.1, vf_coef=0.5, use_advantage_filter=True
    ),
    "mmd_hl": loss_lib.PGLossConfig(
        clip_eps=0.1, vf_coef=0.5, value_head="hl_gauss"
    ),
}
HL_PG = (
    -3.0,
    3.0,
    21,
    0.75,
)  # the HL-Gauss value head's support (v_min, v_max, bins, sigma)


def _pg(cfg, b, ref=True, **kw):
  """pg_loss over the batch `b`.

  Args:
    cfg: The loss config.
    b: The batch.
    ref: Whether to pass the reference logits.
    **kw: More pg_loss arguments.

  Returns:
    pg_loss's output.
  """
  return loss_lib.pg_loss(
      cfg,
      logits=b["logits"],
      value=b["value"],
      legal=b["legal"],
      action=b["action"],
      behavior_logp=b["behavior_logp"],
      mask=b["mask"],
      advantages=b["advantages"],
      targets=b["targets"],
      value_old=b["value_old"],
      value_logits=b.get("value_logits"),
      reference_logits=b["reference_logits"] if ref else None,
      kl_coef=0.2 if ref else 0.0,
      ent_coef=0.05,
      value_support=loss_lib.hl_gauss_support(*HL_PG)
      if cfg.value_head == "hl_gauss"
      else None,
      **kw
  )


def _perturb_illegal(b, seed):
  """The batch with junk in every illegal entry.

  Args:
    b: The batch.
    seed: The seed.

  Returns:
    The perturbed batch.
  """
  rng = np.random.default_rng(seed)
  junk = jnp.asarray(
      rng.choice([-1e30, 1e30, 3e5, -7.0], size=b["logits"].shape), F
  )
  out = dict(b)
  for k in ("logits", "reference_logits", "behavior_log_probs"):
    out[k] = jnp.where(b["legal"], b[k], junk)
  # Mass a (buggy) target puts on illegal actions is never read.
  out["target_policy"] = jnp.where(
      b["legal"],
      b["target_policy"],
      jnp.asarray(rng.random(b["logits"].shape), F),
  )
  return out


def _perturb_invalid_rows(b, seed):
  """The batch with junk in every row the mask excludes.

  Args:
    b: The batch.
    seed: The seed.

  Returns:
    The perturbed batch.
  """
  rng = np.random.default_rng(seed)
  out = dict(b)
  inv = ~b["mask"]
  for k in (
      "logits",
      "reference_logits",
      "value",
      "value_old",
      "targets",
      "advantages",
      "behavior_logp",
      "value_logits",
  ):
    if k in b:
      x = b[k]
      m = inv.reshape(inv.shape + (1,) * (x.ndim - 1))
      out[k] = jnp.where(m, jnp.asarray(rng.normal(size=x.shape) * 1e3, F), x)
  return out


@pytest.mark.parametrize("name", sorted(PG_CFGS))
def test_pg_loss_masking_invariance(name):
  """PG loss masking invariance."""
  cfg = PG_CFGS[name]
  b = _batch(
      0, all_legal_rows=False, hl_bins=21 if cfg.value_head == "hl_gauss" else 0
  )

  def _f(logits, b):
    """The total pg loss as a function of the logits.

    Args:
      logits: The logits.
      b: The batch.

    Returns:
      The loss.
    """
    return _pg(cfg, dict(b, logits=logits))[0]

  base, g = jax.value_and_grad(_f)(b["logits"], b)
  for bb in (_perturb_illegal(b, 1), _perturb_invalid_rows(b, 2)):
    val2, grad2 = jax.value_and_grad(_f)(bb["logits"], bb)
    _close(val2, base, rtol=1e-6)
    _close(
        jnp.where(b["legal"] & b["mask"][:, None], grad2, 0.0),
        jnp.where(b["legal"] & b["mask"][:, None], g, 0.0),
        rtol=1e-5,
        atol=1e-9,
    )
    assert not np.any(
        np.asarray(grad2)[~np.asarray(b["legal"] & b["mask"][:, None])]
    )


def test_losses_are_f32_under_x64():
  """Losses are f32 under x64."""
  b = _batch(1, hl_bins=21)
  for cfg in PG_CFGS.values():
    total, metrics = _pg(cfg, b)
    assert total.dtype == jnp.float32, cfg
    for k, v in metrics.items():
      assert v.dtype == jnp.float32, (k, v.dtype)
  with pytest.raises(AssertionError):
    _pg(PG_CFGS["ppo"], dict(b, logits=b["logits"].astype(jnp.float64)))


def test_pg_loss_is_sum_of_its_terms():
  """PG loss is sum of its terms."""
  b = _batch(2)
  cfg = PG_CFGS["ppo"]
  total, m = _pg(cfg, b)
  _close(
      total,
      m["loss_policy"]
      + 0.5 * m["loss_value"]
      - 0.05 * m["entropy"]
      + 0.2 * m["kl_penalty"],
  )
  total0, m0 = _pg(cfg, b, ref=False)
  assert np.isnan(float(m0["kl_penalty"]))
  _close(
      total0, m0["loss_policy"] + 0.5 * m0["loss_value"] - 0.05 * m0["entropy"]
  )


def test_pg_loss_kl_init_to_uniform_on_first_update():
  """PG loss KL init to uniform on first update."""
  b = _batch(3)
  cfg = loss_lib.PGLossConfig(
      clip_eps=0.2, vf_coef=0.5, kl_init_to_uniform=True
  )
  _, m = _pg(cfg, b, is_first_update=jnp.asarray(True))
  _close(
      m["kl_penalty"],
      loss_lib.masked_mean(
          loss_lib.kl_to_uniform(b["logits"], b["legal"]), b["mask"]
      ),
  )
  _, m2 = _pg(cfg, b, is_first_update=jnp.asarray(False))
  _close(
      m2["kl_penalty"],
      loss_lib.masked_mean(
          loss_lib.kl_to_reference(
              b["logits"], b["reference_logits"], b["legal"]
          ),
          b["mask"],
      ),
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
