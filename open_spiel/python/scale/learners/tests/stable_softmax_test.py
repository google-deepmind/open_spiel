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

"""learners.stable_softmax and the losses' stable_softmax option.

The paper's own worked examples (arXiv:2610.31642, Sections 2 and 3).
Softmax collapse: at logit gaps of 10, 20 and 40 (margins 8.6, 18.6 and 38.6
over four others) the gradient sums to 0 over the classes in fp32 and bf16;
jax.nn's does not past the dtype's cliff (16.6 fp32, 5.5 bf16). The KL terms'
gradient is exactly 0 at policy == reference. At low margins every stable
loss matches the standard one to 1e-6. The config field is out of the config
hash at its default.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners import stable_softmax as ss
from open_spiel.python.scale.learners.tests import toy_loss

F32 = jnp.float32
BF16 = jnp.bfloat16
#: The margin past which jax.nn's top-logit gradient rounds to 0 (the paper).
CLIFF = {F32: 16.6, BF16: 5.5}
GAPS = (10.0, 20.0, 40.0)
K = 5


def _std_ce(z, t):
  """jax.nn's hard-target CE.

  Args:
    z: [..., K] logits.
    t: int [...] targets.

  Returns:
    [...] -log_softmax(z)[t].
  """
  lp = jax.nn.log_softmax(z, axis=-1)
  return -jnp.take_along_axis(lp, t[..., None], axis=-1)[..., 0]


def _gap_logits(gap, dtype):
  """[gap, 0, 0, 0, 0]: the top logit leads four others by `gap`.

  Args:
    gap: the lead.
    dtype: the logits' dtype.

  Returns:
    The logits.
  """
  return jnp.asarray([gap] + [0.0] * (K - 1), dtype)


def _margin(gap):
  """The paper's logit margin -LSE(z_j - z_max) of _gap_logits(gap).

  Args:
    gap: the lead.

  Returns:
    The margin.
  """
  return gap - np.log(K - 1)


def _sums_to_zero(g, rtol) -> bool:
  """|sum g| <= rtol * sum |g|.

  Args:
    g: a gradient over the classes.
    rtol: the tolerance relative to its magnitude.

  Returns:
    Whether it sums to 0.
  """
  g = np.asarray(g, np.float64)
  return abs(g.sum()) <= rtol * np.abs(g).sum()


# --------------------------------------------------------------------------
# The paper's worked examples.
# --------------------------------------------------------------------------


def test_paper_example_hard_target():
  """Section 2: z = [9, -9, -9], t = 0."""
  z, t = jnp.asarray([9.0, -9.0, -9.0], F32), jnp.asarray(0)
  np.testing.assert_allclose(
      ss.log_softmax(z), [-3.046e-8, -18.0, -18.0], rtol=1e-3
  )
  v, g = jax.value_and_grad(ss.cross_entropy)(z, t)
  np.testing.assert_allclose(v, 3.046e-8, rtol=1e-3)
  np.testing.assert_allclose(g, [-3.046e-8, 1.523e-8, 1.523e-8], rtol=1e-3)
  np.testing.assert_allclose(
      jax.grad(lambda z: ss.softmax(z)[0])(z),
      [3.046e-8, -1.523e-8, -1.523e-8],
      rtol=1e-3,
  )
  # jax.nn's: the target gradient is 0 (collapse).
  assert float(jax.grad(_std_ce)(z, t)[0]) == 0.0


def test_paper_example_soft_target():
  """Section 3: a 1e-7 perturbation; the fp64 gradient is matched."""
  z_t = jnp.asarray([2.0, 1.0, 0.0], F32)
  z_s = z_t + jnp.asarray([-1e-7, 1e-7, 0.0], F32)
  g = jax.grad(ss.soft_target_cross_entropy)(z_s, z_t)
  np.testing.assert_allclose(
      g, [-4.595504e-8, 4.144190e-8, 4.513142e-9], rtol=1e-5
  )


# --------------------------------------------------------------------------
# Softmax collapse.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [F32, BF16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("gap", GAPS)
def test_collapse_gradient_sums_to_zero(gap, dtype):
  """CE to the top class and log_softmax[top]: gradients sum to 0."""
  z, t = _gap_logits(gap, dtype), jnp.asarray(0)
  rtol = 1e-5 if dtype == F32 else 1e-2
  new_ce = jax.grad(ss.cross_entropy)(z, t)
  new_lp = jax.grad(lambda z: ss.log_softmax(z)[0])(z)
  assert new_ce.dtype == dtype
  assert _sums_to_zero(new_ce, rtol), new_ce
  assert _sums_to_zero(new_lp, rtol), new_lp
  assert float(new_ce[0]) < 0.0  # the top logit's gradient survives
  old_ce = jax.grad(_std_ce)(z, t)
  old_lp = jax.grad(lambda z: jax.nn.log_softmax(z)[0])(z)
  if _margin(gap) > CLIFF[dtype]:
    assert float(old_ce[0]) == 0.0, old_ce
    assert not _sums_to_zero(old_ce, rtol), old_ce
    assert not _sums_to_zero(old_lp, rtol), old_lp


@pytest.mark.parametrize("gap", GAPS)
def test_collapse_through_the_losses(gap):
  """losses.action_log_prob / cross_entropy_to_action with stable=True.

  The PPO log-ratio and the one-hot BC loss, under a legal mask.

  Args:
    gap: the logit gap of the top class (GAPS).
  """
  z = jnp.asarray([[gap, 0.0, 0.0, 0.0, 0.0, 7.0]], F32)
  legal = jnp.asarray([[True] * K + [False]])
  a = jnp.asarray([0], jnp.int32)
  collapsed = _margin(gap) > CLIFF[F32]
  for stable in (True, False):
    for f in (loss_lib.cross_entropy_to_action, loss_lib.action_log_prob):
      g = jax.grad(lambda z, s=stable, f=f: f(z, legal, a, stable=s).sum())(z)
      assert float(g[0, -1]) == 0.0  # the illegal logit gets nothing
      if stable:
        assert _sums_to_zero(g, 1e-5), (f, g)
      elif collapsed:
        assert float(g[0, 0]) == 0.0 and not _sums_to_zero(g, 1e-5), (f, g)


# --------------------------------------------------------------------------
# KL: gradient exactly 0 at equality.
# --------------------------------------------------------------------------


def _rows(seed, n=16, a=9, scale=3.0, dtype=F32):
  """Random logits and legal masks; the last row has no legal action.

  Args:
    seed: the numpy seed.
    n: rows.
    a: actions.
    scale: the logits' std.
    dtype: the logits' dtype.

  Returns:
    (logits [n, a], legal bool [n, a]).
  """
  rng = np.random.default_rng(seed)
  z = (rng.standard_normal((n, a)) * scale).astype(np.float32)
  legal = rng.random((n, a)) < 0.7
  legal[:, 0] = True
  legal[-1] = False  # a row with no legal action
  return jnp.asarray(z, dtype), jnp.asarray(legal)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kl_gradient_is_exactly_zero_at_equality(seed):
  """kl_to_reference(pi, pi), kl_to_uniform(uniform): gradient exactly 0."""
  z, legal = _rows(seed)
  g = jax.grad(
      lambda x: loss_lib.kl_to_reference(x, z, legal, stable=True).sum()
  )(z)
  assert np.array_equal(np.asarray(g), np.zeros_like(g)), g
  v = loss_lib.kl_to_reference(z, z, legal, stable=True)
  assert np.array_equal(np.asarray(v), np.zeros_like(v)), v
  # Uniform over legal (any constant on the legal logits).
  u = jnp.where(legal, F32(1.25), z)
  g = jax.grad(lambda x: loss_lib.kl_to_uniform(x, legal, stable=True).sum())(u)
  assert np.array_equal(np.asarray(g), np.zeros_like(g)), g
  # The paper's soft-target CE (the gradient q - p).
  zz = jnp.where(legal, z, F32(loss_lib.NEG))
  g = jax.grad(lambda x: ss.soft_target_cross_entropy(x, zz).sum())(zz)
  assert np.array_equal(np.asarray(g), np.zeros_like(g)), g


@pytest.mark.parametrize("dtype", [F32, BF16], ids=["fp32", "bf16"])
def test_reverse_kl_zero_at_equality_any_dtype(dtype):
  """reverse_kl in fp32 and bf16: exactly 0 value and gradient at equality."""
  z, _ = _rows(3, dtype=dtype)
  v, g = jax.value_and_grad(lambda x: ss.reverse_kl(x, z).sum())(z)
  assert float(v) == 0.0 and not np.asarray(g, np.float32).any(), (v, g)


def test_reverse_kl_far_from_equality_is_finite():
  """Differences past expm1's overflow: no NaN; value and gradient ~ fp64."""
  zs = np.asarray([[0.0, -120.0, -5.0], [200.0, 0.0, 0.0], [3.0, 1.0, -150.0]])
  zt = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -110.0, 0.0]])
  for i in range(3):
    v, g = jax.value_and_grad(
        lambda x, i=i: ss.reverse_kl(x, jnp.asarray(zt[i], F32))
    )(jnp.asarray(zs[i], F32))
    assert np.isfinite(float(v)) and np.isfinite(np.asarray(g)).all(), (v, g)
    lq = zs[i] - np.log(np.exp(zs[i] - zs[i].max()).sum()) - zs[i].max()
    lp = zt[i] - np.log(np.exp(zt[i] - zt[i].max()).sum()) - zt[i].max()
    q = np.exp(lq)
    kl = float(np.sum(q * (lq - lp)))
    # fp32 rounding of differences ~100 apart: a few ulps of 100.
    np.testing.assert_allclose(float(v), kl, rtol=2e-5)
    np.testing.assert_allclose(
        np.asarray(g, np.float64), q * (lq - lp - kl), rtol=2e-5, atol=1e-30
    )


# --------------------------------------------------------------------------
# Low margins: the stable losses equal the standard ones.
# --------------------------------------------------------------------------


def _close(a, b, tol=1e-6):
  """Asserts a == b to tol (relative and absolute).

  Args:
    a: values.
    b: values.
    tol: the tolerance.
  """
  np.testing.assert_allclose(
      np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=tol, atol=tol
  )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_values_match_the_standard_path_at_low_margins(seed):
  """Every stable loss primitive vs its standard twin, to 1e-6."""
  z, legal = _rows(seed, scale=2.0)
  ref, _ = _rows(seed + 10, scale=2.0)
  rng = np.random.default_rng(seed)
  a = jnp.asarray(
      [rng.choice(np.flatnonzero(r)) if r.any() else 0 for r in legal],
      jnp.int32,
  )
  t = jax.nn.softmax(jnp.where(legal, ref, -jnp.inf), axis=-1)
  t = jnp.where(legal.any(-1, keepdims=True), t, 0.0).astype(F32)
  live = np.asarray(legal.any(-1))
  for name, f in (
      (
          "masked_log_softmax",
          lambda s: loss_lib.masked_log_softmax(z, legal, stable=s),
      ),
      (
          "action_log_prob",
          lambda s: loss_lib.action_log_prob(z, legal, a, stable=s),
      ),
      ("entropy", lambda s: loss_lib.entropy(z, legal, stable=s)),
      (
          "kl_to_reference",
          lambda s: loss_lib.kl_to_reference(z, ref, legal, stable=s),
      ),
      ("kl_to_uniform", lambda s: loss_lib.kl_to_uniform(z, legal, stable=s)),
      (
          "cross_entropy_to_target",
          lambda s: loss_lib.cross_entropy_to_target(z, t, legal, stable=s),
      ),
      (
          "kl_from_target",
          lambda s: loss_lib.kl_from_target(t, z, legal, stable=s),
      ),
      (
          "aux softmax_ce",
          lambda s: loss_lib.aux_loss("softmax_ce", z, a, stable=s),
      ),
  ):
    new, old = np.asarray(f(True)), np.asarray(f(False))
    if name == "masked_log_softmax":  # legal entries (illegal ~ NEG)
      new, old = new[np.asarray(legal)], old[np.asarray(legal)]
    elif name in ("action_log_prob",):
      new, old = new[live], old[live]
    _close(new, old)
  support = loss_lib.hl_gauss_support(-1.0, 1.0, 21)
  vl, _ = _rows(seed + 20, a=21, scale=2.0)
  target = jnp.asarray(np.linspace(-1.2, 1.2, vl.shape[0]), F32)
  _close(
      loss_lib.hl_gauss_cross_entropy(vl, target, support, stable=True),
      loss_lib.hl_gauss_cross_entropy(vl, target, support),
  )


def test_pg_and_distill_losses_match_at_low_margins():
  """pg_loss (mmd, HL-Gauss value) and distill_loss totals and metrics."""
  z, legal = _rows(5, scale=2.0)
  ref, _ = _rows(6, scale=2.0)
  n = z.shape[0]
  mask = jnp.asarray(legal.any(-1))
  a = jnp.argmax(jnp.where(legal, z, -jnp.inf), axis=-1).astype(jnp.int32)
  support = loss_lib.hl_gauss_support(-1.0, 1.0, 11)
  vl, _ = _rows(7, a=11, scale=1.0)
  rng = np.random.default_rng(0)
  adv = jnp.asarray(rng.standard_normal(n), F32)
  tgt = jnp.asarray(rng.uniform(-1, 1, n), F32)
  blp = loss_lib.action_log_prob(ref, legal, a)
  out = {}
  for s in (False, True):
    cfg = loss_lib.PGLossConfig(value_head="hl_gauss", stable_softmax=s)
    out[s] = loss_lib.pg_loss(
        cfg,
        logits=z,
        value=jnp.zeros(n, F32),
        legal=legal,
        action=a,
        behavior_logp=blp,
        mask=mask,
        advantages=adv,
        targets=tgt,
        value_logits=vl,
        reference_logits=ref,
        kl_coef=0.2,
        ent_coef=0.05,
        value_support=support,
    )
  _close(out[True][0], out[False][0])
  for k in out[False][1]:
    if not np.isnan(out[False][1][k]):
      _close(out[True][1][k], out[False][1][k])
  t = jax.nn.softmax(jnp.where(legal, ref, -jnp.inf), axis=-1)
  t = jnp.where(legal.any(-1, keepdims=True), t, 0.0).astype(F32)
  for policy in ("ce_action", "ce_target", "kl_target"):
    res = {}
    for s in (False, True):
      cfg = loss_lib.DistillLossConfig(
          policy=policy, value="hl_gauss", stable_softmax=s
      )
      res[s] = loss_lib.distill_loss(
          cfg,
          logits=z,
          legal=legal,
          mask=mask,
          action=a,
          target_policy=t,
          value_logits=vl,
          target_value=tgt,
          value_support=support,
      )
    _close(res[True][0], res[False][0])


# --------------------------------------------------------------------------
# Config hash.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls", [loss_lib.PGLossConfig, loss_lib.DistillLossConfig]
)
def test_stable_softmax_is_out_of_the_hash_at_its_default(cls):
  """Default: the key is absent from config_dict; set, it is hashed."""
  assert "stable_softmax" not in common.config_dict(cls())
  on = common.config_dict(cls(stable_softmax=True))["stable_softmax"]
  assert isinstance(on, bool) and on
  assert common.config_hash(cls()) != common.config_hash(
      cls(stable_softmax=True)
  )


def test_pg_config_refuses_stable_softmax_for_an_undeclared_loss():
  """A registered loss without LossSpec.stable_softmax cannot take it."""
  assert not toy_loss.SPEC.stable_softmax
  on = loss_lib.PGLossConfig(stable_softmax=True)
  with pytest.raises(ValueError, match="stable_softmax"):
    pg.PGConfig(
        loss=toy_loss.NAME,
        loss_config=toy_loss.ToyLossConfig(),
        pg=on,
        num_updates=1,
    )
  pg.PGConfig(
      loss=toy_loss.NAME, loss_config=toy_loss.ToyLossConfig(), num_updates=1
  )
  for loss in ("ppo", "mmd"):  # the built-in losses take it
    assert pg.PGConfig(loss=loss, pg=on, num_updates=1).pg.stable_softmax


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
