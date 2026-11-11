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

"""BCLearner: masks, determinism, dtypes, counters and known answers.

BCLearner: masks (garbage at invalid plies or masked value targets changes
nothing), determinism under a key, declared dtypes under x64, an int64 step
counter that does not wrap past 2**31, and a state_dict round trip. Known
answers: the loss of a uniform tabular student on a hand-built target; BC
from a DAgger source on a fixed tabular teacher converges to the teacher on
kuhn (all 12 information states). The learner step on a batch read back
from shards equals the step on the online batch, bitwise.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners.tests import bc_fixtures
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.networks.tests import helpers as net_helpers
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources import shards
from open_spiel.python.scale.sources import targets
from open_spiel.python.scale.sources.tests import helpers
from open_spiel.python.scale.testing import fixtures

KUHN_ONEHOT = net_helpers.KUHN_ONEHOT
OBS_FIELD = net_helpers.OBS_FIELD
F32 = jnp.float32


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


def _kuhn_net(env, value="scalar", aux=False):
  """A Kuhn MLP network policy.

  Args:
    env: The env.
    value: The value head kind.
    aux: Whether to add an opp_card head.

  Returns:
    The policy.
  """
  vh = {"kind": value}
  if value == "hl_gauss":
    vh.update(num_bins=11, v_min=-2.0, v_max=2.0)
  net = {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": [16, 16], "layout": "shared"},
      "policy_head": {"kind": "dense"},
      "value_head": vh,
  }
  if aux:
    net["aux_heads"] = [{"key": "opp_card", "features": 3, "squeeze": False}]
  return policy_lib.NetworkPolicy(
      "kuhn_poker",
      "kuhn_onehot",
      env.num_actions,
      env.num_players,
      KUHN_ONEHOT,
      [12],
      net,
  )


def _teacher_logits(seed=0, scale=1.5):
  """Random teacher logits.

  Args:
    seed: The seed.
    scale: The logit scale.

  Returns:
    f32[12, 2].
  """
  return jnp.asarray(
      np.random.default_rng(seed).normal(size=(12, 2)) * scale, F32
  )


def _source(
    env, student, sp, *, batch=16, num_steps=8, beta=0.5, seed=0, lookahead=True
):
  """A Kuhn dagger source with a tabular teacher.

  Args:
    env: The env.
    student: The student.
    sp: Its params.
    batch: The rows.
    num_steps: The plies.
    beta: The teacher share (a float or a schedule).
    seed: The seed.
    lookahead: Whether to look ahead for returns.

  Returns:
    The DaggerSource.
  """
  teacher = helpers.kuhn_tabular(env)
  return dagger.DaggerSource(
      env,
      student,
      teacher,
      student_params=sp,
      teacher_params=teacher.params_from_logits(_teacher_logits(seed)),
      table=arena_lib.self_play_table(batch, 2, all_seats=True),
      num_steps=num_steps,
      beta=schedules.constant(beta) if isinstance(beta, float) else beta,
      gamma=1.0,
      return_lookahead=lookahead,
  )


def _tree_equal(a, b):
  helpers.assert_tree_equal(a, b)


# --- known-answer loss --------------------------------------------------------
@pytest.mark.parametrize(
    "policy,want",
    [
        ("ce_target", np.log(2.0)),
        ("kl_target", 0.25 * np.log(0.25 / 0.5) + 0.75 * np.log(0.75 / 0.5)),
    ],
)
def test_loss_known_answer_uniform_tabular(env, policy, want):
  """Loss known answer uniform tabular."""
  pol = helpers.kuhn_tabular(env)
  lrn = bc.BCLearner(
      pol,
      bc_fixtures.cfg(
          loss=losses.DistillLossConfig(policy=policy, value="none")
      ),
  )
  _, m = lrn.loss(
      pol.init_params(), bc_fixtures.two_ply_batch(env, [0.25, 0.75])
  )
  np.testing.assert_allclose(float(m["loss_policy"]), want, rtol=1e-6)
  assert (
      float(m["policy_accuracy"]) == 0.0
  )  # argmax of a uniform row is 0, the target's is 1


# --- contract: masks, determinism, dtypes, counters ---------------------------
@pytest.fixture(name="kuhn_batch", scope="module")
def _kuhn_batch_fixture(env):
  """A batch from the Kuhn dagger source.

  Args:
    env: The Kuhn env.

  Returns:
    The batch.
  """
  pol = _kuhn_net(env)
  return _source(env, pol, pol.init_params(jax.random.key(0))).next(
      jax.random.key(1)
  )


def test_invalid_plies_do_not_matter(env, kuhn_batch):
  """Invalid plies do not matter."""
  pol = _kuhn_net(env)
  lrn = bc.BCLearner(pol, bc_fixtures.cfg())
  p = pol.init_params(jax.random.key(0))
  b = kuhn_batch
  v = b.valid
  noisy = b._replace(
      target_policy=jnp.where(v[..., None], b.target_policy, F32(0.7)),
      target_value=jnp.where(v[..., None], b.target_value, F32(-9.0)),
  )
  g1 = jax.grad(lambda q: lrn.loss(q, b)[0])(p)
  g2 = jax.grad(lambda q: lrn.loss(q, noisy)[0])(p)
  _tree_equal(g1, g2)


def test_value_mask_label_is_honoured(env):
  """Value mask label is honoured."""
  pol = _kuhn_net(env)
  p = pol.init_params(jax.random.key(0))
  # No lookahead and T=4: seat-0 rows hold an unfinished hand at ply 3.
  b = _source(env, pol, p, num_steps=4, lookahead=False).next(jax.random.key(1))
  ok = b.labels[targets.TARGET_VALUE_VALID]
  assert bool(
      jnp.any(b.valid & ~ok)
  ), "the batch must hold masked value targets"
  lrn = bc.BCLearner(pol, bc_fixtures.cfg())
  noisy = b._replace(
      target_value=jnp.where(ok[..., None], b.target_value, F32(5.0))
  )
  _tree_equal(lrn.loss(p, b)[1], lrn.loss(p, noisy)[1])
  # ...and it is a real mask: without the label the noisy targets count.
  no_mask = noisy._replace(
      labels={
          k: x for k, x in b.labels.items() if k != targets.TARGET_VALUE_VALID
      }
  )
  assert float(lrn.loss(p, noisy)[1]["loss_value"]) != float(
      lrn.loss(p, no_mask)[1]["loss_value"]
  )


def test_update_deterministic_dtypes_and_minibatch_key(env, kuhn_batch):
  """Update deterministic dtypes and minibatch key."""
  pol = _kuhn_net(env)
  lrn = bc.BCLearner(pol, bc_fixtures.cfg(num_epochs=2, num_minibatches=2))
  s0 = lrn.init(jax.random.key(0), env)
  assert s0.step.dtype == jnp.int64 and int(s0.step) == 0
  s1, m1 = lrn.update(s0, kuhn_batch, jax.random.key(5))
  s1b, _ = lrn.update(s0, kuhn_batch, jax.random.key(5))
  s2, _ = lrn.update(s0, kuhn_batch, jax.random.key(6))
  _tree_equal(s1, s1b)
  assert int(s1.step) == 4  # 2 epochs x 2 minibatches
  diff = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), s1.params, s2.params)
  assert any(
      jax.tree.leaves(diff)
  ), "the shuffle key must matter with minibatches"
  for leaf in jax.tree.leaves(s1.params):
    assert leaf.dtype == F32
  for k, v in m1.items():
    assert v.dtype == F32, k
  assert {
      "loss_total",
      "loss_policy",
      "loss_value",
      "grad_norm",
      "policy_accuracy",
  } <= set(m1)


def test_full_batch_update_ignores_key(env, kuhn_batch):
  """Full batch update ignores key."""
  pol = _kuhn_net(env)
  lrn = bc.BCLearner(pol, bc_fixtures.cfg())
  s0 = lrn.init(jax.random.key(0), env)
  _tree_equal(
      lrn.update(s0, kuhn_batch, jax.random.key(1))[0],
      lrn.update(s0, kuhn_batch, jax.random.key(2))[0],
  )


def test_step_counter_past_2_31_does_not_wrap(env, kuhn_batch):
  """Step counter past 2 31 does not wrap."""
  pol = _kuhn_net(env)
  lrn = bc.BCLearner(pol, bc_fixtures.cfg(num_minibatches=2))
  s0 = lrn.init(jax.random.key(0), env)
  s0 = s0._replace(step=jnp.asarray(2**31 - 1, jnp.int64))
  s1, _ = lrn.update(s0, kuhn_batch, jax.random.key(0))
  assert s1.step.dtype == jnp.int64 and int(s1.step) == 2**31 + 1


def test_state_dict_round_trip(env, kuhn_batch):
  """State dict round trip."""
  pol = _kuhn_net(env)
  lrn = bc.BCLearner(pol, bc_fixtures.cfg())
  s1, _ = lrn.update(
      lrn.init(jax.random.key(0), env), kuhn_batch, jax.random.key(0)
  )
  blob = lrn.state_dict(s1)
  assert set(blob) == {"params", "opt_state", "step"}
  s1r = lrn.load_state(blob)
  _tree_equal(s1r, s1)
  _tree_equal(
      lrn.update(s1r, kuhn_batch, jax.random.key(3)),
      lrn.update(s1, kuhn_batch, jax.random.key(3)),
  )


def test_init_checks_env(env):
  """Init checks env."""
  pol = _kuhn_net(env)
  lrn = bc.BCLearner(pol, bc_fixtures.cfg())
  with pytest.raises(AssertionError):
    lrn.init(jax.random.key(0), leduc.LeducEnv())


def test_hl_gauss_value_and_aux_head_wiring(env, kuhn_batch):
  """HL-Gauss value and aux head wiring.

  A teacher-distillation loss on kuhn: KL + HL-Gauss value + an aux head on a
  privileged label (the opponent's card) + z-loss; equals distill_loss fed by
  hand from the policy's own outputs.

  Args:
    env: The kuhn env fixture.
    kuhn_batch: The hand-built kuhn batch fixture.
  """
  pol = _kuhn_net(env, value="hl_gauss", aux=True)
  cfg = losses.DistillLossConfig(
      policy="kl_target",
      value="hl_gauss",
      value_coef=0.5,
      z_loss_coef=1e-2,
      aux_terms=(losses.AuxTerm("opp_card", "softmax_ce", 0.1),),
  )
  lrn = bc.BCLearner(pol, bc_fixtures.cfg(loss=cfg))
  p = pol.init_params(jax.random.key(2))
  b = kuhn_batch
  total, m = lrn.loss(p, b)
  _, out = pol.evaluate_seq(p, b.init_carry, b.steps, None)
  seat = b.steps.seat[..., None]
  tv = jnp.take_along_axis(b.target_value, seat, -1)[..., 0]
  want, mw = losses.distill_loss(
      cfg,
      logits=out.logits,
      legal=b.steps.legal_mask,
      mask=b.valid,
      target_policy=b.target_policy,
      value_logits=out.aux["value_logits"],
      target_value=tv,
      value_mask=b.valid & b.labels[targets.TARGET_VALUE_VALID],
      action=jnp.argmax(
          jnp.where(b.steps.legal_mask, b.target_policy, -1.0), -1
      ),
      aux_preds={"opp_card": out.aux["opp_card"]},
      labels={"opp_card": b.labels["opp_card"]},
      value_support=pol.value_support(),
  )
  np.testing.assert_allclose(float(total), float(want), rtol=1e-6)
  assert float(m["aux/opp_card"]) > 0 and float(m["z_loss"]) >= 0
  for k in mw:
    np.testing.assert_allclose(float(m[k]), float(mw[k]), rtol=1e-6, err_msg=k)


def test_missing_aux_label_fails_loud(env, kuhn_batch):
  """Missing aux label fails loud."""
  pol = _kuhn_net(env, aux=True)
  cfg = losses.DistillLossConfig(
      policy="kl_target",
      value="none",
      aux_terms=(losses.AuxTerm("opp_hand", "sigmoid_ce", 0.1),),
  )
  lrn = bc.BCLearner(pol, bc_fixtures.cfg(loss=cfg))
  with pytest.raises((AssertionError, KeyError)):
    lrn.loss(pol.init_params(jax.random.key(0)), kuhn_batch)


# --- known answer: BC converges to a fixed tabular teacher on kuhn ------------
def test_bc_converges_to_tabular_teacher_on_kuhn(env):
  """BC converges to tabular teacher on kuhn."""
  pol = helpers.kuhn_tabular(env)
  n = 300
  src = _source(
      env,
      pol,
      pol.init_params(),
      batch=32,
      num_steps=8,
      beta=dagger.beta_schedule(start=1.0, end=0.0, decay_iters=n // 2),
      lookahead=False,
  )
  lrn = bc.BCLearner(
      pol,
      bc.BCConfig(
          loss=losses.DistillLossConfig(policy="kl_target", value="none"),
          learning_rate=0.05,
          lr_end=1e-3,
          total_steps=n,
          weight_decay=0.0,
          max_grad_norm=10.0,
      ),
  )
  st = lrn.init(jax.random.key(0), env)
  key = jax.random.key(42)
  for i in range(n):
    b = src.next(jax.random.fold_in(key, i))
    st, m = lrn.update(st, b, jax.random.fold_in(key, 10**6 + i))
    src.set_student_params(lrn.params(st))
  got = jax.nn.softmax(lrn.params(st)["logits"], axis=-1)
  want = jax.nn.softmax(_teacher_logits(0), axis=-1)
  np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-2)
  assert float(m["loss_policy"]) < 1e-3


# --- shard == online ----------------------------------------------------------
@pytest.mark.parametrize("stateful", [False, True])
def test_shard_equals_online(env, tmp_path, stateful):
  """Shard equals online."""
  if stateful:
    pol = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
    sp = pol.make_params()
    like = pol.init_carry(sp, 1, jnp.zeros((1,), jnp.int32), jax.random.key(0))
    cfg = bc_fixtures.cfg(
        loss=losses.DistillLossConfig(policy="kl_target", value="none")
    )
  else:
    pol = _kuhn_net(env)
    sp = pol.init_params(jax.random.key(0))
    like = None
    cfg = bc_fixtures.cfg()
  src = _source(env, pol, sp, batch=8, num_steps=6)
  online = [src.next(jax.random.key(i)) for i in range(3)]
  path = str(tmp_path / "shards")
  with shards.ShardWriter(
      path,
      env,
      rules_sha="r",
      info_schema_version=1,
      producer="test@0",
      carry_owner=pol.spec,
  ) as w:
    for b in online:
      w.write(b)
  reader = shards.ShardReader(
      path,
      env,
      rules_sha="r",
      info_schema_version=1,
      init_carry_like=like,
      student=pol.spec,
  )
  assert reader.carry == ("stored" if stateful else "none")
  lrn = bc.BCLearner(pol, cfg)
  s_on = s_sh = lrn.init_from_params(sp)
  rows = online[0].valid.shape[0]
  for i, b in enumerate(online):
    from_shard = reader.rows(i * rows, rows)
    _tree_equal(from_shard, b)
    s_on, m_on = lrn.update(s_on, b, jax.random.key(i))
    s_sh, m_sh = lrn.update(s_sh, from_shard, jax.random.key(i))
    _tree_equal(s_sh, s_on)
    _tree_equal(m_sh, m_on)


def test_stateful_student_without_a_carry_starts_from_its_own(env):
  """Stateful student without a carry starts from its own.

  Known answer: a reset-aligned batch with init_carry gives a stateful student
  exactly the loss and update of the same batch carrying the student's own
  init_carry(params, B, seat[:, 0], key(0)).

  Args:
    env: The kuhn env fixture.
  """
  pol = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  sp = pol.make_params(w=jnp.eye(env.num_actions, dtype=jnp.float32))
  b = _source(env, pol, sp, batch=8, num_steps=6).next(jax.random.key(3))
  assert np.asarray(b.steps.new_match[:, 0]).all()
  seat0 = jnp.asarray(b.steps.seat[:, 0]).astype(jnp.int32)
  own = pol.init_carry(sp, seat0.shape[0], seat0, jax.random.key(0))
  lrn = bc.BCLearner(
      pol,
      bc_fixtures.cfg(
          loss=losses.DistillLossConfig(policy="kl_target", value="none")
      ),
  )
  empty, explicit = b._replace(init_carry=()), b._replace(init_carry=own)
  le, me = lrn.loss(sp, empty)
  lx, mx = lrn.loss(sp, explicit)
  assert float(le) == float(lx)
  _tree_equal(me, mx)
  s0 = lrn.init_from_params(sp)
  _tree_equal(
      lrn.update(s0, empty, jax.random.key(1)),
      lrn.update(s0, explicit, jax.random.key(1)),
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
