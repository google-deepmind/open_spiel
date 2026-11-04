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

"""Subprocess body of data_parallel_test (8 forced host devices).

pg      one PGLearner update, data parallel over 8 devices, equals the
1-device update to float rounding (rtol 1e-5) for mmd (flat shuffle,
value EMA, advantage normalisation) and ppo (env_axis, clipped value,
advantage filter); the new state is replicated. A registered loss
variant adds its own case (testing.multidevice.register_dp_case).
hlo     (known answer) the compiled 8-device update all-reduces (psum) and
runs the first dense layer on minibatch/8 rows per device, never on
the whole minibatch.
compile 4 more iterations (rollout + update) compile nothing.
bc      BCLearner (2 epochs x 2 minibatches) data parallel == 1 device.
az      AlphaZero (connect_four, 1 iteration, a one-slot league) on 8 devices
vs 1: self-play counters exactly equal, learner params after 20 Adam steps
within 1e-4 relative L2 (single coordinates within Adam's step bound).
stat    train.run kuhn MMD, 3 seeds x 30 updates on 8 devices vs 1: early
losses per seed within 1e-3 and the two seed-mean exploitabilities
within 3 standard errors of their difference (statistical equality;
the rollouts are exact, the
updates differ by rounding, and a rounding-level flip of one sampled
action makes the runs independent samples).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale import train
from open_spiel.python.scale.configs import az_connect_four
from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.eval import exploitability
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import az
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.parallel import compiles
from open_spiel.python.scale.parallel import data_parallel as dpl
from open_spiel.python.scale.testing import multidevice

tree_map = jax.tree_util.tree_map


def to_device0(tree):
  """A copy of a (sharded) tree on device 0 only (the 1-device reference).

  Args:
    tree: a pytree of arrays.

  Returns:
    the copy on device 0.
  """
  d0 = jax.devices()[0]
  return tree_map(lambda x: jax.device_put(x, d0), tree)


def pg_cases():
  # 128 x 16 self-play: 2048 compacted plies (PGLearner.train_rows), 512-row
  # minibatches (never 256, the hidden width: the hlo check below must not
  # mistake the [256, 256] weight for a whole-minibatch activation).
  """The policy-gradient cases: the kuhn configs and every registered case.

  Returns:
    (name, config) pairs.
  """
  kuhn = config_common.with_overrides(
      config_common.load_config("mmd_kuhn_poker_jax"),
      num_envs=128,
      num_steps=16,
      num_updates=50,
  )
  leduc = config_common.with_overrides(
      config_common.load_config("mmd_leduc_poker"),
      num_envs=64,
      num_steps=16,
      num_updates=50,
  )
  ppo = dataclasses.replace(
      leduc.learner.learner,
      loss="ppo",
      kl_coef=schedules.Schedule("constant", 0.0),
      shuffle="env_axis",
      value_ema=False,
      pg=losses.PGLossConfig(
          clip_eps=0.2, vf_coef=0.5, clip_value=True, use_advantage_filter=True
      ),
  )
  extra = tuple(
      (
          name,
          dataclasses.replace(kuhn, learner=kinds.PG(fn(kuhn.learner.learner))),
      )
      for name, fn in multidevice.dp_cases()
  )
  return (
      ("mmd_flat_kuhn", kuhn),
      ("ppo_env_axis_leduc", dataclasses.replace(leduc, learner=kinds.PG(ppo))),
  ) + extra


def pg_probe(dp):
  """Each PG case's 8-device update equals its 1-device update.

  Args:
    dp: The 8-device data-parallel context.
  """
  for name, cfg in pg_cases():
    parts = runner.one_slot_parts(cfg, mesh=dp.mesh)
    env, policy = parts.env, parts.policy
    learner = pg.PGLearner(policy, cfg.learner.learner)
    s0 = dp.replicate(
        learner.init(jax.random.key(1), env)
    )  # as the League holds it
    src = arena_lib.ArenaSource(
        parts.arena, parts.slot_params(s0.params), parts.table, parts.num_steps
    )
    batch = src.next(jax.random.key(2))
    assert len(batch.action.sharding.device_set) == 8
    ref_learner = pg.PGLearner(
        policy, cfg.learner.learner
    )  # its own jit cache, no scope
    s_ref, m_ref = ref_learner.update(
        to_device0(s0), to_device0(batch), jax.random.key(3)
    )
    s_dp, m_dp = dp.update(learner, s0, batch, jax.random.key(3))
    worst = multidevice.assert_trees_close(
        s_dp, s_ref, rtol=1e-5, atol=1e-6, what=f"{name} state"
    )
    for k, ref in m_ref.items():
      np.testing.assert_allclose(
          m_dp[k], ref, rtol=1e-4, atol=1e-6, err_msg=f"{name} {k}"
      )
    for leaf in jax.tree_util.tree_leaves(s_dp.params):
      assert (
          leaf.sharding.is_fully_replicated
          and len(leaf.sharding.device_set) == 8
      ), leaf.sharding
    assert int(s_dp.update_step) == 1 and s_dp.update_step.dtype == jnp.int64
    print(
        f"pg {name}: 8-device update == 1-device update, max |dparam| "
        f"{worst:.2e}"
    )

    if name == "mmd_flat_kuhn":
      with dp.scope():
        cap = learner.train_rows(
            batch
        )  # self-play: the valid-first compacted plies
      assert cap is not None and cap == ref_learner.train_rows(
          to_device0(batch)
      ), cap
      mb = cap // cfg.learner.learner.num_minibatches
      per = mb // 8
      assert (
          mb != 256
      ), "a 256-row minibatch is indistinguishable from the 256x256 weight"
      with dp.scope():
        txt = (
            # Reaches private _update_jit of the code under test.
            learner._update_jit.lower(  # pylint: disable=protected-access
                dp.replicate(s0),
                dp.shard_rows(batch),
                dp.replicate(jax.random.key(3)),
                capacity=cap,
            )
            .compile()
            .as_text()
        )
      found = multidevice.collectives_in(txt)
      assert found["all-reduce"] > 0, found
      assert (
          f"f32[{per},256]" in txt
      ), f"no per-device f32[{per},256] activation"
      assert (
          f"f32[{mb},256]" not in txt
      ), f"a whole-minibatch f32[{mb},256] activation"
      print(
          f"hlo: all-reduce x{found['all-reduce']}, dense layer on {per} of "
          f"{mb} rows "
          "per device"
      )
      keys = [
          (jax.random.key(20 + i), jax.random.key(40 + i)) for i in range(4)
      ]
      s = s_dp
      with compiles.assert_no_compiles(
          f"{name}: iterations 2-5 (rollout + update)"
      ):
        for kr, ku in keys:
          src.set_params(parts.slot_params(s.params))
          s, _ = dp.update(learner, s, src.next(kr), ku)
      print("compile: iterations 2-5 compiled none")


def small_az_config():
  """A small AlphaZero connect_four run (1 iteration).

  Returns:
    the run config.
  """
  c = dataclasses.replace(
      az_connect_four.az_config(),
      num_simulations=8,
      batch_size=16,
      num_iterations=1,
      num_epochs=2,
      minibatch_size=64,
      replay_buffer_size=1,
  )
  return recipes.alphazero("az_connect_four", c, keep_every=1)


def az_probe(dp):
  """AlphaZero on 8 devices vs 1: equal counters, params within tolerance.

  Args:
    dp: The 8-device data-parallel context.
  """
  league_cfg = small_az_config()
  cfg = league_cfg.learner.az
  r8 = train.run(league_cfg, mesh=dp.mesh)
  r1 = train.run(league_cfg, mesh=dpl.batch_mesh(1))
  m8, m1 = r8.metrics[-1], r1.metrics[-1]
  l8, l1 = r8.league.learners["main"], r1.league.learners["main"]
  assert int(l8.frames) == int(l1.frames) and int(l8.decisions) == int(
      l1.decisions
  )
  assert m8["avg_game_length"] == m1["avg_game_length"], (m8, m1)
  for k in ("loss_total", "loss_policy", "loss_value"):
    np.testing.assert_allclose(m8[k], m1[k], rtol=1e-4, err_msg=k)
  # 20 Adam steps (10 minibatches x 2 epochs): Adam's normalised step turns
  # rounding in a near-zero gradient coordinate into a step of up to lr (1e-3)
  # of either sign, so single coordinates may differ by ~1e-4; the params as a
  # whole agree to rounding.
  rel = multidevice.rel_l2(r8.state.params, r1.state.params)
  worst = multidevice.assert_trees_close(
      r8.state.params,
      r1.state.params,
      rtol=0.0,
      atol=20 * cfg.lr,
      what="az params",
  )
  assert rel < 1e-4, rel
  print(
      f"az: self-play counters equal, params rel L2 {rel:.2e}, max |d| "
      f"{worst:.2e}"
  )

  # BC with epochs x minibatches on the same kind of batch.
  _, base, _, arena, table, _, _ = az.build(cfg, mesh=dp.mesh)
  params = base.init_params(jax.random.key(5))
  src = arena_lib.ArenaSource(
      arena,
      [tree_map(lambda x: x[None], {"base": params, "sampler": {}})],
      table,
      43,
  )
  batch = src.next(jax.random.key(6))
  batch = az.outcome_targets(
      batch._replace(init_carry=batch.init_carry["base"])
  )
  bcfg = bc.BCConfig(
      loss=losses.DistillLossConfig(policy="ce_target", value="mse"),
      num_epochs=2,
      num_minibatches=2,
  )
  learner, ref = bc.BCLearner(base, bcfg), bc.BCLearner(base, bcfg)
  s0 = learner.init_from_params(params)
  s_ref, m_ref = ref.update(
      to_device0(s0), to_device0(batch), jax.random.key(7)
  )
  s_dp, m_dp = dp.update(learner, s0, batch, jax.random.key(7))
  worst = multidevice.assert_trees_close(
      s_dp, s_ref, rtol=1e-5, atol=1e-6, what="bc state"
  )
  for k in m_ref:
    np.testing.assert_allclose(
        m_dp[k], m_ref[k], rtol=1e-4, atol=1e-6, err_msg=f"bc {k}"
    )
  print(f"bc: 2x2 data parallel == 1 device, max |d| {worst:.2e}")


def stat_probe(dp):
  """MMD kuhn exploitability on 8 devices vs 1 over seeds, within a band.

  Args:
    dp: The 8-device data-parallel context.
  """
  base = config_common.with_overrides(
      config_common.load_config("mmd_kuhn_poker_jax"),
      num_envs=64,
      num_steps=16,
      num_updates=30,
  )
  env = config_common.build_env(base)
  tree = exploitability.GameTree(env)
  out = {}
  for seed in (0, 1, 2):
    cfg = config_common.with_overrides(base, seed=seed)
    for n, mesh in ((8, dp.mesh), (1, dpl.batch_mesh(1))):
      res = train.run(cfg, mesh=mesh)
      e = exploitability.exploitability(
          env, res.learner.policy(), to_device0(res.state.params), tree=tree
      ).exploitability
      out[(seed, n)] = (e, [m["loss_total"] for m in res.metrics])
  # The rollouts are exact and an update differs by rounding (pg_probe: one
  # update within rtol 1e-5), but a rounding-level logit difference flips a
  # sampled action now and then, and from that update on the two runs are
  # independent samples of the same training. So: early losses per seed, and the
  # two seed means within 3 standard errors of their difference, sd * sqrt(2/3)
  # with sd pooled over the six runs.
  for seed in (0, 1, 2):
    e8, l8 = out[(seed, 8)]
    e1, l1 = out[(seed, 1)]
    print(
        f"stat seed {seed}: exploitability 8 dev {e8:.5f} 1 dev {e1:.5f} |d| "
        f"{abs(e8 - e1):.2e}"
    )
    np.testing.assert_allclose(
        l8[:3], l1[:3], rtol=1e-3, err_msg=f"seed {seed} early losses"
    )
  e8s = np.array([out[(s, 8)][0] for s in (0, 1, 2)])
  e1s = np.array([out[(s, 1)][0] for s in (0, 1, 2)])
  se_diff = float(np.std(np.concatenate([e8s, e1s]), ddof=1)) * np.sqrt(
      2.0 / 3.0
  )
  band = max(1e-3, 3.0 * se_diff)
  print(
      f"stat mean: 8 dev {e8s.mean():.5f} 1 dev {e1s.mean():.5f} band "
      f"{band:.2e}"
  )
  assert abs(e8s.mean() - e1s.mean()) < band, (e8s, e1s, band)


def refusals(dp):
  """Rows that do not split over 8 devices fail loud, on host and in a trace.

  Args:
    dp: the 8-device DataParallel.

  Raises:
    AssertionError: a split of 12 rows over 8 devices was accepted.
  """
  try:
    dp.shard_rows(np.zeros((12, 3), np.float32))
  except ValueError:
    pass
  else:
    raise AssertionError("shard_rows split 12 rows over 8 devices")
  f = jax.jit(lambda x: dpl.constrain_rows(x, axis=1).sum())
  with dp.scope():
    try:
      f(jnp.zeros((2, 12), jnp.float32))
    except ValueError:
      pass
    else:
      raise AssertionError("constrain_rows pinned 12 rows to 8 devices")
    assert float(f(jnp.ones((2, 16), jnp.float32))) == 32.0
  print("refusals: indivisible rows refused")


PARTS = {"pg": (refusals, pg_probe), "az": (az_probe,), "stat": (stat_probe,)}


def main(part):
  """Runs the probe part `part` on 8 devices, then prints PROBE_OK.

  Args:
    part: a key of PARTS.
  """
  multidevice.expect_devices(8)
  dp = dpl.DataParallel(dpl.batch_mesh(8))
  for fn in PARTS[part]:
    fn(dp)
  print(multidevice.OK)


if __name__ == "__main__":
  # A script: x64 on before any array is made (here, not at import, which
  # changes no process-global state).
  frameworks.enable_x64()
  # After enable_x64 above, which must run first.
  import sys  # pylint: disable=g-import-not-at-top

  main(sys.argv[1])
