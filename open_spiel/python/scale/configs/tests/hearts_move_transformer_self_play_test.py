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

"""configs.hearts_move_transformer_self_play: the recipe, and a CPU run."""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale import train
from open_spiel.python.scale.configs import (
    hearts_move_transformer_self_play as hmt,
)
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.envs import hearts
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners import ppo_magnet
from open_spiel.python.scale.networks import card_move_transformer as cmt
from open_spiel.python.scale.networks import catalog

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_recipe():
  cfg = common.load_config(hmt.NAME)
  assert cfg == hmt.get_config()
  assert cfg.env == "hearts" and dict(cfg.env_kwargs) == hearts.OPTIONS
  assert cfg.league.self_play_seats == "all"
  assert cfg.league.num_envs == 4096 and cfg.league.num_steps == hmt.NUM_STEPS
  assert cfg.run_updates() == 6000 and cfg.keep_every == 40
  lc = cfg.learner.learner
  assert lc.loss == "ppo_magnet" and lc.credit == "seat"
  assert lc.loss_config.field_players == 4
  assert lc.value_scale == 13.0 and lc.gamma == 1.0 and lc.gae_lambda == 0.5
  assert (
      lc.minibatch_size == 4096
      and lc.sgd_steps == 128
      and lc.update_epochs == 1
  )
  assert lc.max_grad_norm == 0.267 and lc.adam_eps == 1e-8
  assert lc.value_rows == hmt.VALUE_ROWS
  assert (
      lc.loss_config.rescale_step == 3000
      and lc.loss_config.rescale_time_scale == 14.0
  )
  assert (lc.lr_sched.anneal_start, lc.lr_sched.anneal_end) == (2999, 3000)
  net = cfg.policy_kwargs["network"]
  assert net["torso"]["kind"] == "card_move_transformer"
  assert (
      net["value_head"]["init_scheme"] == "torch_uniform"
      and net["torso"]["center_keys"]
  )
  assert net["torso"]["in_channels"] == 56 and net["torso"]["n_card"] == 52
  assert (
      cfg.policy_kwargs["num_players"] == 4
      and cfg.policy_kwargs["num_actions"] == 52
  )


def test_schedules():
  """The schedules: "continued" stretches both clocks, "single" never does.

  "continued" stretches them x14 from update 3000; the two configs differ
  only there.
  """
  cont, single = hmt.build(schedule="continued"), hmt.build(schedule="single")
  assert cont == hmt.get_config()
  lc, ls = cont.learner.learner, single.learner.learner
  assert (ls.loss_config.rescale_step, ls.loss_config.rescale_time_scale) == (
      0,
      1.0,
  )
  assert ls.lr_sched.anneal_end == 0 and ls.lr_sched.time_scale == 1.0
  for t, want in (
      (0, 1e-4),
      (2999, 0.5 / 3000**1.1),
      (3000, 5e-6),
      (5999, 5e-6),
  ):
    assert float(lc.lr_sched.value(t)) == pytest.approx(want, rel=1e-9), t
  for t in (0, 2999, 3000, 5999):
    assert float(ls.lr_sched.value(t)) == pytest.approx(
        min(1e-4, max(5e-6, 0.5 / (t + 1) ** 1.1)), rel=1e-9
    ), t
  steps = jnp.asarray([0, 2999, 3000, 5999], jnp.int64)
  tc = np.asarray(lc.loss_config.temperature(steps))
  ts = np.asarray(ls.loss_config.temperature(steps))
  np.testing.assert_allclose(
      ts,
      [
          min(0.1, max(1e-3, 0.05 / (n + 1) ** 0.3))
          for n in (0, 2999, 3000, 5999)
      ],
      rtol=1e-5,
  )
  np.testing.assert_allclose(
      tc,
      [
          min(0.1, max(1e-3, 0.05 / (n + 1) ** 0.3))
          for n in (0, 2999, 3000 * 14, 5999 * 14)
      ],
      rtol=1e-5,
  )
  with pytest.raises(ValueError):
    hmt.build(schedule="nope")
  same = dataclasses.replace(
      single, learner=dataclasses.replace(single.learner, learner=lc)
  )
  assert same == cont


def _small(cfg, num_updates=2):
  """The config at CPU size: a small network, 8 x 40 plies, small batches.

  Args:
    cfg: the run config.
    num_updates: the run length.

  Returns:
    The small config.
  """
  cfg = common.with_overrides(
      cfg, num_updates=num_updates, num_envs=8, num_steps=40
  )
  net = cmt.card_move_transformer(
      depth=1,
      embed_dim_per_head_over8=1,
      n_head=2,
      ff_factor=2,
      in_channels=56,
      n_card=52,
      n_special_action=0,
      compute_dtype="float32",
  )
  lc = dataclasses.replace(
      cfg.learner.learner, minibatch_size=64, sgd_steps=3, forward_chunk_rows=64
  )
  return dataclasses.replace(
      cfg,
      policy_kwargs={**dict(cfg.policy_kwargs), "network": net},
      learner=dataclasses.replace(cfg.learner, learner=lc),
      eval=hmt.evals(every=1, deals=4),
      keep_every=1,
  )


def test_batch_rows_are_tables_in_seat_order():
  """The field credit's layout: each 4 rows are one table's seats 0..3.

  So the loss's field rewards are the PlyLog's every-seat field rewards.
  """
  cfg = _small(hmt.get_config())
  parts = runner.one_slot_parts(cfg)
  learner = pg.PGLearner(parts.policy, cfg.learner.learner)
  s0 = learner.init(jax.random.key(0), parts.env)
  params = parts.slot_params(s0.params)
  dk, pk = arena_lib.independent_keys(jax.random.key(3), 8, 4)
  state = parts.arena.reset(params, parts.table, dk, pk)
  _, batch, log = parts.arena.rollout(params, parts.table, state, 60)
  seat = np.asarray(batch.steps.seat)
  assert seat.shape[0] == 32
  np.testing.assert_array_equal(
      seat.reshape(8, 4, -1),
      np.broadcast_to(np.arange(4)[None, :, None], (8, 4, 60)),
  )
  rewards = np.asarray(log.rewards, np.float32)  # [8, T, 4]
  want = (4 * rewards - rewards.sum(-1, keepdims=True)) / 3  # every seat's
  want = np.transpose(want, (0, 2, 1)).reshape(32, -1)
  got = np.asarray(
      ppo_magnet.field_rewards(jnp.asarray(batch.reward, jnp.float32), 4)
  )
  np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-5)
  assert np.abs(want).sum() > 0  # some hand ended


def test_runs_end_to_end_on_cpu(tmp_path):
  cfg = _small(hmt.get_config())
  res = train.run(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
  ls = res.league.learners["main"]
  assert int(ls.updates) == 2
  # every seat a perspective: 4 x 8 batch rows
  table, _, _, _ = res.league._draw(ls, np.int64(0))  # pylint: disable=protected-access
  assert np.asarray(table.seat_slot).shape == (8, 4)
  assert (np.asarray(table.seat_slot) == 0).all()


def test_network_options():
  """The defaults add the value-head init and key centring to the network.

  lecun and no centring give the catalog network unchanged.
  """
  base = catalog.get(hmt.NETWORK)
  assert (
      hmt.policy_kwargs(value_init="lecun", center_keys=False)["network"]
      == base
  )
  net = hmt.policy_kwargs()["network"]
  assert {k: v for k, v in net["torso"].items() if k != "center_keys"} == base[
      "torso"
  ]
  assert {
      k: v for k, v in net["value_head"].items() if k != "init_scheme"
  } == base["value_head"]
  with pytest.raises(ValueError, match="value_init"):
    hmt.policy_kwargs(value_init="orthogonal")


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider", "-s"]))
