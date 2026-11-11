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

"""configs.euchre_move_transformer_self_play: the recipe, and a CPU run."""

import dataclasses
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale import train
from open_spiel.python.scale.configs import (
    euchre_move_transformer_self_play as emt,
)
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.encoders import euchre_card_planes as ecp
from open_spiel.python.scale.envs import euchre
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import card_move_transformer as cmt
from open_spiel.python.scale.networks import catalog

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_recipe():
  cfg = common.load_config(emt.NAME)
  assert cfg == emt.get_config()
  assert cfg.env == "euchre" and dict(cfg.env_kwargs) == euchre.OPTIONS
  assert cfg.league.self_play_seats == "all"
  assert cfg.league.num_envs == 4096 and cfg.league.num_steps == 84
  assert cfg.run_updates() == 3000 and cfg.keep_every == 100
  lc = cfg.learner.learner
  assert lc.loss == "ppo_magnet" and lc.credit == "seat"
  assert lc.loss_config.field_players == 0  # the env's returns are the team's
  assert (lc.loss_config.rescale_step, lc.loss_config.rescale_time_scale) == (
      0,
      1.0,
  )
  assert lc.value_scale == 2.0 and lc.gamma == 1.0 and lc.gae_lambda == 0.5
  assert (
      lc.minibatch_size == 4096
      and lc.sgd_steps == 128
      and lc.update_epochs == 1
  )
  assert lc.max_grad_norm == 0.267 and lc.adam_eps == 1e-8
  assert lc.value_rows == emt.VALUE_ROWS
  for t in (0, 400, 2999):
    assert float(lc.lr_sched.value(t)) == pytest.approx(
        min(1e-4, max(5e-6, 0.5 / (t + 1) ** 1.1)), rel=1e-9
    ), t
  net = cfg.policy_kwargs["network"]
  assert net["torso"]["kind"] == "card_move_transformer"
  assert (
      net["value_head"]["init_scheme"] == "torch_uniform"
      and net["torso"]["center_keys"]
  )
  assert net["torso"]["in_channels"] == 64 and net["torso"]["n_card"] == 24
  assert net["torso"]["n_special_action"] == 7
  assert (
      cfg.policy_kwargs["num_players"] == 4
      and cfg.policy_kwargs["num_actions"] == 31
  )


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
      in_channels=64,
      n_card=24,
      n_special_action=7,
      compute_dtype="float32",
  )
  lc = dataclasses.replace(
      cfg.learner.learner, minibatch_size=64, sgd_steps=3, forward_chunk_rows=64
  )
  return dataclasses.replace(
      cfg,
      policy_kwargs={**dict(cfg.policy_kwargs), "network": net},
      learner=dataclasses.replace(cfg.learner, learner=lc),
      eval=emt.evals(every=1, deals=4),
      keep_every=1,
  )


def test_rollout_rewards_are_team_points_and_rows_fit():
  """Partners share a reward, opponents get its negative; the rows fit.

  The trained share (decisions + waiting rows) of a uniform-policy rollout
  fits under MAX_DECISION_SHARE.
  """
  cfg = _small(emt.get_config())
  parts = runner.one_slot_parts(cfg)
  learner = pg.PGLearner(parts.policy, cfg.learner.learner)
  s0 = learner.init(jax.random.key(0), parts.env)
  params = parts.slot_params(s0.params)
  n_envs, n_steps = 8, 200
  dk, pk = arena_lib.independent_keys(jax.random.key(3), n_envs, 4)
  state = parts.arena.reset(params, parts.table, dk, pk)
  _, batch, log = parts.arena.rollout(params, parts.table, state, n_steps)
  rewards = np.asarray(log.rewards, np.float32)  # [E, T, 4]
  np.testing.assert_array_equal(rewards[..., 0], rewards[..., 2])
  np.testing.assert_array_equal(rewards[..., 1], rewards[..., 3])
  np.testing.assert_array_equal(rewards[..., 0], -rewards[..., 1])
  assert np.abs(rewards).sum() > 0
  seat = np.asarray(batch.steps.seat)
  np.testing.assert_array_equal(
      seat.reshape(n_envs, 4, -1),
      np.broadcast_to(np.arange(4)[None, :, None], (n_envs, 4, n_steps)),
  )
  info = jax.tree_util.tree_map(
      lambda x: x.reshape((-1,) + x.shape[2:]), batch.steps.info
  )
  wait = np.asarray(jax.jit(jax.vmap(ecp.waiting_row))(info)).reshape(
      seat.shape
  )
  acted = np.asarray(batch.valid)
  assert not (wait & acted).any()
  share = (wait | acted).mean()
  assert 0.25 < share < emt.MAX_DECISION_SHARE, share
  print(
      f"trained share {share:.3f}: decisions {acted.mean():.3f}, waiting"
      f" {wait.mean():.3f}"
  )


def test_runs_end_to_end_on_cpu(tmp_path):
  cfg = _small(emt.get_config())
  res = train.run(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
  ls = res.league.learners["main"]
  assert int(ls.updates) == 2
  table, _, _, _ = res.league._draw(ls, np.int64(0))  # pylint: disable=protected-access
  assert np.asarray(table.seat_slot).shape == (8, 4)
  assert (np.asarray(table.seat_slot) == 0).all()


def test_network_options():
  """The defaults add the value-head init and key centring to the network.

  lecun and no centring give the catalog network unchanged.
  """
  base = catalog.get(emt.NETWORK)
  assert (
      emt.policy_kwargs(value_init="lecun", center_keys=False)["network"]
      == base
  )
  net = emt.policy_kwargs()["network"]
  assert {k: v for k, v in net["torso"].items() if k != "center_keys"} == base[
      "torso"
  ]
  assert {
      k: v for k, v in net["value_head"].items() if k != "init_scheme"
  } == base["value_head"]
  with pytest.raises(ValueError, match="value_init"):
    emt.policy_kwargs(value_init="orthogonal")


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider", "-s"]))
