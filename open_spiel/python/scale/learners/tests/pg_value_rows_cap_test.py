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

"""PGLearner's value_rows forward: only the value-only plies, as every ply.

A config with value_rows (here the hearts card move transformer recipe at CPU
size) trains the value head on plies where its seat does not act. The update
evaluates those plies in a host-sized bucket (pg.VALUE_ROWS_CAPS) instead of
every ply; the targets, values and update must equal the every-ply forward's.
"""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale.configs import (
    hearts_move_transformer_self_play as hmt,
)
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import card_move_transformer as cmt

# pylint: enable=g-import-not-at-top,wrong-import-position


def _small(cfg):
  """The config at CPU size: a small network, 8 x 40 plies, small batches.

  Args:
    cfg: the run config.

  Returns:
    The small config.
  """
  cfg = common.with_overrides(cfg, num_updates=2, num_envs=8, num_steps=40)
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
  )


def test_value_rows_forward_only_evaluates_value_rows():
  """PGLearner.value_rows_cap: the update evaluates only the value-only plies.

  It uses a ladder size >= their count, and its targets, values and update
  equal the every-ply forward's.
  """
  # The test compares the learner's private forward and update steps.
  # pylint: disable=protected-access
  cfg = _small(hmt.get_config())
  parts = runner.one_slot_parts(cfg)
  learner = pg.PGLearner(parts.policy, cfg.learner.learner)
  s0 = learner.init(jax.random.key(0), parts.env)
  params = parts.slot_params(s0.params)
  dk, pk = arena_lib.independent_keys(jax.random.key(3), 8, 4)
  state = parts.arena.reset(params, parts.table, dk, pk)
  _, batch, _ = parts.arena.rollout(params, parts.table, state, 60)
  vr = np.asarray(learner._value_rows_jit(batch.steps.info)) & ~np.asarray(
      batch.valid
  )
  k, n = int(vr.sum()), vr.size
  cap = learner.value_rows_cap(batch)
  assert 0 < k <= cap < n and cap in pg.VALUE_ROWS_CAPS, (k, cap, n)
  assert all(c < k for c in pg.VALUE_ROWS_CAPS if c < cap)
  full = learner._prepare(s0, batch)
  part = jax.jit(learner._prepare, static_argnums=2)(s0, batch, cap)
  for a, b in zip(
      jax.tree_util.tree_leaves(full), jax.tree_util.tree_leaves(part)
  ):
    np.testing.assert_allclose(
        np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6
    )
  # the value-row values themselves, at the value-only plies; 0 elsewhere
  vals = np.asarray(
      learner._value_row_values(s0.params, batch, jnp.asarray(vr), cap)
  )
  every = np.asarray(learner._value_row_values(s0.params, batch))
  np.testing.assert_allclose(vals[vr], every[vr], rtol=1e-5, atol=1e-6)
  assert (vals[~vr] == 0).all()
  capacity = learner.train_rows(batch)
  key = jax.random.key(7)
  s_full, m_full = jax.jit(
      learner._update, static_argnames=("capacity", "value_rows_cap")
  )(s0, batch, key, capacity=capacity, value_rows_cap=None)[:2]
  s_part, m_part = learner._update_jit(
      s0, batch, key, capacity=capacity, value_rows_cap=cap
  )[:2]
  for a, b in zip(
      jax.tree_util.tree_leaves(s_full.params),
      jax.tree_util.tree_leaves(s_part.params),
  ):
    np.testing.assert_allclose(
        np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6
    )
  for name in ("loss_total", "value_loss", "policy_loss"):
    np.testing.assert_allclose(
        np.asarray(m_full[name]), np.asarray(m_part[name]), rtol=1e-4, atol=1e-6
    )


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider", "-s"]))
