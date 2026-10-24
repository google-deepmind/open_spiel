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

"""AlphaZero (learners.az): distillation over a replay of search targets.

* outcome_targets: the Monte Carlo outcome per seat on a hand-built batch
(rewards on opponent plies, a second unfinished hand masked out);
* PlyReplay: flat-shuffle epochs (every stored decision once per epoch
copy, num_epochs identical copies, empty slots invalid);
* the AZ learner (BCLearner, ce_target + mse): the loss of a zero network
is log(#legal) + 0.5 mean z^2, and it falls on a fixed batch;
* train(): a few AZ iterations on connect four end to end (CPU smoke).
"""

import copy
import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import train
from open_spiel.python.scale.configs import az_connect_four
from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.leduc import belief as leduc_belief
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.learners import az
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.networks import policy as network_policy
from open_spiel.python.scale.sources import replay as replay_lib
from open_spiel.python.scale.sources import targets

tree_map = jax.tree_util.tree_map


def _steps(b, t, a=3, p=2, seat=None, legal=None):
  """Toy SeatSteps [b, t].

  Args:
    b: The rows.
    t: The plies.
    a: The actions.
    p: Unused.
    seat: The seats.
    legal: The legal mask.

  Returns:
    The SeatStep.
  """
  del p  # Unused: the steps do not depend on the player count.
  seat = np.zeros((b, t), np.int32) if seat is None else seat
  legal = np.ones((b, t, a), bool) if legal is None else legal
  return types.SeatStep(
      info={"x": jnp.zeros((b, t, 4), jnp.float32)},
      legal_mask=jnp.asarray(legal),
      seat=jnp.asarray(seat),
      to_move=jnp.asarray(seat),
      new_hand=jnp.zeros((b, t), bool),
      new_match=jnp.zeros((b, t), bool),
      last_action=jnp.full((b, t), -1, jnp.int32),
      last_actor=jnp.full((b, t), -1, jnp.int32),
  )


def _batch(
    reward, done, valid, seat=None, a=3, p=2, target_policy=None, legal=None
):
  """A toy TrajectoryBatch.

  Args:
    reward: f32[b, t] rewards.
    done: Done flags.
    valid: Decision flags.
    seat: The seats.
    a: The actions.
    p: The players.
    target_policy: The target policies.
    legal: The legal mask.

  Returns:
    The TrajectoryBatch.
  """
  reward = np.asarray(reward, np.float32)
  b, t = reward.shape
  steps = _steps(b, t, a, p, seat, legal)
  tp = (
      np.full((b, t, a), 1.0 / a, np.float32)
      if target_policy is None
      else np.asarray(target_policy, np.float32)
  )
  return types.TrajectoryBatch(
      steps=steps,
      init_carry=(),
      action=jnp.zeros((b, t), jnp.int32),
      behavior_logits=jnp.zeros((b, t, a), jnp.float32),
      behavior_logp=jnp.zeros((b, t), jnp.float32),
      value_pred=jnp.zeros((b, t, p), jnp.float32),
      target_policy=jnp.asarray(tp),
      target_value=jnp.zeros((b, t, p), jnp.float32),
      reward=jnp.asarray(reward),
      done=jnp.asarray(np.asarray(done, bool)),
      valid=jnp.asarray(np.asarray(valid, bool)),
      labels={},
      aux={},
      bootstrap_step=tree_map(lambda x: x[:, 0], steps),
  )


def test_outcome_targets_known_answer():
  # Row 0 (seat 0): plies 0, 2 are seat 0's; ply 3 (the opponent's) ends the
  # hand with +1 for seat 0; ply 4 is the terminal ply; plies 5, 6 open a
  # second hand that does not finish inside the segment.
  # Row 1 (seat 1): seat 1 acts at 1 and 3; ply 3 ends the hand with -1.
  """Outcome targets known answer."""
  reward = [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0]]
  done = [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
  valid = [[1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0, 1]]
  seat = np.array([[0] * 7, [1] * 7], np.int32)
  out = az.outcome_targets(_batch(reward, done, valid, seat=seat))
  assert out.target_value.dtype == jnp.float32 and out.target_value.shape == (
      2,
      7,
      2,
  )
  # valid is unchanged (policy rows stay); the unfinished second hand's value
  # rows are masked through the one builder's label (BC semantics).
  np.testing.assert_array_equal(np.asarray(out.valid), np.asarray(valid, bool))
  ok = np.asarray(out.labels[targets.TARGET_VALUE_VALID])
  np.testing.assert_array_equal(
      ok, [[1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0]]
  )
  tv = np.asarray(out.target_value)
  np.testing.assert_array_equal(tv[0, [0, 2], 0], [1.0, 1.0])
  np.testing.assert_array_equal(tv[1, [1, 3], 1], [-1.0, -1.0])
  assert not tv[0, :, 1].any() and not tv[1, :, 0].any()  # seat layout
  assert not tv[~ok].any()


def test_outcome_targets_is_the_one_mc_builder():
  """Outcome targets is the one mc builder.

  AZ's targets are sources.targets.with_mc_value_targets(gamma=1), the one
  builder of outcome value targets.
  """
  rng = np.random.default_rng(3)
  b, t = 6, 12
  done = rng.random((b, t)) < 0.2
  valid = (rng.random((b, t)) < 0.6) & ~np.roll(done, 1, axis=1)
  reward = np.where(done, rng.normal(size=(b, t)), 0.0)
  seat = rng.integers(0, 2, size=(b, t)).astype(np.int32)
  batch = _batch(reward, done, valid, seat=seat)
  got = az.outcome_targets(batch)
  want = targets.with_mc_value_targets(batch, gamma=1.0)
  np.testing.assert_array_equal(
      np.asarray(got.target_value), np.asarray(want.target_value)
  )
  np.testing.assert_array_equal(
      np.asarray(got.labels[targets.TARGET_VALUE_VALID]),
      np.asarray(want.labels[targets.TARGET_VALUE_VALID]),
  )
  np.testing.assert_array_equal(np.asarray(got.valid), np.asarray(want.valid))


def test_outcome_targets_first_hand_only_drops_a_finished_second_hand():
  # Hand 1: plies 0-1 (ply 1 ends it, -1 for seat 0), terminal ply 2;
  # hand 2: plies 3-4 (ply 4 ends it, +1), terminal ply 5.
  """Outcome targets first hand only drops a finished second hand."""
  reward = [[0, -1, 0, 0, 1, 0]]
  done = [[0, 1, 0, 0, 1, 0]]
  valid = [[1, 0, 0, 1, 0, 0]]
  both = az.outcome_targets(_batch(reward, done, valid))
  np.testing.assert_array_equal(np.asarray(both.valid), [[1, 0, 0, 1, 0, 0]])
  np.testing.assert_array_equal(
      np.asarray(both.labels[targets.TARGET_VALUE_VALID]), [[1, 0, 0, 1, 0, 0]]
  )
  np.testing.assert_array_equal(
      np.asarray(both.target_value)[0, [0, 3], 0], [-1.0, 1.0]
  )
  first = az.outcome_targets(_batch(reward, done, valid), first_hand_only=True)
  np.testing.assert_array_equal(np.asarray(first.valid), [[1, 0, 0, 0, 0, 0]])
  np.testing.assert_array_equal(
      np.asarray(first.target_value)[0, :, 0], [-1, 0, 0, 0, 0, 0]
  )


def _ply_batch(rows, t, n_valid_per_row, a=3, seed=0):
  """A batch whose plies carry unique ids in their info.

  Args:
    rows: The rows.
    t: The plies.
    n_valid_per_row: Decision plies per row.
    a: The actions.
    seed: The seed.

  Returns:
    The TrajectoryBatch.
  """
  rng = np.random.default_rng(seed)
  valid = np.zeros((rows, t), bool)
  valid[:, :n_valid_per_row] = True
  tp = rng.dirichlet(np.ones(a), size=(rows, t)).astype(np.float32)
  b = _batch(np.zeros((rows, t)), np.zeros((rows, t)), valid, target_policy=tp)
  # tag every ply with a unique id in info so epochs can be audited
  ids = np.arange(rows * t, dtype=np.float32).reshape(rows, t)
  info = {
      "x": jnp.asarray(np.broadcast_to(ids[..., None], (rows, t, 4)).copy())
  }
  return b._replace(steps=b.steps._replace(info=info))


def test_ply_replay_epoch_holds_every_decision_once_per_copy():
  """Ply replay epoch holds every decision once per copy."""
  rows, t = 4, 5
  rep = az.PlyReplay(
      num_slots=2, samples_per_slot=12, minibatch_size=4, num_epochs=3
  )
  assert rep.num_minibatches == (2 * 12) // 4 == 6
  b = _ply_batch(rows, t, 3)  # 12 valid plies of 20
  rep.add(b)
  assert int(rep.write_count) == 1 and rep.write_count.dtype == np.int64
  ep = rep.epoch(jax.random.key(0))
  assert ep.valid.shape == (3 * 6, 4, 1)
  ids = np.asarray(ep.steps.info["x"][..., 0, 0])
  val = np.asarray(ep.valid[..., 0])
  per_copy = [
      ids[i * 6 : (i + 1) * 6][val[i * 6 : (i + 1) * 6]] for i in range(3)
  ]
  want = sorted(
      np.arange(rows * t).reshape(rows, t)[:, :3].reshape(-1).tolist()
  )
  for got in per_copy:
    assert (
        sorted(got.tolist()) == want
    )  # each stored decision exactly once per copy
  for i in range(1, 3):  # one permutation, tiled num_epochs times
    np.testing.assert_array_equal(ids[i * 6 : (i + 1) * 6], ids[:6])
  # The other (empty) slot only contributes invalid padding.
  assert val.sum() == 3 * 12


def test_ply_replay_overwrites_oldest_slot():
  """Ply replay overwrites oldest slot."""
  rep = az.PlyReplay(
      num_slots=2, samples_per_slot=12, minibatch_size=4, num_epochs=1
  )
  for seed in range(3):
    rep.add(_ply_batch(4, 5, 3, seed=seed))
  assert int(rep.write_count) == 3
  ep = rep.epoch(jax.random.key(1))
  tp = np.asarray(ep.target_policy[..., 0, :])[np.asarray(ep.valid[..., 0])]
  fresh = [
      np.asarray(_ply_batch(4, 5, 3, seed=s).target_policy)[:, :3].reshape(
          -1, 3
      )
      for s in (1, 2)
  ]
  stale = np.asarray(_ply_batch(4, 5, 3, seed=0).target_policy)[:, :3].reshape(
      -1, 3
  )

  def keyset(m):
    """The rows of `m`, rounded, as a set.

    Args:
      m: A matrix.

    Returns:
      The set.
    """
    return {tuple(np.round(r, 6)) for r in m}

  assert keyset(tp) == keyset(np.concatenate(fresh))
  assert not keyset(tp) & keyset(stale)


def test_ply_replay_refuses_a_carry():
  """Ply replay refuses a carry."""
  rep = az.PlyReplay(
      num_slots=1, samples_per_slot=4, minibatch_size=2, num_epochs=1
  )
  b = _ply_batch(2, 2, 1)._replace(
      init_carry={"h": jnp.zeros((2, 3), jnp.float32)}
  )
  with pytest.raises(ValueError, match="stateless"):
    rep.add(b)


def _nan_off_domain_batch(rows=4, t=5, n_valid_per_row=2, seed=0):
  """_ply_batch whose non-decision plies read NaN.

  _ply_batch whose decision plies read x > 0 (unique ids + 1) and whose
  other plies read NaN: an encoder that is NaN off its domain (a gin plane
  encoder's case).

  Args:
    rows: batch rows.
    t: plies per row.
    n_valid_per_row: decision plies per row.
    seed: the seed.

  Returns:
    The TrajectoryBatch.
  """
  b = _ply_batch(rows, t, n_valid_per_row, seed=seed)
  x = np.asarray(b.steps.info["x"]) + 1.0
  x[~np.asarray(b.valid)] = np.nan
  return b._replace(steps=b.steps._replace(info={"x": jnp.asarray(x)}))


def test_ply_replay_epoch_rows_are_nan_free_and_masked_with_empty_slots():
  """replay_buffer_size > 1.

  The epoch's padding (slots not yet added) and every non-decision ply read a
  stored decision ply's SeatStep, never zeros or the raw off-domain steps; they
  stay masked (valid False, their targets and labels untouched), the decision
  plies are exactly the stored ones, and the stored bytes are the adds' (the
  fill is the epoch's).
  """
  rows, t = 4, 5
  rep = az.PlyReplay(
      num_slots=3, samples_per_slot=8, minibatch_size=4, num_epochs=2
  )
  b = _nan_off_domain_batch(rows, t, 2)
  rep.add(b)
  raw = rep.state_bytes()
  ep = rep.epoch(jax.random.key(0))
  assert rep.state_bytes() == raw
  x = np.asarray(ep.steps.info["x"][..., 0, :])
  val = np.asarray(ep.valid[..., 0])
  assert np.isfinite(x).all()
  assert (x > 0).all()  # no all-zero (padding) rows
  stored = np.asarray(b.steps.info["x"])[np.asarray(b.valid)]

  def keyset(m):
    """The rows of `m` as a set.

    Args:
      m: A matrix.

    Returns:
      The set.
    """
    return {tuple(r) for r in m}

  assert keyset(x[val]) == keyset(stored)
  assert keyset(x[~val]) <= keyset(stored)  # a decision ply's SeatStep
  assert (~val).sum() > 0 and val.sum() == 2 * rows * 2
  for leaf in jax.tree_util.tree_leaves(
      ep.steps
  ):  # the whole SeatStep, bootstrap too
    assert np.all(np.isfinite(np.asarray(leaf, np.float64)))
  bs = np.asarray(ep.bootstrap_step.info["x"])
  np.testing.assert_array_equal(bs, x)
  # Masked rows keep their own targets: a stored non-decision ply's, or zeros
  # (padding).
  raw_tp = np.asarray(b.target_policy)[~np.asarray(b.valid)]
  allowed = keyset(raw_tp) | {(0.0, 0.0, 0.0)}
  assert keyset(np.asarray(ep.target_policy[..., 0, :])[~val]) <= allowed
  assert (
      (np.asarray(ep.target_policy[..., 0, :])[~val] == 0).all(-1).any()
  )  # padding present


def test_ply_replay_masked_nan_plies_train_finite_with_two_slots():
  """Ply replay masked nan plies train finite with two slots.

  The NaN-free masked plies end to end: an encoder that is NaN on zero rows and
  off its domain, PlyReplay(num_slots=2) with one add (an empty slot to pad):
  every BCLearner update has a finite loss and finite params.
  """
  pol = _net(
      encoder_fn=lambda info: info["x"] / jnp.sum(info["x"], -1, keepdims=True)
  )
  learner = bc.BCLearner(
      pol,
      bc.BCConfig(
          loss=losses.DistillLossConfig(value_coef=0.5),
          learning_rate=1e-2,
          weight_decay=0.0,
      ),
  )
  s = learner.init_from_params(pol.init_params(jax.random.key(0)))
  rep = az.PlyReplay(
      num_slots=2, samples_per_slot=8, minibatch_size=4, num_epochs=1
  )
  rep.add(_nan_off_domain_batch(4, 5, 2))
  ep = rep.epoch(jax.random.key(1))
  upd = jax.jit(learner.update)
  for i in range(ep.valid.shape[0]):
    s, m = upd(s, tree_map(lambda x, i=i: x[i], ep), jax.random.key(i))
    assert np.isfinite(float(m["loss_total"])), (i, m)
  for leaf in jax.tree_util.tree_leaves(learner.params(s)):
    assert np.isfinite(np.asarray(leaf)).all()


def test_ply_replay_keeps_a_nan_decision_ply_loud():
  """Ply replay keeps a nan decision ply loud.

  Only non-decision plies are filled: a decision ply that encodes to NaN (an
  encoder bug) still reaches the loss.
  """
  rep = az.PlyReplay(
      num_slots=2, samples_per_slot=8, minibatch_size=4, num_epochs=1
  )
  b = _nan_off_domain_batch(4, 5, 2)
  x = np.asarray(b.steps.info["x"]).copy()
  x[3, 1] = np.nan  # a decision ply, not the first (the fill's source)
  rep.add(b._replace(steps=b.steps._replace(info={"x": jnp.asarray(x)})))
  ep = rep.epoch(jax.random.key(0))
  xe = np.asarray(ep.steps.info["x"][..., 0, :])
  val = np.asarray(ep.valid[..., 0])
  assert np.isnan(xe[val]).any()
  assert np.isfinite(xe[~val]).all()


def test_ply_replay_epoch_without_a_decision_ply_raises():
  """Ply replay epoch without a decision ply raises."""
  rep = az.PlyReplay(
      num_slots=2, samples_per_slot=8, minibatch_size=4, num_epochs=1
  )
  b = _ply_batch(4, 5, 0)
  rep.add(b)
  with pytest.raises(ValueError, match="no decision ply"):
    rep.epoch(jax.random.key(0))


def _net(a=3, encoder_fn=lambda info: info["x"]):
  """A small MLP network policy on the toy info.

  Args:
    a: The actions.
    encoder_fn: The encoder.

  Returns:
    The policy.
  """
  return network_policy.NetworkPolicy(
      game="toy",
      obs_format="x",
      num_actions=a,
      num_players=2,
      encoder="unused",
      encoder_fn=encoder_fn,
      obs_shape=[4],
      network={
          "compute_dtype": "float32",
          "torso": {"kind": "mlp", "widths": [8], "layout": "shared"},
          "policy_head": {"kind": "dense"},
          "value_head": {"kind": "scalar", "tanh": True},
      },
      value_layout="zero_sum",
  )


def test_az_learner_known_answer_loss_of_a_zero_network():
  """AZ learner known answer loss of a zero network."""
  pol = _net()
  learner = bc.BCLearner(
      pol,
      bc.BCConfig(
          loss=losses.DistillLossConfig(
              policy="ce_target", value="mse", value_coef=0.5
          ),
          weight_decay=0.0,
      ),
  )
  s = learner.init_from_params(
      tree_map(jnp.zeros_like, pol.init_params(jax.random.key(0)))
  )
  legal = np.ones((2, 3, 3), bool)
  legal[0, 1, 2] = False  # 2 legal actions on one ply
  b = _batch(np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 3)), legal=legal)
  tp = np.asarray(b.target_policy).copy()
  tp[0, 1] = [0.5, 0.5, 0.0]
  z = np.array([[1.0, -1.0, 0.5], [0.0, 1.0, -0.5]], np.float32)
  tv = np.zeros((2, 3, 2), np.float32)
  tv[..., 0] = z  # seat 0 rows
  b = b._replace(target_policy=jnp.asarray(tp), target_value=jnp.asarray(tv))
  s2, m = learner.update(s, b, jax.random.key(1))
  n_legal = legal.sum(-1)
  want = np.mean(np.log(n_legal)) + 0.5 * np.mean(z**2)
  np.testing.assert_allclose(float(m["loss_total"]), want, rtol=1e-5)
  assert s2.step.dtype == jnp.int64 and int(s2.step) == 1


def test_az_learner_reduces_the_loss_on_a_fixed_batch():
  """AZ learner reduces the loss on a fixed batch."""
  pol = _net()
  learner = bc.BCLearner(
      pol,
      bc.BCConfig(
          loss=losses.DistillLossConfig(value_coef=0.5),
          learning_rate=1e-2,
          weight_decay=0.0,
          max_grad_norm=1e6,
      ),
  )
  s = learner.init_from_params(pol.init_params(jax.random.key(0)))
  b = _ply_batch(4, 5, 5, seed=3)
  b = b._replace(
      steps=b.steps._replace(
          info={
              "x": jax.random.normal(jax.random.key(2), (4, 5, 4), jnp.float32)
          }
      )
  )
  upd = jax.jit(learner.update)
  first = None
  for i in range(60):
    s, m = upd(s, b, jax.random.key(i))
    first = float(m["loss_total"]) if first is None else first
  assert float(m["loss_total"]) < 0.8 * first, (first, float(m["loss_total"]))
  blob = learner.state_dict(s)
  s_again = learner.load_state(blob)
  for x, y in zip(
      jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(s_again)
  ):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _tiny_c4_config():
  """A tiny Connect Four AZ config.

  Returns:
    The config.
  """
  cfg = az_connect_four.az_config()
  net = copy.deepcopy(dict(cfg.network))
  net["torso"]["widths"] = [16]
  net["value_head"]["hidden"] = 16
  return dataclasses.replace(
      cfg,
      num_simulations=4,
      batch_size=4,
      num_iterations=2,
      num_epochs=2,
      minibatch_size=64,
      replay_buffer_size=2,
      network=net,
  )


def _run(cfg):
  """AZ end to end: a one-learner-slot league (league.kinds.AZ).

  Args:
    cfg: The AZ config.

  Returns:
    The run's output.
  """
  return train.run(recipes.alphazero("az_test", cfg))


def test_az_connect_four_end_to_end_smoke():
  """AZ connect four end to end smoke."""
  cfg = _tiny_c4_config()
  out = _run(cfg)
  hist = out.metrics
  assert len(hist) == 2
  for m in hist:
    for k in (
        "loss_total",
        "loss_policy",
        "loss_value",
        "avg_game_length",
        "frames",
    ):
      assert np.isfinite(m[k]), (k, m)
    assert 7 <= m["avg_game_length"] <= 42
  ls = out.league.learners["main"]
  assert isinstance(ls.frames, np.int64) and ls.frames == 2 * 4 * 43
  assert isinstance(ls.decisions, np.int64)
  assert hist[-1]["frames"] == 2 * 4 * 43 and isinstance(
      hist[-1]["frames"], int
  )
  # the source counts every decision it emitted; AZ keeps each row's first game
  assert (
      ls.decisions
      >= int(round(sum(m["avg_game_length"] for m in hist) * 4))
      > 0
  )
  assert int(out.state.step) == 2 * cfg.num_epochs * az.num_minibatches(cfg)
  p0 = out.learner.policy().init_params(
      jax.random.split(jax.random.key(cfg.seed))[0]
  )
  moved = [
      not np.array_equal(np.asarray(a), np.asarray(b))
      for a, b in zip(
          jax.tree_util.tree_leaves(p0),
          jax.tree_util.tree_leaves(out.state.params),
      )
  ]
  assert any(moved)


def test_az_build_uses_the_one_learner_and_replay():
  """AZ build uses the one learner and replay.

  AZ composes BCLearner (AdamW with a cosine step size) and the FIFO
  ReplayBuffer; it defines no learner or replay storage of its own.
  """
  cfg = _tiny_c4_config()
  _, _, _, _, _, learner, replay = az.build(cfg)
  assert isinstance(learner, bc.BCLearner)
  assert learner.cfg.lr_schedule == "cosine"
  assert learner.cfg.loss.policy == "ce_target"
  assert learner.cfg.loss.value == "mse"
  want_steps = cfg.num_iterations * cfg.num_epochs * az.num_minibatches(cfg)
  assert learner.cfg.total_steps == want_steps
  # Reaches private _buffer of the code under test.
  buf = replay._buffer  # pylint: disable=protected-access
  assert isinstance(buf, replay_lib.ReplayBuffer)
  assert buf.capacity_iterations == cfg.replay_buffer_size
  assert not hasattr(az, "DistillLearner")


def test_az_sampler_is_configurable_determinized_leduc():
  """AZ sampler is configurable determinized leduc.

  AZConfig.sampler picks the SearchPolicy's belief sampler: 'leduc' builds a
  PIMC search over LeducBeliefSampler reading the base params, and one AZ
  iteration runs on leduc (chance mid-hand).
  """
  cfg = az.AZConfig(
      env_name="leduc_poker",
      encoder="leduc_poker_info_state",
      obs_shape=(30,),
      network={
          "compute_dtype": "float32",
          "torso": {"kind": "mlp", "widths": [8], "layout": "shared"},
          "policy_head": {"kind": "dense"},
          "value_head": {"kind": "scalar", "tanh": False},
      },
      sampler="leduc",
      num_worlds=2,
      num_simulations=4,
      batch_size=4,
      num_iterations=1,
      num_epochs=1,
      minibatch_size=16,
      replay_buffer_size=1,
  )
  env, _, sp, _, _, _, _ = az.build(cfg)
  assert isinstance(sp.sampler, leduc_belief.LeducBeliefSampler)
  assert sp.cfg.num_worlds == 2 and sp.shares_base_params and sp.stochastic
  out = _run(cfg)
  assert np.isfinite(out.metrics[0]["loss_total"])
  assert out.league.learners["main"].frames == 4 * (env.max_len + 1)
  with pytest.raises(AssertionError, match="sampler"):
    dataclasses.replace(cfg, sampler="nope")


def test_az_policy_kwargs_pass_through_to_the_base():
  """AZ policy kwargs pass through to the base.

  The base network can be named by a league run's policy_arch / policy_kwargs
  (the registry), not only by the AZConfig; the AZ runner passes them through,
  and a base of another game is refused.
  """
  cfg = _tiny_c4_config()
  env = env_registry.make("connect_four")
  kw = az.base_policy_kwargs(cfg, env)
  default_base = az.build(cfg)[1]
  assert dict(default_base.spec.arch_kwargs) == dict(
      network_policy.NetworkPolicy(**kw).spec.arch_kwargs
  )
  net = dict(kw["network"])
  net["torso"] = dict(net["torso"], widths=[3])
  custom = dict(kw, network=net)
  got_env, base, sp, _, _, learner, _ = az.build(
      cfg, env=env, policy_kwargs=custom
  )
  assert got_env is env and learner.policy() is base and sp.base is base
  assert list(base.spec.arch_kwargs["network"]["torso"]["widths"]) == [3]
  r = kinds.make_driver(kinds.AZ(cfg), env, custom)
  assert r.policy.spec == base.spec
  with pytest.raises(ValueError, match="not the env"):
    az.build(
        cfg,
        env=env,
        policy_kwargs=dict(custom, num_actions=env.num_actions + 1),
    )
  with pytest.raises(ValueError, match="env_name"):
    az.build(cfg, env=env_registry.make("tic_tac_toe"))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
