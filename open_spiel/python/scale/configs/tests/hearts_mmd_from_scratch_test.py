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

"""configs.hearts_mmd_from_scratch: the recipe, rules and from-scratch tests.

Also the moon-shot telemetry every hearts run logs (envs.hearts_telemetry.moon):
the moon read from a hand's final points, the per-update rates on a
constructed batch, that only hearts names it (train.TRAINING_TELEMETRY), and
that observing a run returns the rollout's own batch (one rollout, counted
once) and adds the metrics to the update's train row.
"""

import collections
import dataclasses
import sys
import types as types_lib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps
from open_spiel.python.scale import train

# The hearts config imports the optional JAX games package.
pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.jax.games import hearts as hearts_game
from open_spiel.python.scale.configs import hearts_mmd_from_scratch as hms
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import hearts
from open_spiel.python.scale.envs.hearts_telemetry import moon
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_recipe():
  """The registered config is the module's recipe."""
  cfg = common.load_config("hearts_mmd_from_scratch")
  assert cfg == hms.get_config()
  assert cfg.env == "hearts" and dict(cfg.env_kwargs) == hearts.OPTIONS
  assert cfg.controller == "scheduled_promotion"
  assert cfg.run_updates() == hms.NUM_UPDATES
  assert cfg.league.num_envs == hms.NUM_ENVS
  assert cfg.league.num_steps == hms.NUM_STEPS
  lc = cfg.learner.learner
  assert lc.loss == "mmd" and lc.num_updates == hms.NUM_UPDATES
  assert lc.value_scale == hms.VALUE_SCALE and lc.gamma == 1.0
  assert [(e.name, e.every, e.deals) for e in cfg.eval] == [
      ("vs_uniform", hms.UNIFORM_EVERY, hms.UNIFORM_DEALS)
  ]
  assert cfg.metadata["rules_sha"] == hearts.rules_sha(hearts.OPTIONS)
  assert cfg.metadata["hearts_rules"] == "standard"
  env = common.build_env(cfg)
  assert sides_lib.sides_of(env) == sides_lib.Sides.individual(
      4, constant_sum=False
  )


def test_run_length_override_moves_the_lr_anneal():
  """--num_updates sets the run length and the LR anneal's horizon."""
  cfg = common.with_overrides(
      common.load_config("hearts_mmd_from_scratch"), num_updates=1234
  )
  assert cfg.run_updates() == 1234
  assert cfg.learner.learner.num_updates == 1234


def test_rules_check_refuses_other_options():
  """A non-default rule option fails check_standard_rules."""
  cfg = hms.get_config()
  other = dataclasses.replace(cfg, env_kwargs={**hms.RULES, "jd_bonus": True})
  with pytest.raises(AssertionError):
    hms.check_standard_rules(other)


def test_from_scratch_check_refuses_a_learner_without_a_magnet():
  """A PPO learner (no magnet re-taken from its own params) is refused."""
  cfg = hms.get_config()
  lc = dataclasses.replace(
      cfg.learner.learner, loss="ppo", kl_coef=hms.schedules.constant(0.0)
  )
  ppo = dataclasses.replace(
      cfg, learner=dataclasses.replace(cfg.learner, learner=lc)
  )
  assert isinstance(ppo.learner, kinds.PG)
  with pytest.raises(AssertionError):
    hms.check_from_scratch(ppo)


def _final_points(winners, trick_points):
  """The engine's final points of a hand (pyspiel ComputeScore).

  Args:
    winners: the seat that took each of the 13 tricks.
    trick_points: the points of each trick.

  Returns:
    int32[4].
  """
  raw = np.zeros(4, np.int32)
  for w, p in zip(winners, trick_points):
    raw[w] += p
  # Reaches the engine's private scoring step: the hand is constructed.
  score = hearts_game.Hearts()._final_points  # pylint: disable=protected-access
  return np.asarray(
      score(
          jnp.asarray(raw),
          jnp.asarray(winners, jnp.int8),
          jnp.asarray(trick_points, jnp.int32),
      )
  )


#: 26 points over 13 tricks: nine single hearts, the queen with a heart, two
#: clean tricks.
_TRICK_POINTS = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 14, 3]


def test_moon_outcome_on_a_hand_where_one_seat_takes_all_26():
  """A seat that takes every point shoots the moon; 20+ without is near."""
  assert sum(_TRICK_POINTS) == 26
  pts = _final_points([2] * 13, _TRICK_POINTS)
  np.testing.assert_array_equal(pts, [26, 26, 0, 26])
  shooter, near = moon.moon_outcome(pts)
  assert int(shooter) == 2 and not np.asarray(near).any()
  # Seat 1 takes every trick but one single heart: 25 points, no moon, near.
  pts = _final_points([1, 3] + [1] * 11, _TRICK_POINTS)
  np.testing.assert_array_equal(pts, [0, 25, 0, 1])
  assert pts.sum() == 26 and pts[1] >= moon.NEAR_MOON_POINTS
  shooter, near = moon.moon_outcome(pts)
  assert int(shooter) == -1
  np.testing.assert_array_equal(near, [False, True, False, False])
  # Batched, and read from returns (26 - returns are the points).
  returns = np.array([[0, 26, 0, 0], [20, 19, 20, 19]])
  shooter, _ = moon.moon_outcome(26 - returns)
  np.testing.assert_array_equal(shooter, [1, -1])


def _batch(seat, to_move, points) -> types.TrajectoryBatch:
  """A TrajectoryBatch holding only what the moon metrics read.

  Args:
    seat: int[B, T] perspective seats.
    to_move: int[B, T], -1 on a terminal ply.
    points: int[B, T, 4] the InfoState's points.

  Returns:
    The batch (other fields None; valid where to_move is the row's seat).
  """
  steps = types.SeatStep(
      info={"points": jnp.asarray(points, jnp.int16)},
      legal_mask=None,
      seat=jnp.asarray(seat, jnp.int32),
      to_move=jnp.asarray(to_move, jnp.int32),
      new_hand=None,
      new_match=None,
      last_action=None,
      last_actor=None,
  )
  return types.TrajectoryBatch(
      steps=steps,
      init_carry=None,
      action=None,
      behavior_logits=None,
      behavior_logp=None,
      value_pred=None,
      target_policy=None,
      target_value=None,
      reward=None,
      done=None,
      valid=jnp.asarray(seat) == jnp.asarray(to_move),
      labels={},
      aux={},
      bootstrap_step=None,
  )


def test_moon_batch_metrics_split_learner_and_opponent_seats():
  """Rates per seat-hand over terminal plies: learner vs the other seats."""
  moon_by_0 = [0, 26, 26, 26]
  moon_by_3 = [26, 26, 26, 0]
  near_1 = [3, 20, 3, 0]
  plain = [0, 13, 0, 13]
  seat = [[0] * 3, [1] * 3, [2] * 3, [3] * 3]
  # Row 3's first ply is mid-hand: its points are not a final score and do
  # not count (only terminal plies, to_move -1, do).
  to_move = [[1, 2, -1], [0, 3, -1], [-1, 0, 1], [1, -1, 0]]
  zero = [0, 0, 0, 0]
  points = [
      [zero, zero, moon_by_0],
      [zero, zero, moon_by_3],
      [near_1, zero, zero],
      [moon_by_0, plain, zero],
  ]
  m = jax.jit(moon.batch_metrics)(_batch(seat, to_move, points))
  m = {k: np.asarray(v) for k, v in m.items()}
  assert m["rollout/hands"] == 4
  assert m["rollout/moons/learner"] == 1  # row 0: seat 0 shot it
  assert m["rollout/moons/opponents"] == 1  # row 1: seat 3 did, not seat 1
  assert m["rollout/near_moons/learner"] == 0
  assert m["rollout/near_moons/opponents"] == 1  # row 2: seat 1 of seat 2
  assert m["rollout/moons/learner"].dtype == np.int64
  np.testing.assert_allclose(m["rollout/moon_rate/learner"], 1 / 4)
  np.testing.assert_allclose(m["rollout/moon_rate/opponents"], 1 / 12)
  np.testing.assert_allclose(m["rollout/near_moon_rate/learner"], 0.0)
  np.testing.assert_allclose(m["rollout/near_moon_rate/opponents"], 1 / 12)
  assert m["rollout/moon_rate/learner"].dtype == np.float32
  # No hand ended: counts 0, rates 0 (never a NaN the writer refuses).
  none = moon.batch_metrics(_batch(seat, [[0, 1, 2]] * 4, points))
  assert all(float(v) == 0.0 for v in none.values())


def test_only_hearts_names_training_telemetry():
  """train.run observes hearts runs only, through the training-only module."""
  assert train.TRAINING_TELEMETRY == {"hearts": "envs.hearts_telemetry.moon"}
  assert moon.__name__.endswith(".envs.hearts_telemetry.moon")
  # Not a hearts run: refused; not a PG run: off.
  # observing reads cfg.env and cfg.learner only.
  pg_learner = hms.get_config().learner
  with pytest.raises(ValueError, match="euchre"):
    with moon.observing(
        types_lib.SimpleNamespace(env="euchre", learner=pg_learner)
    ):
      pass
  not_pg = types_lib.SimpleNamespace(env="hearts", learner=object())
  with moon.observing(not_pg):
    # Reaches the active-observer list: no observer for a non-PG run.
    assert not moon._ACTIVE  # pylint: disable=protected-access


def _observers(cfg):
  """The moon observers among the run hooks of cfg.

  Args:
    cfg: the run config.

  Returns:
    The hooks moon.observing installed.
  """
  # Reaches the module's active-observer list to pick its hook out.
  # pylint: disable-next=protected-access
  return [h for h in runner.run_hooks(cfg) if h in moon._ACTIVE]


_ArenaState = collections.namedtuple("_ArenaState", ["policy_key"])


class _StubArena:
  """The arena surface ArenaSource.next uses; rollout returns `batch`."""

  def __init__(self, batch):
    self.batch = batch
    self.rollouts = 0
    self.env = hearts.HeartsEnv()

  def reset(self, params, table, dk, pk):
    """A stub arena state.

    Args:
      params: unused.
      table: unused.
      dk: unused.
      pk: the player keys.

    Returns:
      The state.
    """
    del params, table, dk
    return _ArenaState(pk)

  def rollout(self, params, table, state, num_steps):
    """The fixed batch, counting the rollouts.

    Args:
      params: unused.
      table: unused.
      state: the arena state.
      num_steps: unused.

    Returns:
      (state, the batch, None).
    """
    del params, table, num_steps
    self.rollouts += 1
    return state, self.batch, None


class _StubSource:
  """An ArenaSource's fields, so the real ArenaSource.next runs on it."""

  next = arena_lib.ArenaSource.next  # the real method, unchanged

  def __init__(self, batch):
    self.arena = _StubArena(batch)
    self.params, self.table, self.state = None, None, None
    self.batch, self.num_steps = 4, 3
    self.frames, self.decisions = np.int64(0), np.int64(0)


def test_observing_returns_the_rollouts_batch_and_logs_it():
  """The observed rollout runs once, unchanged; its metrics join the row."""
  moon_by_0 = [0, 26, 26, 26]
  zero = [0, 0, 0, 0]
  seat = [[0] * 3, [1] * 3, [2] * 3, [3] * 3]
  to_move = [[1, 0, -1], [0, 3, -1], [1, 2, 3], [1, 2, 0]]
  points = [
      [zero, zero, moon_by_0],
      [zero, zero, moon_by_0],
      [zero] * 3,
      [zero] * 3,
  ]
  batch = _batch(seat, to_move, points)
  cfg = hms.get_config()
  src = _StubSource(batch)
  assert not _observers(cfg)  # nothing observed outside the block
  with moon.observing(cfg):
    (hook,) = _observers(cfg)
    for n in (1, 2):
      out = src.next(jax.random.key(n))
      assert out is batch  # the rollout's own batch, not a copy
      assert src.arena.rollouts == n and int(src.frames) == 12 * n
      assert int(src.decisions) == 2 * n
      row = {"loss_total": 0.5}
      hook.after_update(None, None, n, dict(row), row)
      assert row["rollout/hands"] == 2.0
      assert row["rollout/moons/learner"] == 1.0  # row 0: seat 0 shot it
      assert row["rollout/moons/opponents"] == 1.0  # row 1: seat 0, not 1
      assert row["rollout/moon_rate/opponents"] == pytest.approx(1 / 6)
      assert row["loss_total"] == 0.5
    with pytest.raises(RuntimeError, match="no observed rollout"):
      hook.after_update(None, None, 3, {}, {})
  assert not _observers(cfg)
  # Outside the block the rollout is not observed.
  assert src.next(jax.random.key(9)) is batch


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
