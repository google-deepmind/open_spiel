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

"""DaggerSource options (sources/dagger.py).

DaggerSource options (sources/dagger.py): opening plies, first-hand
episodes, the routed scripted endgame, the decisions counter.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.connect_four import teacher as c4_teacher
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot
from open_spiel.python.scale.league import sampler as league_sampler
from open_spiel.python.scale.policies import opening
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources import targets

ROUTE = f"{gin_endgame.__name__}:in_domain"


def _c4_source(**kw):
  """A Connect Four dagger source with a prefer-centre teacher.

  Args:
    **kw: DaggerSource overrides.

  Returns:
    (env, source).
  """
  env = connect_four.ConnectFourEnv()
  t = c4_teacher.PreferCenterPolicy()
  args = dict(
      student_params={},
      teacher_params={},
      table=arena_lib.self_play_table(64, 2, all_seats=True),
      num_steps=50,
      beta=schedules.constant(1.0),
      rule=types.ARGMAX,
      return_lookahead=False,
  )
  args.update(kw)
  return env, dagger.DaggerSource(env, t, t, **args)


def test_first_hand_only_restarts_and_masks_later_hands():
  """First hand only restarts and masks later hands."""
  _, src = _c4_source(first_hand_only=True, opening_plies=6)
  for i in range(2):
    b = src.next(jax.random.key(i))
    nh = np.asarray(b.steps.new_hand)
    assert nh[:, 0].all(), "every segment starts from a fresh deal"
    hand = np.cumsum(nh, axis=1)
    assert (hand > 1).any(), "T=50 > 43 plies: some rows reach a second hand"
    v = np.asarray(b.valid)
    assert not v[hand > 1].any()
    # Every decision of the row's seat in the first hand is kept, nothing else.
    seat_moves = np.asarray(b.steps.to_move) == np.asarray(b.steps.seat)
    np.testing.assert_array_equal(v, seat_moves & (hand == 1))
    ok = np.asarray(b.labels[targets.TARGET_VALUE_VALID])
    assert (
        ok <= v
    ).all() and ok.sum() == v.sum(), "every first hand finishes within 50 plies"
  assert isinstance(src.decisions, np.int64)


def test_first_hand_only_refuses_lookahead():
  """First hand only refuses lookahead."""
  with pytest.raises(ValueError, match="look ahead"):
    _c4_source(first_hand_only=True, return_lookahead=True)


def test_decisions_counts_the_masked_valid_rows():
  """Decisions counts the masked valid rows."""
  _, src = _c4_source(first_hand_only=True)
  n = 0
  for i in range(3):
    n += int(np.asarray(src.next(jax.random.key(i)).valid).sum())
  assert int(src.decisions) == n and src.decisions.dtype == np.int64


def test_opening_plies_are_random_and_labelled_by_the_teacher():
  """Opening plies are random and labelled by the teacher."""
  _, src = _c4_source(first_hand_only=True, opening_plies=4)
  assert isinstance(src.slot_policy, opening.OpeningPolicy)
  b = src.next(jax.random.key(7))
  v = np.asarray(b.valid)
  ply = np.cumsum(np.ones_like(v), axis=1) - 1  # first hand starts at t = 0
  tp = np.asarray(b.target_policy)
  teach = np.asarray(jax.vmap(c4_teacher.actions)(b.steps.info))
  act = np.asarray(b.action)
  # Labels: the teacher's one-hot everywhere, the random plies included.
  np.testing.assert_array_equal(tp[v].argmax(-1), teach[v])
  np.testing.assert_allclose(tp[v].max(-1), 1.0)
  early = v & (ply < 4)
  late = v & (ply >= 4)
  assert (
      np.mean(act[early] != teach[early]) > 0.5
  ), "the opening plays at random"
  np.testing.assert_array_equal(act[late], teach[late])


@pytest.fixture(name="gin_batch", scope="module")
def _gin_batch_fixture():
  """A gin dagger source with a routed opponent, and its first batch.

  Returns:
    (source, batch).
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  bot = gin_rule_bot.GinRuleBotPolicy()
  end = gin_endgame.GinScriptedEndgamePolicy()
  opp = routed.RoutedPolicy(bot, end, ROUTE)
  table = league_sampler.entries_table(32, 2, [(1, 0, 1.0)], jax.random.key(0))
  src = dagger.DaggerSource(
      env,
      bot,
      bot,
      student_params={},
      teacher_params={},
      table=table,
      num_steps=120,
      beta=schedules.constant(0.5),
      rule=types.SAMPLE,
      opponents=[arena_lib.Slot(opp, 1, types.ARGMAX)],
      opponent_params=[
          jax.tree.map(lambda x: x[None], opp.make_params({}, {}))
      ],
      return_lookahead=False,
      endgame=dagger.EndgameRoute(end, {}, ROUTE),
  )
  return src, src.next(jax.random.key(0))


def test_endgame_rows_are_not_learner_decisions(gin_batch):
  """Endgame rows are not learner decisions."""
  src, b = gin_batch
  assert isinstance(src.slot_policy, routed.RoutedPolicy)
  info = b.steps.info
  seat_moves = np.asarray(b.steps.to_move) == np.asarray(b.steps.seat)
  end = np.asarray(gin.is_endgame_phase(info.phase))
  v = np.asarray(b.valid)
  assert (
      seat_moves & end
  ).sum() > 0, "the learner seat reaches KNOCK / LAYOFF / WALL"
  assert not v[end].any(), "routed endgame rows are not learner decisions"
  np.testing.assert_array_equal(v, seat_moves & ~end)
  assert int(src.decisions) == int(v.sum())


def test_endgame_returns_credit_the_last_strategic_decision(gin_batch):
  """Endgame returns credit the last strategic decision."""
  _, b = gin_batch
  ok = np.asarray(b.labels[targets.TARGET_VALUE_VALID])
  assert ok.sum() > 0
  assert (ok <= np.asarray(b.valid)).all()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
