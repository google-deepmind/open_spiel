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

"""Tests for RoutedPolicy on gin.

The gin entrant is RoutedPolicy(base, scripted endgame, gin_endgame.in_domain),
used in Arena slots, the Evaluator and HostAgent. Known answers for the routing,
and the learner-side exclusion of routed rows (sources.decisions): a learner
never trains on a KNOCK, LAYOFF or WALL decision.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.core import advantages as adv_lib
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot
from open_spiel.python.scale.envs.tests import gin_cards
from open_spiel.python.scale.envs.tests import gin_helpers
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.sources import decisions

tree_map = jax.tree_util.tree_map
ROUTE = f"{gin_endgame.__name__}:in_domain"


def _step(infos):
  """A SeatStep of gin InfoStates.

  Args:
    infos: The InfoStates.

  Returns:
    The SeatStep.
  """
  info = gin_helpers.batch(*infos)
  b = len(infos)
  return types.SeatStep(
      info=info,
      legal_mask=jnp.ones((b, cm.NUM_ACTIONS), jnp.bool_),
      seat=info.seat.astype(jnp.int32),
      to_move=info.current_player.astype(jnp.int32),
      new_hand=jnp.zeros((b,), jnp.bool_),
      new_match=jnp.zeros((b,), jnp.bool_),
      last_action=jnp.full((b,), types.NO_ACTION, jnp.int32),
      last_actor=jnp.full((b,), types.NO_ACTOR, jnp.int32),
  )


def _routed_uniform():
  """A uniform gin policy routed to the scripted endgame.

  Returns:
    (policy, params).
  """
  base = uniform.UniformPolicy("gin_rummy", cm.NUM_ACTIONS, 2)
  pol = routed.RoutedPolicy(base, gin_endgame.GinScriptedEndgamePolicy(), ROUTE)
  return pol, pol.make_params({}, {})


def test_routing_known_answers():
  """Routing known answers.

  DISCARD -> the base (uniform: zero logits everywhere); KNOCK with 11 cards ->
  the script's min-deadwood discard 3h; the defender's LAYOFF -> Td; a KNOCK
  node where the OTHER seat is to move -> the base (not this seat's decision).
  """
  pol, params = _routed_uniform()
  knocker_melds = np.zeros((2, cm.NUM_MELDS), bool)
  knocker_melds[0] = gin_helpers.melds_mask("7d 8d 9d", "As 2s 3s")
  discard = gin_helpers.make_info(
      hand="As 2s 3s 4c 5c 6c 7d 8d 9d Kh Qh", phase=cm.PHASE_DISCARD
  )
  knock = gin_helpers.make_info(
      hand="As 2s 5s 2c 5c Ad 2d 5d Ah 2h 3h", phase=cm.PHASE_KNOCK, knocker=0
  )
  layoff = gin_helpers.make_info(
      seat=1,
      hand="Td 5s 9c Kh Qs 2h 4d Jc 6h 8s",
      phase=cm.PHASE_LAYOFF,
      knocker=0,
      layed_melds=knocker_melds,
  )
  others_knock = gin_helpers.make_info(
      hand="As 2s 5s 2c 5c Ad 2d 5d Ah 2h",
      phase=cm.PHASE_KNOCK,
      knocker=1,
      current_player=1,
  )
  infos = [discard, knock, layoff, others_knock]
  step = _step(infos)
  out = pol.evaluate(
      params, pol.init_carry(params, 4, step.seat, None), step, None
  )
  assert out.logits.dtype == jnp.float32 and out.value.dtype == jnp.float32
  lg = np.asarray(out.logits)
  np.testing.assert_array_equal(
      np.asarray(routed.override_rows(pol, step.info)),
      [False, True, True, False],
  )
  np.testing.assert_array_equal(lg[0], 0.0)
  np.testing.assert_array_equal(lg[3], 0.0)
  assert (
      int(np.argmax(lg[1])) == gin_cards.card("3h")
      and np.isneginf(np.delete(lg[1], gin_cards.card("3h"))).all()
  )
  assert int(np.argmax(lg[2])) == gin_cards.card("Td")


def test_override_rows_any_leading_shape():
  """Override rows any leading shape."""
  pol, _ = _routed_uniform()
  infos = [
      gin_helpers.make_info(phase=ph)
      for ph in (
          cm.PHASE_DISCARD,
          cm.PHASE_KNOCK,
          cm.PHASE_WALL,
          cm.PHASE_LAYOFF,
          cm.PHASE_DRAW,
          cm.PHASE_KNOCK,
      )
  ]
  flat = gin_helpers.batch(*infos)
  want = np.asarray(gin_endgame.in_domain(flat))
  np.testing.assert_array_equal(want, [False, True, True, True, False, True])
  bt = tree_map(lambda x: x.reshape((2, 3) + x.shape[1:]), flat)
  np.testing.assert_array_equal(
      np.asarray(routed.override_rows(pol, bt)), want.reshape(2, 3)
  )
  with pytest.raises(TypeError):
    routed.override_rows(
        uniform.UniformPolicy("gin_rummy", cm.NUM_ACTIONS, 2), flat
    )


@pytest.fixture(name="rollout", scope="module")
def _rollout_fixture():
  """Routed rule bot, self-play, every seat a perspective: 8 rows x 300 ply.

  Returns:
    (env, policy, batch, log).
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  pol = routed.RoutedPolicy(
      gin_rule_bot.GinRuleBotPolicy(),
      gin_endgame.GinScriptedEndgamePolicy(),
      ROUTE,
  )
  params = pol.make_params({}, {})
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.ARGMAX)])
  b, t = 8, 300
  table = arena_lib.self_play_table(b, 2, all_seats=True)
  dk, pk = arena_lib.independent_keys(jax.random.key(11), b, 2)
  _, tb, log = ar.collect(
      [tree_map(lambda x: x[None], params)], table, dk, pk, t
  )
  return env, pol, tb, log


def test_exclude_override_drops_exactly_the_routed_decisions(rollout):
  """Exclude override drops exactly the routed decisions."""
  _, pol, tb, _ = rollout
  out, n = decisions.exclude_override(pol, tb)
  valid, kept = np.asarray(tb.valid), np.asarray(out.valid)
  phase = np.asarray(tb.steps.info.phase)
  endgame = np.asarray(gin.is_endgame_phase(phase))
  # a valid row's seat is to move
  np.testing.assert_array_equal(kept, valid & ~endgame)
  assert isinstance(n, np.int64) and int(n) == int((valid & endgame).sum()) > 0
  assert kept.sum() > 0
  for f in ("reward", "done", "action", "behavior_logp", "value_pred"):
    np.testing.assert_array_equal(
        np.asarray(getattr(out, f)), np.asarray(getattr(tb, f))
    )


def test_terminal_reward_reaches_the_last_strategic_decision(rollout):
  """Terminal reward reaches the last strategic decision.

  Known answer: with gamma = lambda = 1 and zero values, every kept
  (strategic) decision's TD return is its seat's score of that hand, the
  routed endgame plies in between included in the reward sum.

  Args:
    rollout: the rollout fixture.
  """
  _, pol, tb, _ = rollout
  out, _ = decisions.exclude_override(pol, tb)
  b, t = out.valid.shape
  zeros = jnp.zeros((b, t), jnp.float32)
  g = adv_lib.gae(
      out.reward,
      zeros,
      out.valid,
      out.done,
      jnp.zeros((b,), jnp.float32),
      gamma=1.0,
      lam=1.0,
  )
  ret, valid = np.asarray(g.returns), np.asarray(out.valid)
  reward, done = np.asarray(out.reward), np.asarray(out.done)
  checked = 0
  for r in range(b):
    ends = np.flatnonzero(done[r])
    for i in np.flatnonzero(valid[r]):
      later = ends[ends >= i]
      if not later.size:
        continue  # hand cut by the segment end: bootstraps instead
      np.testing.assert_allclose(
          ret[r, i], reward[r, later[0]], rtol=0, atol=1e-4
      )
      checked += 1
  assert checked > 100


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
