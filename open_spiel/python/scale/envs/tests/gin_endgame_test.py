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

"""The scripted gin endgame: contract and known answers.

Read from the InfoState only, and the knock_discard hook.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.tests import gin_cards
from open_spiel.python.scale.envs.tests import gin_games
from open_spiel.python.scale.envs.tests import gin_helpers
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.testing import check_policy

tree_map = jax.tree_util.tree_map
P0 = "As 2s 3s 4c 5c 6c 7d 8d 9d Kh"
P1 = "6s 8s Ts Qs 2c 8c Tc Kc 3d 5d"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A gin env with Oklahoma rules.

  Returns:
    The env.
  """
  return gin.GinEnv(oklahoma_frac=1.0)


def _act(*infos):
  """The scripted endgame actions of `infos`.

  Args:
    *infos: The GinInfos.

  Returns:
    The actions, as a list.
  """
  batch = gin_helpers.batch(*infos)
  return np.asarray(jax.jit(gin_endgame.actions)(batch)).tolist()


def test_spec_and_registry():
  """Spec and registry."""
  pol = registry.build(gin_endgame.ARCH, {})
  assert isinstance(pol, gin_endgame.GinScriptedEndgamePolicy)
  assert pol.spec.capability == types.Capability.NATIVE_JAX
  assert (
      pol.spec.arch_name == "gin_rummy.scripted_endgame"
      and pol.decider == "scripted"
  )


def test_knock_discard_leaves_the_least_deadwood():
  """Knock discard leaves the least deadwood.

  With As 2s 5s 2c 5c Ad 2d 5d Ah 2h 3h after a knock, discarding 3h lays
  gin (sets of aces, twos and fives); the rule bot's discard Ad keeps
  1 deadwood.
  """
  i = gin_helpers.make_info(
      hand="As 2s 5s 2c 5c Ad 2d 5d Ah 2h 3h", phase=cm.PHASE_KNOCK, knocker=0
  )
  assert _act(i) == [gin_cards.card("3h")]


def test_knock_discard_hook_is_overridable():
  """Knock discard hook is overridable."""

  class AlwaysFirstLegal(gin_endgame.GinScriptedEndgamePolicy):
    """The scripted endgame with the first legal knock discard."""

    ARCH = "test.first_legal_endgame"

    @staticmethod
    def knock_discard(info, bot_action):
      """The first legal knock discard of each row.

      Args:
        info: A batched GinInfo.
        bot_action: Unused.

      Returns:
        i32[B] discards.
      """
      del bot_action  # Unused.
      legal = jax.vmap(lambda i: gin_endgame.knock_cache(i)[0])(info)
      return jnp.argmax(legal, axis=-1).astype(jnp.int32)

  i = gin_helpers.make_info(
      hand="As 2s 5s 2c 5c Ad 2d 5d Ah 2h 3h", phase=cm.PHASE_KNOCK, knocker=0
  )
  b = gin_helpers.batch(i)
  got = np.asarray(
      gin_endgame.actions(b, AlwaysFirstLegal.knock_discard)
  ).tolist()
  assert got == [gin_cards.card("As")]


def test_known_answers_knocker_lays_highest_index_meld_then_passes():
  """Known answers knocker lays highest index meld then passes."""
  hand = "As 2s 3s 4c 5c 6c 7d 8d 9d Td"
  ids = sorted(
      gin_cards.meld_id(m) for m in ("As 2s 3s", "4c 5c 6c", "7d 8d 9d Td")
  )
  i = gin_helpers.make_info(hand=hand, phase=cm.PHASE_KNOCK, knocker=0)
  assert _act(i) == [cm.ACTION_MELD_BASE + ids[-1]]
  done = gin_helpers.make_info(hand="", phase=cm.PHASE_KNOCK, knocker=0)
  assert _act(done) == [cm.ACTION_PASS]


def test_known_answers_layoffs():
  """Known answers layoffs.

  The defender lays Td off onto the knocker's 7d8d9d (deadwood -10), then
  passes when nothing more can be laid off; against gin it never lays off.
  """
  knocker_melds = np.zeros((2, cm.NUM_MELDS), bool)
  knocker_melds[0] = gin_helpers.melds_mask("7d 8d 9d", "As 2s 3s")
  hand = "Td 5s 9c Kh Qs 2h 4d Jc 6h 8s"
  lay = gin_helpers.make_info(
      seat=1,
      hand=hand,
      phase=cm.PHASE_LAYOFF,
      knocker=0,
      layed_melds=knocker_melds,
  )
  after = gin_helpers.make_info(
      seat=1,
      hand="5s 9c Kh Qs 2h 4d Jc 6h 8s",
      phase=cm.PHASE_LAYOFF,
      knocker=0,
      layed_melds=knocker_melds,
      layoffs_mask=gin_cards.hand_mask("Td").astype(bool),
  )
  vs_gin = lay._replace(knocker_gin=jnp.bool_(True))
  assert _act(lay, after, vs_gin) == [
      gin_cards.card("Td"),
      cm.ACTION_PASS,
      cm.ACTION_PASS,
  ]


def test_known_answers_layoff_tie_breaks():
  """Known answers layoff tie breaks.

  Equal best reachable deadwood: the card leaving less deadwood right away
  wins (Td over 6d onto 7d8d9d); then, at equal value, the higher index (Th
  over Td onto 7d8d9d and 7h8h9h).
  """
  others = "2s 4s Qs 3c 5c Kc Ks Jc"
  m1 = np.zeros((2, cm.NUM_MELDS), bool)
  m1[0] = gin_helpers.melds_mask("7d 8d 9d")
  m2 = np.zeros((2, cm.NUM_MELDS), bool)
  m2[0] = gin_helpers.melds_mask("7d 8d 9d", "7h 8h 9h")
  first = gin_helpers.make_info(
      seat=1,
      hand="6d Td " + others,
      phase=cm.PHASE_LAYOFF,
      knocker=0,
      layed_melds=m1,
  )
  index = gin_helpers.make_info(
      seat=1,
      hand="Td Th " + others,
      phase=cm.PHASE_LAYOFF,
      knocker=0,
      layed_melds=m2,
  )
  assert _act(first, index) == [gin_cards.card("Td"), gin_cards.card("Th")]


def test_known_answer_defender_melds_after_layoffs():
  """Known answer defender melds after layoffs."""
  hand = "Jh Qh Kh 2c 3d 9s"
  i = gin_helpers.make_info(
      seat=1, hand=hand, phase=cm.PHASE_LAYOFF, knocker=0, finished_layoffs=True
  )
  assert _act(i) == [cm.ACTION_MELD_BASE + gin_cards.meld_id("Jh Qh Kh")]


def test_known_answer_wall_knocks_iff_legal():
  """Known answer wall knocks iff legal."""
  # Hand + upcard 7d: 7d8d9d melds, Jd Kh left, discard Kh: 10 <= 10, knock.
  # With Qs instead: 37 deadwood after the best discard, pass.
  w = gin_helpers.make_info(
      hand="As 2s 3s 4c 5c 6c 8d 9d Jd Kh",
      phase=cm.PHASE_WALL,
      upcard=gin_cards.card("7d"),
      stock_count=2,
  )
  n = w._replace(upcard=jnp.int32(gin_cards.card("Qs")))
  assert _act(w, n) == [cm.ACTION_KNOCK, cm.ACTION_PASS]


def test_knock_cache_equals_the_engine_cache_at_knock_nodes(env):
  """Knock cache equals the engine cache at knock nodes.

  knock_cache recomputes the engine's KNOCK-phase discard cache from the
  mover's own hand; the two agree at every KNOCK node.

  Args:
    env: The module `env` fixture.
  """
  g = gin_games.games(64, 4)
  mask = gin_games.decision_mask(env, g) & (
      np.asarray(g.states.game["phase"]) == cm.PHASE_KNOCK
  )
  st = gin_games.flat_states(g, mask)
  assert jax.tree_util.tree_leaves(st)[0].shape[0] > 50
  info = jax.vmap(env.info_state)(st, jax.vmap(env.actor)(st))
  legal, dw = jax.jit(jax.vmap(gin_endgame.knock_cache))(info)
  np.testing.assert_array_equal(
      np.asarray(legal), np.asarray(st.game["cached_legal_discard_mask"])
  )
  np.testing.assert_array_equal(
      np.asarray(dw), np.asarray(st.game["cached_all_discard_deadwoods"])
  )


def test_actions_are_legal_at_every_endgame_node(env):
  """Actions are legal at every endgame node.

  Args:
    env: The module `env` fixture.
  """
  g = gin_games.games(64, 4)
  phase = np.asarray(g.states.game["phase"])
  mask = gin_games.decision_mask(env, g) & np.isin(phase, [4, 5, 6])
  st = gin_games.flat_states(g, mask)
  info = jax.vmap(env.info_state)(st, jax.vmap(env.actor)(st))
  a = np.asarray(jax.jit(gin_endgame.actions)(info))
  legal = np.asarray(jax.vmap(env.legal_mask)(st))
  assert legal[np.arange(len(a)), a].all()
  assert len(a) > 200


def test_policy_contract_and_out_of_domain_nan(env):
  """Policy contract and out of domain nan.

  Args:
    env: The module `env` fixture.
  """
  outcomes = (
      [0] + gin_cards.cards(P0) + gin_cards.cards(P1) + [gin_cards.card("Td")]
  )
  s = env.deal(jnp.asarray(outcomes, jnp.int32))
  s = env.apply(s, jnp.int32(cm.ACTION_DRAW_UPCARD))
  s_knock = env.apply(s, jnp.int32(cm.ACTION_KNOCK))
  s_melds = env.apply(s_knock, jnp.int32(gin_cards.card("Kh")))
  b = 3
  st1 = tree_map(lambda x: jnp.stack([x] * b), s_knock)
  st2 = tree_map(lambda x: jnp.stack([x] * b), s_melds)
  step1 = gin_helpers.seat_steps_at(env, st1, True)
  step2 = gin_helpers.seat_steps_at(env, st2, False)
  pol = gin_endgame.GinScriptedEndgamePolicy()
  check_policy.check_policy(pol, {}, {}, step1, step2)
  out = pol.evaluate({}, (), step1, jax.random.key(0))
  np.testing.assert_array_equal(
      np.asarray(out.aux["action"]), [gin_cards.card("Kh")] * b
  )
  assert np.asarray(out.logits)[:, gin_cards.card("Kh")].max() == 0.0
  # A strategic node is outside the script's domain: NaN logits.
  st0 = tree_map(lambda x: jnp.stack([x] * b), s)
  out0 = pol.evaluate(
      {}, (), gin_helpers.seat_steps_at(env, st0, True), jax.random.key(0)
  )
  assert np.isnan(np.asarray(out0.logits)).all()


if __name__ == "__main__":
  # The package runs with x64 on, as its entry points and conftest.py set
  # it; set here for `python <file>`, never at import (which changes no
  # process-global state).
  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
