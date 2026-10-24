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

"""The gin rule bot (simple_bot over InfoState): contract and known answers."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.core import decide
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot
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


@pytest.fixture(name="deal", scope="module")
def _deal_fixture(env):
  """The state after dealing P0, P1 and the Td upcard.

  Args:
    env: The env.

  Returns:
    The state.
  """
  return env.deal(
      jnp.asarray(
          [0]
          + gin_cards.cards(P0)
          + gin_cards.cards(P1)
          + [gin_cards.card("Td")],
          jnp.int32,
      )
  )


def test_spec_and_registry():
  """Spec and registry."""
  pol = registry.build(gin_rule_bot.ARCH, {})
  assert isinstance(pol, gin_rule_bot.GinRuleBotPolicy)
  s = pol.spec
  assert (s.game, s.num_actions, s.num_players, s.stateful) == (
      "gin_rummy",
      241,
      2,
      False,
  )
  assert s.capability == types.Capability.NATIVE_JAX and s.framework == "jax"


def test_known_answers_take_then_knock(env, deal):
  """Known answers take then knock.

  p0 holds As2s3s 4c5c6c 7d8d9d Kh and sees Td: Td completes a run, so take
  it; then the 11-card hand discards to 0 deadwood, so knock.

  Args:
    env: The module `env` fixture.
    deal: The module known-deal fixture.
  """
  info = env.info_state(deal, jnp.int32(0))
  assert (
      int(gin_rule_bot.actions(gin_helpers.batch(info))[0])
      == cm.ACTION_DRAW_UPCARD
  )
  s = env.apply(deal, jnp.int32(cm.ACTION_DRAW_UPCARD))
  assert (
      int(
          gin_rule_bot.actions(
              gin_helpers.batch(env.info_state(s, jnp.int32(0)))
          )[0]
      )
      == cm.ACTION_KNOCK
  )


def test_known_answers_from_hand_built_infostates():
  """Known answers from hand built infostates."""
  # DRAW, nothing useful in the upcard, stock left: draw from stock.
  i = gin_helpers.make_info(
      hand=P1, phase=cm.PHASE_DRAW, upcard=gin_cards.card("Ah")
  )
  # FIRST_UPCARD, upcard useless: pass.
  j = gin_helpers.make_info(
      hand=P1,
      phase=cm.PHASE_FIRST_UPCARD,
      upcard=gin_cards.card("Ah"),
      pass_count=0,
  )
  # WALL (stock 2), upcard useless: pass (the hand ends drawn).
  w = gin_helpers.make_info(
      hand=P1, phase=cm.PHASE_WALL, upcard=gin_cards.card("Ah"), stock_count=2
  )
  # DISCARD, 11 cards, no knock: the highest-rank deadwood card (Kc).
  d = gin_helpers.make_info(hand=P1 + " 4h", phase=cm.PHASE_DISCARD)
  got = np.asarray(gin_rule_bot.actions(gin_helpers.batch(i, j, w, d))).tolist()
  assert got == [
      cm.ACTION_DRAW_STOCK,
      cm.ACTION_PASS,
      cm.ACTION_PASS,
      gin_cards.card("Kc"),
  ]
  # Stock size is all the bot reads of the stock: DRAW with stock 2 (the wall
  # cannot be drawn from) and a useless upcard passes, as simple_bot does.
  i2 = i._replace(stock_count=jnp.int32(2))
  assert int(gin_rule_bot.actions(gin_helpers.batch(i2))[0]) == cm.ACTION_PASS


def test_policy_contract(env, deal):
  """Policy contract.

  Args:
    env: The module `env` fixture.
    deal: The module `deal` fixture.
  """
  pol = gin_rule_bot.GinRuleBotPolicy()
  b = 4
  s0 = tree_map(lambda x: jnp.stack([x] * b), deal)
  s1 = jax.vmap(env.apply)(s0, jnp.full((b,), cm.ACTION_DRAW_UPCARD, jnp.int32))
  step1 = gin_helpers.seat_steps_at(env, s0, True)
  step2 = gin_helpers.seat_steps_at(env, s1, False)
  check_policy.check_policy(
      pol, pol.init_params(None), pol.init_params(None), step1, step2
  )
  out = pol.evaluate({}, (), step1, jax.random.key(0))
  a, probs = decide.decide(
      out, step1.legal_mask, types.ARGMAX, jax.random.key(0)
  )
  np.testing.assert_array_equal(np.asarray(a), [cm.ACTION_DRAW_UPCARD] * b)
  np.testing.assert_array_equal(np.asarray(out.aux["action"]), np.asarray(a))
  assert np.asarray(probs)[:, cm.ACTION_DRAW_UPCARD].min() == 1.0
  a2, _ = decide.decide(out, step1.legal_mask, types.SAMPLE, jax.random.key(3))
  np.testing.assert_array_equal(np.asarray(a2), np.asarray(a))


def test_not_to_move_gives_nan_logits(env, deal):
  """Not to move gives nan logits.

  Out of domain (not this seat's turn) the logits are NaN, so a misrouted
  slot fails the Arena's bad-ply check instead of playing silently.

  Args:
    env: The module `env` fixture.
    deal: The module known-deal fixture.
  """
  pol = gin_rule_bot.GinRuleBotPolicy()
  st = tree_map(lambda x: x[None], deal)
  step = plies.seat_step(env, st, 1, True, True, -1, -1)
  assert np.isnan(
      np.asarray(pol.evaluate({}, (), step, jax.random.key(0)).logits)
  ).all()


def test_actions_are_legal_on_seeded_games(env):
  """Actions are legal on seeded games.

  Args:
    env: The module `env` fixture.
  """
  g = gin_games.games(64, 3)
  mask = gin_games.decision_mask(env, g)
  st = gin_games.flat_states(g, mask)
  actor = jax.vmap(env.actor)(st)
  info = jax.vmap(env.info_state)(st, actor)
  a = np.asarray(jax.jit(gin_rule_bot.actions)(info))
  legal = np.asarray(jax.vmap(env.legal_mask)(st))
  strategic = ~np.asarray(gin.is_endgame_phase(info.phase))
  ok = legal[np.arange(len(a)), np.clip(a, 0, 240)]
  # simple_bot plays a legal move at every strategic node (its own endgame
  # moves are simplified; gin entrants route the endgame to a scripted policy).
  assert ok[strategic].all(), np.nonzero(~ok & strategic)[0][:5]
  assert strategic.sum() > 1000


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
