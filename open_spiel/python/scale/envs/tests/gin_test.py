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

"""GinEnv: the Env contract, known answers, the raw-layer deal, InfoState."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.tests import gin_cards
from open_spiel.python.scale.envs.tests import gin_chance_fixtures
from open_spiel.python.scale.testing import check_env

# A gin hand for player 0 once it takes the upcard Td: As2s3s 4c5c6c 7d8d9d +
# Kh.
P0 = "As 2s 3s 4c 5c 6c 7d 8d 9d Kh"
# No meld for player 1: 6+8+10+10+2+8+10+10+3+5 = 72 points.
P1 = "6s 8s Ts Qs 2c 8c Tc Kc 3d 5d"
UPCARD = "Td"
P1_POINTS = 72


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A gin env with Oklahoma rules.

  Returns:
    The env.
  """
  return gin.GinEnv(oklahoma_frac=1.0)


def _known_deal(env, first_player=0):
  """The state after dealing P0, P1 and the upcard.

  Args:
    env: The env.
    first_player: The first player outcome.

  Returns:
    The state.
  """
  outcomes = (
      [first_player]
      + gin_cards.cards(P0)
      + gin_cards.cards(P1)
      + [gin_cards.card(UPCARD)]
  )
  return env.deal(jnp.asarray(outcomes, jnp.int32))


def _legal(env, s):
  """The legal actions of `s`.

  Args:
    env: The env.
    s: The state.

  Returns:
    The set of legal actions.
  """
  return set(np.nonzero(np.asarray(env.legal_mask(s)))[0].tolist())


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, hide_fn=gin.hide_hidden_cards, num_playouts=16)


def test_check_env_passes_with_a_variant_node():
  """Check env passes with a variant node."""
  check_env.check_env(
      gin.GinEnv(oklahoma_frac=0.5),
      hide_fn=gin.hide_hidden_cards,
      num_playouts=8,
      seed=3,
  )


def test_sizes_and_bound(env):
  """Sizes and bound.

  Args:
    env: The module `env` fixture.
  """
  assert (env.num_players, env.num_actions, env.max_chance_outcomes) == (
      2,
      241,
      52,
  )
  # EVENTS_MAX (160) + wall pass (1) + knocker: discard, 3 melds, pass (5)
  # + defender: <= 10 layoffs + pass + <= 3 melds + pass, at most 10 cards (12)
  # ... bounded above by 15.
  assert gin.MAX_LEN == 160 + 1 + 5 + 15 == env.max_len
  assert env.zero_sum and env.name == "gin_rummy"


def test_root_meta_chance_nodes():
  """Root meta chance nodes."""
  e = gin.GinEnv(oklahoma_frac=0.25)
  s = e.root()
  assert bool(e.is_chance(s)) and not bool(e.terminal(s))
  assert not np.asarray(e.legal_mask(s)).any()
  p = np.asarray(e.chance_probs(s))
  assert p.dtype == np.float32 and p.shape == (52,)
  np.testing.assert_allclose(p[:2], [0.75, 0.25], atol=1e-7)
  assert p[2:].sum() == 0
  s = e.apply_chance(s, jnp.int32(1))  # oklahoma
  np.testing.assert_allclose(np.asarray(e.chance_probs(s))[:2], [0.5, 0.5])
  # No variant node when every deal is Oklahoma (oklahoma_frac=1.0).
  e1 = gin.GinEnv(oklahoma_frac=1.0)
  np.testing.assert_allclose(
      np.asarray(e1.chance_probs(e1.root()))[:2], [0.5, 0.5]
  )
  assert bool(e1.root().game["oklahoma"])
  e0 = gin.GinEnv(oklahoma_frac=None)
  assert not bool(e0.root().game["oklahoma"])


def test_known_deal_first_decision(env):
  """Known deal first decision.

  Args:
    env: The module `env` fixture.
  """
  s = _known_deal(env)
  assert not bool(env.is_chance(s))
  assert int(env.actor(s)) == 0
  assert _legal(env, s) == {cm.ACTION_DRAW_UPCARD, cm.ACTION_PASS}
  info = env.info_state(s, jnp.int32(0))
  np.testing.assert_array_equal(np.asarray(info.hand), gin_cards.hand_mask(P0))
  assert int(info.upcard) == gin_cards.card(UPCARD)
  assert int(info.stock_count) == 31
  assert int(info.knock_card) == 10  # Oklahoma: the value of the Td upcard
  assert int(info.phase) == cm.PHASE_FIRST_UPCARD
  # Player 1 moves first when the first-player node says so.
  s1 = _known_deal(env, first_player=1)
  assert int(env.actor(s1)) == 1
  # The engine cache follows the first player (hand + upcard, as
  # init_dealt_state).
  assert int(s1.game["cached_min_deadwood"]) > 0


def test_known_gin_hand_returns(env):
  """p0 takes Td, knocks, discards Kh, lays its three melds, passes: gin."""
  s = _known_deal(env)
  k = jax.random.key(0)
  s = env.step(s, jnp.int32(cm.ACTION_DRAW_UPCARD), k)
  assert int(env.actor(s)) == 0 and cm.ACTION_KNOCK in _legal(env, s)
  s = env.step(s, jnp.int32(cm.ACTION_KNOCK), k)
  assert int(s.game["phase"]) == cm.PHASE_KNOCK
  assert gin_cards.card("Kh") in _legal(env, s)
  s = env.step(s, jnp.int32(gin_cards.card("Kh")), k)
  for meld in ("As 2s 3s", "4c 5c 6c", "7d 8d 9d Td"):
    a = cm.ACTION_MELD_BASE + gin_cards.meld_id(meld)
    assert a in _legal(env, s), meld
    s = env.step(s, jnp.int32(a), k)
  s = env.step(s, jnp.int32(cm.ACTION_PASS), k)
  assert int(s.game["phase"]) == cm.PHASE_LAYOFF and int(env.actor(s)) == 1
  info1 = env.info_state(s, jnp.int32(1))
  assert bool(info1.knocker_gin) and int(info1.knocker) == 0
  s = env.step(s, jnp.int32(cm.ACTION_PASS), k)
  assert bool(env.terminal(s)) and int(env.actor(s)) == -1
  np.testing.assert_array_equal(
      np.asarray(env.returns(s)),
      [P1_POINTS + cm.GIN_BONUS, -(P1_POINTS + cm.GIN_BONUS)],
  )
  assert np.asarray(env.returns(s)).dtype == np.float32
  np.testing.assert_array_equal(
      np.asarray(env.info_state(s, jnp.int32(1)).scores), [97, -97]
  )


def test_events_private_card_is_own_only(env):
  """Both pass the first upcard, p0 draws from stock: only p0 sees the card."""
  s = _known_deal(env)
  s = env.apply(s, jnp.int32(cm.ACTION_PASS))
  s = env.apply(s, jnp.int32(cm.ACTION_PASS))
  s = env.apply(s, jnp.int32(cm.ACTION_DRAW_STOCK))
  assert bool(env.is_chance(s))
  drawn = gin_cards.card("Jh")
  s = env.apply_chance(s, jnp.int32(drawn))
  i0, i1 = env.info_state(s, jnp.int32(0)), env.info_state(s, jnp.int32(1))
  assert int(i0.ev_count) == int(i1.ev_count) == 3
  # Relative actor codes: 1 = the observing seat, 2 = the opponent.
  np.testing.assert_array_equal(np.asarray(i0.ev_actor)[:3], [1, 2, 1])
  np.testing.assert_array_equal(np.asarray(i1.ev_actor)[:3], [2, 1, 2])
  np.testing.assert_array_equal(np.asarray(i0.ev_type)[:3], [1, 1, 3])
  # The declined upcard is the public card of the passes and the stock draw.
  np.testing.assert_array_equal(
      np.asarray(i0.ev_public)[:3], [gin_cards.card(UPCARD) + 1] * 3
  )
  assert int(np.asarray(i0.ev_private)[2]) == drawn + 1
  assert not np.asarray(i1.ev_private).any()
  assert int(np.asarray(i0.hand)[drawn]) == 1
  assert int(np.asarray(env.labels(s, jnp.int32(1))["opp_hand"])[drawn]) == 1
  assert int(i1.stock_count) == 30


def test_info_state_fields_and_dtypes(env):
  """Info state fields and dtypes.

  Args:
    env: The module `env` fixture.
  """
  s = _known_deal(env)
  info = env.info_state(s, jnp.int32(1))
  want = {
      "seat": jnp.int32,
      "hand": jnp.int8,
      "discard_pile": jnp.int8,
      "upcard": jnp.int32,
      "known_cards": jnp.int8,
      "phase": jnp.int32,
      "current_player": jnp.int32,
      "knock_card": jnp.int32,
      "oklahoma": jnp.bool_,
      "stock_count": jnp.int32,
      "pass_count": jnp.int32,
      "num_draw_upcard": jnp.int32,
      "repeated_move": jnp.bool_,
      "upcard_taken": jnp.int32,
      "allow_upcard_throwback": jnp.bool_,
      "layed_melds": jnp.bool_,
      "layoffs_mask": jnp.bool_,
      "knocker": jnp.int32,
      "finished_layoffs": jnp.bool_,
      "knocker_gin": jnp.bool_,
      "done": jnp.bool_,
      "scores": jnp.int32,
      "ev_actor": jnp.int8,
      "ev_type": jnp.int8,
      "ev_public": jnp.int8,
      "ev_private": jnp.int8,
      "ev_count": jnp.int32,
      "ev_error": jnp.int32,
  }
  assert set(info._fields) == set(want)
  for name, dt in want.items():
    assert getattr(info, name).dtype == dt, (name, getattr(info, name).dtype)
  assert info.layed_melds.shape == (2, cm.NUM_MELDS)
  assert info.ev_actor.shape == (gin.EVENTS_MAX,)
  # Seat 1's hand, never seat 0's.
  np.testing.assert_array_equal(np.asarray(info.hand), gin_cards.hand_mask(P1))
  np.testing.assert_array_equal(
      np.asarray(env.labels(s, jnp.int32(1))["opp_hand"]),
      gin_cards.hand_mask(P0).astype(bool),
  )
  assert env.labels(s, jnp.int32(1))["opp_hand"].dtype == jnp.bool_
  assert env.world_from_info(info) is None


def test_visible_action_is_public(env):
  """Visible action is public.

  Args:
    env: The module `env` fixture.
  """
  s = _known_deal(env)
  for q in (0, 1):
    assert (
        int(env.visible_action(s, jnp.int32(cm.ACTION_PASS), jnp.int32(q)))
        == cm.ACTION_PASS
    )


def test_hide_hidden_cards_moves_only_hidden_cards(env):
  """The privacy hook keeps the seat's hand, public cards and counts."""
  s = _known_deal(env)
  for seed in range(8):
    h = gin.hide_hidden_cards(env, s, jnp.int32(0), jax.random.key(seed))
    g, hg = s.game, h.game
    np.testing.assert_array_equal(
        np.asarray(hg["player0_hand"]), np.asarray(g["player0_hand"])
    )
    assert int(hg["player1_hand"].sum()) == 10 and int(hg["deck"].sum()) == 31
    union = np.asarray(g["player1_hand"]) + np.asarray(g["deck"])
    np.testing.assert_array_equal(
        np.asarray(hg["player1_hand"]) + np.asarray(hg["deck"]), union
    )
  moved = [
      not np.array_equal(
          np.asarray(
              gin.hide_hidden_cards(
                  env, s, jnp.int32(0), jax.random.key(i)
              ).game["player1_hand"]
          ),
          np.asarray(s.game["player1_hand"]),
      )
      for i in range(4)
  ]
  assert all(moved)


def test_init_is_deterministic_and_deals_both_first_players(env):
  """Init is deterministic and deals both first players.

  Args:
    env: The module `env` fixture.
  """
  keys = jax.random.split(jax.random.key(11), 256)
  s = jax.jit(jax.vmap(env.init))(keys)
  s2 = jax.jit(jax.vmap(env.init))(keys)
  for x, y in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(s2)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  first = np.asarray(jax.vmap(env.actor)(s))
  assert set(first.tolist()) == {0, 1}
  assert abs(first.mean() - 0.5) < 4 * np.sqrt(0.25 / 256)
  assert (np.asarray(s.game["player0_hand"]).sum(-1) == 10).all()
  assert (np.asarray(s.game["deck"]).sum(-1) == 31).all()
  assert np.asarray(s.game["oklahoma"]).all()


def test_endgame_phase_predicate():
  """Endgame phase predicate."""
  phases = jnp.arange(8, dtype=jnp.int32)
  np.testing.assert_array_equal(
      np.asarray(gin.is_endgame_phase(phases)),
      [p in (cm.PHASE_KNOCK, cm.PHASE_LAYOFF, cm.PHASE_WALL) for p in range(8)],
  )
  assert types.NO_ACTION == -1


# The engine-chance known answer, one shard per file (gin_chance_fixtures).
@pytest.mark.parametrize("okla", gin_chance_fixtures.SHARDS[0])
def test_one_step_engine_chance_equals_step_then_install(okla):
  """One step engine chance equals step then install (shard 0)."""
  gin_chance_fixtures.check_one_step_engine_chance(okla)


def test_the_shards_cover_every_oklahoma_frac_once():
  """The shards check each oklahoma_frac once, and each shard file exists."""
  got = []
  for k in sorted(gin_chance_fixtures.SHARDS):
    got.extend(gin_chance_fixtures.SHARDS[k])
  assert got == list(gin_chance_fixtures.OKLAHOMA_FRACS), got
  here = os.path.dirname(os.path.abspath(__file__))
  for k in gin_chance_fixtures.SHARDS:
    if k:
      assert os.path.exists(os.path.join(here, f"gin_shard{k}_test.py")), k


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
