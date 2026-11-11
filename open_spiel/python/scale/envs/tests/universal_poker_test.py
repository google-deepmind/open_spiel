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

"""NoLimitHoldemEnv (universal_poker) tests.

The Env contract, privacy, known terminal returns (by hand and against pyspiel),
the max_len bound, the InfoState, x64 dtypes, the GameSpec row and serving.

Cards: id = 4 * rank + suit (rank 0 = deuce .. 12 = ace). Dealt: seat 0's two
hole cards, seat 1's two, then the board. Seat 0 is the big blind (2), seat 1
the small blind (1) and acts first preflop; seat 0 acts first after.
Actions: 0 fold, 1 call/check, k >= 2 raise to k total.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import no_limit_holdem as nlh
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.envs import universal_poker as up
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import wire
from open_spiel.python.scale.testing import check_env
from open_spiel.python.scale.testing import optional_pyspiel_games

F, C = up.FOLD, up.CALL


def card(rank: str, suit: int) -> int:
  """The card id of a rank and suit.

  Args:
    rank: The rank character.
    suit: The suit (0-3).

  Returns:
    The card id.
  """
  return 4 * "23456789TJQKA".index(rank) + suit


AA = (card("A", 0), card("A", 1))
KK = (card("K", 0), card("K", 1))
SEVEN_TWO = (card("7", 2), card("2", 1))
# No K (KK would make trips), no straight or flush for any hand used with it.
LOW_BOARD = (
    card("9", 2),
    card("J", 2),
    card("4", 3),
    card("3", 1),
    card("6", 0),
)
CHECK_ROUND = (C, C)
# Board straight A-K-Q-J-T, three suits, no flush possible: both seats play it.
BROADWAY = (
    card("A", 2),
    card("K", 3),
    card("Q", 0),
    card("J", 1),
    card("T", 2),
)
DEUCE_TREY_A = (card("2", 0), card("3", 1))
DEUCE_TREY_B = (card("2", 2), card("3", 3))


def worst_case_path():
  """The longest hand (MAX_LEN = 107 decisions).

  Limp, check; on each later round check, min-raises, call; 99 raises of 2 chips
  from 2 to 200.

  Returns:
    The actions.
  """
  targets = list(range(4, up.STACK + 1, 2))
  assert len(targets) == 99
  per = (33, 33, 33)
  out, i = [C, C], 0
  for n in per:
    out += [C] + targets[i : i + n] + [C]
    i += n
  return out


def events_of(holes0, holes1, rounds, board):
  """Raw events of a hand.

  Raw events: the 4 hole cards, then rounds[0], then per later round its
  board card(s) and actions (a round of None = no decisions, cards only).

  Args:
    holes0: Seat 0's hole cards.
    holes1: Seat 1's hole cards.
    rounds: The actions of each round (None: no decisions).
    board: The board cards.

  Returns:
    The events.
  """
  ev = list(holes0) + list(holes1) + list(rounds[0])
  deals = (board[:3], board[3:4], board[4:5])
  for r, cards in zip(rounds[1:], deals):
    ev += list(cards) + list(r or ())
  return ev


KNOWN = [
    # SB folds preflop: loses its 1-chip blind.
    ("sb_folds", list(AA) + list(KK) + [F], [1.0, -1.0]),
    # SB raises to 6, BB folds: SB wins the 2-chip blind.
    ("bb_folds", list(AA) + list(KK) + [6, F], [-2.0, 2.0]),
    # Checked down: AA beats 7-2 on a low board, pot 2 each.
    (
        "checkdown",
        events_of(AA, SEVEN_TWO, [CHECK_ROUND] * 4, LOW_BOARD),
        [2.0, -2.0],
    ),
    # Board straight: split pot.
    (
        "split",
        events_of(DEUCE_TREY_A, DEUCE_TREY_B, [CHECK_ROUND] * 4, BROADWAY),
        [0.0, 0.0],
    ),
    # Preflop all-in and call; the board runs out; AA holds against KK.
    (
        "allin",
        list(AA) + list(KK) + [200, C] + list(LOW_BOARD),
        [200.0, -200.0],
    ),
    # Flop: check, bet 8, raise 20, fold: the bettor loses 8 (2 blind + 6).
    (
        "flop_raise_fold",
        events_of(KK, AA, [CHECK_ROUND, (C, 8, 20, F)], LOW_BOARD),
        [8.0, -8.0],
    ),
]


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A no-limit hold'em env.

  Returns:
    The env.
  """
  return up.NoLimitHoldemEnv()


def _raw(env, events, end_at_chance=False):
  """Applies `events` from the root, checking actions are legal.

  Args:
    env: The env.
    events: The chance outcomes and actions.
    end_at_chance: Whether the result is a chance node.

  Returns:
    The state.
  """
  s = env.root()
  for e in events:
    assert not bool(env.terminal(s)), events
    if bool(env.is_chance(s)):
      s = env.apply_chance(s, e)
    else:
      assert bool(env.legal_mask(s)[e]), (events, e)
      s = env.apply(s, e)
  assert bool(env.is_chance(s)) == end_at_chance
  return s


def test_x64_is_on():
  """X64 is on."""
  assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, hide_fn=up.hide_opponent_hole, num_playouts=128)


@pytest.mark.parametrize(
    "label,events,expected", KNOWN, ids=[k[0] for k in KNOWN]
)
def test_known_terminal_returns(env, label, events, expected):
  """Known terminal returns.

  Args:
    env: The module `env` fixture.
    label: The parametrized case.
    events: The parametrized case.
    expected: The parametrized case.
  """
  s = _raw(env, events)
  assert bool(env.terminal(s))
  assert int(env.actor(s)) == -1  # The engine -4, normalized
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  assert env.returns(s).dtype == jnp.float32
  assert not bool(
      env.terminal(_raw(env, events[:-1], end_at_chance=label == "allin"))
  )


def test_known_returns_match_pyspiel(env):
  """Known returns match pyspiel.

  Args:
    env: The module `env` fixture.
  """
  game = optional_pyspiel_games.load_game(up.GAME_STRING)
  for _, events, expected in KNOWN:
    st = game.new_initial_state()
    for e in events:
      st.apply_action(e)
    assert st.is_terminal()
    np.testing.assert_array_equal(
        np.asarray(st.returns(), np.float32), expected
    )
    np.testing.assert_array_equal(
        np.asarray(env.returns(_raw(env, events))), expected
    )


def _worst_case_events():
  """The events of the longest hand: aces vs kings on a low board.

  Returns:
    The events.
  """
  acts = worst_case_path()
  ev = list(AA) + list(KK) + acts[:2]
  i = 2
  for cards in (LOW_BOARD[:3], LOW_BOARD[3:4], LOW_BOARD[4:5]):
    ev += list(cards)
    n = 1 + 33 + 1
    ev += acts[i : i + n]
    i += n
  assert i == len(acts) == up.MAX_LEN == 107
  return ev


def test_max_len_is_reached_by_the_worst_case_path(env):
  """Max len is reached by the worst case path.

  The longest hand has exactly MAX_LEN decisions and ends terminal (the
  bound's derivation is in envs/universal_poker.py); facing the 99th raise
  (all-in) only fold and call remain; pyspiel agrees on the whole path.

  Args:
    env: The module `env` fixture.
  """
  ev = _worst_case_events()
  s = _raw(env, ev)
  assert bool(env.terminal(s))
  assert int(env.info_state(s, 0).num_actions) == up.MAX_LEN
  np.testing.assert_array_equal(np.asarray(s.spent), [200, 200])
  # AA (seat 0) vs KK on the low board: seat 0 wins the full stack.
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [200.0, -200.0])
  s99 = _raw(env, ev[:-1])  # facing the all-in: fold or call only
  np.testing.assert_array_equal(
      np.flatnonzero(np.asarray(env.legal_mask(s99))), [F, C]
  )
  st = optional_pyspiel_games.load_game(up.GAME_STRING).new_initial_state()
  for e in ev:
    st.apply_action(e)
  assert st.is_terminal() and len(st.history()) == up.MAX_LEN + 9
  np.testing.assert_array_equal(
      np.asarray(st.returns(), np.float32), [200.0, -200.0]
  )


def test_legal_masks_actor_and_chance(env):
  """Legal masks actor and chance.

  Args:
    env: The module `env` fixture.
  """
  s = env.root()
  assert bool(env.is_chance(s))
  np.testing.assert_allclose(
      np.asarray(env.chance_probs(s)), np.full(52, 1 / 52), rtol=1e-6
  )
  s = _raw(env, list(AA) + list(KK))
  assert int(env.actor(s)) == 1  # the small blind acts first preflop
  legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
  # min raise to 4
  np.testing.assert_array_equal(legal, [F, C] + list(range(4, 201)))
  # limp: the BB may check or raise, not fold
  s = _raw(env, list(AA) + list(KK) + [C])
  assert int(env.actor(s)) == 0
  np.testing.assert_array_equal(
      np.flatnonzero(np.asarray(env.legal_mask(s))), [C] + list(range(4, 201))
  )
  # flop pending
  s = _raw(env, list(AA) + list(KK) + [C, C], end_at_chance=True)
  probs = np.asarray(env.chance_probs(s))
  assert probs.dtype == np.float32 and probs.shape == (52,)
  assert (probs[list(AA + KK)] == 0).all()
  np.testing.assert_allclose(probs[probs > 0], 1 / 48, rtol=1e-6)
  s = _raw(env, list(AA) + list(KK) + [C, C] + list(LOW_BOARD[:3]))
  assert int(env.actor(s)) == 0  # seat 0 first after the flop
  # raise to 10, re-raise to 30 (by 20)
  s = _raw(env, list(AA) + list(KK) + [10, 30])
  legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
  np.testing.assert_array_equal(legal, [F, C] + list(range(50, 201)))


def test_step_resolves_the_board(env):
  """Step resolves the board.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, list(AA) + list(KK) + [C])
  # closes preflop: flop dealt
  s2 = env.step(s, jnp.int32(C), jax.random.key(3))
  assert not bool(env.is_chance(s2)) and int(env.actor(s2)) == 0
  board = np.asarray(env.info_state(s2, 0).board)
  assert (board[:3] >= 0).all() and (board[3:] == -1).all()
  assert not set(board[:3].tolist()) & set(AA + KK)
  # an int64 action under x64
  s3 = env.step(s, jnp.int64(200), jax.random.key(4))
  # all-in called: board runs out
  s3 = env.step(s3, jnp.int32(C), jax.random.key(5))
  assert bool(env.terminal(s3))
  assert (np.asarray(env.info_state(s3, 1).board) >= 0).all()


def test_info_state_is_own_hole_and_public_history(env):
  """Info state is own hole and public history.

  Args:
    env: The module `env` fixture.
  """
  ev = list(KK) + list(AA) + [6, 18, C] + list(LOW_BOARD[:3]) + [C, 30]
  s = _raw(env, ev)
  i0, i1 = env.info_state(s, 0), env.info_state(s, 1)
  assert tuple(np.asarray(i0.hole)) == KK and tuple(np.asarray(i1.hole)) == AA
  assert int(i0.seat) == 0 and int(i1.seat) == 1
  np.testing.assert_array_equal(
      np.asarray(i0.board), list(LOW_BOARD[:3]) + [-1, -1]
  )
  assert int(i0.round) == 1
  np.testing.assert_array_equal(np.asarray(i0.spent), [18, 30])
  assert int(i0.num_actions) == 5
  np.testing.assert_array_equal(
      np.asarray(i0.actions)[:6], [6, 18, C, C, 30, -1]
  )
  np.testing.assert_array_equal(
      np.asarray(i0.action_rounds)[:6], [0, 0, 0, 1, 1, -1]
  )
  assert (np.asarray(i0.actions)[5:] == -1).all()
  for f in (
      "board",
      "round",
      "spent",
      "actions",
      "action_rounds",
      "num_actions",
  ):
    np.testing.assert_array_equal(
        np.asarray(getattr(i0, f)), np.asarray(getattr(i1, f))
    )
  assert tuple(np.asarray(env.labels(s, 0)["opp_hole"])) == AA
  assert tuple(np.asarray(env.labels(s, 1)["opp_hole"])) == KK
  assert env.world_from_info(i0) is None
  assert int(env.visible_action(s, jnp.int32(30), 0)) == 30  # betting is public


def test_round_is_clipped_after_the_river(env):
  """Round is clipped after the river.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, events_of(AA, SEVEN_TWO, [CHECK_ROUND] * 4, LOW_BOARD))
  assert int(s.round) == 4  # The engine's value past the river
  info = env.info_state(s, 0)
  assert int(info.round) == 3
  np.testing.assert_array_equal(
      np.asarray(info.action_rounds)[:8], [0, 0, 1, 1, 2, 2, 3, 3]
  )


def test_info_state_ignores_opponent_hole(env):
  """Info state ignores opponent hole.

  Same own hole, same public history, different opponent holes ->
  identical InfoState (the twin form of the privacy check).

  Args:
    env: The module `env` fixture.
  """
  hist = [6, C] + list(LOW_BOARD[:3]) + [C]
  a = env.info_state(_raw(env, list(AA) + list(KK) + hist), 0)
  b = env.info_state(_raw(env, list(AA) + list(SEVEN_TWO) + hist), 0)
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_hide_fn_redraws_the_opponent_hole_consistently(env):
  """Hide fn redraws the opponent hole consistently.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, list(AA) + list(KK) + [C, C] + list(LOW_BOARD[:3]))
  seen = set()
  for k in range(32):
    h = up.hide_opponent_hole(env, s, jnp.int32(0), jax.random.key(k))
    opp = tuple(int(c) for c in np.asarray(h.private_cards[1]))
    seen.add(opp)
    assert opp[0] != opp[1]
    assert not set(opp) & set(AA + LOW_BOARD[:3])
    assert tuple(np.asarray(h.private_cards[0])) == AA
    assert h.private_cards.dtype == s.private_cards.dtype
    dealt = set(AA + LOW_BOARD[:3] + opp)
    np.testing.assert_array_equal(
        np.asarray(h.deck_mask), [c not in dealt for c in range(52)]
    )
  assert len(seen) > 16


def test_declared_dtypes_survive_x64(env):
  """Declared dtypes survive x64.

  Args:
    env: The module `env` fixture.
  """
  s = env.init(jax.random.key(0))
  a = jnp.argmax(env.legal_mask(s))  # int64 under x64
  s2 = env.step(s, a, jax.random.key(1))
  for x, y in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(s2)):
    assert x.dtype == y.dtype and x.shape == y.shape
  for st in (s, s2):
    for seat in (0, 1):
      for leaf in jax.tree_util.tree_leaves(env.info_state(st, seat)):
        assert leaf.dtype == jnp.int32, leaf.dtype
    assert env.actor(st).dtype == jnp.int32
    assert env.legal_mask(st).dtype == jnp.bool_
    assert env.returns(st).dtype == jnp.float32
    assert env.visible_action(st, a, 0).dtype == jnp.int32


def test_uniform_self_play_through_the_arena(env):
  """Uniform self play through the arena.

  Args:
    env: The module `env` fixture.
  """
  pol = uniform.UniformPolicy(env.name, env.num_actions, env.num_players)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b = 64
  table = arena_lib.self_play_table(b, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(1), b, 2)
  params = [
      jax.tree_util.tree_map(
          lambda x: x[None], pol.init_params(jax.random.key(0))
      )
  ]
  _, batch, log = ar.collect(params, table, dk, pk, 2 * (env.max_len + 1))
  done = np.asarray(log.done)
  assert done.sum() >= b
  r = np.asarray(log.rewards)[done]
  np.testing.assert_allclose(r.sum(-1), 0.0)
  assert (np.abs(r) <= up.STACK).all()
  assert not np.asarray(log.bad).any()
  assert batch.action.dtype == np.int32


def test_registry_row_and_serving():
  """Registry row and serving."""
  assert "universal_poker" in registry.builtin_names()
  env = registry.make("universal_poker")
  assert isinstance(env, up.NoLimitHoldemEnv)
  assert registry.spec_of(env) == registry.spec("universal_poker")
  assert "universal_poker" in games.served_games()
  optional_pyspiel_games.require(up.GAME_STRING)
  pyspiel = pytest.importorskip("pyspiel")
  params = wire.game_params(pyspiel.load_game(up.GAME_STRING))
  assert games.env_for("universal_poker", params).name == "universal_poker"
  # pyspiel's universal_poker defaults are a different game: refused.
  with pytest.raises(ValueError, match="bettingAbstraction"):
    games.env_for(
        "universal_poker",
        wire.game_params(pyspiel.load_game("universal_poker")),
    )
  with pytest.raises(ValueError, match="must state"):
    games.env_for("universal_poker", {})
  with pytest.raises(ValueError, match="stack"):
    games.env_for("universal_poker", {**params, "stack": "100 100"})


def test_registry_rules_are_the_engines_configuration():
  """Registry rules are the engines configuration."""
  # Reaches private members of the code under test.
  assert registry._UNIVERSAL_POKER_RULES == nlh._SUPPORTED_KWARGS  # pylint: disable=protected-access
  pins = registry.game("universal_poker").serving.pyspiel_params
  optional_pyspiel_games.require(up.GAME_STRING)
  pyspiel = pytest.importorskip("pyspiel")
  full = dict(pyspiel.load_game(up.GAME_STRING).get_parameters())
  assert set(pins) == set(full)  # every pyspiel parameter is pinned ...
  for k, v in full.items():
    assert pins[k] == (v,), (k, v, pins[k])  # ... at the canonical game's value


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
