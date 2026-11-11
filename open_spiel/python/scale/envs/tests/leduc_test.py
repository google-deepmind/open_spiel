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

"""LeducEnv: the Env contract, privacy and known answers.

Cards are the engine's (OpenSpiel's) ids 0..5; rank = id // 2 (J J Q Q K K).
Actions: fold 0, call/check 1, raise 2. Ante 1; raises 2 (round 1), 4 (round 2).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.testing import check_env

F, C, R = 0, 1, 2
J0, J1, Q0, Q1, K0, K1 = range(6)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Leduc env.

  Returns:
    The env.
  """
  return leduc.LeducEnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, hide_fn=leduc.hide_opponent_card, num_playouts=128)


def _play_raw(env, events, end_at_chance=False):
  """Plays `events` through the raw layer.

  Raw layer: at a chance node apply the next event as a chance outcome,
  else as a player action.

  Args:
    env: The leduc env.
    events: The chance outcomes and player actions, in order.
    end_at_chance: Whether the play must end at a chance node.

  Returns:
    The final state.
  """
  s = env.root()
  for e in events:
    assert not bool(env.terminal(s))
    s = env.apply_chance(s, e) if bool(env.is_chance(s)) else env.apply(s, e)
  assert bool(env.is_chance(s)) == end_at_chance
  return s


@pytest.mark.parametrize(
    "events,expected",
    [
        # p0 raises, p1 folds: p0 wins p1's ante.
        ((J0, K0, R, F), [1.0, -1.0]),
        # check-check, public Q, check-check: showdown K beats J (pot 2).
        ((J0, K0, C, C, Q0, C, C), [-1.0, 1.0]),
        # raise-call (3 each), public J pairs p0; round 2 check, raise 4, call.
        ((J0, K0, R, C, J1, C, R, C), [7.0, -7.0]),
        # equal ranks, no pair: split pot.
        ((J0, J1, C, C, Q0, C, C), [0.0, 0.0]),
        # raise, re-raise, call (5 each); public K pairs p1; check-check.
        ((Q0, K1, R, R, C, K0, C, C), [-5.0, 5.0]),
        # check, raise, call (3 each); round 2 p0 raises 4, p1 folds its 3.
        ((Q0, K1, C, R, C, J0, R, F), [3.0, -3.0]),
    ],
)
def test_known_terminal_returns(env, events, expected):
  """Known terminal returns.

  Args:
    env: The module `env` fixture.
    events: The parametrized case.
    expected: The parametrized case.
  """
  s = _play_raw(env, events)
  assert bool(env.terminal(s))
  assert int(env.actor(s)) == -1
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  assert env.returns(s).dtype == jnp.float32


def test_known_returns_match_pyspiel(env):
  """Known returns match pyspiel.

  Args:
    env: The module `env` fixture.
  """
  pyspiel = pytest.importorskip("pyspiel")
  game = pyspiel.load_game("leduc_poker")
  for events in [
      (J0, K0, R, F),
      (J0, K0, R, C, J1, C, R, C),
      (Q0, K1, R, R, C, K0, C, C),
      (Q0, K1, C, R, C, J0, R, F),
      (J0, J1, C, C, Q0, C, C),
  ]:
    st = game.new_initial_state()
    for e in events:
      st.apply_action(e)
    assert st.is_terminal()
    np.testing.assert_array_equal(
        np.asarray(env.returns(_play_raw(env, events))),
        np.asarray(st.returns(), np.float32),
    )


def test_legal_masks_and_actor(env):
  """Legal masks and actor.

  Args:
    env: The module `env` fixture.
  """
  s = _play_raw(env, (J0, K0))
  assert int(env.actor(s)) == 0
  np.testing.assert_array_equal(
      np.asarray(env.legal_mask(s)), [False, True, True]
  )
  s = _play_raw(env, (J0, K0, R, R))  # two raises: no third
  assert int(env.actor(s)) == 0
  np.testing.assert_array_equal(
      np.asarray(env.legal_mask(s)), [True, True, False]
  )
  # round over: public card pending
  s = _play_raw(env, (J0, K0, C, C), end_at_chance=True)
  assert bool(env.is_chance(s)) and not bool(env.terminal(s))
  probs = np.asarray(env.chance_probs(s))
  np.testing.assert_allclose(probs, [0, 0.25, 0.25, 0.25, 0, 0.25], atol=1e-7)


def test_step_resolves_public_card(env):
  """Step resolves public card.

  Args:
    env: The module `env` fixture.
  """
  s = _play_raw(env, (J0, K0, C))
  s2 = env.step(s, jnp.int32(C), jax.random.key(3))
  assert not bool(env.is_chance(s2))
  assert int(env.actor(s2)) == 0
  pub = int(env.info_state(s2, 0).public_card)
  assert pub in (J1, Q0, Q1, K1)


def test_info_state_is_own_card_and_public_history(env):
  """Info state is own card and public history.

  Args:
    env: The module `env` fixture.
  """
  s = _play_raw(env, (Q0, K1, R, C, J0, C))
  i0, i1 = env.info_state(s, 0), env.info_state(s, 1)
  assert int(i0.card) == Q0 and int(i1.card) == K1
  assert int(i0.public_card) == J0 == int(i1.public_card)
  assert int(i0.seat) == 0 and int(i1.seat) == 1
  assert int(i0.round) == 2
  np.testing.assert_array_equal(np.asarray(i0.round1), [R, C, -1, -1])
  np.testing.assert_array_equal(np.asarray(i0.round2), [C, -1, -1, -1])
  np.testing.assert_array_equal(np.asarray(i0.ante), [3, 3])
  # Only the seat's own card differs between the two seats' InfoStates.
  for f in ("public_card", "round", "round1", "round2", "ante"):
    np.testing.assert_array_equal(
        np.asarray(getattr(i0, f)), np.asarray(getattr(i1, f))
    )
  assert int(env.labels(s, 0)["opp_card"]) == K1
  assert int(env.labels(s, 1)["opp_card"]) == Q0
  assert env.world_from_info(i0) is None
  assert int(env.visible_action(s, jnp.int32(R), 1)) == R


def test_public_card_absent_before_round_two(env):
  """Public card absent before round two.

  Args:
    env: The module `env` fixture.
  """
  s = _play_raw(env, (Q0, K1, R))
  assert int(env.info_state(s, 1).public_card) == -1
  np.testing.assert_array_equal(
      np.asarray(env.info_state(s, 1).round2), [-1] * 4
  )


def test_info_state_ignores_opponent_card(env):
  """Info state ignores opponent card.

  Same seat card, same public history, different opponent cards ->
  identical InfoState for the seat (the twin form of the privacy check).

  Args:
    env: The module `env` fixture.
  """
  a = env.info_state(_play_raw(env, (Q0, K1, R, C, J0, C)), 0)
  b = env.info_state(_play_raw(env, (Q0, J1, R, C, J0, C)), 0)
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_hide_fn_moves_the_opponent_card_and_keeps_the_deck_consistent(env):
  """Hide fn moves the opponent card and keeps the deck consistent.

  Args:
    env: The module `env` fixture.
  """
  s = _play_raw(env, (Q0, K1, C, C, J0))
  seen = set()
  for k in range(32):
    h = leduc.hide_opponent_card(env, s, jnp.int32(0), jax.random.key(k))
    opp = int(h.private_cards[1])
    seen.add(opp)
    assert opp not in (Q0, J0) and int(h.private_cards[0]) == Q0
    dealt = {Q0, J0, opp}
    np.testing.assert_array_equal(
        np.asarray(h.deck_mask), [c not in dealt for c in range(6)]
    )
  assert len(seen) > 1
  assert types.NO_ACTION == -1


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
