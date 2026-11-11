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

"""LiarsDiceEnv tests.

The game is OpenSpiel liars_dice(numdice=2,dice_sides=5), reset-face. Tested:
the Env contract, privacy, and known answers.

Chance outcome o rolls face o + 1; seat 0 rolls both its dice, then seat 1.
Bid id = (quantity - 1) * 5 + (face - 1), ids 0..19; LIAR = 20. The highest
face (5) is wild: it counts toward every bid face.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import liars_dice
from open_spiel.python.scale.testing import check_env

LIAR = liars_dice.LIAR


def bid(quantity, face):
  """The bid action id.

  Args:
    quantity: The bid quantity.
    face: The bid face.

  Returns:
    The action id.
  """
  return (quantity - 1) * liars_dice.DICE_SIDES + (face - 1)


# Seat 0 rolls faces 5, 1 (outcomes 4, 0); seat 1 rolls 3, 4 (outcomes 2, 3).
DEAL = (4, 0, 2, 3)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A liar's dice env.

  Returns:
    The env.
  """
  return liars_dice.LiarsDiceEnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(
      env, hide_fn=liars_dice.hide_opponent_dice, num_playouts=128
  )


def _raw(env, events, end_at_chance=False):
  """Applies `events` from the root.

  Args:
    env: The env.
    events: The chance outcomes and actions.
    end_at_chance: Whether the result is a chance node.

  Returns:
    The state.
  """
  s = env.root()
  for e in events:
    assert not bool(env.terminal(s))
    s = env.apply_chance(s, e) if bool(env.is_chance(s)) else env.apply(s, e)
  assert bool(env.is_chance(s)) == end_at_chance
  return s


KNOWN = [
    # "two 1s": seat 0's 1 and its wild 5 make two -> the bidder wins.
    ((bid(2, 1), LIAR), [1.0, -1.0]),
    # "three 1s": only two -> the caller (seat 1) wins.
    ((bid(3, 1), LIAR), [-1.0, 1.0]),
    # seat 1 bids "two 4s" (its 4 + seat 0's wild 5); seat 0 calls and loses.
    ((bid(1, 3), bid(2, 4), LIAR), [-1.0, 1.0]),
    # "two 3s": seat 1's 3 and seat 0's wild 5; seat 1 raises to "three 3s",
    # seat 0 calls and wins.
    ((bid(2, 3), bid(3, 3), LIAR), [1.0, -1.0]),
]


@pytest.mark.parametrize("bids,expected", KNOWN)
def test_known_terminal_returns(env, bids, expected):
  """Known terminal returns.

  Args:
    env: The module `env` fixture.
    bids: The parametrized case.
    expected: The parametrized case.
  """
  s = _raw(env, DEAL + bids)
  assert bool(env.terminal(s))
  assert int(env.actor(s)) == -1  # The engine says -4; normalized
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  assert env.returns(s).dtype == jnp.float32
  assert not bool(env.terminal(_raw(env, DEAL + bids[:-1])))


def test_known_returns_match_pyspiel(env):
  """Known returns match pyspiel.

  Args:
    env: The module `env` fixture.
  """
  del env  # Unused.
  pyspiel = pytest.importorskip("pyspiel")
  game = pyspiel.load_game("liars_dice(numdice=2,dice_sides=5)")
  for bids, expected in KNOWN:
    st = game.new_initial_state()
    for e in DEAL + bids:
      st.apply_action(e)
    assert st.is_terminal()
    np.testing.assert_array_equal(
        np.asarray(st.returns(), np.float32), expected
    )


def test_chance_and_legal_masks(env):
  """Chance and legal masks.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, DEAL[:1], end_at_chance=True)
  np.testing.assert_allclose(
      np.asarray(env.chance_probs(s)), [0.2] * 5, atol=1e-7
  )
  s = _raw(env, DEAL)
  assert int(env.actor(s)) == 0
  # No bid yet: every bid, but not LIAR.
  np.testing.assert_array_equal(
      np.asarray(env.legal_mask(s)), [True] * 20 + [False]
  )
  s = _raw(env, DEAL + (bid(2, 4),))
  assert int(env.actor(s)) == 1
  np.testing.assert_array_equal(
      np.asarray(env.legal_mask(s)), [a > bid(2, 4) for a in range(21)]
  )
  s = _raw(env, DEAL + (bid(4, 5),))  # the highest bid: only LIAR remains
  np.testing.assert_array_equal(
      np.asarray(env.legal_mask(s)), [False] * 20 + [True]
  )


def test_info_state_is_own_dice_and_the_bids(env):
  """Info state is own dice and the bids.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, DEAL + (bid(1, 3), bid(2, 4)))
  i0, i1 = env.info_state(s, 0), env.info_state(s, 1)
  assert int(i0.seat) == 0 and int(i1.seat) == 1
  np.testing.assert_array_equal(np.asarray(i0.dice), [1, 5])  # faces, sorted
  np.testing.assert_array_equal(np.asarray(i1.dice), [3, 4])
  want = [bid(1, 3), bid(2, 4)] + [-1] * 19
  np.testing.assert_array_equal(np.asarray(i0.bids), want)
  np.testing.assert_array_equal(np.asarray(i1.bids), want)
  assert int(i0.num_bids) == 2 == int(i1.num_bids)
  np.testing.assert_array_equal(
      np.asarray(env.labels(s, 0)["opp_dice"]), [3, 4]
  )
  np.testing.assert_array_equal(
      np.asarray(env.labels(s, 1)["opp_dice"]), [1, 5]
  )
  assert env.world_from_info(i0) is None
  # bids are public
  assert int(env.visible_action(s, jnp.int32(LIAR), 1)) == LIAR


def test_info_state_ignores_opponent_dice(env):
  """Info state ignores opponent dice.

  Args:
    env: The module `env` fixture.
  """
  a = env.info_state(_raw(env, (4, 0, 2, 3, bid(1, 2))), 1)
  b = env.info_state(_raw(env, (1, 1, 2, 3, bid(1, 2))), 1)
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_hide_fn_rerolls_only_the_opponent(env):
  """Hide fn rerolls only the opponent.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, DEAL + (bid(1, 3),))
  seen = set()
  for k in range(32):
    h = liars_dice.hide_opponent_dice(env, s, jnp.int32(0), jax.random.key(k))
    np.testing.assert_array_equal(np.asarray(h["dice"][0]), [1, 5])
    d1 = tuple(np.asarray(h["dice"][1]).tolist())
    assert d1 == tuple(sorted(d1)) and all(1 <= f <= 5 for f in d1)
    assert h["dice"].dtype == s["dice"].dtype
    seen.add(d1)
  assert len(seen) > 1


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
