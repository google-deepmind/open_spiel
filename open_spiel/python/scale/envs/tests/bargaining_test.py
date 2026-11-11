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

"""BargainingEnv (OpenSpiel bargaining) tests.

The Env contract, privacy, options and known answers.

The deal is one chance node over the 1000 built-in instances (pool, seat 0's
values, seat 1's values); instance 0 is pool (1, 2, 3), values (8, 1, 0) and
(4, 0, 2). An offer [a, b, c] is what its PROPOSER keeps; AGREE (action 120)
accepts the last offer: the proposer gets values[proposer] . offer, the other
seat values[other] . (pool - offer). General-sum; no agreement within
max_turns offers (or an end drawn at a prob_end chance node) pays 0, 0.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import bargaining
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.testing import check_env

AGREE = bargaining.AGREE


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A bargaining env.

  Returns:
    The env.
  """
  return bargaining.BargainingEnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy.

  Args:
    env: The module `env` fixture.
  """
  assert not env.zero_sum and env.max_len == 10 and env.num_actions == 121
  check_env.check_env(
      env, hide_fn=bargaining.hide_opponent_values, num_playouts=128
  )


def test_check_env_with_prob_end_and_discount():
  """Check env with prob end and discount."""
  env = registry.make("bargaining", max_turns=4, discount=0.9, prob_end=0.25)
  assert env.max_len == 4
  check_env.check_env(
      env, hide_fn=bargaining.hide_opponent_values, num_playouts=64
  )


def test_offer_ids_follow_openspiel_order():
  """Offer ids follow openspiel order."""
  assert bargaining.offer_id(0, 0, 0) == 0
  # pyspiel: "Offer: Book: 1, Hat: 1, Basketball: 0"
  assert bargaining.offer_id(1, 1, 0) == 9
  assert bargaining.offer_id(0, 2, 0) == 15
  assert bargaining.offer_id(0, 0, 7) == AGREE - 1
  assert AGREE == 120


def _raw(env, events):
  """Applies `events`, chance outcomes at chance nodes, from the root.

  Args:
    env: The env.
    events: The chance outcomes and actions.

  Returns:
    The state.
  """
  s = env.root()
  for e in events:
    assert not bool(env.terminal(s))
    s = env.apply_chance(s, e) if bool(env.is_chance(s)) else env.apply(s, e)
  return s


O = bargaining.offer_id  # pylint: disable=invalid-name
KNOWN = [
    # seat 0 keeps (1, 1, 0) = 8 + 1; seat 1 gets (0, 1, 3) . (4, 0, 2) = 6.
    ((0, O(1, 1, 0), AGREE), [9.0, 6.0]),
    # seat 1 counters keeping (0, 2, 0) = 0; seat 0 gets (1, 0, 3) . (8, 1, 0) =
    # 8.
    ((0, O(1, 1, 0), O(0, 2, 0), AGREE), [8.0, 0.0]),
    # ten offers, no agreement.
    ((0,) + (O(1, 0, 0),) * 10, [0.0, 0.0]),
]


@pytest.mark.parametrize("events,expected", KNOWN)
def test_known_terminal_returns(env, events, expected):
  """Known terminal returns.

  Args:
    env: The module `env` fixture.
    events: The parametrized case.
    expected: The parametrized case.
  """
  s = _raw(env, events)
  assert bool(env.terminal(s))
  # The engine says chance (-1) at terminal; the env says -1
  assert int(env.actor(s)) == -1
  assert not bool(env.is_chance(s))
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  assert env.returns(s).dtype == jnp.float32
  assert not bool(env.terminal(_raw(env, events[:-1])))


def test_discount_known_answer():
  """Discount known answer."""
  # The discount applies from the third decision on: AGREE after two offers.
  env = registry.make("bargaining", discount=0.5)
  np.testing.assert_array_equal(
      np.asarray(env.returns(_raw(env, (0, O(1, 1, 0), O(0, 2, 0), AGREE)))),
      [4.0, 0.0],
  )
  np.testing.assert_array_equal(
      np.asarray(env.returns(_raw(env, (0, O(1, 1, 0), AGREE)))), [9.0, 6.0]
  )


def test_known_returns_match_pyspiel(env):
  """Known returns match pyspiel.

  Args:
    env: The module `env` fixture.
  """
  del env  # Unused.
  pyspiel = pytest.importorskip("pyspiel")
  for game_str, events, expected in [
      ("bargaining", e, x) for e, x in KNOWN
  ] + [(
      "bargaining(discount=0.5)",
      (0, O(1, 1, 0), O(0, 2, 0), AGREE),
      [4.0, 0.0],
  )]:
    st = pyspiel.load_game(game_str).new_initial_state()
    for e in events:
      st.apply_action(e)
    assert st.is_terminal()
    np.testing.assert_array_equal(
        np.asarray(st.returns(), np.float32), expected
    )


def test_legal_masks(env):
  """Legal masks.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, (0,))
  assert int(env.actor(s)) == 0
  m = np.asarray(env.legal_mask(s))
  assert not m[AGREE] and m.sum() == 2 * 3 * 4  # offers within pool (1, 2, 3)
  assert m[O(1, 2, 3)] and not m[O(2, 0, 0)]
  s = _raw(env, (0, O(1, 1, 0)))
  assert int(env.actor(s)) == 1 and bool(env.legal_mask(s)[AGREE])
  probs = np.asarray(env.chance_probs(env.root()))
  # The 1000 instances, then the mid-hand continue / end outcomes (pyspiel's
  # ids 1000 / 1001), which the deal never draws.
  assert probs.shape == (1002,) and probs.dtype == np.float32
  np.testing.assert_allclose(probs[:1000], 1e-3, rtol=1e-6)
  np.testing.assert_array_equal(probs[1000:], 0.0)


def test_info_state_is_own_values_and_public_offers(env):
  """Info state is own values and public offers.

  Args:
    env: The module `env` fixture.
  """
  s = _raw(env, (0, O(1, 1, 0), O(0, 2, 0)))
  i0, i1 = env.info_state(s, 0), env.info_state(s, 1)
  assert int(i0.seat) == 0 and int(i1.seat) == 1
  np.testing.assert_array_equal(np.asarray(i0.pool), [1, 2, 3])
  np.testing.assert_array_equal(np.asarray(i0.values), [8, 1, 0])
  np.testing.assert_array_equal(np.asarray(i1.values), [4, 0, 2])
  want = np.full((10, 3), -1)
  want[0] = [1, 1, 0]
  want[1] = [0, 2, 0]
  for i in (i0, i1):
    np.testing.assert_array_equal(np.asarray(i.offers), want)
    assert int(i.num_offers) == 2
  np.testing.assert_array_equal(
      np.asarray(env.labels(s, 0)["opp_values"]), [4, 0, 2]
  )
  assert env.world_from_info(i0) is None


def test_info_state_ignores_opponent_values(env):
  """Info state ignores opponent values.

  Two instances with the same pool and seat-0 values, different seat-1
  values: seat 0's InfoState is identical.

  Args:
    env: The module `env` fixture.
  """
  # Reaches private _instances of the code under test.
  inst = np.asarray(env.game._instances)  # pylint: disable=protected-access
  key = {}
  pair = None
  for i, row in enumerate(inst):
    k = (tuple(row[0]), tuple(row[1]))
    if k in key and tuple(inst[key[k]][2]) != tuple(row[2]):
      pair = (key[k], i)
      break
    key.setdefault(k, i)
  assert pair is not None
  a = env.info_state(_raw(env, (pair[0], O(0, 0, 0))), 0)
  b = env.info_state(_raw(env, (pair[1], O(0, 0, 0))), 0)
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_options_spec_and_validation():
  """Options spec and validation."""
  assert registry.game("bargaining").default_kwargs == {
      "max_turns": 10,
      "discount": 1.0,
      "prob_end": 0.0,
  }
  s = registry.spec("bargaining", discount=1, max_turns=4)
  assert s.kwargs == {"discount": 1.0, "max_turns": 4, "prob_end": 0.0}
  env = registry.make(s)
  assert registry.spec_of(env) == s and env.max_len == 4
  assert registry.rules_sha(env) != registry.rules_sha(
      registry.make("bargaining")
  )
  with pytest.raises(ValueError, match="max_turns"):
    registry.spec("bargaining", max_turns=0)
  with pytest.raises(TypeError, match="max_turns"):
    registry.spec("bargaining", max_turns=4.0)
  with pytest.raises(ValueError, match="discount"):
    registry.spec("bargaining", discount=0.0)
  with pytest.raises(ValueError, match="prob_end"):
    registry.spec("bargaining", prob_end=1.0)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
