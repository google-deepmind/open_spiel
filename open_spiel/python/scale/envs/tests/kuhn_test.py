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

"""KuhnEnv: the Env contract and known answers."""

import collections
import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.testing import check_env

PASS, BET = 0, 1
J, Q, K = 0, 1, 2


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, hide_fn=kuhn.hide_opponent_card, num_playouts=128)


def _play(env, s, actions):
  """Steps `actions` from `s`.

  Args:
    env: The env.
    s: The state.
    actions: The actions.

  Returns:
    The state.
  """
  for a in actions:
    assert not bool(env.terminal(s))
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


@pytest.mark.parametrize(
    "cards,actions,expected",
    [
        ((K, J), (PASS, PASS), [1.0, -1.0]),
        ((J, K), (PASS, PASS), [-1.0, 1.0]),
        ((K, Q), (BET, BET), [2.0, -2.0]),
        ((Q, K), (BET, BET), [-2.0, 2.0]),
        ((J, K), (BET, PASS), [1.0, -1.0]),
        ((K, J), (PASS, BET, PASS), [-1.0, 1.0]),
        ((Q, J), (PASS, BET, BET), [2.0, -2.0]),
    ],
)
def test_known_terminal_returns(env, cards, actions, expected):
  """Known terminal returns.

  Args:
    env: The module `env` fixture.
    cards: The parametrized case.
    actions: The parametrized case.
    expected: The parametrized case.
  """
  s = _play(env, env.deal(*cards), actions)
  assert bool(env.terminal(s))
  assert int(env.actor(s)) == -1
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)


def test_actor_and_legal(env):
  """Actor and legal.

  Args:
    env: The module `env` fixture.
  """
  s = env.deal(K, J)
  assert int(env.actor(s)) == 0
  np.testing.assert_array_equal(np.asarray(env.legal_mask(s)), [True, True])
  s = env.step(s, jnp.int32(PASS), jax.random.key(0))
  assert int(env.actor(s)) == 1


def test_raw_layer_replays_a_known_deal(env):
  """Raw layer replays a known deal.

  Root is a chance node; apply_chance with the outcomes K, J gives deal(K, J).

  Args:
    env: The module `env` fixture.
  """
  s = env.root()
  assert bool(env.is_chance(s))
  np.testing.assert_allclose(
      np.asarray(env.chance_probs(s)), [1 / 3] * 3, atol=1e-6
  )
  assert env.chance_probs(s).dtype == jnp.float32
  s = env.apply_chance(env.apply_chance(s, K), J)
  assert not bool(env.is_chance(s))
  ref = env.deal(K, J)
  for x, y in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(ref)):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  assert int(env.visible_action(s, jnp.int32(BET), 1)) == BET


def test_deterministic_deal_from_key(env):
  """Deterministic deal from key.

  Args:
    env: The module `env` fixture.
  """
  s1 = env.init(jax.random.key(7))
  s2 = env.init(jax.random.key(7))
  assert int(s1.p0_card) == int(s2.p0_card) and int(s1.p1_card) == int(
      s2.p1_card
  )
  assert int(env.actor(s1)) == 0


def test_deal_distribution_uniform_over_six(env):
  """Deal distribution uniform over six.

  Args:
    env: The module `env` fixture.
  """
  n = 12000
  states = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.key(3), n))
  deals = collections.Counter(
      zip(
          np.asarray(states.p0_card).tolist(),
          np.asarray(states.p1_card).tolist(),
      )
  )
  assert set(deals) == set(itertools.permutations(range(3), 2))
  for c in deals.values():
    assert abs(c / n - 1 / 6) < 4 * np.sqrt((1 / 6) * (5 / 6) / n)


def test_info_state_is_own_card_and_public_history(env):
  """Info state is own card and public history.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, env.deal(Q, K), (PASS, BET))
  i0, i1 = env.info_state(s, 0), env.info_state(s, 1)
  assert int(i0.card) == Q and int(i1.card) == K
  np.testing.assert_array_equal(np.asarray(i0.bets), [PASS, BET, -1])
  assert int(i0.num_bets) == 2
  assert int(env.labels(s, 0)["opp_card"]) == K
  assert env.world_from_info(i0) is None


def test_info_index_is_a_bijection_on_decision_nodes(env):
  """Info index is a bijection on decision nodes.

  Args:
    env: The module `env` fixture.
  """
  seen = set()
  for c0 in range(3):
    for c1 in range(3):
      if c0 == c1:
        continue
      for hist in ((), (PASS,), (BET,), (PASS, BET)):
        s = _play(env, env.deal(c0, c1), hist)
        seat = int(env.actor(s))
        idx = int(kuhn.info_index(env.info_state(s, seat)))
        seen.add((idx, (c0, c1)[seat], hist))
  by_idx = collections.defaultdict(set)
  for idx, card, hist in seen:
    by_idx[idx].add((card, hist))
  assert set(by_idx) == set(range(kuhn.NUM_INFO_STATES))
  assert all(len(v) == 1 for v in by_idx.values())
  assert types.NO_ACTION == -1


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
