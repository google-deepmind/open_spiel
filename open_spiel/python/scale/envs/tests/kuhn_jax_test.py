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

"""KuhnJaxEnv, the built-in game.

The Env contract, known payoffs and, where the JAX games are installed, equality
with the JAX-engine adapter on every state.
"""

import ast
import inspect
import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps
from open_spiel.python.scale.envs import kuhn_jax
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.testing import check_env

PASS, BET = 0, 1
J, Q, K = 0, 1, 2
HISTORIES = (
    (PASS, PASS),
    (BET, BET),
    (BET, PASS),
    (PASS, BET, PASS),
    (PASS, BET, BET),
)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A JAX Kuhn env.

  Returns:
    The env.
  """
  return kuhn_jax.KuhnJaxEnv()


def test_check_env_passes(env):
  """Check env passes.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(
      env, hide_fn=kuhn_jax.hide_opponent_card, num_playouts=128
  )


def _play(env, s, actions):
  """Applies `actions` from `s`.

  Args:
    env: The env.
    s: The state.
    actions: The actions.

  Returns:
    The state.
  """
  for a in actions:
    assert not bool(env.terminal(s))
    s = env.apply(s, jnp.int32(a))
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
        ((J, Q), (PASS, BET, BET), [-2.0, 2.0]),
    ],
)
def test_known_payoffs(env, cards, actions, expected):
  """Known payoffs.

  Args:
    env: The module `env` fixture.
    cards: The parametrized case.
    actions: The parametrized case.
    expected: The parametrized case.
  """
  s = _play(env, env.deal(*cards), actions)
  assert bool(env.terminal(s)) and int(env.actor(s)) == -1
  np.testing.assert_array_equal(
      np.asarray(env.returns(s)), np.asarray(expected, np.float32)
  )
  assert env.returns(s).dtype == jnp.float32


def test_chance_layout_and_seats(env):
  """Chance layout and seats.

  Args:
    env: The module `env` fixture.
  """
  s = env.root()
  assert bool(env.is_chance(s))
  np.testing.assert_allclose(env.chance_probs(s), [1 / 3] * 3)
  s = env.apply_chance(s, Q)
  np.testing.assert_allclose(env.chance_probs(s), [0.5, 0.0, 0.5])
  s = env.apply_chance(s, K)
  assert not bool(env.is_chance(s)) and int(env.actor(s)) == 0
  assert int(env.actor(env.apply(s, PASS))) == 1
  assert int(env.actor(_play(env, s, (PASS, BET)))) == 0
  info = env.info_state(s, 1)
  assert int(info.card) == K and int(kuhn_jax.info_index(info)) == K * 4


def test_the_game_dependency_free_module_imports_no_game_package():
  """The game dependency free module imports no game package."""
  tree = ast.parse(inspect.getsource(kuhn_jax))
  mods = {
      n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
  }
  for n in ast.walk(tree):
    if isinstance(n, ast.Import):
      mods.update(a.name for a in n.names)
  allowed = ("__future__", "typing", "jax", subprocess_env.PACKAGE)
  bad = sorted(
      m
      for m in mods
      if not any(m == a or m.startswith(a + ".") for a in allowed)
  )
  assert not bad, bad


def test_registered_as_its_own_game(env):
  """Registered as its own game.

  Args:
    env: The module `env` fixture.
  """
  del env  # Unused.
  made = registry.make("kuhn_poker_jax")
  assert (
      isinstance(made, kuhn_jax.KuhnJaxEnv)
      and registry.spec_of(made).name == "kuhn_poker_jax"
  )
  assert len(registry.rules_sha(made)) == 64


def test_equal_to_the_engine_adapter_on_every_state(env):
  """Equal to the engine adapter on every state.

  Args:
    env: The module `env` fixture.
  """
  pytest.importorskip(deps.GAMES_PACKAGE)
  # Lazy: kuhn needs the optional JAX games package, skipped above if absent.
  from open_spiel.python.scale.envs import kuhn  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  ref = kuhn.KuhnEnv()
  for c0, c1 in itertools.permutations(range(3), 2):
    a, b = env.deal(c0, c1), ref.deal(c0, c1)
    for hist in HISTORIES:
      for n in range(len(hist) + 1):
        sa, sb = _play(env, a, hist[:n]), _play(ref, b, hist[:n])
        for f in ("terminal", "actor", "legal_mask", "returns"):
          np.testing.assert_array_equal(
              np.asarray(getattr(env, f)(sa)),
              np.asarray(getattr(ref, f)(sb)),
              err_msg=f"{f} {hist[:n]}",
          )
        for seat in (0, 1):
          for x, y in zip(env.info_state(sa, seat), ref.info_state(sb, seat)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  sa, sb = env.root(), ref.root()
  np.testing.assert_allclose(env.chance_probs(sa), ref.chance_probs(sb))
  np.testing.assert_allclose(
      env.chance_probs(env.apply_chance(sa, 1)),
      ref.chance_probs(ref.apply_chance(sb, 1)),
  )
  k = jax.random.key(7)
  for x, y in zip(env.init(k), ref.init(k)[:2]):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
