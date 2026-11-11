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

"""GinEnv's deal fast path.

A deal card that does not complete the deal changes only the receiving hand, the
deck and the deal count, so it skips the engine step (whose meld-cache
recompute, 216 scenarios plus the 32K-combination knocker melds, is selected
away while the phase stays DEAL). Known answers: the fast card equals
gin_game.step field for field at every such node; apply_chance, batched (vmap:
the Arena's deal and chance resolve) and unbatched, equals the engine-only path
over whole deals and stock draws; the vmapped program keeps a batch-level cond
(the full engine step only when some row needs it).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import game as gin_game
from open_spiel.python.scale.core import chance as chance_lib
from open_spiel.python.scale.envs import gin

tree_map = jax.tree_util.tree_map


def _eq(got, want, msg=""):
  assert set(got) == set(want), set(got) ^ set(want)
  for name in want:
    a, b = np.asarray(got[name]), np.asarray(want[name])
    assert a.dtype == b.dtype and a.shape == b.shape, (
        msg,
        name,
        a.dtype,
        b.dtype,
    )
    np.testing.assert_array_equal(a, b, err_msg=f"{msg} {name}")


@pytest.mark.parametrize("okla", [False, True])
def test_fast_deal_card_equals_the_engine_step(okla):
  """Fast deal card equals the engine step.

  Args:
    okla: The parametrized case.
  """
  n = 0
  for seed in range(16):
    g = gin_game.init_state(oklahoma=okla, **gin.DEFAULT_RULES)
    order = np.asarray(jax.random.permutation(jax.random.key(seed), 52))
    # Reaches private _DEAL_CARDS of the code under test.
    for i in range(gin._DEAL_CARDS - 1):  # pylint: disable=protected-access
      o = jnp.int32(order[i])
      # Reaches private _deal_card_is_fast of the code under test.
      assert bool(gin._deal_card_is_fast(g))  # pylint: disable=protected-access
      _eq(
          # Reaches private _deal_card_fast of the code under test.
          gin._deal_card_fast(g, o),  # pylint: disable=protected-access
          gin_game.step(g, o),
          f"seed {seed} card {i}",
      )
      g = gin_game.step(g, o)
      n += 1
    # the completing card: the engine's
    assert not bool(gin._deal_card_is_fast(g))  # pylint: disable=protected-access
  assert n == 16 * 20


def _engine_only_apply_chance(env, s, o):
  """apply_chance without the fast path.

  The engine step on every engine chance node.

  Args:
    env: The gin env (unused).
    s: The GinState at a chance node.
    o: The chance outcome.

  Returns:
    The next GinState.
  """
  del env  # Unused.
  o = jnp.asarray(o, jnp.int32)
  g_var = dict(s.game)
  g_var["oklahoma"] = o == 1
  s_var = gin.GinState(
      g_var, jnp.int32(gin.STAGE_FIRST_PLAYER), s.first_player, s.events
  )
  s_fp = s._replace(stage=jnp.int32(gin.STAGE_ENGINE), first_player=o)
  in_deal = s.game["prev_player"] < 0
  # Reaches private _DEAL_CARDS of the code under test.
  completes = in_deal & (s.game["deal_cards_dealt"] == gin._DEAL_CARDS - 1)  # pylint: disable=protected-access
  # Reaches private _engine_chance_step of the code under test.
  g_eng = gin._engine_chance_step(s.game, o, completes, s.first_player)  # pylint: disable=protected-access
  s_eng = s._replace(
      game=g_eng, events=gin.record_event(s.events, s.game, o, True)
  )
  # Reaches private _where of the code under test.
  out = gin._where(s.stage == gin.STAGE_FIRST_PLAYER, s_fp, s_eng)  # pylint: disable=protected-access
  # Reaches private _where of the code under test.
  return gin._where(s.stage == gin.STAGE_VARIANT, s_var, out)  # pylint: disable=protected-access


def _resolve(env, apply_chance, s, key, count):
  """Resolves chance nodes from `s` with keys folded from `key`.

  Args:
    env: The env.
    apply_chance: The chance step to use.
    s: The state.
    key: The PRNG key.
    count: The chance counter to start from.

  Returns:
    (the state at a decision or terminal node, the counter).
  """

  def cond(c):
    """Whether the state is still at a live chance node.

    Args:
      c: (state, count).

    Returns:
      bool.
    """
    st, _ = c
    return env.is_chance(st) & ~env.terminal(st)

  def body(c):
    """Applies one sampled chance outcome.

    Args:
      c: (state, count).

    Returns:
      (next state, count + 1).
    """
    st, n = c
    o = chance_lib.sample_outcome(env, st, jax.random.fold_in(key, n))
    return apply_chance(st, o), n + 1

  return jax.lax.while_loop(cond, body, (s, count))


@pytest.mark.parametrize("frac", [1.0, 0.5])
def test_batched_deals_and_draws_equal_the_engine_only_path(frac):
  """Batched deals and draws equal the engine only path.

  Whole deals of 24 rows under vmap (the Arena's _deal), then play with
  uniform actions and resolve every stock draw under vmap, both paths from
  the same keys: equal states at every ply.

  Args:
    frac: The env's oklahoma_frac.
  """
  env = gin.GinEnv(oklahoma_frac=frac)
  b = 24
  keys = jax.random.split(jax.random.key(7), b)

  def ref(s, o):
    """The engine-only chance step.

    Args:
      s: The state.
      o: The chance outcome.

    Returns:
      The state.
    """
    return _engine_only_apply_chance(env, s, o)

  deal = jax.jit(
      jax.vmap(lambda k: _resolve(env, env.apply_chance, env.root(), k, 0))
  )
  deal_ref = jax.jit(jax.vmap(lambda k: _resolve(env, ref, env.root(), k, 0)))
  # Each jitted call returns a (state, count) pair; pylint cannot see it.
  (s, n), (r, nr) = deal(keys), deal_ref(keys)  # pylint: disable=unbalanced-tuple-unpacking
  np.testing.assert_array_equal(np.asarray(n), np.asarray(nr))
  _eq(s.game, r.game, "deal")
  _eq(s.events._asdict(), r.events._asdict(), "deal events")
  step = jax.jit(jax.vmap(env.apply))
  res = jax.jit(
      jax.vmap(lambda st, k, c: _resolve(env, env.apply_chance, st, k, c))
  )
  res_ref = jax.jit(jax.vmap(lambda st, k, c: _resolve(env, ref, st, k, c)))
  rng = np.random.default_rng(0)
  draws = 0
  for ply in range(30):
    legal = np.asarray(jax.vmap(env.legal_mask)(s))
    a = np.array(
        [rng.choice(np.nonzero(m)[0]) if m.any() else 0 for m in legal],
        np.int32,
    )
    # one shared step; then each path resolves the draws
    s = step(s, jnp.asarray(a))
    k = jax.random.split(jax.random.key(1000 + ply), b)
    draws += int(np.asarray(jax.vmap(env.is_chance)(s)).sum())
    (s2, n2), (r2, nr2) = res(s, k, n), res_ref(s, k, n)
    np.testing.assert_array_equal(np.asarray(n2), np.asarray(nr2))
    _eq(s2.game, r2.game, f"ply {ply}")
    s, n = s2, n2
  assert draws > 100, draws


def test_unbatched_apply_chance_equals_the_engine_only_path():
  """Unbatched apply chance equals the engine only path."""
  env = gin.GinEnv(oklahoma_frac=1.0)
  for seed in range(4):
    k = jax.random.key(seed)
    s, _ = jax.jit(
        lambda kk: _resolve(env, env.apply_chance, env.root(), kk, 0)
    )(k)
    r, _ = jax.jit(
        lambda kk: _resolve(
            env,
            lambda a, o: _engine_only_apply_chance(env, a, o),
            env.root(),
            kk,
            0,
        )
    )(k)
    _eq(s.game, r.game, f"seed {seed}")


def test_vmapped_apply_chance_keeps_a_cond():
  """Vmapped apply chance keeps a cond."""
  env = gin.GinEnv(oklahoma_frac=1.0)
  s = tree_map(lambda x: jnp.stack([x] * 4), env.root())
  jaxpr = jax.make_jaxpr(jax.vmap(env.apply_chance))(
      s, jnp.zeros((4,), jnp.int32)
  )
  prims = set()

  def walk(j):
    for e in j.eqns:
      prims.add(e.primitive.name)
      for v in e.params.values():
        for x in v if isinstance(v, (tuple, list)) else (v,):
          if hasattr(x, "eqns"):
            walk(x)
          elif hasattr(getattr(x, "jaxpr", None), "eqns"):
            walk(x.jaxpr)

  walk(jaxpr.jaxpr)
  assert "cond" in prims, sorted(prims)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
