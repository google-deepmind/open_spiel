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

"""GinEnv's two engines agree on everything the Env exposes.

Random games, both engines stepped with the same chance outcomes and actions,
vmapped as the Arena runs them: at every ply the actor, chance flag, terminal
flag, legal mask, chance probabilities, returns, both seats' info_state and
labels are equal, and so is every engine state field. Play is greedy-biased
(knock when legal, else the least-deadwood
discard 70% of the time) so the knock, layoff and wall phases are reached,
and the test asserts they were. The engine choice: the `engine` argument,
set_default_engine, loud refusal of an unknown name, and an unchanged
EnvSpec and rules identity.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry

tree_map = jax.tree_util.tree_map


def _queries(env, s):
  """Every query of the env at `s`, by name.

  Args:
    env: The env.
    s: The state.

  Returns:
    {query name: value}.
  """
  q = dict(
      actor=env.actor(s),
      chance=env.is_chance(s),
      terminal=env.terminal(s),
      legal=env.legal_mask(s),
      probs=env.chance_probs(s),
      returns=env.returns(s),
      info0=env.info_state(s, 0),
      info1=env.info_state(s, 1),
      labels0=env.labels(s, 0),
      labels1=env.labels(s, 1),
      stage=s.stage,
      first_player=s.first_player,
      events=s.events,
  )
  q["game"] = dict(s.game)
  return q


def _equal(a, b):
  """Whether two pytrees are equal leaf by leaf.

  Args:
    a: A pytree.
    b: The other pytree.

  Returns:
    bool[].
  """
  la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
  assert len(la) == len(lb)
  return jnp.all(jnp.stack([jnp.all(x == y) for x, y in zip(la, lb)]))


def _play(oklahoma_frac, rows, plies, seed):
  """Plays the reference and dense_cache engines in lockstep and compares them.

  Args:
    oklahoma_frac: The Oklahoma fraction.
    rows: The rows.
    plies: The plies.
    seed: The seed.

  Returns:
    (query names, per-ply equality, phases, done flags).
  """
  ca = gin.GinEnv(oklahoma_frac=oklahoma_frac, engine="reference")
  tp = gin.GinEnv(oklahoma_frac=oklahoma_frac, engine="dense_cache")
  names = sorted(_queries(ca, ca.root()))

  def one(sa, sb, k):
    """Steps both engines one ply on one row.

    Args:
      sa: The state under the first engine.
      sb: The state under the second engine.
      k: PRNG key.

    Returns:
      (next sa, next sb, states equal, phase, done).
    """
    qa, qb = _queries(ca, sa), _queries(tp, sb)
    eq = jnp.stack([_equal(qa[n], qb[n]) for n in names])
    k_o, k_a, k_g = jax.random.split(k, 3)
    o = jax.random.categorical(k_o, jnp.log(qa["probs"]))
    legal = qa["legal"]
    noise = jax.random.uniform(k_a, legal.shape)
    rand = jnp.argmax(jnp.where(legal, noise, -1.0))
    dw = sa.game["cached_all_discard_deadwoods"].astype(jnp.float32)
    cards = jnp.arange(cm.NUM_ACTIONS) < cm.NUM_CARDS
    dw_full = jnp.zeros(cm.NUM_ACTIONS).at[: cm.NUM_CARDS].set(dw)
    greedy_discard = jnp.argmin(
        jnp.where(legal & cards, dw_full + 0.01 * noise, 1e9)
    )
    phase = sa.game["phase"]
    a = jnp.where(
        legal[cm.ACTION_KNOCK],
        cm.ACTION_KNOCK,
        jnp.where(
            (phase == cm.PHASE_DISCARD) & (jax.random.uniform(k_g) < 0.7),
            greedy_discard,
            rand,
        ),
    ).astype(jnp.int32)
    ch = qa["chance"]
    na = tree_map(
        lambda x, y: jnp.where(ch, x, y),
        ca.apply_chance(sa, o),
        ca.apply(sa, a),
    )
    nb = tree_map(
        lambda x, y: jnp.where(ch, x, y),
        tp.apply_chance(sb, o),
        tp.apply(sb, a),
    )
    done = qa["terminal"]
    na = tree_map(lambda r, x: jnp.where(done, r, x), ca.root(), na)
    nb = tree_map(lambda r, x: jnp.where(done, r, x), tp.root(), nb)
    return na, nb, eq, phase, done

  def step(carry, k):
    """One ply over every row.

    Args:
      carry: (sa, sb).
      k: PRNG key.

    Returns:
      ((sa, sb), (eq, phase, done)).
    """
    sa, sb = carry
    na, nb, eq, phase, done = jax.vmap(one)(sa, sb, jax.random.split(k, rows))
    return (na, nb), (eq, phase, done)

  def batch(s):
    """Broadcasts a state to the rows.

    Args:
      s: An unbatched state.

    Returns:
      The batched state.
    """
    return tree_map(lambda x: jnp.broadcast_to(x, (rows,) + jnp.shape(x)), s)

  _, (eq, phase, done) = jax.jit(
      lambda k: jax.lax.scan(
          step, (batch(ca.root()), batch(tp.root())), jax.random.split(k, plies)
      )
  )(jax.random.key(seed))
  return names, np.asarray(eq), np.asarray(phase), np.asarray(done)


@pytest.mark.parametrize("oklahoma_frac", [1.0, 0.5])
def test_dense_cache_engine_equals_the_reference_engine_on_random_games(
    oklahoma_frac,
):
  """Dense cache engine equals the reference engine on random games.

  Args:
    oklahoma_frac: The parametrized case.
  """
  names, eq, phase, done = _play(oklahoma_frac, rows=32, plies=700, seed=3)
  bad = {
      n: int((~eq[:, :, i]).sum())
      for i, n in enumerate(names)
      if not eq[:, :, i].all()
  }
  assert not bad, (
      "engine 'dense_cache' differs from 'reference' (plies x rows "
      f"mismatched): {bad}"
  )
  assert int(done.sum()) >= 64, int(done.sum())  # hands completed
  for ph in (cm.PHASE_KNOCK, cm.PHASE_LAYOFF, cm.PHASE_WALL, cm.PHASE_DEAL):
    assert int((phase == ph).sum()) > 0, f"phase {ph} never reached"


def test_engine_choice_is_execution_only(monkeypatch):
  """Engine choice is execution only.

  Args:
    monkeypatch: pytest monkeypatch fixture.
  """
  # Reaches private _DEFAULT_ENGINE of the code under test.
  monkeypatch.setitem(gin._DEFAULT_ENGINE, "name", "reference")  # pylint: disable=protected-access
  assert gin.GinEnv().engine == "reference"
  gin.set_default_engine("dense_cache")
  env = gin.GinEnv(oklahoma_frac=1.0)
  assert env.engine == "dense_cache"
  # the argument wins
  assert gin.GinEnv(engine="reference").engine == "reference"
  ref = gin.GinEnv(oklahoma_frac=1.0, engine="reference")
  assert registry.spec_of(env) == registry.spec_of(ref)
  assert registry.rules_sha(env) == registry.rules_sha(ref)
  assert registry.make("gin_rummy", oklahoma_frac=1.0).engine == "dense_cache"
  with pytest.raises(ValueError, match="fast"):
    gin.set_default_engine("fast")
  with pytest.raises(ValueError, match="fast"):
    gin.GinEnv(engine="fast")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
