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

"""check_env: the Env contract, for every env author.

Checks, over random playouts driven by uniform-legal actions:
  1. determinism: same deal/chance keys -> identical states and info_states;
  2. chance resolution: init/step never leave the caller at a chance node
     (actor is in [0, P) or the state is terminal, and -1 iff terminal), and
     with chance outcomes, different deal keys give different deals;
  3. a non-terminal state has a non-empty legal mask; a terminal state has an
     empty one;
  4. returns sum to zero at terminal when env.zero_sum;
  5. privacy: hide_fn(env, s, seat, key) rewrites only information hidden from
     `seat`; info_state(s, seat) must not change, and hide_fn must change the
     state at least once (a vacuous hook is an error);
  6. labels are a Mapping on every reachable state and seat; info_state is
     jittable and vmappable; visible_action is an i32 scalar;
  7. the raw layer: chance_probs is f32[max_chance_outcomes], non-negative and
     sums to 1 at every chance node of a raw playout (root/apply/apply_chance),
     is_chance is False on resolved states, and init/step equal the raw layer
     with chance.resolve (event n of a call keyed fold_in(key, n)), i.e.
     replaying a hand's own chance outcomes through apply_chance reproduces it;
  8. world_from_info: None, or a state whose info_state for that seat
     round-trips to the same InfoState.
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import chance
from open_spiel.python.scale.core import types
from open_spiel.python.scale.testing.errors import ContractError
from open_spiel.python.scale.testing.errors import require

tree_map = jax.tree_util.tree_map


def _trees_equal(a, b) -> bool:
  """Whether two pytrees are equal leaf by leaf.

  Args:
    a: A pytree.
    b: The other pytree.

  Returns:
    True when equal.
  """
  la, ta = jax.tree_util.tree_flatten(a)
  lb, tb = jax.tree_util.tree_flatten(b)
  if ta != tb:
    return False
  return all(
      np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)
  )


def _random_legal(mask, key):
  """A uniformly random legal action (any action when none is legal).

  Args:
    mask: The legal mask.
    key: The key.

  Returns:
    The action.
  """
  logits = jnp.where(mask, 0.0, -jnp.inf)
  return jax.random.categorical(key, jnp.where(jnp.any(mask), logits, 0.0))


def _playout(env: types.Env, deal_key, play_key):
  """States [max_len + 1] of one hand (terminal states repeat at the end).

  Args:
    env: The env.
    deal_key: The deal key.
    play_key: The play key.

  Returns:
    The states, leaves [max_len + 1, ...].
  """
  s0 = env.init(deal_key)

  def body(s, k):
    """Plays one random legal action; a no-op at a terminal state.

    Args:
      s: The state.
      k: The ply's key.

    Returns:
      (next state, next state) for `jax.lax.scan`.
    """
    ka, kc = jax.random.split(k)
    term = env.terminal(s)
    a = _random_legal(env.legal_mask(s), ka)
    s2 = env.step(s, a, kc)
    s2 = tree_map(lambda x, y: jnp.where(term, x, y), s, s2)
    return s2, s2

  _, traj = jax.lax.scan(body, s0, jax.random.split(play_key, env.max_len))
  return tree_map(lambda x, y: jnp.concatenate([x[None], y]), s0, traj)


def check_env(
    env: types.Env,
    hide_fn: Optional[Callable[[Any, Any, Any, Any], Any]] = None,
    num_playouts: int = 64,
    seed: int = 0,
) -> None:
  """Raises ContractError on the first broken contract.

  Args:
    env: The env.
    hide_fn: The env's privacy hook, hide_fn(env, state, seat, key), or None
      to skip the privacy check.
    num_playouts: The number of random playouts.
    seed: The seed.

  Raises:
    ContractError: a contract is broken.
  """
  for attr in (
      "name",
      "num_players",
      "num_actions",
      "max_len",
      "max_chance_outcomes",
      "zero_sum",
  ):
    require(hasattr(env, attr), f"env lacks {attr}")
  p, a = env.num_players, env.num_actions
  require(p >= 1 and a >= 1 and env.max_len >= 1, "bad env sizes")

  key = jax.random.key(seed)
  kd, kp, kh = jax.random.split(key, 3)
  deal_keys = jax.random.split(kd, num_playouts)
  play_keys = jax.random.split(kp, num_playouts)
  playout = jax.jit(jax.vmap(lambda d, q: _playout(env, d, q)))

  # 1. Determinism.
  t1 = playout(deal_keys, play_keys)
  t2 = playout(deal_keys, play_keys)
  require(_trees_equal(t1, t2), "env is not deterministic under the same keys")
  # Flatten [N, L+1] states.
  n, l1 = num_playouts, env.max_len + 1
  flat = tree_map(lambda x: x.reshape((n * l1,) + x.shape[2:]), t1)

  actor = np.asarray(jax.vmap(env.actor)(flat))
  term = np.asarray(jax.vmap(env.terminal)(flat))
  legal = np.asarray(jax.vmap(env.legal_mask)(flat))
  rets = np.asarray(jax.vmap(env.returns)(flat))

  require(
      legal.shape == (n * l1, a) and legal.dtype == np.bool_,
      f"legal_mask must be bool[{a}], got {legal.dtype}{legal.shape[1:]}",
  )
  require(
      rets.shape == (n * l1, p) and rets.dtype == np.float32,
      f"returns must be f32[{p}], got {rets.dtype}{rets.shape[1:]}",
  )
  # Every playout ends terminal within max_len decisions.
  last_term = np.asarray(
      jax.vmap(env.terminal)(tree_map(lambda x: x[:, -1], t1))
  )
  require(last_term.all(), f"a hand did not end within max_len={env.max_len}")

  # 2. Chance resolution.
  require(
      ((actor >= 0) & (actor < p) | term).all(),
      "a reachable non-terminal state has actor outside [0, P) "
      "(an unresolved chance node?)",
  )
  require((actor[term] == -1).all(), "actor must be -1 at terminal")
  if env.max_chance_outcomes > 0:
    inits = jax.vmap(env.init)(deal_keys)
    first = tree_map(np.asarray, inits)
    distinct = {
        tuple(
            np.concatenate(
                [np.ravel(l[i]) for l in jax.tree_util.tree_leaves(first)]
            ).tolist()
        )
        for i in range(n)
    }
    require(len(distinct) > 1, "different deal keys never change the deal")

  # 3. Legal masks.
  require(
      legal[~term].any(axis=-1).all(),
      "empty legal mask at a non-terminal state",
  )
  require(not legal[term].any(), "non-empty legal mask at a terminal state")

  # 4. Zero-sum.
  if env.zero_sum:
    sums = rets[term].sum(axis=-1)
    require(
        np.allclose(sums, 0.0, atol=1e-5), f"returns not zero-sum: {sums[:5]}"
    )

  # 6. info_state / labels / visible_action jit + vmap.
  seats = jnp.arange(p, dtype=jnp.int32)
  info_all = jax.jit(
      jax.vmap(lambda s: jax.vmap(lambda q: env.info_state(s, q))(seats))
  )
  info = info_all(flat)
  labels = jax.jit(
      jax.vmap(lambda s: jax.vmap(lambda q: env.labels(s, q))(seats))
  )(flat)
  require(
      hasattr(labels, "items"),
      f"labels must be a Mapping, got {type(labels).__name__}",
  )
  for k, v in labels.items():
    require(
        np.asarray(v).shape[:2] == (n * l1, p),
        f"labels[{k!r}] is not per state and seat",
    )
  va = jax.eval_shape(
      env.visible_action,
      tree_map(lambda x: x[0], flat),
      jnp.int32(0),
      jnp.int32(0),
  )
  require(
      not va.shape and va.dtype == jnp.int32,
      f"visible_action must return an i32 scalar, got {va.dtype}{va.shape}",
  )

  # 7. Raw layer and chance.
  _check_raw_layer(env, flat, term, deal_keys, play_keys)

  # 8. world_from_info round trip (host-side, a few states).
  for i in range(min(n * l1, 16)):
    for q in range(p):
      inf = tree_map(lambda x, i=i, q=q: x[i, q], info)
      w = env.world_from_info(inf)
      if w is None:
        continue
      require(
          _trees_equal(env.info_state(w, jnp.int32(q)), inf),
          f"world_from_info(info) does not round-trip: info_state(world, {q}) "
          "!= info",
      )

  # 5. Privacy.
  if hide_fn is None:
    return
  hkeys = jax.random.split(kh, n * l1 * p).reshape((n * l1, p) + kh.shape)
  hidden = jax.jit(
      jax.vmap(
          lambda s, ks: jax.vmap(lambda q, k: hide_fn(env, s, q, k))(seats, ks)
      )
  )(
      flat, hkeys
  )  # [N*L, P] states
  info_h = jax.jit(jax.vmap(jax.vmap(env.info_state)))(
      hidden, jnp.broadcast_to(seats, (n * l1, p))
  )
  changed = False
  for q in range(p):
    s_q = tree_map(lambda x, q=q: x[:, q], hidden)
    if not _trees_equal(s_q, flat):
      changed = True
  require(
      changed, "hide_fn never changed a state: the privacy check is vacuous"
  )
  for q in range(p):
    a_q = tree_map(lambda x, q=q: np.asarray(x[:, q]), info)
    b_q = tree_map(lambda x, q=q: np.asarray(x[:, q]), info_h)
    if not _trees_equal(a_q, b_q):
      raise ContractError(
          f"info_state(seat={q}) changed when information hidden from seat {q} "
          "changed: the InfoState leaks hidden information"
      )


def _check_raw_layer(env: types.Env, flat, term, deal_keys, play_keys) -> None:
  """Checks the raw layer: init, step and chance_probs.

  Args:
    env: The env.
    flat: The states, flattened.
    term: Unused.
    deal_keys: The deal PRNG keys.
    play_keys: The play PRNG keys.
  """
  del term
  m = env.max_chance_outcomes
  require(
      not np.asarray(jax.vmap(env.is_chance)(flat)).any(),
      "is_chance is True on a state returned by init/step",
  )
  # init == root + chance.resolve; step == apply + chance.resolve.
  init_ref = jax.jit(jax.vmap(lambda k: chance.resolve(env, env.root(), k)[0]))(
      deal_keys
  )
  require(
      _trees_equal(jax.jit(jax.vmap(env.init))(deal_keys), init_ref),
      "init differs from root() + chance.resolve (replaying the deal's chance "
      "outcomes through apply_chance does not reproduce it)",
  )
  s0 = jax.vmap(env.init)(deal_keys)
  mask = jax.vmap(env.legal_mask)(s0)
  a = jax.vmap(_random_legal)(mask, play_keys).astype(jnp.int32)
  step_ref = jax.jit(
      jax.vmap(lambda s, x, k: chance.resolve(env, env.apply(s, x), k)[0])
  )(s0, a, play_keys)
  require(
      _trees_equal(jax.jit(jax.vmap(env.step))(s0, a, play_keys), step_ref),
      "step differs from apply() + chance.resolve",
  )
  if m == 0:
    require(
        not bool(env.is_chance(env.root())),
        "max_chance_outcomes == 0 but root is chance",
    )
    return

  # Raw playouts: record chance_probs at every chance node.
  def raw_playout(key):
    """Plays one raw playout, chance nodes included.

    Args:
      key: The PRNG key.

    Returns:
      (is chance per ply, chance probabilities per ply).
    """

    def body(carry, k):
      """Plays one raw ply: a chance outcome or a random legal action.

      Args:
        carry: (state,).
        k: The ply's key.

      Returns:
        ((next state,), (is chance, chance probabilities)).
      """
      (s,) = carry
      kc, ka = jax.random.split(k)
      is_c = env.is_chance(s)
      probs = env.chance_probs(s)
      o = chance.sample_outcome(env, s, kc)
      act = _random_legal(env.legal_mask(s), ka).astype(jnp.int32)
      s_c = env.apply_chance(s, o)
      s_a = env.apply(s, act)
      nxt = tree_map(lambda x, y: jnp.where(is_c, x, y), s_c, s_a)
      nxt = tree_map(lambda x, y: jnp.where(env.terminal(s), x, y), s, nxt)
      return (nxt,), (is_c, probs)

    steps = 4 * (env.max_len + 8)
    _, (is_c, probs) = jax.lax.scan(
        body, (env.root(),), jax.random.split(key, steps)
    )
    return is_c, probs

  is_c, probs = jax.jit(jax.vmap(raw_playout))(deal_keys)
  is_c, probs = np.asarray(is_c), np.asarray(probs)
  require(
      probs.dtype == np.float32 and probs.shape[-1] == m,
      f"chance_probs must be f32[{m}], got {probs.dtype}{probs.shape[2:]}",
  )
  require(
      is_c.any(), "max_chance_outcomes > 0 but a raw playout met no chance node"
  )
  pc = probs[is_c]
  require((pc >= 0).all(), "negative chance probability")
  require(
      np.allclose(pc.sum(-1), 1.0, atol=1e-5),
      f"chance_probs do not sum to 1 at a chance node: {pc.sum(-1)[:5]}",
  )
