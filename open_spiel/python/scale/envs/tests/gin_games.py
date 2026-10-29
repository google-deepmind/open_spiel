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

"""Seeded gin games for the gin tests: raw-layer playouts, mixed behaviour.

Every hand is played from `GinEnv.root()` through the raw layer: chance nodes
(the variant and first-player nodes, the 21 deal cards, every stock draw) are
sampled from `chance_probs`; player nodes by a per-game mix chosen so the
games cover knocks, gin, undercuts and wall hands:
  * strategic phases (FIRST_UPCARD / DRAW / DISCARD): the rule bot, or a
    uniformly random legal action with probability eps (per game, from EPS);
  * endgame phases (KNOCK / LAYOFF / WALL): the scripted endgame, or a random
    legal action with probability END_RAND (per game), so the endgame tests
    meet layoff and meld states the script alone never reaches;
  * in half of the games (NO_KNOCK) a DISCARD-phase knock is replaced by a
    random legal discard, so more hands reach the wall (wall knocks, draws)
    and knock late (undercuts).

Test infrastructure only (it composes components that have their own tests).
"""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import chance
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot

tree_map = jax.tree_util.tree_map

# 2 meta chance nodes + 21 deal cards + MAX_LEN decisions + 29 stock draws.
RAW_STEPS = 2 + 21 + gin.MAX_LEN + 29
EPS = (0.0, 0.05, 0.3, 1.0)
END_RAND = (0.0, 0.5)
NO_KNOCK = (False, True)


class Games(NamedTuple):
  """Seeded gin games, every ply.

  Attributes:
    states: [N, T] the state BEFORE raw step t.
    is_chance: bool[N, T].
    live: bool[N, T]: the state is not terminal (step t happens).
    action: i32[N, T]: the player action or the chance outcome.
    final: [N] terminal states.
  """

  states: gin.GinState
  is_chance: jax.Array
  live: jax.Array
  action: jax.Array
  final: gin.GinState


def _game_params(n):
  """The behaviour parameters of game i, cycling through the grids.

  Args:
    n: The games.

  Returns:
    (eps, end_rand, no_knock), each [n].
  """
  i = np.arange(n)
  eps = np.asarray(EPS, np.float32)[i % len(EPS)]
  end = np.asarray(END_RAND, np.float32)[(i // len(EPS)) % len(END_RAND)]
  no_knock = np.asarray(NO_KNOCK)[
      (i // (len(EPS) * len(END_RAND))) % len(NO_KNOCK)
  ]
  return jnp.asarray(eps), jnp.asarray(end), jnp.asarray(no_knock)


def _behaviour(env, s, key, eps, end_rand, no_knock):
  """i32[N] actions at player nodes (garbage at chance/terminal rows).

  Args:
    env: The env.
    s: The states.
    key: The PRNG key.
    eps: Random-move rates.
    end_rand: Random-move rates in the endgame.
    no_knock: Whether knocking is skipped.

  Returns:
    i32[N], the actions.
  """
  actor = jax.vmap(env.actor)(s)
  seat = jnp.clip(actor, 0, 1).astype(jnp.int32)
  info = jax.vmap(env.info_state)(s, seat)
  legal = jax.vmap(env.legal_mask)(s)
  k_rand, k_mix, k_card = jax.random.split(key, 3)
  any_legal = jnp.any(legal, axis=-1, keepdims=True)
  logits = jnp.where(legal | ~any_legal, jnp.float32(0.0), -jnp.inf)
  rand = jax.random.categorical(k_rand, logits, axis=-1).astype(jnp.int32)
  card_legal = legal & (jnp.arange(legal.shape[-1]) < 52)[None, :]
  any_card = jnp.any(card_legal, axis=-1, keepdims=True)
  rand_card = jax.random.categorical(
      k_card,
      jnp.where(card_legal | ~any_card, jnp.float32(0.0), -jnp.inf),
      axis=-1,
  ).astype(jnp.int32)
  bot = gin_rule_bot.actions(info)
  end = gin_endgame.actions(info)
  is_end = gin.is_endgame_phase(info.phase)
  u = jax.random.uniform(k_mix, actor.shape, dtype=jnp.float32)
  use_rand = u < jnp.where(is_end, end_rand, eps)
  a = jnp.where(use_rand, rand, jnp.where(is_end, end, bot))
  skip_knock = no_knock & (info.phase == 3) & (a == 55)  # DISCARD-phase KNOCK
  return jnp.where(skip_knock, rand_card, a).astype(jnp.int32)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _play(env, n, key):
  """Plays `n` games of `env` with the mixed behaviour policy.

  Args:
    env: the gin Env.
    n: the number of games.
    key: a PRNG key.

  Returns:
    The Games, with the per-ply traces [N, T].
  """
  eps, end_rand, no_knock = _game_params(n)
  s0 = tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), env.root())

  def body(s, k):
    """One ply of every game.

    Args:
      s: The states.
      k: PRNG key.

    Returns:
      (next states, (states, is chance, live, action)).
    """
    k_chance, k_play = jax.random.split(k)
    term = jax.vmap(env.terminal)(s)
    is_c = jax.vmap(env.is_chance)(s)
    outcome = jax.vmap(lambda st, kk: chance.sample_outcome(env, st, kk))(
        s, jax.random.split(k_chance, n)
    )
    act = jnp.where(
        is_c, outcome, _behaviour(env, s, k_play, eps, end_rand, no_knock)
    )
    s_c = jax.vmap(env.apply_chance)(s, act)
    s_a = jax.vmap(env.apply)(s, act)
    nxt = tree_map(
        lambda c, a: jnp.where(
            is_c.reshape(is_c.shape + (1,) * (c.ndim - 1)), c, a
        ),
        s_c,
        s_a,
    )
    nxt = tree_map(
        lambda old, new: jnp.where(
            term.reshape(term.shape + (1,) * (old.ndim - 1)), old, new
        ),
        s,
        nxt,
    )
    return nxt, (s, is_c, ~term, act)

  final, (st, is_c, live, act) = jax.lax.scan(
      body, s0, jax.random.split(key, RAW_STEPS)
  )

  def swap(x):
    """Swaps the first two axes.

    Args:
      x: A leaf.

    Returns:
      The swapped leaf.
    """
    return jnp.swapaxes(x, 0, 1)

  return Games(tree_map(swap, st), swap(is_c), swap(live), swap(act), final)


@functools.cache
def games(n: int, seed: int, oklahoma_frac=1.0) -> Games:
  """`n` seeded games (cached per process); raises if a hand did not finish.

  Args:
    n: Number of games.
    seed: The seed.
    oklahoma_frac: The Oklahoma deal fraction.

  Returns:
    The Games.
  """
  env = gin.GinEnv(oklahoma_frac=oklahoma_frac)
  out = _play(env, n, jax.random.key(seed))
  done = np.asarray(jax.vmap(env.terminal)(out.final))
  assert (
      done.all()
  ), f"{(~done).sum()} of {n} games unfinished after {RAW_STEPS} raw steps"
  return out


def chunks(n: int, seed: int, size: int):
  """games(n, seed) in chunks of `size` games: [(chunk Games), ...].

  Args:
    n: Number of games.
    seed: The seed.
    size: Games per chunk.

  Returns:
    The chunks.
  """
  g = games(n, seed)

  def chunk(i):
    """The chunk of games starting at game `i`.

    Args:
      i: The first game.

    Returns:
      The chunk Games.
    """
    return tree_map(lambda x: x[i : i + size], g)

  return [chunk(i) for i in range(0, n, size)]


def flat_states(g: Games, mask):
  """States where mask[N, T] holds, flattened to [M].

  Args:
    g: The Games.
    mask: bool[N, T] the plies to keep.

  Returns:
    The flattened states.
  """
  idx = np.nonzero(np.asarray(mask))
  return tree_map(lambda x: x[idx], g.states)


def decision_mask(env, g: Games):
  """bool[N, T]: live player decision nodes.

  Args:
    env: Unused.
    g: The Games.

  Returns:
    The mask.
  """
  del env  # Unused.
  return np.asarray(g.live) & ~np.asarray(g.is_chance)


def outcome_classes(final_game: dict[str, Any]) -> dict[str, Any]:
  """Counts of hand outcomes over terminal engine states (batched dict).

  Args:
    final_game: The terminal engine states (batched dict).

  Returns:
    Counts per outcome class.
  """
  f = {k: np.asarray(v) for k, v in final_game.items()}
  knocked = f["knocker"] >= 0
  scored = knocked & (f["p0_score"] != 0)
  knocker_won = np.where(
      f["knocker"] == 0, f["p0_score"] > 0, f["p0_score"] < 0
  )
  gin_hand = knocked & (f["knocker_deadwood"] == 0)
  wall_knock = knocked & (f["deck"].sum(-1) <= 2)
  return {
      "knock_win": int((scored & knocker_won & ~gin_hand).sum()),
      "gin": int(gin_hand.sum()),
      "undercut": int((knocked & ~knocker_won & ~gin_hand).sum()),
      "wall_draw": int((~knocked & (f["deck"].sum(-1) <= 2)).sum()),
      "wall_knock": int(wall_knock.sum()),
      "draw_other": int((~knocked & (f["deck"].sum(-1) > 2)).sum()),
  }
