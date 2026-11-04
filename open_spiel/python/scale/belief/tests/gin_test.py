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

"""Tests for GinBeliefSampler.

Exact counts of consistent worlds on small reachable
positions, uniformity, exactness of the world rebuilt with the true hidden
cards, consistency of every sampled world with the InfoState, uniform
marginals over the unseen cards, known opponent cards always placed,
privacy (permuting hidden information leaves the output unchanged), dtypes
and determinism.
"""

import collections
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.belief.tests import gin_positions
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.gin import belief as gin_belief
from open_spiel.python.scale.envs.tests import gin_games

tree_map = jax.tree_util.tree_map


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A gin env.

  Returns:
    The env.
  """
  return gin.GinEnv()


@pytest.fixture(name="sampler", scope="module")
def _sampler_fixture(env):
  """The gin belief sampler of `env`.

  Args:
    env: The gin env.

  Returns:
    The sampler.
  """
  return gin_belief.GinBeliefSampler(env)


def _info(env, s, seat):
  """The InfoState of `seat` at `s`.

  Args:
    env: The env.
    s: The state.
    seat: The seat.

  Returns:
    The InfoState.
  """
  return jax.jit(env.info_state)(s, jnp.int32(seat))


def _b(x):
  """Adds a leading batch axis of 1.

  Args:
    x: A pytree.

  Returns:
    The batched tree.
  """
  return tree_map(lambda a: jnp.asarray(a)[None], x)


def _det(sampler, info1, k, seed):
  """sample_determinized over one InfoState, leaves [K, ...] (numpy).

  Args:
    sampler: The sampler.
    info1: One unbatched InfoState.
    k: The worlds to sample.
    seed: The seed.

  Returns:
    The k worlds, numpy leaves [K, ...].
  """
  fn = jax.jit(lambda i, key: sampler.sample_determinized({}, (), i, k, key))
  w = fn(_b(info1), jax.random.key(seed))
  return tree_map(lambda x: np.asarray(x)[0], w)


def _true_hidden(env, s, seat):
  """The opponent cards `seat` cannot see at `s`.

  Args:
    env: Unused.
    s: The state.
    seat: The seat.

  Returns:
    bool[52], the hidden cards.
  """
  del env  # Unused.
  g = s.game
  opp_hand = np.asarray(g["player1_hand" if seat == 0 else "player0_hand"]) > 0
  known = np.asarray(g["known_cards"])[1 - seat] > 0
  return opp_hand & ~known


def _true_labels(s, seat):
  """The cards the opponent drew from the stock, by event row.

  Args:
    s: The state.
    seat: The seat.

  Returns:
    int8[EVENTS_MAX], 0 elsewhere.
  """
  ev = s.events
  opp_rows = (np.asarray(ev.actor) == 1 - seat) & (
      np.asarray(ev.type) == gin.EVT_DRAW_STOCK
  )
  opp_rows &= np.arange(gin.EVENTS_MAX) < int(ev.count)
  return np.where(opp_rows, np.asarray(ev.private_card), 0).astype(np.int8)


def _assert_tree_equal(a, b, what):
  """Asserts two pytrees are equal leaf by leaf, dtypes included.

  Args:
    a: A pytree.
    b: The other pytree.
    what: The failure message prefix.
  """
  la, ta = jax.tree_util.tree_flatten_with_path(a)
  lb, tb = jax.tree_util.tree_flatten_with_path(b)
  assert ta == tb, what
  for (path, x), (_, y) in zip(la, lb):
    x, y = np.asarray(x), np.asarray(y)
    assert x.dtype == y.dtype, (
        what,
        jax.tree_util.keystr(path),
        x.dtype,
        y.dtype,
    )
    np.testing.assert_array_equal(
        x, y, err_msg=f"{what} {jax.tree_util.keystr(path)}"
    )


# -- known answers: tiny reachable positions -----------------------------------


def test_case_a_exact_count_of_consistent_worlds(env, sampler):
  """Case a exact count of consistent worlds.

  Opponent hand fully known except its two late stock draws, 4 cards in
  the stock: |U| = 6, h = 2. Worlds = C(6,2) hands x 4! stock orders x 2
  labellings of its two stock draws = 720, each sampled about equally often.

  Args:
    env: the env fixture.
    sampler: the sampler fixture.
  """
  s, _ = gin_positions.case_a(env)
  info = _info(env, s, 0)
  assert int(np.asarray(gin_belief.unseen(info)).sum()) == 6
  assert int(gin_belief.hidden_count(info)) == 2
  assert int(info.stock_count) == 4
  assert gin_belief.count_worlds(info) == {
      "hands": 15,
      "stock_orders": 24,
      "opp_draw_labels": 2,
      "total": 720,
  }
  k = 20000
  w = _det(sampler, info, k, seed=3)
  orders = np.asarray(
      jax.jit(jax.vmap(gin_belief.stock_order))(
          jnp.asarray(w.states.game["deck"]), jnp.asarray(w.chance_scores)
      )
  )[:, :4]
  assert (orders >= 0).all()
  opp = np.asarray(w.states.game["player1_hand"]) > 0
  rows = _true_labels(s, 0) > 0  # the opponent's two stock-draw rows
  labels = np.asarray(w.states.events.private_card)[:, rows]
  seen = collections.Counter(
      (tuple(np.flatnonzero(opp[i])), tuple(orders[i]), tuple(labels[i]))
      for i in range(k)
  )
  assert len(seen) == 720, len(seen)
  counts = np.asarray(list(seen.values()), np.float64)
  expected = k / 720
  chi2 = float(((counts - expected) ** 2 / expected).sum())
  # 719 dof: mean 719, sd ~37.9; 5 sd
  assert chi2 < 719 + 5 * math.sqrt(2 * 719), chi2


def test_case_c_draw_labellings_known_answer(env):
  """Case c draw labellings known answer.

  Two late stock draws, a dealt card shed between them, h = 3: the newest
  draw's card is one of the 3 hidden cards, the older one's one of the 3 left
  of (hidden + the shed card): 3 x 3 = 9 labellings, uniform, all consistent.

  Args:
    env: the env fixture.
  """
  s, _ = gin_positions.case_c(env)
  info = _info(env, s, 0)
  assert int(gin_belief.hidden_count(info)) == 3
  c = gin_belief.count_worlds(info)
  assert c["opp_draw_labels"] == 9, c
  assert c["hands"] == math.comb(7, 3) and c["stock_orders"] == math.factorial(
      4
  )
  hidden = jnp.asarray(_true_hidden(env, s, 0))
  fn = jax.jit(
      jax.vmap(lambda kk: gin_belief.sample_opp_labels(info, hidden, kk))
  )
  labels = np.asarray(fn(jax.random.split(jax.random.key(0), 9000)))
  rows = _true_labels(s, 0) > 0
  assert rows.sum() == 2
  got = collections.Counter(tuple(r[rows]) for r in labels)
  assert len(got) == 9, got
  # the real labelling is one of them
  assert tuple(_true_labels(s, 0)[rows]) in got
  assert min(got.values()) > 9000 / 9 * 0.85, got


def test_known_opponent_cards_always_placed(env, sampler):
  """Picked-up discards the opponent still holds are in its hand everywhere."""
  s, _ = gin_positions.case_a(env)
  info = _info(env, s, 0)
  known = np.asarray(info.known_cards)[1] > 0
  assert known.sum() == 8
  w = _det(sampler, info, 256, seed=1)
  opp = np.asarray(w.states.game["player1_hand"]) > 0
  assert (opp[:, known]).all()
  assert (opp.sum(-1) == 10).all()


# -- exactness and consistency on real games -----------------------------------


def _domain_states(n_games=24, seed=7, per_game=6):
  """Random gin decision states drawn from simulated games.

  Args:
    n_games: Games simulated.
    seed: The seed.
    per_game: States drawn per game at most.

  Returns:
    (env, states).
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  g = gin_games.games(n_games, seed)
  mask = gin_games.decision_mask(env, g)
  phase = np.asarray(g.states.game["phase"])
  mask &= (
      (phase == cm.PHASE_FIRST_UPCARD)
      | (phase == cm.PHASE_DRAW)
      | (phase == cm.PHASE_DISCARD)
  )
  rng = np.random.default_rng(seed)
  picks = []
  for i in range(n_games):
    ts = np.flatnonzero(mask[i])
    for t in rng.choice(ts, size=min(per_game, len(ts)), replace=False):
      picks.append((i, int(t)))
  idx = tuple(np.asarray(picks).T)
  return env, tree_map(lambda x: x[idx], g.states)


def test_world_with_the_true_hidden_cards_is_the_real_state():
  """World with the true hidden cards is the real state.

  assemble(info, true hidden hand, true draw cards) == the real state, leaf
  for leaf, for both seats (the searcher to move or not).
  """
  env, states = _domain_states()
  template = env.root().game
  n = int(states.stage.shape[0])
  for seat in (0, 1):
    infos = jax.vmap(env.info_state, in_axes=(0, None))(states, jnp.int32(seat))
    hidden = np.stack([
        _true_hidden(env, tree_map(lambda x, i=i: x[i], states), seat)
        for i in range(n)
    ])
    labels = np.stack([
        _true_labels(tree_map(lambda x, i=i: x[i], states), seat)
        for i in range(n)
    ])
    rebuilt = jax.jit(
        jax.vmap(lambda i, h, l: gin_belief.assemble(i, h, l, template))
    )(infos, jnp.asarray(hidden), jnp.asarray(labels))
    _assert_tree_equal(rebuilt, states, f"seat {seat}")


def test_every_sampled_world_is_consistent_with_the_infostate(sampler):
  """Every sampled world is consistent with the infostate.

  Each world: the seat's InfoState is exactly the input; the seat's legal
  actions (where it is to move) are the real ones; the opponent's view is a
  real one (its stock cards are in its hand or shed later; its hand has the
  real size).

  Args:
    sampler: the sampler fixture.
  """
  env, states = _domain_states(n_games=8, per_game=4)
  for seat in (0, 1):
    infos = jax.vmap(env.info_state, in_axes=(0, None))(states, jnp.int32(seat))
    k = 16
    w = jax.jit(
        lambda i, key, k=k: sampler.sample_determinized({}, (), i, k, key)
    )(infos, jax.random.key(seat))
    n = int(states.stage.shape[0])
    flat = tree_map(
        lambda x, n=n, k=k: x.reshape((n * k,) + x.shape[2:]), w.states
    )
    rep = tree_map(lambda x, k=k: jnp.repeat(x, k, axis=0), infos)
    again = jax.vmap(env.info_state, in_axes=(0, None))(flat, jnp.int32(seat))
    _assert_tree_equal(again, rep, f"info seat {seat}")
    mine = np.repeat(np.asarray(states.game["current_player"]) == seat, k)
    legal_w = np.asarray(jax.vmap(env.legal_mask)(flat))
    legal_r = np.asarray(
        jnp.repeat(jax.vmap(env.legal_mask)(states), k, axis=0)
    )
    np.testing.assert_array_equal(legal_w[mine], legal_r[mine])
    opp_key = "player1_hand" if seat == 0 else "player0_hand"
    real_sizes = np.repeat(np.asarray(states.game[opp_key]).sum(-1), k)
    np.testing.assert_array_equal(
        np.asarray(flat.game[opp_key]).sum(-1), real_sizes
    )
    # Stock and chance scores: finite exactly on the stock, a permutation's
    # values.
    sc = np.asarray(w.chance_scores).reshape(n * k, -1)
    deck = np.asarray(flat.game["deck"]) > 0
    np.testing.assert_array_equal(np.isfinite(sc), deck)
    # The opponent's stock-draw cards: in its hand now or shed by it later.
    ev = jax.tree_util.tree_map(np.asarray, flat.events)
    for r in range(0, n * k, 7):
      cnt = int(ev.count[r])
      for i in range(cnt):
        if ev.actor[r, i] == 1 - seat and ev.type[r, i] == gin.EVT_DRAW_STOCK:
          c = int(ev.private_card[r, i]) - 1
          assert c >= 0
          later_shed = any(
              ev.actor[r, j] == 1 - seat
              and ev.type[r, j] == gin.EVT_DISCARD
              and int(ev.public_card[r, j]) - 1 == c
              for j in range(i + 1, cnt)
          )
          held = bool(np.asarray(flat.game[opp_key])[r, c])
          assert held or later_shed, (r, i, c)


def test_marginals_uniform_over_the_unseen_cards(sampler):
  """Marginals uniform over the unseen cards.

  Each unseen card is in the opponent's hidden hand with probability
  h / |U| (binomial, 5 sd), and a known card or a seen card never is.

  Args:
    sampler: the sampler fixture.
  """
  env, states = _domain_states(n_games=4, per_game=2)
  k = 4096
  for r in range(int(states.stage.shape[0])):
    s = tree_map(lambda x, r=r: x[r], states)
    seat = int(s.game["current_player"])
    info = _info(env, s, seat)
    u = np.asarray(gin_belief.unseen(info))
    h = int(gin_belief.hidden_count(info))
    w = _det(sampler, info, k, seed=r)
    opp = (
        np.asarray(
            w.states.game["player1_hand" if seat == 0 else "player0_hand"]
        )
        > 0
    )
    known = np.asarray(info.known_cards)[1 - seat] > 0
    hidden = opp & ~known[None]
    assert (hidden.sum(-1) == h).all()
    assert not hidden[:, ~u].any()
    p = h / u.sum()
    freq = hidden[:, u].mean(0)
    sd = math.sqrt(p * (1 - p) / k)
    assert np.all(np.abs(freq - p) < 5 * sd + 1e-12), (freq, p)


# -- privacy -------------------------------------------------------------------


def test_permuting_hidden_information_leaves_input_and_worlds_unchanged(
    sampler,
):
  """Permuting hidden information leaves input and worlds unchanged.

  hide_hidden_cards re-deals the opponent's hidden hand and the stock
  (and scrambles its private transcript cards and engine caches): the seat's
  InfoState, i.e. the sampler's whole input, is unchanged, so its worlds are
  bitwise the same for the same key.

  Args:
    sampler: the sampler fixture.
  """
  env, states = _domain_states(n_games=6, per_game=3)
  n = int(states.stage.shape[0])
  seats = np.asarray(states.game["current_player"]).astype(np.int32)
  keys = jax.random.split(jax.random.key(11), n)
  hidden = jax.jit(
      jax.vmap(lambda s, q, k: gin.hide_hidden_cards(env, s, q, k))
  )(states, jnp.asarray(seats), keys)
  changed = np.any(
      np.asarray(hidden.game["deck"]) != np.asarray(states.game["deck"]), -1
  )
  assert changed.sum() >= n // 2, changed
  info_a = jax.vmap(env.info_state)(states, jnp.asarray(seats))
  info_b = jax.vmap(env.info_state)(hidden, jnp.asarray(seats))
  _assert_tree_equal(info_a, info_b, "info")
  fn = jax.jit(
      lambda i: sampler.sample_determinized({}, (), i, 4, jax.random.key(5))
  )
  _assert_tree_equal(fn(info_a), fn(info_b), "worlds")


# -- registry, dtypes, determinism, stock order --------------------------------


def test_registered_and_built_from_the_env(env):
  """Registered and built from the env."""
  s = registry.build("gin", env)
  assert isinstance(s, gin_belief.GinBeliefSampler)
  assert registry.name_of(s) == "gin"
  assert s.params_source == "none" and not s.is_perfect_information
  assert not s.kwargs and not s.stateful and not s.init_params()
  with pytest.raises(ValueError, match="gin env"):
    gin_belief.GinBeliefSampler(env_registry.make("leduc_poker"))


def test_dtypes_shapes_and_determinism(env, sampler):
  """Dtypes shapes and determinism."""
  s, _ = gin_positions.case_a(env)
  info = _b(_info(env, s, 0))
  fn = jax.jit(lambda i, key: sampler.sample_determinized({}, (), i, 3, key))
  a, b, c = (
      fn(info, jax.random.key(0)),
      fn(info, jax.random.key(0)),
      fn(info, jax.random.key(1)),
  )
  _assert_tree_equal(a, b, "same key")
  assert not np.array_equal(
      np.asarray(a.chance_scores), np.asarray(c.chance_scores)
  )
  root = env.root()
  for (path, x), (_, y) in zip(
      jax.tree_util.tree_flatten_with_path(a.states)[0],
      jax.tree_util.tree_flatten_with_path(root)[0],
  ):
    assert x.dtype == y.dtype, (jax.tree_util.keystr(path), x.dtype, y.dtype)
    assert x.shape == (1, 3) + y.shape, (jax.tree_util.keystr(path), x.shape)
  assert a.chance_scores.dtype == jnp.float32 and a.chance_scores.shape == (
      1,
      3,
      52,
  )
  w, logw = sampler.sample_worlds({}, (), info, 3, jax.random.key(0))
  _assert_tree_equal(w, a.states, "sample_worlds == sample_determinized states")
  np.testing.assert_array_equal(np.asarray(logw), np.zeros((1, 3), np.float32))


def test_stock_order_known_answer():
  """Stock order known answer."""
  deck = np.zeros(52, np.int8)
  deck[[3, 10, 40]] = 1
  scores = np.full(52, -np.inf, np.float32)
  scores[[3, 10, 40]] = [1.0, 7.0, 4.0]
  got = np.asarray(
      gin_belief.stock_order(jnp.asarray(deck), jnp.asarray(scores))
  )
  assert list(got[:3]) == [10, 40, 3] and (got[3:] == -1).all()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
