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

"""GinBeliefSampler calibration against the true deals of held-out games.

Held-out games: gin_games.games(HELD_OUT_GAMES, HELD_OUT_SEED), full gin hands
played through the raw layer (uniform deal and stock draws, the rule bot /
mixed behaviour at the strategic phases); the seed is used by no other test,
and nothing in the sampler was tuned on them. At each sampled FIRST_UPCARD /
DRAW / DISCARD node the seat to move's InfoState goes to the sampler; K
worlds give p_c, the sampled probability that unseen card c is in the
opponent's hidden hand, scored against the TRUE deal y_c (the real state's
opponent hand minus its publicly known cards):

  * support: every true hidden card is unseen with p_c > 0, and no seen or
    known card is ever sampled (p_c = 0 off the unseen set);
  * calibration in the large, per state: sum_c p_c = sum_c y_c = h exactly
    (every world has the opponent's true hidden-hand size);
  * reliability: pooled over states and cards, binned by p, each bin's true
    frequency is within 4 binomial sd (+ the K-sample noise) of its mean p,
    and the expected calibration error is below ECE_MAX.

Reported, not asserted: the same reliability restricted to unseen cards of a
rank the opponent shed, and of a rank it took from the discard pile. A
uniform-consistent belief treats the opponent's moves as uninformative, so a
hand-dependent opponent (the rule bot keeps pairs and sheds lone high cards)
makes those strata miscalibrated even though the pooled reliability holds.
"""

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import belief as gin_belief
from open_spiel.python.scale.envs.tests import gin_games

tree_map = jax.tree_util.tree_map
HELD_OUT_GAMES = 48
HELD_OUT_SEED = 7_654_321  # no other test uses this seed
PER_GAME = 8
K = 128
CHUNK = 16
N_BINS = 10
ECE_MAX = 0.02


def _held_out_states():
  """Held-out gin decision states (first upcard, draw, discard).

  Returns:
    (env, states).
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  g = gin_games.games(HELD_OUT_GAMES, HELD_OUT_SEED)
  mask = np.asarray(gin_games.decision_mask(env, g))
  phase = np.asarray(g.states.game["phase"])
  mask &= (
      (phase == cm.PHASE_FIRST_UPCARD)
      | (phase == cm.PHASE_DRAW)
      | (phase == cm.PHASE_DISCARD)
  )
  rng = np.random.default_rng(HELD_OUT_SEED)
  picks = []
  for i in range(HELD_OUT_GAMES):
    ts = np.flatnonzero(mask[i])
    for t in rng.choice(ts, size=min(PER_GAME, len(ts)), replace=False):
      picks.append((i, int(t)))
  idx = tuple(np.asarray(picks).T)
  return env, tree_map(lambda x: x[idx], g.states)


def _truth(states, seat):
  """bool[N, 52]: the opponent's hidden hand in the real states (seat view).

  Args:
    states: The states, leaves [N, ...].
    seat: int[N], the seat of each view.

  Returns:
    bool[N, 52], the hidden cards.
  """
  g = tree_map(np.asarray, states.game)
  opp_hand = (
      np.where((seat == 0)[:, None], g["player1_hand"], g["player0_hand"]) > 0
  )
  known = (
      np.where(
          (seat == 0)[:, None], g["known_cards"][:, 1], g["known_cards"][:, 0]
      )
      > 0
  )
  return opp_hand & ~known


def _opp_rank_strata(states, seat):
  """Cards sharing a rank with a card the opponent shed or took.

  From the real public transcript.

  Args:
    states: the states, leaves [N, ...].
    seat: i32[N], the seat of each row.

  Returns:
    (shed, took) bool[N, 52]: discarded / taken from the discard pile.
  """
  ev = tree_map(np.asarray, states.events)
  n = ev.actor.shape[0]
  shed = np.zeros((n, 52), bool)
  took = np.zeros((n, 52), bool)
  rank = np.arange(52) % 13
  for r in range(n):
    for i in range(int(ev.count[r])):
      if ev.actor[r, i] != 1 - seat[r]:
        continue
      c = int(ev.public_card[r, i]) - 1
      if c < 0:
        continue
      if ev.type[r, i] == gin.EVT_DISCARD:
        shed[r] |= rank == rank[c]
      elif ev.type[r, i] == gin.EVT_DRAW_UPCARD:
        took[r] |= rank == rank[c]
  return shed, took


@pytest.fixture(name="calib", scope="module")
def _calib_fixture():
  """Per-card sampled and true hidden-hand probabilities on held-out states.

  Returns:
    The calibration inputs.
  """
  env, states = _held_out_states()
  seat = np.asarray(states.game["current_player"]).astype(np.int32)
  infos = jax.vmap(env.info_state)(states, jnp.asarray(seat))
  sampler = gin_belief.GinBeliefSampler(env)

  @jax.jit
  def chunk_p(info, key):
    """Per-card sampled probability of being in the opponent's hidden hand.

    Returns:
      f64[n, 52], the fraction of worlds with the card hidden and unknown.

    Args:
      info: a chunk of InfoStates, leaves [n, ...].
      key: a PRNG key.
    """
    w = sampler.sample_determinized({}, (), info, K, key)
    s = info.seat.astype(jnp.int32)
    opp = (
        jnp.where(
            (s == 0)[:, None, None],
            w.states.game["player1_hand"],
            w.states.game["player0_hand"],
        )
        > 0
    )
    kc = info.known_cards > 0
    known = jnp.where((s == 0)[:, None], kc[:, 1], kc[:, 0])
    return jnp.mean((opp & ~known[:, None]).astype(jnp.float64), axis=1)

  n = int(seat.shape[0])

  def sampled(base):
    """The sampled probabilities of every state, in chunks.

    Returns:
      f64[N, 52] from worlds drawn with base key `base`.

    Args:
      base: the base PRNG seed.
    """
    return np.concatenate([
        np.asarray(
            chunk_p(
                tree_map(lambda x, i=i: x[i : i + CHUNK], infos),
                jax.random.fold_in(jax.random.key(base), i),
            )
        )
        for i in range(0, n, CHUNK)
    ])

  p = sampled(7)
  # an independent draw assigns the reliability bins, so binning on sampling
  # noise does not select the cells whose p is averaged (no regression bias)
  p_bin = sampled(8)
  u = np.asarray(jax.vmap(gin_belief.unseen)(infos))
  h = np.asarray(jax.vmap(gin_belief.hidden_count)(infos))
  y = _truth(states, seat)
  shed, took = _opp_rank_strata(states, seat)
  return dict(p=p, p_bin=p_bin, y=y, u=u, h=h, shed=shed, took=took, n=n)


def _reliability(p_bin, p, y):
  """Reliability rows per equal-width bin of `p_bin`, and the ECE.

  `p_bin` is an independent draw of p.

  Args:
    p_bin: the probabilities that assign the bins.
    p: the sampled probabilities.
    y: the true outcomes.

  Returns:
    ([(n, mean p, true frequency)] over bins with cells, ECE).
  """
  edges = np.linspace(0.0, 1.0, N_BINS + 1)
  b = np.clip(np.digitize(p_bin, edges) - 1, 0, N_BINS - 1)
  rows, ece = [], 0.0
  for k in range(N_BINS):
    m = b == k
    if m.sum():
      rows.append((int(m.sum()), float(p[m].mean()), float(y[m].mean())))
      ece += m.sum() / p.size * abs(p[m].mean() - y[m].mean())
  return rows, float(ece)


def test_support_contains_the_true_deal(calib):
  """Support contains the true deal."""
  p, y, u = calib["p"], calib["y"], calib["u"]
  assert calib["n"] >= 300, calib["n"]
  assert not y[~u].any()  # the true hidden hand lies in the unseen set
  assert (p[~u] == 0).all()  # seen / known cards are never sampled
  # every true hidden card has positive sampled probability
  assert (p[y] > 0).all()


def test_calibration_in_the_large_per_state(calib):
  """Calibration in the large per state."""
  p, y, h = calib["p"], calib["y"], calib["h"]
  np.testing.assert_allclose(p.sum(-1), h, rtol=0, atol=1e-9)
  np.testing.assert_array_equal(y.sum(-1), h)


def test_reliability_against_the_true_deals(calib):
  """Reliability against the true deals."""
  u = calib["u"]
  p, y = calib["p"][u], calib["y"][u].astype(np.float64)
  rows, ece = _reliability(calib["p_bin"][u], p, y)
  brier = float(np.mean((p - y) ** 2))
  print(
      f"\n[gin belief calibration, {calib['n']} held-out states x K={K}, "
      f"{p.size} unseen-card cells] ECE {ece:.4f}, Brier {brier:.4f}"
  )
  for n, mp, fy in rows:
    print(f"  bin n={n:6d} mean p {mp:.4f} true freq {fy:.4f}")
  for n, mp, fy in rows:
    if n < 100:
      continue
    sd = math.sqrt(max(mp * (1 - mp), 1e-6) / n)
    assert abs(fy - mp) <= 4 * sd + 4 * math.sqrt(0.25 / (K * n)) + 1e-9, (
        n,
        mp,
        fy,
    )
  assert ece <= ECE_MAX, ece


def test_reported_strata_opponent_moves_carry_no_information(calib):
  """Reported strata opponent moves carry no information.

  Reported (module doc): reliability on cards of a rank the opponent shed
  or took. No bound: the uniform-consistent belief's known limitation.

  Args:
    calib: the calibration fixture.
  """
  u = calib["u"]
  for name in ("shed", "took"):
    m = u & calib[name]
    p, y = calib["p"][m], calib["y"][m].astype(np.float64)
    if p.size == 0:
      print(f"[gin belief stratum {name}] no cells")
      continue
    se = math.sqrt(max(p.mean() * (1 - p.mean()), 1e-9) / p.size)
    print(
        f"[gin belief stratum opponent-{name}-rank] cells {p.size}: "
        f"mean p {p.mean():.4f}, "
        f"true freq {y.mean():.4f}, z {(y.mean() - p.mean()) / se:+.2f}"
    )
    assert np.isfinite(p).all()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
