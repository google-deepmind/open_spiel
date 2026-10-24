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

"""Tests for gin PIMC search.

That is envs.gin.search over SearchPolicy with determinized chance and
fold_routed:

  * known answers on small reachable positions: at my DISCARD with two cards
    left in the stock every action ends the hand through the scripted
    endgame folded into the dynamics, so each world's root Q of every legal
    action is exactly the scripted playout's score / value_scale; at my DRAW
    the root Q of the stock draw is the stand-in's value after drawing the
    first card of that world's sampled stock order;
  * search beats or equals the raw network on fixed positions whose best
    action is known (the same in every sampled world and the true state);
  * a KNOCK / LAYOFF / WALL root is not searched: the scripted action;
  * serve == eval bitwise through HostAgent;
  * the registry rebuilds the policy from its arch_kwargs; refusals for
    unsupported combinations.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.belief.tests import gin_positions
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.gin import belief as gin_belief
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import search as gin_search
from open_spiel.python.scale.envs.tests import gin_games
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies.tests import gin_search_fixtures

POSITIONS = gin_search_fixtures.POSITIONS
SEARCH_SHARDS = gin_search_fixtures.SEARCH_SHARDS
StandIn = gin_search_fixtures.StandIn
W = gin_search_fixtures.W
# Short aliases of the shared fixtures (functions, so pylint's variable rule).
# pylint: disable=invalid-name
_run = gin_search_fixtures.run
_scripted_playout = gin_search_fixtures.scripted_playout
_search = gin_search_fixtures.make_search
_small_net = gin_search_fixtures.small_net
_step = gin_search_fixtures.root_step
# pylint: enable=invalid-name
tree_map = jax.tree_util.tree_map


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A gin env.

  Returns:
    The env.
  """
  return gin.GinEnv()


# -- mctx known answers --------------------------------------------------------


@pytest.mark.parametrize("value_scale", [1.0, 50.0])
def test_wall_discard_root_q_is_the_folded_scripted_score_in_every_world(
    env, value_scale
):
  """Wall discard root q is the folded scripted score in every world."""
  s, _ = gin_positions.wall_discard(env)
  legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
  # 11 discards + knock
  assert cm.ACTION_KNOCK in legal and len(legal) == 12, legal
  sp = _search(
      env, StandIn(), num_simulations=32, num_worlds=4, value_scale=value_scale
  )
  res, out = _run(sp, env, s)
  for k in range(4):
    world = tree_map(lambda x, k=k: x[0, k], res.worlds)
    for a in legal:
      want = _scripted_playout(env, world, a, 0, value_scale)
      np.testing.assert_allclose(
          res.q[0, k, a],
          want,
          rtol=0,
          atol=1e-6,
          err_msg=f"world {k} action {a}",
      )
  # PIMC: the policy's search value / Q are the means over the K worlds.
  np.testing.assert_allclose(
      out.aux[search.AUX_SEARCH_Q][0], res.q[0].mean(0), atol=1e-6
  )
  np.testing.assert_allclose(
      out.aux[search.AUX_SEARCH_VALUE][0], res.values[0].mean(), atol=1e-6
  )


def test_draw_root_q_follows_each_worlds_stock_order(env):
  """Draw root q follows each worlds stock order.

  At my DRAW (two legal actions, two simulations: each child visited
  once, Q = r + v(child)): Q(draw upcard) = <hand + upcard, W>; Q(draw stock)
  = <hand + the first card of THAT world's stock order, W>.

  Args:
    env: the GinEnv fixture.
  """
  s, _ = gin_positions.case_a(env)
  legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
  assert list(legal) == [cm.ACTION_DRAW_UPCARD, cm.ACTION_DRAW_STOCK], legal
  sp = _search(env, StandIn(), num_simulations=2, num_worlds=6)
  params = sp.init_params(jax.random.key(0))
  step = _step(env, s, 0)
  key = jax.random.key(4)
  res, _ = _run(sp, env, s, seed=4)
  # The worlds' stock orders: the sampler's own draw for this key (SearchPolicy
  # derives the sampler key as the first of split(key, 4)).
  k_world = jax.random.split(key, 4)[0]
  det = sp.sampler.sample_determinized(
      params["sampler"], (), step.info, 6, k_world
  )
  hand = np.asarray(s.game["player0_hand"]).astype(np.float32)
  up = int(s.game["upcard"])
  want_up = float((hand + np.eye(52, dtype=np.float32)[up]) @ W)
  firsts = set()
  for k in range(6):
    world = tree_map(lambda x, k=k: np.asarray(x)[0, k], det.states)
    np.testing.assert_array_equal(
        world.game["deck"], res.worlds.game["deck"][0, k]
    )
    first = int(
        np.asarray(
            gin_belief.stock_order(world.game["deck"], det.chance_scores[0, k])
        )[0]
    )
    firsts.add(first)
    want_stock = float((hand + np.eye(52, dtype=np.float32)[first]) @ W)
    assert res.q[0, k, cm.ACTION_DRAW_UPCARD] == np.float32(want_up), (
        k,
        res.q[0, k],
    )
    assert res.q[0, k, cm.ACTION_DRAW_STOCK] == np.float32(want_stock), (
        k,
        first,
        res.q[0, k],
    )
  assert len(firsts) > 1, "the worlds should not all draw the same card"


# -- search >= raw network on known-answer positions ---------------------------


def test_the_shards_cover_every_position_once():
  """SEARCH_SHARDS covers every position once, and each shard file exists."""
  got = []
  for shard in SEARCH_SHARDS.values():
    got.extend(shard)
  got.sort()
  assert got == list(range(len(POSITIONS))), SEARCH_SHARDS
  here = os.path.dirname(os.path.abspath(__file__))
  for k in (1, *SEARCH_SHARDS):
    assert os.path.exists(os.path.join(here, f"gin_search_shard{k}_test.py")), k


# -- routed roots --------------------------------------------------------------


def test_endgame_roots_are_not_searched_they_play_the_scripted_action():
  """Endgame roots are not searched they play the scripted action."""
  env = gin.GinEnv(oklahoma_frac=1.0)
  g = gin_games.games(8, 3)
  mask = gin_games.decision_mask(env, g) & np.asarray(
      gin.is_endgame_phase(g.states.game["phase"])
  )
  idx = tuple(a[:6] for a in np.nonzero(mask))
  states = tree_map(lambda x: x[idx], g.states)
  seats = jnp.asarray(np.asarray(states.game["current_player"]), jnp.int32)
  step = plies.seat_step(
      env, states, seats, False, False, types.NO_ACTION, types.NO_ACTOR
  )
  sp = _search(env, StandIn(), num_simulations=4, num_worlds=2)
  params = sp.init_params(jax.random.key(0))

  def go(p, st, key):
    """Evaluates one step after observing it.

    Args:
      p: the params.
      st: the SeatStep.
      key: a PRNG key.

    Returns:
      The PolicyOutput.
    """
    c = sp.init_carry(p, st.seat.shape[0], st.seat, key)
    return sp.evaluate(p, sp.observe(p, c, st), st, key)

  out = jax.jit(go)(params, step, jax.random.key(1))
  want = gin_endgame.GinScriptedEndgamePolicy().evaluate({}, (), step, None)
  np.testing.assert_array_equal(np.asarray(out.logits), np.asarray(want.logits))
  hot = np.asarray(want.logits) == 0
  np.testing.assert_array_equal(
      np.asarray(out.aux[types.AUX_TARGET_POLICY]), hot.astype(np.float32)
  )


def _endgame_steps(env):
  """Seat steps at up to six reachable KNOCK / LAYOFF / WALL decisions.

  Args:
    env: the GinEnv.

  Returns:
    The SeatStep, leaves [B, ...].
  """
  g = gin_games.games(8, 3)
  mask = gin_games.decision_mask(env, g) & np.asarray(
      gin.is_endgame_phase(g.states.game["phase"])
  )
  idx = tuple(a[:6] for a in np.nonzero(mask))
  assert idx[0].size, "no endgame decision in the sampled games"
  states = tree_map(lambda x: x[idx], g.states)
  seats = jnp.asarray(np.asarray(states.game["current_player"]), jnp.int32)
  return plies.seat_step(
      env, states, seats, False, False, types.NO_ACTION, types.NO_ACTOR
  )


def test_the_plain_member_plays_the_scripted_endgame_where_the_net_is_nan():
  """gin_mctx_pimc's plain member is the base routed to the scripted endgame.

  A gin network is NaN at KNOCK / LAYOFF / WALL (its encoder is defined only
  away from those phases). The plain member (the pool member, gate entrant
  and plain sibling of expert iteration) plays those plies with the scripted
  endgame, as every gin entrant does, so its outputs there are the scripted
  endgame's, not the base's NaN; its params wrap the base's and unwrap back.
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  factory = gin_search.GinMctxPimc({})
  base = StandIn(bias=np.full(gin_search_fixtures.A, np.nan, np.float32))
  plain = factory.plain_member(base)
  assert routed.is_scripted_phase(plain) and plain.base is base
  base_params = base.init_params(jax.random.key(0))
  params = factory.plain_params(plain, base_params)
  assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
      plain.init_params(jax.random.key(0))
  )
  assert factory.unplain_params(plain, params) is base_params
  with pytest.raises(ValueError, match="already a RoutedPolicy"):
    factory.plain_member(plain)

  step = _endgame_steps(env)

  def go(p, st, key):
    """Evaluates the plain member on one step after observing it.

    Args:
      p: the plain member's params.
      st: the SeatStep.
      key: a PRNG key.

    Returns:
      The PolicyOutput.
    """
    c = plain.init_carry(p, st.seat.shape[0], st.seat, key)
    return plain.evaluate(p, plain.observe(p, c, st), st, key)

  out = jax.jit(go)(params, step, jax.random.key(1))
  want = gin_endgame.GinScriptedEndgamePolicy().evaluate({}, (), step, None)
  logits = np.asarray(out.logits)
  assert not np.isnan(logits).any(), logits
  np.testing.assert_array_equal(logits, np.asarray(want.logits))
  # The base alone (the plain member before the fix) is NaN on these plies.
  bare = base.evaluate(base_params, (), step, None)
  assert np.isnan(np.asarray(bare.logits)).all()


# -- serve == eval -------------------------------------------------------------


# -- config identity and refusals ----------------------------------------------


def test_arch_kwargs_record_the_config_and_the_registry_rebuilds_it():
  """Architecture kwargs record the config and the registry rebuilds it."""
  env = gin.GinEnv(oklahoma_frac=1.0)
  net, _ = _small_net(env)
  pol = gin_search.make(
      net,
      env,
      gin_search.mcts_config(num_simulations=8, num_worlds=4, value_scale=50.0),
  )
  sp = gin_search.search_policy(pol)
  m = sp.spec.arch_kwargs["mcts"]
  assert (
      m["chance"] == "determinized"
      and m["fold_routed"]
      and m["value_scale"] == 50.0
  )
  assert sp.spec.arch_kwargs["sampler"] == {"name": "gin", "kwargs": {}}
  again = registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)
  assert again.spec == pol.spec
  assert isinstance(again.base.sampler, gin_belief.GinBeliefSampler)


def test_mcts_config_json_round_trips():
  """Mcts config json round trips."""
  d = search.MCTSConfig().to_json()
  assert {"chance", "fold_routed", "value_scale", "act"} <= set(d)
  assert search.MCTSConfig(**d) == search.MCTSConfig()


def test_refusals(env):
  """Refusals."""
  with pytest.raises(ValueError, match="RoutedPolicy"):
    search.SearchPolicy(
        StandIn(),
        env,
        gin_belief.GinBeliefSampler(env),
        search.MCTSConfig(chance="determinized", fold_routed=True),
    )
  with pytest.raises(ValueError, match="determinized"):
    search.SearchPolicy(
        gin_search.routed_net(StandIn()),
        env,
        gin_belief.GinBeliefSampler(env),
        search.MCTSConfig(fold_routed=True),
    )
  with pytest.raises(ValueError, match="fixes"):
    gin_search.mcts_config(mode="single_agent")
  with pytest.raises(ValueError):
    search.MCTSConfig(chance="nope")


# -- value units from the artifact -----------------------------------


def _net_artifact(tmp_path, name, value_scale):
  """Writes a routed gin network artifact with an optional value_scale.

  Args:
    tmp_path: The folder.
    name: The artifact name.
    value_scale: The value scale, or None.

  Returns:
    (env, the loaded Artifact).
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  net, net_params = _small_net(env)
  rp = routed.with_scripted_phase(net, env_registry.game("gin_rummy").endgame)
  d = str(tmp_path / name)
  artifact.write_artifact(
      d,
      5,
      rp,
      rp.make_params(net_params, {}),
      rules_sha="r",
      config_hash="c",
      algo="test",
      env_name=env.name,
      extensions={} if value_scale is None else {"value_scale": value_scale},
  )
  return env, artifact.load_artifact(d, 5)


def test_from_artifact_reads_and_requires_the_recorded_value_scale(tmp_path):
  """From artifact reads and requires the recorded value scale."""
  env, art = _net_artifact(tmp_path, "v50", 50.0)
  pol, params = gin_search.from_artifact(art, env)  # the recorded units
  assert gin_search.search_policy(pol).cfg.value_scale == 50.0
  assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
      pol.init_params(jax.random.key(0))
  )
  pol2, _ = gin_search.from_artifact(
      art, env, gin_search.mcts_config(num_simulations=4, value_scale=50.0)
  )
  assert gin_search.search_policy(pol2).cfg.num_simulations == 4
  # a config in other units
  with pytest.raises(ValueError, match="value_scale"):
    gin_search.from_artifact(art, env, gin_search.mcts_config(value_scale=1.0))
  env, bare = _net_artifact(tmp_path, "none", None)
  with pytest.raises(ValueError, match="value_scale"):  # units not recorded
    gin_search.from_artifact(bare, env)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
