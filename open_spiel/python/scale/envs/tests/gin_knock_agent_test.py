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

"""Tests for the agent's own knock discard (endgame.AGENT_KNOCK_ROUTE).

On AGENT_KNOCK_ROUTE the knocker's 11-card discard is the base policy's
decision. The scripted endgame plays everything else: KNOCK meld laying,
LAYOFF and WALL. These tests check:

  * known answers for the route;
  * every consumer keeps such a composition as it is;
  * a network picks the knock discard with finite logits over the engine's
    legal knock discards;
  * full games through every phase;
  * the learner trajectory keeps the decision with its behaviour logp;
  * the decision's features do not depend on what the knocker cannot see.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import reference_eval
from open_spiel.python.scale.envs.gin import search as gin_search
from open_spiel.python.scale.envs.tests import gin_games
from open_spiel.python.scale.envs.tests import gin_helpers
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.sources import decisions

tree_map = jax.tree_util.tree_map
ENCODER = "gin_rummy_observation_tensor"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The Oklahoma gin env.

  Returns:
    The GinEnv.
  """
  return gin.GinEnv(oklahoma_frac=1.0)


def _net():
  """A small observation-tensor network policy for gin.

  Returns:
    The policy.
  """
  enc = enc_registry.get(ENCODER)
  kw = common.network_policy_kwargs(
      "gin_rummy",
      ENCODER,
      list(enc.shape),
      cm.NUM_ACTIONS,
      2,
      catalog.mlp([64], "relu", "separate"),
  )
  return registry.build("network", kw)


def _agent(net):
  """`net` on the agent knock route.

  Args:
    net: The network policy.

  Returns:
    The RoutedPolicy.
  """
  pol = routed.with_scripted_phase(
      net, gin_endgame.ARCH, gin_endgame.AGENT_KNOCK_ROUTE
  )
  assert isinstance(pol, routed.RoutedPolicy)
  return pol


def _knock_nodes(env, n=64, seed=4):
  """(states, infos) of the 11-card KNOCK nodes of seeded games.

  Args:
    env: The gin env.
    n: The games.
    seed: The seed.

  Returns:
    (states, infos).
  """
  g = gin_games.games(n, seed)
  phase = np.asarray(g.states.game["phase"])
  st = gin_games.flat_states(
      g, gin_games.decision_mask(env, g) & (phase == cm.PHASE_KNOCK)
  )
  info = jax.vmap(env.info_state)(st, jax.vmap(env.actor)(st))
  eleven = np.asarray(gin_endgame.is_knock_discard(info))
  st = tree_map(lambda x: x[eleven], st)
  info = tree_map(lambda x: x[eleven], info)
  assert int(eleven.sum()) >= 20, int(eleven.sum())
  return st, info


def test_route_known_answers():
  """The agent route leaves only the knocker's 11-card discard to the base."""
  knocker_melds = np.zeros((2, cm.NUM_MELDS), bool)
  knocker_melds[0] = gin_helpers.melds_mask("7d 8d 9d", "As 2s 3s")
  infos = [
      gin_helpers.make_info(
          hand="As 2s 3s 4c 5c 6c 7d 8d 9d Kh Qh", phase=cm.PHASE_DISCARD
      ),
      gin_helpers.make_info(
          hand="As 2s 5s 2c 5c Ad 2d 5d Ah 2h 3h",
          phase=cm.PHASE_KNOCK,
          knocker=0,
      ),
      gin_helpers.make_info(
          hand="As 2s 5s 2c 5c Ad 2d 5d Ah 2h", phase=cm.PHASE_KNOCK, knocker=0
      ),
      gin_helpers.make_info(
          seat=1,
          hand="Td 5s 9c Kh Qs 2h 4d Jc 6h 8s",
          phase=cm.PHASE_LAYOFF,
          knocker=0,
          layed_melds=knocker_melds,
      ),
      gin_helpers.make_info(
          hand="As 2s 3s 4c 5c 6c 7d 8d 9d Kh",
          phase=cm.PHASE_WALL,
          stock_count=2,
      ),
      # The other seat's knock discard is not this row's decision.
      gin_helpers.make_info(
          hand="As 2s 5s 2c 5c Ad 2d 5d Ah 2h 3h",
          phase=cm.PHASE_KNOCK,
          knocker=1,
          current_player=1,
      ),
  ]
  info = gin_helpers.batch(*infos)
  np.testing.assert_array_equal(
      np.asarray(gin_endgame.in_domain(info)),
      [False, True, True, True, True, False],
  )
  np.testing.assert_array_equal(
      np.asarray(gin_endgame.is_knock_discard(info)),
      [False, True, False, False, False, False],
  )
  np.testing.assert_array_equal(
      np.asarray(gin_endgame.in_domain_agent_knock(info)),
      [False, False, True, True, True, False],
  )
  base = uniform.UniformPolicy("gin_rummy", cm.NUM_ACTIONS, 2)
  np.testing.assert_array_equal(
      np.asarray(routed.override_rows(_agent(base), info)),
      [False, False, True, True, True, False],
  )


def test_every_consumer_keeps_the_agent_route():
  """Both declared routes are scripted phases; nobody wraps them twice."""
  base = uniform.UniformPolicy("gin_rummy", cm.NUM_ACTIONS, 2)
  default = routed.with_scripted_phase(base, gin_endgame.ARCH)
  agent = _agent(base)
  assert default.spec.arch_kwargs["route"] == gin_endgame.ROUTE
  assert agent.spec.arch_kwargs["route"] == gin_endgame.AGENT_KNOCK_ROUTE
  assert routed.phase_routes(agent.override) == gin_endgame.ROUTES
  for pol in (default, agent):
    assert routed.is_scripted_phase(pol)
    assert routed.is_scripted_phase(pol, gin_endgame.ARCH)
    assert routed.with_scripted_phase(pol, gin_endgame.ARCH) is pol
    params = pol.make_params({}, {})
    assert routed.routed_pair(pol, params, gin_endgame.ARCH)[0] is pol
    e = match.Entrant("x", pol, params, types.ARGMAX)
    assert reference_eval.route_endgame(e, None) is e
    assert gin_search.routed_net(pol) is pol
  # An explicit route is matched exactly.
  agent_route = gin_endgame.AGENT_KNOCK_ROUTE
  assert not routed.is_scripted_phase(default, route=agent_route)
  assert routed.is_scripted_phase(agent, route=agent_route)
  with pytest.raises(ValueError, match="is not played on route"):
    routed.with_scripted_phase(
        base, gin_endgame.ARCH, f"{gin_endgame.__name__}:is_knock_discard"
    )
  # The serialised spec rebuilds the same route.
  rebuilt = registry.build(agent.spec.arch_name, dict(agent.spec.arch_kwargs))
  assert rebuilt.route is agent.route


def test_network_picks_the_knock_discard(env):
  """Finite logits; the legal mask is the engine's legal knock discards."""
  st, info = _knock_nodes(env)
  net = _net()
  pol = _agent(net)
  params = pol.make_params(net.init_params(jax.random.key(1)), {})
  b = int(info.seat.shape[0])
  step = gin_helpers.seat_steps_at(env, st, False)
  legal = np.asarray(step.legal_mask)
  cache_legal, _ = jax.vmap(gin_endgame.knock_cache)(info)
  np.testing.assert_array_equal(
      legal[:, : cm.NUM_CARDS], np.asarray(cache_legal)
  )
  assert not legal[:, cm.NUM_CARDS :].any()
  assert legal.sum(axis=1).min() >= 1
  np.testing.assert_array_equal(
      np.asarray(routed.override_rows(pol, info)), np.zeros(b, bool)
  )
  carry = pol.init_carry(params, b, step.seat, jax.random.key(0))
  out = pol.evaluate(params, carry, step, jax.random.key(2))
  lg = np.asarray(out.logits)
  assert np.isfinite(lg).all()
  # The routed composition's logits there are the network's own.
  net_params = params["base"]
  net_carry = net.init_carry(net_params, b, step.seat, jax.random.key(0))
  nout = net.evaluate(net_params, net_carry, step, jax.random.key(2))
  np.testing.assert_allclose(lg, np.asarray(nout.logits), rtol=0, atol=0)
  pick = np.argmax(np.where(legal, lg, -np.inf), axis=1)
  assert legal[np.arange(b), pick].all()

  # The network does not always take the script's min-deadwood card.
  def script(i):
    """The script's min-deadwood knock discard for one row.

    Args:
      i: one row's GinInfo.

    Returns:
      The card the script discards.
    """
    return gin_endgame.min_deadwood_discard(*gin_endgame.knock_cache(i))

  scripted = np.asarray(jax.vmap(script)(info))
  assert (pick != scripted).any()


@pytest.fixture(name="rollout", scope="module")
def _rollout_fixture():
  """The agent-route network in self-play, every seat a perspective.

  Returns:
    (env, net, routed policy, params, trajectory batch).
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  net = _net()
  pol = _agent(net)
  params = pol.make_params(net.init_params(jax.random.key(3)), {})
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 64, 400
  table = arena_lib.self_play_table(b, 2, all_seats=True)
  dk, pk = arena_lib.independent_keys(jax.random.key(17), b, 2)
  _, tb, _ = ar.collect([tree_map(lambda x: x[None], params)], table, dk, pk, t)
  return env, net, pol, params, tb


def test_full_games_through_every_phase(rollout):
  """Every phase is played; every action taken is legal."""
  _, _, _, _, tb = rollout
  valid = np.asarray(tb.valid)
  phase = np.asarray(tb.steps.info.phase)
  action = np.asarray(tb.action)
  legal = np.asarray(tb.steps.legal_mask)
  i, j = np.nonzero(valid)
  assert legal[i, j, action[i, j]].all()
  played = set(np.unique(phase[valid]).tolist())
  assert {
      cm.PHASE_FIRST_UPCARD,
      cm.PHASE_DRAW,
      cm.PHASE_DISCARD,
      cm.PHASE_KNOCK,
      cm.PHASE_LAYOFF,
      cm.PHASE_WALL,
  } <= played, played
  knock11 = np.asarray(gin_endgame.is_knock_discard(tb.steps.info)) & valid
  assert knock11.sum() > 0
  assert (action[knock11] < cm.NUM_CARDS).all()
  assert int(np.asarray(tb.done).sum()) > 0


def test_learner_trajectory_keeps_the_knock_discard(rollout):
  """exclude_override keeps the knock discard with the network's logp."""
  _, net, pol, params, tb = rollout
  out, n = decisions.exclude_override(pol, tb)
  valid, kept = np.asarray(tb.valid), np.asarray(out.valid)
  info = tb.steps.info
  endgame = np.asarray(gin.is_endgame_phase(info.phase))
  knock11 = np.asarray(gin_endgame.is_knock_discard(info))
  np.testing.assert_array_equal(kept, valid & ~(endgame & ~knock11))
  assert int(n) == int((valid & endgame & ~knock11).sum()) > 0
  rows = kept & knock11
  assert rows.sum() > 0
  # The behaviour logp of the kept knock discards is the network's: recompute
  # the network's logits on those steps and its sampling probability.
  i, j = np.nonzero(rows)
  steps = tree_map(lambda x: x[i, j], tb.steps)
  m = len(i)
  net_params = params["base"]
  net_carry = net.init_carry(net_params, m, steps.seat, jax.random.key(0))
  nout = net.evaluate(net_params, net_carry, steps, jax.random.key(0))
  lg = np.asarray(nout.logits, np.float64)
  np.testing.assert_allclose(
      np.asarray(tb.behavior_logits)[i, j], lg, rtol=1e-5, atol=1e-5
  )
  legal = np.asarray(steps.legal_mask)
  masked = np.where(legal, lg, -np.inf)
  logp = masked - np.logaddexp.reduce(masked, axis=1, keepdims=True)
  a = np.asarray(tb.action)[i, j]
  np.testing.assert_allclose(
      np.asarray(tb.behavior_logp)[i, j],
      logp[np.arange(m), a],
      rtol=1e-4,
      atol=1e-4,
  )
  assert np.isfinite(np.asarray(tb.behavior_logp)[i, j]).all()


def test_knock_discard_features_do_not_leak(env):
  """Re-dealing what the knocker cannot see changes nothing it decides on."""
  st, info = _knock_nodes(env)
  seat = jax.vmap(env.actor)(st)
  enc = enc_registry.get(ENCODER)
  x = np.asarray(jax.vmap(enc.fn)(info))
  legal = np.asarray(jax.vmap(env.legal_mask)(st))
  assert np.isfinite(x).all()
  for k in range(3):
    keys = jax.random.split(jax.random.key(100 + k), int(seat.shape[0]))
    st2 = jax.vmap(lambda s, p, kk: gin.hide_hidden_cards(env, s, p, kk))(
        st, seat, keys
    )
    opp = np.where(
        np.asarray(seat)[:, None] == 0,
        np.asarray(st2.game["player1_hand"]),
        np.asarray(st2.game["player0_hand"]),
    )
    opp0 = np.where(
        np.asarray(seat)[:, None] == 0,
        np.asarray(st.game["player1_hand"]),
        np.asarray(st.game["player0_hand"]),
    )
    if k == 0:
      assert (opp != opp0).any()  # the re-deal moved hidden cards
    info2 = jax.vmap(env.info_state)(st2, seat)
    np.testing.assert_array_equal(np.asarray(jax.vmap(enc.fn)(info2)), x)
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(env.legal_mask)(st2)), legal
    )


if __name__ == "__main__":
  # The package runs with x64 on, as its entry points and conftest.py set
  # it; set here for `python <file>`, never at import (which changes no
  # process-global state).
  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
