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

"""Gin search fixtures: positions, shard map and the shared search checks."""

import collections

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.belief.tests import gin_positions
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import search as gin_search
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve.tests import arena_ref
from open_spiel.python.scale.serve.tests import helpers
from open_spiel.python.scale.serve.tests import serve_eval_checks

tree_map = jax.tree_util.tree_map

A = cm.NUM_ACTIONS

F32 = jnp.float32

# Stand-in value weights: multiples of 1/8, so hand sums are exact in f32.
W = (np.arange(52) % 7).astype(np.float32) / np.float32(8.0)


class StandIn(types.Policy):
  """A gin policy/value stand-in, stateless.

  logits = `bias` on every row, value of the row's seat = <hand, W> (the
  other seat: its negation).

  Attributes:
    bias: f32[A], the logits of every row.
    spec: the PolicySpec.
  """

  def __init__(self, bias=None):
    self.bias = (
        np.zeros(A, np.float32)
        if bias is None
        else np.asarray(bias, np.float32)
    )
    self.spec = types.PolicySpec(
        game="gin_rummy",
        obs_format="gin_info",
        num_actions=A,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name="test.gin_standin",
        arch_kwargs={},
    )

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def encode(self, info):
    """No features.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      f32[B, 0].
    """
    return jnp.zeros((info.seat.shape[0], 0), F32)

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step
    return carry

  def evaluate(self, params, carry, step, key):
    """The bias logits and the hand values.

    Args:
      params: the params.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput.
    """
    del params, carry, key
    info = step.info
    v = (info.hand.astype(F32) @ jnp.asarray(W)).astype(F32)
    mine = jnp.stack([v, -v], axis=-1)
    value = jnp.where((info.seat == 0)[:, None], mine, mine[:, ::-1]).astype(
        F32
    )
    b = info.seat.shape[0]
    return types.PolicyOutput(
        logits=jnp.broadcast_to(jnp.asarray(self.bias), (b, A)),
        value=value,
        aux={},
    )


def make_search(env, net, **kw):
  """The gin search policy over `net` with mcts_config(**kw).

  Args:
    env: the GinEnv.
    net: the policy/value network searched over.
    **kw: mcts_config overrides.

  Returns:
    The search policy.
  """
  cfg = gin_search.mcts_config(**kw)
  return gin_search.search_policy(gin_search.make(net, env, cfg))


def root_step(env, s, seat):
  """The batch-of-one seat step at state s, as at a match's first ply.

  Args:
    env: the GinEnv.
    s: the (unbatched) state.
    seat: the seat to act.

  Returns:
    The SeatStep.
  """
  s1 = tree_map(lambda x: jnp.asarray(x)[None], s)
  return plies.seat_step(
      env, s1, jnp.int32(seat), True, True, types.NO_ACTION, types.NO_ACTOR
  )


def run(sp, env, s, seat=0, seed=0):
  """Searches and evaluates sp at s for `seat`; numpy results.

  Args:
    sp: the search policy.
    env: the GinEnv.
    s: the (unbatched) state.
    seat: the seat to act.
    seed: the PRNG seed of the search.

  Returns:
    (SearchResult, PolicyOutput) as numpy trees.
  """
  params = sp.init_params(jax.random.key(0))
  step = root_step(env, s, seat)

  def go(p, st, key):
    """Searches and evaluates one step from a fresh observed carry.

    Args:
      p: the params.
      st: the SeatStep.
      key: a PRNG key.

    Returns:
      (SearchResult, PolicyOutput).
    """
    c = sp.init_carry(p, 1, st.seat, key)
    c = sp.observe(p, c, st)
    return sp.search(p, c, st, key), sp.evaluate(p, c, st, key)

  res, out = jax.jit(go)(params, step, jax.random.key(seed))
  return tree_map(np.asarray, res), tree_map(np.asarray, out)


def scripted_playout(env, s, a, seat, value_scale):
  """The score of `seat` / value_scale after action a at s.

  Every later ply is played by the scripted endgame (it must be a KNOCK /
  LAYOFF / WALL ply).

  Args:
    env: the GinEnv.
    s: the state.
    a: the action.
    seat: the seat scored.
    value_scale: the score's divisor.

  Returns:
    The score of `seat` / value_scale at the end of the hand.

  Raises:
    AssertionError: the hand did not end within 64 scripted plies.
  """
  apply, info_state = jax.jit(env.apply), jax.jit(env.info_state)
  s = apply(s, jnp.int32(a))
  for _ in range(64):
    if bool(env.terminal(s)):
      return float(np.asarray(env.returns(s))[seat]) / value_scale
    mover = int(env.actor(s))
    assert mover >= 0 and bool(gin.is_endgame_phase(s.game["phase"])), (
        "the playout left the scripted endgame",
        int(s.game["phase"]),
    )
    info = tree_map(
        lambda x: jnp.asarray(x)[None], info_state(s, jnp.int32(mover))
    )
    s = apply(s, gin_endgame.actions(info)[0])
  raise AssertionError("no terminal within 64 scripted plies")


POSITIONS = [
    dict(),  # hearts gin + junk vs OPP_HAND
    dict(hand1=[0, 2, 4, 6, 8, 13, 15, 17, 19, 21], upcard=30),
    dict(hand1=[14, 16, 18, 20, 22, 27, 29, 31, 33, 35], upcard=25),
]

# The slow tests run in shard files, so each file stays within a test
# runner's per-file time budget: gin_search_shard1_test.py runs
# check_gin_search_serve_equals_eval, gin_search_shard<k>_test.py runs
# check_search_beats_or_equals_the_raw_network at the positions
# SEARCH_SHARDS[k].
SEARCH_SHARDS = {2: (0,), 3: (1,), 4: (2,)}


def check_search_beats_or_equals_the_raw_network(env, pos):
  """Search beats or equals the raw network.

  The stand-in network prefers discarding a heart of its gin run; the
  best action (by the exact scripted playout) is the same in the true state
  and every sampled world (the known answer); search plays it, and its true
  value is >= the raw network's action's.

  Args:
    env: the GinEnv.
    pos: the index of the position in POSITIONS.
  """
  s, _ = gin_positions.wall_discard(env, **POSITIONS[pos])
  legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
  bias = np.zeros(A, np.float32)
  bias[39] = 3.0  # Ah: breaks the gin run
  net = StandIn(bias)
  true_q = {int(a): scripted_playout(env, s, a, 0, 1.0) for a in legal}
  best = max(true_q, key=true_q.get)
  sp = make_search(env, net, num_simulations=32, num_worlds=8)
  res, out = run(sp, env, s, seed=pos)
  # precondition: the answer is known (the same in every world)
  for k in range(8):
    world = tree_map(lambda x, k=k: x[0, k], res.worlds)
    qk = {int(a): scripted_playout(env, world, a, 0, 1.0) for a in legal}
    assert max(qk, key=qk.get) == best, (k, qk, true_q)
  greedy = types.DecisionRule("argmax", 1.0)
  legal_row = jnp.asarray(np.asarray(env.legal_mask(s))[None])
  a_search = int(
      decide_lib.decide(
          jnp.asarray(out.logits), legal_row, greedy, jax.random.key(0)
      )[0][0]
  )
  raw = net.evaluate({}, (), root_step(env, s, 0), None)
  a_raw = int(
      decide_lib.decide(raw.logits, legal_row, greedy, jax.random.key(0))[0][0]
  )
  assert a_search == best, (a_search, best, true_q)
  assert true_q[a_search] >= true_q[a_raw], (true_q, a_search, a_raw)


def small_net(env):
  """A small MLP over the gin observation tensor (644 features).

  Args:
    env: the GinEnv.

  Returns:
    (network policy, its params).
  """
  enc = "gin_rummy_observation_tensor"
  cfg = {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": [16], "layout": "shared"},
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
  net = policy_lib.NetworkPolicy(
      env.name, enc, env.num_actions, 2, enc, [644], cfg
  )
  return net, net.init_params(jax.random.key(2))


def check_gin_search_serve_equals_eval():
  """Serve == eval through HostAgent, NUMERIC (gin_search_shard1_test)."""
  env = gin.GinEnv(oklahoma_frac=1.0)
  net, net_params = small_net(env)
  pol = gin_search.make(
      net,
      env,
      gin_search.mcts_config(num_simulations=4, num_worlds=2, value_scale=50.0),
  )
  params = gin_search.make_params(pol, net_params)
  assert serve_eval_checks.serve_eval_rule(pol) == serve_eval_checks.NUMERIC
  cases = arena_ref.run(
      env,
      pol,
      params,
      types.SAMPLE,
      batch=4,
      num_steps=6,
      hands_per_match=1,
      seed=5,
  )
  # HostAgent compiles once per history length (a search per compile), so
  # only the two most common lengths are served; every state served is
  # checked.
  by_len = collections.Counter(len(c.steps.seat) for c in cases)
  lengths = [t for t, _ in by_len.most_common(2)]
  cases = [c for c in cases if len(c.steps.seat) in lengths]
  assert len(cases) >= 6, (len(cases), by_len)
  art = helpers.artifact(pol, params, types.SAMPLE)
  agents = [host_agent.HostAgent(art, seat, env=env) for seat in range(2)]
  serve_eval_checks.check_cases(env, agents, cases, types.SAMPLE, exact=False)
  searched = [c for c in cases if int(np.asarray(c.legal).sum()) > 1]
  assert searched, "no searched decision"
  for c in searched[:2]:
    assert (
        agents[c.seat]
        .act(c.hands, seat_key=c.seat_key, first_ply=c.first_ply)
        .decider
        == "search"
    )
