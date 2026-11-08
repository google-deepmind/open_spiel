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

"""Length buckets.

serve.replay, "Length buckets"; serve.host_agent, "Compiles".

* the padded replay gives the SeatSteps of stacking exactly T plies, bit
  for bit, on tic-tac-toe, kuhn (a two-hand match), leduc, connect_four and
  a long gin rummy game, every seat, and pads to bucket_size(T); the
  last-ply-only replay (stateless policies) gives its last ply, bit for bit;
* HostAgent's decisions (logits, value, distribution, action) equal the
  unbucketed path's, the policy run on the exact [1, T] history, bit for
  bit, for stateless networks and a recurrent one;
* compiles: a whole connect_four game served without warmup traces
  O(log T) shapes, not O(T); after HostAgent.warmup / CheckpointBot.warmup
  no decision traces or compiles anything (jax.monitoring's events);
* the seats of one artifact share their env and jitted calls through the
  PolicyServer (the second seat's warmup compiles nothing), and a second
  CheckpointBot.warmup of warm agents compiles nothing and plays no playout.
"""

import sys
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve.tests import checkpoint_bot_helpers as cbh
from open_spiel.python.scale.serve.tests import helpers
import pyspiel

tree_map = jax.tree_util.tree_map

_TRACE = "/jax/core/compile/jaxpr_trace_duration"
_COMPILE = "/jax/core/compile/backend_compile_duration"
_EVENTS = {_TRACE: 0, _COMPILE: 0}


def _listener(event, *args, **kwargs):
  del args, kwargs
  if event in _EVENTS:
    _EVENTS[event] += 1


jax.monitoring.register_event_duration_secs_listener(_listener)


class _Count:
  """Traces and backend compiles inside the block."""

  def __enter__(self):
    self._start = dict(_EVENTS)
    return self

  def __exit__(self, *exc):
    self.traces = _EVENTS[_TRACE] - self._start[_TRACE]
    self.compiles = _EVENTS[_COMPILE] - self._start[_COMPILE]


_REF_SEAT_STEP = weakref.WeakKeyDictionary()


def _exact_replay(env, hands, seat):
  """The unbucketed reference replay, without length buckets.

  Every batched call on exactly the hand's / match's plies, stacked on the
  device.

  Args:
    env: the env.
    hands: the match's hands, each a list of actions.
    seat: the seat observing.

  Returns:
    The SeatStep [1, T].
  """
  fns = replay._fns(env)  # pylint: disable=protected-access
  states, new_hand, last_action, last_actor = [], [], [], []
  for hand in hands:
    hs, acts, _ = replay.replay_hand(env, hand, fns)
    n = len(hs)
    la = np.full((n,), types.NO_ACTION, np.int32)
    lact = np.full((n,), types.NO_ACTOR, np.int32)
    if acts:
      prev = tree_map(lambda *xs: jnp.stack(xs), *hs[:-1])
      vis = np.asarray(fns.visible(prev, jnp.asarray(acts, jnp.int32)))
      la[1:] = vis[:, int(seat)]
      lact[1:] = np.asarray(fns.actors(prev))
    states.extend(hs)
    new_hand.extend([True] + [False] * (n - 1))
    last_action.append(la)
    last_actor.append(lact)
  t = len(states)
  new_match = np.zeros((t,), np.bool_)
  new_match[0] = True
  # the reference jitted seat_step, on exactly T plies
  if env not in _REF_SEAT_STEP:
    _REF_SEAT_STEP[env] = jax.jit(lambda *a: plies.seat_step(env, *a))
  steps = _REF_SEAT_STEP[env](
      tree_map(lambda *xs: jnp.stack(xs), *states),
      jnp.int32(seat),
      jnp.asarray(new_hand, jnp.bool_),
      jnp.asarray(new_match),
      jnp.asarray(np.concatenate(last_action)),
      jnp.asarray(np.concatenate(last_actor)),
  )
  return tree_map(lambda x: x[None], steps)


def _assert_tree_equal(a, b):
  """Asserts two pytrees are equal: structure, dtypes, shapes, values.

  Args:
    a: A pytree.
    b: A pytree.
  """
  la, ta = jax.tree_util.tree_flatten(a)
  lb, tb = jax.tree_util.tree_flatten(b)
  assert ta == tb, (ta, tb)
  for x, y in zip(la, lb):
    x, y = np.asarray(x), np.asarray(y)
    assert x.dtype == y.dtype and x.shape == y.shape, (
        x.dtype,
        y.dtype,
        x.shape,
        y.shape,
    )
    np.testing.assert_array_equal(x, y)


def _playout(env, rng, max_actions=100_000):
  """A random raw history of one hand to its end.

  Args:
    env: The env.
    rng: A numpy Generator.
    max_actions: The bound on the hand's length.

  Returns:
    The actions, chance outcomes included.

  Raises:
    AssertionError: The hand did not end within max_actions.
  """
  fns = replay._fns(env)  # pylint: disable=protected-access
  s, hist = fns.root(), []
  for _ in range(max_actions):
    is_chance, terminal, legal, probs = fns.node(s)
    if bool(terminal):
      return hist
    if bool(is_chance):
      p = np.asarray(probs, np.float64)
      a = int(rng.choice(p.shape[0], p=p / p.sum()))
      s = fns.apply_chance(s, jnp.int32(a))
    else:
      a = int(rng.choice(np.flatnonzero(np.asarray(legal))))
      s = fns.apply(s, jnp.int32(a))
    hist.append(a)
  raise AssertionError("playout did not end")


def _prefixes(env, hands, every=1):
  """(hands up to a decision or the end, the actor) of a match.

  Every `every`-th decision of the last hand, and the terminal end (actor -1).

  Args:
    env: the env.
    hands: the match's hands.
    every: keep every every-th decision.

  Returns:
    A list of (hands, actor).
  """
  fns = replay._fns(env)  # pylint: disable=protected-access
  done, last = [list(h) for h in hands[:-1]], list(hands[-1])
  out, s, k = [], fns.root(), 0
  for i, a in enumerate(last):
    is_chance = bool(fns.node(s)[0])
    if not is_chance:
      if k % every == 0:
        out.append((done + [last[:i]], int(fns.decision_node(s)[0])))
      k += 1
      s = fns.apply(s, jnp.int32(a))
    else:
      s = fns.apply_chance(s, jnp.int32(a))
  out.append((done + [last], -1))
  return out


def _net(game, encoder, widths=(32,), seed=0):
  """A stateless network artifact's (env, policy, params) of `game`.

  Args:
    game: The game.
    encoder: The encoder.
    widths: The hidden widths.
    seed: The init seed.

  Returns:
    (env, policy, params).
  """
  env = games.env_for(game, {})
  shape = list(enc_registry.get(encoder).shape)
  pol = policy_lib.NetworkPolicy(
      **common.network_policy_kwargs(
          game,
          encoder,
          shape,
          env.num_actions,
          env.num_players,
          catalog.mlp(list(widths), "relu", "shared"),
      )
  )
  return env, pol, pol.init_params(jax.random.key(seed))


def _game(name, seed=0):
  """(env, policy, params, hands) of a stateless network and a random match.

  Args:
    name: The game.
    seed: The seed.

  Returns:
    (env, policy, params, hands).
  """
  rng = np.random.default_rng(seed)
  if name == "tic_tac_toe":
    env, pol, params = _net(name, "tic_tac_toe_observation_tensor")
  elif name == "gin_rummy":
    env, pol, params = _net(name, "gin_rummy_observation_tensor")
  else:
    env = games.env_for(name, {})
    pol, params = helpers.network(env, "mlp")
  hands = [_playout(env, rng)]
  if name == "kuhn_poker":
    hands = [_playout(env, rng), _playout(env, rng)]
  return env, pol, params, hands


GAMES = (
    "tic_tac_toe",
    "kuhn_poker",
    "leduc_poker",
    "connect_four",
    "gin_rummy",
)
EVERY = {
    "gin_rummy": 6
}  # a long game: every 6th decision (its compiles dominate)


def test_bucket_sizes():
  """Bucket sizes."""
  assert [replay.bucket_size(n) for n in (1, 2, 3, 4, 5, 8, 9, 100)] == [
      1,
      2,
      4,
      4,
      8,
      8,
      16,
      128,
  ]
  assert replay.buckets(1) == (1,) and replay.buckets(9) == (1, 2, 4, 8, 16)
  with pytest.raises(ValueError, match="at least one ply"):
    replay.bucket_size(0)


@pytest.mark.parametrize("game", GAMES)
def test_padded_replay_equals_exact_stacking(game):
  """Padded replay equals exact stacking."""
  env, _, _, hands = _game(game)
  checked = 0
  for prefix, _ in _prefixes(env, hands, EVERY.get(game, 1)):
    for seat in range(env.num_players):
      pad = replay.replay_seat_steps_padded(env, prefix, seat)
      want = _exact_replay(env, prefix, seat)
      t = int(want.seat.shape[1])
      assert pad.num_plies == t and pad.length == replay.bucket_size(t)
      got = replay.replay_seat_steps(env, prefix, seat)
      assert got.num_plies == t and got.hand_starts == pad.hand_starts
      _assert_tree_equal(got.steps, want)
      _assert_tree_equal(pad.last, tree_map(lambda x: x[:, -1], want))
      # The stateless path's last-ply-only replay: the same last ply.
      last = replay.replay_last_seat_step(env, prefix, seat)
      assert last.num_plies == t
      _assert_tree_equal(last.last, pad.last)
      _assert_tree_equal(last.state, pad.state)
      checked += 1
  if game == "gin_rummy":
    assert max(len(h) for h in hands) >= 60, [len(h) for h in hands]
  assert checked >= 2 * env.num_players


def _unbucketed(agent, hands, seat_key):
  """(logits, value) f32 of the policy run on the exact [1, T] history.

  The unbucketed reference path: _serve_rows over the whole history.

  Args:
    agent: the HostAgent.
    hands: the match's hands.
    seat_key: the seat's PRNG key.

  Returns:
    (logits, value).
  """
  steps = _exact_replay(agent.env, hands, agent.seat)
  t = int(steps.seat.shape[1])
  init_key, eval_key, _ = agent._keys(seat_key, jnp.int64(t - 1))  # pylint: disable=protected-access
  out = agent._serve(  # pylint: disable=protected-access
      agent.params,
      steps,
      jnp.asarray([agent.seat], jnp.int32),
      init_key,
      eval_key,
  )
  return (
      np.asarray(out.logits, np.float32)[0],
      np.asarray(out.value, np.float32)[0],
  )


def _check_decisions(env, pol, params, hands, every, rule=types.SAMPLE):
  """Checks each agent's decisions against the unbucketed reference.

  On every `every`-th prefix of the hands: logits, value, probabilities and
  action.

  Args:
    env: The env.
    pol: The policy.
    params: Its params.
    hands: The hands' histories.
    every: The prefix stride.
    rule: The decision rule.

  Returns:
    The decisions checked.
  """
  agents = [
      host_agent.HostAgent(helpers.artifact(pol, params, rule), s, env=env)
      for s in range(env.num_players)
  ]
  n = 0
  for prefix, actor in _prefixes(env, hands, every):
    if actor < 0:
      continue
    agent = agents[actor]
    key = jax.random.key(17 + n)
    d = agent.act(prefix, seat_key=key)
    if d.decider == "forced":
      continue
    logits, value = _unbucketed(agent, prefix, key)
    np.testing.assert_array_equal(d.logits, logits)
    np.testing.assert_array_equal(d.value, value)
    _, eval_key_unused, decide_key = agent._keys(  # pylint: disable=protected-access
        key,
        jnp.int64(replay.replay_seat_steps(env, prefix, actor).num_plies - 1),
    )
    del eval_key_unused
    action, probs = agent._decide(  # pylint: disable=protected-access
        jnp.asarray(logits)[None],
        jnp.asarray(d.legal)[None],
        decide_key,
    )
    np.testing.assert_array_equal(d.probs, np.asarray(probs[0]))
    assert d.action == int(action[0])
    n += 1
  return n


@pytest.mark.parametrize("game", GAMES)
def test_decisions_equal_the_unbucketed_path(game):
  """Decisions equal the unbucketed path."""
  env, pol, params, hands = _game(game, seed=1)
  n = _check_decisions(env, pol, params, hands, EVERY.get(game, 1))
  assert n >= 2, n


@pytest.mark.parametrize("game", ["kuhn_poker", "leduc_poker"])
def test_recurrent_decisions_equal_the_unbucketed_path(game):
  """A stateful policy observes the exact, unpadded history."""
  env = games.env_for(game, {})
  pol, params = helpers.network(env, "rnn")
  assert pol.spec.stateful
  rng = np.random.default_rng(3)
  hands = [_playout(env, rng), _playout(env, rng)]
  assert _check_decisions(env, pol, params, hands, 1) >= 2


def _fresh_agents(name, pol, params):
  """HostAgents of every seat on a new env object.

  replay's jitted calls are cached per env, so a new env retraces them.

  Args:
    name: the game name.
    pol: the policy.
    params: its params.

  Returns:
    (env, a HostAgent per seat).
  """
  env = games.env_for(name, {})
  return env, [
      host_agent.HostAgent(helpers.artifact(pol, params), s, env=env)
      for s in range(env.num_players)
  ]


# Traces per extra bucket: the jitted seat_step, visible and actors (the
# padding is numpy, put on the device without a trace). An unbucketed
# replay traces at least this many per ply.
PER_BUCKET = 3


def test_connect_four_game_traces_log_many_shapes():
  """A long game's extra traces grow with the buckets, not the plies.

  Every decision of a 30+ ply connect_four game, with no warmup, against only
  its first 3 decisions, each on a fresh env: the extra traces grow with the
  extra buckets, not with the extra plies. (A difference, so traces made once
  per process, the env's own nested jits, cancel or favour it.)
  """
  env, pol, params, _ = _game("connect_four")
  rng = np.random.default_rng(11)
  hands = [_playout(env, rng)]
  while len(hands[0]) < 30:
    hands = [_playout(env, rng)]
  prefixes = [(p, a) for p, a in _prefixes(env, hands) if a >= 0]
  counts = []
  for n in (3, len(prefixes)):
    env_n, agents = _fresh_agents("connect_four", pol, params)
    assert env_n is not env
    with _Count() as c:
      for prefix, actor in prefixes[:n]:
        agents[actor].act(prefix, seat_key=jax.random.key(0))
    counts.append(c.traces)

  # connect_four has no chance: decision i is at T = i + 1 plies, after m = i
  # actions. New shapes of the long game: seat_step at bucket_size(T), visible
  # and actors at bucket_size(m) (m >= 1).
  def shapes(n):
    """The seat-step and visible-action bucket lengths of n decisions.

    Args:
      n: the decisions.

    Returns:
      (seat-step buckets, visible buckets).
    """
    return (
        {replay.bucket_size(i + 1) for i in range(n)},
        {replay.bucket_size(i) for i in range(1, n)},
    )

  (s_short, v_short), (s_long, v_long) = shapes(3), shapes(len(prefixes))
  bound = len(s_long - s_short) + 2 * len(v_long - v_short)
  assert len(s_long - s_short) <= len(replay.buckets(len(prefixes)))
  assert counts[1] - counts[0] <= bound, (counts, bound)
  assert counts[1] - counts[0] < len(prefixes) - 3, (counts, len(prefixes))


@pytest.mark.parametrize("game", ["connect_four", "gin_rummy"])
def test_no_decision_compiles_after_host_agent_warmup(game):
  """No decision compiles after host agent warmup."""
  env, pol, params, hands = _game(game, seed=2)
  n_plies = sum(len(h) for h in hands) + 1
  warm = []
  for max_plies in (2, n_plies):
    _, agents = _fresh_agents(game, pol, params)
    with _Count() as w:
      for a in agents:
        a.warmup(max_plies)
    warm.append(w.traces)
  extra_buckets = len(replay.buckets(n_plies)) - len(replay.buckets(2))
  assert warm[1] - warm[0] <= PER_BUCKET * extra_buckets, (warm, extra_buckets)
  env, agents = _fresh_agents(game, pol, params)
  for a in agents:
    a.warmup(n_plies)
  with _Count() as c:
    for prefix, actor in _prefixes(env, hands):
      if actor >= 0:
        agents[actor].act(prefix, seat_key=jax.random.key(1))
  assert (c.traces, c.compiles) == (0, 0), (c.traces, c.compiles)


@pytest.mark.parametrize("game", ["tic_tac_toe", "connect_four", "leduc_poker"])
@pytest.mark.parametrize("deterministic", [True, False])
def test_no_move_compiles_after_checkpoint_bot_warmup(game, deterministic):
  """No move compiles after checkpoint bot warmup."""
  if game == "tic_tac_toe":
    _, pol, params = _net(game, "tic_tac_toe_observation_tensor")
  else:
    pol, params = helpers.network(games.env_for(game, {}), "mlp")
  bot = cbh.memory_bot(pol, params, deterministic=deterministic, seed=3)
  pg = pyspiel.load_game(bot.game_string)
  bot.check_serves(pg)
  bot.warmup(range(bot.num_players), games_per_seat=0)
  with _Count() as c:
    games_played = cbh.self_play(bot, pg, deals=range(3))
  assert len(games_played) == 3
  assert (c.traces, c.compiles) == (0, 0), (c.traces, c.compiles)


def test_seats_share_compiles_and_a_warm_bot_skips_its_warmup():
  """Seats share their compiles; warming a warm bot does nothing."""
  _, pol, params = _net("tic_tac_toe", "tic_tac_toe_observation_tensor")
  bot = cbh.memory_bot(pol, params, seed=3)
  pg = pyspiel.load_game(bot.game_string)
  bot.check_serves(pg)
  name, gp = "tic_tac_toe", dict(pg.get_parameters())
  srv = bot._server  # pylint: disable=protected-access
  a0, a1 = (
      srv.agent(bot.path, bot.step, s, name, gp, rule=bot.rule) for s in (0, 1)
  )
  assert a1.env is a0.env
  assert a1._serve_last is a0._serve_last  # pylint: disable=protected-access
  with _Count() as w0:
    a0.warmup(10)
  with _Count() as w1:
    a1.warmup(10)
  assert w0.traces > 0 and (w1.traces, w1.compiles) == (0, 0), (
      w0.traces,
      w1.traces,
      w1.compiles,
  )
  assert a0.warmed_plies == a1.warmed_plies == 10
  calls = []
  real = bot.choose_action
  bot.choose_action = lambda *a: calls.append(a) or real(*a)
  with _Count() as c:
    bot.warmup(range(2))
  assert not calls, "warm agents: no playout"
  assert (c.traces, c.compiles) == (0, 0), (c.traces, c.compiles)


def test_shared_agent_must_match():
  """A shared agent must serve the same policy, env, rule and pad_rows."""
  env, pol, params, _ = _game("kuhn_poker")
  art = helpers.artifact(pol, params)
  a0 = host_agent.HostAgent(art, 0, env=env)
  with pytest.raises(ValueError, match="same policy, env"):
    host_agent.HostAgent(art, 1, env=games.env_for("kuhn_poker", {}), share=a0)
  with pytest.raises(ValueError, match="same policy, env"):
    host_agent.HostAgent(art, 1, env=env, pad_rows=1, share=a0)


def test_warmup_refuses_bad_lengths():
  """Warmup refuses bad lengths."""
  env, pol, params, _ = _game("kuhn_poker")
  agent = host_agent.HostAgent(helpers.artifact(pol, params), 0, env=env)
  with pytest.raises(ValueError, match="max_plies"):
    agent.warmup(0)
  bot = cbh.memory_bot(pol, params)
  with pytest.raises(ValueError, match="max_plies"):
    bot.warmup([0], max_plies=0)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
