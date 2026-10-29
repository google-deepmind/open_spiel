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

"""Tests for SearchPolicy.

Known answers on tiny explicit trees (toy_games) whose
minimax and expectimax values are worked out by hand, including chance nodes
and both discount cases (the seat changes, -1; the same seat moves again
after chance, +1), and Connect Four positions with a forced win in one and a
forced block. Also: the Policy contract (check_policy), dtypes under x64,
PIMC averaging, the privacy boundary (search reads InfoState only), Arena
integration and the registry rebuild.
"""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.belief import perfect_information
from open_spiel.python.scale.belief import registry as belief_registry
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies.tests import search_fixtures
from open_spiel.python.scale.policies.tests import toy_games
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import steps as steps_lib

Q = search.AUX_SEARCH_Q
TP = types.AUX_TARGET_POLICY
SV = search.AUX_SEARCH_VALUE
tree_map = jax.tree_util.tree_map
# Short aliases of the shared fixtures (functions, so pylint's variable rule).
# pylint: disable=invalid-name
_c4_net = search_fixtures.c4_net
_cfg = search_fixtures.cfg
_leduc_search = search_fixtures.leduc_search
_leduc_state = search_fixtures.leduc_state
_root_step = search_fixtures.root_step
_run = search_fixtures.run
_toy_node = search_fixtures.toy_node
_toy_root = search_fixtures.toy_root
_uniform = search_fixtures.uniform_policy
# pylint: enable=invalid-name


# --- tiny trees with chance nodes: exact values -------------------------------


def test_chance_node_value_is_the_exact_expectation():
  """Chance node value is the exact expectation.

  The afterstate (chance node) value is sum_o p(o) (r_o + d_o v(s_o)):
  for the coin gamble .3 * (+1) + .7 * (-1) = -0.4 exactly, chance logits
  log(.3), log(.7); the safe action reaches no chance node (one outcome, p 1).
  """
  env = toy_games.coin()
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(),
  )
  node = _toy_node(env)
  # Reaches private _decision_fn of the code under test.
  out, _ = pol._decision_fn(  # pylint: disable=protected-access
      {}, jax.random.key(0), jnp.asarray([1], jnp.int32), node
  )
  np.testing.assert_allclose(
      np.asarray(out.afterstate_value), [-0.4], atol=1e-6
  )
  np.testing.assert_allclose(
      np.exp(np.asarray(out.chance_logits[0])), [0.3, 0.7], atol=1e-6
  )
  # Reaches private _decision_fn of the code under test.
  out0, _ = pol._decision_fn(  # pylint: disable=protected-access
      {}, jax.random.key(0), jnp.asarray([0], jnp.int32), node
  )
  np.testing.assert_allclose(
      np.asarray(out0.afterstate_value), [0.1], atol=1e-6
  )
  np.testing.assert_allclose(
      np.exp(np.asarray(out0.chance_logits[0])), [1.0, 0.0], atol=1e-7
  )


def test_chance_node_is_an_expectation_not_a_max():
  """Chance node is an expectation not a max."""
  env = toy_games.coin()
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=40),
  )
  out = _run(pol, env, _toy_root(env), 0)
  q = np.asarray(out.aux[Q][0])
  w = np.asarray(out.aux[TP][0])
  assert np.argmax(w) == 0, w  # safe +0.1 beats the gamble (EV -0.4)
  np.testing.assert_allclose(q[0], 0.1, atol=1e-6)
  assert q[1] < 0.0, q  # a max over outcomes would give +1
  np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)


# MCTS Q is a mean over visits (it includes the exploration of losing lines),
# so these known answers are the best move and the ordering/sign of Q; 1000
# simulations let the visit counts catch up with the values on these trees.


def test_seat_change_after_chance_is_discount_minus_one():
  """Seat change after chance is discount minus one."""
  env = toy_games.coin_reply()
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=1000),
  )
  out = _run(pol, env, _toy_root(env), 0)
  q, w = np.asarray(out.aux[Q][0]), np.asarray(out.aux[TP][0])
  # Q(a0) = -0.15 < Q(a1) = 0 (wrong sign: +0.9)
  assert np.argmax(w) == 1, (w, q)
  np.testing.assert_allclose(q[1], 0.0, atol=1e-6)
  assert q[0] < 0.0, q


def test_same_seat_after_chance_is_discount_plus_one():
  """Same seat after chance is discount plus one."""
  env = toy_games.coin_again()
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=1000),
  )
  out = _run(pol, env, _toy_root(env), 0)
  q, w = np.asarray(out.aux[Q][0]), np.asarray(out.aux[TP][0])
  # Q(a0) = -0.1 > Q(a1) = -0.3 (wrong sign: -1)
  assert np.argmax(w) == 0, (w, q)
  np.testing.assert_allclose(q[1], -0.3, atol=1e-6)
  assert -0.3 < q[0] < 0.0, q


def test_zero_sum_minimax_on_a_deterministic_tree():
  """Zero sum minimax on a deterministic tree."""
  env = toy_games.reply()
  for kind in ("muzero", "gumbel"):
    pol = search.SearchPolicy(
        _uniform(env),
        env,
        perfect_information.PerfectInformationSampler(env),
        _cfg(num_simulations=200, search=kind),
    )
    out = _run(pol, env, _toy_root(env), 0)
    q, w = np.asarray(out.aux[Q][0]), np.asarray(out.aux[TP][0])
    assert np.argmax(w) == 1, (kind, w, q)  # minimax: -1 vs +0.3
    assert q[0] < 0.0 < q[1] and abs(q[1] - 0.3) < 0.05, (kind, q)


def test_single_agent_folds_the_other_seat_into_dynamics():
  """Single agent folds the other seat into dynamics."""
  env = toy_games.reply()
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=32, mode="single_agent", opponent_rule="argmax"),
  )
  out = _run(pol, env, _toy_root(env), 0)
  q, w = np.asarray(out.aux[Q][0]), np.asarray(out.aux[TP][0])
  # The folded seat plays argmax of uniform logits = its first legal action:
  # every visit of a0 returns +1 and of a1 +0.5 (terminal rewards, discount 0).
  np.testing.assert_allclose(q, [1.0, 0.5], atol=1e-6)
  assert np.argmax(w) == 0, (w, q)
  v = np.asarray(out.value[0])
  assert v[1] == 0.0  # single_agent value layout: the searching seat only


def test_stochastic_tree_refuses_gumbel():
  """Stochastic tree refuses gumbel."""
  env = toy_games.coin()
  with pytest.raises(ValueError, match="gumbel"):
    search.SearchPolicy(
        _uniform(env),
        env,
        perfect_information.PerfectInformationSampler(env),
        _cfg(search="gumbel"),
    )


# --- Connect Four: forced moves -----------------------------------------------


def _c4(actions):
  """Connect Four after `actions`, batched.

  Args:
    actions: The actions.

  Returns:
    (env, state).
  """
  env = connect_four.ConnectFourEnv()
  s = env.init(jax.random.key(0))
  for a in actions:
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return env, tree_map(lambda x: x[None], s)


@pytest.mark.parametrize("kind", ["muzero", "gumbel"])
# win in one
@pytest.mark.parametrize(
    "actions,seat,best",
    [
        ((0, 1, 0, 1, 0, 6), 0, 0),
        # block or lose
        ((0, 1, 0, 1, 0), 1, 0),
    ],
)
def test_c4_forced_moves(kind, actions, seat, best):
  """Connect four forced moves."""
  env, s = _c4(actions)
  assert int(env.actor(tree_map(lambda x: x[0], s))) == seat
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=200, search=kind),
  )
  out = _run(pol, env, s, seat)
  w = np.asarray(out.aux[TP][0])
  assert np.argmax(w) == best, w


# --- contract, dtypes, temperature --------------------------------------------


def _two_params(pol):
  """Two different params of `pol`.

  Args:
    pol: The policy.

  Returns:
    (params a, params b).
  """
  pa = pol.init_params(jax.random.key(1))
  return pa, tree_map(
      lambda x: x + 0.25 if jnp.issubdtype(x.dtype, jnp.floating) else x, pa
  )


def test_check_policy_stateless():
  """Check policy stateless."""
  env = connect_four.ConnectFourEnv()
  pol = search.SearchPolicy(
      _c4_net(),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=8),
  )
  assert not pol.spec.stateful
  step1, step2 = steps_lib.two_ply_steps(env, 4, seat=0, seed=2)
  check_policy.check_policy(pol, *_two_params(pol), step1, step2)


def test_search_policy_owns_no_temperature():
  """Search policy owns no temperature.

  No temperature fields on MCTSConfig; the schedule is the
  Arena slot's (core.arena.TemperatureDrop).
  """
  names = {f.name for f in dataclasses.fields(search.MCTSConfig)}
  assert not {n for n in names if "temperature" in n}, names


def test_dtypes_and_logits_at_temperature_one():
  """Dtypes and logits at temperature one."""
  env, s = _c4((3,))
  pol = search.SearchPolicy(
      _c4_net(),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=8),
  )
  params = pol.init_params(jax.random.key(0))
  step = _root_step(env, s, 1)
  c = pol.init_carry(params, 1, step.seat, jax.random.key(1))
  outs = []
  for ply in range(3):
    st = step._replace(
        new_hand=jnp.asarray([ply == 0]), new_match=jnp.asarray([ply == 0])
    )
    c = pol.observe(params, c, st)
    outs.append(pol.evaluate(params, c, st, jax.random.key(5)))
  for o in outs:
    assert o.logits.dtype == jnp.float32 and o.value.dtype == jnp.float32
    assert o.aux[TP].dtype == jnp.float32 and o.aux[Q].dtype == jnp.float32
  w = np.asarray(outs[0].aux[TP][0])
  # Known answer: log(w + 1e-8) in f32, the same at every ply.
  want = np.log(w.astype(np.float32) + np.float32(1e-8)).astype(np.float32)
  for o in outs:
    # XLA vs numpy log
    np.testing.assert_allclose(np.asarray(o.logits[0]), want, rtol=1e-6)
  v = np.asarray(outs[0].value[0])
  np.testing.assert_allclose(v[1], np.asarray(outs[0].aux[SV][0]), atol=1e-6)
  np.testing.assert_allclose(v[0], -v[1], atol=1e-6)


def test_not_to_move_gives_zeros():
  """Not to move gives zeros."""
  env, s = _c4((3,))
  pol = search.SearchPolicy(
      _c4_net(),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=4),
  )
  out = _run(pol, env, s, 0)  # seat 1 is to move
  assert not np.asarray(out.logits).any() and not np.asarray(out.value).any()
  assert not np.asarray(out.aux[TP]).any()


def test_registry_rebuild_gives_the_same_policy():
  """Registry rebuild gives the same policy."""
  env = connect_four.ConnectFourEnv()
  pol = search.SearchPolicy(
      _c4_net(),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=8),
  )
  again = registry.build(search.ARCH, pol.spec.arch_kwargs)
  assert again.spec == pol.spec
  _, s = _c4((3, 3))
  params = pol.init_params(jax.random.key(0))
  a, b = _run(pol, env, s, 0, params=params), _run(
      again, env, s, 0, params=params
  )
  np.testing.assert_array_equal(np.asarray(a.logits), np.asarray(b.logits))


# --- leduc: PIMC, stochastic chance node, privacy -----------------------------


def test_pimc_averages_root_visits_and_values_over_worlds():
  """Pimc averages root visits and values over worlds."""
  env, pol = _leduc_search(k=4)
  s = _leduc_state(env, (0, 4), (1,))  # seat 1 to act after a check
  params = pol.init_params(jax.random.key(0))
  step = _root_step(env, s, 1)
  c = pol.observe(
      params, pol.init_carry(params, 1, step.seat, jax.random.key(1)), step
  )
  res = pol.search(params, c, step, jax.random.key(3))
  assert res.weights.shape == (1, 4, 3) and res.values.shape == (1, 4)
  out = pol.evaluate(params, c, step, jax.random.key(3))
  np.testing.assert_allclose(
      np.asarray(out.aux[TP]), np.asarray(res.weights.mean(axis=1)), atol=1e-6
  )
  np.testing.assert_allclose(
      np.asarray(out.aux[SV]), np.asarray(res.values.mean(axis=1)), atol=1e-6
  )
  # Each world's root is the seat's InfoState (never the true hidden card).
  infos = jax.vmap(env.info_state, in_axes=(0, None))(
      tree_map(lambda x: x[0], res.worlds), jnp.int32(1)
  )
  for got, want in zip(
      jax.tree_util.tree_leaves(infos), jax.tree_util.tree_leaves(step.info)
  ):
    np.testing.assert_array_equal(
        np.asarray(got), np.broadcast_to(np.asarray(want), got.shape)
    )


def test_round2_chance_node_in_the_leduc_tree():
  """Round2 chance node in the leduc tree.

  A round-1 call that closes the round leads to the public-card chance
  node: the search must expand it (visits finite, values bounded by the pot).
  """
  env, pol = _leduc_search(k=2, sims=32)
  s = _leduc_state(env, (0, 4), (1,))  # seat 1: call closes round 1 -> chance
  out = _run(pol, env, s, 1)
  w = np.asarray(out.aux[TP][0])
  # fold illegal
  assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-5 and w[0] == 0.0
  assert np.isfinite(np.asarray(out.aux[Q])).all()
  assert abs(float(out.aux[SV][0])) <= 13.0


def test_search_reads_infostate_only():
  """Search reads infostate only.

  Privacy: two worlds that differ only in the opponent's card (the seat's
  InfoState is identical) give identical search outputs under the same key.
  """
  env, pol = _leduc_search(k=3)
  params = pol.init_params(jax.random.key(0))
  outs = []
  for opp in (1, 3, 5):
    s = _leduc_state(env, (0, opp), (1, 2))  # seat 0 holds J0; check, raise
    outs.append(_run(pol, env, s, 0, key=9, params=params))
  for o in outs[1:]:
    for a, b in zip(
        jax.tree_util.tree_leaves(o), jax.tree_util.tree_leaves(outs[0])
    ):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_leduc_search_seat_in_the_arena():
  """Leduc search seat in the arena."""
  env, pol = _leduc_search(k=2, sims=8)
  opp = _uniform(env)
  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(pol, 1, types.SAMPLE),
          arena_lib.Slot(opp, 1, types.SAMPLE),
      ],
  )
  table = arena_lib.rotated_table(2, seat_slots=[0, 1], seat_members=[0, 0])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(0), 2, 2)
  params = [tree_map(lambda x: x[None], pol.init_params(jax.random.key(1))), {}]
  # raises on a bad ply
  _, batch, _ = ar.collect(params, table, dk, pk, num_steps=12)
  tp = np.asarray(batch.target_policy)
  valid = np.asarray(batch.valid)
  assert valid.any()
  np.testing.assert_allclose(tp[valid].sum(-1), 1.0, atol=1e-5)
  assert not tp[~valid].any()
  legal = np.asarray(batch.steps.legal_mask)
  assert not (tp[valid] * ~legal[valid]).any()


def test_c4_self_play_in_the_arena_emits_search_targets():
  """Connect four self play in the arena emits search targets."""
  env = connect_four.ConnectFourEnv()
  pol = search.SearchPolicy(
      _c4_net(),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=4),
  )
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  table = arena_lib.self_play_table(2, 2, all_seats=True)
  dk, pk = arena_lib.independent_keys(jax.random.key(0), 2, 2)
  params = [tree_map(lambda x: x[None], pol.init_params(jax.random.key(1)))]
  _, batch, _ = ar.collect(params, table, dk, pk, num_steps=6)
  valid = np.asarray(batch.valid)
  # every ply is one seat's decision (no terminal in 6 plies)
  assert valid.sum() == 2 * 6
  np.testing.assert_allclose(
      np.asarray(batch.target_policy)[valid].sum(-1), 1.0, atol=1e-5
  )


# --- registry -----------------------------------------------------------------


def test_search_is_a_builtin_architecture():
  """Search is a builtin architecture."""
  assert search.ARCH in registry.registered()


# --- EnvSpec identity, the sampler registry, non-acting rows --

_SEEN = []  # rows of each RecordingSampler.sample_worlds call


class RecordingSampler(perfect_information.PerfectInformationSampler):
  """PerfectInformationSampler that records the rows it samples worlds for.

  The rows are recorded by a host callback.
  """

  def sample_worlds(self, params, carry, info, k, key):
    # one call per searched row (a multi-row search runs each row alone under
    # vmap, and a vmapped debug callback runs once per row): rows in node's
    # leading axis, recorded per call
    """Records the rows, then samples as the parent.

    Args:
      params: the params.
      carry: the carry.
      info: the InfoState, leaves [B, ...].
      k: worlds per row.
      key: a PRNG key.

    Returns:
      The parent's (worlds, log-weights).
    """
    jax.debug.callback(
        lambda node: _SEEN.append(int(np.shape(node)[0])), info.node
    )
    return super().sample_worlds(params, carry, info, k, key)


# Named per module copy: `python <file>` runs this module as __main__ and
# pytest then imports it again, each copy with its own RecordingSampler class.
belief_registry.register(f"test_recording.{__name__}", RecordingSampler)


def test_arch_kwargs_record_the_env_spec_and_rebuild_its_options():
  """Architecture kwargs record the env spec and rebuild its options.

  Known answer: coin_scaled(scale=2) doubles every return, so the rebuilt
  policy's Q(safe) is 0.2 (scale 1 would give 0.1): the env's options survive
  the registry rebuild.
  """
  env = env_registry.make("toy_coin_scaled", scale=2)
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=40),
  )
  assert pol.spec.arch_kwargs["env"] == {
      "name": "toy_coin_scaled",
      "kwargs": {"scale": 2.0},
  }
  again = registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)
  assert env_registry.spec_of(again.env) == env_registry.spec(
      "toy_coin_scaled", scale=2.0
  )
  assert again.spec == pol.spec
  q = np.asarray(_run(again, again.env, _toy_root(again.env), 0).aux[Q][0])
  np.testing.assert_allclose(q[0], 0.2, atol=1e-6)


def test_arch_kwargs_record_the_sampler_by_name_and_the_full_mcts_config():
  """Architecture kwargs record the sampler by name and the full mcts config."""
  env = toy_games.coin()
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(),
  )
  kw = pol.spec.arch_kwargs
  assert kw["sampler"] == {"name": "perfect_information", "kwargs": {}}
  assert kw["mcts"] == dataclasses.asdict(_cfg())
  assert registry.build(pol.spec.arch_name, kw).spec == pol.spec


def test_unregistered_env_and_sampler_are_refused():
  """Unregistered env and sampler are refused."""
  nodes = [
      toy_games.node("p0", (1, 2)),
      toy_games.node("term", returns=(1.0, -1.0)),
      toy_games.node("term", returns=(-1.0, 1.0)),
  ]
  rogue = toy_games.TreeGameEnv(
      "toy_unregistered",
      "nowhere:none",
      nodes,
      num_actions=2,
      max_chance_outcomes=0,
      max_len=1,
  )
  with pytest.raises(KeyError, match="no env named"):
    search.SearchPolicy(
        _uniform(rogue),
        rogue,
        perfect_information.PerfectInformationSampler(rogue),
        _cfg(),
    )

  class Unregistered(perfect_information.PerfectInformationSampler):
    """A sampler class that is never registered."""

    pass

  env = toy_games.coin()
  with pytest.raises(KeyError, match="not a registered belief sampler"):
    search.SearchPolicy(_uniform(env), env, Unregistered(env), _cfg())


def _rows_step(env, seats):
  """SeatSteps of toy roots for `seats`.

  Args:
    env: The toy env.
    seats: The seats.

  Returns:
    The SeatStep.
  """
  s = _toy_root(env, len(seats))
  return plies.seat_step(
      env,
      s,
      jnp.asarray(seats, jnp.int32),
      True,
      True,
      types.NO_ACTION,
      types.NO_ACTOR,
  )


@pytest.mark.parametrize(
    "seats,searched",
    [
        ((0, 1, 1, 1), [2]),  # one acting row: a chunk of ceil(4/2) = 2 rows
        ((0, 0, 1, 0), [4]),  # three: the full batch
        ((1, 1, 1, 1), []),  # none: no search at all
    ],
)
def test_non_acting_rows_are_not_searched(seats, searched):
  """Non acting rows are not searched.

  At toy_coin's root seat 0 moves: rows of seat 1 are non-acting. The
  sampler (called once per search, on the searched rows) sees only the chunk
  holding the acting rows; their known answer Q(safe) = 0.1 holds and the
  other rows are zeros.

  Args:
    seats: the seat of each row.
    searched: the expected sizes of the sampler's calls.
  """
  env = toy_games.coin()
  pol = search.SearchPolicy(
      _uniform(env), env, RecordingSampler(env), _cfg(num_simulations=40)
  )
  params = pol.init_params(jax.random.key(0))
  step = _rows_step(env, seats)
  carry = pol.observe(
      params,
      pol.init_carry(params, len(seats), step.seat, jax.random.key(1)),
      step,
  )
  _SEEN.clear()
  out = pol.evaluate(params, carry, step, jax.random.key(2))
  jax.block_until_ready(out)
  jax.effects_barrier()
  assert sum(_SEEN) == sum(searched), (_SEEN, searched)
  acting = np.asarray(seats) == 0
  q = np.asarray(out.aux[Q])
  np.testing.assert_allclose(q[acting, 0], 0.1, atol=1e-6)
  for x in (out.logits, out.value, out.aux[TP], out.aux[SV], out.aux[Q]):
    assert not np.asarray(x)[~acting].any()


def test_search_chunks():
  """Search chunks."""
  assert search.search_chunks(1) == (1,)
  assert search.search_chunks(4) == (2, 4)
  assert search.search_chunks(5) == (3, 5)


def test_a_rows_search_does_not_depend_on_the_other_rows():
  """A rows search does not depend on the other rows.

  owned_only gathers the acting rows into a chunk, so the search must be
  a per-row program (core.arena.owned_only): a row's noise is its own (the
  row's index folded into the key), never where it lands in the chunk.
  Case: toy reply, muzero, 2 simulations,
  dirichlet_fraction 0.9, every row the same root; row 3's search with seats
  (0, 0, 0, 0) and with (1, 1, 1, 0) (rows 0..2 not acting, so row 3 is
  gathered to the front of a smaller chunk) is the same, bit for bit, and
  equals row 3's search with every row acting.
  """
  env = toy_games.reply()
  pol = search.SearchPolicy(
      _uniform(env),
      env,
      perfect_information.PerfectInformationSampler(env),
      _cfg(num_simulations=2, dirichlet_fraction=0.9),
  )
  params = pol.init_params(jax.random.key(0))
  mover, other = 0, 1  # p0 moves at reply's root
  outs = {}
  for seats in (
      (mover,) * 4,
      (other, other, other, mover),
      (other, mover, other, mover),
  ):
    step = _rows_step(env, seats)
    carry = pol.observe(
        params, pol.init_carry(params, 4, step.seat, jax.random.key(1)), step
    )
    outs[seats] = pol.evaluate(params, carry, step, jax.random.key(5))
  ref = outs[(mover,) * 4]
  for seats, out in outs.items():
    for x, y in zip(
        jax.tree_util.tree_leaves((out.logits, out.aux[TP], out.aux[Q])),
        jax.tree_util.tree_leaves((ref.logits, ref.aux[TP], ref.aux[Q])),
    ):
      np.testing.assert_array_equal(
          np.asarray(x)[3], np.asarray(y)[3], err_msg=str(seats)
      )
  # the noise is live: rows with the same input differ (their own keys)
  tp = np.asarray(ref.aux[TP])
  assert any(not np.array_equal(tp[0], tp[i]) for i in (1, 2, 3)), tp


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
