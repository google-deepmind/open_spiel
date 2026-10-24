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

"""Arena integration over components that pass alone.

The components: kuhn, uniform, tabular, HistoryPolicy, CallbackTabularPolicy,
decide.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import chance
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core.tests import arena_fixtures
from open_spiel.python.scale.core.tests import named_fns
from open_spiel.python.scale.core.tests import numpy_policies
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import fixtures

tree_map = jax.tree_util.tree_map
PASS, BET = 0, 1
KUHN_INDEX = f"{kuhn.__name__}:info_index"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The kuhn env, shared by the module.

  Returns:
    a KuhnEnv.
  """
  return kuhn.KuhnEnv()


def _stack(params, k=1):
  """Stacks `params` k times on a new leading axis.

  Args:
    params: The params.
    k: The copies.

  Returns:
    The stacked params.
  """
  return tree_map(lambda x: jnp.stack([x] * k), params)


def _uniform(env):
  """A uniform Kuhn policy.

  Args:
    env: The Kuhn env.

  Returns:
    The policy.
  """
  return uniform.UniformPolicy("kuhn_poker", env.num_actions, env.num_players)


def _tabular(env, index_fn=KUHN_INDEX):
  """A Kuhn tabular policy.

  Args:
    env: The env.
    index_fn: The info-state index function.

  Returns:
    The policy.
  """
  return tabular.TabularPolicy(
      "kuhn_poker",
      kuhn.NUM_INFO_STATES,
      env.num_actions,
      env.num_players,
      index_fn,
  )


def _always(pol, action):
  """Tabular params that play `action` everywhere.

  Args:
    pol: The tabular policy.
    action: The action.

  Returns:
    The params.
  """
  logits = jnp.zeros((kuhn.NUM_INFO_STATES, 2)).at[:, action].set(10.0)
  return pol.params_from_logits(logits)


def _history(env, **kw):
  """A Kuhn HistoryPolicy.

  Args:
    env: The Kuhn env.
    **kw: HistoryPolicy options.

  Returns:
    The policy.
  """
  return fixtures.HistoryPolicy(
      "kuhn_poker", env.num_actions, env.num_players, **kw
  )


def _hand_returns(log, seat):
  """Per-hand returns of `seat` over all rows (done plies only).

  Args:
    log: The arena log.
    seat: The seat.

  Returns:
    The returns of `seat` at each hand's end.
  """
  r = np.asarray(log.rewards)[..., seat]
  return r[np.asarray(log.done)]


def _hand_ply(nh):
  """Each ply's index in its hand.

  Args:
    nh: bool[B, T], the new_hand flags.

  Returns:
    int64[B, T].
  """
  b, t = nh.shape
  hp = np.zeros((b, t), np.int64)
  for i in range(1, t):
    hp[:, i] = np.where(nh[:, i], 0, hp[:, i - 1] + 1)
  return hp


def test_uniform_self_play_known_mean(env):
  """Uniform vs uniform kuhn: E[p0] = 1/8 exactly (by hand); E[p0+p1] = 0."""
  pol = _uniform(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 1024, 64
  table = arena_lib.self_play_table(b, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(0), b, 2)
  _, batch, log = ar.collect([_stack({})], table, dk, pk, t)
  r0 = _hand_returns(log, 0)
  assert r0.size > 10000
  se = r0.std() / np.sqrt(r0.size)
  assert abs(r0.mean() - 0.125) < 4 * se, (r0.mean(), se)
  np.testing.assert_allclose(_hand_returns(log, 0) + _hand_returns(log, 1), 0.0)
  # Shapes of the emitted batch.
  assert batch.action.shape == (b, t) and batch.behavior_logits.shape == (
      b,
      t,
      2,
  )
  assert batch.value_pred.shape == (b, t, 2) and batch.reward.shape == (b, t)
  assert batch.behavior_logp.shape == (b, t) and batch.done.shape == (b, t)
  assert batch.steps.info.card.shape == (b, t)
  assert batch.bootstrap_step.info.card.shape == (b,)
  assert bool(
      np.asarray(batch.steps.new_hand)[:, 0].all()
  ), "a fresh reset starts at new_hand"
  np.testing.assert_array_equal(np.asarray(batch.done), np.asarray(log.done))


def test_duplicate_deals_share_deals_and_reduce_variance(env):
  """Seat-swapped duplicate rows share every deal.

  Always-bet (slot 0) vs uniform (slot 1): E[slot 0, as p0] = 1/2 and E[as p1]
  = 1/4 (hand computed), so a duplicate pair's hand sum has mean 3/4, and
  pairing removes card luck: the paired variance is below the variance of
  unpaired sums.

  Args:
    env: the kuhn env fixture.
  """
  tab = _tabular(env)
  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(tab, 1, types.ARGMAX),
          arena_lib.Slot(_uniform(env), 1, types.SAMPLE),
      ],
  )
  n = 512
  table = arena_lib.rotated_table(
      n, seat_slots=[0, 1], seat_members=[0, 0], perspective=0
  )
  np.testing.assert_array_equal(table.seat_slot[:2], [[0, 1], [1, 0]])
  np.testing.assert_array_equal(table.perspective_seat[:2], [0, 1])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(1), n, 2)
  _, batch, _ = ar.collect(
      [_stack(_always(tab, BET)), _stack({})], table, dk, pk, 48
  )

  own = np.asarray(batch.steps.info.card)
  opp = np.asarray(batch.labels["opp_card"])
  seat = np.asarray(batch.steps.seat)
  nh = np.asarray(batch.steps.new_hand)
  done = np.asarray(batch.done)
  rew = np.asarray(batch.reward)
  p0 = np.where(seat == 0, own, opp)
  p1 = np.where(seat == 0, opp, own)
  paired, unpaired = [], []
  hand_rets = []
  for i in range(n):
    deals, rets = [], []
    for row in (2 * i, 2 * i + 1):
      deals.append(list(zip(p0[row][nh[row]], p1[row][nh[row]])))
      rets.append(rew[row][done[row]])
    m = min(len(deals[0]), len(deals[1]), len(rets[0]), len(rets[1]))
    assert m >= 8
    assert deals[0][:m] == deals[1][:m], f"deal {i} differs across the swap"
    hand_rets.append((rets[0][:8], rets[1][:8]))
  for i in range(n):
    a, b_ = hand_rets[i]
    paired.extend(a + b_)
    unpaired.extend(a + hand_rets[(i + 1) % n][1])
  paired, unpaired = np.array(paired), np.array(unpaired)
  se = paired.std() / np.sqrt(paired.size)
  assert abs(paired.mean() - 0.75) < 4 * se, (paired.mean(), se)
  assert paired.var() < 0.9 * unpaired.var(), (paired.var(), unpaired.var())


def test_exact_equal_under_same_keys(env):
  """Equal keys give bit-equal rollouts; other policy keys change the actions.

  Args:
    env: the kuhn env fixture.
  """
  pol = _uniform(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  table = arena_lib.self_play_table(64, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(2), 64, 2)
  out1 = ar.collect([_stack({})], table, dk, pk, 32)
  out2 = ar.collect([_stack({})], table, dk, pk, 32)
  for x, y in zip(
      jax.tree_util.tree_leaves(out1[1:]), jax.tree_util.tree_leaves(out2[1:])
  ):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  _, pk2 = arena_lib.independent_keys(jax.random.key(3), 64, 2)
  out3 = ar.collect([_stack({})], table, dk, pk2, 32)
  assert not np.array_equal(
      np.asarray(out1[2].action), np.asarray(out3[2].action)
  )
  # Same deal keys -> same first deal regardless of policy keys.
  np.testing.assert_array_equal(
      np.asarray(out1[1].steps.info.card[:, 0]),
      np.asarray(out3[1].steps.info.card[:, 0]),
  )


def test_row_independent_of_batch_composition(env):
  """A row's actions do not depend on the other rows of its batch.

  Args:
    env: the kuhn env fixture.
  """
  pol = _uniform(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  dk, pk = arena_lib.independent_keys(jax.random.key(4), 16, 2)
  _, full, _ = ar.collect(
      [_stack({})], arena_lib.self_play_table(16, 2), dk, pk, 20
  )
  _, half, _ = ar.collect(
      [_stack({})], arena_lib.self_play_table(8, 2), dk[8:], pk[8:], 20
  )
  np.testing.assert_array_equal(
      np.asarray(full.action[8:]), np.asarray(half.action)
  )


def test_always_bet_known_returns_and_terminal_ply(env):
  """Both seats always bet (argmax).

  each hand is bet, bet (done), then one observe-only terminal ply; every hand
  is a ±2 showdown by card.

  Args:
    env: the kuhn env fixture.
  """
  pol = _tabular(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.ARGMAX)])
  table = arena_lib.SlotTable(
      np.zeros((32, 2), np.int32),
      np.zeros((32, 2), np.int32),
      np.zeros((32,), np.int32),
  )
  dk, pk = arena_lib.independent_keys(jax.random.key(5), 32, 2)
  _, batch, log = ar.collect([_stack(_always(pol, BET))], table, dk, pk, 21)
  done, term = np.asarray(log.done), np.asarray(log.terminal_ply)
  assert (
      done[:, 1::3].all()
      and not done[:, 0::3].any()
      and not done[:, 2::3].any()
  )
  assert (
      term[:, 2::3].all()
      and not term[:, 0::3].any()
      and not term[:, 1::3].any()
  )
  act = np.asarray(log.action)
  assert (act[~term] == BET).all() and (act[term] == -1).all()
  assert (np.asarray(log.actor)[term] == -1).all() and (
      np.asarray(log.slot)[term] == -1
  ).all()
  own = np.asarray(batch.steps.info.card)
  opp = np.asarray(batch.labels["opp_card"])
  expected = 2.0 * np.sign(own - opp)
  np.testing.assert_array_equal(np.asarray(batch.reward)[done], expected[done])
  np.testing.assert_array_equal(np.asarray(batch.reward)[~done], 0.0)
  # The terminal ply: to_move -1, nothing legal, invalid, sees the ending bet
  # and the terminal InfoState (two bets).
  st = batch.steps
  assert (np.asarray(st.to_move)[term] == -1).all()
  assert (
      not np.asarray(st.legal_mask)[term].any()
      and not np.asarray(batch.valid)[term].any()
  )
  assert (np.asarray(st.last_action)[term] == BET).all()
  assert (np.asarray(st.last_actor)[term] == 1).all()
  assert (np.asarray(st.info.num_bets)[term] == 2).all()
  # The next hand's first ply: NO_ACTION / NO_ACTOR.
  nh = np.asarray(st.new_hand)
  assert (np.asarray(st.last_action)[nh] == types.NO_ACTION).all()
  assert (np.asarray(st.last_actor)[nh] == types.NO_ACTOR).all()


def test_observe_every_ply_and_logits_zero_off_acting_rows(env):
  """HistoryPolicy logits[0] = observes since the hand began = hand ply + 1.

  Emitted logits are zero where the perspective seat did not act.

  Args:
    env: the kuhn env fixture.
  """
  pol = _history(env)
  params = pol.make_params(b=jnp.array([0.0, 0.5]))
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.ARGMAX)])
  b, t = 8, 12
  table = arena_lib.self_play_table(b, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(6), b, 2)
  _, batch, log = ar.collect([_stack(params)], table, dk, pk, t)
  hand_ply = _hand_ply(np.asarray(batch.steps.new_hand))
  valid = np.asarray(batch.valid)
  assert valid.any() and (~valid).any()
  np.testing.assert_array_equal(
      valid,
      np.asarray(log.actor) == np.asarray(table.perspective_seat)[:, None],
  )
  bl = np.asarray(batch.behavior_logits)
  np.testing.assert_array_equal(bl[..., 0][valid], (hand_ply + 1)[valid])
  assert (bl[~valid] == 0).all()
  assert (
      np.asarray(batch.behavior_logp) == 0
  ).all(), "argmax behaviour is deterministic"
  # The non-acting seat's legal mask is hidden.
  lm = np.asarray(batch.steps.legal_mask)
  assert not lm[~valid].any() and lm[valid].any(axis=-1).all()


def test_observe_seq_replay_equals_live(env):
  """Stateless replay of the perspective seat's steps reproduces live logits."""
  pol = _history(env)
  params = pol.make_params(
      w=jnp.array([[0.3, -0.2], [0.1, 0.4]]), b=jnp.array([0.0, 1.5])
  )
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 8, 10
  dk, pk = arena_lib.independent_keys(jax.random.key(7), b, 2)
  _, batch, _ = ar.collect(
      [_stack(params)], arena_lib.self_play_table(b, 2), dk, pk, t
  )
  valid = np.asarray(batch.valid)
  for k in range(t):
    prefix = tree_map(lambda x, k=k: x[:, : k + 1], batch.steps)
    c = pol.observe_seq(params, batch.init_carry, prefix)
    step_k = tree_map(lambda x, k=k: x[:, k], batch.steps)
    lg = np.asarray(pol.evaluate(params, c, step_k, None).logits)
    np.testing.assert_allclose(
        lg[valid[:, k]],
        np.asarray(batch.behavior_logits[:, k])[valid[:, k]],
        rtol=0,
        atol=1e-6,
    )
  # evaluate_seq over the whole segment gives the same logits on valid plies.
  _, outs = pol.evaluate_seq(params, batch.init_carry, batch.steps, None)
  np.testing.assert_allclose(
      np.asarray(outs.logits)[valid],
      np.asarray(batch.behavior_logits)[valid],
      atol=1e-6,
  )


def test_terminal_ply_lets_a_policy_see_how_the_hand_ended(env):
  """2-hand match, both seats always bet.

  replaying the perspective seat's steps through hand 1's terminal ply gives
  hist = [0, 2] (both bets seen, including the hand-ending one).

  Args:
    env: the kuhn env fixture.
  """
  pol = _history(env)
  params = pol.make_params(b=jnp.array([0.0, 10.0]))  # always BET under argmax
  ar = arena_lib.Arena(
      env, [arena_lib.Slot(pol, 1, types.ARGMAX)], hands_per_match=2
  )
  b = 4
  dk, pk = arena_lib.independent_keys(jax.random.key(13), b, 2)
  _, batch, log = ar.collect(
      [_stack(params)], arena_lib.self_play_table(b, 2), dk, pk, 6
  )
  term = np.asarray(log.terminal_ply)
  np.testing.assert_array_equal(np.flatnonzero(term[0]), [2, 5])
  c = pol.observe_seq(
      params, batch.init_carry, tree_map(lambda x: x[:, :3], batch.steps)
  )
  np.testing.assert_array_equal(
      np.asarray(c["hist"]), np.tile([[0.0, 2.0]], (b, 1))
  )
  nh, nm = np.asarray(batch.steps.new_hand), np.asarray(batch.steps.new_match)
  assert (
      nh[:, 3].all() and not nm[:, 3].any()
  ), "hand 2 opens without a new match"


def test_behavior_logp_matches_empirical_frequencies(env):
  """SAMPLE at T=2 over logits [0, log 9].

  probs [1/4, 3/4]. behavior_logp is the log-probability of the action taken
  under that tempered distribution, and the empirical BET frequency matches
  it.

  Args:
    env: the kuhn env fixture.
  """
  pol = _tabular(env)
  logits = jnp.zeros((kuhn.NUM_INFO_STATES, 2)).at[:, BET].set(jnp.log(9.0))
  ar = arena_lib.Arena(
      env, [arena_lib.Slot(pol, 1, types.DecisionRule("sample", 2.0))]
  )
  b, t = 512, 24
  dk, pk = arena_lib.independent_keys(jax.random.key(14), b, 2)
  _, batch, _ = ar.collect(
      [_stack(pol.params_from_logits(logits))],
      arena_lib.self_play_table(b, 2),
      dk,
      pk,
      t,
  )
  valid = np.asarray(batch.valid)
  act = np.asarray(batch.action)[valid]
  lp = np.asarray(batch.behavior_logp)[valid]
  np.testing.assert_allclose(np.exp(lp[act == BET]), 0.75, atol=1e-6)
  np.testing.assert_allclose(np.exp(lp[act == PASS]), 0.25, atol=1e-6)
  freq = (act == BET).mean()
  assert abs(freq - 0.75) < 4 * np.sqrt(0.75 * 0.25 / act.size), (
      freq,
      act.size,
  )
  assert (np.asarray(batch.behavior_logp)[~valid] == 0).all()


def test_stacked_opponents_indexed_per_row(env):
  """Slot 1 stacks K=2 tabular opponents [always pass, always bet]."""
  pol = _tabular(env)
  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(pol, 1, types.ARGMAX),
          arena_lib.Slot(pol, 2, types.ARGMAX),
      ],
  )
  b = 16
  member = (np.arange(b) % 2).astype(np.int32)
  table = arena_lib.SlotTable(
      seat_slot=np.tile(np.array([[0, 1]], np.int32), (b, 1)),
      seat_member=np.stack([np.zeros(b, np.int32), member], axis=1),
      perspective_seat=np.zeros(b, np.int32),
  )
  opp = tree_map(
      lambda x, y: jnp.stack([x, y]), _always(pol, PASS), _always(pol, BET)
  )
  dk, pk = arena_lib.independent_keys(jax.random.key(8), b, 2)
  _, _, log = ar.collect([_stack(_always(pol, PASS)), opp], table, dk, pk, 24)
  actor, action = np.asarray(log.actor), np.asarray(log.action)
  for row in range(b):
    np.testing.assert_array_equal(action[row][actor[row] == 1], member[row])
    np.testing.assert_array_equal(action[row][actor[row] == 0], PASS)
    np.testing.assert_array_equal(
        np.asarray(log.member)[row][actor[row] == 1], member[row]
    )


def test_callback_batched_slot_matches_native(env):
  """A CALLBACK_BATCHED slot plays exactly like the same NATIVE_JAX tables.

  The slot runs host numpy tables through pure_callback, K=2, member ids in
  params; the NATIVE_JAX slot is the same tables as a tabular policy.

  Args:
    env: the kuhn env fixture.
  """
  tab = _tabular(env)
  cb = fixtures.CallbackTabularPolicy(
      "kuhn_poker", kuhn.NUM_INFO_STATES, 2, 2, KUHN_INDEX
  )
  rng = np.random.default_rng(0)
  tables = rng.normal(size=(2, kuhn.NUM_INFO_STATES, 2)).astype(np.float32)
  cb.set_host_tables(tables)
  b = 16
  member = (np.arange(b) % 2).astype(np.int32)
  table = arena_lib.SlotTable(
      seat_slot=np.tile(np.array([[0, 1]], np.int32), (b, 1)),
      seat_member=np.stack([np.zeros(b, np.int32), member], axis=1),
      perspective_seat=np.zeros(b, np.int32),
  )
  dk, pk = arena_lib.independent_keys(jax.random.key(15), b, 2)
  p0 = _stack(_always(tab, PASS))
  native = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(tab, 1, types.SAMPLE),
          arena_lib.Slot(tab, 2, types.SAMPLE),
      ],
  )
  _, _, log_n = native.collect(
      [p0, {"logits": jnp.asarray(tables)}], table, dk, pk, 24
  )
  called = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(tab, 1, types.SAMPLE),
          arena_lib.Slot(cb, 2, types.SAMPLE),
      ],
  )
  _, _, log_c = called.collect(
      [p0, {"member": jnp.arange(2, dtype=jnp.int32)}], table, dk, pk, 24
  )
  np.testing.assert_array_equal(
      np.asarray(log_c.action), np.asarray(log_n.action)
  )


def test_mixed_architectures_and_no_retrace_across_params(env):
  """Two architectures share an Arena; new params do not retrace.

  Args:
    env: the kuhn env fixture.
  """
  calls = {"n": 0}

  class Counting(fixtures.HistoryPolicy):
    """A HistoryPolicy that counts its evaluate traces."""

    def evaluate(self, params, carry, step, key):
      """Counts the trace, then evaluates as HistoryPolicy.

      Args:
        params: the policy params.
        carry: the carry.
        step: the SeatStep.
        key: the evaluate key.

      Returns:
        HistoryPolicy's PolicyOutput.
      """
      calls["n"] += 1
      return super().evaluate(params, carry, step, key)

  hist = Counting("kuhn_poker", env.num_actions, env.num_players)
  ar = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(hist, 1, types.SAMPLE),
          arena_lib.Slot(_uniform(env), 1, types.SAMPLE),
      ],
  )
  table = arena_lib.rotated_table(4, [0, 1], [0, 0])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(9), 4, 2)
  _, batch, log = ar.collect(
      [_stack(hist.init_params()), _stack({})], table, dk, pk, 8
  )
  n_after_first = calls["n"]
  assert n_after_first >= 1
  ar.collect(
      [_stack(hist.make_params(b=jnp.array([1.0, -1.0]))), _stack({})],
      table,
      dk,
      pk,
      8,
  )
  assert calls["n"] == n_after_first, "params change retraced the rollout"
  # Per-row outputs of the mixed table: slot 0 (HistoryPolicy, c = e0) acts
  # on the perspective seat's plies with logits[0] = hand ply + 1.
  valid = np.asarray(batch.valid)
  np.testing.assert_array_equal(np.asarray(log.slot)[valid], 0)
  hp = _hand_ply(np.asarray(batch.steps.new_hand))
  np.testing.assert_array_equal(
      np.asarray(batch.behavior_logits)[..., 0][valid], (hp + 1)[valid]
  )
  opp = (np.asarray(log.actor) >= 0) & ~valid
  np.testing.assert_array_equal(np.asarray(log.slot)[opp], 1)


def test_validate_fails_loud(env):
  """Arena.validate raises on bad members, perspective and unstacked params.

  Args:
    env: the kuhn env fixture.
  """
  pol = _tabular(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1), arena_lib.Slot(pol, 2)])
  p = [_stack(pol.init_params()), _stack(pol.init_params(), 2)]
  good = arena_lib.rotated_table(2, [0, 1], [0, 1])
  ar.validate(p, good)
  bad_member = good._replace(seat_member=np.full_like(good.seat_member, 2))
  bad_persp = good._replace(perspective_seat=1 - good.perspective_seat)
  for bad in (bad_member, bad_persp):
    with pytest.raises(ValueError):
      ar.validate(p, bad)
  with pytest.raises(ValueError, match="stacked"):
    ar.validate([pol.init_params(), p[1]], good)


def test_host_driven_slot_rejected_by_in_graph_arena(env):
  """The in-graph Arena refuses a HOST_DRIVEN slot.

  Args:
    env: the kuhn env fixture.
  """
  host_pol = numpy_policies.NumpyHistoryPolicy("kuhn_poker", 2, 2)
  with pytest.raises(AssertionError, match="HostArena"):
    arena_lib.Arena(env, [arena_lib.Slot(host_pol, 1)])


def test_out_of_range_tabular_index_fails_loud(env):
  """An out-of-range tabular index is a bad ply.

  Args:
    env: the kuhn env fixture.
  """
  pol = _tabular(env, named_fns.OOB_INDEX)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.ARGMAX)])
  dk, pk = arena_lib.independent_keys(jax.random.key(16), 8, 2)
  with pytest.raises(arena_lib.BadPlyError, match="bad plies"):
    ar.collect(
        [_stack(pol.init_params())], arena_lib.self_play_table(8, 2), dk, pk, 4
    )


def test_arena_source_continues_hands(env):
  """ArenaSource keeps its state.

  with T = 2 < hand length 3 (bet, bet, terminal), call 2 starts on hand 1's
  terminal ply, and call 1's bootstrap_step is call 2's first step.

  Args:
    env: the kuhn env fixture.
  """
  pol = _tabular(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.ARGMAX)])
  table = arena_lib.self_play_table(8, 2)
  src = arena_lib.ArenaSource(
      ar, [_stack(_always(pol, BET))], table, num_steps=2
  )
  b1 = src.next(jax.random.key(10))
  b2 = src.next(jax.random.key(11))
  assert isinstance(b1, types.TrajectoryBatch) and b1.action.shape == (8, 2)
  assert np.asarray(b1.done)[:, 1].all() and not np.asarray(b1.done)[:, 0].any()
  assert (
      np.asarray(b2.steps.to_move)[:, 0] == -1
  ).all(), "call 2 did not continue the hand"
  assert np.asarray(b2.steps.new_hand)[:, 1].all()
  for x, y in zip(
      jax.tree_util.tree_leaves(b1.bootstrap_step),
      jax.tree_util.tree_leaves(tree_map(lambda z: z[:, 0], b2.steps)),
  ):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
  src.set_params([_stack(_always(pol, PASS))])
  src.restart()
  b3 = src.next(jax.random.key(12))
  assert (np.asarray(b3.action)[:, 0] == PASS).all() and np.asarray(
      b3.steps.new_hand
  )[:, 0].all()


def test_rollout_continuation_equals_one_rollout(env):
  """Two rollouts of 6 plies equal one rollout of 12.

  Args:
    env: the kuhn env fixture.
  """
  pol = _uniform(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1)])
  table = arena_lib.self_play_table(8, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(11), 8, 2)
  st = ar.reset([_stack({})], table, dk, pk)
  st, a, _ = ar.rollout([_stack({})], table, st, 6)
  _, b2, _ = ar.rollout([_stack({})], table, st, 6)
  _, whole, _ = ar.collect([_stack({})], table, dk, pk, 12)
  np.testing.assert_array_equal(
      np.concatenate([np.asarray(a.action), np.asarray(b2.action)], 1),
      np.asarray(whole.action),
  )


def test_all_seats_perspective_keeps_every_decision(env):
  """With every seat a perspective seat, every decision is a valid row.

  Args:
    env: the kuhn env fixture.
  """
  pol = _uniform(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 8, 16
  dk, pk = arena_lib.independent_keys(jax.random.key(17), b, 2)
  table = arena_lib.self_play_table(b, 2, all_seats=True)
  _, batch, log = ar.collect([_stack({})], table, dk, pk, t)
  assert batch.action.shape == (2 * b, t)
  valid = np.asarray(batch.valid).reshape(b, 2, t)
  np.testing.assert_array_equal(
      valid.sum(1), (np.asarray(log.actor) >= 0).astype(int)
  )
  np.testing.assert_array_equal(
      np.asarray(batch.steps.seat).reshape(b, 2, t)[:, 1], 1
  )


def test_private_actions_hidden_by_visible_action(env):
  """An env whose seat-1 actions are private to seat 1.

  seat 0's last_action is NO_ACTION after a seat-1 action, seat 1 sees its own
  action.

  Args:
    env: the kuhn env fixture.
  """
  del env  # unused: the test builds its own env

  class PrivateP1(kuhn.KuhnEnv):
    """Kuhn with seat 1's actions private to seat 1."""

    def visible_action(self, s, a, seat):
      """Hides seat 1's actions from seat 0.

      Args:
        s: the state.
        a: the action.
        seat: the seat.

      Returns:
        NO_ACTION for seat 0 after a seat-1 action, else a.
      """
      hidden = (self.actor(s) == 1) & (seat == 0)
      return jnp.where(hidden, types.NO_ACTION, jnp.asarray(a, jnp.int32))

  penv = PrivateP1()
  pol = _uniform(penv)
  ar = arena_lib.Arena(penv, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 16, 12
  dk, pk = arena_lib.independent_keys(jax.random.key(18), b, 2)
  _, batch, _ = ar.collect(
      [_stack({})], arena_lib.self_play_table(b, 2, all_seats=True), dk, pk, t
  )
  la = np.asarray(batch.steps.last_action).reshape(b, 2, t)
  lactor = np.asarray(batch.steps.last_actor).reshape(b, 2, t)
  after_p1 = lactor[:, 0] == 1
  assert after_p1.any()
  assert (la[:, 0][after_p1] == types.NO_ACTION).all()
  assert (la[:, 1][after_p1] >= 0).all()
  after_p0 = lactor[:, 0] == 0
  np.testing.assert_array_equal(la[:, 0][after_p0], la[:, 1][after_p0])


def test_aux_targets_reach_the_batch(env):
  """A policy emitting aux target_policy/target_value fills the batch.

  TrajectoryBatch.target_* (a SearchPolicy's visits and root value) is filled
  where the policy acted; other aux keys land in batch.aux.

  Args:
    env: the kuhn env fixture.
  """

  class WithTargets(uniform.UniformPolicy):
    """A uniform policy that emits aux targets."""

    def evaluate(self, params, carry, step, key):
      """Uniform evaluation plus aux targets.

      Args:
        params: the policy params.
        carry: the carry.
        step: the SeatStep.
        key: the evaluate key.

      Returns:
        the PolicyOutput with aux targets.
      """
      out = super().evaluate(params, carry, step, key)
      b = out.logits.shape[0]
      return out._replace(
          aux={
              types.AUX_TARGET_POLICY: jnp.full((b, 2), 0.5, jnp.float32),
              types.AUX_TARGET_VALUE: jnp.full((b, 2), 3.0, jnp.float32),
              "extra": jnp.full((b,), 7.0, jnp.float32),
          }
      )

  pol = WithTargets("kuhn_poker", 2, 2)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  dk, pk = arena_lib.independent_keys(jax.random.key(19), 8, 2)
  _, batch, _ = ar.collect(
      [_stack({})], arena_lib.self_play_table(8, 2), dk, pk, 10
  )
  valid = np.asarray(batch.valid)
  assert (np.asarray(batch.target_policy)[valid] == 0.5).all()
  assert (np.asarray(batch.target_policy)[~valid] == 0).all()
  assert (np.asarray(batch.target_value)[valid] == 3.0).all()
  assert set(batch.aux) == {"extra"}
  assert (np.asarray(batch.aux["extra"])[valid] == 7.0).all()


def test_non_finite_teacher_value_is_a_bad_ply(env):
  """A NaN aux target_value on a decided row is a bad ply.

  A non-float32 target_value is refused at trace time.

  Args:
    env: the kuhn env fixture.
  """

  def with_value(v):
    """A uniform teacher whose target_value is v.

    Args:
      v: (value, dtype) of the emitted target_value.

    Returns:
      the teacher policy.
    """

    class Teacher(uniform.UniformPolicy):
      """A uniform policy emitting a constant target_value."""

      def evaluate(self, params, carry, step, key):
        """Uniform evaluation plus the constant target_value.

        Args:
          params: the policy params.
          carry: the carry.
          step: the SeatStep.
          key: the evaluate key.

        Returns:
          the PolicyOutput with target_value.
        """
        out = super().evaluate(params, carry, step, key)
        b = out.logits.shape[0]
        return out._replace(
            aux={types.AUX_TARGET_VALUE: jnp.full((b, 2), v[0], v[1])}
        )

    return Teacher("kuhn_poker", 2, 2)

  dk, pk = arena_lib.independent_keys(jax.random.key(20), 8, 2)
  ar = arena_lib.Arena(
      env, [arena_lib.Slot(with_value((jnp.nan, jnp.float32)), 1, types.SAMPLE)]
  )
  with pytest.raises(arena_lib.BadPlyError, match="bad plies"):
    ar.collect([_stack({})], arena_lib.self_play_table(8, 2), dk, pk, 4)
  ar = arena_lib.Arena(
      env, [arena_lib.Slot(with_value((1, jnp.int32)), 1, types.SAMPLE)]
  )
  with pytest.raises(TypeError, match="float32"):
    ar.collect([_stack({})], arena_lib.self_play_table(8, 2), dk, pk, 4)


def test_multi_hand_matches_and_cross_hand_carry(env):
  """With hands_per_match=3, new_match comes every third hand.

  A cross-hand HistoryPolicy (c = e0, d = e1) sees
  logits[0] = plies into the hand (resets at new_hand) and logits[1] = hand
  number within the match (resets only at new_match).

  Args:
    env: the kuhn env fixture.
  """
  pol = _history(env, cross_hand=True)
  params = pol.make_params(d=jnp.array([0.0, 1.0]))
  ar = arena_lib.Arena(
      env, [arena_lib.Slot(pol, 1, types.SAMPLE)], hands_per_match=3
  )
  b, t = 4, 30
  dk, pk = arena_lib.independent_keys(jax.random.key(12), b, 2)
  _, batch, _ = ar.collect(
      [_stack(params)], arena_lib.self_play_table(b, 2), dk, pk, t
  )
  nh, nm = np.asarray(batch.steps.new_hand), np.asarray(batch.steps.new_match)
  valid = np.asarray(batch.valid)
  bl = np.asarray(batch.behavior_logits)
  hp = _hand_ply(nh)
  for row in range(b):
    hands = np.flatnonzero(nh[row])
    matches = np.flatnonzero(nm[row])
    assert len(hands) >= 5
    np.testing.assert_array_equal(matches, hands[::3])
    hand_no = np.cumsum(nh[row])  # 1-based hand number since reset
    in_match = (hand_no - 1) % 3 + 1
    v = valid[row]
    np.testing.assert_array_equal(bl[row, v, 0], hp[row, v] + 1)
    np.testing.assert_array_equal(bl[row, v, 1], in_match[v])


def test_chance_events_keyed_by_hand_and_event_index():
  """Known answer.

  replaying a hand's actions on the host, with chance event n of hand h
  sampled from fold_in(fold_in(fold_in(deal_key, h), 0), n), gives the deal
  and the board the Arena produced.
  """
  env = arena_fixtures.Leduc()
  pol = uniform.UniformPolicy("leduc_poker", env.num_actions, env.num_players)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 8, 40
  dk, pk = arena_lib.independent_keys(jax.random.key(20), b, 2)
  _, batch, log = ar.collect(
      [_stack({})], arena_lib.self_play_table(b, 2), dk, pk, t
  )
  term = np.asarray(log.terminal_ply)
  actor, action = np.asarray(log.actor), np.asarray(log.action)
  pub, priv = np.asarray(batch.labels["public"]), np.asarray(
      batch.labels["private"]
  )
  resolve = jax.jit(lambda st, k, n: chance.resolve(env, st, k, n))
  step = jax.jit(
      lambda st, a_, k, n: chance.resolve(env, env.apply(st, a_), k, n)
  )
  boards = 0
  for row in range(b):
    ends = np.flatnonzero(term[row])
    starts = np.concatenate([[0], ends[:-1] + 1])
    for h, (a, e) in enumerate(zip(starts, ends)):
      ck = jax.random.fold_in(jax.random.fold_in(dk[row], h), 0)
      s, n = resolve(env.root(), ck, 0)
      np.testing.assert_array_equal(np.asarray(s.private_cards), priv[row, a])
      for ply in range(a, e):
        assert actor[row, ply] >= 0
        s, n = step(s, action[row, ply], ck, n)
        assert int(s.public_card) == pub[row, ply + 1], (row, h, ply)
      boards += int(n) > 2
  assert boards > 5, boards


def test_temperature_drop_known_answer(env):
  """Slot TemperatureDrop.

  the actor acts at the rule's T before the hand's ply `ply` and at
  drop.temperature from it on. Known answer: with logits (0.3, -0.4) at every
  infoset, a decision at hand ply h has logp = log softmax((0.3, -0.4) /
  T(h))[action], T(0) = 1, T(h>=1) = 0.25; behavior_logits stay the policy's
  own logits.

  Args:
    env: the kuhn env fixture.
  """
  pol = _tabular(env)
  raw = np.array([0.3, -0.4], np.float32)
  params = pol.params_from_logits(
      jnp.tile(jnp.asarray(raw)[None], (kuhn.NUM_INFO_STATES, 1))
  )
  slot = arena_lib.Slot(
      pol, 1, types.SAMPLE, temperature_drop=arena_lib.TemperatureDrop(1, 0.25)
  )
  ar = arena_lib.Arena(env, [slot])
  b, t = 64, 24
  dk, pk = arena_lib.independent_keys(jax.random.key(23), b, 2)
  table = arena_lib.self_play_table(b, 2, all_seats=True)
  _, batch, _ = ar.collect([_stack(params)], table, dk, pk, t)
  valid = np.asarray(batch.valid)
  hp = _hand_ply(np.asarray(batch.steps.new_hand))
  act = np.asarray(batch.action)
  logp = np.asarray(batch.behavior_logp)
  seen = set()
  for temp_ply, temp in ((0, 1.0), (1, 0.25), (2, 0.25)):
    z = raw.astype(np.float64) / np.float64(temp)  # f64, decide's arithmetic
    want = z - np.log(np.exp(z).sum())
    sel = valid & (hp == temp_ply)
    assert sel.any(), temp_ply
    np.testing.assert_allclose(
        logp[sel], want[act[sel]].astype(np.float32), rtol=1e-6
    )
    seen.add(temp_ply)
  bl = np.asarray(batch.behavior_logits)[valid]
  np.testing.assert_array_equal(bl, np.broadcast_to(raw, bl.shape))
  # Without the drop every ply decides at T = 1.
  ar1 = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  _, b1, _ = ar1.collect([_stack(params)], table, dk, pk, t)
  sel = np.asarray(b1.valid) & (_hand_ply(np.asarray(b1.steps.new_hand)) >= 1)
  z = raw.astype(np.float64)
  np.testing.assert_allclose(
      np.asarray(b1.behavior_logp)[sel],
      (z - np.log(np.exp(z).sum()))[np.asarray(b1.action)[sel]],
      rtol=1e-6,
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
