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

"""BridgeEnv tests.

The Env contract with its privacy hook, the InfoState layout, the acting seat at
dummy's cards, the registry row and serving.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import bridge as game
from open_spiel.python.scale.envs import bridge
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.testing import check_env
from open_spiel.python.scale.testing import infoset_invariance as twins

P = game.PASS_ACTION


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A bridge env.

  Returns:
    The env.
  """
  return bridge.BridgeEnv()


@pytest.fixture(name="plays", scope="module")
def _plays_fixture(env):
  """16 random boards, every ply: (states [16, 372], actor [16, 372]).

  Args:
    env: The bridge env.

  Returns:
    (states, actor).
  """
  states, actor, _ = twins.playouts(env, 16, env.max_len, seed=3)
  return states, actor


def _at(states, i, t):
  """Ply `t` of board `i`.

  Args:
    states: The states, leaves [boards, plies, ...].
    i: The board.
    t: The ply.

  Returns:
    The state.
  """
  return jax.tree_util.tree_map(lambda x: x[i, t], states)


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, hide_fn=bridge.hide_hidden_cards, num_playouts=16)


def test_check_env_passes_for_a_fixed_board():
  """Check env passes for a fixed board."""
  check_env.check_env(
      bridge.BridgeEnv(dealer=2, dealer_vul=True, random_dealer_vul=False),
      hide_fn=bridge.hide_hidden_cards,
      num_playouts=8,
  )


def test_info_state_matches_info_template(env, plays):
  """Info state matches info template.

  Args:
    env: The module `env` fixture.
    plays: The module `plays` fixture.
  """
  states, _ = plays
  info = env.info_state(_at(states, 0, 60), jnp.int32(1))
  want = bridge.info_template()
  assert set(info) == set(want)
  for k, v in want.items():
    assert info[k].dtype == v.dtype and info[k].shape == v.shape, k


def test_info_state_contents_against_the_state(env, plays):
  """Info state contents against the state.

  Args:
    env: The module `env` fixture.
    plays: The module `plays` fixture.
  """
  states, _ = plays
  info_fn = jax.jit(env.info_state)
  for i in range(4):
    for t in (0, 20, 80, 150, env.max_len):
      s = _at(states, i, t)
      deal = np.asarray(s.deal)
      hist, count = game.action_history(s)
      for seat in range(4):
        info = info_fn(s, jnp.int32(seat))
        np.testing.assert_array_equal(np.asarray(info["hand"]), deal == seat)
        assert int(np.asarray(info["hand"]).sum()) == 13
        shown = bool(game.dummy_exposed(s))
        assert bool(info["dummy_shown"]) == shown
        dummy = int(game.dummy(s))
        np.testing.assert_array_equal(
            np.asarray(info["dummy"]), (deal == dummy) & shown
        )
        np.testing.assert_array_equal(
            np.asarray(info["actions"]), np.asarray(hist)
        )
        assert int(info["num_actions"]) == int(count)
        assert (
            int(info["dealer"]) == int(s.dealer) and int(info["seat"]) == seat
        )
        np.testing.assert_array_equal(
            np.asarray(info["vul"]), np.asarray(s.vul)
        )


def test_declarer_acts_for_dummy(env, plays):
  """Declarer acts for dummy.

  Args:
    env: The module `env` fixture.
    plays: The module `plays` fixture.
  """
  states, actor = plays
  seen = 0
  for i in range(16):
    for t in range(env.max_len):
      s = _at(states, i, t)
      if int(s.phase) != game.PHASE_PLAY:
        continue
      on_move, declarer = int(s.on_move), int(s.declarer)
      if on_move == (declarer + 2) % 4:
        seen += 1
        assert int(actor[i, t]) == declarer
        legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
        assert legal.size and (np.asarray(s.holder)[legal] == on_move).all()
  assert seen > 0


def test_hide_keeps_counts_and_changes_hidden_cards(env, plays):
  """Hide keeps counts and changes hidden cards.

  Args:
    env: The module `env` fixture.
    plays: The module `plays` fixture.
  """
  states, _ = plays
  s = _at(states, 0, 10)  # in the auction: three hands hidden
  for seat in range(4):
    h = bridge.hide_hidden_cards(env, s, jnp.int32(seat), jax.random.key(seat))
    assert np.bincount(np.asarray(h.deal), minlength=4).tolist() == [13] * 4
    held = np.asarray(h.holder)
    assert (
        np.bincount(held[held >= 0], minlength=4).tolist()
        == np.bincount(
            np.asarray(s.holder)[np.asarray(s.holder) >= 0], minlength=4
        ).tolist()
    )
    assert (np.asarray(h.deal) != np.asarray(s.deal)).any()
    for k, v in env.info_state(h, jnp.int32(seat)).items():
      np.testing.assert_array_equal(
          np.asarray(v), np.asarray(env.info_state(s, seat)[k]), k
      )


def test_returns_and_labels(env, plays):
  """Returns and labels.

  Args:
    env: The module `env` fixture.
    plays: The module `plays` fixture.
  """
  states, _ = plays
  for i in range(16):
    s = _at(states, i, env.max_len)
    assert bool(env.terminal(s))
    r = np.asarray(env.returns(s))
    assert r.dtype == np.float32 and r.sum() == 0.0
    np.testing.assert_array_equal(r, np.asarray(s.score, np.float32))
    np.testing.assert_array_equal(
        np.asarray(env.labels(s, 0)["deal_holder"]), np.asarray(s.deal)
    )


def test_random_boards_cover_dealers_and_vulnerability(env):
  """Random boards cover dealers and vulnerability.

  Args:
    env: The module `env` fixture.
  """
  inits = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.key(0), 256))
  pairs = {
      (int(d), tuple(v.tolist()))
      for d, v in zip(np.asarray(inits.dealer), np.asarray(inits.vul))
  }
  assert len(pairs) == 16
  assert (
      np.asarray(jax.vmap(env.actor)(inits)) == np.asarray(inits.dealer)
  ).all()


def test_registry_row_options_and_covers():
  """Registry row options and covers."""
  assert "bridge" in registry.builtin_names()
  assert registry.spec("bridge").kwargs == {
      "dealer": 0,
      "dealer_vul": False,
      "non_dealer_vul": False,
      "random_dealer_vul": True,
      "compact_history": False,
  }
  fixed = registry.spec(
      "bridge", dealer=3, non_dealer_vul=True, random_dealer_vul=False
  )
  assert registry.spec_of(registry.make(fixed)) == fixed
  assert registry.covers(registry.spec("bridge"), fixed)
  assert not registry.covers(fixed, registry.spec("bridge"))
  with pytest.raises(ValueError, match="random_dealer_vul"):
    registry.spec("bridge", dealer=1)
  with pytest.raises(ValueError, match="dealer"):
    registry.spec("bridge", dealer=4, random_dealer_vul=False)
  assert registry.make("bridge").world_from_info(None) is None


def test_replays_a_pyspiel_board_through_the_raw_layer():
  """Replays a pyspiel board through the raw layer."""
  pyspiel = pytest.importorskip("pyspiel")
  env = bridge.BridgeEnv(dealer=0, random_dealer_vul=False)
  os_state = pyspiel.load_game(
      "bridge(use_double_dummy_result=false)"
  ).new_initial_state()
  s = env.root()
  rng = np.random.default_rng(0)
  while not os_state.is_terminal():
    if os_state.is_chance_node():
      a = int(rng.choice([o for o, _ in os_state.chance_outcomes()]))
      assert bool(env.is_chance(s))
      s = env.apply_chance(s, a)
    else:
      legal = os_state.legal_actions()
      assert np.flatnonzero(np.asarray(env.legal_mask(s))).tolist() == legal
      assert int(env.actor(s)) == os_state.current_player()
      a = int(rng.choice(legal)) if rng.random() > 0.7 or P not in legal else P
      s = env.apply(s, a)
    os_state.apply_action(a)
  assert bool(env.terminal(s))
  np.testing.assert_array_equal(
      np.asarray(env.returns(s)), np.asarray(os_state.returns(), np.float32)
  )


def test_compact_history_rebuilds_the_action_list_at_every_ply(env, plays):
  """Compact history rebuilds the action list at every ply.

  Args:
    env: The module `env` fixture.
    plays: The module `plays` fixture.
  """
  del env  # Unused.
  states, _ = plays
  flat = jax.tree_util.tree_map(
      lambda x: x.reshape((-1,) + x.shape[2:]), states
  )
  compact = jax.jit(jax.vmap(bridge.compact_history))(flat)
  infos = {**compact, "dealer": flat.dealer.astype(jnp.int8)}
  acts, count = jax.jit(jax.vmap(bridge.expand_history))(infos)
  want, want_count = jax.jit(jax.vmap(game.action_history))(flat)
  np.testing.assert_array_equal(np.asarray(acts), np.asarray(want))
  np.testing.assert_array_equal(np.asarray(count), np.asarray(want_count))
  # The coverage: doubled and redoubled auctions, finished boards, a pass-out.
  assert (np.asarray(flat.dbl) == 2).any() and (
      np.asarray(flat.n_played) == 52
  ).any()
  calls = np.asarray(compact["calls"])
  assert calls.dtype == np.uint8 and calls.shape[1] == bridge.MAX_NONPASS_CALLS
  assert calls[calls != bridge.PAD_U8].max() <= 4 * 36 + 3


def test_compact_history_of_a_longest_auction():
  """Compact history of a longest auction.

  The 319-call auction: three opening passes, then every bid passed twice,
  doubled, passed twice, redoubled and passed twice (three passes after
  7NT XX): 105 non-pass calls, the most the compact form holds.
  """
  env = bridge.BridgeEnv(dealer=1, random_dealer_vul=False)
  s = env.root()
  for c in range(bridge.NUM_CARDS):
    s = env.apply_chance(s, c)
  calls = [P, P, P]
  for b in range(game.NUM_BIDS):
    calls += [55 + b, P, P, 53, P, P, 54, P, P]
  calls.append(P)
  assert len(calls) == game.MAX_AUCTION_LENGTH
  for call in calls:
    assert bool(env.legal_mask(s)[call]), (len(calls), call)
    s = env.apply(s, call)
  assert int(s.phase) == game.PHASE_PLAY
  cenv = bridge.BridgeEnv(
      dealer=1, random_dealer_vul=False, compact_history=True
  )
  info = cenv.info_state(s, jnp.int32(0))
  assert (
      int((np.asarray(info["calls"]) != bridge.PAD_U8).sum())
      == bridge.MAX_NONPASS_CALLS
  )
  acts, count = bridge.expand_history(info)
  want, want_count = game.action_history(s)
  np.testing.assert_array_equal(np.asarray(acts), np.asarray(want))
  assert int(count) == int(want_count) == game.MAX_AUCTION_LENGTH


def test_compact_env_info_state_and_contract(plays):
  """Compact env info state and contract.

  Args:
    plays: The module `plays` fixture.
  """
  cenv = bridge.BridgeEnv(compact_history=True)
  states, _ = plays
  info = cenv.info_state(_at(states, 1, 90), jnp.int32(2))
  want = bridge.compact_info_template()
  assert set(info) == set(want)
  for k, v in want.items():
    assert info[k].dtype == v.dtype and info[k].shape == v.shape, k
  raw_bytes = sum(v.nbytes for v in bridge.info_template().values())
  compact_bytes = sum(v.nbytes for v in want.values())
  assert compact_bytes < raw_bytes / 2.5, (compact_bytes, raw_bytes)
  check_env.check_env(cenv, hide_fn=bridge.hide_hidden_cards, num_playouts=8)


def test_compact_history_is_an_option_not_another_game():
  """Compact history is an option not another game."""
  spec = registry.spec("bridge", compact_history=True)
  assert registry.spec_of(registry.make(spec)) == spec
  assert registry.make(spec).compact_history
  assert registry.covers(spec, registry.spec("bridge"))
  fixed = registry.spec("bridge", dealer=2, random_dealer_vul=False)
  fixed_compact = registry.spec(
      "bridge", dealer=2, random_dealer_vul=False, compact_history=True
  )
  assert registry.covers(fixed_compact, fixed) and registry.covers(
      fixed, fixed_compact
  )
  with pytest.raises(TypeError, match="compact_history"):
    registry.spec("bridge", compact_history=1)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
