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

"""Tests for contract bridge in JAX.

Move-for-move equivalence with pyspiel over random boards is in
openspiel_equivalence_test.py; this file checks known answers (scores, IMPs,
the longest auction, the duplicate cycle, the dealer-relative auction view)
and the JAX contract (jit, vmap, fixed dtypes).
"""

import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import bridge
from open_spiel.python.jax.games import bridge_reference
from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import registry
import pyspiel

_GAME_STRING = "bridge(use_double_dummy_result=false)"
P = bridge.PASS_ACTION
X = bridge.DOUBLE_ACTION
XX = bridge.REDOUBLE_ACTION


def _bid(level: int, strain: int) -> int:
  """The action id of a bid.

  Args:
    level: The level, 1..7.
    strain: The strain.

  Returns:
    The bid's action id.
  """
  return bridge.FIRST_BID_ACTION + (level - 1) * bridge.NUM_STRAINS + strain


def _longest_auction() -> list[int]:
  """The 319-call auction: every bid passed round, doubled and redoubled.

  Returns:
    The 319 call actions.
  """
  calls = [P, P, P]
  for b in range(bridge.NUM_BIDS):
    calls += [bridge.FIRST_BID_ACTION + b, P, P, X, P, P, XX, P, P]
  return calls + [P]


class BridgeTest(parameterized.TestCase):
  """The JAX bridge plays OpenSpiel's bridge.

  Attributes:
    apply: The jitted apply_action.
    game: The game.
    mask: The jitted legal_actions_mask.
  """

  def setUp(self):
    """Builds the card-play game and its jitted step."""
    super().setUp()
    self.game = bridge.Bridge(use_double_dummy_result=False)
    self.apply = jax.jit(self.game.apply_action)
    self.mask = jax.jit(self.game.legal_actions_mask)

  def _dealt(self, game=None, seed=0):
    """A state after a random deal, with the deal order used.

    Args:
      game: The game (default: the test's).
      seed: The shuffle seed.

    Returns:
      (state, deal order).
    """
    game = game or self.game
    apply = self.apply if game is self.game else jax.jit(game.apply_action)
    state = game.new_initial_state()
    order = list(range(bridge.NUM_CARDS))
    random.Random(seed).shuffle(order)
    if game.random_dealer_vul:
      state = apply(state, jnp.int32(0))
    for card in order:
      state = apply(state, jnp.int32(card))
    return state, order

  def _os_dealt(self, order, game_string=_GAME_STRING):
    """An OpenSpiel state after dealing `order`.

    Args:
      order: The deal order.
      game_string: The game string.

    Returns:
      The OpenSpiel state.
    """
    os_state = pyspiel.load_game(game_string).new_initial_state()
    for card in order:
      os_state.apply_action(card)
    return os_state

  def test_deal_is_52_chance_nodes_then_the_dealer_calls(self):
    """The deal is 52 chance nodes, then the dealer calls."""
    state = self.game.new_initial_state()
    for i in range(bridge.NUM_CARDS):
      self.assertEqual(int(state.current_player), game_lib.CHANCE_PLAYER_ID)
      probs = np.asarray(self.game.chance_outcomes(state))
      self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
      self.assertEqual(int((probs > 0).sum()), bridge.NUM_CARDS - i)
      state = self.apply(state, jnp.int32(i))
    self.assertEqual(int(state.current_player), 0)
    np.testing.assert_array_equal(
        np.asarray(state.deal), np.arange(bridge.NUM_CARDS) % 4
    )

  def test_passed_out_board_scores_zero_like_pyspiel(self):
    """A passed-out board scores zero, as in pyspiel."""
    state, order = self._dealt()
    os_state = self._os_dealt(order)
    for _ in range(4):
      state = self.apply(state, jnp.int32(P))
      os_state.apply_action(P)
    self.assertTrue(bool(state.is_terminal))
    self.assertTrue(os_state.is_terminal())
    np.testing.assert_array_equal(self.game.returns(state), os_state.returns())
    self.assertFalse(bool(np.asarray(self.mask(state)).any()))

  def test_longest_auction_then_play_matches_pyspiel(self):
    """The longest auction and its play match pyspiel."""
    state, order = self._dealt(seed=3)
    os_state = self._os_dealt(order)
    calls = _longest_auction()
    self.assertLen(calls, bridge.MAX_AUCTION_LENGTH)
    for i, call in enumerate(calls):
      legal = np.flatnonzero(np.asarray(self.mask(state)))
      self.assertEqual(legal.tolist(), os_state.legal_actions(), f"call {i}")
      self.assertEqual(int(state.current_player), os_state.current_player())
      state = self.apply(state, jnp.int32(call))
      os_state.apply_action(call)
    self.assertEqual(int(state.phase), bridge.PHASE_PLAY)
    self.assertEqual(
        (int(state.level), int(state.strain), int(state.dbl)),
        (7, bridge.NO_TRUMP, bridge.REDOUBLED),
    )
    # Play the lowest legal card every time.
    while not os_state.is_terminal():
      legal = np.flatnonzero(np.asarray(self.mask(state)))
      self.assertEqual(legal.tolist(), os_state.legal_actions())
      self.assertEqual(int(state.current_player), os_state.current_player())
      state = self.apply(state, jnp.int32(legal[0]))
      os_state.apply_action(int(legal[0]))
    self.assertTrue(bool(state.is_terminal))
    np.testing.assert_array_equal(self.game.returns(state), os_state.returns())
    history, count = bridge.action_history(state)
    self.assertEqual(int(count), bridge.MAX_GAME_LENGTH)
    self.assertEqual(
        np.asarray(history).tolist(), os_state.history()[bridge.NUM_CARDS :]
    )

  def test_declarer_plays_dummy_cards(self):
    """Declarer plays dummy's cards."""
    state, order = self._dealt(seed=5)
    # North opens 1S, all pass: North declares, East leads, South is dummy.
    for call in (_bid(1, 3), P, P, P):
      state = self.apply(state, jnp.int32(call))
    self.assertEqual(int(state.declarer), 0)
    self.assertEqual(int(state.current_player), 1)
    self.assertFalse(bool(bridge.dummy_exposed(state)))
    lead = int(np.flatnonzero(np.asarray(self.mask(state)))[0])
    state = self.apply(state, jnp.int32(lead))
    self.assertTrue(bool(bridge.dummy_exposed(state)))
    self.assertEqual(int(bridge.dummy(state)), 2)
    # South (dummy) is due; North acts with South's cards.
    self.assertEqual(int(state.on_move), 2)
    self.assertEqual(int(state.current_player), 0)
    legal = np.flatnonzero(np.asarray(self.mask(state)))
    self.assertTrue(all(order.index(c) % 4 == 2 for c in legal))

  @parameterized.parameters(
      # (level, strain, dbl, tricks, vul, declarer score)
      (1, 4, 0, 7, False, 90),
      (3, 4, 0, 9, False, 400),
      (3, 4, 0, 9, True, 600),
      (4, 2, 0, 10, True, 620),
      (6, 3, 0, 12, False, 980),
      (7, 4, 0, 13, True, 2220),
      (1, 0, 1, 7, False, 140),
      (2, 3, 1, 8, False, 470),
      (1, 4, 2, 8, False, 760),
      (3, 4, 0, 11, False, 460),
      (4, 3, 0, 9, False, -50),
      (4, 3, 0, 8, True, -200),
      (4, 3, 1, 9, False, -100),
      (4, 3, 1, 7, False, -500),
      (4, 3, 1, 6, False, -800),
      (4, 3, 1, 8, True, -500),
      (4, 3, 2, 9, True, -400),
      (7, 4, 2, 0, True, -7600),
      (0, 0, 0, 0, False, 0),
  )
  def test_score_known_answers(self, level, strain, dbl, tricks, vul, want):
    """score() gives the known answers.

    Args:
      level: The contract level.
      strain: The strain.
      dbl: The doubling.
      tricks: The tricks taken.
      vul: Whether declarer is vulnerable.
      want: The expected score.
    """
    self.assertEqual(int(bridge.score(level, strain, dbl, tricks, vul)), want)

  def test_imps_known_answers(self):
    """imps() gives the known answers."""
    diffs = [0, 10, -10, 20, 40, 50, -430, 490, 500, 3990, 4000, 7600, -7600]
    want = [0, 0, 0, 1, 1, 2, -10, 10, 11, 23, 24, 24, -24]
    np.testing.assert_array_equal(bridge.imps(jnp.asarray(diffs)), want)

  def test_duplicate_cycle_covers_every_dealer_and_vulnerability(self):
    """The 16-board cycle covers every dealer and vulnerability."""
    pairs = {bridge.board_dealer_vul(b) for b in range(16)}
    self.assertLen(pairs, 16)
    # Board 1: North deals, none vulnerable; board 2: East, NS vulnerable;
    # board 3: South, EW vulnerable; board 4: West, all vulnerable.
    self.assertEqual(bridge.board_dealer_vul(0), (0, (False, False)))
    self.assertEqual(bridge.board_dealer_vul(1), (1, (True, False)))
    self.assertEqual(bridge.board_dealer_vul(2), (2, (False, True)))
    self.assertEqual(bridge.board_dealer_vul(3), (3, (True, True)))

  def test_random_dealer_vul_draws_a_cycle_position(self):
    """random_dealer_vul draws a cycle position first."""
    game = bridge.Bridge(use_double_dummy_result=False, random_dealer_vul=True)
    apply = jax.jit(game.apply_action)
    s0 = game.new_initial_state()
    probs = np.asarray(game.chance_outcomes(s0))
    np.testing.assert_allclose(probs[:16], 1 / 16)
    self.assertEqual(float(probs[16:].sum()), 0.0)
    for b in range(16):
      s = apply(s0, jnp.int32(b))
      dealer, vul = bridge.board_dealer_vul(b)
      self.assertEqual(int(s.dealer), dealer)
      self.assertEqual(tuple(np.asarray(s.vul).tolist()), vul)
      self.assertEqual(int(s.current_player), game_lib.CHANCE_PLAYER_ID)
      self.assertEqual(
          int((np.asarray(game.chance_outcomes(s)) > 0).sum()), bridge.NUM_CARDS
      )
    with self.assertRaises(ValueError):
      bridge.Bridge(
          use_double_dummy_result=False, random_dealer_vul=True, dealer=2
      )

  def test_fixed_dealer_vul_matches_pyspiel(self):
    """A fixed dealer and vulnerability match pyspiel."""
    string = (
        "bridge(use_double_dummy_result=false,dealer=3,dealer_vul=true,"
        "non_dealer_vul=false)"
    )
    game = registry.load_game(string)
    state, order = self._dealt(game=game, seed=7)
    os_state = self._os_dealt(order, string)
    self.assertEqual(int(state.current_player), os_state.current_player())
    np.testing.assert_array_equal(np.asarray(state.vul), [False, True])
    apply = jax.jit(game.apply_action)
    for call in (_bid(1, 0), X, P, P, P):
      state = apply(state, jnp.int32(call))
      os_state.apply_action(call)
    rng = random.Random(0)
    while not os_state.is_terminal():
      legal = np.flatnonzero(np.asarray(game.legal_actions_mask(state)))
      self.assertEqual(legal.tolist(), os_state.legal_actions())
      a = int(rng.choice(legal.tolist()))
      state = apply(state, jnp.int32(a))
      os_state.apply_action(a)
    np.testing.assert_array_equal(game.returns(state), os_state.returns())

  def test_auction_view_is_dealer_relative(self):
    """The auction view is dealer-relative."""
    # East deals and passes: each player sees that pass at East's seat
    # relative to its own, (1 - p) % 4. Checked against pyspiel when its
    # view is dealer-relative too (bridge_reference).
    game_string = "bridge(use_double_dummy_result=false,dealer=1)"
    game = bridge.Bridge(use_double_dummy_result=False, dealer=1)
    state, order = self._dealt(game=game, seed=1)
    state = game.apply_action(state, jnp.int32(P))  # East passes
    for p in range(4):
      obs = np.asarray(game.observation_tensor(state, jnp.int32(p)))
      opening_passes = obs[bridge.NUM_OBSERVATION_TYPES + 4 :][:4]
      np.testing.assert_array_equal(opening_passes, np.eye(4)[(1 - p) % 4])
    if not bridge_reference.pyspiel_auction_view_is_dealer_relative():
      self.skipTest(bridge_reference.SKIP_REASON)
    os_state = self._os_dealt(order, game_string)
    os_state.apply_action(P)
    for p in range(4):
      np.testing.assert_array_equal(
          np.asarray(game.observation_tensor(state, jnp.int32(p))),
          np.asarray(os_state.observation_tensor(p), np.float32),
      )

  def test_matches_pyspiel_observation_when_north_deals(self):
    """The observation equals pyspiel's when North deals."""
    state, order = self._dealt(seed=11)
    os_state = self._os_dealt(order)
    obs = jax.jit(self.game.observation_tensor)
    rng = random.Random(1)
    while not os_state.is_terminal():
      for p in range(4):
        np.testing.assert_array_equal(
            obs(state, jnp.int32(p)),
            np.asarray(os_state.observation_tensor(p), np.float32),
        )
      legal = np.flatnonzero(np.asarray(self.mask(state))).tolist()
      calls = [a for a in legal if a >= P]
      a = P if calls and rng.random() < 0.6 else rng.choice(legal)
      state = self.apply(state, jnp.int32(a))
      os_state.apply_action(int(a))

  def test_jit_vmap_and_fixed_dtypes(self):
    """The game runs under jit and vmap with fixed dtypes."""
    states = jax.vmap(lambda _: self.game.new_initial_state())(jnp.arange(3))
    keys = jax.random.split(jax.random.key(0), 3)
    perms = jax.vmap(lambda k: jax.random.permutation(k, 52))(keys)

    def deal(s, perm):
      """Deals `perm` into `s`.

      Args:
        s: The state.
        perm: The card order.

      Returns:
        The dealt state.
      """
      return jax.lax.fori_loop(
          0, 52, lambda i, st: self.game.apply_action(st, perm[i]), s
      )

    dealt = jax.jit(jax.vmap(deal))(states, perms)
    self.assertTrue(bool((dealt.phase == bridge.PHASE_AUCTION).all()))
    s0 = self.game.new_initial_state()
    s1 = self.apply(s0, np.int64(5))
    for name, a, b in zip(bridge.GameState._fields, s0, s1):
      self.assertEqual(a.dtype, b.dtype, name)
      self.assertEqual(a.shape, b.shape, name)
    obs = jax.jit(jax.vmap(self.game.observation_tensor))(
        dealt, jnp.zeros(3, jnp.int32)
    )
    self.assertEqual(obs.shape, (3, 571))

  def test_registered_and_refuses_double_dummy(self):
    """The bridge game is registered and refuses double-dummy results."""
    self.assertIn("bridge", registry.registered_names())
    self.assertIsInstance(registry.load_game(_GAME_STRING), bridge.Bridge)
    with self.assertRaises(ValueError):
      registry.load_game("bridge(use_double_dummy_result=true)")
    # OpenSpiel's default bridge is the double-dummy game (terminal after the
    # auction); the default load refuses rather than playing another game.
    self.assertTrue(
        pyspiel.load_game("bridge")
        .get_type()
        .parameter_specification["use_double_dummy_result"]
    )
    with self.assertRaisesRegex(ValueError, "use_double_dummy_result=false"):
      registry.load_game("bridge")
    with self.assertRaisesRegex(ValueError, "use_double_dummy_result=false"):
      bridge.Bridge()
    self.assertEqual(
        self.game.obs_shape,
        tuple(pyspiel.load_game(_GAME_STRING).observation_tensor_shape()),
    )
    self.assertEqual(
        self.game.max_game_length,
        pyspiel.load_game(_GAME_STRING).max_game_length(),
    )


if __name__ == "__main__":
  absltest.main()
