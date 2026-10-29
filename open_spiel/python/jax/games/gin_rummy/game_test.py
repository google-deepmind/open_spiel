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

"""GinRummy: the Game interface, deals, turn order and dtype stability."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import gin_rummy
from open_spiel.python.jax.games.gin_rummy import game
from open_spiel.python.jax.games.gin_rummy import layoffs
import pyspiel


class GinRummySmokeTest(absltest.TestCase):
  """Smoke tests of the gin rummy game.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = gin_rummy.GinRummy()

  def test_basic_properties(self):
    """The game's sizes are gin rummy's."""
    self.assertEqual(self.game.num_players, 2)
    self.assertEqual(self.game.num_actions, gin_rummy.NUM_ACTIONS)
    self.assertEqual(self.game.obs_shape, (gin_rummy.OBSERVATION_TENSOR_SIZE,))
    self.assertEqual(self.game.max_chance_outcomes, gin_rummy.NUM_CARDS)

  def test_initial_state_is_chance_node(self):
    """The initial state is a chance node."""
    state = self.game.new_initial_state()
    self.assertTrue(bool(self.game.is_chance_node(state)))
    self.assertEqual(
        int(self.game.current_player(state)), game_lib.CHANCE_PLAYER_ID
    )

  def test_chance_outcomes_normalized(self):
    """The chance outcomes sum to 1."""
    state = self.game.new_initial_state()
    probs = self.game.chance_outcomes(state)
    self.assertEqual(probs.shape, (gin_rummy.NUM_CARDS,))
    self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

  def test_apply_action_jit(self):
    """`apply_action` must trace cleanly under jit."""
    jitted = jax.jit(self.game.apply_action)
    state = self.game.new_initial_state()
    state2 = jitted(state, jnp.int32(0))
    self.assertEqual(int(state2["phase"]), gin_rummy.PHASE_DEAL)

  def test_random_rollout_to_terminal(self):
    """A random rollout reaches a terminal state."""
    rng = np.random.default_rng(7)
    state = self.game.new_initial_state()
    steps = 0
    while not bool(state["done"]):
      if bool(self.game.is_chance_node(state)):
        p = np.asarray(self.game.chance_outcomes(state))
        action = int(rng.choice(p.size, p=p))
      else:
        mask = np.asarray(self.game.legal_actions_mask(state))
        legal = np.flatnonzero(mask)
        self.assertNotEmpty(
            legal,
            f"no legal actions at step {steps}, phase {int(state['phase'])}",
        )
        action = int(rng.choice(legal))
      state = self.game.apply_action(state, jnp.int32(action))
      steps += 1
      self.assertLess(steps, 1000, "rollout exceeded step cap")
    self.assertTrue(bool(state["done"]))
    returns = np.asarray(self.game.returns(state))
    self.assertEqual(returns.shape, (2,))
    self.assertAlmostEqual(float(returns.sum()), 0.0, places=4)

  def test_observation_tensor_shape_and_dtype(self):
    """The observation tensor has the right shape and dtype."""
    state = self.game.new_initial_state()
    rng = np.random.default_rng(0)
    while bool(self.game.is_chance_node(state)) and not bool(state["done"]):
      p = np.asarray(self.game.chance_outcomes(state))
      action = int(rng.choice(p.size, p=p))
      state = self.game.apply_action(state, jnp.int32(action))
    obs = self.game.observation_tensor(state, jnp.int32(0))
    self.assertEqual(obs.shape, (gin_rummy.OBSERVATION_TENSOR_SIZE,))
    self.assertEqual(obs.dtype, jnp.float32)

  def test_observation_tensor_uses_state_knock_card(self):
    """The observation tensor shows the state's knock card."""
    state = self.game.new_initial_state()
    state = dict(state)
    state["knock_card"] = jnp.int8(4)
    obs = np.asarray(self.game.observation_tensor(state, jnp.int32(0)))
    knock_start = 2 + 2 * gin_rummy.NUM_CARDS + 2
    np.testing.assert_array_equal(
        obs[knock_start : knock_start + gin_rummy.KNOCK_THRESHOLD],
        np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    )

  def test_observation_from_os_matches_pyspiel(self):
    """observation_from_os equals pyspiel's observation tensor."""
    os_game = pyspiel.load_game("gin_rummy")
    os_state = os_game.new_initial_state()
    actions = [
        1,
        4,
        5,
        6,
        17,
        18,
        19,
        30,
        31,
        32,
        2,
        3,
        16,
        29,
        43,
        44,
        45,
        7,
        20,
        33,
        0,
        52,
        55,
        1,
    ]
    for action in actions:
      os_state.apply_action(action)

    np.testing.assert_array_equal(
        np.asarray(self.game.observation_from_os(os_state, 0)),
        np.asarray(os_state.observation_tensor(0), dtype=np.float32),
    )

  def test_information_state_tensor_not_implemented(self):
    """information_state_tensor raises NotImplementedError."""
    state = self.game.new_initial_state()
    with self.assertRaises(NotImplementedError):
      self.game.information_state_tensor(state, jnp.int32(0))

  def _first_upcard_deal_with_first_player(self, first_player):
    """A dealt FIRST_UPCARD state whose first-to-act player is ``first_player``.

    init_dealt_state randomizes ``first_player``, so search seeds
    deterministically for one that matches.

    Args:
      first_player: The player who must act first.

    Returns:
      The state.
    """
    for seed in range(64):
      state, _ = game.init_dealt_state(jax.random.PRNGKey(seed), False, False)
      if (
          int(state["current_player"]) == first_player
          and int(state["phase"]) == gin_rummy.PHASE_FIRST_UPCARD
      ):
        return state
    self.fail(f"no dealt state with first_player={first_player} in 64 seeds")

  def test_both_pass_stock_draw_goes_to_first_to_act_player(self):
    """After two passes, the first stock draw goes to the first player.

    After both players pass the first upcard, the first stock draw goes to the
    player who acted first, for either first player. `init_state` always deals
    player 0 first and `init_dealt_state` deals either, so both are dealt here
    and the both-pass line is played directly.
    """
    for first_player in (0, 1):
      state = self._first_upcard_deal_with_first_player(first_player)
      # First-to-act passes (pass_count 0 -> 1, turn switches to the opponent).
      state = game.step(state, jnp.int32(gin_rummy.ACTION_PASS))
      self.assertEqual(int(state["pass_count"]), 1)
      self.assertEqual(int(state["current_player"]), 1 - first_player)
      # Opponent passes too (both passed): the drawer is set for the stock draw.
      state = game.step(state, jnp.int32(gin_rummy.ACTION_PASS))
      self.assertEqual(int(state["pass_count"]), 2)
      self.assertEqual(
          int(state["phase"]),
          gin_rummy.PHASE_FIRST_UPCARD,
          "both-pass stays in FirstUpcard until draw_stock",
      )
      self.assertEqual(
          int(state["current_player"]),
          first_player,
          "the first stock draw must go to the first-to-act player",
      )


class RandomizedKnockLimitTest(absltest.TestCase):
  """Randomized knock limit decoupled from the upcard."""

  def _deal_many(self, n, *, oklahoma, randomized, seed=0):
    """Deals `n` hands and reads their knock cards and upcards.

    Args:
      n: The hands.
      oklahoma: Whether Oklahoma rules hold.
      randomized: Whether the knock limit is randomised.
      seed: The seed.

    Returns:
      (knock cards, upcards), each of length n.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    deal = jax.vmap(
        lambda k: gin_rummy.init_dealt_state(k, oklahoma, randomized)[0]
    )(keys)
    return np.asarray(deal["knock_card"]), np.asarray(deal["upcard"])

  def test_standard_oklahoma_is_special_case(self):
    """Without randomization the knock limit is the upcard's."""
    # randomized_knock_limit=False gives the upcard-derived limit exactly.
    knock, upcard = self._deal_many(256, oklahoma=True, randomized=False)
    expected = np.asarray(
        jax.vmap(game.oklahoma_knock_card)(jnp.asarray(upcard))
    )
    np.testing.assert_array_equal(knock, expected)

  def test_randomized_independent_of_upcard(self):
    """With randomization one upcard maps to several knock limits."""
    # Under randomization a single upcard value must map to more than one
    # knock card across deals — the defining property the upcard-derived
    # path (one upcard -> one limit) can never satisfy.
    knock, upcard = self._deal_many(512, oklahoma=True, randomized=True)
    saw_multi = False
    for u in np.unique(upcard):
      if np.unique(knock[upcard == u]).size > 1:
        saw_multi = True
        break
    self.assertTrue(
        saw_multi,
        "randomized knock limit is not decoupled from the upcard: every "
        "upcard mapped to a single limit",
    )

  def test_randomized_covers_full_range_uniformly(self):
    """Randomized knock limits cover 0..KNOCK_THRESHOLD roughly uniformly."""
    # Uniform over {0..KNOCK_THRESHOLD}: every discrete limit should appear,
    # unlike the upcard-rank-induced distribution (T/J/Q/K all collapse to
    # 10), and no value should dominate.
    knock, _ = self._deal_many(4096, oklahoma=True, randomized=True)
    self.assertTrue((knock >= 0).all())
    self.assertTrue((knock <= gin_rummy.KNOCK_THRESHOLD).all())
    values, counts = np.unique(knock, return_counts=True)
    self.assertEqual(values.size, gin_rummy.KNOCK_THRESHOLD + 1)
    # Roughly uniform: expected ~4096/11 ≈ 372 per bucket. Loose bound — the
    # point is no bucket is starved or hogging, not a tight chi-square.
    self.assertLess(counts.max(), counts.min() * 3)

  def test_randomized_ignored_for_standard_deals(self):
    """Randomization leaves standard deals at KNOCK_THRESHOLD."""
    # On non-Oklahoma deals the knock card stays pinned at KNOCK_THRESHOLD
    # regardless of the randomization flag — it only governs Oklahoma deals.
    knock, _ = self._deal_many(128, oklahoma=False, randomized=True)
    np.testing.assert_array_equal(
        knock, np.full_like(knock, gin_rummy.KNOCK_THRESHOLD)
    )


class PublicSurfaceTest(absltest.TestCase):
  """The public names downstream engines use instead of private ones."""

  def test_oklahoma_knock_card_is_public(self):
    """oklahoma_knock_card is public and correct."""
    self.assertIs(gin_rummy.oklahoma_knock_card, game.oklahoma_knock_card)
    self.assertIn("oklahoma_knock_card", gin_rummy.__all__)
    self.assertIn("oklahoma_knock_card", game.__all__)
    # Ace => gin only (0); face cards => 10; otherwise the pip value.
    for upcard, want in ((0, 0), (12, 10), (4, 5), (13 + 9, 10)):
      self.assertEqual(int(gin_rummy.oklahoma_knock_card(upcard)), want)

  def test_lazy_constant_warmup_is_public(self):
    """The lazy-constant warmup is public."""
    for module in (gin_rummy, game):
      self.assertIs(module.init_lazy_constants, layoffs.init_lazy_constants)
      self.assertIs(
          module.lazy_constants_initialized, layoffs.lazy_constants_initialized
      )
      self.assertIn("init_lazy_constants", module.__all__)
      self.assertIn("lazy_constants_initialized", module.__all__)
    gin_rummy.init_lazy_constants()
    self.assertTrue(gin_rummy.lazy_constants_initialized())


class PublicWallMemoryTest(absltest.TestCase):
  """Public knowledge of the wall pick is retained."""

  def test_public_pick_memory_on_retained_wall_history(self):
    """The wall pick is remembered as public knowledge."""
    # A legal Oklahoma history that reaches the wall and picks the upcard.
    # Independently reconstruct public knowledge from visible card movement,
    # without consulting the engine's known-card transition conditions.
    history = [
        47,
        40,
        50,
        30,
        3,
        4,
        22,
        10,
        0,
        35,
        51,
        5,
        37,
        25,
        32,
        26,
        2,
        38,
        33,
        18,
        13,
        54,
        52,
        2,
        52,
        50,
        53,
        48,
        37,
        53,
        9,
        10,
        53,
        28,
        48,
        52,
        47,
        53,
        23,
        23,
        53,
        34,
        34,
        52,
        28,
        53,
        7,
        7,
        53,
        41,
        41,
        53,
        8,
        8,
        53,
        24,
        24,
        53,
        36,
        36,
        53,
        42,
        42,
        53,
        11,
        11,
        53,
        45,
        45,
        53,
        31,
        31,
        52,
        26,
        53,
        20,
        20,
        53,
        17,
        13,
        53,
        16,
        30,
        53,
        15,
        15,
        53,
        43,
        43,
        53,
        1,
        17,
        53,
        21,
        21,
        53,
        14,
        34,
        53,
        49,
        49,
        53,
        6,
        32,
        53,
        19,
        19,
        53,
        12,
        33,
        53,
        27,
        16,
        53,
        44,
        6,
        53,
        39,
        40,
        55,
        38,
        85,
        63,
        118,
        54,
        105,
        123,
        54,
    ]
    state = game.init_state(oklahoma=True)
    known = np.zeros((2, 52), dtype=np.int8)
    saw_wall_pick = False
    for ply, action in enumerate(history):
      self.assertTrue(bool(game.legal_actions_mask(state)[action]), ply)
      player, upcard = int(state["current_player"]), int(state["upcard"])
      old_hands = np.stack(
          [np.asarray(state["player0_hand"]), np.asarray(state["player1_hand"])]
      )
      after = game.step(state, jnp.int32(action))
      hands = np.stack(
          [np.asarray(after["player0_hand"]), np.asarray(after["player1_hand"])]
      )
      if (
          player in (0, 1)
          and upcard >= 0
          and not old_hands[player, upcard]
          and hands[player, upcard]
      ):
        known[player, upcard] = 1
        if int(state["phase"]) == gin_rummy.PHASE_WALL:
          saw_wall_pick = True
          self.assertEqual(upcard, 40)
      known *= hands
      np.testing.assert_array_equal(
          after["known_cards"],
          known,
          err_msg=f"after ply {ply}, action {action}",
      )
      state = after
    self.assertTrue(saw_wall_pick)
    self.assertTrue(bool(state["done"]))
    self.assertEqual(int(state["known_cards"][1, 40]), 0)


if __name__ == "__main__":
  absltest.main()
