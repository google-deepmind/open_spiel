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

"""Heads-up no-limit hold'em tests.

Parameters, betting, showdown scoring with and without the lookup table,
dtypes, and random play against OpenSpiel.
"""

import itertools
import math
import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import no_limit_holdem
from open_spiel.python.jax.games import optional_pyspiel_games
from open_spiel.python.jax.games import registry
import pyspiel

_NUM_EQUIVALENCE_PLAYOUTS = 1000
# The universal_poker parameters, under this package's game name.
_JAX_GAME_STRING = "no_limit_holdem" + (
    no_limit_holdem.UNIVERSAL_POKER_GAME_STRING.removeprefix("universal_poker")
)


# The OpenSpiel reference game needs ACPC, an optional OpenSpiel dependency.
_needs_reference_game = optional_pyspiel_games.skip_unless_available(
    no_limit_holdem.UNIVERSAL_POKER_GAME_STRING
)


def _card(rank: int, suit: int = 0) -> int:
  """The card index of `rank` and `suit`.

  Args:
    rank: The rank.
    suit: The suit.

  Returns:
    The card index.
  """
  return rank * no_limit_holdem.NUM_SUITS + suit


class NoLimitHoldemBasicTest(parameterized.TestCase):
  """The game's parameters, betting, hand evaluation and returns.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = no_limit_holdem.NoLimitHoldem()

  def test_registered(self):
    """The game is registered under its names and game strings."""
    self.assertIn("no_limit_holdem", registry.registered_names())
    self.assertIsInstance(
        registry.load_game("no_limit_holdem"), no_limit_holdem.NoLimitHoldem
    )
    self.assertIsInstance(
        registry.load_game(_JAX_GAME_STRING),
        no_limit_holdem.NoLimitHoldem,
    )
    self.assertIsInstance(
        registry.load_game("no_limit_holdem", use_showdown_cache=False),
        no_limit_holdem.NoLimitHoldem,
    )
    self.assertIsInstance(
        registry.load_game("no_limit_holdem(use_showdown_cache=false)"),
        no_limit_holdem.NoLimitHoldem,
    )

  def test_rejects_other_universal_poker_params(self):
    """Other universal_poker parameters are refused."""
    with self.assertRaisesRegex(
        ValueError,
        "heads-up no-limit configuration; mismatched parameters",
    ):
      registry.load_game(
          _JAX_GAME_STRING.replace("stack=200 200", "stack=100 100")
      )
    with self.assertRaisesRegex(
        ValueError,
        "heads-up no-limit configuration; unsupported parameters",
    ):
      registry.load_game(_JAX_GAME_STRING[:-1] + ",calcOddsNumSims=100)")

  def test_rejects_non_bool_showdown_cache_flag(self):
    """A non-bool use_showdown_cache raises TypeError."""
    with self.assertRaisesRegex(TypeError, "use_showdown_cache must be a bool"):
      no_limit_holdem.NoLimitHoldem(use_showdown_cache="false")

  def test_initial_chance_outcomes(self):
    """The first chance node is uniform over the 52 cards."""
    state = self.game.new_initial_state()
    self.assertEqual(int(state.current_player), game_lib.CHANCE_PLAYER_ID)
    probs = self.game.chance_outcomes(state)
    self.assertEqual(probs.shape, (no_limit_holdem.NUM_CARDS,))
    np.testing.assert_allclose(
        np.array(probs), np.full(no_limit_holdem.NUM_CARDS, 1.0 / 52.0)
    )

  @_needs_reference_game
  def test_preflop_legal_actions_match_pyspiel(self):
    """Preflop legal actions equal pyspiel's."""
    os_game = pyspiel.load_game(no_limit_holdem.UNIVERSAL_POKER_GAME_STRING)
    os_state = os_game.new_initial_state()
    state = self.game.new_initial_state()
    for action in [0, 1, 2, 3]:
      os_state.apply_action(action)
      state = self.game.apply_action(state, jnp.int32(action))
    self.assertEqual(int(state.current_player), 1)
    jax_legal = np.flatnonzero(np.array(self.game.legal_actions_mask(state)))
    self.assertEqual(jax_legal.tolist(), os_state.legal_actions())

  @parameterized.named_parameters(
      (
          "royal_flush",
          [
              _card(8, 0),
              _card(9, 0),
              _card(10, 0),
              _card(11, 0),
              _card(12, 0),
              _card(0, 1),
              _card(1, 2),
          ],
          [
              _card(8, 1),
              _card(8, 2),
              _card(8, 3),
              _card(12, 1),
              _card(12, 2),
              _card(0, 0),
              _card(1, 1),
          ],
      ),
      (
          "full_house_beats_flush",
          [
              _card(5, 0),
              _card(5, 1),
              _card(5, 2),
              _card(9, 0),
              _card(9, 1),
              _card(0, 2),
              _card(1, 3),
          ],
          [
              _card(2, 0),
              _card(4, 0),
              _card(6, 0),
              _card(8, 0),
              _card(10, 0),
              _card(0, 1),
              _card(1, 2),
          ],
      ),
      (
          "wheel_straight",
          [
              _card(12, 0),
              _card(0, 1),
              _card(1, 2),
              _card(2, 3),
              _card(3, 0),
              _card(7, 1),
              _card(8, 2),
          ],
          [
              _card(12, 1),
              _card(0, 2),
              _card(1, 3),
              _card(2, 0),
              _card(4, 1),
              _card(7, 2),
              _card(8, 3),
          ],
      ),
      (
          "straight_flush_beats_quads",
          [
              _card(4, 2),
              _card(5, 2),
              _card(6, 2),
              _card(7, 2),
              _card(8, 2),
              _card(0, 1),
              _card(1, 3),
          ],
          [
              _card(12, 0),
              _card(12, 1),
              _card(12, 2),
              _card(12, 3),
              _card(8, 0),
              _card(0, 2),
              _card(1, 1),
          ],
      ),
      (
          "two_pair_beats_pair",
          [
              _card(11, 0),
              _card(11, 1),
              _card(6, 0),
              _card(6, 1),
              _card(2, 0),
              _card(0, 2),
              _card(1, 3),
          ],
          [
              _card(12, 0),
              _card(12, 1),
              _card(10, 0),
              _card(8, 1),
              _card(6, 2),
              _card(0, 0),
              _card(1, 1),
          ],
      ),
      (
          "ace_high_kicker_beats_king_high",
          [
              _card(12, 0),
              _card(10, 1),
              _card(8, 2),
              _card(6, 3),
              _card(4, 0),
              _card(2, 1),
              _card(0, 2),
          ],
          [
              _card(11, 0),
              _card(10, 2),
              _card(8, 3),
              _card(6, 0),
              _card(4, 1),
              _card(2, 2),
              _card(0, 3),
          ],
      ),
      (
          # Four eights on the board: the fifth card is the best other rank
          # among all seven, here the ace in the hole, not the ten.
          "four_of_a_kind_kicker_from_hole",
          [
              _card(12, 0),
              _card(4, 0),
              _card(6, 0),
              _card(6, 1),
              _card(6, 2),
              _card(6, 3),
              _card(8, 1),
          ],
          [
              _card(10, 1),
              _card(0, 1),
              _card(6, 0),
              _card(6, 1),
              _card(6, 2),
              _card(6, 3),
              _card(8, 1),
          ],
      ),
      (
          "four_of_a_kind_kicker_beats_board_kicker",
          [
              _card(11, 0),
              _card(0, 0),
              _card(3, 0),
              _card(3, 1),
              _card(3, 2),
              _card(3, 3),
              _card(9, 1),
          ],
          [
              _card(2, 1),
              _card(1, 1),
              _card(3, 0),
              _card(3, 1),
              _card(3, 2),
              _card(3, 3),
              _card(9, 1),
          ],
      ),
      (
          "counterfeit_two_pair_kicker",
          [
              _card(12, 0),
              _card(11, 0),
              _card(8, 0),
              _card(8, 1),
              _card(5, 0),
              _card(5, 1),
              _card(2, 0),
          ],
          [
              _card(12, 1),
              _card(10, 0),
              _card(8, 2),
              _card(8, 3),
              _card(5, 2),
              _card(5, 3),
              _card(2, 1),
          ],
      ),
  )
  def test_hand_evaluator_ordering(self, winning_cards, losing_cards):
    """The winning seven cards score higher than the losing ones.

    Args:
      winning_cards: The winning seven cards.
      losing_cards: The losing seven cards.
    """
    self.assertGreater(
        int(no_limit_holdem.best_seven_card_score(jnp.array(winning_cards))),
        int(no_limit_holdem.best_seven_card_score(jnp.array(losing_cards))),
    )

  def test_uncached_seven_card_evaluator_matches_cached(self):
    """The uncached seven-card evaluator equals the cached one."""
    rng = np.random.default_rng(0)
    random_hands = np.stack([
        rng.choice(no_limit_holdem.NUM_CARDS, size=7, replace=False)
        for _ in range(1024)
    ]).astype(np.int32)
    score_cached = jax.jit(jax.vmap(no_limit_holdem.best_seven_card_score))
    score_uncached = jax.jit(
        jax.vmap(no_limit_holdem.best_seven_card_score_uncached)
    )
    np.testing.assert_array_equal(
        np.array(score_cached(jnp.asarray(random_hands))),
        np.array(score_uncached(jnp.asarray(random_hands))),
    )

  def test_jit_apply_action_and_tensors(self):
    """apply_action and the tensors run under jit with the right shapes."""
    apply_jit = jax.jit(self.game.apply_action)
    obs_jit = jax.jit(self.game.observation_tensor)
    info_jit = jax.jit(self.game.information_state_tensor)
    state = self.game.new_initial_state()
    for action in [0, 1, 2, 3]:
      state = apply_jit(state, jnp.int32(action))
    obs = obs_jit(state, jnp.int32(0))
    info = info_jit(state, jnp.int32(0))
    self.assertEqual(obs.shape, self.game.obs_shape)
    self.assertEqual(info.shape, self.game.info_state_shape)

  def test_five_card_category_ordering_through_seven_card_path(self):
    """Hand categories order correctly through the seven-card path."""
    categories_low_to_high = [
        [_card(12, 0), _card(10, 1), _card(8, 2), _card(6, 3), _card(4, 0)],
        [_card(12, 0), _card(12, 1), _card(8, 2), _card(6, 3), _card(4, 0)],
        [_card(12, 0), _card(12, 1), _card(8, 2), _card(8, 3), _card(4, 0)],
        [_card(12, 0), _card(12, 1), _card(12, 2), _card(6, 3), _card(4, 0)],
        [_card(0, 0), _card(1, 1), _card(2, 2), _card(3, 3), _card(4, 0)],
        [_card(12, 2), _card(10, 2), _card(8, 2), _card(6, 2), _card(4, 2)],
        [_card(12, 0), _card(12, 1), _card(12, 2), _card(8, 0), _card(8, 1)],
        [_card(12, 0), _card(12, 1), _card(12, 2), _card(12, 3), _card(4, 0)],
        [_card(8, 1), _card(9, 1), _card(10, 1), _card(11, 1), _card(12, 1)],
    ]
    pad = [_card(2, 0), _card(3, 1)]
    scores = [
        int(no_limit_holdem.best_seven_card_score(jnp.array(hand + pad)))
        for hand in categories_low_to_high
    ]
    self.assertEqual(scores, sorted(scores))
    self.assertLen(set(scores), len(scores))

  def test_lut_matches_categorical_evaluator_for_all_five_card_hands(self):
    """The lookup table equals the categorical evaluator on every hand."""
    # Reaches private members of the code under test.
    # pylint: disable=protected-access
    self.assertIsInstance(no_limit_holdem._FIVE_CARD_SCORE_LUT, np.ndarray)
    num_hands = math.comb(no_limit_holdem.NUM_CARDS, 5)
    self.assertEqual(
        no_limit_holdem._FIVE_CARD_SCORE_LUT.shape,
        (math.comb(no_limit_holdem.NUM_RANKS + 4, 5) * 2,),
    )

    hands = np.fromiter(
        itertools.chain.from_iterable(
            itertools.combinations(range(no_limit_holdem.NUM_CARDS), 5)
        ),
        dtype=np.int16,
        count=num_hands * 5,
    ).reshape(num_hands, 5)
    ranks = np.sort(hands // no_limit_holdem.NUM_SUITS, axis=1)
    suits = hands % no_limit_holdem.NUM_SUITS
    flush = np.all(suits == suits[:, :1], axis=1).astype(np.int32)
    indexes = no_limit_holdem._rank_flush_index(ranks, flush)
    reference_scores = no_limit_holdem._np_score_5(hands)
    np.testing.assert_array_equal(
        no_limit_holdem._FIVE_CARD_SCORE_LUT[indexes], reference_scores
    )

    lut_score = jax.jit(jax.vmap(no_limit_holdem._score_5))
    chunk_size = 262144
    for start in range(0, num_hands, chunk_size):
      stop = min(start + chunk_size, num_hands)
      np.testing.assert_array_equal(
          np.array(lut_score(jnp.asarray(hands[start:stop], dtype=jnp.int32))),
          reference_scores[start:stop],
      )

  def test_numpy_reference_matches_categorical_evaluator(self):
    """The numpy reference equals the categorical evaluator."""
    # Reaches private members of the code under test.
    # pylint: disable=protected-access
    hands = np.array(
        [
            [
                _card(8, 0),
                _card(9, 0),
                _card(10, 0),
                _card(11, 0),
                _card(12, 0),
            ],
            [
                _card(12, 0),
                _card(12, 1),
                _card(12, 2),
                _card(12, 3),
                _card(4, 0),
            ],
            [
                _card(12, 0),
                _card(12, 1),
                _card(12, 2),
                _card(8, 0),
                _card(8, 1),
            ],
            [_card(12, 2), _card(10, 2), _card(8, 2), _card(6, 2), _card(4, 2)],
            [_card(0, 0), _card(1, 1), _card(2, 2), _card(3, 3), _card(4, 0)],
            [
                _card(12, 0),
                _card(12, 1),
                _card(12, 2),
                _card(6, 3),
                _card(4, 0),
            ],
            [_card(12, 0), _card(12, 1), _card(8, 2), _card(8, 3), _card(4, 0)],
            [_card(12, 0), _card(12, 1), _card(8, 2), _card(6, 3), _card(4, 0)],
            [_card(12, 0), _card(10, 1), _card(8, 2), _card(6, 3), _card(4, 0)],
        ],
        dtype=np.int32,
    )
    reference_score = jax.jit(jax.vmap(no_limit_holdem._score_5_categorical))
    np.testing.assert_array_equal(
        no_limit_holdem._np_score_5(hands),
        np.array(reference_score(jnp.asarray(hands))),
    )

  def _assert_sequence_matches_pyspiel(self, actions):
    """Plays `actions` in both games and checks they agree.

    Args:
      actions: The actions.

    Returns:
      (OpenSpiel state, JAX state).
    """
    os_game = pyspiel.load_game(no_limit_holdem.UNIVERSAL_POKER_GAME_STRING)
    os_state = os_game.new_initial_state()
    state = self.game.new_initial_state()
    for action in actions:
      if not os_state.is_chance_node():
        self.assertIn(action, os_state.legal_actions())
        self.assertTrue(bool(self.game.legal_actions_mask(state)[action]))
      os_state.apply_action(action)
      state = self.game.apply_action(state, jnp.int32(action))
    self.assertEqual(bool(state.is_terminal), os_state.is_terminal())
    self.assertEqual(int(state.current_player), os_state.current_player())
    np.testing.assert_allclose(
        np.array(self.game.returns(state)), os_state.returns(), atol=1e-6
    )
    return os_state, state

  @_needs_reference_game
  def test_fold_return_matches_pyspiel(self):
    """A fold's returns equal pyspiel's."""
    self._assert_sequence_matches_pyspiel([0, 1, 2, 3, 4, no_limit_holdem.FOLD])

  @_needs_reference_game
  def test_checked_down_showdown_matches_pyspiel(self):
    """A checked-down showdown equals pyspiel's."""
    self._assert_sequence_matches_pyspiel(
        [0, 1, 2, 3, 1, 1, 4, 5, 6, 1, 1, 7, 1, 1, 8, 1, 1]
    )

  @_needs_reference_game
  def test_all_in_call_runs_out_board(self):
    """An all-in call runs out the board."""
    os_state, state = self._assert_sequence_matches_pyspiel(
        [0, 1, 2, 3, 200, 1, 4, 5, 6, 7, 8]
    )
    self.assertTrue(os_state.is_terminal())
    self.assertTrue(bool(state.is_terminal))
    self.assertEqual(int(state.board_count), no_limit_holdem.MAX_BOARD_CARDS)

  @_needs_reference_game
  def test_raise_boundaries_match_pyspiel(self):
    """The raise sizes allowed equal pyspiel's."""
    os_state, state = self._assert_sequence_matches_pyspiel([0, 1, 2, 3])
    np.testing.assert_array_equal(
        np.flatnonzero(np.array(self.game.legal_actions_mask(state))),
        os_state.legal_actions(),
    )
    self.assertNotIn(2, os_state.legal_actions())
    self.assertNotIn(3, os_state.legal_actions())
    self.assertIn(4, os_state.legal_actions())
    self.assertIn(200, os_state.legal_actions())

    os_state.apply_action(132)
    state = self.game.apply_action(state, jnp.int32(132))
    np.testing.assert_array_equal(
        np.flatnonzero(np.array(self.game.legal_actions_mask(state))),
        os_state.legal_actions(),
    )
    self.assertEqual(os_state.legal_actions(), [0, 1, 200])

  def test_four_of_a_kind_kicker_is_a_held_rank(self):
    """A four-of-a-kind kicker is a rank the hand holds."""
    # Only the kicker separates these hands, and it must be a rank the hand
    # holds: the ace (12) and the queen (10) both lose to nothing else.
    board = [_card(6, 0), _card(6, 1), _card(6, 2), _card(6, 3), _card(8, 1)]
    ace = [_card(12, 0), _card(4, 0)] + board
    queen = [_card(10, 1), _card(0, 1)] + board
    for score in (
        no_limit_holdem.best_seven_card_score,
        no_limit_holdem.best_seven_card_score_uncached,
    ):
      self.assertGreater(
          int(score(jnp.array(ace))), int(score(jnp.array(queen)))
      )
    # A board with quads and the best kicker on it is a split pot.
    board_ace = [
        _card(6, 0),
        _card(6, 1),
        _card(6, 2),
        _card(6, 3),
        _card(12, 1),
    ]
    self.assertEqual(
        int(
            no_limit_holdem.best_seven_card_score(
                jnp.array([_card(10, 0), _card(0, 0)] + board_ace)
            )
        ),
        int(
            no_limit_holdem.best_seven_card_score(
                jnp.array([_card(9, 2), _card(1, 2)] + board_ace)
            )
        ),
    )

  @_needs_reference_game
  def test_quads_showdown_with_hole_kicker_matches_pyspiel(self):
    """A quads showdown decided by a hole kicker equals pyspiel's."""
    # P0 holds 2s Qs, P1 holds 6c Ac; the board is 8d 8c 8s 8h Td.
    os_state, state = self._assert_sequence_matches_pyspiel(
        [3, 43, 48, 16, 200, 1, 25, 24, 27, 26, 33]
    )
    self.assertTrue(os_state.is_terminal())
    np.testing.assert_allclose(os_state.returns(), [-200.0, 200.0])
    np.testing.assert_allclose(
        np.array(self.game.returns(state)), [-200.0, 200.0]
    )

  @_needs_reference_game
  def test_raises_close_when_a_round_reaches_the_action_limit(self):
    """Raising closes when a round reaches its action limit."""
    # A betting round holds at most MAX_ROUND_ACTIONS actions and every
    # raise leaves room for each player to answer it: after 63 minimum
    # raises preflop only fold and call remain, as in OpenSpiel.
    raises = list(range(4, 4 + 2 * 63, 2))
    os_state, state = self._assert_sequence_matches_pyspiel(
        [0, 1, 2, 3] + raises
    )
    self.assertEqual(int(state.round_actions), 63)
    self.assertEqual(os_state.legal_actions(), [0, 1])
    np.testing.assert_array_equal(
        np.flatnonzero(np.array(self.game.legal_actions_mask(state))), [0, 1]
    )
    # One raise fewer leaves raising open.
    os_state, state = self._assert_sequence_matches_pyspiel(
        [0, 1, 2, 3] + raises[:-1]
    )
    self.assertIn(raises[-1], os_state.legal_actions())
    self.assertTrue(bool(self.game.legal_actions_mask(state)[raises[-1]]))
    # The limit is per round: after the call the flop opens raising again.
    os_state, state = self._assert_sequence_matches_pyspiel(
        [0, 1, 2, 3] + raises + [no_limit_holdem.CALL, 4, 5, 6]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(np.array(self.game.legal_actions_mask(state))),
        os_state.legal_actions(),
    )
    self.assertIn(200, os_state.legal_actions())

  def test_history_overflow_fails_loudly(self):
    """Overflowing the history raises."""
    state = self.game.new_initial_state()._replace(
        history_len=jnp.int32(no_limit_holdem.MAX_GAME_LENGTH)
    )
    with self.assertRaisesRegex(Exception, "MAX_GAME_LENGTH"):
      next_state = self.game.apply_action(state, jnp.int32(0))
      jax.tree_util.tree_map(lambda x: x.block_until_ready(), next_state)

  def test_history_free_mode_has_no_information_state(self):
    """Without history tracking there is no information state."""
    game = no_limit_holdem.NoLimitHoldem(track_information_state_history=False)
    state = game.new_initial_state()
    for action in [0, 1, 2, 3]:
      state = game.apply_action(state, jnp.int32(action))
    self.assertEqual(int(state.history_len), 0)
    self.assertEqual(
        game.observation_tensor(state, jnp.int32(0)).shape, game.obs_shape
    )
    with self.assertRaisesRegex(ValueError, "no information state"):
      game.information_state_tensor(state, jnp.int32(0))
    with self.assertRaisesRegex(ValueError, "no information state"):
      _ = game.info_state_shape

  def test_incomplete_showdown_fails_loudly(self):
    """Returns at a showdown without its cards raise."""
    state = self.game.new_initial_state()._replace(
        is_terminal=jnp.bool_(True),
        current_player=jnp.int32(game_lib.TERMINAL_PLAYER_ID),
    )
    with self.assertRaisesRegex(Exception, "showdown requires"):
      self.game.returns(state).block_until_ready()

  def test_jit_returns_showdown_does_not_call_host_validation(self):
    """Jitted showdown returns do not call the host validation."""
    # Reaches private members of the code under test.
    # pylint: disable=protected-access
    state = self.game.new_initial_state()._replace(
        private_cards=jnp.array(
            [
                [_card(12, 0), _card(12, 1)],
                [_card(11, 0), _card(11, 1)],
            ],
            dtype=jnp.int32,
        ),
        board_cards=jnp.array(
            [_card(12, 2), _card(8, 0), _card(6, 1), _card(4, 2), _card(2, 3)],
            dtype=jnp.int32,
        ),
        is_terminal=jnp.bool_(True),
        current_player=jnp.int32(game_lib.TERMINAL_PLAYER_ID),
    )
    original_check = no_limit_holdem._check_showdown_cards_complete

    def fail_if_called(cards):
      del cards
      raise AssertionError("host showdown validation leaked into jit returns")

    try:
      no_limit_holdem._check_showdown_cards_complete = fail_if_called
      returns = jax.jit(self.game.returns)(state)
      np.testing.assert_allclose(np.array(returns), np.array([1.0, -1.0]))
    finally:
      no_limit_holdem._check_showdown_cards_complete = original_check

  def test_uncached_terminal_returns_match_cached(self):
    """Uncached terminal returns equal cached ones."""
    cached_game = no_limit_holdem.NoLimitHoldem(use_showdown_cache=True)
    uncached_game = no_limit_holdem.NoLimitHoldem(use_showdown_cache=False)
    state = cached_game.new_initial_state()._replace(
        private_cards=jnp.array(
            [
                [_card(12, 0), _card(12, 1)],
                [_card(11, 0), _card(11, 1)],
            ],
            dtype=jnp.int32,
        ),
        board_cards=jnp.array(
            [_card(12, 2), _card(8, 0), _card(6, 1), _card(4, 2), _card(2, 3)],
            dtype=jnp.int32,
        ),
        spent=jnp.array([200, 200], dtype=jnp.int32),
        is_terminal=jnp.bool_(True),
        current_player=jnp.int32(game_lib.TERMINAL_PLAYER_ID),
    )
    cached_returns = jax.jit(cached_game.returns)(state)
    uncached_returns = jax.jit(uncached_game.returns)(state)
    np.testing.assert_allclose(
        np.array(uncached_returns), np.array(cached_returns)
    )

  def test_batched_jit_smoke(self):
    """A vmapped, jitted batch of deals runs."""
    game = self.game

    @jax.jit
    def run_batch():
      """Deals four games in a vmapped scan.

      Returns:
        i32[4]: each game's current player after the deal.
      """
      states = jax.vmap(lambda _: game.new_initial_state())(jnp.arange(4))
      actions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

      def deal_one(state):
        """Applies the deal actions to one state.

        Args:
          state: The state.

        Returns:
          The dealt state.
        """

        def body(carry, action):
          """Applies one action.

          Args:
            carry: The state.
            action: The action.

          Returns:
            (next state, None) for `jax.lax.scan`.
          """
          return game.apply_action(carry, action), None

        state, _ = jax.lax.scan(body, state, actions)
        return state

      return jax.vmap(deal_one)(states).current_player

    np.testing.assert_array_equal(np.array(run_batch()), np.full(4, 1))


class NoLimitHoldemEquivalenceTest(absltest.TestCase):
  """Random play agrees with OpenSpiel."""

  @_needs_reference_game
  def test_random_playthrough_matches_pyspiel(self):
    """Random playthroughs agree with pyspiel move for move."""
    game = no_limit_holdem.NoLimitHoldem()
    apply_jit = jax.jit(game.apply_action)
    legal_jit = jax.jit(game.legal_actions_mask)
    returns_jit = jax.jit(game.returns)
    obs_jit = jax.jit(game.observation_tensor)
    info_jit = jax.jit(game.information_state_tensor)

    for seed in range(_NUM_EQUIVALENCE_PLAYOUTS):
      os_game = pyspiel.load_game(no_limit_holdem.UNIVERSAL_POKER_GAME_STRING)
      os_state = os_game.new_initial_state()
      state = game.new_initial_state()
      rng = random.Random(seed)

      while not os_state.is_terminal():
        self.assertEqual(bool(state.is_terminal), os_state.is_terminal())
        self.assertEqual(int(state.current_player), os_state.current_player())

        if os_state.is_chance_node():
          os_outcomes = os_state.chance_outcomes()
          os_actions, probs = zip(*os_outcomes)
          jax_probs = np.array(game.chance_outcomes(state))
          for os_action, prob in os_outcomes:
            self.assertAlmostEqual(float(jax_probs[os_action]), prob)
          action = rng.choices(os_actions, weights=probs)[0]
        else:
          mask = np.array(legal_jit(state))
          jax_legal = np.flatnonzero(mask).tolist()
          self.assertEqual(jax_legal, os_state.legal_actions())
          np.testing.assert_allclose(
              np.array(returns_jit(state)), os_state.returns(), atol=1e-6
          )
          for player in range(no_limit_holdem.NUM_PLAYERS):
            np.testing.assert_array_equal(
                np.array(obs_jit(state, jnp.int32(player))),
                np.array(os_state.observation_tensor(player), dtype=np.float32),
            )
            np.testing.assert_array_equal(
                np.array(info_jit(state, jnp.int32(player))),
                np.array(
                    os_state.information_state_tensor(player), dtype=np.float32
                ),
            )
          action = rng.choice(jax_legal)

        os_state.apply_action(action)
        state = apply_jit(state, jnp.int32(action))

      self.assertTrue(bool(state.is_terminal))
      np.testing.assert_allclose(
          np.array(returns_jit(state)), os_state.returns(), atol=1e-6
      )


if __name__ == "__main__":
  absltest.main()
