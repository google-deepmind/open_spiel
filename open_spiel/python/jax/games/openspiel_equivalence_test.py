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

"""Move-for-move equivalence of every JAX game with OpenSpiel.

The harness is openspiel_equivalence (seeded random playouts in lockstep with
OpenSpiel, compared field by field at every state). This file checks that
every registered game is covered at OpenSpiel's defaults and runs the
playouts that are not split over the shard files, plus no_limit_holdem's
known-answer cases.
"""

import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import bridge_reference
from open_spiel.python.jax.games import no_limit_holdem
from open_spiel.python.jax.games import openspiel_equivalence as equivalence
from open_spiel.python.jax.games import registry
import pyspiel

_card = equivalence.card  # pylint: disable=invalid-name


class PyspielEquivalenceTest(absltest.TestCase):
  """Every JAX game plays OpenSpiel's game move for move."""

  def test_every_registered_game_is_covered(self):
    """Every registered JAX game has a config or a reference-build entry."""
    configured = {config.name for config in equivalence.GAME_CONFIGS}
    self.assertEmpty(configured & set(equivalence.NEEDS_REFERENCE_BUILD))
    self.assertEqual(
        set(registry.registered_names()),
        configured | set(equivalence.NEEDS_REFERENCE_BUILD),
        msg=(
            "every JAX game needs a GAME_CONFIGS entry, or an entry in "
            "NEEDS_REFERENCE_BUILD saying which OpenSpiel build checks it"
        ),
    )

  def test_every_openspiel_name_is_checked_at_its_defaults(self):
    """Every OpenSpiel name is checked at OpenSpiel's defaults."""
    # A JAX game under an OpenSpiel name plays OpenSpiel's game at the same
    # parameters, defaults included, or refuses them: each such name has a
    # config at OpenSpiel's default parameters, or is listed as refusing them.
    openspiel_names = set(pyspiel.registered_names())
    for name in registry.registered_names():
      if name not in set(pyspiel.registered_names()):
        continue  # not an OpenSpiel name (no_limit_holdem)
      with self.subTest(game=name):
        if name in equivalence.NEEDS_REFERENCE_BUILD:
          continue
        os_game = pyspiel.load_game(name)
        # Some games list their resolved parameters, others only the spec.
        defaults = {
            **os_game.get_type().parameter_specification,
            **os_game.get_parameters(),
        }
        at_defaults = [
            config
            for config in equivalence.GAME_CONFIGS
            if config.name == name
            and config.jax_game_str in (None, name)
            and all(
                defaults.get(k) == v
                for k, v in registry.parse_game_string(config.game_str)[
                    1
                ].items()
            )
        ]
        if name in equivalence.REFUSES_OPENSPIEL_DEFAULTS:
          self.assertEmpty(at_defaults)
          with self.assertRaises(ValueError):
            registry.load_game(name)
        else:
          self.assertNotEmpty(
              at_defaults,
              f"{name}: no GAME_CONFIGS entry at OpenSpiel's "
              f"default parameters {defaults}",
          )
    self.assertIn("go", openspiel_names)

  def test_default_sizes_match_openspiel(self):
    """Default sizes equal OpenSpiel's."""
    # The lockstep maps OpenSpiel's legal actions into the JAX mask, so a JAX
    # game with extra never-legal actions would pass it; the sizes are
    # compared here, at OpenSpiel's defaults.
    for name in registry.registered_names():
      if (
          name not in set(pyspiel.registered_names())
          or name in equivalence.NEEDS_REFERENCE_BUILD
          or name in equivalence.REFUSES_OPENSPIEL_DEFAULTS
      ):
        continue
      with self.subTest(game=name):
        game, os_game = registry.load_game(name), pyspiel.load_game(name)
        self.assertEqual(game.num_actions, os_game.num_distinct_actions())
        self.assertEqual(game.num_players, os_game.num_players())
        self.assertEqual(
            game.max_chance_outcomes, os_game.max_chance_outcomes()
        )

  def test_bridge_configs_cover_every_dealer(self):
    """The bridge configs cover every dealer."""
    # Every bridge config is OpenSpiel's card-play bridge at some dealer and
    # vulnerability, checked tensor for tensor, and the configs cover every
    # dealer (the auction view seats calls as OpenSpiel does whoever dealt).
    configs = [
        config for config in equivalence.GAME_CONFIGS if config.name == "bridge"
    ]
    for config in configs:
      self.assertTrue(config.check_observation, config.game_str)
      self.assertTrue(config.check_information_state, config.game_str)
      params = registry.parse_game_string(config.game_str)[1]
      self.assertIs(params["use_double_dummy_result"], False, config.game_str)
      self.assertLessEqual(
          set(params),
          {"use_double_dummy_result", "dealer", "dealer_vul", "non_dealer_vul"},
          config.game_str,
      )
    self.assertEqual(
        {
            registry.parse_game_string(c.game_str)[1].get("dealer", 0)
            for c in configs
        },
        {0, 1, 2, 3},
    )
    self.assertGreaterEqual(sum(config.playouts for config in configs), 2000)

  def _check_move_for_move(
      self,
      configs: list[equivalence.GameConfig],
      file_shard: tuple[int, int] = (0, 1),
  ) -> None:
    for game_str, count in equivalence.run_playouts(
        self, configs, file_shard=file_shard
    ).items():
      if count:
        print(f"openspiel_equivalence: game={game_str} truncated={count}")

  def test_move_for_move_equivalence(self):
    """Every non-optional config agrees move for move."""
    # Every config whose reference game an OpenSpiel build may lack has its
    # own test below, which skips on such a build.
    self.assertEqual(
        [
            config
            for config in equivalence.GAME_CONFIGS
            if equivalence.needs_optional_game(config)
        ],
        [equivalence.NO_LIMIT_HOLDEM_CONFIG],
    )
    self._check_move_for_move(
        equivalence.move_for_move_configs(), (0, equivalence.MOVE_SHARDS)
    )

  def test_the_move_for_move_shards_are_files(self):
    """Each move-for-move shard file exists."""
    here = os.path.dirname(os.path.abspath(__file__))
    for k in range(1, equivalence.MOVE_SHARDS):
      self.assertTrue(
          os.path.exists(
              os.path.join(here, f"openspiel_equivalence_shard{k}_test.py")
          ),
          k,
      )

  def test_move_for_move_equivalence_bridge_other_dealers(self):
    """Bridge boards dealt by East, South or West agree move for move."""
    # The JAX auction view is dealer-relative; an earlier pyspiel seats calls
    # as if North dealt, so these boards are compared only with a pyspiel
    # that has OpenSpiel's dealer-relative view.
    configs = [
        config
        for config in equivalence.GAME_CONFIGS
        if equivalence.bridge_other_dealer(config)
    ]
    self.assertLen(configs, 4)
    if not bridge_reference.pyspiel_auction_view_is_dealer_relative():
      self.skipTest(bridge_reference.SKIP_REASON)
    self._check_move_for_move(configs)

  @equivalence.needs_universal_poker
  def test_move_for_move_equivalence_no_limit_holdem(self):
    """No-limit hold'em agrees move for move."""
    self._check_move_for_move(
        [equivalence.NO_LIMIT_HOLDEM_CONFIG], (0, equivalence.MOVE_SHARDS)
    )

  @equivalence.needs_universal_poker
  def test_no_limit_holdem_all_in_preflop_runs_out_equivalently(self):
    """A preflop all-in runs out as in pyspiel."""
    os_state, state = self._assert_no_limit_sequence_equivalent(
        [0, 1, 2, 3, 200, 1, 4, 5, 6, 7, 8]
    )
    self.assertTrue(os_state.is_terminal())
    self.assertTrue(bool(state.is_terminal))
    self.assertEqual(int(state.board_count), no_limit_holdem.MAX_BOARD_CARDS)

  @equivalence.needs_universal_poker
  def test_no_limit_holdem_min_raise_sequence_matches_pyspiel(self):
    """A min-raise sequence matches pyspiel."""
    os_state, state = self._assert_no_limit_sequence_equivalent(
        [0, 1, 2, 3, 4, 6, 8, 1, 9, 10, 11, 1, 1, 12, 1, 1, 13, 1, 1]
    )
    self.assertTrue(os_state.is_terminal())
    self.assertTrue(bool(state.is_terminal))
    np.testing.assert_array_equal(
        np.asarray(state.spent), np.asarray([8, 8], dtype=np.int32)
    )

  @equivalence.needs_universal_poker
  def test_no_limit_holdem_max_raise_all_in_mid_street_matches_pyspiel(self):
    """A max-raise all-in mid street matches pyspiel."""
    os_state, state = self._assert_no_limit_sequence_equivalent(
        [0, 1, 2, 3, 1, 1, 4, 5, 6, 200, 1, 7, 8]
    )
    self.assertTrue(os_state.is_terminal())
    self.assertTrue(bool(state.is_terminal))
    np.testing.assert_array_equal(
        np.asarray(state.spent), np.asarray([200, 200], dtype=np.int32)
    )

  @equivalence.needs_universal_poker
  def test_no_limit_holdem_stack_depth_invariants_match_pyspiel(self):
    """Stack-depth invariants match pyspiel."""
    self._assert_no_limit_sequence_equivalent(
        [0, 1, 2, 3, 17, 43, 93, 143, 193, 1, 4, 5, 6, 1, 1, 7, 1, 1, 8, 1, 1]
    )

  @equivalence.needs_universal_poker
  def test_no_limit_holdem_showdown_ranks_match_pyspiel_for_every_hand_class(
      self,
  ):
    """Showdown ranks match pyspiel for every hand class."""
    cases = [
        (
            "high_card",
            "High Card",
            [_card(12, 0), _card(11, 1)],
            [_card(10, 2), _card(9, 3)],
            [_card(7, 0), _card(5, 1), _card(3, 2), _card(1, 3), _card(0, 0)],
        ),
        (
            "one_pair",
            "Pair",
            [_card(12, 0), _card(12, 1)],
            [_card(11, 2), _card(10, 3)],
            [_card(8, 0), _card(6, 1), _card(4, 2), _card(2, 3), _card(0, 0)],
        ),
        (
            "two_pair",
            "Two Pair",
            [_card(12, 0), _card(11, 1)],
            [_card(10, 2), _card(9, 3)],
            [_card(12, 2), _card(11, 3), _card(7, 0), _card(4, 1), _card(1, 2)],
        ),
        (
            "three_of_a_kind",
            "Three of a Kind",
            [_card(12, 0), _card(12, 1)],
            [_card(11, 2), _card(10, 3)],
            [_card(12, 2), _card(8, 0), _card(6, 1), _card(3, 2), _card(0, 3)],
        ),
        (
            "straight",
            "Straight",
            [_card(6, 0), _card(7, 1)],
            [_card(12, 2), _card(12, 3)],
            [_card(8, 2), _card(9, 3), _card(10, 0), _card(3, 1), _card(0, 2)],
        ),
        (
            "flush",
            "Flush",
            [_card(12, 0), _card(7, 0)],
            [_card(11, 1), _card(11, 2)],
            [_card(10, 0), _card(5, 0), _card(2, 0), _card(8, 3), _card(0, 1)],
        ),
        (
            "full_house",
            "Full House",
            [_card(12, 0), _card(12, 1)],
            [_card(10, 2), _card(9, 3)],
            [
                _card(12, 2),
                _card(11, 0),
                _card(11, 1),
                _card(4, 2),
                _card(0, 3),
            ],
        ),
        (
            "four_of_a_kind",
            "Four of a Kind",
            [_card(12, 0), _card(12, 1)],
            [_card(11, 2), _card(10, 3)],
            [_card(12, 2), _card(12, 3), _card(8, 0), _card(4, 1), _card(0, 2)],
        ),
        (
            "straight_flush",
            "Straight Flush",
            [_card(6, 0), _card(7, 0)],
            [_card(12, 1), _card(12, 2)],
            [_card(8, 0), _card(9, 0), _card(10, 0), _card(4, 1), _card(0, 2)],
        ),
    ]

    for name, expected_rank, p0_hole, p1_hole, board in cases:
      with self.subTest(hand_class=name):
        actions = p0_hole + p1_hole + [200, 1] + board
        os_state, state = self._assert_no_limit_sequence_equivalent(actions)
        self.assertTrue(os_state.is_terminal())
        self.assertGreater(float(os_state.returns()[0]), 0.0)
        np.testing.assert_allclose(
            np.asarray(no_limit_holdem.NoLimitHoldem().returns(state)),
            os_state.returns(),
            atol=1e-6,
        )
        self.assertEqual(
            os_state.to_struct().best_hand_rank_types[0], expected_rank
        )

  @equivalence.needs_universal_poker
  def test_no_limit_holdem_showdown_return_outcomes_match_pyspiel(self):
    """Showdown returns match pyspiel for wins and splits."""
    cases = [
        (
            "p0_wins",
            [_card(12, 0), _card(12, 1)],
            [_card(2, 2), _card(3, 3)],
            [_card(8, 0), _card(6, 1), _card(4, 2), _card(1, 3), _card(0, 0)],
        ),
        (
            "p1_wins",
            [_card(2, 0), _card(3, 1)],
            [_card(12, 2), _card(12, 3)],
            [_card(8, 0), _card(6, 1), _card(4, 2), _card(1, 3), _card(0, 0)],
        ),
        (
            "split_board_straight",
            [_card(0, 0), _card(2, 1)],
            [_card(4, 2), _card(6, 3)],
            [
                _card(8, 0),
                _card(9, 1),
                _card(10, 2),
                _card(11, 3),
                _card(12, 0),
            ],
        ),
    ]

    for name, p0_hole, p1_hole, board in cases:
      with self.subTest(outcome=name):
        actions = p0_hole + p1_hole + [200, 1] + board
        os_state, state = self._assert_no_limit_sequence_equivalent(actions)
        self.assertTrue(os_state.is_terminal())
        np.testing.assert_allclose(
            np.asarray(no_limit_holdem.NoLimitHoldem().returns(state)),
            os_state.returns(),
            atol=1e-6,
        )

  def _assert_no_limit_sequence_equivalent(
      self, actions: list[int]
  ) -> tuple[pyspiel.State, Any]:
    """Plays `actions` in both no-limit games, comparing every node.

    Args:
      actions: The actions.

    Returns:
      (OpenSpiel state, JAX state) at the end.
    """
    os_game = pyspiel.load_game(no_limit_holdem.UNIVERSAL_POKER_GAME_STRING)
    game = no_limit_holdem.NoLimitHoldem()
    apply_action = jax.jit(game.apply_action)
    fns = equivalence.CompiledFns(
        legal_actions_mask=jax.jit(game.legal_actions_mask),
        returns=jax.jit(game.returns),
        observation_tensor=jax.jit(game.observation_tensor),
        information_state_tensor=jax.jit(game.information_state_tensor),
    )
    os_state = os_game.new_initial_state()
    jax_state = game.new_initial_state()

    for step, action in enumerate(actions):
      legal_jax = equivalence.compare_state(
          game,
          os_game,
          os_state,
          jax_state,
          fns,
          equivalence.NO_LIMIT_HOLDEM_CONFIG,
          0,
          step,
      )
      self._assert_no_limit_stack_invariants(game, os_state, jax_state)
      self.assertFalse(os_state.is_terminal(), msg=f"extra action={action}")

      if os_state.is_chance_node():
        self.assertIn(action, [a for a, _ in os_state.chance_outcomes()])
        os_action = game.action_to_os(jax_state, action)
      else:
        self.assertIn(action, legal_jax)
        os_action = game.action_to_os(jax_state, action)

      os_state.apply_action(int(os_action))
      jax_state = apply_action(jax_state, jnp.array(action))

    equivalence.compare_state(
        game,
        os_game,
        os_state,
        jax_state,
        fns,
        equivalence.NO_LIMIT_HOLDEM_CONFIG,
        0,
        len(actions),
    )
    self._assert_no_limit_stack_invariants(game, os_state, jax_state)
    return os_state, jax_state

  def _assert_no_limit_stack_invariants(
      self,
      game: no_limit_holdem.NoLimitHoldem,
      os_state: pyspiel.State,
      state: Any,
  ) -> None:
    """Asserts the commitments are within the stack and match OpenSpiel.

    Args:
      game: The JAX game.
      os_state: The OpenSpiel state.
      state: The JAX state.
    """
    spent = np.asarray(state.spent, dtype=np.int32)
    self.assertGreaterEqual(int(np.min(spent)), 0)
    self.assertLessEqual(int(np.max(spent)), no_limit_holdem.STACK)

    os_struct = os_state.to_struct()
    np.testing.assert_array_equal(
        spent, np.asarray(os_struct.player_contributions, dtype=np.int32)
    )
    self.assertEqual(int(np.sum(spent)), int(os_struct.pot_size))

    if bool(state.is_terminal):
      returns = np.asarray(game.returns(state), dtype=np.float32)
      np.testing.assert_allclose(np.sum(returns), 0.0, atol=1e-6)


class OccupiedCellTurnTest(parameterized.TestCase):
  """A move to an occupied cell passes the turn as in OpenSpiel.

  The classical variants (OpenSpiel's defaults) give the mover another move;
  the abrupt ones pass the turn.
  """

  @parameterized.parameters(
      ("phantom_ttt_ir", 1),
      ("phantom_ttt_ir(gameversion=classical)", 1),
      ("phantom_ttt_ir(gameversion=abrupt)", 0),
      ("dark_hex_ir", 1),
      ("dark_hex_ir(gameversion=cdh)", 1),
      ("dark_hex_ir(gameversion=adh)", 0),
  )
  def test_player_after_occupied_cell_matches_pyspiel(self, game_str, after):
    """The player after a move to an occupied cell equals pyspiel's.

    Args:
      game_str: The game string.
      after: The expected player.
    """
    game = registry.load_game(game_str)
    os_state = pyspiel.load_game(game_str).new_initial_state()
    state = game.new_initial_state()
    for action in (4, 4):  # player 1 tries player 0's cell
      state = game.apply_action(state, jnp.int32(action))
      os_state.apply_action(action)
      self.assertEqual(
          int(game.current_player(state)), os_state.current_player()
      )
    self.assertEqual(int(game.current_player(state)), after)


if __name__ == "__main__":
  absltest.main()
