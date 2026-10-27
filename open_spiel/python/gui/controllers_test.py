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


"""Tests for the controllers that play seats."""

from absl.testing import absltest

from open_spiel.python.games import hangman  # pylint: disable=unused-import
from open_spiel.python.games import tic_tac_toe  # pylint: disable=unused-import
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import mcts_controller
import pyspiel

# The simulations MCTS is capped at when the C++ search gives it no clock.
_CAP = mcts_controller.SIMULATIONS[contract.MctsLimits.SIMULATIONS_ONLY].maximum


def _availability(game: pyspiel.Game) -> dict[str, bool]:
  """Returns whether each controller can play `game`."""
  return {
      descriptor.controller.name: descriptor.available
      for descriptor in controllers.descriptors(game)
  }


def _mcts(game: pyspiel.Game) -> controllers.Descriptor:
  """Returns the MCTS descriptor for `game`."""
  (mcts,) = [
      descriptor
      for descriptor in controllers.descriptors(game)
      if descriptor.controller.name == controllers.MCTS
  ]
  return mcts


def _mcts_reason(game: pyspiel.Game) -> str | None:
  """Returns why MCTS cannot play `game`, or None."""
  return _mcts(game).unavailable_reason


class _Mutates:
  """A controller that plays on the state it is given, then chooses 4."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Plays the top-left corner on `state`, then chooses the centre."""
    del seat
    state.apply_action(0)
    return 4


class _Returns:
  """A controller that returns a fixed value."""

  def __init__(self, value: object) -> None:
    self._value = value

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the fixed value, whatever it is."""
    del state, seat
    return self._value  # type: ignore[return-value]


class _Raises:
  """A controller that raises."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Raises, as a controller with a bug would."""
    del state, seat
    raise KeyError("the card in hand")


class DescriptorsTest(absltest.TestCase):

  def test_every_controller_is_described_human_first(self) -> None:
    kinds = [
        descriptor.controller.name
        for descriptor in controllers.descriptors(
            pyspiel.load_game("tic_tac_toe")
        )
    ]
    self.assertEqual(
        kinds, [controllers.HUMAN, controllers.RANDOM, controllers.MCTS]
    )

  def test_every_controller_plays_a_native_perfect_information_game(
      self,
  ) -> None:
    self.assertEqual(
        _availability(pyspiel.load_game("tic_tac_toe")),
        {"human": True, "random": True, "mcts": True},
    )

  def test_mcts_refuses_a_game_it_cannot_search_fairly(self) -> None:
    for name in (
        "kuhn_poker",  # Hidden cards.
        "matrix_rps",  # Simultaneous moves.
        "stones_and_gems",  # Chance sampled inside the state.
        "cliff_walking",  # Rewards before the end.
    ):
      with self.subTest(name):
        game = pyspiel.load_game(name)
        self.assertEqual(
            _availability(game), {"human": True, "random": True, "mcts": False}
        )
        self.assertIn(game.get_type().short_name, _mcts_reason(game) or "")

  def test_mcts_refuses_a_simultaneous_game_served_turn_based(self) -> None:
    game = pyspiel.load_game_as_turn_based("matrix_rps")
    self.assertIsNotNone(_mcts_reason(game))

  def test_mcts_searches_a_game_with_explicit_chance(self) -> None:
    self.assertIsNone(_mcts_reason(pyspiel.load_game("pig")))

  def test_mcts_plays_a_game_implemented_in_python(self) -> None:
    self.assertIsNone(_mcts_reason(pyspiel.load_game("python_tic_tac_toe")))

  def test_a_controller_that_reads_the_full_state_refuses_hidden_info(
      self,
  ) -> None:
    game = pyspiel.load_game("kuhn_poker")
    for name, kind in controllers.kinds().items():
      with self.subTest(name):
        self.assertEqual(
            kind.unavailable_reason(game) is None, not kind.reads_full_state
        )

  def test_mcts_refuses_a_game_that_hides_despite_its_declaration(
      self,
  ) -> None:
    game = pyspiel.load_game("python_hangman")
    self.assertIn("python_hangman hides", _mcts_reason(game) or "")

  def test_only_mcts_reads_the_full_state(self) -> None:
    self.assertEqual(
        {
            name
            for name, kind in controllers.kinds().items()
            if kind.reads_full_state
        },
        {controllers.MCTS},
    )

  def test_descriptors_describe_the_registered_controllers(self) -> None:
    self.assertEqual(
        tuple(
            descriptor.controller
            for descriptor in controllers.descriptors(
                pyspiel.load_game("kuhn_poker")
            )
        ),
        controllers.registered(),
    )

  def test_available_lists_only_the_controllers_that_can_play(self) -> None:
    self.assertEqual(
        [
            controller.name
            for controller in controllers.available(
                pyspiel.load_game("kuhn_poker")
            )
        ],
        [controllers.HUMAN, controllers.RANDOM],
    )
    self.assertEqual(
        controllers.available(pyspiel.load_game("tic_tac_toe")),
        controllers.registered(),
    )

  def test_a_controller_without_settings_has_an_empty_schema(self) -> None:
    human, uniform, _ = controllers.registered()
    self.assertEqual(human.config_schema, {})
    self.assertEqual(uniform.config_schema, {})

  def test_the_mcts_schema_admits_no_unknown_settings(self) -> None:
    schema = _mcts(pyspiel.load_game("chess")).controller.config_schema
    self.assertEqual(schema["type"], "object")
    self.assertIs(schema["additionalProperties"], False)

  def test_the_mcts_schema_states_defaults_and_bounds(self) -> None:
    self.assertEqual(
        _mcts(pyspiel.load_game("chess")).controller.config_schema[
            "properties"
        ],
        {
            "max_simulations": {
                "type": "integer",
                "default": 250,
                "description": (
                    "Simulations per move, each ending in a random rollout."
                ),
                "minimum": 1,
                "maximum": 250,
            },
            "uct_c": {
                "type": "number",
                "default": 2.0,
                "description": (
                    "The exploration constant of the UCT selection rule."
                ),
                "minimum": 0.0,
                "maximum": 100.0,
            },
            "solve": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Whether proven wins and losses are backed up the search "
                    "tree."
                ),
            },
        },
    )


class CreateTest(absltest.TestCase):

  def test_an_unknown_controller_is_refused(self) -> None:
    with self.assertRaisesRegex(controllers.UnavailableError, "'alphazero'"):
      controllers.create(
          controllers.Spec("alphazero"), pyspiel.load_game("chess"), seed=0
      )

  def test_a_refusal_is_the_contracts_controller_unavailable(self) -> None:
    with self.assertRaises(errors.ControllerUnavailableError) as raised:
      controllers.create(
          controllers.Spec("alphazero"), pyspiel.load_game("chess"), seed=0
      )
    self.assertEqual(
        raised.exception.to_error().code,
        errors.ErrorCode.CONTROLLER_UNAVAILABLE,
    )

  def test_a_controller_that_cannot_play_the_game_is_refused(self) -> None:
    with self.assertRaisesRegex(controllers.UnavailableError, "kuhn_poker"):
      controllers.create(
          controllers.Spec(controllers.MCTS),
          pyspiel.load_game("kuhn_poker"),
          seed=0,
      )

  def test_invalid_settings_are_refused(self) -> None:
    invalid: tuple[tuple[controllers.Config, str], ...] = (
        ({"depth": 3}, "no setting 'depth'"),
        ({"max_simulations": "many"}, "takes integer values"),
        ({"max_simulations": True}, "takes integer values"),
        ({"max_simulations": 2.5}, "takes integer values"),
        ({"max_simulations": 0}, "lies in"),
        ({"max_simulations": 10**400}, "lies in"),
        ({"uct_c": -1.0}, r"lies in \[0.0, 100.0\]"),
        ({"uct_c": float("inf")}, "lies in"),
        ({"uct_c": float("nan")}, "lies in"),
        ({"solve": 1}, "takes boolean values"),
    )
    for config, message in invalid:
      with (
          self.subTest(config=config),
          self.assertRaisesRegex(controllers.UnavailableError, message),
      ):
        controllers.create(
            controllers.Spec(controllers.MCTS, config),
            pyspiel.load_game("tic_tac_toe"),
            seed=0,
        )

  def test_a_long_controller_name_is_cut_short_in_the_message(self) -> None:
    with self.assertRaises(controllers.UnavailableError) as raised:
      controllers.create(
          controllers.Spec("x" * 10_000), pyspiel.load_game("chess"), seed=0
      )
    self.assertLess(len(str(raised.exception)), 200)

  def test_a_controller_without_settings_refuses_any(self) -> None:
    with self.assertRaises(controllers.UnavailableError):
      controllers.create(
          controllers.Spec(controllers.RANDOM, {"seed": 1}),
          pyspiel.load_game("tic_tac_toe"),
          seed=0,
      )

  def test_an_integer_is_a_valid_number(self) -> None:
    controllers.create(
        controllers.Spec(controllers.MCTS, {"uct_c": 1}),
        pyspiel.load_game("tic_tac_toe"),
        seed=0,
    )


class HumanTest(absltest.TestCase):

  def test_a_human_seat_is_never_played_by_the_server(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    human = controllers.create(
        controllers.Spec(controllers.HUMAN), game, seed=0
    )
    with self.assertRaisesRegex(ValueError, "played by a person"):
      controllers.choose(human, game.new_initial_state(), 0)


class RandomTest(absltest.TestCase):

  def _play_out(self, seed: int) -> list[int]:
    """Returns the history of a game the random controller plays alone."""
    game = pyspiel.load_game("tic_tac_toe")
    bot = controllers.create(
        controllers.Spec(controllers.RANDOM), game, seed=seed
    )
    state = game.new_initial_state()
    while not state.is_terminal():
      state.apply_action(controllers.choose(bot, state, state.current_player()))
    return list(state.history())

  def test_the_same_seed_plays_the_same_game(self) -> None:
    self.assertEqual(self._play_out(seed=7), self._play_out(seed=7))

  def test_different_seeds_play_different_games(self) -> None:
    games = {tuple(self._play_out(seed)) for seed in range(10)}
    self.assertGreater(len(games), 1)

  def test_it_chooses_the_same_move_at_a_position_however_often_asked(
      self,
  ) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    bot = controllers.create(controllers.Spec(controllers.RANDOM), game, seed=5)
    state = game.new_initial_state()
    state.apply_action(4)
    moves = {controllers.choose(bot, state, 1) for _ in range(10)}
    self.assertLen(moves, 1)

  def test_it_plays_for_the_seat_in_an_imperfect_information_game(
      self,
  ) -> None:
    game = pyspiel.load_game("kuhn_poker")
    bot = controllers.create(controllers.Spec(controllers.RANDOM), game, seed=0)
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    self.assertIn(controllers.choose(bot, state, 0), state.legal_actions(0))


class MctsTest(absltest.TestCase):

  def _winning_position(self, game: pyspiel.Game) -> pyspiel.State:
    """X holds cells 0 and 1, O holds 3 and 4, and X wins at 2."""
    state = game.new_initial_state()
    for action in (0, 3, 1, 4):
      state.apply_action(action)
    return state

  def test_it_takes_a_winning_move(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    bot = controllers.create(controllers.Spec(controllers.MCTS), game, seed=0)
    self.assertEqual(
        controllers.choose(bot, self._winning_position(game), 0), 2
    )

  def test_the_same_seed_chooses_the_same_moves(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    spec = controllers.Spec(controllers.MCTS, {"max_simulations": 20})
    moves = [
        controllers.choose(
            controllers.create(spec, game, seed=3),
            game.new_initial_state(),
            0,
        )
        for _ in range(2)
    ]
    self.assertEqual(moves[0], moves[1])

  def test_it_chooses_the_same_move_at_a_position_however_often_asked(
      self,
  ) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    spec = controllers.Spec(controllers.MCTS, {"max_simulations": 20})
    bot = controllers.create(spec, game, seed=3)
    state = game.new_initial_state()
    state.apply_action(4)
    moves = {controllers.choose(bot, state, 1) for _ in range(5)}
    self.assertLen(moves, 1)

  def test_max_simulations_is_bounded(self) -> None:
    game = pyspiel.load_game("tic_tac_toe")
    bound = _CAP
    controllers.create(
        controllers.Spec(controllers.MCTS, {"max_simulations": bound}),
        game,
        seed=0,
    )
    with self.assertRaisesRegex(
        controllers.UnavailableError, f"lies in \\[1, {bound}\\]"
    ):
      controllers.create(
          controllers.Spec(controllers.MCTS, {"max_simulations": bound + 1}),
          game,
          seed=0,
      )

  def test_it_takes_a_winning_move_in_a_python_game(self) -> None:
    game = pyspiel.load_game("python_tic_tac_toe")
    bot = controllers.create(controllers.Spec(controllers.MCTS), game, seed=0)
    self.assertEqual(
        controllers.choose(bot, self._winning_position(game), 0), 2
    )


class ChooseTest(absltest.TestCase):

  def test_a_controller_that_raises_fails_its_seat(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    with self.assertRaises(controllers.ControllerError) as raised:
      controllers.choose(_Raises(), state, 0)
    self.assertEqual(raised.exception.seat, 0)
    self.assertEqual(raised.exception.error, "KeyError")
    self.assertIsInstance(raised.exception.__cause__, KeyError)

  def test_a_controller_that_returns_a_non_action_fails_its_seat(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    for value in (9, -1, None, True, "4", 4.0):
      with self.subTest(value=value):
        with self.assertRaises(controllers.ControllerError) as raised:
          controllers.choose(_Returns(value), state, 0)
        self.assertEqual(raised.exception.seat, 0)
        self.assertEqual(raised.exception.error, "IllegalAction")

  def test_an_occupied_cell_is_not_a_legal_action(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    state.apply_action(4)
    with self.assertRaises(controllers.ControllerError) as raised:
      controllers.choose(_Returns(4), state, 1)
    self.assertEqual(raised.exception.seat, 1)

  def test_a_failure_is_the_contracts_controller_failed_naming_the_seat(
      self,
  ) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    with self.assertRaises(errors.ControllerFailedError) as raised:
      controllers.choose(_Raises(), state, 0)
    error = raised.exception.to_error()
    self.assertEqual(error.code, errors.ErrorCode.CONTROLLER_FAILED)
    self.assertEqual(error.seat, 0)
    self.assertNotIn("the card in hand", error.message)

  def test_a_failure_message_is_withheld_from_other_viewers(self) -> None:
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    with self.assertRaises(controllers.ControllerError) as raised:
      controllers.choose(_Raises(), state, 0)
    self.assertEqual(
        raised.exception.failure(visible=False),
        contract.ControllerFailure(error="KeyError", message=None),
    )
    self.assertEqual(
        raised.exception.failure(visible=True),
        contract.ControllerFailure(
            error="KeyError", message="'the card in hand'"
        ),
    )

  def test_a_controller_cannot_change_the_position(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    self.assertEqual(controllers.choose(_Mutates(), state, 0), 4)
    self.assertEqual(state.history(), [])

  def test_a_seat_not_to_move_is_not_asked(self) -> None:
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    with self.assertRaisesRegex(ValueError, "Seat 1 is not to move"):
      controllers.choose(_Returns(4), state, 1)


if __name__ == "__main__":
  absltest.main()
