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


"""Tests for the chess adapter."""

import json
import random

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui import presenters
from open_spiel.python.gui.adapters import chess_adapter

_BLACK = adapter.Viewpoint(seat=0, omniscient=False)
_WHITE = adapter.Viewpoint(seat=1, omniscient=False)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)
_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_VIEWPOINTS = (_BLACK, _WHITE, _SPECTATOR, _ORACLE)

_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_VIEW_FIELDS = ["chessCheck", "chessLastMove", "chessOutcome", "fen"]
_NO_OUTCOME = {"checkmate": False, "stalemate": False, "threefold": False}


def _play(state: pyspiel.State, *moves: str) -> pyspiel.State:
  """Plays `moves`, in standard algebraic notation, on `state`."""
  for move in moves:
    state.apply_action(state.parse_move_to_action(move))
  return state


def _new(game: str = "chess") -> pyspiel.State:
  """The initial state of `game`."""
  return pyspiel.load_game(game).new_initial_state()


def _present(
    state: pyspiel.State, viewpoint: adapter.Viewpoint = _WHITE
) -> dict[str, contract.JsonValue]:
  """The chess adapter's view of `state` for `viewpoint`."""
  return chess_adapter.ADAPTER.present(state, viewpoint)


class ChessAdapterTest(absltest.TestCase):
  """The chess adapter's views and hints."""

  def test_registered_with_the_chess_renderer(self) -> None:
    self.assertIs(presenters.adapter_for("chess"), chess_adapter.ADAPTER)
    self.assertEqual(chess_adapter.ADAPTER.renderer, "chess")

  def test_initial_position(self) -> None:
    self.assertEqual(
        _present(_new()),
        {
            "fen": _START,
            "chessCheck": None,
            "chessLastMove": None,
            "chessOutcome": _NO_OUTCOME,
        },
    )

  def test_last_move(self) -> None:
    view = _present(_play(_new(), "e4"), _BLACK)
    self.assertEqual(view["chessLastMove"], {"from": "e2", "to": "e4"})
    state = _play(_new(), "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O")
    view = _present(state)
    # Castling is the king's move of two squares.
    self.assertEqual(view["chessLastMove"], {"from": "e1", "to": "g1"})

  def test_check_names_the_side_in_check(self) -> None:
    view = _present(_play(_new(), "e4", "f5", "Qh5+"), _BLACK)
    self.assertEqual(view["chessCheck"], "black")
    view = _present(_play(_new(), "e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+"))
    self.assertEqual(view["chessCheck"], "white")
    self.assertEqual(view["chessOutcome"], _NO_OUTCOME)

  def test_checkmate(self) -> None:
    view = _present(_play(_new(), "f3", "e5", "g4", "Qh4#"), _SPECTATOR)
    self.assertEqual(view["chessCheck"], "white")
    self.assertEqual(view["chessLastMove"], {"from": "d8", "to": "h4"})
    self.assertEqual(
        view["chessOutcome"],
        {"checkmate": True, "stalemate": False, "threefold": False},
    )

  def test_stalemate(self) -> None:
    state = pyspiel.load_game("chess").new_initial_state(
        "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1"
    )
    state = _play(state, "Qg6")
    self.assertTrue(state.is_terminal())
    view = _present(state)
    self.assertIsNone(view["chessCheck"])
    self.assertEqual(
        view["chessOutcome"],
        {"checkmate": False, "stalemate": True, "threefold": False},
    )

  def test_threefold_repetition(self) -> None:
    state = _play(_new(), *(["Nf3", "Nf6", "Ng1", "Ng8"] * 2))
    self.assertTrue(state.is_terminal())
    self.assertEqual(
        _present(state)["chessOutcome"],
        {"checkmate": False, "stalemate": False, "threefold": True},
    )

  def test_moves_are_hinted_from_and_to(self) -> None:
    state = _new()
    hints = chess_adapter.ADAPTER.describe_actions(
        state, state.legal_actions(), _WHITE
    )
    self.assertEqual(len(hints), 20)
    self.assertEqual(
        hints[state.parse_move_to_action("e4")], {"from": "e2", "to": "e4"}
    )
    self.assertEqual(
        hints[state.parse_move_to_action("Nf3")], {"from": "g1", "to": "f3"}
    )

  def test_only_the_given_actions_are_hinted(self) -> None:
    state = _new()
    actions = state.legal_actions()[:3]
    hints = chess_adapter.ADAPTER.describe_actions(state, actions, _WHITE)
    self.assertEqual(sorted(hints), sorted(actions))

  def test_castling_moves_the_king_two_squares(self) -> None:
    state = _play(_new(), "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
    action = state.parse_move_to_action("O-O")
    hints = chess_adapter.ADAPTER.describe_actions(state, [action], _WHITE)
    self.assertEqual(hints[action], {"from": "e1", "to": "g1"})

  def test_promotions_share_their_squares(self) -> None:
    state = pyspiel.load_game("chess").new_initial_state(
        "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1"
    )
    hints = chess_adapter.ADAPTER.describe_actions(
        state, state.legal_actions(), _WHITE
    )
    promotions = {
        hint["promotion"]: (hint["from"], hint["to"])
        for hint in hints.values()
        if isinstance(hint, dict) and "promotion" in hint
    }
    self.assertEqual(sorted(promotions), ["b", "n", "q", "r"])
    for squares in promotions.values():
      self.assertEqual(squares, ("b7", "b8"))

  def test_check_agrees_with_the_game_over_random_games(self) -> None:
    rng = random.Random(7)
    checks = 0
    for _ in range(6):
      state = _new()
      while not state.is_terminal() and state.move_number() < 300:
        action = rng.choice(state.legal_actions())
        san = state.action_to_string(action)
        state.apply_action(action)
        view = _present(state, _ORACLE)
        expected = san.endswith(("+", "#"))
        self.assertEqual(view["chessCheck"] is not None, expected, san)
        outcome = view["chessOutcome"]
        assert isinstance(outcome, dict)
        self.assertEqual(outcome["checkmate"], san.endswith("#"), san)
        checks += expected
    self.assertGreater(checks, 0)

  def test_every_viewer_is_given_the_same_view_and_hints(self) -> None:
    # The privacy check. Chess hides nothing: each seat, a spectator and the
    # oracle get the same view, holding only the board's four fields, and
    # the same hints, so no viewer is shown anything another is not.
    rng = random.Random(3)
    state = _new()
    for _ in range(80):
      if state.is_terminal():
        break
      views = set()
      for viewpoint in _VIEWPOINTS:
        view = _present(state, viewpoint)
        self.assertEqual(sorted(view), _VIEW_FIELDS)
        views.add(json.dumps(view, sort_keys=True))
      self.assertEqual(len(views), 1)
      actions = state.legal_actions()
      hints = {
          json.dumps(
              chess_adapter.ADAPTER.describe_actions(state, actions, viewpoint),
              sort_keys=True,
          )
          for viewpoint in _VIEWPOINTS
      }
      self.assertEqual(len(hints), 1)
      state.apply_action(rng.choice(actions))

  def test_no_action_is_withheld_or_relabelled(self) -> None:
    state = _new()
    action = state.legal_actions()[0]
    self.assertIsNone(chess_adapter.ADAPTER.public_action_label(state, action))

  def test_presentation_is_json(self) -> None:
    state = _play(_new(), "e4")
    presentation = presenters.present(
        "chess", state, _BLACK, state.legal_actions()
    )
    assert presentation is not None
    self.assertEqual(presentation.renderer, "chess")
    assert presentation.actions is not None
    self.assertEqual(len(presentation.actions), 20)
    json.dumps(presentation.view, allow_nan=False)

  def test_chess960_opens_with_no_last_move(self) -> None:
    state = _new("chess(chess960=true)")
    state.apply_action(state.chance_outcomes()[0][0])
    view = _present(state)
    self.assertIsNone(view["chessLastMove"])
    hints = chess_adapter.ADAPTER.describe_actions(
        state, state.legal_actions(), _WHITE
    )
    self.assertEqual(len(hints), len(state.legal_actions()))

  def test_parse_board_refuses_a_bad_placement(self) -> None:
    with self.assertRaises(ValueError):
      chess_adapter.parse_board("rnbqkbnr/ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")


if __name__ == "__main__":
  absltest.main()
