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


"""Tests for the checkers adapter."""

import json
import random
from typing import Any

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import loader
from open_spiel.python.gui import presenters
from open_spiel.python.gui import views
from open_spiel.python.gui.adapters import checkers_adapter

_ADAPTER = checkers_adapter.ADAPTER
_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)

# The most plies of a random game played.
_PLIES = 400


def _seat(seat: int) -> adapter.Viewpoint:
  """The viewpoint of seat `seat`."""
  return adapter.Viewpoint(seat=seat, omniscient=False)


def _view(state: pyspiel.State, viewpoint: adapter.Viewpoint) -> Any:
  """The view of `state` for `viewpoint`, untyped for reading."""
  return _ADAPTER.present(state, viewpoint)


def _hints(
    state: pyspiel.State, actions: list[int], viewpoint: adapter.Viewpoint
) -> Any:
  """The hints of `actions` for `viewpoint`, untyped for reading."""
  return _ADAPTER.describe_actions(state, actions, viewpoint)


def _state(*labels: str, game: str = "checkers") -> pyspiel.State:
  """The position after the moves `labels`, each played by its label."""
  state = pyspiel.load_game(game).new_initial_state()
  for label in labels:
    by_label = {
        state.action_to_string(state.current_player(), action): action
        for action in state.legal_actions()
    }
    state.apply_action(by_label[label])
  return state


# White jumps c7 to e5, taking Black's man on d6, and must jump on from e5
# to c3, taking d4.
_DOUBLE_JUMP = ("c3b4", "f6g5", "e3d4", "d6c5", "b4d6", "c7e5")


def _random_states(seed: int) -> list[pyspiel.State]:
  """Every position of a random game, the terminal one last."""
  rng = random.Random(seed)
  state = pyspiel.load_game("checkers").new_initial_state()
  states = [state.clone()]
  for _ in range(_PLIES):
    if state.is_terminal():
      break
    state.apply_action(rng.choice(state.legal_actions()))
    states.append(state.clone())
  return states


def _sq(row: int, col: int) -> dict[str, int]:
  """The square at `row`, `col` as the struct names it."""
  return {"row": row, "col": col}


class BoardTest(absltest.TestCase):

  def test_presents_the_opening(self) -> None:
    struct = _view(_state(), _ORACLE)["struct"]
    self.assertEqual(_ADAPTER.renderer, "checkers")
    self.assertEqual((struct["rows"], struct["columns"]), (8, 8))
    self.assertEqual(struct["currentPlayer"], 0)
    board = struct["board"]
    # Row 0 is rank 1. Black, player 0, has twelve men on ranks 1-3, White
    # twelve on ranks 6-8, all on the dark squares: a1 is dark.
    for row in range(8):
      for col in range(8):
        piece = board[row][col]
        if piece:
          self.assertEqual((row + col) % 2, 0, (row, col))
          self.assertEqual(piece, "b" if row < 3 else "w", (row, col))
    self.assertEqual(sum(row.count("b") for row in board), 12)
    self.assertEqual(sum(row.count("w") for row in board), 12)
    self.assertEqual(board[0][0], "b")  # a1
    self.assertEqual(board[2][0], "b")  # a3

  def test_reads_any_board_size(self) -> None:
    struct = _view(_state(game="checkers(rows=10,columns=10)"), _ORACLE)[
        "struct"
    ]
    self.assertEqual((struct["rows"], struct["columns"]), (10, 10))
    self.assertEqual(len(struct["board"]), 10)
    self.assertEqual(len(struct["board"][0]), 10)
    struct = _view(_state(game="checkers(rows=6,columns=6)"), _ORACLE)["struct"]
    self.assertEqual((struct["rows"], struct["columns"]), (6, 6))

  def test_kings_appear_in_play(self) -> None:
    for seed in range(20):
      for state in _random_states(seed):
        text = str(state)
        board = _view(state, _ORACLE)["struct"]["board"]
        flat = [piece for row in board for piece in row]
        # "8" is also the row number of the top row.
        self.assertEqual(flat.count("B"), text.count("8") - 1)
        self.assertEqual(flat.count("W"), text.count("*"))
        if "B" in flat and "W" in flat:
          return
    self.fail("No random game crowned a king of each side.")

  def test_movable_sources_are_the_pieces_with_a_move(self) -> None:
    view = _view(_state(), _seat(0))
    self.assertCountEqual(
        view["movableSources"],
        [_sq(2, 0), _sq(2, 2), _sq(2, 4), _sq(2, 6)],
    )
    self.assertEqual(
        _view(_random_states(0)[-1], _ORACLE)["movableSources"], []
    )


class MovesTest(absltest.TestCase):

  def test_parses_moves_and_jumps(self) -> None:
    self.assertEqual(
        checkers_adapter.parse_action_label("a3b4"),
        {
            "type": "move",
            "moves": [{"from": _sq(2, 0), "to": _sq(3, 1)}],
            "displayLabel": "a3→b4",
        },
    )
    jump = checkers_adapter.parse_action_label("c7e5")
    self.assertEqual(jump["type"], "jump")
    self.assertEqual(
        checkers_adapter.parse_action_label("a10b9")["moves"],
        [{"from": _sq(9, 0), "to": _sq(8, 1)}],
    )

  def test_refuses_a_label_it_cannot_parse_naming_it(self) -> None:
    for label in ("", "x", "a3", "a3b4x", "pass", "a3-b4"):
      with self.subTest(label=label):
        with self.assertRaisesRegex(ValueError, repr(label)):
          checkers_adapter.parse_action_label(label)

  def test_parses_each_kind_of_label_the_game_writes(self) -> None:
    # A plain move from the start, and the jump that ends _DOUBLE_JUMP's
    # first capture, labelled by the served game itself.
    start = _state()
    move = start.action_to_string(0, start.legal_actions()[0])
    before_jump = _state(*_DOUBLE_JUMP[:-1])
    jumps = [
        before_jump.action_to_string(before_jump.current_player(), action)
        for action in before_jump.legal_actions()
    ]
    self.assertIn(_DOUBLE_JUMP[-1], jumps)
    self.assertEqual(checkers_adapter.parse_action_label(move)["type"], "move")
    self.assertEqual(
        checkers_adapter.parse_action_label(_DOUBLE_JUMP[-1])["type"], "jump"
    )
    # Every label of a random game parses.
    for state in _random_states(1):
      if state.is_terminal():
        continue
      player = state.current_player()
      for action in state.legal_actions():
        parsed = checkers_adapter.parse_action_label(
            state.action_to_string(player, action)
        )
        self.assertIn(parsed["type"], ("move", "jump"))

  def test_hints_each_move_from_its_square_to_its_landing(self) -> None:
    state = _state()
    hints = _hints(state, state.legal_actions(), _seat(0))
    by_label = {
        state.action_to_string(0, action): hint
        for action, hint in hints.items()
    }
    self.assertEqual(len(by_label), 7)
    self.assertEqual(
        by_label["a3b4"],
        {
            "actionType": "move",
            "displayLabel": "a3→b4",
            "moves": [{"from": _sq(2, 0), "to": _sq(3, 1)}],
            "fromSquare": _sq(2, 0),
            "toSquare": _sq(3, 1),
        },
    )

  def test_a_multiple_capture_is_one_jump_at_a_time(self) -> None:
    state = _state(*_DOUBLE_JUMP)
    # White has jumped once and still moves: only the jumping piece may.
    self.assertEqual(state.current_player(), 1)
    hints = list(_hints(state, state.legal_actions(), _seat(1)).values())
    self.assertEqual(len(hints), 1)
    self.assertEqual(hints[0]["actionType"], "jump")
    self.assertEqual(hints[0]["fromSquare"], _sq(4, 4))
    self.assertEqual(hints[0]["toSquare"], _sq(2, 2))
    self.assertEqual(_view(state, _seat(1))["movableSources"], [_sq(4, 4)])


class PlayTest(absltest.TestCase):

  def test_plays_whole_games(self) -> None:
    for seed in range(8):
      for state in _random_states(seed):
        view = _view(state, _seat(seed % 2))
        json.dumps(view, allow_nan=False)
        if not state.is_terminal():
          hints = _hints(
              state, state.legal_actions(), _seat(state.current_player())
          )
          self.assertCountEqual(hints, state.legal_actions())
          sources = {
              (hint["fromSquare"]["row"], hint["fromSquare"]["col"])
              for hint in hints.values()
          }
          self.assertEqual(
              sources,
              {(s["row"], s["col"]) for s in view["movableSources"]},
          )

  def test_refuses_another_game(self) -> None:
    with self.assertRaises(ValueError):
      checkers_adapter.extract_board_state(
          pyspiel.load_game("tic_tac_toe").new_initial_state()
      )


def _position(state: pyspiel.State) -> views.Position:
  """A session position at `state`, for building frames."""
  loaded = loader.load("checkers", {})
  seats = tuple(
      contract.SeatView(seat, contract.HUMAN, {}, False, None)
      for seat in range(loaded.num_seats)
  )
  return views.Position(
      info=contract.SessionInfo(
          id="s",
          game=loaded.name,
          parameters=dict(loaded.parameters),
          num_seats=loaded.num_seats,
          chance_policy=contract.ChancePolicy.AUTO,
          turn_based=loaded.turn_based,
          hides_information=loaded.hides_information,
          seed=1,
          controllers=controllers.available(loaded.game),
      ),
      loaded=loaded,
      status=contract.SessionStatus.PLAYING,
      seats=seats,
      state=state,
      ply=0,
      history=(),
  )


class PrivacyTest(absltest.TestCase):

  def test_every_viewer_is_sent_the_same_public_view(self) -> None:
    # Checkers hides nothing: no seat holds a hidden piece, so every viewer,
    # seat, spectator or oracle, is sent the same struct, movable squares
    # and move hints, and the view holds only the board and its moves.
    self.assertIs(presenters.adapter_for("checkers"), _ADAPTER)
    viewers = {
        "seat0": contract.Viewer(contract.ViewerKind.SEAT, 0),
        "seat1": contract.Viewer(contract.ViewerKind.SEAT, 1),
        "spectator": contract.Viewer(contract.ViewerKind.SPECTATOR),
        "oracle": contract.Viewer(contract.ViewerKind.ORACLE),
    }
    for state in _random_states(5)[:-1:7]:
      position = _position(state)
      built = {
          name: views.build(position, viewer).presentation
          for name, viewer in viewers.items()
      }
      shown = set()
      hints = set()
      for presentation in built.values():
        assert presentation is not None and presentation.actions is not None
        self.assertEqual(presentation.renderer, "checkers")
        view: Any = presentation.view
        self.assertEqual(set(view), {"struct", "movableSources"})
        self.assertEqual(
            set(view["struct"]), {"board", "rows", "columns", "currentPlayer"}
        )
        self.assertEqual(
            view["struct"], checkers_adapter.extract_board_state(state)
        )
        shown.add(json.dumps(view, sort_keys=True))
        hints.add(repr(presentation.actions))
        self.assertCountEqual(
            [hint.id for hint in presentation.actions], state.legal_actions()
        )
      self.assertEqual(len(shown), 1)
      self.assertEqual(len(hints), 1)


if __name__ == "__main__":
  absltest.main()
