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

# STRUCT-GLUE(chess ObservationStruct): chess has no struct yet, so the
# adapter builds the view the board reads from the state's FEN and history.
"""The chess adapter: the position the chess board reads, for every viewer.

Chess hides nothing, so every viewer, seat, spectator or oracle, is given
the same view; the board turns itself to the viewer's seat. The view:

  {"fen": "...", "chessCheck": "white" | "black" | null,
   "chessLastMove": {"from": "e2", "to": "e4"} | null,
   "chessOutcome": {"checkmate": false, "stalemate": false,
                    "threefold": false}}

- `fen`: the position, as the board and its move list read it.
- `chessCheck`: the side whose king is in check, null when none is.
- `chessLastMove`: the squares of the move that reached the position.
- `chessOutcome`: how a finished game ended, as far as the position says
  (the board names the result and its reason from these and the returns).

Each legal move is hinted {"from": "e2", "to": "e4"}, with "promotion": "q"
(or "r", "b", "n") on a promotion, so the board can find the action for a
move dragged or clicked between two squares.

In OpenSpiel's chess, seat 0 plays black and seat 1 plays white.
"""

from collections.abc import Mapping, Sequence

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "chess"

_SIZE = 8
_BLACK_SEAT = 0
_WHITE_SEAT = 1

_KNIGHT_STEPS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
_KING_STEPS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
_ROOK_LINES = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_LINES = ((1, 1), (1, -1), (-1, 1), (-1, -1))

Board = list[list[str | None]]


def _fen(state: pyspiel.State) -> str:
  """The FEN of a chess state; raises on any other state."""
  board = getattr(state, "board", None)
  if board is None:
    raise TypeError(f"Not a chess state: {type(state).__name__}.")
  return str(board().to_fen())


def parse_board(fen: str) -> Board:
  """Returns the FEN's placement as rows from rank 8, each from file a."""
  placement = fen.split()[0]
  rows: Board = []
  for text in placement.split("/"):
    row: list[str | None] = []
    for letter in text:
      if letter.isdigit():
        row.extend([None] * int(letter))
      elif letter.lower() in "pnbrqk":
        row.append(letter)
      else:
        raise ValueError(f"Unexpected FEN placement: {fen!r}")
    if len(row) != _SIZE:
      raise ValueError(f"Unexpected FEN placement: {fen!r}")
    rows.append(row)
  if len(rows) != _SIZE:
    raise ValueError(f"Unexpected FEN placement: {fen!r}")
  return rows


def _white(piece: str) -> bool:
  """Whether `piece`, a FEN letter, is white."""
  return piece.isupper()


def _attacked(board: Board, row: int, column: int, by_white: bool) -> bool:
  """Whether the side `by_white` attacks the square at `row`, `column`."""

  def piece_at(r: int, c: int) -> str | None:
    """The piece at row `r`, column `c`; None off the board."""
    if 0 <= r < _SIZE and 0 <= c < _SIZE:
      return board[r][c]
    return None

  def is_attacker(piece: str | None, kinds: str) -> bool:
    """Whether `piece` is one of `kinds` of the attacking side."""
    return (
        piece is not None
        and _white(piece) == by_white
        and piece.lower() in kinds
    )

  # A white pawn attacks up the board (towards row 0).
  pawn_row = row + 1 if by_white else row - 1
  if any(is_attacker(piece_at(pawn_row, column + d), "p") for d in (-1, 1)):
    return True
  if any(
      is_attacker(piece_at(row + dr, column + dc), "n")
      for dr, dc in _KNIGHT_STEPS
  ):
    return True
  if any(
      is_attacker(piece_at(row + dr, column + dc), "k")
      for dr, dc in _KING_STEPS
  ):
    return True
  for lines, kinds in ((_ROOK_LINES, "rq"), (_BISHOP_LINES, "bq")):
    for dr, dc in lines:
      r, c = row + dr, column + dc
      while 0 <= r < _SIZE and 0 <= c < _SIZE:
        piece = board[r][c]
        if piece is not None:
          if is_attacker(piece, kinds):
            return True
          break
        r, c = r + dr, c + dc
  return False


def king_in_check(board: Board, white: bool) -> tuple[int, int] | None:
  """Returns the square of the `white` side's king if it is in check."""
  king = "K" if white else "k"
  for r, row in enumerate(board):
    for c, piece in enumerate(row):
      if piece == king:
        return (r, c) if _attacked(board, r, c, not white) else None
  return None  # No king: a test position.


def _chess960(state: pyspiel.State) -> bool:
  """Whether the state's game is Chess960."""
  return bool(state.get_game().get_parameters().get("chess960", False))


def _lan(state: pyspiel.State, action: int) -> str:
  """The move's long algebraic text, such as "e2e4" or "e7e8q"."""
  board = state.board()
  move = pyspiel.chess.action_to_move(action, board)
  return str(move.to_lan(_chess960(state), board)).lower()


def _move(lan: str) -> dict[str, contract.JsonValue]:
  """The squares of a long algebraic move, {"from": "e2", "to": "e4"}."""
  return {"from": lan[0:2], "to": lan[2:4]}


def _last_move(state: pyspiel.State, white_to_move: bool) -> str | None:
  """The long algebraic text of the move that reached `state`, if any."""
  history = state.history()
  # Chess960 opens with a chance node that draws the position.
  if _chess960(state) and len(history) == 1:
    return None
  if not history:
    return None
  parent = state.clone()
  parent.undo_action(_BLACK_SEAT if white_to_move else _WHITE_SEAT, history[-1])
  return _lan(parent, history[-1])


class ChessAdapter(adapter.Adapter):
  """Presents a chess position to the chess board."""

  renderer = "chess"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, contract.JsonValue]:
    # Chess hides nothing: every viewpoint is given the same view.
    del viewpoint
    fen = _fen(state)
    fields = fen.split()
    white_to_move = len(fields) < 2 or fields[1] == "w"
    in_check = king_in_check(parse_board(fen), white_to_move) is not None
    last = _last_move(state, white_to_move)
    terminal = state.is_terminal()
    stuck = terminal and not state.board().has_legal_moves()
    return {
        "fen": fen,
        "chessCheck": (
            ("white" if white_to_move else "black") if in_check else None
        ),
        "chessLastMove": None if last is None else _move(last),
        "chessOutcome": {
            "checkmate": stuck and in_check,
            "stalemate": stuck and not in_check,
            "threefold": terminal and bool(state.is_repetition_draw()),
        },
    }

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: adapter.Viewpoint,
  ) -> Mapping[int, contract.JsonValue]:
    del viewpoint  # Every viewer given the actions may see them.
    hints: dict[int, contract.JsonValue] = {}
    for action in actions:
      lan = _lan(state, action)
      hint = _move(lan)
      if len(lan) == 5:
        hint["promotion"] = lan[4]
      hints[action] = hint
    return hints


ADAPTER = ChessAdapter()
