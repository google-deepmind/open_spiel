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

# STRUCT-GLUE(checkers StateStruct): the game has no state struct, so the
# adapter reads the board from the state's text into the struct-shaped record
# the board draws, and parses each legal move's label into its squares.
"""The checkers adapter: the board as a struct, and each move's squares.

The view is the record the `checkers` client renderer reads:

- `struct`: `board`, rows x columns from the bottom (row 0 is rank 1, column
  0 is file a), each cell "" (empty), "b" or "B" (player 0's man or king,
  Black), "w" or "W" (player 1's man or king, White); `rows` and `columns`,
  the board's size; and `currentPlayer`.
- `movableSources`: the squares, as {row, col}, of the pieces with a legal
  move.

Each legal action's hint names its kind (`actionType`, "move" or "jump"),
its `displayLabel` ("c3→d4"), its `moves` ({from, to} per leg) and its
`fromSquare` and `toSquare`.

The game plays a multiple capture as one action per jump, the same player to
move until the capturing piece can jump no more.

Checkers hides nothing, so every viewer is sent the same view and hints;
the client turns the board round for the second seat.
"""

from collections.abc import Mapping, Sequence
import re

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "checkers"

Json = contract.JsonValue

# The game's text glyph of each piece, to the struct's: player 0's man and
# king ("o", "8"), player 1's ("+", "*").
_GLYPHS = {"o": "b", "8": "B", "+": "w", "*": "W"}
_EMPTY = "."
_SQUARE = re.compile(r"([a-z])(\d+)")
# A whole label: two or more squares, nothing else.
_LABEL = re.compile(r"(?:[a-z]\d+){2,}")


def _ensure_checkers_state(state: pyspiel.State) -> pyspiel.State:
  """Returns `state`, a checkers state.

  Raises:
    ValueError: If `state` is not a checkers state.
  """
  game_name = state.get_game().get_type().short_name
  if game_name != GAME:
    raise ValueError(f"Expected a checkers state, got {game_name}.")
  return state


def _square(row: int, col: int) -> dict[str, Json]:
  """A square as the struct names it."""
  return {"row": row, "col": col}


def parse_action_label(label: str) -> dict[str, Json]:
  """Parses a move's label into its kind, its legs and a display label.

  A label is "a3b4" (from a3 to b4) or "a3c5" (a jump). Any board size: a
  column is one letter and a row any number of digits, so a 10x10 board's
  "a10b9" and "j4i5" parse too.

  Args:
    label: The game's label for the move.

  Returns:
    `type` ("move" or "jump"), `displayLabel` (the squares joined by
    arrows), and `moves`: {from, to} per leg, each a {row, col} with row 0
    at the bottom.

  Raises:
    ValueError: If `label` is not two or more squares, naming the label.
  """
  squares = label.strip().lower()
  if not _LABEL.fullmatch(squares):
    raise ValueError(f"Cannot parse the checkers move label {label!r}.")
  matches = _SQUARE.findall(squares)
  moves: list[Json] = []
  for (from_col, from_row), (to_col, to_row) in zip(matches, matches[1:]):
    moves.append({
        "from": _square(int(from_row) - 1, ord(from_col) - ord("a")),
        "to": _square(int(to_row) - 1, ord(to_col) - ord("a")),
    })
  # A jump (a capture) moves two squares diagonally.
  first_from, first_to = matches[0], matches[1]
  is_jump = abs(int(first_to[1]) - int(first_from[1])) == 2
  return {
      "type": "jump" if is_jump else "move",
      "moves": moves,
      "displayLabel": "→".join(f"{col}{row}" for col, row in matches),
  }


def extract_board_state(state: pyspiel.State) -> dict[str, Json]:
  """Reads the struct-shaped board from the game's text.

  Any board size (the game's `rows` and `columns` parameters): the size is
  read from the text itself. The text is one line per row, top row first,
  each a right-aligned row number then one glyph per column, then a footer
  of column letters indented by the row-number width. The row number is
  fixed-width, never matched as digits: "8" is also player 0's king, so
  "88.+" is row 8 with a king on column a.

  Args:
    state: A checkers state.

  Returns:
    `board` (rows x columns from the bottom), `rows`, `columns` and
    `currentPlayer`.

  Raises:
    ValueError: If the text is not a checkers board.
  """
  _ensure_checkers_state(state)
  text = str(state)
  lines = text.rstrip("\n").split("\n")
  if len(lines) < 2:
    raise ValueError(f"Unparseable checkers board text: {text!r}")
  footer = lines[-1].strip()
  row_lines = lines[:-1]
  rows = len(row_lines)
  columns = len(footer)
  label_width = len(str(rows))
  if columns == 0 or not footer.isalpha():
    raise ValueError(f"Checkers footer {lines[-1]!r} is not column letters")
  board = [["" for _ in range(columns)] for _ in range(rows)]
  for line in row_lines:
    label = line[:label_width].strip()
    content = line[label_width:]
    if not label.isdigit() or len(content) != columns:
      raise ValueError(
          f"Checkers row {line!r} does not match a {rows}x{columns} board"
      )
    row_idx = int(label) - 1  # 0 = bottom
    if not 0 <= row_idx < rows:
      raise ValueError(f"Checkers row number {label} outside 1..{rows}")
    for col_idx, char in enumerate(content):
      piece = _GLYPHS.get(char)
      if piece is not None:
        board[row_idx][col_idx] = piece
      elif char != _EMPTY:
        raise ValueError(f"Unknown checkers glyph {char!r} in row {line!r}")
  cells: list[Json] = [list(row) for row in board]
  return {
      "board": cells,
      "rows": rows,
      "columns": columns,
      "currentPlayer": state.current_player(),
  }


def augment_action(label: str) -> dict[str, Json]:
  """The board's fields for a move with label `label`.

  Returns:
    `actionType` ("move" or "jump"), `displayLabel`, `moves`, `fromSquare`
    (where the piece starts) and `toSquare` (where it ends).

  Raises:
    ValueError: If `label` is not a move's squares, naming the label.
  """
  parsed = parse_action_label(label)
  entry: dict[str, Json] = {
      "actionType": parsed["type"],
      "displayLabel": parsed["displayLabel"],
  }
  moves = parsed["moves"]
  assert isinstance(moves, list) and moves
  first, last = moves[0], moves[-1]
  assert isinstance(first, dict) and isinstance(last, dict)
  entry["moves"] = moves
  entry["fromSquare"] = first["from"]
  entry["toSquare"] = last["to"]
  return entry


def compute_movable_sources(state: pyspiel.State) -> list[Json]:
  """The squares of the pieces that have a legal move, each once, in order."""
  if state.is_terminal():
    return []
  player = state.current_player()
  sources: dict[tuple[int, int], Json] = {}
  for action in state.legal_actions():
    from_square = augment_action(state.action_to_string(player, action)).get(
        "fromSquare"
    )
    if isinstance(from_square, dict):
      row, col = from_square["row"], from_square["col"]
      assert isinstance(row, int) and isinstance(col, int)
      sources.setdefault((row, col), from_square)
  return list(sources.values())


class CheckersAdapter(adapter.Adapter):
  """Presents a checkers position as the board's struct and its moves."""

  renderer = "checkers"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, Json]:
    """Returns the board's struct and the squares that may move."""
    del viewpoint  # The game hides nothing.
    return {
        "struct": extract_board_state(state),
        "movableSources": compute_movable_sources(state),
    }

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: adapter.Viewpoint,
  ) -> Mapping[int, Json]:
    """Returns each move's kind, legs and start and end squares."""
    del viewpoint  # The game hides nothing.
    _ensure_checkers_state(state)
    player = state.current_player()
    return {
        action: augment_action(state.action_to_string(player, action))
        for action in actions
    }


ADAPTER = CheckersAdapter()
