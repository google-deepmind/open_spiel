# Copyright 2019 DeepMind Technologies Limited
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

"""Some basic heuristic evaluation functions for games."""

from typing import Callable
import pyspiel


def evaluate_state(state: pyspiel.State, player: int):
  """Evaluates the given state for the specified player.

  Args:
    state: The state to evaluate.
    player: The player to evaluate for.

  Returns:
    The heuristic value of the state for the specified player. A higher value
    means the state is better for the specified player.
  """
  game_str = state.get_game().get_type().short_name
  if game_str not in HEURISTIC_CALLBACKS:
    raise ValueError(f"No heuristic callback found for game {game_str}.")
  return HEURISTIC_CALLBACKS[game_str](state, player)


def evaluate_state_chess(state: pyspiel.State, player: int):
  """Evaluates the given state."""
  board = state.board()
  value = 0
  for row in range(8):
    for col in range(8):
      square = pyspiel.chess.Square(row, col)
      piece = board.at(square)
      if piece.type == pyspiel.chess.PieceType.EMPTY:
        continue
      elif piece.type == pyspiel.chess.PieceType.PAWN:
        piece_val = 1
      elif piece.type == pyspiel.chess.PieceType.KNIGHT:
        piece_val = 3
      elif piece.type == pyspiel.chess.PieceType.BISHOP:
        piece_val = 3
      elif piece.type == pyspiel.chess.PieceType.ROOK:
        piece_val = 5
      elif piece.type == pyspiel.chess.PieceType.QUEEN:
        piece_val = 9
      elif piece.type == pyspiel.chess.PieceType.KING:
        piece_val = 0
      else:
        raise ValueError(f"Unknown piece type: {piece.piece_type}")
      # note that white is 1, black is 0
      if ((piece.color == pyspiel.chess.Color.WHITE and player == 1) or
          (piece.color == pyspiel.chess.Color.BLACK and player == 0)):
        value += piece_val
      else:
        value -= piece_val
  return value


HEURISTIC_CALLBACKS: dict[str, Callable[[pyspiel.State, int], float]] = {
    "chess": evaluate_state_chess,
}

