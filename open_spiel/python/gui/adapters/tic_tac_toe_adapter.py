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

# STRUCT-GLUE(tic_tac_toe ObservationStruct): the board panel replays the
# action history onto its nine cells itself, so the view is the position's
# history and turn, not a struct.
"""The tic_tac_toe adapter: the position as the tic-tac-toe panel reads it.

The `tic_tac_toe` client renderer is the game's Gameplay panel, which marks
its 3 by 3 board by replaying the history (X first). The view holds, under
the names the panel reads:

- `history`, `historyStr`: the actions applied (each the cell marked, row by
  row from the top left), and the game's text for them.
- `currentPlayer`, `currentPlayerLabel`: the seat to move, and who that is in
  words ("player 0"; "terminal" once the game is over).
- `isTerminal`, `returns`: whether the game is over, and each seat's return
  then (null before).
- `turnNumber`: the number of actions applied.
- `gameParams`: the game's parameters.

Tic-tac-toe hides nothing, so every viewer is omniscient and gets the same
view; the adapter refuses a viewer that is not, rather than guess what to
withhold. The legal actions reach the panel in the frame.
"""

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "tic_tac_toe"


def describe_player(state: pyspiel.State, player: int) -> str:
  """Who is to move, in words, as the board's status row names them."""
  if state.is_terminal():
    return "terminal"
  if player == pyspiel.PlayerId.CHANCE:
    return "chance"
  if player == pyspiel.PlayerId.TERMINAL:
    return "terminal"
  if player == pyspiel.PlayerId.SIMULTANEOUS:
    return "simultaneous"
  return f"player {player}"


def _require_public(state: pyspiel.State, viewpoint: adapter.Viewpoint) -> None:
  """Fails unless the game hides nothing, which the whole view assumes."""
  game_type = state.get_game().get_type()
  if game_type.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
    raise ValueError(
        f"{game_type.short_name} hides information; its view would leak it."
    )
  if not viewpoint.omniscient:
    raise ValueError(
        "A game that hides nothing has only omniscient viewers; got"
        f" {viewpoint}."
    )


def _parameters(state: pyspiel.State) -> dict[str, contract.JsonValue]:
  """The game's parameters, defaults included."""
  params: dict[str, contract.JsonValue] = {}
  for name, value in state.get_game().get_parameters().items():
    if not isinstance(value, (bool, int, float, str)):
      raise TypeError(f"Parameter {name}={value!r} is not a JSON scalar.")
    params[name] = value
  return params


def session_state(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> dict[str, contract.JsonValue]:
  """The position under the field names the board panel reads."""
  _require_public(state, viewpoint)
  history = [int(action) for action in state.history()]
  is_terminal = state.is_terminal()
  current_player = int(state.current_player())
  returns: list[contract.JsonValue] | None = None
  if is_terminal:
    returns = [float(value) for value in state.returns()]
  return {
      "history": list(history),
      "historyStr": state.history_str(),
      "currentPlayer": current_player,
      "currentPlayerLabel": describe_player(state, current_player),
      "isTerminal": is_terminal,
      "returns": returns,
      "turnNumber": len(history),
      "gameParams": _parameters(state),
  }


class TicTacToeAdapter(adapter.Adapter):
  """Presents a tic-tac-toe position as the session its panel reads."""

  renderer = "tic_tac_toe"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, contract.JsonValue]:
    return session_state(state, viewpoint)


ADAPTER = TicTacToeAdapter()
