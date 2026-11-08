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

# STRUCT-GLUE(leduc_poker StateStruct): leduc_poker has no state struct, so
# the adapter replays the hand into the shared poker table (poker_table.py),
# which builds the universal-poker-shaped struct the `poker` board reads.
"""The leduc_poker adapter: the hand as the poker board reads it.

Leduc hold'em deals each seat one private card from a deck of two suits of
one more rank than there are seats, after an ante of one chip. A round of
fixed-limit betting follows (raises of two chips, at most two a round), then
one public card is turned and a second round is bet (raises of four). In the
board's words a fold is `f`, a check or call `c`, and a bet or raise to n
chips in all `r<n>`; the public card starts the second round (`/`).

The view, and who sees what, is `poker_table.present`'s: a seat sees its own
card, the public card is public once turned, and every card still in at a
showdown is tabled.
"""

from collections.abc import Mapping, Sequence
import re

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui.adapters import poker_table

GAME = "leduc_poker"
_TITLE = "Leduc Hold'em"

_FOLD = 0
_CALL = 1
_RAISE = 2
_KINDS = {_FOLD: "f", _CALL: "c", _RAISE: "r"}
# Chips: each seat's ante, and the fixed raise of each betting round.
_ANTE = 1
_RAISE_SIZES = (2, 4)
_SUITS = "sh"
# The ranks, lowest first, of the usual two- and three-seat decks; larger
# decks take the highest ranks of a full deck.
_SMALL_DECK_RANKS = "JQKA"
_FULL_DECK_RANKS = "23456789TJQKA"
# The pot as the game's text states it, to check the replay against.
_POT = re.compile(r"^Pot: (\d+)$", re.MULTILINE)
# Parameters that change what an action or a card means, which this adapter
# does not read.
_UNSUPPORTED = {"action_mapping": False, "suit_isomorphism": False}


def _card_code(card: int, num_ranks: int) -> str:
  """The code of the card numbered `card`: its rank, then its suit."""
  if num_ranks <= len(_SMALL_DECK_RANKS):
    ranks = _SMALL_DECK_RANKS[:num_ranks]
  elif num_ranks <= len(_FULL_DECK_RANKS):
    ranks = _FULL_DECK_RANKS[-num_ranks:]
  else:
    raise ValueError(f"leduc_poker with {num_ranks} ranks has no card names.")
  return f"{ranks[card // len(_SUITS)]}{_SUITS[card % len(_SUITS)]}"


def _move(table: poker_table.Table, action: int) -> tuple[str, int]:
  """The betting move `action` is at `table`, and the raise it makes."""
  kind = _KINDS.get(action)
  if kind is None:
    raise ValueError(f"Unexpected leduc_poker action {action}.")
  size = _RAISE_SIZES[len(table.board)] if kind == "r" else 0
  return kind, size


def _table(state: pyspiel.State) -> poker_table.Table:
  """The hand so far, replayed from `state`'s history."""
  game = state.get_game()
  parameters = game.get_parameters()
  for name, supported in _UNSUPPORTED.items():
    if parameters.get(name, supported) != supported:
      raise ValueError(f"The leduc_poker adapter does not read {name}.")
  num_seats = state.num_players()
  num_ranks = num_seats + 1
  table = poker_table.Table(
      title=_TITLE, num_seats=num_seats, ante=_ANTE, board_sizes=(0, 1)
  )
  replay = game.new_initial_state()
  dealt = 0
  for action in state.history():
    if not replay.is_chance_node():
      table.move(replay.current_player(), *_move(table, action))
    elif dealt < num_seats:
      table.deal_hole(dealt, _card_code(action, num_ranks))
      dealt += 1
    elif not table.board:
      table.deal_board(_card_code(action, num_ranks))
    else:
      raise ValueError(f"Unexpected leduc_poker deal: {state.history()}")
    replay.apply_action(action)
  if not state.is_terminal():
    stated = _POT.search(str(state))
    pot = sum(table.contributions)
    if stated is None or int(stated.group(1)) != pot:
      raise ValueError(
          f"The leduc_poker replay's pot {pot} disagrees with "
          f"the game's text: {str(state)!r}"
      )
  return table


class LeducPokerAdapter(adapter.Adapter):
  """Presents a leduc_poker hand to the poker board."""

  renderer = "poker"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, contract.JsonValue]:
    """Returns the board's view of `state` for `viewpoint`."""
    player = (
        poker_table.TERMINAL_PLAYER
        if state.is_terminal()
        else int(state.current_player())
    )
    return poker_table.present(_table(state), player, viewpoint)

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: adapter.Viewpoint,
  ) -> Mapping[int, contract.JsonValue]:
    """Names each betting action in the board's words."""
    del viewpoint
    table = _table(state)
    seat = state.current_player()
    return {
        action: {"label": table.label(seat, *_move(table, action))}
        for action in actions
    }

  def public_action_label(
      self, state: pyspiel.State, action: int
  ) -> str | None:
    """Betting is public, and so is the public card; a seat's card is not."""
    player = state.current_player()
    if player != pyspiel.PlayerId.CHANCE:
      return str(state.action_to_string(player, action))
    num_seats = state.num_players()
    # Seats' cards are dealt first, one each; the next card is the public one.
    if len(state.history()) < num_seats:
      return None
    return f"Public card {_card_code(action, num_seats + 1)}"


ADAPTER = LeducPokerAdapter()
