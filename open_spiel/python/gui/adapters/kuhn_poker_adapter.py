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

# STRUCT-GLUE(kuhn_poker StateStruct): kuhn_poker has no state struct, so the
# adapter replays the hand into the shared poker table (poker_table.py),
# which builds the universal-poker-shaped struct the `poker` board reads.
"""The kuhn_poker adapter: the hand as the poker board reads it.

Each seat antes one chip and is dealt one card from a deck of one more card
than there are seats. In turn each seat passes or bets one chip; after a bet,
a pass folds and a bet calls. In the board's words a pass is a check (`c`)
or a fold (`f`), and a bet is a bet to two chips (`r2`) or a call (`c`).

The view, and who sees what, is `poker_table.present`'s: a seat sees its own
card, and every card still in at a showdown is tabled.
"""

from collections.abc import Mapping, Sequence

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui.adapters import poker_table

GAME = "kuhn_poker"
_TITLE = "Kuhn Poker"

_PASS = 0
_BET = 1
# Chips: each seat's ante, and the size of the one bet.
_ANTE = 1
_BET_SIZE = 1
# Kuhn's cards have ranks only; they are drawn as spades.
_SUIT = "s"
# The ranks, lowest first, of the usual two- and three-seat decks; larger
# decks take the highest ranks of a full deck.
_SMALL_DECK_RANKS = "JQKA"
_FULL_DECK_RANKS = "23456789TJQKA"


def _rank_labels(num_cards: int) -> str:
  """The rank of each card of a deck of `num_cards`, lowest first."""
  if num_cards <= len(_SMALL_DECK_RANKS):
    return _SMALL_DECK_RANKS[:num_cards]
  if num_cards > len(_FULL_DECK_RANKS):
    raise ValueError(f"kuhn_poker with {num_cards} cards has no card names.")
  return _FULL_DECK_RANKS[-num_cards:]


def _kind(table: poker_table.Table, action: int) -> tuple[str, int]:
  """The betting move `action` is at `table`, and the raise it makes."""
  bet_made = table.raises > 0
  if action == _BET:
    return ("c", 0) if bet_made else ("r", _BET_SIZE)
  if action == _PASS:
    return ("f", 0) if bet_made else ("c", 0)
  raise ValueError(f"Unexpected kuhn_poker action {action}.")


def _table(state: pyspiel.State) -> poker_table.Table:
  """The hand so far, replayed from `state`'s history."""
  num_seats = state.num_players()
  ranks = _rank_labels(num_seats + 1)
  table = poker_table.Table(
      title=_TITLE, num_seats=num_seats, ante=_ANTE, board_sizes=(0,)
  )
  replay = state.get_game().new_initial_state()
  dealt = 0
  for action in state.history():
    if replay.is_chance_node():
      if dealt >= num_seats:
        raise ValueError(f"Unexpected kuhn_poker deal: {state.history()}")
      table.deal_hole(dealt, f"{ranks[action]}{_SUIT}")
      dealt += 1
    else:
      table.move(replay.current_player(), *_kind(table, action))
    replay.apply_action(action)
  # The replay's stakes must be the game's own.
  if state.is_terminal():
    returns = state.returns()
    for seat, paid in enumerate(table.contributions):
      if returns[seat] < 0 and -returns[seat] != paid:
        raise ValueError(
            f"kuhn_poker seat {seat} lost {-returns[seat]}, not its {paid}."
        )
  return table


class KuhnPokerAdapter(adapter.Adapter):
  """Presents a kuhn_poker hand to the poker board."""

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
        action: {"label": table.label(seat, *_kind(table, action))}
        for action in actions
    }

  def public_action_label(
      self, state: pyspiel.State, action: int
  ) -> str | None:
    """A seat's pass or bet is public; the cards dealt are not."""
    player = state.current_player()
    if player == pyspiel.PlayerId.CHANCE:
      return None
    return str(state.action_to_string(player, action))


ADAPTER = KuhnPokerAdapter()
