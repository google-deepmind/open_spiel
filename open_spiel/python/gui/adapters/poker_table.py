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

# STRUCT-GLUE(kuhn_poker StateStruct, leduc_poker StateStruct): neither game
# has a state struct, so a poker adapter replays the hand into this table and
# builds the universal-poker-shaped struct the `poker` board reads.
"""The poker table the kuhn_poker and leduc_poker adapters share.

The `poker` client renderer is the poker board, which reads a struct shaped
like universal_poker's state struct (`board_cards`, `player_hands`,
`player_contributions`, `starting_stacks`, `pot_size`, `current_player`,
`betting_history`, `blinds`, `best_hand_rank_types`,
`best_five_card_hands`), the history's labels in universal_poker's words
(`player=0 move=Bet2`), and the legal actions under the same words. A game's
adapter replays its hand into a `Table` (each seat's cards, the public cards,
each betting action), and `present` builds that view.

The betting string is ACPC's: one action per character, `f` a fold, `c` a
check or call, `r<n>` a bet or raise to `n` chips in all (the ante counted),
and `/` between rounds. Chips are the hand's totals, the ante included.

PRIVACY. A seat's hole cards, and the hand strength computed from them, are
served to the seat itself and to an omniscient viewer. At a showdown the
rules table every hand still in the pot, so once the hand is over and two or
more seats are still in it, those seats' cards are served to everyone. A
folded hand stays face down, and a hand won by fold shows no one's cards: the
winner is never made to show. The public cards, the pot, every contribution
and every betting action are public at the table. In the history the cards
dealt to the seats read "Chance outcome (hidden)" to every viewer but the
oracle, as their cards are shown on the table and nowhere else; a public card
reads `Deal <card>`.
"""

from collections.abc import Mapping, Sequence
import dataclasses

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

Json = contract.JsonValue

# The struct's `current_player` once the hand is over.
TERMINAL_PLAYER = -4
# The history label of a card dealt to a seat, for a viewer who may not see
# it: the board names no dealt card in its history.
HIDDEN_DEAL = "Chance outcome (hidden)"
# Hand strengths, in the board's words.
HIGH_CARD = "High Card"
PAIR = "Pair"


@dataclasses.dataclass
class Table:
  """A poker hand, replayed: the cards, the stakes and the betting.

  Attributes:
    title: The game's name, shown on the board's hand bar.
    num_seats: The seats at the table.
    ante: Each seat's ante, in chips.
    board_sizes: The public cards turned before each round's betting.
    hole: Each seat's hole card code ("Ks"), or None before its deal.
    board: The public cards turned so far.
    contributions: Each seat's chips in the pot.
    folded: Whether each seat has folded.
    rounds: The ACPC betting of each round so far.
    labels: Each history entry's label, in the board's words, and the seat
      whose card a chance entry deals (None for a seat's action or a public
      card).
    level: The contribution every seat still in must match.
    raises: The bets and raises made in the current round.
  """

  title: str
  num_seats: int
  ante: int
  board_sizes: tuple[int, ...]
  hole: list[str | None] = dataclasses.field(default_factory=list)
  board: list[str] = dataclasses.field(default_factory=list)
  contributions: list[int] = dataclasses.field(default_factory=list)
  folded: list[bool] = dataclasses.field(default_factory=list)
  rounds: list[list[str]] = dataclasses.field(default_factory=lambda: [[]])
  labels: list[tuple[str, int | None]] = dataclasses.field(default_factory=list)
  level: int = 0
  raises: int = 0

  def __post_init__(self) -> None:
    self.hole = [None] * self.num_seats
    self.contributions = [self.ante] * self.num_seats
    self.folded = [False] * self.num_seats
    self.level = self.ante

  def deal_hole(self, seat: int, card: str) -> None:
    """Records `card` dealt face down to `seat`."""
    if self.hole[seat] is not None:
      raise ValueError(f"Seat {seat} is dealt a second card {card}.")
    self.hole[seat] = card
    self.labels.append((f"player=-1 move=Deal {card}", seat))

  def deal_board(self, card: str) -> None:
    """Records `card` turned face up, which starts the next round."""
    self.board.append(card)
    self.rounds.append([])
    self.raises = 0
    self.labels.append((f"player=-1 move=Deal {card}", None))

  def owed(self, seat: int) -> int:
    """The chips `seat` must put in to stay in."""
    return self.level - self.contributions[seat]

  def move(self, seat: int, kind: str, raise_size: int = 0) -> str:
    """Records `seat`'s fold ("f"), check or call ("c") or raise ("r").

    Returns:
      The move in the board's words: "Fold", "Call", or "Bet<n>" for a bet
      or raise to n chips in all.
    """
    if kind == "f":
      self.folded[seat] = True
      words = "Fold"
    elif kind == "c":
      self.contributions[seat] = self.level
      words = "Call"
    elif kind == "r":
      if raise_size <= 0:
        raise ValueError(f"A raise of {raise_size} chips.")
      self.level += raise_size
      self.contributions[seat] = self.level
      self.raises += 1
      words = f"Bet{self.level}"
      kind = f"r{self.level}"
    else:
      raise ValueError(f"Unknown betting move {kind!r}.")
    self.rounds[-1].append(kind)
    self.labels.append((f"player={seat} move={words}", None))
    return words

  def label(self, seat: int, kind: str, raise_size: int = 0) -> str:
    """The legal action's label: the move `move` would record, unmade."""
    if kind == "f":
      words = "Fold"
    elif kind == "c":
      words = "Call"
    elif kind == "r":
      words = f"Bet{self.level + raise_size}"
    else:
      raise ValueError(f"Unknown betting move {kind!r}.")
    return f"player={seat} move={words}"

  @property
  def betting_history(self) -> str:
    """The ACPC betting string: "cr2c", "r3c/cr7f"."""
    return "/".join("".join(actions) for actions in self.rounds)


def showdown_seats(struct: Mapping[str, object]) -> frozenset[int]:
  """The seats whose hands are tabled: those still in at a finished showdown.

  Empty while the hand is in play and for a hand won by fold. A seat is still
  in when it has matched the largest contribution (these games have no
  stacks, so no seat is all-in for less): a fold is open only to a seat that
  owes chips, so a folded seat is always short. Cross-checked against the
  folds in the betting string (each `f` is one fold); a disagreement raises.
  """
  if struct.get("current_player") != TERMINAL_PLAYER:
    return frozenset()
  spent = struct.get("player_contributions")
  if not isinstance(spent, list) or not spent:
    raise ValueError(f"poker contributions are not a list: {spent!r:.200}")
  betting = struct.get("betting_history")
  if not isinstance(betting, str):
    raise ValueError(f"poker betting_history is not a string: {betting!r}")
  top = max(int(paid) for paid in spent if isinstance(paid, int))
  live = frozenset(seat for seat, paid in enumerate(spent) if paid == top)
  folds = betting.count("f")
  if len(live) + folds != len(spent):
    raise ValueError(
        f"poker: {len(live)} seats still in and {folds} folds in "
        f"{betting!r} do not add up to {len(spent)} seats"
    )
  return live if len(live) > 1 else frozenset()


# Per-seat lists in the struct whose entries are a seat's private cards or
# are computed from them.
_PER_SEAT_PRIVATE = (
    "player_hands",
    "best_five_card_hands",
    "best_hand_rank_types",
)


def viewer_struct(
    struct: Mapping[str, Json], viewpoint: adapter.Viewpoint
) -> dict[str, Json]:
  """Keeps the viewer's own private entries and any hand tabled at showdown.

  Every other seat's entry is an empty string, which the board draws as an
  empty card slot.
  """
  tabled = showdown_seats(struct)
  out = dict(struct)
  for key in _PER_SEAT_PRIVATE:
    values = struct[key]
    if not isinstance(values, list):
      raise ValueError(f"poker {key} is not a list: {values!r:.200}")
    out[key] = [
        value if viewpoint.sees_private(seat) or seat in tabled else ""
        for seat, value in enumerate(values)
    ]
  return out


def _strength(hole: str | None, board: Sequence[str]) -> tuple[str, str]:
  """A seat's hand strength and the cards that make it.

  A hole card that pairs a public card is a pair; otherwise the hand is the
  hole card alone, a high card.
  """
  if hole is None:
    return "", ""
  for card in board:
    if card[0] == hole[0]:
      return PAIR, hole + card
  return HIGH_CARD, hole


def struct_of(table: Table, current_player: int) -> dict[str, Json]:
  """The table as the full universal-poker-shaped struct, nothing hidden.

  A game with no public cards (Kuhn) has no hand categories: the higher card
  wins. Its seats' rank types are empty, so the board names none, and each
  seat's best hand is its card alone.
  """
  strengths = [_strength(hole, table.board) for hole in table.hole]
  if sum(table.board_sizes) == 0:
    strengths = [("", cards) for _, cards in strengths]
  return {
      "board_cards": "".join(table.board),
      "player_hands": [hole or "" for hole in table.hole],
      "player_contributions": list(table.contributions),
      "starting_stacks": [],
      "pot_size": sum(table.contributions),
      "current_player": current_player,
      "betting_history": table.betting_history,
      "blinds": [],
      "ante": table.ante,
      "best_hand_rank_types": [rank for rank, _ in strengths],
      "best_five_card_hands": [cards for _, cards in strengths],
  }


def present(
    table: Table, current_player: int, viewpoint: adapter.Viewpoint
) -> dict[str, Json]:
  """The board's view of `table` for `viewpoint`.

  - `struct`: the universal-poker-shaped struct, filtered for the viewer.
  - `historyActionStrings`: each history entry's label in the board's
    words; a card dealt to a seat is hidden from every viewer but the oracle.
  - `pokerGame`: what the board's hand bar and layout need of the game: its
    title, its betting ("limit": one bet size), the hole cards a seat holds
    and the public cards turned before each round.
  """
  struct = viewer_struct(struct_of(table, current_player), viewpoint)
  history: list[Json] = [
      HIDDEN_DEAL if dealt is not None and not viewpoint.omniscient else text
      for text, dealt in table.labels
  ]
  game: dict[str, Json] = {
      "title": table.title,
      "betting": "limit",
      "holeCards": 1,
      "boardSizes": list(table.board_sizes),
  }
  return {"struct": struct, "historyActionStrings": history, "pokerGame": game}
