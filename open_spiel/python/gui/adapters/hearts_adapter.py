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

# STRUCT-GLUE(hearts ObservationStruct): the board reads its own struct shape
# (hands by compass, the trick in play, the last finished trick and its
# winner, the points, the pass direction and the phase), read from the
# state's text and filtered for the viewer, not the engine's struct.
"""The hearts adapter: the board's struct for one viewer.

The view is the record the `hearts` client renderer reads:

- `struct`: the table, as the board's state hook reads it:
  - `hands`: each compass point ("N", "E", "S", "W") to its card codes
    ("Qs", "Th"). A hand the viewer may not see is one `None` per card,
    since how many cards a seat holds is public and the board draws a back
    for each. A seat sees its own hand; the oracle sees every hand; a
    spectator sees none.
  - `currentTrick`: the trick in play, each card with the seat that played
    it (`player`) and its compass point (`position`).
  - `completedTricks`: the number of finished tricks.
  - `lastTrick`: the last finished trick, its cards and the compass point of
    its winner, or None before the first; the board collects it toward the
    winner when `index` grows.
  - `points`: each compass point's points.
  - `passDirection`: "left", "right", "across" or "none".
  - `phase`: "passing" or "playing".
  - `currentPlayer`: the seat to act, or a negative player id.

Every legal action, a pass or a play, is hinted with the card it moves
(`cardCode`). `public_action_label` names every play in the history (it is
laid face up) and says a pass was made without naming its card; the deal and
the pass direction stay hidden chance outcomes.

The state's text is laid out by the engine as:

  Pass Direction: Left
  <blank>
  four lines of North's hand by suit ("        S 42"), four of West and
  East ("S JT763         S KQ8"), four of South
  Passed Cards: / Received Cards: (one line per seat, "N: KD 5C JC ")
  Tricks: (a header, then one line per trick, indented three columns per
  seat before the leader)
  Points: (one line per seat, "N: 0")
"""

from collections.abc import Mapping, Sequence
import re

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "hearts"

Json = contract.JsonValue

# Card action mapping: action = rank_idx * 4 + suit_idx, with ranks 2 to A
# and suits C, D, H, S.
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

COMPASS = ("N", "E", "S", "W")
_NUM_SEATS = len(COMPASS)

# The history of a hearts hand: the pass direction (chance), the 52 dealt
# cards (chance), then, unless the direction is "no pass", the twelve passed
# cards (3 per player) before the first lead.
_FIRST_PASS_INDEX = 1 + 52
_NUM_PASS_ACTIONS = 3 * 4
_NO_PASS = 0
PASS_HIDDEN_LABEL = "Passes a card (hidden)"


def action_to_card_code(action: int) -> str:
  """The card code of the engine's card `action`: "Ah", "2c"."""
  if not 0 <= action < len(RANKS) * len(SUITS):
    raise ValueError(f"Not a hearts card action: {action}")
  return f"{RANKS[action // 4]}{SUITS[action % 4]}"


def action_label_to_card_code(label: str) -> str:
  """The card code of the engine's card text: "2C" is "2c", "10H" is "Th"."""
  if len(label) < 2:
    return label
  rank = label[:-1]
  suit = label[-1].lower()
  if rank == "10":
    rank = "T"
  return f"{rank}{suit}"


def parse_hand_string(hand_str: str) -> list[str]:
  """The ranks of one suit line's cards: "KQ93" or "none" or empty."""
  hand_str = hand_str.strip()
  if not hand_str or hand_str.lower() == "none":
    return []
  ranks = []
  i = 0
  while i < len(hand_str):
    char = hand_str[i]
    if char == "1" and i + 1 < len(hand_str) and hand_str[i + 1] == "0":
      ranks.append("T")
      i += 2
    else:
      ranks.append(char)
      i += 1
  return ranks


def completed_trick(line: str, index: int) -> dict[str, Json]:
  """One finished trick from its text line, with its winner.

  The engine indents each trick line by three spaces per seat before the
  leader, under the header ``N  E  S  W  N  E  S``, and writes the cards in
  play order. Hearts has no trumps: the highest card of the led suit wins.
  `index` counts finished tricks from 1; `winner` is a compass letter; each
  card is ``{"card", "player", "position"}``.
  """
  indent = len(line) - len(line.lstrip(" "))
  if indent % 3 != 0 or not 0 <= indent // 3 < _NUM_SEATS:
    raise ValueError(f"hearts: trick line has no leader column: {line!r}")
  leader = indent // 3
  labels = line.split()
  if len(labels) != _NUM_SEATS:
    raise ValueError(f"hearts: finished trick without 4 cards: {line!r}")
  cards: list[dict[str, Json]] = []
  for i, label in enumerate(labels):
    player = (leader + i) % _NUM_SEATS
    code = action_label_to_card_code(label)
    if len(code) != 2 or code[0] not in RANK_IDX or code[1] not in SUITS:
      raise ValueError(f"hearts: unparseable card {label!r} in {line!r}")
    cards.append({"card": code, "player": player, "position": COMPASS[player]})
  led = str(cards[0]["card"])[1]
  winner = max(
      (c for c in cards if str(c["card"])[1] == led),
      key=lambda c: RANK_IDX[str(c["card"])[0]],
  )
  return {
      "index": index,
      "cards": list(cards),
      "winner": COMPASS[int(str(winner["player"]))],
  }


def _pass_direction(lines: Sequence[str]) -> str:
  """The pass direction of the text's first "Pass Direction:" line."""
  for line in lines:
    if "Pass Direction:" in line:
      direction = line.split(":")[-1].strip().lower()
      for name in ("left", "right", "across"):
        if name in direction:
          return name
      return "none"
  return "none"


def _hand_lines(lines: Sequence[str]) -> list[str]:
  """The non-blank lines between the direction and the tricks or points."""
  hand_lines: list[str] = []
  in_hands = False
  for line in lines:
    if "Pass Direction" in line:
      in_hands = True
      continue
    if "Tricks:" in line or "Points:" in line:
      break
    if in_hands and line.strip():
      hand_lines.append(line)
  return hand_lines


def _read_hands(lines: Sequence[str]) -> dict[str, list[str]]:
  """Every compass point's hand, read from the text's twelve hand lines."""
  hands: dict[str, list[str]] = {seat: [] for seat in COMPASS}
  hand_lines = _hand_lines(lines)
  if len(hand_lines) < 12:
    return hands
  suits = ["s", "h", "d", "c"]
  # North and South: "        S J62", centred.
  for seat, first in (("N", 0), ("S", 8)):
    for i, suit in enumerate(suits):
      match = re.match(r"\s+[SHDC]\s+(.*)$", hand_lines[first + i])
      if match:
        for rank in parse_hand_string(match.group(1).strip()):
          hands[seat].append(f"{rank}{suit}")
  # West and East: "S 854           S KQ93".
  for i, suit in enumerate(suits):
    parts = re.split(r"\s{4,}", hand_lines[4 + i])
    west = re.match(r"[SHDC]\s*(.*)", parts[0].strip())
    if west:
      for rank in parse_hand_string(west.group(1)):
        hands["W"].append(f"{rank}{suit}")
    if len(parts) >= 2:
      east = re.match(r"[SHDC]\s*(.*)", parts[-1].strip())
      if east:
        for rank in parse_hand_string(east.group(1)):
          hands["E"].append(f"{rank}{suit}")
  return hands


def _trick_lines(lines: Sequence[str]) -> list[str]:
  """The non-blank lines of the text's Tricks: section, header first."""
  trick_lines: list[str] = []
  in_tricks = False
  for line in lines:
    if "Tricks:" in line:
      in_tricks = True
      continue
    if in_tricks and "Points:" in line:
      break
    if in_tricks and line.strip():
      trick_lines.append(line)
  return trick_lines


def _read_tricks(
    lines: Sequence[str], current_player: int
) -> tuple[list[Json], int, dict[str, Json] | None]:
  """The trick in play, the number finished and the last finished one."""
  current_trick: list[Json] = []
  completed = 0
  last_trick: dict[str, Json] | None = None
  trick_lines = _trick_lines(lines)
  if len(trick_lines) < 2:
    return current_trick, completed, last_trick
  for trick_line in trick_lines[1:]:  # The first line is the seat header.
    cards = trick_line.split()
    if len(cards) == _NUM_SEATS:
      completed += 1
      last_trick = completed_trick(trick_line, completed)
    elif cards:
      # The leader of the trick in play: the seat to act, less the cards
      # played to it.
      lead = (current_player - len(cards)) % 4 if current_player >= 0 else 0
      for i, card in enumerate(cards):
        player = (lead + i) % _NUM_SEATS
        current_trick.append({
            "card": action_label_to_card_code(card),
            "player": player,
            "position": COMPASS[player],
        })
  return current_trick, completed, last_trick


def _read_points(lines: Sequence[str]) -> dict[str, Json]:
  """Each compass point's points, from the "Points:" section alone.

  The "Passed Cards:" section has lines of the same shape ("S: 3S 7D QH"),
  so an unanchored match would read a passed card's rank as points.
  """
  points: dict[str, Json] = {seat: 0 for seat in COMPASS}
  in_points = False
  for line in lines:
    if line.strip() == "Points:":
      in_points = True
      continue
    match = re.fullmatch(r"([NESW]):\s*(-?\d+)", line.strip())
    if in_points and match:
      points[match.group(1)] = int(match.group(2))
  return points


def extract_hearts_state(state: pyspiel.State) -> dict[str, Json]:
  """The board's struct of `state`, every hand shown."""
  game_name = state.get_game().get_type().short_name
  if game_name != GAME:
    raise ValueError(f"Expected a hearts state, got {game_name}.")
  lines = str(state).strip().split("\n")
  pass_direction = _pass_direction(lines)
  current_player = state.current_player()
  current_trick, completed, last_trick = _read_tricks(lines, current_player)
  # The hand is still passing until the text shows the tricks or the cards
  # received, which appear only once every seat has passed.
  has_tricks = any("Tricks:" in line for line in lines)
  has_received = any("Received Cards:" in line for line in lines)
  passing = pass_direction != "none" and not has_tricks and not has_received
  hands: dict[str, Json] = {
      seat: list(cards) for seat, cards in _read_hands(lines).items()
  }
  return {
      "hands": hands,
      "currentTrick": current_trick,
      "completedTricks": completed,
      "lastTrick": last_trick,
      "points": _read_points(lines),
      "passDirection": pass_direction,
      "phase": "passing" if passing else "playing",
      "currentPlayer": current_player,
  }


def mask_hands(
    struct: Mapping[str, Json], visible: Sequence[str]
) -> dict[str, Json]:
  """A copy of `struct` whose hands keep only the `visible` compass points.

  Every other hand is one None per card. Fails loud on a struct without the
  four-hand shape, which would otherwise be served whole.
  """
  hands = struct.get("hands")
  if not isinstance(hands, dict):
    raise ValueError(f"struct has no hands dict to mask: {sorted(struct)!r}")
  masked: dict[str, Json] = {}
  for compass, cards in hands.items():
    if compass not in COMPASS or not isinstance(cards, list):
      raise ValueError(f"unexpected hands entry {compass!r}: {cards!r}")
    masked[compass] = list(cards) if compass in visible else [None] * len(cards)
  out = dict(struct)
  out["hands"] = masked
  return out


def viewer_struct(
    struct: Mapping[str, Json], viewpoint: adapter.Viewpoint
) -> dict[str, Json]:
  """The struct `viewpoint` may see: its own hand, every other as backs.

  The trick in play, the points, the pass direction and the phase are on the
  table for everyone, and so are the sizes of the hidden hands.
  """
  visible = [
      c for seat, c in enumerate(COMPASS) if viewpoint.sees_private(seat)
  ]
  return mask_hands(struct, visible)


class HeartsAdapter(adapter.Adapter):
  """Presents a hearts position to one viewer as the board's struct."""

  renderer = "hearts"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, Json]:
    """The struct, holding only what the viewer may see."""
    return {"struct": viewer_struct(extract_hearts_state(state), viewpoint)}

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: adapter.Viewpoint,
  ) -> Mapping[int, Json]:
    """Each pass or play with the card it moves, as `cardCode`."""
    del state, viewpoint
    return {
        action: {"cardCode": action_to_card_code(action)} for action in actions
    }

  def public_action_label(
      self, state: pyspiel.State, action: int
  ) -> str | None:
    """A play's own label; a pass without its card; None for chance.

    A card played to a trick is face up for the table. A passed card is seen
    only by the passer (the receiver learns it from its own hand), so the
    others read that a pass was made. The deal is hidden.
    """
    player = state.current_player()
    if player == pyspiel.PlayerId.CHANCE:
      return None
    history = state.history()
    index = len(history)
    if (
        history
        and history[0] != _NO_PASS
        and _FIRST_PASS_INDEX <= index < _FIRST_PASS_INDEX + _NUM_PASS_ACTIONS
    ):
      return PASS_HIDDEN_LABEL
    return str(state.action_to_string(player, action))


ADAPTER = HeartsAdapter()
