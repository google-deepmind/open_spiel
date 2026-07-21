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

# Lint as python3
"""Belote implemented in Python.

Classic (non-contract) French Belote for 4 players in 2 fixed partnerships
(players 0 & 2 vs players 1 & 3). Trump is chosen via the "prise" procedure:
5 cards are dealt to each player, the next card of the stock is turned face
up, and players in turn may take it (round 1) or, if everyone passes, choose
one of the three other suits (round 2). If everyone passes twice, the deal is
redealt with the next player as dealer. Card play follows standard suit- and
trump-following obligations, and scoring uses the standard 162-point deck
(152 card points + 10 for the last trick), with an all-or-nothing rule: the
declaring team keeps its points only if it scores strictly more than 81;
otherwise the defending team collects all 162 points.

The "belote/rebelote" bonus (20 extra points awarded to whichever team has a
single player holding both the King and Queen of the trump suit) is optional
and controlled via the "use_belote_rebelote" game parameter (off by default).
When enabled, the bonus is credited to the holder's team regardless of
whether the declaring team makes its contract.
"""

import bisect

import numpy as np

import pyspiel

_NUM_PLAYERS = 4
_NUM_SUITS = 4
_NUM_RANKS = 8
_NUM_CARDS = _NUM_SUITS * _NUM_RANKS
_MAX_SCORE = 162
_BELOTE_REBELOTE_BONUS = 20

_SUIT_NAMES = ["C", "D", "H", "S"]
_RANK_NAMES = ["7", "8", "9", "10", "J", "Q", "K", "A"]

# Card strength, low to high, when the card's suit is NOT trump.
_NONTRUMP_ORDER = ["7", "8", "9", "J", "Q", "K", "10", "A"]
# Card strength, low to high, when the card's suit IS trump.
_TRUMP_ORDER = ["7", "8", "Q", "K", "10", "A", "9", "J"]

_NONTRUMP_POINTS = {
    "7": 0,
    "8": 0,
    "9": 0,
    "J": 2,
    "Q": 3,
    "K": 4,
    "10": 10,
    "A": 11,
}
_TRUMP_POINTS = {
    "7": 0,
    "8": 0,
    "9": 14,
    "J": 20,
    "Q": 3,
    "K": 4,
    "10": 10,
    "A": 11,
}
_NONTRUMP_STRENGTH_BY_RANK = [
    _NONTRUMP_ORDER.index(name) for name in _RANK_NAMES
]
_TRUMP_STRENGTH_BY_RANK = [_TRUMP_ORDER.index(name) for name in _RANK_NAMES]
_NONTRUMP_POINTS_BY_RANK = [_NONTRUMP_POINTS[name] for name in _RANK_NAMES]
_TRUMP_POINTS_BY_RANK = [_TRUMP_POINTS[name] for name in _RANK_NAMES]

# Card are defined as 0, 1, ..., 31
PASS_ACTION = _NUM_CARDS  # 32
TAKE_ACTION = _NUM_CARDS + 1  # 33
CHOOSE_SUIT_ACTION_BASE = _NUM_CARDS + 2  # + suit index (0..3) : 34, 35, 36, 37

_GAME_TYPE = pyspiel.GameType(
    short_name="python_belote",
    long_name="Python Belote",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "dealer": 0,
        "use_belote_rebelote": False,
    },
)
_GAME_INFO = pyspiel.GameInfo(
    # Card plays (0..31) + pass + take + 4 choose-suit actions.
    num_distinct_actions=_NUM_CARDS + 2 + _NUM_SUITS,
    max_chance_outcomes=_NUM_CARDS,
    num_players=_NUM_PLAYERS,
    # Loose bounds that also cover the optional belote/rebelote bonus.
    min_utility=-float(_MAX_SCORE + _BELOTE_REBELOTE_BONUS),
    max_utility=float(_MAX_SCORE + _BELOTE_REBELOTE_BONUS),
    utility_sum=0.0,
    # Dealing (~32 draws) + bidding (up to 8 calls) + card play (32 plays).
    max_game_length=_NUM_CARDS + 8 + _NUM_CARDS,
)


def card_suit(card) -> int:
  """Returns the suit index (0..3) of `card`."""
  return card // _NUM_RANKS


def card_rank_name(card) -> str:
  """Returns the rank name (e.g. "A") of `card`."""
  return _RANK_NAMES[card % _NUM_RANKS]


def card_string(card) -> str:
  """Returns the human-readable string (e.g. "AS") for `card`."""
  return card_rank_name(card) + _SUIT_NAMES[card_suit(card)]


def card_points(card, trump_suit) -> int:
  """Returns the point value of `card` given the current `trump_suit`."""
  rank = card % _NUM_RANKS
  return (_TRUMP_POINTS_BY_RANK[rank]
          if card_suit(card) == trump_suit else _NONTRUMP_POINTS_BY_RANK[rank])


def card_strength(card, trump_suit) -> int:
  """Returns the relative ranking strength of `card` given `trump_suit`."""
  rank = card % _NUM_RANKS
  return (_TRUMP_STRENGTH_BY_RANK[rank] if card_suit(card) == trump_suit else
          _NONTRUMP_STRENGTH_BY_RANK[rank])


def team_of(player) -> int:
  """Returns the team id (0 or 1) that `player` belongs to."""
  return player % 2


def partner_of(player) -> int:
  """Returns the id of `player`'s partner."""
  return (player + 2) % _NUM_PLAYERS


def _order_from(start) -> list[int]:
  return [(start + i) % _NUM_PLAYERS for i in range(_NUM_PLAYERS)]


def _initial_deal_schedule(dealer) -> list[int | None]:
  """Deal order for the first 5 cards/player (3 then 2) plus the turned card."""
  order = _order_from((dealer + 1) % _NUM_PLAYERS)
  schedule = []
  for player in order:
    schedule.extend([player] * 3)
  for player in order:
    schedule.extend([player] * 2)
  schedule.append(None)  # The next stock card is turned face up.
  return schedule


class BeloteGame(pyspiel.Game):
  """A Python version of Belote."""

  def __init__(self, params=None) -> None:
    super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
    self.dealer = self.get_parameters().get("dealer", 0)
    self.use_belote_rebelote = self.get_parameters().get(
        "use_belote_rebelote", False)

  def new_initial_state(self) -> "BeloteState":
    """Returns a state corresponding to the start of a game."""
    return BeloteState(self)

  def make_py_observer(self,
                       iig_obs_type=None,
                       params=None) -> "BeloteObserver":
    """Returns an object used for observing game state."""
    return BeloteObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params,
    )


class BeloteState(pyspiel.State):
  """A python version of the Belote state."""

  # Attribute count reflects the game's own state (deal/bid/play phases,
  # trick history, running scores, ...); splitting it up would not make
  # the logic clearer.
  # pylint: disable=too-many-instance-attributes

  def __init__(self, game) -> None:
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._dealer = game.dealer
    self.hands = [[] for _ in range(_NUM_PLAYERS)]
    self._deck = list(range(_NUM_CARDS))
    self._turned_card = None

    self._phase = "deal"
    self._deal_schedule = _initial_deal_schedule(self._dealer)
    self._deal_index = 0
    self._after_deal_phase = "bid1"

    self._bid_turn_order = _order_from((self._dealer + 1) % _NUM_PLAYERS)
    self._bid_pointer = 0

    self._taker = -1
    self._trump_suit = -1
    self._declarer_team = -1
    self._use_belote_rebelote = game.use_belote_rebelote
    self._belote_team = -1

    self._trick = []
    self._trick_leader = -1
    self._current_player_play = -1
    self._tricks_played = 0
    self._played_cards = []
    self._trick_history = []
    self._team_points = [0, 0]
    self._returns = [0.0] * _NUM_PLAYERS

  def current_player(self) -> int:
    """Returns id of the current player to act."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    if self._phase == "deal":
      return pyspiel.PlayerId.CHANCE
    if self._phase in ("bid1", "bid2"):
      return self._bid_turn_order[self._bid_pointer]
    return self._current_player_play

  def _legal_actions(self, player) -> list[int]:
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    if self._phase == "bid1":
      return [PASS_ACTION, TAKE_ACTION]
    if self._phase == "bid2":
      turned_suit = card_suit(self._turned_card)
      return [PASS_ACTION] + [
          CHOOSE_SUIT_ACTION_BASE + s
          for s in range(_NUM_SUITS)
          if s != turned_suit
      ]
    if self._phase == "play":
      return self._legal_card_plays(player)
    return []

  # pylint: disable-next=too-many-return-statements
  def _legal_card_plays(self, player) -> list[int]:
    """Cards `player` may legally play, given suit/trump-following rules."""
    hand = self.hands[player]
    if not self._trick:  # No cards played for the trick, any card may be led.
      return sorted(hand)

    led_suit = card_suit(self._trick[0][1])
    trump = self._trump_suit
    same_suit_cards = [c for c in hand if card_suit(c) == led_suit]
    current_winner = self._trick_winner(self._trick)
    partner_winning = partner_of(player) == current_winner

    if same_suit_cards:
      if led_suit != trump:  # If the led suit is not trump, must follow suit.
        return sorted(same_suit_cards)
      # Trump was led: must play higher than the best trump so far if
      # possible, even if the partner currently holds the trick.
      highest = max(
          card_strength(c, trump)
          for _, c in self._trick
          if card_suit(c) == trump)
      higher = [c for c in same_suit_cards if card_strength(c, trump) > highest]
      return sorted(higher) if higher else sorted(same_suit_cards)

    # No cards of the led suit: may play trump if possible.
    trump_cards = [c for c in hand if card_suit(c) == trump]
    if trump_cards and led_suit != trump:
      if (partner_winning
         ):  # If the partner is currently winning, any card may be played.
        return sorted(hand)

      trumps_played = [c for _, c in self._trick if card_suit(c) == trump]
      if not trumps_played:  # No trumps have been played yet, play any trump.
        return sorted(trump_cards)

      # Need to play a higher trump if possible.
      highest = max(card_strength(c, trump) for c in trumps_played)
      higher = [c for c in trump_cards if card_strength(c, trump) > highest]
      return sorted(higher) if higher else sorted(trump_cards)

    # No cards of the led suit and no trumps: may play any card.
    return sorted(hand)

  def _is_better(self, card, other, led_suit, trump) -> bool:
    """Whether `card` beats `other` within the same trick."""
    card_trump = card_suit(card) == trump
    other_trump = card_suit(other) == trump

    # Exactly one card is trump, so `card` wins iff it is the trump card.
    if card_trump != other_trump:
      return card_trump

    # Both cards are trump, compare by trump ranking order
    if card_trump and other_trump:
      return card_strength(card, trump) > card_strength(other, trump)

    card_led = card_suit(card) == led_suit
    other_led = card_suit(other) == led_suit

    # Exactly one card follows the led suit, so `card` wins iff it follows it.
    if card_led != other_led:
      return card_led

    # Both cards follow led suit, compare by non-trump ranking order.
    if card_led and other_led:
      return card_strength(card, trump) > card_strength(other, trump)

    # Neither card is trump nor led suit: card cannot beat other.
    return False

  def _trick_winner(self, trick) -> int:
    """Returns the player currently winning `trick` (partial or complete)."""
    led_suit = card_suit(trick[0][1])
    best_player, best_card = trick[0]
    for player, card in trick[1:]:
      if self._is_better(card, best_card, led_suit, self._trump_suit):
        best_player, best_card = player, card

    return best_player

  def chance_outcomes(self) -> list[tuple[int, float]]:
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    probability = 1.0 / len(self._deck)
    return [(card, probability) for card in self._deck]

  def _enter_play_phase(self) -> None:
    self._trick_leader = (self._dealer + 1) % _NUM_PLAYERS
    self._current_player_play = self._trick_leader
    self._trick = []
    self._tricks_played = 0
    self._team_points = [0, 0]
    if self._use_belote_rebelote:
      self._belote_team = self._find_belote_team()

  def _find_belote_team(self) -> int:
    """Returns the team id of the player holding K+Q of trump, or -1."""
    trump_king = self._trump_suit * _NUM_RANKS + _RANK_NAMES.index("K")
    trump_queen = self._trump_suit * _NUM_RANKS + _RANK_NAMES.index("Q")
    for player, hand in enumerate(self.hands):
      if trump_king in hand and trump_queen in hand:
        return team_of(player)
    return -1

  def _apply_deal_action(self, card) -> None:
    self._deck.remove(card)
    destination = self._deal_schedule[self._deal_index]
    if destination is None:
      self._turned_card = card
    else:
      self.hands[destination].append(card)
    self._deal_index += 1
    if self._deal_index == len(self._deal_schedule):
      self._phase = self._after_deal_phase
      self._deal_schedule = []
      self._deal_index = 0
      if self._phase == "play":
        self._enter_play_phase()

  def _start_completion_deal(self, schedule, next_phase) -> None:
    self._deal_schedule = schedule
    self._deal_index = 0
    self._after_deal_phase = next_phase
    self._phase = "deal"

  def _apply_bid1_action(self, action, player) -> None:
    if action == TAKE_ACTION:
      self._taker = player
      self._trump_suit = card_suit(self._turned_card)
      self._declarer_team = team_of(player)
      self.hands[player].append(self._turned_card)

      order = _order_from((self._dealer + 1) % _NUM_PLAYERS)
      target_counts = {p: (2 if p == player else 3) for p in order}
      dealt_counts = {p: 0 for p in order}
      schedule = []
      while any(dealt_counts[p] < target_counts[p] for p in order):
        for p in order:
          if dealt_counts[p] < target_counts[p]:
            schedule.append(p)
            dealt_counts[p] += 1
      self._start_completion_deal(schedule, "play")
    else:
      self._bid_pointer += 1
      if self._bid_pointer == _NUM_PLAYERS:
        self._phase = "bid2"
        self._bid_pointer = 0

  def _apply_bid2_action(self, action, player) -> None:
    if action == PASS_ACTION:
      self._bid_pointer += 1
      if self._bid_pointer == _NUM_PLAYERS:
        # Everyone passed twice: reshuffle and redeal, dealer rotates.
        self._dealer = (self._dealer + 1) % _NUM_PLAYERS
        self.hands = [[] for _ in range(_NUM_PLAYERS)]
        self._turned_card = None
        self._deck = list(range(_NUM_CARDS))
        self._bid_turn_order = _order_from((self._dealer + 1) % _NUM_PLAYERS)
        self._bid_pointer = 0
        self._deal_schedule = _initial_deal_schedule(self._dealer)
        self._deal_index = 0
        self._after_deal_phase = "bid1"
        self._phase = "deal"
    else:
      suit = action - CHOOSE_SUIT_ACTION_BASE
      self._taker = player
      self._trump_suit = suit
      self._declarer_team = team_of(player)
      # insort (not append) to keep `self._deck` sorted, which lets
      # chance_outcomes() skip an explicit sort on every deal.
      bisect.insort(self._deck, self._turned_card)

      order = _order_from((self._dealer + 1) % _NUM_PLAYERS)
      schedule = order * 3
      self._start_completion_deal(schedule, "play")

  def _finalize_scores(self):
    declarer_team = self._declarer_team
    other_team = 1 - declarer_team
    declarer_points = self._team_points[declarer_team]
    other_points = self._team_points[other_team]
    if declarer_points > _MAX_SCORE // 2:
      final_declarer, final_other = declarer_points, other_points
    else:
      final_declarer, final_other = 0, _MAX_SCORE
    if self._belote_team == declarer_team:
      final_declarer += _BELOTE_REBELOTE_BONUS
    elif self._belote_team == other_team:
      final_other += _BELOTE_REBELOTE_BONUS
    diff = float(final_declarer - final_other)
    self._returns = [
        diff if team_of(p) == declarer_team else -diff
        for p in range(_NUM_PLAYERS)
    ]

  def _apply_play_action(self, card, player) -> None:
    self.hands[player].remove(card)
    self._trick.append((player, card))
    self._played_cards.append(card)
    if len(self._trick) < _NUM_PLAYERS:
      self._current_player_play = (player + 1) % _NUM_PLAYERS
      return

    winner = self._trick_winner(self._trick)
    points = sum(card_points(c, self._trump_suit) for _, c in self._trick)
    self._tricks_played += 1
    if self._tricks_played == _NUM_CARDS // _NUM_PLAYERS:
      points += 10
    self._team_points[team_of(winner)] += points
    self._trick_history.append([c for _, c in self._trick])

    self._trick = []
    self._trick_leader = winner
    self._current_player_play = winner
    if self._tricks_played == _NUM_CARDS // _NUM_PLAYERS:
      self._finalize_scores()
      self._phase = "done"

  def _apply_action(self, action) -> None:
    """Applies an action and updates the state."""
    if self._phase == "deal":
      self._apply_deal_action(action)
    elif self._phase == "bid1":
      self._apply_bid1_action(action, self._bid_turn_order[self._bid_pointer])
    elif self._phase == "bid2":
      self._apply_bid2_action(action, self._bid_turn_order[self._bid_pointer])
    elif self._phase == "play":
      self._apply_play_action(action, self._current_player_play)

  def _action_to_string(self, player, action) -> str:
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal: {card_string(action)}"
    if action == PASS_ACTION:
      return "Pass"
    if action == TAKE_ACTION:
      return "Take"
    if CHOOSE_SUIT_ACTION_BASE <= action < CHOOSE_SUIT_ACTION_BASE + _NUM_SUITS:
      return f"Choose trump: {_SUIT_NAMES[action - CHOOSE_SUIT_ACTION_BASE]}"
    return f"Play: {card_string(action)}"

  def is_terminal(self) -> bool:
    """Returns True if the game is over."""
    return self._phase == "done"

  def returns(self) -> list[float]:
    """Total reward for each player over the course of the game so far."""
    return list(self._returns)

  def __str__(self) -> str:
    """String for debug purposes. No particular semantics are required."""
    lines = [
        f"Dealer: {self._dealer}",
        f"Phase: {self._phase}",
        f"Hands: {[sorted(h) for h in self.hands]}",
    ]
    if self._turned_card is not None:
      lines.append(f"Turned card: {card_string(self._turned_card)}")
    if self._trump_suit >= 0:
      lines.append(
          f"Trump: {_SUIT_NAMES[self._trump_suit]}, Taker: {self._taker}")
    if self._phase in ("play", "done"):
      lines.append(f"Trick: {[(p, card_string(c)) for p, c in self._trick]}")
      lines.append(f"Team points: {self._team_points}")
      if self._belote_team >= 0:
        lines.append(f"Belote/rebelote team: {self._belote_team}")
    return "\n".join(lines)


# BeloteObserver reads BeloteState's private fields directly, which is the
# established pattern for observers in this package (see e.g. ant_foraging.py).
# pylint: disable=protected-access
class BeloteObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params=None) -> None:
    """Initializes an empty observation tensor."""
    del params
    self.iig_obs_type = iig_obs_type

    pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("hand", _NUM_CARDS, (_NUM_CARDS,)))
    if iig_obs_type.public_info:
      pieces.append(("dealer", _NUM_PLAYERS, (_NUM_PLAYERS,)))
      pieces.append(("turned_card", _NUM_CARDS, (_NUM_CARDS,)))
      pieces.append(("trump_suit", _NUM_SUITS + 1, (_NUM_SUITS + 1,)))
      pieces.append(("declarer", _NUM_PLAYERS, (_NUM_PLAYERS,)))
      pieces.append(("current_trick", _NUM_CARDS, (_NUM_CARDS,)))
      pieces.append(("cards_played", _NUM_CARDS, (_NUM_CARDS,)))
      pieces.append(("team_points", 2, (2,)))
      if iig_obs_type.perfect_recall:
        num_tricks = _NUM_CARDS // _NUM_PLAYERS
        pieces.append(("trick_history", num_tricks * _NUM_CARDS, (num_tricks,
                                                                  _NUM_CARDS)))

    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player) -> None:  # pylint: disable=too-many-branches
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if "player" in self.dict:
      self.dict["player"][player] = 1
    if "hand" in self.dict:
      for card in state.hands[player]:
        self.dict["hand"][card] = 1
    if "dealer" in self.dict:
      self.dict["dealer"][state._dealer] = 1
    if "turned_card" in self.dict and state._turned_card is not None:
      self.dict["turned_card"][state._turned_card] = 1
    if "trump_suit" in self.dict:
      index = state._trump_suit if state._trump_suit >= 0 else _NUM_SUITS
      self.dict["trump_suit"][index] = 1
    if "declarer" in self.dict and state._taker >= 0:
      self.dict["declarer"][state._taker] = 1
    if "current_trick" in self.dict:
      for _, card in state._trick:
        self.dict["current_trick"][card] = 1
    if "cards_played" in self.dict:
      for card in state._played_cards:
        self.dict["cards_played"][card] = 1
    if "team_points" in self.dict:
      self.dict["team_points"][0] = state._team_points[0] / float(_MAX_SCORE)
      self.dict["team_points"][1] = state._team_points[1] / float(_MAX_SCORE)
    if "trick_history" in self.dict:
      for trick_idx, cards in enumerate(state._trick_history):
        for card in cards:
          self.dict["trick_history"][trick_idx][card] = 1

  def string_from(self, state, player) -> str:
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "hand" in self.dict:
      pieces.append(
          f"hand:{[card_string(c) for c in sorted(state.hands[player])]}")
    if "dealer" in self.dict:
      pieces.append(f"dealer:{state._dealer}")
    if "turned_card" in self.dict and state._turned_card is not None:
      pieces.append(f"turned:{card_string(state._turned_card)}")
    if "trump_suit" in self.dict and state._trump_suit >= 0:
      pieces.append(f"trump:{_SUIT_NAMES[state._trump_suit]}")
    if "declarer" in self.dict and state._taker >= 0:
      pieces.append(f"declarer:{state._taker}")
    if "current_trick" in self.dict:
      pieces.append(f"trick:{[card_string(c) for _, c in state._trick]}")
    if "team_points" in self.dict:
      pieces.append(f"points:{state._team_points}")
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, BeloteGame)
