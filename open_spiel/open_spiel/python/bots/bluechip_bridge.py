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
"""Wraps third-party bridge bots to make them usable in OpenSpiel.

This code enables OpenSpiel interoperation for bots which implement the BlueChip
bridge protocol. This is widely used, e.g. in the World computer bridge
championships. For a rough outline of the protocol, see:
http://www.bluechipbridge.co.uk/protocol.htm

No formal specification is available. This implementation has been verified
to work correctly with WBridge5.

This bot controls a single player in the full game of bridge, including both the
bidding and play phase. It chooses its actions by invoking an external bot which
plays the full game of bridge. This means that each time the bot is asked for an
action, it sends up to three actions (one for each other player) to the external
bridge bot, and obtains an action in return.
"""

import re
import pyspiel

# Example session:
#
# Recv: Connecting "WBridge5" as ANYPL using protocol version 18
# Send: WEST ("WBridge5") seated
# Recv: WEST ready for teams
# Send: Teams: N/S "silent" E/W "bidders"
# Recv: WEST ready to start
# Send: Start of board
# Recv: WEST ready for deal
# Send: Board number 8. Dealer WEST. Neither vulnerable.
# Recv: WEST ready for cards
# Send: WEST's cards: S A T 9 5. H K 6 5. D Q J 8 7 6. C 7.
# Recv: WEST PASSES
# Recv: WEST ready for  NORTH's bid
# Send: EAST PASSES
# Recv: WEST ready for EAST's bid
# Send: EAST bids 1C
# Recv: WEST ready for  SOUTH's bid

# The game we support
GAME_STR = "bridge(use_double_dummy_result=False)"

# Template regular expressions for messages we receive
_CONNECT = 'Connecting "(?P<client_name>.*)" as ANYPL using protocol version 18'
_PLAYER_ACTION = ("(?P<seat>NORTH|SOUTH|EAST|WEST) "
                  "((?P<pass>PASSES)|(?P<dbl>DOUBLES)|(?P<rdbl>REDOUBLES)|bids "
                  "(?P<bid>[^ ]*)|(plays (?P<play>[23456789tjqka][cdhs])))"
                  "(?P<alert> Alert.)?")
_READY_FOR_OTHER = ("{seat} ready for "
                    "(((?P<other>[^']*)'s ((bid)|(card to trick \\d+)))"
                    "|(?P<dummy>dummy))")

# Templates for fixed messages we receive
_READY_FOR_TEAMS = "{seat} ready for teams"
_READY_TO_START = "{seat} ready to start"
_READY_FOR_DEAL = "{seat} ready for deal"
_READY_FOR_CARDS = "{seat} ready for cards"
_READY_FOR_BID = "{seat} ready for {other}'s bid"

# Templates for messages we send
_SEATED = '{seat} ("{client_name}") seated'
_TEAMS = 'Teams: N/S "north-south" E/W "east-west"'
_START_BOARD = "start of board"
_DEAL = "Board number {board}. Dealer NORTH. Neither vulnerable."
_CARDS = "{seat}'s cards: {hand}"
_OTHER_PLAYER_ACTION = "{player} {action}"
_PLAYER_TO_LEAD = "{seat} to lead"
_DUMMY_CARDS = "Dummy's cards: {}"

# BlueChip bridge protocol message constants
_SEATS = ["NORTH", "EAST", "SOUTH", "WEST"]
_TRUMP_SUIT = ["C", "D", "H", "S", "NT"]
_NUMBER_TRUMP_SUITS = len(_TRUMP_SUIT)
_SUIT = _TRUMP_SUIT[:4]
_NUMBER_SUITS = len(_SUIT)
_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_LSUIT = [x.lower() for x in _SUIT]
_LRANKS = [x.lower() for x in _RANKS]

# OpenSpiel action ids
_ACTION_PASS = 52
_ACTION_DBL = 53
_ACTION_RDBL = 54
_ACTION_BID = 55  # First bid, i.e. 1C


def _bid_to_action(action_str):
  """Returns an OpenSpiel action id (an integer) from a BlueChip bid string."""
  level = int(action_str[0])
  trumps = _TRUMP_SUIT.index(action_str[1:])
  return _ACTION_BID + (level - 1) * _NUMBER_TRUMP_SUITS + trumps


def _play_to_action(action_str):
  """Returns an OpenSpiel action id (an integer) from a BlueChip card string."""
  rank = _LRANKS.index(action_str[0])
  suit = _LSUIT.index(action_str[1])
  return rank * _NUMBER_SUITS + suit


def _action_to_string(action):
  """Converts OpenSpiel action id (an integer) to a BlueChip action string.

  Args:
    action: an integer action id corresponding to a bid.

  Returns:
    A string in BlueChip format, e.g. 'PASSES' or 'bids 1H', or 'plays ck'.
  """
  if action == _ACTION_PASS:
    return "PASSES"
  elif action == _ACTION_DBL:
    return "DOUBLES"
  elif action == _ACTION_RDBL:
    return "REDOUBLES"
  elif action >= _ACTION_BID:
    level = str((action - _ACTION_BID) // _NUMBER_TRUMP_SUITS + 1)
    trumps = _TRUMP_SUIT[(action - _ACTION_BID) % _NUMBER_TRUMP_SUITS]
    return "bids " + level + trumps
  else:
    rank = action // _NUMBER_SUITS
    suit = action % _NUMBER_SUITS
    return "plays " + _LRANKS[rank] + _LSUIT[suit]


def _expect_regex(controller, regex):
  """Reads a line from the controller, parses it using the regular expression."""
  line = controller.read_line()
  match = re.match(regex, line)
  if not match:
    raise ValueError("Received '{}' which does not match regex '{}'".format(
        line, regex))
  return match.groupdict()


def _expect(controller, expected):
  """Reads a line from the controller, checks it matches expected line exactly."""
  line = controller.read_line()
  if expected != line:
    raise ValueError("Received '{}' but expected '{}'".format(line, expected))


def _hand_string(cards):
  """Returns the hand of the to-play player in the state in BlueChip format."""
  if len(cards) != 13:
    raise ValueError("Must have 13 cards")
  suits = [[] for _ in range(4)]
  for card in reversed(sorted(cards)):
    suit = card % 4
    rank = card // 4
    suits[suit].append(_RANKS[rank])
  for i in range(4):
    if suits[i]:
      suits[i] = _TRUMP_SUIT[i] + " " + " ".join(suits[i]) + "."
    else:
      suits[i] = _TRUMP_SUIT[i] + " -."
  return " ".join(suits)


def _connect(controller, seat):
  """Performs the initial handshake with a BlueChip bot."""
  client_name = _expect_regex(controller, _CONNECT)["client_name"]
  controller.send_line(_SEATED.format(seat=seat, client_name=client_name))
  _expect(controller, _READY_FOR_TEAMS.format(seat=seat))
  controller.send_line(_TEAMS)
  _expect(controller, _READY_TO_START.format(seat=seat))


def _new_deal(controller, seat, hand, board):
  """Informs a BlueChip bots that there is a new deal."""
  controller.send_line(_START_BOARD)
  _expect(controller, _READY_FOR_DEAL.format(seat=seat))
  controller.send_line(_DEAL.format(board=board))
  _expect(controller, _READY_FOR_CARDS.format(seat=seat))
  controller.send_line(_CARDS.format(seat=seat, hand=hand))


class BlueChipBridgeBot(pyspiel.Bot):
  """An OpenSpiel bot, wrapping a BlueChip bridge bot implementation."""

  def __init__(self, game, player_id, controller_factory):
    """Initializes an OpenSpiel `Bot` wrapping a BlueChip-compatible bot.

    Args:
      game: The OpenSpiel game object, should be an instance of
        `bridge(use_double_dummy_result=false)`.
      player_id: The id of the player the bot will act as, 0 = North (dealer), 1
        = East, 2 = South, 3 = West.
      controller_factory: Callable that returns new BlueChip controllers which
        must support methods `read_line` and `send_line`, and `terminate`.
    """
    pyspiel.Bot.__init__(self)
    if str(game) != GAME_STR:
      raise ValueError(f"BlueChipBridgeBot invoked with {game}")
    self._game = game
    self._player_id = player_id
    self._controller_factory = controller_factory
    self._seat = _SEATS[player_id]
    self._num_actions = 52
    self.dummy = None
    self.is_play_phase = False
    self.cards_played = 0
    self._board = 0
    self._state = self._game.new_initial_state()
    self._controller = None

  def player_id(self):
    return self._player_id

  def restart(self):
    """Indicates that we are starting a new episode."""
    # If we already have a fresh state, there is nothing to do.
    if not self._state.history():
      return
    self._num_actions = 52
    self.dummy = None
    self.is_play_phase = False
    self.cards_played = 0
    # We didn't see the end of the episode, so the external bot will still
    # be expecting it. If we can autoplay other people's actions to the end
    # (e.g. everyone passes or players play their last card), then do that.
    if not self._state.is_terminal():
      state = self._state.clone()
      while (not state.is_terminal()
             and state.current_player() != self._player_id):
        legal_actions = state.legal_actions()
        if _ACTION_PASS in legal_actions:
          state.apply(_ACTION_PASS)
        elif len(legal_actions) == 1:
          state.apply_action(legal_actions[0])
      if state.is_terminal():
        self.inform_state(state)
    # Otherwise, we will have to restart the external bot, because
    # the protocol makes no provision for this case.
    if not self._state.is_terminal():
      self._controller.terminate()
      self._controller = None
    self._state = self._game.new_initial_state()

  def _update_for_state(self):
    """Called for all non-chance nodes, whether or not we have to act."""
    # Get the actions in the game so far.
    actions = self._state.history()
    self.is_play_phase = (not self._state.is_terminal() and
                          max(self._state.legal_actions()) < 52)
    self.cards_played = sum(1 if a < 52 else 0 for a in actions) - 52

    # If this is the first time we've seen the deal, send our hand.
    if len(actions) == 52:
      self._board += 1
      _new_deal(self._controller, self._seat,
                _hand_string(actions[self._player_id:52:4]), self._board)

    # Send actions since last `step` call.
    for other_player_action in actions[self._num_actions:]:
      other = _expect_regex(self._controller,
                            _READY_FOR_OTHER.format(seat=self._seat))
      other_player = other["other"]
      if other_player == "Dummy":
        other_player = _SEATS[self.dummy]
      self._controller.send_line(
          _OTHER_PLAYER_ACTION.format(
              player=other_player,
              action=_action_to_string(other_player_action)))
    self._num_actions = len(actions)

    # If the opening lead has just been made, give the dummy.
    if self.is_play_phase and self.cards_played == 1:
      self.dummy = self._state.current_player() ^ 2
      if self._player_id != self.dummy:
        other = _expect_regex(self._controller,
                              _READY_FOR_OTHER.format(seat=self._seat))
        dummy_cards = _hand_string(actions[self.dummy:52:4])
        self._controller.send_line(_DUMMY_CARDS.format(dummy_cards))

    # If the episode is terminal, send (fake) timing info.
    if self._state.is_terminal():
      self._controller.send_line(
          "Timing - N/S : this board  [1:15],  total  [0:11:23].  "
          "E/W : this board  [1:18],  total  [0:10:23]"
      )
      self.dummy = None
      self.is_play_phase = False
      self.cards_played = 0

  def inform_action(self, state, player, action):
    del player, action
    self.inform_state(state)

  def inform_state(self, state):
    # Connect if we need to.
    if self._controller is None:
      self._controller = self._controller_factory()
      _connect(self._controller, self._seat)

    full_history = state.history()
    known_history = self._state.history()
    if full_history[:len(known_history)] != known_history:
      raise ValueError(
          "Supplied state is inconsistent with bot's internal state\n"
          f"Supplied state:\n{state}\n"
          f"Internal state:\n{self._state}\n")
    for action in full_history[len(known_history):]:
      self._state.apply_action(action)
      if not self._state.is_chance_node():
        self._update_for_state()

  def step(self, state):
    """Returns an action for the given state."""
    # Bring the external bot up-to-date.
    self.inform_state(state)

    # If we're on a new trick, tell the bot it is its turn.
    if self.is_play_phase and self.cards_played % 4 == 0:
      self._controller.send_line(_PLAYER_TO_LEAD.format(seat=self._seat))

    # Get our action from the bot.
    our_action = _expect_regex(self._controller, _PLAYER_ACTION)
    self._num_actions += 1
    if our_action["pass"]:
      return _ACTION_PASS
    elif our_action["dbl"]:
      return _ACTION_DBL
    elif our_action["rdbl"]:
      return _ACTION_RDBL
    elif our_action["bid"]:
      return _bid_to_action(our_action["bid"])
    elif our_action["play"]:
      return _play_to_action(our_action["play"])

  def terminate(self):
    self._controller.terminate()
    self._controller = None

