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

"""Wraps third-party bridge bots to make them usable in OpenSpiel.

This code enables OpenSpiel interoperation for bots which implement the BlueChip
bridge protocol. This is widely used, e.g. in the World computer bridge
championships. For a rough outline of the protocol, see:
http://www.bluechipbridge.co.uk/protocol.htm

No formal specification is available. This implementation has been verified
to work correctly with WBridge5.

This bot controls a single player in the game of uncontested bridge bidding. It
chooses its actions by invoking an external bot which plays the full game of
bridge. This means that each time the bot is asked for an action, it sends up to
three actions (forced passes from both opponents, plus partner's most recent
action) to the external bridge bot, and obtains an action in return.

Since we are restricting ourselves to the uncontested bidding game, we have
no support for Doubling, Redoubling, or the play of the cards.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

# Template regular expressions for messages we receive
_CONNECT = 'Connecting "(?P<client_name>.*)" as ANYPL using protocol version 18'
_SELF_BID_OR_PASS = "{seat} ((?P<pass>PASSES)|bids (?P<bid>[^ ]*))( Alert.)?"

# Templates for fixed messages we receive
_READY_FOR_TEAMS = "{seat} ready for teams"
_READY_TO_START = "{seat} ready to start"
_READY_FOR_DEAL = "{seat} ready for deal"
_READY_FOR_CARDS = "{seat} ready for cards"
_READY_FOR_BID = "{seat} ready for {other}'s bid"

# Templates for messages we send
_SEATED = '{seat} ("{client_name}") seated'
_TEAMS = 'Teams: N/S "opponents" E/W "bidders"'
_START_BOARD = "start of board"
# The board number is arbitrary, but "8" is consistent with the dealer and
# vulnerability we want (in the standard numbering). See Law 2:
# http://web2.acbl.org/documentlibrary/play/Laws-of-Duplicate-Bridge.pdf
_DEAL = "Board number 8. Dealer WEST. Neither vulnerable."
_CARDS = "{seat}'s cards: {hand}"
_OTHER_PLAYER_PASS = "{player} PASSES"
_OTHER_PLAYER_BID = "{player} bids {bid}"

# BlueChip bridge protocol message constants
_SEATS = ["WEST", "EAST"]
_OPPONENTS = ["NORTH", "SOUTH"]
_TRUMP_SUIT = ["C", "D", "H", "S", "NT"]
_NUMBER_TRUMP_SUITS = len(_TRUMP_SUIT)
_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

# OpenSpiel constants
_PASS_ACTION = 0


def _string_to_action(call_str):
  """Converts a BlueChip bid string to an OpenSpiel action id (an integer).

  Args:
    call_str: string representing a bid in the BlueChip format, i.e. "[level]
      (as a digit) + [trump suit (S, H, D, C or NT)]", e.g. "1C".

  Returns:
    An integer action id - see `bridge_uncontested_bidding.cc`, functions
    `Denomination` and `Level`.
    0 is reserved for Pass, so bids are in order from 1 upwards: 1 = 1C,
    2 = 1D, etc.
  """
  level = int(call_str[0])
  trumps = _TRUMP_SUIT.index(call_str[1:])
  return (level - 1) * _NUMBER_TRUMP_SUITS + trumps + 1


def _action_to_string(action):
  """Converts OpenSpiel action id (an integer) to a BlueChip bid string.

  Args:
    action: an integer action id corresponding to a bid.

  Returns:
    A string in BlueChip format.

  Inverse of `_string_to_action`. See documentation there.
  """
  level = str((action - 1) // _NUMBER_TRUMP_SUITS + 1)
  trumps = _TRUMP_SUIT[(action - 1) % _NUMBER_TRUMP_SUITS]
  return level + trumps


def _expect_regex(client, regex):
  """Reads a line from the client, parses it using the regular expression."""
  line = client.read_line()
  match = re.match(regex, line)
  if not match:
    raise ValueError("Received '{}' which does not match regex '{}'".format(
        line, regex))
  return match.groupdict()


def _expect(client, expected):
  """Reads a line from the client, checks it matches expected line exactly."""
  line = client.read_line()
  if expected != line:
    raise ValueError("Received '{}' but expected '{}'".format(line, expected))


def _hand_string(state_vec):
  """Returns the hand of the to-play player in the state in BlueChip format."""
  # See UncontestedBiddingState::InformationStateTensor
  # The first 52 elements are whether or not we hold the given card (cards
  # ordered suit-by-suit, in ascending order of rank).
  suits = []
  for suit in reversed(range(4)):
    cards = []
    for rank in reversed(range(13)):
      if state_vec[rank * 4 + suit]:
        cards.append(_RANKS[rank])
    suits.append(_TRUMP_SUIT[suit] + " " + (" ".join(cards) if cards else "-") +
                 ".")
  return " ".join(suits)


def _actions(state_vec):
  """Returns the player actions that have been taken in the game so far."""
  # See UncontestedBiddingState::InformationStateTensor
  # The first 52 elements are the cards held, then two elements for each
  # possible action, specifying which of the two players has taken it (if
  # either player has). Then two elements specifying which player we are.
  actions = state_vec[52:-2]
  return [index // 2 for index, value in enumerate(actions) if value]


def _connect(client, seat, state_vec):
  """Performs the initial handshake with a BlueChip bot."""
  client.start()
  client_name = _expect_regex(client, _CONNECT)["client_name"]
  client.send_line(_SEATED.format(seat=seat, client_name=client_name))
  _expect(client, _READY_FOR_TEAMS.format(seat=seat))
  client.send_line(_TEAMS)
  _expect(client, _READY_TO_START.format(seat=seat))
  client.send_line(_START_BOARD)
  _expect(client, _READY_FOR_DEAL.format(seat=seat))
  client.send_line(_DEAL)
  _expect(client, _READY_FOR_CARDS.format(seat=seat))
  client.send_line(_CARDS.format(seat=seat, hand=_hand_string(state_vec)))


class BlueChipBridgeBot(pyspiel.Bot):
  """An OpenSpiel bot, wrapping a BlueChip bridge bot implementation."""

  def __init__(self, game, player_id, client):
    """Initializes an OpenSpiel `Bot` wrapping a BlueChip-compatible bot.

    Args:
      game: The OpenSpiel game object, should be an instance of
        bridge_uncontested_bidding, without forced actions.
      player_id: The id of the player the bot will act as, 0 = West (dealer), 1
        = East.
      client: The BlueChip bot; must support methods `start`, `read_line`, and
        `send_line`.
    """
    pyspiel.Bot.__init__(self)
    self._game = game
    self._player_id = player_id
    self._client = client
    self._seat = _SEATS[player_id]
    self._partner = _SEATS[1 - player_id]
    self._left_hand_opponent = _OPPONENTS[player_id]
    self._right_hand_opponent = _OPPONENTS[1 - player_id]
    self._connected = False

  def player_id(self):
    return self._player_id

  def restart(self):
    """Indicates that the next step may be from a non-sequential state."""
    self._connected = False

  def restart_at(self, state):
    """Indicates that the next step may be from a non-sequential state."""
    self._connected = False

  def step(self, state):
    """Returns the action and policy for the bot in this state."""
    state_vec = state.information_state_tensor(self.player_id())

    # Connect if necessary.
    if not self._connected:
      _connect(self._client, self._seat, state_vec)
      self._connected = True

    # Get the actions in the game so far.
    actions = _actions(state_vec)

    # Unless this is the first or second action in the game, our
    # left-hand-opponent will have passed since our last turn.
    if len(actions) > 1:
      _expect(
          self._client,
          _READY_FOR_BID.format(
              seat=self._seat, other=self._left_hand_opponent))
      self._client.send_line(
          _OTHER_PLAYER_PASS.format(player=self._left_hand_opponent))

    # Unless there aren't any prior actions, our partner will have bid
    # or passed since our last turn, and so we need to send partner's action
    # to the bot.
    if actions:
      _expect(self._client,
              _READY_FOR_BID.format(seat=self._seat, other=self._partner))
      if actions[-1] == _PASS_ACTION:
        self._client.send_line(_OTHER_PLAYER_PASS.format(player=self._partner))
      else:
        self._client.send_line(
            _OTHER_PLAYER_BID.format(
                player=self._partner, bid=_action_to_string(actions[-1])))

    # Unless there aren't any prior actions, our right-hand-opponent will have
    # passed since our last turn.
    if actions:
      _expect(
          self._client,
          _READY_FOR_BID.format(
              seat=self._seat, other=self._right_hand_opponent))
      self._client.send_line(
          _OTHER_PLAYER_PASS.format(player=self._right_hand_opponent))

    # Get our action from the bot.
    our_action = _expect_regex(self._client,
                               _SELF_BID_OR_PASS.format(seat=self._seat))
    action = 0 if our_action["pass"] else _string_to_action(our_action["bid"])
    return (action, 1.0), action
