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

"""Tests for open_spiel.python.bots.bluechip_bridge_uncontested_bidding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import absltest
from open_spiel.python.bots import bluechip_bridge_uncontested_bidding
import pyspiel

_BID_1D = bluechip_bridge_uncontested_bidding._string_to_action("1D")
_BID_1H = bluechip_bridge_uncontested_bidding._string_to_action("1H")
_BID_2H = bluechip_bridge_uncontested_bidding._string_to_action("2H")


class BluechipBridgeWrapperTest(absltest.TestCase):

  def test_complete_session_east(self):
    game = pyspiel.load_game("bridge_uncontested_bidding")
    mock_client = absltest.mock.Mock(
        **{
            "read_line.side_effect": [
                'Connecting "WBridge5" as ANYPL using protocol version 18',
                "EAST ready for teams",
                "EAST ready to start",
                "EAST ready for deal",
                "EAST ready for cards",
                "EAST ready for WEST's bid",
                "EAST ready for NORTH's bid",
                "EAST bids 1H",
                "EAST ready for SOUTH's bid",
                "EAST ready for WEST's bid",
                "EAST ready for NORTH's bid",
                "EAST PASSES",
            ]
        })
    bot = bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(
        game, 1, mock_client)
    state = game.deserialize_state("A86.J543.K642.A3 J.KQ962.T953.J96")
    state.apply_action(_BID_1D)
    policy, action = bot.step(state)
    self.assertEqual(action, _BID_1H)
    self.assertEqual(policy, (_BID_1H, 1.0))
    state.apply_action(action)
    state.apply_action(_BID_2H)
    policy, action = bot.step(state)
    self.assertEqual(action, bluechip_bridge_uncontested_bidding._PASS_ACTION)
    self.assertEqual(policy,
                     (bluechip_bridge_uncontested_bidding._PASS_ACTION, 1.0))
    # Finished - now check that the game state is correct.
    self.assertEqual(str(state), "A86.J543.K642.A3 J.KQ962.T953.J96 1D-1H-2H")
    # Check that we received the expected messages.
    mock_client.assert_has_calls([
        absltest.mock.call.start(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('EAST ("WBridge5") seated'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('Teams: N/S "opponents" E/W "bidders"'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("start of board"),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            "Board number 8. Dealer WEST. Neither vulnerable."),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            "EAST's cards: S J. H K Q 9 6 2. D T 9 5 3. C J 9 6."),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("WEST bids 1D"),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("NORTH PASSES"),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("SOUTH PASSES"),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("WEST bids 2H"),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("NORTH PASSES"),
    ])

  def test_complete_session_west(self):
    game = pyspiel.load_game("bridge_uncontested_bidding")
    mock_client = absltest.mock.Mock(
        **{
            "read_line.side_effect": [
                'Connecting "WBridge5" as ANYPL using protocol version 18',
                "WEST ready for teams",
                "WEST ready to start",
                "WEST ready for deal",
                "WEST ready for cards",
                "WEST bids 1D Alert.",
                "WEST ready for NORTH's bid",
                "WEST ready for EAST's bid",
                "WEST ready for SOUTH's bid",
                "WEST bids 2H",
                "WEST ready for NORTH's bid",
                "WEST ready for EAST's bid",
                "WEST ready for SOUTH's bid",
            ]
        })
    bot = bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(
        game, 0, mock_client)
    state = game.deserialize_state("A86.J543.K642.A3 J.KQ962.T953.J96")
    policy, action = bot.step(state)
    self.assertEqual(action, _BID_1D)
    self.assertEqual(policy, (_BID_1D, 1.0))
    state.apply_action(action)
    state.apply_action(_BID_1H)
    policy, action = bot.step(state)
    self.assertEqual(action, _BID_2H)
    self.assertEqual(policy, (_BID_2H, 1.0))
    state.apply_action(action)
    # Finished - now check that the game state is correct.
    self.assertEqual(str(state), "A86.J543.K642.A3 J.KQ962.T953.J96 1D-1H-2H")
    # Check that we received the expected messages.
    mock_client.assert_has_calls([
        absltest.mock.call.start(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('WEST ("WBridge5") seated'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('Teams: N/S "opponents" E/W "bidders"'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("start of board"),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            "Board number 8. Dealer WEST. Neither vulnerable."),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            "WEST's cards: S A 8 6. H J 5 4 3. D K 6 4 2. C A 3."),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("NORTH PASSES"),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("EAST bids 1H"),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line("SOUTH PASSES"),
        absltest.mock.call.read_line(),
    ])

  def test_invalid_fixed_message(self):
    game = pyspiel.load_game("bridge_uncontested_bidding")
    mock_client = absltest.mock.Mock(
        **{
            "read_line.side_effect": [
                'Connecting "WBridge5" as ANYPL using protocol version 18',
                "WEST ready for cards",
            ]
        })
    bot = bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(
        game, 0, mock_client)
    state = game.deserialize_state("A86.J543.K642.A3 J.KQ962.T953.J96")
    with self.assertRaisesRegex(
        ValueError,
        "Received 'WEST ready for cards' but expected 'WEST ready for teams'"):
      bot.step(state)

  def test_invalid_variable_message(self):
    game = pyspiel.load_game("bridge_uncontested_bidding")
    mock_client = absltest.mock.Mock(
        **{
            "read_line.side_effect": [
                'Connecting "WBridge5" as ANYPL using protocol version 18',
                "WEST ready for teams",
                "WEST ready to start",
                "WEST ready for deal",
                "WEST ready for cards",
                "NORTH bids 1S",
            ]
        })
    bot = bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(
        game, 0, mock_client)
    state = game.deserialize_state("A86.J543.K642.A3 J.KQ962.T953.J96")
    with self.assertRaisesRegex(
        ValueError,
        "Received 'NORTH bids 1S' which does not match regex 'WEST"):
      bot.step(state)

  def test_string_to_action_to_string_roundtrip(self):
    for level, trump in itertools.product(
        range(1, 8), bluechip_bridge_uncontested_bidding._TRUMP_SUIT):
      bid = str(level) + trump
      action = bluechip_bridge_uncontested_bidding._string_to_action(bid)
      self.assertEqual(
          bid, bluechip_bridge_uncontested_bidding._action_to_string(action))

  def test_action_to_string_to_action_roundtrip(self):
    for action in range(1, 36):
      bid = bluechip_bridge_uncontested_bidding._action_to_string(action)
      self.assertEqual(
          action, bluechip_bridge_uncontested_bidding._string_to_action(bid))


if __name__ == "__main__":
  absltest.main()
