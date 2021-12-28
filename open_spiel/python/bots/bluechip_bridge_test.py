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

from absl.testing import absltest
from open_spiel.python.bots import bluechip_bridge
import pyspiel


class BluechipBridgeWrapperTest(absltest.TestCase):

  def test_complete_deal_east(self):
    # Plays a complete deal, with the mock external bot playing East.
    # The deal is as follows:
    #
    # Vul: None
    #         S AKJ8
    #         H 4
    #         D JT9532
    #         C 32
    # S 3             S Q9
    # H KQJ8762       H AT5
    # D K4            D A87
    # C KQ4           C AJT96
    #         S T76542
    #         H 93
    #         D Q6
    #         C 875
    #
    # West  North East  South
    #       Pass  1N    Pass
    # 2D    Pass  2H    Pass
    # 3S    Dbl   4C    Pass
    # 4D    Pass  4N    Pass
    # 5D    Pass  6H    Pass
    # Pass  Pass
    #
    # N  E  S  W  N  E  S
    #       S7 S3 SK S9
    # DJ D8 D6 DK
    #          H2 H4 HT H9
    #    H5 H3 H6 C3
    #          C4 C2 CT C5
    #    C6 C7 CQ D2
    #          CK D3 CJ C8
    #          D4 D5 DA DQ
    #    C9 S2 H7 S8
    #          HK SJ HA S4
    #    CA S5 H8 D9
    #          HQ DT D7 S6
    #          HJ SA SQ ST
    #
    # Declarer tricks: 12

    game = pyspiel.load_game('bridge(use_double_dummy_result=false)')
    mock_client = absltest.mock.Mock(
        **{
            'read_line.side_effect': [
                'Connecting "WBridge5" as ANYPL using protocol version 18',
                'EAST ready for teams',
                'EAST ready to start',
                'EAST ready for deal',
                'EAST ready for cards',
                "EAST ready for NORTH's bid",
                'EAST bids 1NT',
                "EAST ready for SOUTH's bid",
                "EAST ready for WEST's bid",
                "EAST ready for NORTH's bid",
                'EAST bids 2H',
                "EAST ready for SOUTH's bid",
                "EAST ready for WEST's bid",
                "EAST ready for NORTH's bid",
                'EAST bids 4C Alert.',
                "EAST ready for SOUTH's bid",
                "EAST ready for WEST's bid",
                "EAST ready for NORTH's bid",
                'EAST bids 4NT',
                "EAST ready for SOUTH's bid",
                "EAST ready for WEST's bid",
                "EAST ready for NORTH's bid",
                'EAST bids 6H',
                "EAST ready for SOUTH's bid",
                "EAST ready for WEST's bid",
                "EAST ready for NORTH's bid",
                "EAST ready for SOUTH's card to trick 1",
                'EAST ready for dummy',
                'WEST plays 3s',
                "EAST ready for NORTH's card to trick 1",
                'EAST plays 9s',
                "EAST ready for NORTH's card to trick 2",
                'EAST plays 8d',
                "EAST ready for SOUTH's card to trick 2",
                'WEST plays kd',
                'WEST plays 2h',
                "EAST ready for NORTH's card to trick 3",
                'EAST plays th',
                "EAST ready for SOUTH's card to trick 3",
                'EAST plays 5h',
                "EAST ready for SOUTH's card to trick 4",
                'WEST plays 6h',
                "EAST ready for NORTH's card to trick 4",
                'WEST plays 4c',
                "EAST ready for NORTH's card to trick 5",
                'EAST plays tc',
                "EAST ready for SOUTH's card to trick 5",
                'EAST plays 6c',
                "EAST ready for SOUTH's card to trick 6",
                'WEST plays qc',
                "EAST ready for NORTH's card to trick 6",
                'WEST plays kc',
                "EAST ready for NORTH's card to trick 7",
                'EAST plays jc',
                "EAST ready for SOUTH's card to trick 7",
                'WEST plays 4d',
                "EAST ready for NORTH's card to trick 8",
                'EAST plays ad',
                "EAST ready for SOUTH's card to trick 8",
                'EAST plays 9c',
                "EAST ready for SOUTH's card to trick 9",
                'WEST plays 7h',
                "EAST ready for NORTH's card to trick 9",
                'WEST plays kh',
                "EAST ready for NORTH's card to trick 10",
                'EAST plays ah',
                "EAST ready for SOUTH's card to trick 10",
                'EAST plays ac',
                "EAST ready for SOUTH's card to trick 11",
                'WEST plays 8h',
                "EAST ready for NORTH's card to trick 11",
                'WEST plays qh',
                "EAST ready for NORTH's card to trick 12",
                'EAST plays 7d',
                "EAST ready for SOUTH's card to trick 12",
                'WEST plays jh',
                "EAST ready for NORTH's card to trick 13",
                'EAST plays qs',
            ]
        })
    bot = bluechip_bridge.BlueChipBridgeBot(game, 1, lambda: mock_client)
    state = game.new_initial_state()
    history = [
        33, 25, 3, 44, 47, 28, 23, 46, 1, 43, 30, 26, 29, 48, 24, 42, 13, 21,
        17, 8, 5, 34, 6, 7, 37, 49, 11, 38, 51, 32, 20, 9, 0, 14, 35, 22, 10,
        50, 15, 45, 39, 16, 12, 18, 27, 31, 41, 40, 4, 36, 19, 2, 52, 59, 52,
        61, 52, 62, 52, 68, 53, 70, 52, 71, 52, 74, 52, 76, 52, 82, 52, 52, 52,
        23, 7, 47, 31, 37, 25, 17, 45, 2, 10, 34, 30, 14, 6, 18, 4, 8, 0, 32,
        12, 16, 20, 40, 1, 44, 5, 36, 24, 9, 13, 49, 41, 28, 3, 22, 27, 46, 39,
        50, 11, 48, 15, 26, 29, 42, 33, 21, 19, 38, 51, 43, 35
    ]

    # Check the bot provides the expected actions
    for action in history:
      if state.current_player() == 1:
        bot_action = bot.step(state)
        self.assertEqual(action, bot_action)
      state.apply_action(action)

    # Check the session went as expected; send_line calls are us sending
    # data to the (mock) external bot.
    mock_client.assert_has_calls([
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('EAST ("WBridge5") seated'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            'Teams: N/S "north-south" E/W "east-west"'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('start of board'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            'Board number 1. Dealer NORTH. Neither vulnerable.'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            "EAST's cards: C A J T 9 6. D A 8 7. H A T 5. S Q 9."),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('WEST bids 2D'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('WEST bids 3S'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH DOUBLES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('WEST bids 4D'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('WEST bids 5D'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('WEST PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH PASSES'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 7s'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line(
            "Dummy's cards: C K Q 4. D K 4. H K Q J 8 7 6 2. S 3."),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays ks'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays jd'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 6d'),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 4h'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 9h'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 3h'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 3c'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 2c'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 5c'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 7c'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 2d'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 3d'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 8c'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 5d'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays qd'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 2s'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 8s'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays js'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 4s'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 5s'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays 9d'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays td'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('SOUTH plays 6s'),
        absltest.mock.call.send_line('EAST to lead'),
        absltest.mock.call.read_line(),
        absltest.mock.call.read_line(),
        absltest.mock.call.send_line('NORTH plays as'),
        absltest.mock.call.read_line(),
    ])


if __name__ == '__main__':
  absltest.main()
