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

"""Tests for the C++ Seep game and its pyspiel.seep Python bindings."""

import copy
import json

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import pyspiel


class GamesSeepTest(parameterized.TestCase):

  def test_load_game_and_metadata(self):
    game = pyspiel.load_game("seep")
    game_type = game.get_type()

    self.assertEqual(game_type.short_name, "seep")
    self.assertEqual(game.num_players(), 4)
    self.assertEqual(game.num_distinct_actions(), 2917)
    self.assertEqual(game.max_chance_outcomes(), 52)
    self.assertEqual(game.max_game_length(), 49)
    self.assertEqual(game.observation_tensor_shape(), [242])

  def test_bindings_phases_and_deal_to_bid(self):
    seep = pyspiel.seep
    game = pyspiel.load_game("seep")
    state = game.new_initial_state()

    self.assertEqual(state.current_phase(), seep.Phase.DEAL_INITIAL)
    self.assertTrue(state.is_chance_node())

    # First test an initial deal where Player 0's 4 hand cards are all < 9
    # ("3C", "4D", "6H", "8S"): a full reshuffle of those 4 cards is triggered!
    low_hand_deal = ["3C", "4D", "6H", "8S"]
    for card_str in low_hand_deal:
      state.apply_action(seep.card_from_string(card_str))
    self.assertEqual(state.current_phase(), seep.Phase.DEAL_INITIAL)
    self.assertEqual(state.num_redeals(), 1)
    self.assertEqual(state.num_cards_dealt(), 0)
    self.assertEmpty(state.floor_piles())
    self.assertEmpty(state.hand(0))
    self.assertLen(state.legal_actions(), 52)

    # Deal 4 bidder cards to Player 0 (including 9S and KS).
    bidder_cards = ["3C", "4D", "9S", "KS"]
    for card_str in bidder_cards:
      state.apply_action(seep.card_from_string(card_str))

    self.assertEqual(state.current_phase(), seep.Phase.BID)
    self.assertEqual(state.current_player(), 0)
    # Floor cards are not dealt or revealed until AFTER Player 0 bids.
    self.assertEmpty(state.floor_piles())
    self.assertLen(state.hand(0), 4)
    obs_json = json.loads(state.observation_string(0))
    self.assertEmpty(obs_json["floor_piles"])
    self.assertEqual(obs_json["hands"][0], ["3C", "4D", "9S", "KS"])

    # Player 0 holds 9S (9) and KS (13), so legal bids are 9 and 13.
    self.assertEqual(
        state.legal_actions(),
        [seep.make_bid_action(9), seep.make_bid_action(13)],
    )

    state.apply_action(seep.make_bid_action(9))
    self.assertEqual(state.bid_value(), 9)
    # After the bid is made, the 4 floor cards are dealt face-up in DEAL_FLOOR.
    self.assertEqual(state.current_phase(), seep.Phase.DEAL_FLOOR)
    self.assertTrue(state.is_chance_node())
    self.assertEmpty(state.floor_piles())

    floor_cards = ["2C", "5D", "7H", "8C"]
    for card_str in floor_cards:
      state.apply_action(seep.card_from_string(card_str))

    # Player 0 must immediately play turn 0 from their initial 4 cards before
    # the remaining 44 cards are dealt.
    self.assertEqual(state.current_phase(), seep.Phase.PLAY)
    self.assertEqual(state.current_player(), 0)
    self.assertLen(state.floor_piles(), 4)
    self.assertLen(state.hand(0), 4)
    state.apply_action(state.legal_actions()[0])
    self.assertEqual(state.current_phase(), seep.Phase.DEAL_REMAINDER)

  def test_card_and_action_helpers_and_custom_play_state(self):
    seep = pyspiel.seep
    self.assertEqual(seep.card_from_string("INVALID"), pyspiel.INVALID_ACTION)
    td = seep.card_from_string("TD")
    self.assertEqual(seep.card_string(td), "TD")
    self.assertEqual(seep.card_capture_value(td), 10)
    self.assertEqual(seep.card_point_value(td), 6)

    bid_act = seep.make_bid_action(11)
    self.assertEqual(seep.bid_value_from_action(bid_act), 11)
    play_act = seep.make_play_action(td, seep.Intent.BUILD_10)
    self.assertEqual(
        seep.decode_play_action(play_act), (td, seep.Intent.BUILD_10)
    )

    game = pyspiel.load_game("seep")
    state = game.new_initial_state()
    house = seep.House(
        value=9,
        is_cemented=False,
        cards=[seep.card_from_string("4S"), seep.card_from_string("5S")],
    )
    self.assertEqual(house.point_value(), 9)
    self.assertEqual(house.to_string(), "9[4S,5S]")

    state.set_custom_play_state(
        current_player=0,
        bid_value=9,
        num_cards_played=10,
        hands=[[seep.card_from_string("9S")], [], [], []],
        floor_piles=[house],
    )
    self.assertEqual(state.current_phase(), seep.Phase.PLAY)
    state.apply_action(
        seep.make_play_action(seep.card_from_string("9S"), seep.Intent.CAPTURE)
    )
    self.assertEqual(state.team_card_points(), [18, 0])
    self.assertEqual(state.team_seeps(), [1, 0])
    self.assertEqual(state.team_total_scores(), [68, 0])
    self.assertEqual(state.last_capture_player(), 0)

  def test_compulsory_capture_and_kachha_vs_pakka_rules(self):
    seep = pyspiel.seep
    game = pyspiel.load_game("seep")
    state = game.new_initial_state()

    # 5 and 3 on floor, player plays 8 -> cannot throw 8, must capture 5 + 3.
    state.set_custom_play_state(
        current_player=0,
        bid_value=9,
        num_cards_played=4,
        hands=[[seep.card_from_string("8S")], [], [], []],
        floor_piles=[
            seep.House(3, False, [seep.card_from_string("3D")]),
            seep.House(5, False, [seep.card_from_string("5S")]),
        ],
    )
    self.assertEqual(
        state.legal_actions(),
        [
            seep.make_play_action(
                seep.card_from_string("8S"), seep.Intent.CAPTURE
            )
        ],
    )

    # Kachha 11 + 2 on floor, player plays K -> must capture kachha 11 + 2.
    state.set_custom_play_state(
        current_player=0,
        bid_value=9,
        num_cards_played=4,
        hands=[[seep.card_from_string("KS")], [], [], []],
        floor_piles=[
            seep.House(
                11,
                False,
                [seep.card_from_string("5C"), seep.card_from_string("6S")],
            ),
            seep.House(2, False, [seep.card_from_string("2D")]),
        ],
    )
    self.assertEqual(
        state.legal_actions(),
        [
            seep.make_play_action(
                seep.card_from_string("KS"), seep.Intent.CAPTURE
            )
        ],
    )

    # Pakka 11 + 2 on floor, player plays K -> cannot capture pakka 11 + 2; must
    # throw K loose.
    state.set_custom_play_state(
        current_player=0,
        bid_value=9,
        num_cards_played=4,
        hands=[[seep.card_from_string("KS")], [], [], []],
        floor_piles=[
            seep.House(
                11,
                True,
                [
                    seep.card_from_string("5C"),
                    seep.card_from_string("6S"),
                    seep.card_from_string("JC"),
                ],
            ),
            seep.House(2, False, [seep.card_from_string("2D")]),
        ],
    )
    self.assertEqual(
        state.legal_actions(),
        [seep.make_play_action(seep.card_from_string("KS"), seep.Intent.THROW)],
    )

    # Two Q's in hand + kachha 10 and 2 on floor -> can play Q:BUILD_12 to make
    # a pakka 12 house while keeping the second Q in hand (whereas with only one
    # Q in hand, BUILD_12 is illegal and only CAPTURE is legal).
    kachha10_and_2 = [
        seep.House(
            10,
            False,
            [seep.card_from_string("4C"), seep.card_from_string("6S")],
        ),
        seep.House(2, False, [seep.card_from_string("2D")]),
    ]
    qc = seep.card_from_string("QC")
    qs = seep.card_from_string("QS")

    # Single Q in hand: only CAPTURE is legal.
    state.set_custom_play_state(
        current_player=0,
        bid_value=9,
        num_cards_played=4,
        hands=[[qc], [], [], []],
        floor_piles=kachha10_and_2,
    )
    self.assertEqual(
        state.legal_actions(),
        [seep.make_play_action(qc, seep.Intent.CAPTURE)],
    )

    # Two Q's in hand: both CAPTURE and BUILD_12 are legal (THROW is illegal).
    state.set_custom_play_state(
        current_player=0,
        bid_value=9,
        num_cards_played=4,
        hands=[[qc, qs], [], [], []],
        floor_piles=kachha10_and_2,
    )
    self.assertIn(
        seep.make_play_action(qc, seep.Intent.BUILD_12), state.legal_actions()
    )
    self.assertNotIn(
        seep.make_play_action(qc, seep.Intent.THROW), state.legal_actions()
    )
    state.apply_action(seep.make_play_action(qc, seep.Intent.BUILD_12))
    self.assertLen(state.floor_piles(), 1)
    self.assertEqual(state.floor_piles()[0].value, 12)
    self.assertTrue(state.floor_piles()[0].is_cemented)
    self.assertLen(state.floor_piles()[0].cards, 4)
    self.assertEqual(state.hand(0), [qs])

  def test_pickle_and_json_structs(self):
    seep = pyspiel.seep
    game = pyspiel.load_game("seep")
    state = game.new_initial_state()
    for card_str in ["3C", "4D", "9S", "KS"]:
      state.apply_action(seep.card_from_string(card_str))
    state.apply_action(seep.make_bid_action(9))
    for card_str in ["2C", "5D", "7H", "8C"]:
      state.apply_action(seep.card_from_string(card_str))

    # Pickle/deepcopy and serialize_game_and_state round-trip for game & state.
    copied_game = copy.deepcopy(game)
    copied_state = copy.deepcopy(state)
    self.assertEqual(str(game), str(copied_game))
    self.assertEqual(str(state), str(copied_state))

    serialized = pyspiel.serialize_game_and_state(game, state)
    deserialized_game, deserialized_state = pyspiel.deserialize_game_and_state(
        serialized
    )
    self.assertEqual(str(game), str(deserialized_game))
    self.assertEqual(str(state), str(deserialized_state))

    # JSON struct and observation struct checks.
    state_json = json.loads(state.to_json())
    self.assertIn("phase", state_json)
    self.assertIn("floor_piles", state_json)

    obs_json = json.loads(state.to_observation_struct(0).to_json())
    self.assertEqual(obs_json["observing_player"], 0)
    self.assertNotEmpty(obs_json["hands"][0])
    self.assertEmpty(obs_json["hands"][1])
    self.assertEmpty(obs_json["hands"][2])
    self.assertEmpty(obs_json["hands"][3])

  def test_illegal_bids_and_actions_rejected(self):
    seep = pyspiel.seep
    game = pyspiel.load_game("seep")
    state = game.new_initial_state()
    for card_str in ["AC", "TC", "QD", "KS"]:
      state.apply_action(seep.card_from_string(card_str))

    self.assertEqual(state.current_phase(), seep.Phase.BID)
    self.assertEmpty(state.floor_piles())
    self.assertEqual(
        state.legal_actions(),
        [
            seep.make_bid_action(10),
            seep.make_bid_action(12),
            seep.make_bid_action(13),
        ],
    )
    # Trying to bid Ace of Clubs (value 1 < 9) or play AC during Bid phase fails
    with self.assertRaises(pyspiel.SpielError):
      state.apply_action(seep.make_bid_action(1))
    with self.assertRaises(pyspiel.SpielError):
      state.apply_action(
          seep.make_play_action(seep.card_from_string("AC"), seep.Intent.THROW)
      )
    # Trying to bid 9 or 11 (which are >= 9 but not in Player 0's hand) fails
    with self.assertRaises(pyspiel.SpielError):
      state.apply_action(seep.make_bid_action(9))
    with self.assertRaises(pyspiel.SpielError):
      state.apply_action(seep.make_bid_action(11))

    # Bidding 10 transitions to DEAL_FLOOR, where the 4 floor cards are dealt
    # face-up, and then transitions to PLAY (Turn 0) with only the 4 initial
    # cards in Player 0's hand:
    # - AC:Build10 has 2 choices: 8S+AH (choice 0) or 7D+2C (choice 1)
    # - TC:Capture has 2 choices: 8S+2C (choice 0) or AH+2C+7D (choice 1)
    # - TC:Build10 is illegal because Player 0 only holds one 10 in the 4 cards.
    state.apply_action(seep.make_bid_action(10))
    self.assertEqual(state.current_phase(), seep.Phase.DEAL_FLOOR)
    for card_str in ["AH", "2C", "7D", "8S"]:
      state.apply_action(seep.card_from_string(card_str))
    self.assertEqual(state.current_phase(), seep.Phase.PLAY)
    self.assertLen(state.hand(0), 4)
    ac = seep.card_from_string("AC")
    tc = seep.card_from_string("TC")
    self.assertEqual(
        state.legal_actions(),
        [
            seep.make_play_action(ac, seep.Intent.BUILD_10, 0),
            seep.make_play_action(ac, seep.Intent.BUILD_10, 1),
            seep.make_play_action(tc, seep.Intent.CAPTURE, 0),
            seep.make_play_action(tc, seep.Intent.CAPTURE, 1),
        ],
    )
    with self.assertRaises(pyspiel.SpielError):
      state.apply_action(seep.make_play_action(tc, seep.Intent.BUILD_10))

    # Verify AC:Build10 choice 0 uses 8S+AH, while choice 1 uses 7D+2C.
    s_choice0 = copy.deepcopy(state)
    s_choice0.apply_action(seep.make_play_action(ac, seep.Intent.BUILD_10, 0))
    self.assertEqual(
        [h.to_string() for h in s_choice0.floor_piles()],
        ["2C", "7D", "10[AC,AH,8S]"],
    )
    s_choice1 = copy.deepcopy(state)
    s_choice1.apply_action(seep.make_play_action(ac, seep.Intent.BUILD_10, 1))
    self.assertEqual(
        [h.to_string() for h in s_choice1.floor_piles()],
        ["AH", "8S", "10[AC,2C,7D]"],
    )

  def test_disjoint_combinations_must_all_be_captured_or_cemented(self):
    seep = pyspiel.seep
    game = pyspiel.load_game("seep")
    state = game.new_initial_state()
    # Floor has 7D + 3C (= 10) AND 8S + 2C (= 10): two disjoint combinations of 10.
    floor = [
        seep.House(2, False, [seep.card_from_string("2C")]),
        seep.House(3, False, [seep.card_from_string("3C")]),
        seep.House(7, False, [seep.card_from_string("7D")]),
        seep.House(8, False, [seep.card_from_string("8S")]),
    ]
    tc = seep.card_from_string("TC")
    td = seep.card_from_string("TD")
    state.set_custom_play_state(
        current_player=0,
        bid_value=10,
        num_cards_played=4,
        hands=[[tc, td], [], [], []],
        floor_piles=floor,
    )
    # Only choice_index = 0 is legal for TC:Capture and TC:Build10 (cannot leave
    # 7D+3C behind!).
    self.assertEqual(
        state.legal_actions(),
        [
            seep.make_play_action(tc, seep.Intent.CAPTURE, 0),
            seep.make_play_action(tc, seep.Intent.BUILD_10, 0),
            seep.make_play_action(td, seep.Intent.CAPTURE, 0),
            seep.make_play_action(td, seep.Intent.BUILD_10, 0),
        ],
    )
    # 1. Capturing with TC picks up all 4 floor cards (8+2 and 7+3 -> Seep!).
    cap_state = state.clone()
    cap_state.apply_action(seep.make_play_action(tc, seep.Intent.CAPTURE, 0))
    self.assertEmpty(cap_state.floor_piles())
    self.assertLen(cap_state.team_captured_cards()[0], 5)
    self.assertEqual(cap_state.team_seeps()[0], 1)

    # 2. Building 10 with TC (while holding TD) cements 8+2, 7+3, and TC into a
    # single cemented 10P house of 5 cards (cannot leave 7+3 alone!).
    build_state = state.clone()
    build_state.apply_action(seep.make_play_action(tc, seep.Intent.BUILD_10, 0))
    self.assertLen(build_state.floor_piles(), 1)
    self.assertTrue(build_state.floor_piles()[0].is_cemented)
    self.assertEqual(
        build_state.floor_piles()[0].to_string(), "10P[2C,3C,7D,8S,TC]"
    )

  def test_cannot_break_own_kachha_house_but_partner_and_opponent_can(self):
    seep = pyspiel.seep
    game = pyspiel.load_game("seep")
    state = game.new_initial_state()
    # Player 0 owns a kachha 9 house (owners_mask = 1 << 0 = 1).
    kachha9_owned_by_p0 = seep.House(
        9,
        False,
        [seep.card_from_string("4C"), seep.card_from_string("5D")],
        owners_mask=1 << 0,
    )
    c3s = seep.card_from_string("3S")
    qd = seep.card_from_string("QD")
    c9c = seep.card_from_string("9C")

    # 1. Player 0 (the owner) CANNOT break their own kachha 9 house with 3S -> 12
    state.set_custom_play_state(
        current_player=0,
        bid_value=9,
        num_cards_played=4,
        hands=[[c3s, c9c, qd], [], [], []],
        floor_piles=[kachha9_owned_by_p0],
    )
    self.assertNotIn(
        seep.make_play_action(c3s, seep.Intent.BUILD_12),
        state.legal_actions(),
    )

    # 2. Opponent (Player 1) CAN break Player 0's kachha 9 house with 3S -> 12
    state.set_custom_play_state(
        current_player=1,
        bid_value=9,
        num_cards_played=5,
        hands=[[c9c], [c3s, qd], [], []],
        floor_piles=[kachha9_owned_by_p0],
    )
    self.assertIn(
        seep.make_play_action(c3s, seep.Intent.BUILD_12),
        state.legal_actions(),
    )

    # 3. Partner (Player 2) CAN ALSO break Player 0's kachha 9 house with 3S -> 12
    # and ownership transfers to Player 2 (owners_mask == 1 << 2).
    state.set_custom_play_state(
        current_player=2,
        bid_value=9,
        num_cards_played=6,
        hands=[[c9c], [], [c3s, qd], []],
        floor_piles=[kachha9_owned_by_p0],
    )
    break_act = seep.make_play_action(c3s, seep.Intent.BUILD_12)
    self.assertIn(break_act, state.legal_actions())
    state.apply_action(break_act)
    self.assertEqual(state.floor_piles()[0].value, 12)
    self.assertEqual(state.floor_piles()[0].owners_mask, 1 << 2)

  def test_house_owner_card_binding_lifecycle(self):
    seep = pyspiel.seep
    game = pyspiel.load_game("seep")
    state = game.new_initial_state()

    ac = seep.card_from_string("AC")
    c2d = seep.card_from_string("2D")
    c3d = seep.card_from_string("3D")
    c4c = seep.card_from_string("4C")
    c5c = seep.card_from_string("5C")
    c6s = seep.card_from_string("6S")
    c7d = seep.card_from_string("7D")
    tc = seep.card_from_string("TC")
    qc = seep.card_from_string("QC")
    qd = seep.card_from_string("QD")
    qh = seep.card_from_string("QH")
    qs = seep.card_from_string("QS")
    ks = seep.card_from_string("KS")

    # 1. P0 builds 12[2D, TC] (owners_mask = 1 << 0). Floor also has loose AC.
    p0_kachha12_floor = [
        seep.House(1, False, [ac]),
        seep.House(12, False, [c2d, tc], owners_mask=1 << 0),
    ]
    # With 1 Q (QD) + KS: P0 cannot play QD:Build13, must keep QD for 12 house.
    state.set_custom_play_state(
        current_player=0,
        bid_value=10,
        num_cards_played=4,
        hands=[[qd, ks], [], [], []],
        floor_piles=p0_kachha12_floor,
    )
    self.assertNotIn(
        seep.make_play_action(qd, seep.Intent.BUILD_13), state.legal_actions()
    )
    self.assertIn(
        seep.make_play_action(qd, seep.Intent.CAPTURE), state.legal_actions()
    )

    # 2. With 2 Q's (QD, QS) + KS: P0 CAN play QD:Build13 (retaining QS for 12).
    state.set_custom_play_state(
        current_player=0,
        bid_value=10,
        num_cards_played=4,
        hands=[[qd, qs, ks], [], [], []],
        floor_piles=p0_kachha12_floor,
    )
    self.assertIn(
        seep.make_play_action(qd, seep.Intent.BUILD_13), state.legal_actions()
    )

    # 3. Opponent P1 cements P0's 12 house (QC:Build12 -> 12P, owners_mask = 3).
    #    P0 with 1 Q (QD) is STILL bound to the 12P house (cannot play QD:Build13).
    state.set_custom_play_state(
        current_player=1,
        bid_value=10,
        num_cards_played=5,
        hands=[[qd, ks], [qc, qh], [], []],
        floor_piles=p0_kachha12_floor,
    )
    state.apply_action(seep.make_play_action(qc, seep.Intent.BUILD_12))
    state.set_custom_play_state(
        current_player=0,
        bid_value=10,
        num_cards_played=8,
        hands=[[qd, ks], [qh], [], []],
        floor_piles=state.floor_piles(),
    )
    self.assertNotIn(
        seep.make_play_action(qd, seep.Intent.BUILD_13), state.legal_actions()
    )
    self.assertIn(
        seep.make_play_action(qd, seep.Intent.CAPTURE), state.legal_actions()
    )

    # 4. After P0's 12 house is picked up, Partner P2 or Opponent P1 creates a
    #    new 5 + 7 = 12 house (kachha or pakka). P0 with 1 Q (QD) is NOT bound
    #    to that house, so both QD:Capture and QD:Build13 (QD + AC -> 13) are legal!
    for is_pakka in (False, True):
      state.set_custom_play_state(
          current_player=0,
          bid_value=10,
          num_cards_played=10,
          hands=[[qd, ks], [], [], []],
          floor_piles=[
              seep.House(1, False, [ac]),
              seep.House(12, is_pakka, [c5c, c7d], owners_mask=1 << 2),
          ],
      )
      self.assertIn(
          seep.make_play_action(qd, seep.Intent.CAPTURE), state.legal_actions()
      )
      self.assertIn(
          seep.make_play_action(qd, seep.Intent.BUILD_13), state.legal_actions()
      )

    # 5. If P0 built a kachha 10 house and loose 3 is on floor, playing KS (13)
    #    must pick up 10 + 3 (KS:Throw is illegal).
    state.set_custom_play_state(
        current_player=0,
        bid_value=10,
        num_cards_played=8,
        hands=[[tc, ks], [], [], []],
        floor_piles=[
            seep.House(3, False, [c3d]),
            seep.House(10, False, [c4c, c6s], owners_mask=1 << 0),
        ],
    )
    self.assertNotIn(
        seep.make_play_action(ks, seep.Intent.THROW), state.legal_actions()
    )
    self.assertIn(
        seep.make_play_action(ks, seep.Intent.CAPTURE), state.legal_actions()
    )

    # 6. Partner Exception on kachha or pakka 12 house owned by P0:
    #    - Partner P2 can play 7D + 5C -> 12 onto P0's 12 house even with NO Q!
    #    - Partner P2 with a single Q (QC) can both Capture and Build12 (cement
    #      or re-cement by placing QC onto the 12 pile) without a second Q!
    #    - Opponent P1 cannot do either without retaining a Q in hand.
    for is_pakka in (False, True):
      p0_house_and_5 = [
          seep.House(5, False, [c5c]),
          seep.House(12, is_pakka, [c2d, tc], owners_mask=1 << 0),
      ]
      state.set_custom_play_state(
          current_player=2,
          bid_value=10,
          num_cards_played=6,
          hands=[[qd], [], [c7d], []],
          floor_piles=p0_house_and_5,
      )
      self.assertIn(
          seep.make_play_action(c7d, seep.Intent.BUILD_12),
          state.legal_actions(),
      )

      state.set_custom_play_state(
          current_player=2,
          bid_value=10,
          num_cards_played=6,
          hands=[[qd], [], [qc], []],
          floor_piles=p0_house_and_5,
      )
      self.assertIn(
          seep.make_play_action(qc, seep.Intent.CAPTURE), state.legal_actions()
      )
      self.assertIn(
          seep.make_play_action(qc, seep.Intent.BUILD_12), state.legal_actions()
      )

      state.set_custom_play_state(
          current_player=1,
          bid_value=10,
          num_cards_played=5,
          hands=[[qd], [c7d, qc], [], []],
          floor_piles=p0_house_and_5,
      )
      self.assertNotIn(
          seep.make_play_action(qc, seep.Intent.BUILD_12), state.legal_actions()
      )

    # 7. Cementing partner's 13 house with KS:Build13 when a 10 house and loose
    #    3S are also on the floor:
    #    - If 10 house is kachha, KS:Build13 MUST merge 13 + (10 + 3S) + KS into
    #      a single 6-card 13P house (no option to leave 10 + 3S behind).
    #    - If 10 house is pakka (10P), KS:Build13 CANNOT merge 10P + 3S; only
    #      13 + KS becomes 13P while 3S and 10P remain on the floor.
    c2h = seep.card_from_string("2H")
    c3s = seep.card_from_string("3S")
    c8c = seep.card_from_string("8C")
    c9d = seep.card_from_string("9D")
    kc = seep.card_from_string("KC")

    state.set_custom_play_state(
        current_player=0,
        bid_value=10,
        num_cards_played=16,
        hands=[[kc, ks], [], [], []],
        floor_piles=[
            seep.House(3, False, [c3s]),
            seep.House(10, False, [c2h, c8c], owners_mask=1 << 1),
            seep.House(13, False, [c4c, c9d], owners_mask=1 << 2),
        ],
    )
    self.assertIn(
        seep.make_play_action(ks, seep.Intent.BUILD_13, 0),
        state.legal_actions(),
    )
    self.assertNotIn(
        seep.make_play_action(ks, seep.Intent.BUILD_13, 1),
        state.legal_actions(),
    )
    state.apply_action(seep.make_play_action(ks, seep.Intent.BUILD_13, 0))
    self.assertEqual(
        [h.to_string() for h in state.floor_piles()],
        ["13P[2H,3S,4C,8C,9D,KS]"],
    )

    state.set_custom_play_state(
        current_player=0,
        bid_value=10,
        num_cards_played=16,
        hands=[[kc, ks], [], [], []],
        floor_piles=[
            seep.House(3, False, [c3s]),
            seep.House(10, True, [c2h, c8c, tc], owners_mask=1 << 1),
            seep.House(13, False, [c4c, c9d], owners_mask=1 << 2),
        ],
    )
    state.apply_action(seep.make_play_action(ks, seep.Intent.BUILD_13, 0))
    self.assertEqual(
        [h.to_string() for h in state.floor_piles()],
        ["3S", "10P[2H,8C,TC]", "13P[4C,9D,KS]"],
    )

  def test_random_simulations_preserve_100_deck_points_and_zero_sum(self):
    game = pyspiel.load_game("seep")
    rng = np.random.default_rng(42)

    for _ in range(15):
      state = game.new_initial_state()
      while not state.is_terminal():
        legal = state.legal_actions()
        self.assertNotEmpty(legal)
        action = int(rng.choice(legal))
        state.apply_action(action)

      card_points = state.team_card_points()
      self.assertEqual(card_points[0] + card_points[1], 100)

      returns = state.returns()
      self.assertLen(returns, 4)
      self.assertEqual(returns[0], returns[2])
      self.assertEqual(returns[1], returns[3])
      self.assertAlmostEqual(sum(returns), 0.0)


if __name__ == "__main__":
  absltest.main()
