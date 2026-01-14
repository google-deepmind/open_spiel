// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/dou_dizhu/dou_dizhu_utils.h"

#include <cstring>
#include <iostream>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace dou_dizhu {

void SingleRankHandTest() {
  std::array<int, kNumRanks> hand1{}, hand2{};
  hand1[6] = 3;
  int action_id1 = SingleRankHandToActionId(hand1);
  SPIEL_CHECK_EQ(FormatSingleHand(SingleRankHand(action_id1)), "999");

  hand2[13] = 1;
  int action_id2 = SingleRankHandToActionId(hand2);
  SPIEL_CHECK_EQ(FormatSingleHand(SingleRankHand(action_id2)), "(BWJ)");

  // 558999TJJJJKKK
  std::array<int, kNumRanks> current_hand = {0, 0, 2, 0, 0, 1, 3, 1, 4, 0, 3};

  std::vector<Action> actions1, actions2, actions3;

  // The only hands that are greater than 999 are JJJ and KKK
  SearchSingleRankActions(&actions1, current_hand, /*prev_action=*/action_id1);
  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 2);

  // No hands greater than BWJ
  SearchSingleRankActions(&actions2, current_hand, /*prev_action=*/action_id2);
  SPIEL_CHECK_EQ(static_cast<int>(actions2.size()), 0);

  // 6 solos + 4 pairs + 3 trios + 1 bomb = 14
  SearchSingleRankActions(&actions3, current_hand,
                          /*prev_action=*/kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 14);
}

void ChainOnlyHandTest() {
  std::array<int, kNumRanks> hand1 = {0, 0, 0, 3, 3, 3};
  int action_id1 = ChainOnlyHandToActionId(hand1);

  SPIEL_CHECK_EQ(FormatSingleHand(ChainOnlyHand(action_id1)), "666777888");

  std::array<int, kNumRanks> hand2 = {2, 2, 2, 2, 2, 2, 2, 2, 2};

  int action_id2 = ChainOnlyHandToActionId(hand2);
  SPIEL_CHECK_EQ(FormatSingleHand(ChainOnlyHand(action_id2)),
                 "33445566778899TTJJ");

  // 5566777888999TTTJJQQKKAA22(BWJ)(CJ)
  std::array<int, kNumRanks> current_hand = {0, 0, 2, 2, 3, 3, 3, 3,
                                             2, 2, 2, 2, 2, 1, 1};

  std::vector<Action> actions1, actions2, actions3;
  SearchChainOnlyActions(&actions1, current_hand, /*prev_action=*/action_id1);

  // The only hands greater than 666777888 are 777888999 and 888999TTT
  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 2);

  SearchChainOnlyActions(&actions2, current_hand, /*prev_action=*/action_id2);

  // The only hands greater than 334455....TTJJ are 5566....QQKK and
  // 6677.....KKAA
  SPIEL_CHECK_EQ(static_cast<int>(actions2.size()), 2);

  SearchChainOnlyActions(&actions3, current_hand,
                         /*prev_action=*/kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 63);
}

void SingleTrioCombHandTest() {
  std::array<int, kNumRanks> hand1{}, hand2{};

  // 999-(CJ)
  hand1[6] = 3;
  hand1[14] = 1;
  int action_id1 = SingleTrioCombHandToActionId(hand1);
  SPIEL_CHECK_EQ(FormatSingleHand(SingleTrioCombHand(action_id1)), "999(CJ)");

  // 333-22
  hand2[12] = 2;
  hand2[0] = 3;

  int action_id2 = SingleTrioCombHandToActionId(hand2);
  SPIEL_CHECK_EQ(FormatSingleHand(SingleTrioCombHand(action_id2)), "33322");

  // 666777TTTQQQ222(BWJ)(CJ)
  std::array<int, kNumRanks> current_hand = {0, 0, 0, 3, 3, 0, 0, 3,
                                             0, 3, 0, 0, 3, 1, 1};

  std::vector<Action> actions1, actions2, actions3;

  // The hands that are greater than 333222 uses trios 666, 777, TTT, QQQ, 222
  // And we just enuemerate all possible pairs
  SearchSingleTrioCombActions(&actions1, current_hand,
                              /*prev_action=*/action_id1);
  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 18);

  SearchSingleTrioCombActions(&actions2, current_hand,
                              /*prev_action=*/action_id2);
  SPIEL_CHECK_EQ(static_cast<int>(actions2.size()), 20);

  SearchSingleTrioCombActions(&actions3, current_hand, kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 50);
}

void AirplaneCombHandTest() {
  // 888999TTTJJJQQQ-7772(CJ)
  std::array<int, kNumRanks> hand1 = {0, 0, 0, 0, 3, 3, 3, 3,
                                      3, 3, 0, 0, 1, 0, 1};

  int action_id1 = AirplaneCombHandToActionId(hand1, /*chain_head=*/5,
                                              /*kicker_type=*/kSolo);
  SPIEL_CHECK_EQ(FormatSingleHand(AirplaneCombHand(action_id1)),
                 "777888999TTTJJJQQQ2(CJ)");

  // TTTJJJQQQKKK-33445522
  std::array<int, kNumRanks> hand2 = {2, 2, 2, 0, 0, 0, 0, 3,
                                      3, 3, 3, 0, 2, 0, 0};
  int action_id2 = AirplaneCombHandToActionId(hand2, /*chain_head=*/7,
                                              /*kicker_type=*/kPair);
  SPIEL_CHECK_EQ(FormatSingleHand(AirplaneCombHand(action_id2)),
                 "334455TTTJJJQQQKKK22");

  // 667899TTTJJJJQQQKKKAAA222(BWJ)(CJ)
  std::array<int, kNumRanks> current_hand = {0, 0, 0, 2, 1, 1, 2, 3,
                                             4, 3, 3, 3, 3, 1, 1};
  std::vector<Action> actions1, actions2, actions3;
  SearchAirplaneCombActions(&actions1, current_hand,
                            /*prev_action=*/action_id1);
  // C(7, 5) - C(5, 3) + 3*(C(6, 3) - C(4, 1)) + C(3, 2) * 5 + 2 + C(6, 2) - 1 =
  // 90
  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 90);

  // The only hand that is greater than TTTJJJQQQKKK-33445522 is
  // JJJQQQKKKAAA-6699TT22
  SearchAirplaneCombActions(&actions2, current_hand,
                            /*prev_action=*/action_id2);
  SPIEL_CHECK_EQ(static_cast<int>(actions2.size()), 1);

  SearchAirplaneCombActions(&actions3, current_hand,
                            /*prev_action=*/kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 1052);
}

}  // namespace dou_dizhu
}  // namespace open_spiel

int main() {
  open_spiel::dou_dizhu::SingleRankHandTest();
  open_spiel::dou_dizhu::ChainOnlyHandTest();
  open_spiel::dou_dizhu::SingleTrioCombHandTest();
  open_spiel::dou_dizhu::AirplaneCombHandTest();
}
