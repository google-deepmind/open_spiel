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

#include <iostream>
#include <cstring>
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/spiel.h"

#include "open_spiel/games/dou_dizhu/dou_dizhu_utils.h"


namespace open_spiel{
namespace dou_dizhu{



void SingleRankHandTest(){
  std::array<int, kNumRanks> hand1{};
  // 999
  hand1[6] = 3;
  int action_id1 = SingleRankHandToActionId(hand1);
  // std::cout << "Begin test for transforming hands to Ids" << std::endl;
  // std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(hand1)) << std::endl;
  // std::cout << action_id << std::endl;
  // std::cout << FormatSingleHand(SingleRankHand(action_id)) << std::endl;
  SPIEL_CHECK_EQ(FormatSingleHand(SingleRankHand(action_id1)), "999");

  std::array<int, kNumRanks> hand2{};
  // BWJ
  hand2[13] = 1;
  int action_id2 = SingleRankHandToActionId(hand2);
  SPIEL_CHECK_EQ(FormatSingleHand(SingleRankHand(action_id2)), "(BWJ)");



  std::array<int, kNumRanks> current_hand{};
  // 558999TJJJJKKK
  current_hand[2] = 2;
  current_hand[5] = 1;
  current_hand[6] = 3;
  current_hand[7] = 1;
  current_hand[8] = 4;
  current_hand[10] = 3;
  std::vector<Action> actions1;
  // std::cout << "Begin test for search actions" << std::endl;
  // std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(current_hand)) << std::endl;
  SearchSingleRankActions(actions1, current_hand, /*prev_action=*/action_id1);
  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 2);


  std::vector<Action> actions2;
  SearchSingleRankActions(actions2, current_hand, /*prev_action=*/action_id2);
  SPIEL_CHECK_EQ(static_cast<int>(actions2.size()), 0);



  std::vector<Action> actions3;
  SearchSingleRankActions(actions3, current_hand, /*prev_action=*/kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 14);
  // std::cout << "Possible actions:" << std::endl;
  // for(auto action: actions3){
  //   std::array<int, kNumRanks> possible_hand = SingleRankHand(action);
  //   std::cout << FormatSingleHand(possible_hand) << std::endl;
  // }
}


void ChainOnlyHandTest(){
  std::array<int, kNumRanks> hand1{};
  // 666777888
  hand1[3] = 3;
  hand1[4] = 3;
  hand1[5] = 3;
  int action_id1 = ChainOnlyHandToActionId(hand1);
  // std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(hand1)) << std::endl;
  // std::cout << action_id1 << std::endl;
  // std::cout << FormatSingleHand(ChainOnlyHand(action_id1)) << std::endl;

  SPIEL_CHECK_EQ(FormatSingleHand(ChainOnlyHand(action_id1)), "666777888");



  std::array<int, kNumRanks> hand2{};
  // 33445566778899TTJJ
  hand2[0] = 2;
  hand2[1] = 2;
  hand2[2] = 2;
  hand2[3] = 2;
  hand2[4] = 2;
  hand2[5] = 2;
  hand2[6] = 2;
  hand2[7] = 2;
  hand2[8] = 2;

  int action_id2 = ChainOnlyHandToActionId(hand2);
  SPIEL_CHECK_EQ(FormatSingleHand(ChainOnlyHand(action_id2)), "33445566778899TTJJ");


  std::array<int, kNumRanks> current_hand{};
  // 5566777888999TTTJJQQKKAA22(BWJ)(CJ)
  current_hand[2] = 2;
  current_hand[3] = 2;
  current_hand[4] = 3;
  current_hand[5] = 3;
  current_hand[6] = 3;
  current_hand[7] = 3;
  current_hand[8] = 2;
  current_hand[9] = 2;
  current_hand[10] = 2;
  current_hand[11] = 2;
  current_hand[12] = 2;
  current_hand[13] = 1;
  current_hand[14] = 1;
  std::vector<Action> actions1;
  std::cout << "Begin test for search actions" << std::endl;
  std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(current_hand)) << std::endl;
  SearchChainOnlyActions(actions1, current_hand, /*prev_action=*/action_id1);


  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 2);

  std::vector<Action> actions2;
  SearchChainOnlyActions(actions2, current_hand, /*prev_action=*/action_id2);


  // std::cout << "Possible actions:" << std::endl;

  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 2);


  std::vector<Action> actions3;
  SearchChainOnlyActions(actions3, current_hand, /*prev_action=*/kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 63);

  // std::cout << "Possible actions:" << std::endl;
  // for(auto action: actions){
  //   std::array<int, kNumRanks> possible_hand = ChainOnlyHand(action);
  //   std::cout << FormatSingleHand(possible_hand) << std::endl;
  // }
}


void SingleTrioCombHandTest(){
  std::array<int, kNumRanks> hand1{};

  //999-(CJ)
  hand1[6] = 3;
  hand1[14] = 1;
  int action_id1 = SingleTrioCombHandToActionId(hand1);
  // std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(hand1)) << std::endl;
  // std::cout << action_id1 << std::endl;
  // std::cout << FormatSingleHand(SingleTrioCombHand(action_id1)) << std::endl;

  SPIEL_CHECK_EQ(FormatSingleHand(SingleTrioCombHand(action_id1)), "999(CJ)");


  std::array<int, kNumRanks> hand2{};

  // 333-22
  hand2[12] = 2;
  hand2[0] = 3;

  int action_id2 = SingleTrioCombHandToActionId(hand2);
  SPIEL_CHECK_EQ(FormatSingleHand(SingleTrioCombHand(action_id2)), "33322");

  std::array<int, kNumRanks> current_hand{};
  // 666777TTTQQQ222(BWJ)(CJ)
  current_hand[3] = 3;
  current_hand[4] = 3;

  current_hand[7] = 3;
  current_hand[9] = 3;

  current_hand[12] = 3;
  current_hand[13] = 1;
  current_hand[14] = 1;


  std::cout << "Begin test for search actions" << std::endl;
  std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(current_hand)) << std::endl;

  std::vector<Action> actions1;

  SearchSingleTrioCombActions(actions1, current_hand, /*prev_action=*/action_id1);
  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 18);


  std::vector<Action> actions2;
  SearchSingleTrioCombActions(actions2, current_hand, /*prev_action=*/action_id2);
  SPIEL_CHECK_EQ(static_cast<int>(actions2.size()), 20);



  std::vector<Action> actions3;
  SearchSingleTrioCombActions(actions3, current_hand, kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 50);
  // std::cout << "Possible actions:" << std::endl;
  // for(auto action: actions2){
  //   std::array<int, kNumRanks> possible_hand = SingleTrioCombHand(action);
  //   std::cout << FormatSingleHand(possible_hand) << std::endl;
  // }
}

void AirplaneCombHandTest(){
  std::array<int, kNumRanks> hand1{};

  // 888999TTTJJJQQQ-7772(CJ)

  hand1[5] = 3;
  hand1[6] = 3;
  hand1[7] = 3;
  hand1[8] = 3;
  hand1[9] = 3;

  hand1[4] = 3;
  hand1[12] = 1;
  hand1[14] = 1;
  int action_id1 = AirplaneCombHandToActionId(hand1, /*chain_head=*/5, /*kicker_type=*/kSolo);
  // std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(hand)) << std::endl;
  // std::cout << action_id << std::endl;
  // std::cout << FormatSingleHand(AirplaneCombHand(action_id)) << std::endl;
  SPIEL_CHECK_EQ(FormatSingleHand(AirplaneCombHand(action_id1)), "777888999TTTJJJQQQ2(CJ)");

  

  std::array<int, kNumRanks> hand2{};

  // TTTJJJQQQKKK-33445522
  hand2[7] = 3;
  hand2[8] = 3;
  hand2[9] = 3;
  hand2[10] = 3;


  hand2[0] = 2;
  hand2[1] = 2;
  hand2[2] = 2;
  hand2[12] = 2;
  int action_id2 = AirplaneCombHandToActionId(hand2, /*chain_head=*/7, /*kicker_type=*/kPair);
  // std::cout << "second" << std::endl;
  // std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(hand)) << std::endl;
  // std::cout << action_id << std::endl;
  // std::cout << FormatSingleHand(AirplaneCombHand(action_id)) << std::endl;
  SPIEL_CHECK_EQ(FormatSingleHand(AirplaneCombHand(action_id2)), "334455TTTJJJQQQKKK22");





  std::array<int, kNumRanks> current_hand{};

  // 667899TTTJJJJQQQKKKAAA222(BWJ)(CJ)


  current_hand[3] = 2;
  current_hand[4] = 1;
  current_hand[5] = 1;
  current_hand[6] = 2;
  current_hand[7] = 3;
  current_hand[8] = 4;
  current_hand[9] = 3;
  current_hand[10] = 3;
  current_hand[11] = 3;
  current_hand[12] = 3;
  current_hand[13] = 1;
  current_hand[14] = 1;
  std::vector<Action> actions1;
  std::cout << "Begin test for search actions" << std::endl;
  std::cout << absl::StrFormat("Hands: %s", FormatSingleHand(current_hand)) << std::endl;
  SearchAirplaneCombActions(actions1, current_hand, /*prev_action=*/action_id1);
  // C(7, 5) - C(5, 3) + 3*(C(6, 3) - C(4, 1)) + C(3, 2) * 5 + 2 + C(6, 2) - 1 = 90
  SPIEL_CHECK_EQ(static_cast<int>(actions1.size()), 90);

  std::vector<Action> actions2;
  SearchAirplaneCombActions(actions2, current_hand, /*prev_action=*/action_id2);
  SPIEL_CHECK_EQ(static_cast<int>(actions2.size()), 1);



  std::vector<Action> actions3;
  SearchAirplaneCombActions(actions3, current_hand, /*prev_action=*/kInvalidAction);
  SPIEL_CHECK_EQ(static_cast<int>(actions3.size()), 1052);
  // std::cout << "Possible actions:" << std::endl;
  // for(auto action: actions){
  //   std::array<int, kNumRanks> possible_hand = AirplaneCombHand(action);
  //   std::cout << FormatSingleHand(possible_hand) << std::endl;
  // }
}


} // namespace dou_dizhu
} // namespace open_spiel


int main(){
  open_spiel::dou_dizhu::SingleRankHandTest();
  open_spiel::dou_dizhu::ChainOnlyHandTest();
  open_spiel::dou_dizhu::SingleTrioCombHandTest();
  open_spiel::dou_dizhu::AirplaneCombHandTest();
}