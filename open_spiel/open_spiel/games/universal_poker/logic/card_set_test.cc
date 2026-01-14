// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/universal_poker/logic/card_set.h"

#include <iostream>
#include <limits>
#include <map>
#include <string>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

void BasicCardSetTests() {
  CardSet cs("AhKsQhJhTh");

  std::cout << "CardSet: " << cs.ToString() << std::endl;
  for (auto card : cs.ToCardArray()) {
    std::cout << "Card: " << card << std::endl;
  }
  std::cout << "Rank: " << cs.RankCards() << std::endl;
  std::cout << "Count Cards: " << cs.NumCards() << std::endl;

  CardSet deck(4, 13);
  std::cout << "CardSet: " << deck.ToString() << std::endl;
  std::cout << "Rank: " << deck.RankCards() << std::endl;
  std::cout << "Count Cards: " << deck.NumCards() << std::endl;

  for (auto combo : deck.Combinations(3)) {
    std::cout << "CardSet: " << combo.ToString() << std::endl;
  }

  for (auto combo : deck.Combinations(1)) {
    std::cout << "CardSet: " << combo.ToString() << std::endl;
  }
}

void BasicHandRankTests() {
  CardSet cs("AhKhQhJhTh");
  SPIEL_CHECK_EQ(cs.GetHandRank(), kStraightFlush);
  CardSet cs2("AhAsAc2d2c");
  SPIEL_CHECK_EQ(cs2.GetHandRank(), kFullHouse);
  CardSet cs3("2h3h4h5h7h");
  SPIEL_CHECK_EQ(cs3.GetHandRank(), kFlush);
  CardSet cs4("2c3d4h5s6c");
  SPIEL_CHECK_EQ(cs4.GetHandRank(), kStraight);
  CardSet cs5("AhAdAsAc2d");
  SPIEL_CHECK_EQ(cs5.GetHandRank(), kFourOfAKind);
  CardSet cs6("AhAdAs2c3d");
  SPIEL_CHECK_EQ(cs6.GetHandRank(), kThreeOfAKind);
  CardSet cs7("AhAd2c2d3h");
  SPIEL_CHECK_EQ(cs7.GetHandRank(), kTwoPair);
  CardSet cs8("AhAd2c3d4h");
  SPIEL_CHECK_EQ(cs8.GetHandRank(), kPair);
  CardSet cs9("AhKdQcJs2h");
  SPIEL_CHECK_EQ(cs9.GetHandRank(), kHighCard);
  CardSet cs10("Ah2h3h4h5h");
  SPIEL_CHECK_EQ(cs10.GetHandRank(), kStraightFlush);

  std::cout << "CardSet: " << cs.ToString() << std::endl;
  std::cout << "Rank: " << cs.RankCards() << std::endl;
  std::cout << "HandRank: " << cs.GetHandRank() << std::endl;
}

void BoundaryTest() {
  std::cout << "\n--- Rank Boundary Test ---\n";
  open_spiel::universal_poker::logic::CardSet deck(4, 13);
  std::map<open_spiel::universal_poker::logic::HandRankType, int> min_r;
  std::map<open_spiel::universal_poker::logic::HandRankType, int> max_r;
  std::map<int, std::string> hand_str_for_rank;

  const open_spiel::universal_poker::logic::HandRankType types[] = {
      open_spiel::universal_poker::logic::kHighCard,
      open_spiel::universal_poker::logic::kPair,
      open_spiel::universal_poker::logic::kTwoPair,
      open_spiel::universal_poker::logic::kThreeOfAKind,
      open_spiel::universal_poker::logic::kStraight,
      open_spiel::universal_poker::logic::kFlush,
      open_spiel::universal_poker::logic::kFullHouse,
      open_spiel::universal_poker::logic::kFourOfAKind,
      open_spiel::universal_poker::logic::kStraightFlush};

  for (auto type : types) {
    min_r[type] = std::numeric_limits<int>::max();
    max_r[type] = std::numeric_limits<int>::min();
  }

  for (const auto& hand : deck.Combinations(5)) {
    int rank = hand.RankCards();
    auto type = hand.GetHandRank();
    if (rank < min_r[type]) {
      min_r[type] = rank;
      hand_str_for_rank[rank] = hand.ToString();
    }
    if (rank > max_r[type]) {
      max_r[type] = rank;
      hand_str_for_rank[rank] = hand.ToString();
    }
  }

  auto print_rank = [&](int rank) {
    std::cout << "  Rank: " << rank
              << ", Hand: " << hand_str_for_rank[rank] << ", Type: "
              << open_spiel::universal_poker::logic::HandRankToString(
                     open_spiel::universal_poker::logic::CardSet(
                         hand_str_for_rank[rank])
                         .GetHandRank())
              << "\n";
  };

  std::cout << "HighCard MAX vs Pair MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kHighCard]);
  print_rank(min_r[open_spiel::universal_poker::logic::kPair]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kHighCard],
                 min_r[open_spiel::universal_poker::logic::kPair]);

  std::cout << "Pair MAX vs TwoPair MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kPair]);
  print_rank(min_r[open_spiel::universal_poker::logic::kTwoPair]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kPair],
                 min_r[open_spiel::universal_poker::logic::kTwoPair]);

  std::cout << "TwoPair MAX vs ThreeOfAKind MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kTwoPair]);
  print_rank(min_r[open_spiel::universal_poker::logic::kThreeOfAKind]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kTwoPair],
                 min_r[open_spiel::universal_poker::logic::kThreeOfAKind]);

  std::cout << "ThreeOfAKind MAX vs Straight MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kThreeOfAKind]);
  print_rank(min_r[open_spiel::universal_poker::logic::kStraight]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kThreeOfAKind],
                 min_r[open_spiel::universal_poker::logic::kStraight]);

  std::cout << "Straight MAX vs Flush MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kStraight]);
  print_rank(min_r[open_spiel::universal_poker::logic::kFlush]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kStraight],
                 min_r[open_spiel::universal_poker::logic::kFlush]);

  std::cout << "Flush MAX vs FullHouse MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kFlush]);
  print_rank(min_r[open_spiel::universal_poker::logic::kFullHouse]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kFlush],
                 min_r[open_spiel::universal_poker::logic::kFullHouse]);

  std::cout << "FullHouse MAX vs FourOfAKind MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kFullHouse]);
  print_rank(min_r[open_spiel::universal_poker::logic::kFourOfAKind]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kFullHouse],
                 min_r[open_spiel::universal_poker::logic::kFourOfAKind]);

  std::cout << "FourOfAKind MAX vs StraightFlush MIN:\n";
  print_rank(max_r[open_spiel::universal_poker::logic::kFourOfAKind]);
  print_rank(min_r[open_spiel::universal_poker::logic::kStraightFlush]);
  SPIEL_CHECK_LT(max_r[open_spiel::universal_poker::logic::kFourOfAKind],
                 min_r[open_spiel::universal_poker::logic::kStraightFlush]);
  std::cout << "--- Rank Boundary Test End ---\n";
}

void HandRankToStringTests() {
  SPIEL_CHECK_EQ(HandRankToString(kStraightFlush), "Straight Flush");
  SPIEL_CHECK_EQ(HandRankToString(kFourOfAKind), "Four of a Kind");
  SPIEL_CHECK_EQ(HandRankToString(kFullHouse), "Full House");
  SPIEL_CHECK_EQ(HandRankToString(kFlush), "Flush");
  SPIEL_CHECK_EQ(HandRankToString(kStraight), "Straight");
  SPIEL_CHECK_EQ(HandRankToString(kThreeOfAKind), "Three of a Kind");
  SPIEL_CHECK_EQ(HandRankToString(kTwoPair), "Two Pair");
  SPIEL_CHECK_EQ(HandRankToString(kPair), "Pair");
  SPIEL_CHECK_EQ(HandRankToString(kHighCard), "High Card");
}

void GetBest5CardsTests() {
  CardSet seven_cards;
  seven_cards.AddCard(CardSet("Ah").ToCardArray()[0]);
  CardSet best_set = seven_cards.GetBest5Cards();
  CardSet expected("Ah");
  SPIEL_CHECK_EQ(best_set.RankCards(), expected.RankCards());
  seven_cards.AddCard(CardSet("Kh").ToCardArray()[0]);
  best_set = seven_cards.GetBest5Cards();
  expected = CardSet("AhKh");
  SPIEL_CHECK_EQ(best_set.RankCards(), expected.RankCards());
  seven_cards.AddCard(CardSet("Qh").ToCardArray()[0]);
  seven_cards.AddCard(CardSet("Jh").ToCardArray()[0]);
  seven_cards.AddCard(CardSet("Th").ToCardArray()[0]);
  seven_cards.AddCard(CardSet("Ac").ToCardArray()[0]);
  seven_cards.AddCard(CardSet("Kc").ToCardArray()[0]);
  best_set = seven_cards.GetBest5Cards();
  expected = CardSet("AhKhQhJhTh");
  SPIEL_CHECK_EQ(best_set.RankCards(), expected.RankCards());

  CardSet flush7;
  flush7.AddCard(CardSet("2h").ToCardArray()[0]);
  flush7.AddCard(CardSet("3h").ToCardArray()[0]);
  flush7.AddCard(CardSet("4h").ToCardArray()[0]);
  flush7.AddCard(CardSet("5h").ToCardArray()[0]);
  flush7.AddCard(CardSet("7h").ToCardArray()[0]);
  flush7.AddCard(CardSet("Ac").ToCardArray()[0]);
  flush7.AddCard(CardSet("Kc").ToCardArray()[0]);
  CardSet best_flush_hand = flush7.GetBest5Cards();
  CardSet expected_flush("7h5h4h3h2h");
  SPIEL_CHECK_EQ(best_flush_hand.RankCards(), expected_flush.RankCards());
}

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::logic::BasicCardSetTests();
  open_spiel::universal_poker::logic::HandRankToStringTests();
  open_spiel::universal_poker::logic::GetBest5CardsTests();
  open_spiel::universal_poker::logic::BoundaryTest();
}
