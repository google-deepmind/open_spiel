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

#include "open_spiel/games/tarok/cards.h"

#include <algorithm>
#include <random>
#include <utility>

namespace open_spiel {
namespace tarok {

Card::Card(CardSuit suit, int rank, int points, std::string short_name,
           std::string long_name)
    : suit(suit),
      rank(rank),
      points(points),
      short_name(short_name),
      long_name(long_name) {}

const std::string Card::ToString() const { return long_name; }

const std::array<Card, 54> InitializeCardDeck() {
  return {// taroks
          Card(CardSuit::kTaroks, 8, 5, "T1", "Pagat"),
          Card(CardSuit::kTaroks, 9, 1, "T2", "II"),
          Card(CardSuit::kTaroks, 10, 1, "T3", "III"),
          Card(CardSuit::kTaroks, 11, 1, "T4", "IIII"),
          Card(CardSuit::kTaroks, 12, 1, "T5", "V"),
          Card(CardSuit::kTaroks, 13, 1, "T6", "VI"),
          Card(CardSuit::kTaroks, 14, 1, "T7", "VII"),
          Card(CardSuit::kTaroks, 15, 1, "T8", "VIII"),
          Card(CardSuit::kTaroks, 16, 1, "T9", "IX"),
          Card(CardSuit::kTaroks, 17, 1, "T10", "X"),
          Card(CardSuit::kTaroks, 18, 1, "T11", "XI"),
          Card(CardSuit::kTaroks, 19, 1, "T12", "XII"),
          Card(CardSuit::kTaroks, 20, 1, "T13", "XIII"),
          Card(CardSuit::kTaroks, 21, 1, "T14", "XIV"),
          Card(CardSuit::kTaroks, 22, 1, "T15", "XV"),
          Card(CardSuit::kTaroks, 23, 1, "T16", "XVI"),
          Card(CardSuit::kTaroks, 24, 1, "T17", "XVII"),
          Card(CardSuit::kTaroks, 25, 1, "T18", "XVIII"),
          Card(CardSuit::kTaroks, 26, 1, "T19", "XIX"),
          Card(CardSuit::kTaroks, 27, 1, "T20", "XX"),
          Card(CardSuit::kTaroks, 28, 5, "T21", "Mond"),
          Card(CardSuit::kTaroks, 29, 5, "T22", "Skis"),
          // hearts
          Card(CardSuit::kHearts, 0, 1, "H4", "4 of Hearts"),
          Card(CardSuit::kHearts, 1, 1, "H3", "3 of Hearts"),
          Card(CardSuit::kHearts, 2, 1, "H2", "2 of Hearts"),
          Card(CardSuit::kHearts, 3, 1, "H1", "1 of Hearts"),
          Card(CardSuit::kHearts, 4, 2, "HJ", "Jack of Hearts"),
          Card(CardSuit::kHearts, 5, 3, "HKN", "Knight of Hearts"),
          Card(CardSuit::kHearts, 6, 4, "HQ", "Queen of Hearts"),
          Card(CardSuit::kHearts, 7, 5, "HKI", "King of Hearts"),
          // diamonds
          Card(CardSuit::kDiamonds, 0, 1, "D4", "4 of Diamonds"),
          Card(CardSuit::kDiamonds, 1, 1, "D3", "3 of Diamonds"),
          Card(CardSuit::kDiamonds, 2, 1, "D2", "2 of Diamonds"),
          Card(CardSuit::kDiamonds, 3, 1, "D1", "1 of Diamonds"),
          Card(CardSuit::kDiamonds, 4, 2, "DJ", "Jack of Diamonds"),
          Card(CardSuit::kDiamonds, 5, 3, "DKN", "Knight of Diamonds"),
          Card(CardSuit::kDiamonds, 6, 4, "DQ", "Queen of Diamonds"),
          Card(CardSuit::kDiamonds, 7, 5, "DKI", "King of Diamonds"),
          // spades
          Card(CardSuit::kSpades, 0, 1, "S7", "7 of Spades"),
          Card(CardSuit::kSpades, 1, 1, "S8", "8 of Spades"),
          Card(CardSuit::kSpades, 2, 1, "S9", "9 of Spades"),
          Card(CardSuit::kSpades, 3, 1, "S10", "10 of Spades"),
          Card(CardSuit::kSpades, 4, 2, "SJ", "Jack of Spades"),
          Card(CardSuit::kSpades, 5, 3, "SKN", "Knight of Spades"),
          Card(CardSuit::kSpades, 6, 4, "SQ", "Queen of Spades"),
          Card(CardSuit::kSpades, 7, 5, "SKI", "King of Spades"),
          // clubs
          Card(CardSuit::kClubs, 0, 1, "C7", "7 of Clubs"),
          Card(CardSuit::kClubs, 1, 1, "C8", "8 of Clubs"),
          Card(CardSuit::kClubs, 2, 1, "C9", "9 of Clubs"),
          Card(CardSuit::kClubs, 3, 1, "C10", "10 of Clubs"),
          Card(CardSuit::kClubs, 4, 2, "CJ", "Jack of Clubs"),
          Card(CardSuit::kClubs, 5, 3, "CKN", "Knight of Clubs"),
          Card(CardSuit::kClubs, 6, 4, "CQ", "Queen of Clubs"),
          Card(CardSuit::kClubs, 7, 5, "CKI", "King of Clubs")};
}

DealtCards DealCards(int num_players, int seed) {
  std::vector<Action> cards(54);
  std::iota(cards.begin(), cards.end(), 0);
  Shuffle(&cards, std::mt19937(seed));

  // first six cards are talon
  auto begin = cards.begin();
  auto end = begin + 6;
  std::vector<Action> talon(begin, end);

  // deal the rest of the cards to players
  int num_cards_per_player = 48 / num_players;
  std::vector<std::vector<Action>> players_cards;
  players_cards.reserve(num_players);

  std::advance(begin, 6);
  for (int i = 0; i < num_players; i++) {
    std::advance(end, num_cards_per_player);
    std::vector<Action> player_cards(begin, end);
    // player's cards are sorted since legal actions need to be returned in
    // ascending order
    std::sort(player_cards.begin(), player_cards.end());
    players_cards.push_back(player_cards);
    std::advance(begin, num_cards_per_player);
  }

  return {talon, players_cards};
}

void Shuffle(std::vector<Action>* actions, std::mt19937&& rng) {
  for (int i = actions->size() - 1; i > 0; i--) {
    std::swap(actions->at(i), actions->at(rng() % (i + 1)));
  }
}

int CardPoints(const std::vector<Action>& actions,
               const std::array<Card, 54>& deck) {
  // counting is done in batches of three (for every batch we sum up points from
  // three cards and subtract 2 points, if the last batch has less than three
  // cards we subtract 1 point), mathematically, this is equevalent to
  // subtracting 2/3 from each card
  float points = 0;
  for (auto const& action : actions) {
    points += deck.at(action).points;
  }
  points -= actions.size() * 0.666f;
  return static_cast<int>(round(points));
}

}  // namespace tarok
}  // namespace open_spiel
