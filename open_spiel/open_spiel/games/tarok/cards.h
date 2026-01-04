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

#ifndef OPEN_SPIEL_GAMES_TAROK_CARDS_H_
#define OPEN_SPIEL_GAMES_TAROK_CARDS_H_

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace tarok {

// a subset of card actions that are used throughout the codebase and add to
// readability, for more info see TarokState::LegalActions()
inline constexpr int kPagatAction = 0;
inline constexpr int kMondAction = 20;
inline constexpr int kSkisAction = 21;
inline constexpr int kKingOfHeartsAction = 29;
inline constexpr int kKingOfDiamondsAction = 37;
inline constexpr int kKingOfSpadesAction = 45;
inline constexpr int kKingOfClubsAction = 53;

enum class CardSuit { kHearts, kDiamonds, kSpades, kClubs, kTaroks };

struct Card {
  Card(CardSuit suit, int rank, int points, std::string short_name,
       std::string long_name);

  const std::string ToString() const;

  const CardSuit suit;
  const int rank;
  const int points;
  const std::string short_name;
  const std::string long_name;
};

const std::array<Card, 54> InitializeCardDeck();

// a type for a pair holding talon and players' private cards
using DealtCards =
    std::tuple<std::vector<Action>, std::vector<std::vector<Action>>>;
DealtCards DealCards(int num_players, int seed);

// we use our own implementation since std::shuffle is non-deterministic across
// different versions of the standard library implementation
void Shuffle(std::vector<Action>* actions, std::mt19937&& rng);

int CardPoints(const std::vector<Action>& actions,
               const std::array<Card, 54>& deck);

}  // namespace tarok
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TAROK_CARDS_H_
