// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/algorithms/get_legal_actions_map.h"

#include <unordered_map>

#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel_utils.h"

namespace algorithms = open_spiel::algorithms;
namespace kuhn_poker = open_spiel::kuhn_poker;
namespace leduc_poker = open_spiel::leduc_poker;

using LegalActionsMap =
    std::unordered_map<std::string, std::vector<open_spiel::Action>>;

namespace {
void KuhnTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("kuhn_poker");

  LegalActionsMap map_p0 =
      algorithms::GetLegalActionsMap(*game,
                                     /*depth_limit=*/-1, open_spiel::Player{0});
  SPIEL_CHECK_EQ(map_p0.size(), kuhn_poker::kNumInfoStatesP0);

  LegalActionsMap map_p1 =
      algorithms::GetLegalActionsMap(*game,
                                     /*depth_limit=*/-1, open_spiel::Player{1});
  SPIEL_CHECK_EQ(map_p1.size(), kuhn_poker::kNumInfoStatesP1);

  LegalActionsMap map_both = algorithms::GetLegalActionsMap(
      *game, /*depth_limit=*/-1, open_spiel::kInvalidPlayer);
  SPIEL_CHECK_EQ(map_both.size(),
                 kuhn_poker::kNumInfoStatesP0 + kuhn_poker::kNumInfoStatesP1);
  // They should all have two legal actions: pass and bet.
  for (const auto& legal_actions : map_both) {
    SPIEL_CHECK_EQ(legal_actions.second.size(), 2);
  }
}

void LeducTest() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("leduc_poker");
  LegalActionsMap map_both = algorithms::GetLegalActionsMap(
      *game, /*depth_limit=*/-1, open_spiel::kInvalidPlayer);
  SPIEL_CHECK_EQ(map_both.size(), leduc_poker::kNumInfoStates);
}

void GoofspielTest() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(
      "goofspiel", {{"num_cards", open_spiel::GameParameter(3)}});
  LegalActionsMap map_both = algorithms::GetLegalActionsMap(
      *game, /*depth_limit=*/-1, open_spiel::kInvalidPlayer);
  SPIEL_CHECK_GT(map_both.size(), 0);
}

}  // namespace

int main(int argc, char** argv) {
  KuhnTest();
  LeducTest();
  GoofspielTest();
}
