// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/parcheesi.h"

#include <algorithm>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace parcheesi {
namespace {

namespace testing = open_spiel::testing;

bool ActionsContains(const std::vector<Action>& legal_actions, Action action) {
  return std::find(legal_actions.begin(), legal_actions.end(), action) !=
         legal_actions.end();
}

void SpielMoveParcheesiMoveEncodingDecodingTest() {
  std::shared_ptr<const Game> game = LoadGame("parcheesi");
  std::unique_ptr<State> state = game->NewInitialState();
  ParcheesiState* pstate = static_cast<ParcheesiState*>(state.get());
  
  pstate->SetPlayer(0);
  ParcheesiMove parcheesiMove = ParcheesiMove(0, -1, 0, 3, false);
  Action spielMove = pstate->ParcheesiMoveToSpielMove(parcheesiMove);
  ParcheesiMove decodedParcheesiMove = pstate->SpielMoveToParcheesiMove(spielMove);

  SPIEL_CHECK_EQ(parcheesiMove.die_index, decodedParcheesiMove.die_index);
  SPIEL_CHECK_EQ(parcheesiMove.old_pos, decodedParcheesiMove.old_pos);
  SPIEL_CHECK_EQ(parcheesiMove.new_pos, decodedParcheesiMove.new_pos);
  SPIEL_CHECK_EQ(parcheesiMove.token_index, decodedParcheesiMove.token_index);
  SPIEL_CHECK_EQ(parcheesiMove.breaking_block, decodedParcheesiMove.breaking_block);
}

}  // namespace
}  // namespace parcheesi
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("parcheesi");
  open_spiel::parcheesi::SpielMoveParcheesiMoveEncodingDecodingTest();
}
