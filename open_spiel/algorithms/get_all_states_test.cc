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

#include "open_spiel/algorithms/get_all_states.h"

#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel_utils.h"

namespace algorithms = open_spiel::algorithms;
namespace ttt = open_spiel::tic_tac_toe;

int main(int argc, char **argv) {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("tic_tac_toe");
  auto states = algorithms::GetAllStates(*game, -1, /*include_terminals=*/true,
                                         /*include_chance_states=*/true);
  SPIEL_CHECK_EQ(states.size(), ttt::kNumberStates);
}
