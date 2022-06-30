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

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"

#include <iostream>

namespace open_spiel {
namespace universal_poker {
namespace acpc_cpp {

void BasicACPCTests() {
  const std::string gameDesc(
      "GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 2\nstack = 1200 "
      "1200\nblind = 100 100\nfirstPlayer = 1 1\nnumSuits = 2\nnumRanks = "
      "3\nnumHoleCards = 1\nnumBoardCards = 0 1\nEND GAMEDEF");

  ACPCGame game(gameDesc);
  ACPCState state(&game);

  std::cout << game.ToString() << std::endl;
  std::cout << state.ToString() << std::endl;

  while (!state.IsFinished()) {
    int32_t minRaise = 0, maxRaise = 0;
    if (state.RaiseIsValid(&minRaise, &maxRaise)) {
      minRaise = state.MaxSpend() > minRaise ? state.MaxSpend() : minRaise;
    }

    const ACPCState::ACPCActionType available_actions[] = {
        ACPCState::ACPC_CALL, ACPCState::ACPC_FOLD,
        // ACPCState::ACPC_RAISE
    };

    for (const auto &action : available_actions) {
      if (state.IsValidAction(action, 0)) {
        state.DoAction(action, 0);
        std::cout << state.ToString() << std::endl;
      }
    }
  }

  std::cout << state.ValueOfState(0) << std::endl;
  std::cout << state.ValueOfState(1) << std::endl;

  SPIEL_CHECK_EQ(game.TotalMoney(), 2400);
}

}  // namespace acpc_cpp
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::acpc_cpp::BasicACPCTests();
}
