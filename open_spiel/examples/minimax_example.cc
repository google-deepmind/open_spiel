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

#include "open_spiel/algorithms/minimax.h"

#include <memory>

#include "open_spiel/games/breakthrough.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

using open_spiel::breakthrough::BreakthroughState;

inline constexpr int kSearchDepth = 2;

namespace open_spiel {
namespace {

int BlackPieceAdvantage(const State& state) {
  const auto& bstate = down_cast<const BreakthroughState&>(state);
  return bstate.pieces(breakthrough::kBlackPlayerId) -
         bstate.pieces(breakthrough::kWhitePlayerId);
}

void PlayGame() {
  std::shared_ptr<const Game> game =
      LoadGame("breakthrough", {{"rows", GameParameter(6)},
                                {"columns", GameParameter(6)}});
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    std::cout << std::endl << state->ToString() << std::endl;

    Player player = state->CurrentPlayer();
    std::pair<double, Action> value_action = algorithms::AlphaBetaSearch(
        *game, state.get(), [player](const State& state) {
            return (player == breakthrough::kBlackPlayerId ?
                    BlackPieceAdvantage(state) :
                    -BlackPieceAdvantage(state));
            },
        kSearchDepth, player);

    std::cout << std::endl << "Player " << player << " choosing action "
              << state->ActionToString(player, value_action.second)
              << " with heuristic value (to black) " << value_action.first
              << std::endl;

    state->ApplyAction(value_action.second);
  }

  std::cout << "Terminal state: " << std::endl;
  std::cout << state->ToString() << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::PlayGame();
}
