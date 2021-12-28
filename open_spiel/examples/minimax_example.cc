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

#include <memory>

#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/breakthrough.h"
#include "open_spiel/games/pig.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

inline constexpr int kSearchDepth = 2;
inline constexpr int kSearchDepthPig = 10;
inline constexpr int kWinscorePig = 30;
inline constexpr int kDiceoutcomesPig = 2;
inline constexpr int kSeed = 726345721;

namespace open_spiel {
namespace {

int BlackPieceAdvantage(const State& state) {
  const auto& bstate = down_cast<const breakthrough::BreakthroughState&>(state);
  return bstate.pieces(breakthrough::kBlackPlayerId) -
         bstate.pieces(breakthrough::kWhitePlayerId);
}

void PlayBreakthrough() {
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

int FirstPlayerAdvantage(const State& state) {
  const auto& pstate = down_cast<const pig::PigState&>(state);
  return pstate.score(0) - pstate.score(1);
}

void PlayPig(std::mt19937& rng) {
  std::shared_ptr<const Game> game =
      LoadGame("pig", {{"winscore", GameParameter(kWinscorePig)},
                       {"diceoutcomes", GameParameter(kDiceoutcomesPig)}});
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    std::cout << std::endl << state->ToString() << std::endl;

    Player player = state->CurrentPlayer();
    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      ActionsAndProbs outcomes = state->ChanceOutcomes();
      Action action = open_spiel::SampleAction(outcomes, rng).first;
      std::cerr << "Sampled action: " << state->ActionToString(player, action)
                << std::endl;
      state->ApplyAction(action);
    } else {
      std::pair<double, Action> value_action = algorithms::ExpectiminimaxSearch(
          *game, state.get(),
          [player](const State& state) {
            return (player == Player{0} ? FirstPlayerAdvantage(state)
                                        : -FirstPlayerAdvantage(state));
          },
          kSearchDepthPig, player);

      std::cout << std::endl
                << "Player " << player << " choosing action "
                << state->ActionToString(player, value_action.second)
                << " with heuristic value " << value_action.first << std::endl;

      state->ApplyAction(value_action.second);
    }
  }

  std::cout << "Terminal state: " << std::endl;
  std::cout << state->ToString() << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  std::mt19937 rng(kSeed);  // Random number generator.
  open_spiel::PlayBreakthrough();
  open_spiel::PlayPig(rng);
}
