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

#include "open_spiel/games/crossword/crossword_lib.h"

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/games/crossword/crossword.h"
#include "open_spiel/games/crossword/crossword_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/status.h"

namespace open_spiel {
namespace crossword {

void SimulateRandomGame(std::shared_ptr<const open_spiel::Game> game,
                        int seed) {
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<ActionStructSampler> sampler =
      state->GetActionStructSampler(/*seed=*/seed);
  std::mt19937 rng(seed);

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      std::vector<std::pair<open_spiel::Action, double>> outcomes =
          state->ChanceOutcomes();
      open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
      std::cerr << "sampled outcome: "
                << state->ActionToString(open_spiel::kChancePlayerId, action)
                << std::endl;
      state->ApplyAction(action);
    } else {
      std::unique_ptr<ActionStruct> action = sampler->SampleActionStruct();
      if (action == nullptr) {
        SpielFatalError("SimulateRandomGame: failed to sample action.");
      }
      auto* crossword_action =
          down_cast<CrosswordActionStruct*>(action.get());
      std::cout << "Applying action: " << crossword_action->ToString() << "\n";
      Status status = state->ApplyActionStruct(*action);
      if (!status.ok()) {
        SpielFatalError("SimulateRandomGame: failed to apply action: " +
                        status.ToString());
      }
      std::cout << "State: \n" << state->ToString() << "\n";
    }
  }
}

void SimulateWinningGame(std::shared_ptr<const open_spiel::Game> game,
                         int seed) {
  std::unique_ptr<State> state = game->NewInitialState();
  auto* crossword_state = down_cast<CrosswordState*>(state.get());
  const std::vector<int>& clue_solved = crossword_state->clue_solved();
  std::mt19937 rng(seed);
  double util_return = 0.0;

  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      std::vector<std::pair<open_spiel::Action, double>> outcomes =
          state->ChanceOutcomes();
      open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
      std::cerr << "sampled outcome: "
                << state->ActionToString(open_spiel::kChancePlayerId, action)
                << std::endl;
      state->ApplyAction(action);
    } else {
      std::vector<int> unsolved_clue_indices;
      for (int i = 0; i < clue_solved.size(); ++i) {
        if (clue_solved[i] == 0) {
          unsolved_clue_indices.push_back(i);
        }
      }
      int sampled_idx = absl::Uniform<int>(rng, 0,
                                           unsolved_clue_indices.size());
      int clue_index = unsolved_clue_indices[sampled_idx];
      const Clue& clue = crossword_state->board().clue(clue_index);
      std::string cid = ClueId(clue);
      std::string answer = crossword_state->board().answer(cid);
      CrosswordActionStruct action = CrosswordActionStruct(cid, answer);
      Status status = state->ValidateActionStruct(action);
      if (!status.ok()) {
        SpielFatalError("SimulateWinningGame: failed to validate action: " +
                        status.ToString());
      }
      status = state->ApplyActionStruct(action);
      if (!status.ok()) {
        SpielFatalError("SimulateRandomGame: failed to apply action: " +
                        status.ToString());
      }
      SPIEL_CHECK_GT(state->Rewards()[0], 0);
      SPIEL_CHECK_GT(state->Returns()[0], util_return);
      util_return = state->Returns()[0];
      std::cout << "State: \n" << state->ToString() << "\n";
    }
  }
}

}  // namespace crossword
}  // namespace open_spiel
