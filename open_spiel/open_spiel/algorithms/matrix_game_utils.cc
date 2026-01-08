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

#include "open_spiel/algorithms/matrix_game_utils.h"

#include <memory>

#include "open_spiel/algorithms/deterministic_policy.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

using open_spiel::matrix_game::MatrixGame;

std::shared_ptr<const MatrixGame> LoadMatrixGame(const std::string& name) {
  std::shared_ptr<const Game> game = LoadGame(name);
  // Make sure it is indeed a matrix game.
  const MatrixGame* matrix_game = dynamic_cast<const MatrixGame*>(game.get());
  if (matrix_game == nullptr) {
    // If it is not already a matrix game, check if it is a 2-player NFG.
    // If so, convert it.
    const NormalFormGame* nfg = dynamic_cast<const NormalFormGame*>(game.get());
    if (nfg != nullptr && nfg->NumPlayers() == 2) {
      return AsMatrixGame(nfg);
    } else {
      SpielFatalError(absl::StrCat("Cannot load ", name, " as a matrix game."));
    }
  }
  return std::static_pointer_cast<const MatrixGame>(game);
}

std::shared_ptr<const MatrixGame> AsMatrixGame(const Game* game) {
  const NormalFormGame* nfg = dynamic_cast<const NormalFormGame*>(game);
  SPIEL_CHECK_TRUE(nfg != nullptr);
  return AsMatrixGame(nfg);
}

std::shared_ptr<const MatrixGame> AsMatrixGame(const NormalFormGame* game) {
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  std::unique_ptr<State> initial_state = game->NewInitialState();
  std::vector<std::vector<Action>> legal_actions = {
      initial_state->LegalActions(0), initial_state->LegalActions(1)};

  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  std::vector<double> row_utils;
  std::vector<double> col_utils;
  int num_rows = legal_actions[0].size();
  int num_cols = legal_actions[1].size();

  GameType type = game->GetType();
  type.min_num_players = 2;
  type.max_num_players = 2;

  for (int r = 0; r < num_rows; ++r) {
    Action row_action = legal_actions[0][r];
    row_names.push_back(initial_state->ActionToString(0, row_action));

    for (int c = 0; c < num_cols; ++c) {
      Action col_action = legal_actions[1][c];
      if (col_names.size() < num_cols) {
        col_names.push_back(initial_state->ActionToString(1, col_action));
      }

      std::unique_ptr<State> clone = initial_state->Clone();
      clone->ApplyActions({row_action, col_action});
      SPIEL_CHECK_TRUE(clone->IsTerminal());
      std::vector<double> returns = clone->Returns();
      SPIEL_CHECK_EQ(returns.size(), 2);

      row_utils.push_back(returns[0]);
      col_utils.push_back(returns[1]);
    }
  }

  return std::shared_ptr<MatrixGame>(
      new MatrixGame(type, {}, row_names, col_names, row_utils, col_utils));
}

std::shared_ptr<const MatrixGame> ExtensiveToMatrixGame(const Game& game) {
  SPIEL_CHECK_EQ(game.NumPlayers(), 2);

  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  std::vector<std::vector<double>> row_player_utils;
  std::vector<std::vector<double>> col_player_utils;

  GameType type = game.GetType();

  std::vector<DeterministicTabularPolicy> policies = {
      DeterministicTabularPolicy(game, 0), DeterministicTabularPolicy(game, 1)};

  bool first_row = true;
  do {
    policies[1].ResetDefaultPolicy();
    row_names.push_back(policies[0].ToString(" --- "));
    std::vector<double> row_utils;
    std::vector<double> col_utils;
    do {
      if (first_row) {
        col_names.push_back(policies[1].ToString(" --- "));
      }
      std::unique_ptr<State> state = game.NewInitialState();
      std::vector<double> returns =
          ExpectedReturns(*state, {&policies[0], &policies[1]}, -1);
      row_utils.push_back(returns[0]);
      col_utils.push_back(returns[1]);
    } while (policies[1].NextPolicy());
    first_row = false;
    row_player_utils.push_back(row_utils);
    col_player_utils.push_back(col_utils);
  } while (policies[0].NextPolicy());

  return matrix_game::CreateMatrixGame(type.short_name, type.long_name,
                                       row_names, col_names, row_player_utils,
                                       col_player_utils);
}

}  // namespace algorithms
}  // namespace open_spiel
