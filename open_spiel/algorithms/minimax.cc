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

#include <algorithm>  // std::max
#include <limits>

#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

// An alpha-beta algorithm.
//
// Implements a min-max algorithm with alpha-beta pruning.
// See for example https://en.wikipedia.org/wiki/Alpha-beta_pruning
//
// Arguments:
//   state: The current state node of the game.
//   depth: The maximum depth for the min/max search.
//   alpha: best value that the MAX player can guarantee (if the value is <=
//     alpha, the MAX player will avoid it).
//   beta: the best value that the MIN currently can guarantee (if the value is
//     >= than beta, the MIN player will avoid it).
//   value_function: An optional functioon mapping a Spiel `State` to a
//     numerical value, to be used as the value for a node when we reach
//     `maximum_depth` and the node is not terminal.
//   maximizing_player_id: The id of the MAX player. The other player is assumed
//     to be MIN.
//
// Returns:
//   The optimal value of the sub-game starting in state (given alpha/beta).
double _alpha_beta(State* state, int depth, double alpha, double beta,
                   std::function<double(const State&)> value_function,
                   Player maximizing_player, Action* best_action) {
  if (state->IsTerminal()) {
    return state->PlayerReturn(maximizing_player);
  }

  if (depth == 0 && !value_function) {
    SpielFatalError(
        "We assume we can walk the full depth of the tree. "
        "Try increasing depth or provide a value_function.");
  }

  if (depth == 0) {
    return value_function(*state);
  }

  Player player = state->CurrentPlayer();
  if (player == maximizing_player) {
    double value = -std::numeric_limits<double>::infinity();

    for (auto action : state->LegalActions()) {
      state->ApplyAction(action);
      double child_value =
          _alpha_beta(state, /*depth=*/depth - 1, /*alpha=*/alpha,
                      /*beta=*/beta, value_function, maximizing_player,
                      /*best_action=*/nullptr);
      state->UndoAction(player, action);

      if (child_value > value) {
        value = child_value;
        if (best_action != nullptr) {
          *best_action = action;
        }
      }

      alpha = std::max(alpha, value);
      if (alpha >= beta) {
        break;  // beta cut-off
      }
    }

    return value;
  } else {
    double value = std::numeric_limits<double>::infinity();

    for (auto action : state->LegalActions()) {
      state->ApplyAction(action);
      double child_value =
          _alpha_beta(state, /*depth=*/depth - 1, /*alpha=*/alpha,
                      /*beta=*/beta, value_function, maximizing_player,
                      /*best_action=*/nullptr);
      state->UndoAction(player, action);

      if (child_value < value) {
        value = child_value;
        if (best_action != nullptr) {
          *best_action = action;
        }
      }

      beta = std::min(beta, value);
      if (alpha >= beta) {
        break;  // alpha cut-off
      }
    }

    return value;
  }
}
}  // namespace

std::pair<double, Action> AlphaBetaSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player) {
  if (game.NumPlayers() != 2) {
    SpielFatalError("Game must be a 2-player game");
  }
  GameType game_info = game.GetType();
  if (game_info.chance_mode != GameType::ChanceMode::kDeterministic) {
    SpielFatalError(absl::StrCat("The game must be a Deterministic one, not ",
                                 game_info.chance_mode));
  }
  if (game_info.information != GameType::Information::kPerfectInformation) {
    SpielFatalError(
        absl::StrCat("The game must be a perfect information one, not ",
                     game_info.information));
  }
  if (game_info.dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        absl::StrCat("The game must be turn-based, not ", game_info.dynamics));
  }
  if (game_info.utility != GameType::Utility::kZeroSum) {
    SpielFatalError(
        absl::StrCat("The game must be 0-sum, not  ", game_info.utility));
  }

  std::unique_ptr<State> search_root;
  if (state == nullptr) {
    search_root = game.NewInitialState();
  } else {
    search_root = state->Clone();
  }

  if (maximizing_player == kInvalidPlayer) {
    maximizing_player = search_root->CurrentPlayer();
  }

  double infinity = std::numeric_limits<double>::infinity();
  Action best_action = kInvalidAction;
  double value = _alpha_beta(
      search_root.get(), /*depth=*/depth_limit, /*alpha=*/-infinity,
      /*beta=*/infinity, value_function, maximizing_player, &best_action);

  return std::pair<double, Action>(value, best_action);
}

}  // namespace algorithms
}  // namespace open_spiel
