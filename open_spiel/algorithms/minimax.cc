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

#include "open_spiel/algorithms/minimax.h"

#include <algorithm>  // std::find, std::max
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

Action BestActions::Single() const {
  return this->actions.empty() ? kInvalidAction : this->actions[0];
}

Action BestActions::SampleUniformly(std::mt19937& rng) const {
  if (this->actions.empty()) {
    return kInvalidAction;
  }

  std::uniform_int_distribution<int> dist(0, this->actions.size() - 1);
  return this->actions[dist(rng)];
}

size_t BestActions::Size() const { return this->actions.size(); }

bool BestActions::ContainsAction(Action action) const {
  return std::find(this->actions.begin(), this->actions.end(), action) !=
         this->actions.end();
}

bool BestActions::Equals(const std::vector<Action>& action_list) const {
  if (this->actions.size() != action_list.size()) {
    return false;
  }

  for (Action action : action_list) {
    if (!this->ContainsAction(action)) {
      return false;
    }
  }
  return true;
}

void BestActions::Clear() { this->actions.clear(); }

void BestActions::Add(Action action) { this->actions.push_back(action); }

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
//   value_function: An optional function mapping a Spiel `State` to a
//     numerical value, to be used as the value for a node when we reach
//     `depth_limit` and the node is not terminal.
//   maximizing_player_id: The id of the MAX player. The other player is assumed
//     to be MIN.
//   use_undo: use the State::Undo for faster run-time.
//
// Returns:
//   The optimal value of the sub-game starting in state (given alpha/beta).
double _alpha_beta(State* state, int depth, double alpha, double beta,
                   std::function<double(const State&)> value_function,
                   Player maximizing_player, BestActions* best_actions,
                   bool use_undo) {
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

  const double kInf = std::numeric_limits<double>::infinity();
  const bool is_root = best_actions != nullptr;

  Player player = state->CurrentPlayer();
  if (player == maximizing_player) {
    double value = -kInf;

    for (Action action : state->LegalActions()) {
      double child_value = 0;
      if (use_undo) {
        state->ApplyAction(action);
        child_value =
            _alpha_beta(state, /*depth=*/depth - 1, /*alpha=*/alpha,
                        /*beta=*/beta, value_function, maximizing_player,
                        /*best_action=*/nullptr, use_undo);
        state->UndoAction(player, action);
      } else {
        std::unique_ptr<State> child_state = state->Child(action);
        child_value =
            _alpha_beta(child_state.get(), /*depth=*/depth - 1, /*alpha=*/alpha,
                        /*beta=*/beta, value_function, maximizing_player,
                        /*best_action=*/nullptr, use_undo);
      }

      if (child_value > value) {
        value = child_value;
        if (is_root) {
          best_actions->Clear();
          best_actions->Add(action);
        }
      } else if (is_root && child_value == value) {
        best_actions->Add(action);
      }

      alpha = is_root ? std::max(alpha, std::nextafter(value, -kInf))
                      : std::max(alpha, value);
      if (alpha >= beta) {
        break;  // beta cut-off
      }
    }

    return value;
  } else {
    double value = std::numeric_limits<double>::infinity();

    for (Action action : state->LegalActions()) {
      double child_value = 0;
      if (use_undo) {
        state->ApplyAction(action);
        child_value =
            _alpha_beta(state, /*depth=*/depth - 1, /*alpha=*/alpha,
                        /*beta=*/beta, value_function, maximizing_player,
                        /*best_action=*/nullptr, use_undo);
        state->UndoAction(player, action);
      } else {
        std::unique_ptr<State> child_state = state->Child(action);
        child_value =
            _alpha_beta(child_state.get(), /*depth=*/depth - 1, /*alpha=*/alpha,
                        /*beta=*/beta, value_function, maximizing_player,
                        /*best_action=*/nullptr, use_undo);
      }

      if (child_value < value) {
        value = child_value;
        if (is_root) {
          best_actions->Clear();
          best_actions->Add(action);
        }
      } else if (is_root && child_value == value) {
        best_actions->Add(action);
      }

      beta = is_root ? std::min(beta, std::nextafter(value, kInf))
                     : std::min(beta, value);
      if (alpha >= beta) {
        break;  // alpha cut-off
      }
    }

    return value;
  }
}

// Expectiminimax algorithm.
//
// Runs expectiminimax until the specified depth.
// See https://en.wikipedia.org/wiki/Expectiminimax for details.
//
// Arguments:
//   state: The state to start the search from.
//   depth: The depth of the search (not counting chance nodes).
//   value_function: A value function, taking in a state and returning a value,
//     in terms of the maximizing_player_id.
//   maximizing_player_id: The player running the search (current player at root
//     of the search tree).
//
// Returns:
//   The optimal value of the sub-game starting in state.
double _expectiminimax(const State* state, int depth,
                       std::function<double(const State&)> value_function,
                       Player maximizing_player, BestActions* best_actions,
                       double tolerance) {
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

  const bool is_root = best_actions != nullptr;

  // in root, store the true action values for each action
  std::vector<std::pair<Action, double>> action_values;

  Player player = state->CurrentPlayer();
  double value;
  if (state->IsChanceNode()) {
    value = 0;
    for (const auto& actionprob : state->ChanceOutcomes()) {
      std::unique_ptr<State> child_state = state->Child(actionprob.first);
      double child_value = _expectiminimax(child_state.get(), depth,
                                           value_function, maximizing_player,
                                           /*best_action=*/nullptr, tolerance);
      value += actionprob.second * child_value;
    }
  } else if (player == maximizing_player) {
    value = -std::numeric_limits<double>::infinity();

    for (Action action : state->LegalActions()) {
      std::unique_ptr<State> child_state = state->Child(action);
      double child_value = _expectiminimax(child_state.get(),
                                           /*depth=*/depth - 1, value_function,
                                           maximizing_player,
                                           /*best_action=*/nullptr, tolerance);

      value = std::max(value, child_value);

      if (is_root) {
        action_values.push_back({action, child_value});
      }
    }
  } else {
    value = std::numeric_limits<double>::infinity();

    for (Action action : state->LegalActions()) {
      std::unique_ptr<State> child_state = state->Child(action);
      double child_value = _expectiminimax(child_state.get(),
                                           /*depth=*/depth - 1, value_function,
                                           maximizing_player,
                                           /*best_action=*/nullptr, tolerance);

      value = std::min(value, child_value);
      if (is_root) {
        action_values.push_back({action, child_value});
      }
    }
  }

  if (is_root) {
    for (const auto& [a, v] : action_values) {
      if (std::abs(v - value) <= tolerance) {
        best_actions->Add(a);
      }
    }
  }

  return value;
}
}  // namespace

std::pair<double, BestActions> AlphaBetaSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player, bool use_undo) {
  SPIEL_CHECK_LE(game.NumPlayers(), 2);

  // Check to ensure the correct setup intended for this algorithm.
  // Note: do no check perfect vs. imperfect information to support use of
  // minimax as a subroutine of PIMC.
  GameType game_info = game.GetType();
  SPIEL_CHECK_EQ(game_info.chance_mode, GameType::ChanceMode::kDeterministic);
  SPIEL_CHECK_EQ(game_info.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game_info.utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(game_info.reward_model, GameType::RewardModel::kTerminal);

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
  BestActions best_actions = BestActions();
  double value =
      _alpha_beta(search_root.get(), /*depth=*/depth_limit, /*alpha=*/-infinity,
                  /*beta=*/infinity, value_function, maximizing_player,
                  &best_actions, use_undo);

  return {value, best_actions};
}

std::pair<double, BestActions> ExpectiminimaxSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player) {
  SPIEL_CHECK_LE(game.NumPlayers(), 2);

  GameType game_info = game.GetType();
  SPIEL_CHECK_EQ(game_info.chance_mode,
                 GameType::ChanceMode::kExplicitStochastic);
  SPIEL_CHECK_EQ(game_info.information,
                 GameType::Information::kPerfectInformation);
  SPIEL_CHECK_EQ(game_info.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game_info.utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(game_info.reward_model, GameType::RewardModel::kTerminal);

  std::unique_ptr<State> search_root;
  if (state == nullptr) {
    search_root = game.NewInitialState();
  } else {
    search_root = state->Clone();
  }

  if (maximizing_player == kInvalidPlayer) {
    SPIEL_CHECK_FALSE(search_root->IsChanceNode());
    maximizing_player = search_root->CurrentPlayer();
  }

  BestActions best_actions = BestActions();
  double value =
      _expectiminimax(search_root.get(), /*depth=*/depth_limit, value_function,
                      maximizing_player, &best_actions, 1e-9);

  return {value, best_actions};
}

}  // namespace algorithms
}  // namespace open_spiel
