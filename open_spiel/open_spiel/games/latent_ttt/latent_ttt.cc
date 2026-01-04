// Copyright 2025 DeepMind Technologies Limited
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

#include "open_spiel/games/latent_ttt/latent_ttt.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace latent_ttt {
namespace {

using tic_tac_toe::PlayerToState;
using tic_tac_toe::StateToString;

const GameType kGameType{/*short_name=*/"latent_ttt",
                         /*long_name=*/"Latent Tic Tac Toe",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LatentTTTGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

LatentTTTState::LatentTTTState(std::shared_ptr<const Game> game)
    : State(game), state_(game) {
  std::fill(begin(x_view_), end(x_view_), tic_tac_toe::CellState::kEmpty);
  std::fill(begin(o_view_), end(o_view_), tic_tac_toe::CellState::kEmpty);
}

void LatentTTTState::DoApplyAction(Action move) {
  Player cur_player = CurrentPlayer();
  auto& cur_view = cur_player == 0 ? x_view_ : o_view_;
  int prev_pending_cell = -1;
  if (!action_sequence_.empty() &&
      action_sequence_.back().first != cur_player) {
    prev_pending_cell = action_sequence_.back().second;
  }

  if (state_.BoardAt(move) == tic_tac_toe::CellState::kEmpty) {
    state_.ApplyAction(move);
  } else {
    state_.ChangePlayer();
  }

  cur_view[move] = state_.BoardAt(move);
  // Reveal opponent's previously pending selection to the acting player.
  if (prev_pending_cell != -1) {
    cur_view[prev_pending_cell] = state_.BoardAt(prev_pending_cell);
  }
  action_sequence_.push_back(std::pair<int, Action>(cur_player, move));
}

std::vector<Action> LatentTTTState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  const Player player = CurrentPlayer();
  const auto& cur_view = player == 0 ? x_view_ : o_view_;
  for (Action move = 0; move < tic_tac_toe::kNumCells; ++move) {
    if (cur_view[move] == tic_tac_toe::CellState::kEmpty) {
      moves.push_back(move);
    }
  }
  return moves;
}

std::string LatentTTTState::ViewToString(Player player) const {
  std::string str;
  const auto& cur_view = player == 0 ? x_view_ : o_view_;
  for (int r = 0; r < tic_tac_toe::kNumRows; ++r) {
    for (int c = 0; c < tic_tac_toe::kNumCols; ++c) {
      int idx = r * tic_tac_toe::kNumCols + c;
      absl::StrAppend(&str, StateToString(cur_view[idx]));
    }
    if (r < (tic_tac_toe::kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  absl::StrAppend(&str, "\n");
  return str;
}

std::string LatentTTTState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int opp_pending_index = -1;
  if (!action_sequence_.empty()) {
    const auto& last = action_sequence_.back();
    if (last.first != player) {
      opp_pending_index = static_cast<int>(action_sequence_.size()) - 1;
    }
  }

  std::string str;
  absl::StrAppend(&str, ViewToString(player));

  for (int i = 0; i < static_cast<int>(action_sequence_.size()); ++i) {
    if (i == opp_pending_index) continue;
    const auto& p = action_sequence_[i];
    absl::StrAppend(&str, p.first, ",");
    absl::StrAppend(&str, p.second, " ");
  }
  return str;
}

std::string LatentTTTState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ViewToString(player);
}

void LatentTTTState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(
      values.size(),
      tic_tac_toe::kCellStates * tic_tac_toe::kNumRows * tic_tac_toe::kNumCols);
  TensorView<3> view(
      values,
      {tic_tac_toe::kCellStates, tic_tac_toe::kNumRows, tic_tac_toe::kNumCols},
      true);

  std::fill(values.begin(), values.end(), 0.0f);

  const auto& cur_view = player == 0 ? x_view_ : o_view_;
  for (int r = 0; r < tic_tac_toe::kNumRows; ++r) {
    for (int c = 0; c < tic_tac_toe::kNumCols; ++c) {
      int idx = r * tic_tac_toe::kNumCols + c;
      auto cell_state = cur_view[idx];
      view[{static_cast<int>(cell_state), r, c}] = 1.0f;
    }
  }
}

void LatentTTTState::UndoAction(Player player, Action move) {
  const auto& last = action_sequence_.back();
  SPIEL_CHECK_EQ(last.first, player);
  SPIEL_CHECK_EQ(last.second, move);

  if (state_.BoardAt(move) == PlayerToState(player)) {
    state_.UndoAction(player, move);
  } else {
    state_.ChangePlayer();
  }

  auto& cur_view = player == 0 ? x_view_ : o_view_;
  cur_view[move] = tic_tac_toe::CellState::kEmpty;

  int prev_pending_cell = -1;
  if (action_sequence_.size() >= 2) {
    const auto& prev = action_sequence_[action_sequence_.size() - 2];
    if (prev.first == 1 - player) prev_pending_cell = prev.second;
  }
  if (prev_pending_cell != -1) {
    cur_view[prev_pending_cell] = tic_tac_toe::CellState::kEmpty;
  }

  action_sequence_.pop_back();
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> LatentTTTState::Clone() const {
  return std::unique_ptr<State>(new LatentTTTState(*this));
}

std::string LatentTTTGame::ActionToString(Player player,
                                          Action action_id) const {
  return absl::StrCat(tic_tac_toe::StateToString(PlayerToState(player)), "(",
                      action_id / tic_tac_toe::kNumCols, ",",
                      action_id % tic_tac_toe::kNumCols, ")");
}

LatentTTTGame::LatentTTTGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace latent_ttt
}  // namespace open_spiel
