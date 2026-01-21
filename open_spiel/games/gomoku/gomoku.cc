// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/gomoku/gomoku.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace gomoku {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"gomoku",
    /*long_name=*/"Gomoku",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
   {{"size", GameParameter(kDefaultSize)},
    {"dims", GameParameter(kDefaultDims)},
    {"connect", GameParameter(kDefaultConnect)},
    {"wrap", GameParameter(kDefaultWrap)}
	 }
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GomokuGame(params));
}

std::ostream& operator<<(std::ostream& os, Stone s) {
  switch (s) {
    case Stone::kEmpty: return os << "Empty";
    case Stone::kBlack: return os << "Black";
    case Stone::kWhite: return os << "White";
  }
  return os << "Unknown";
}

GomokuGame::GomokuGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size")),
      dims_(ParameterValue<int>("dims")),
      connect_(ParameterValue<int>("connect")),
      wrap_(ParameterValue<bool>("wrap")) {
				total_size_ = 1;
				for (int i = 0; i < dims_; ++i) {
					total_size_ *= size_;
				}
}


REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);



void GomokuState::DoApplyAction(Action move) {
	SPIEL_CHECK_EQ(board_.AtIndex(move), Stone::kEmpty);
  board_.AtIndex(move) = current_player_ == kBlackPlayer
                    ? Stone::kBlack
                    : Stone::kWhite;
  current_player_ = 1 - current_player_;
  move_count_ += 1;
}

std::vector<Action> GomokuState::LegalActions() const {
  if (IsTerminal()) return {};

  std::vector<Action> actions;
  actions.reserve(board_.NumCells());

  for (Action i = 0; i < static_cast<Action>(board_.NumCells()); ++i) {
    if (board_.AtIndex(i) == Stone::kEmpty) {
      actions.push_back(i);
    }
  }
  return actions;
}


std::string GomokuState::ActionToString(Player player,
                                           Action action_id) const {
  return game_->ActionToString(player, action_id);
}


std::string GomokuState::ToString() const {
  std::string s;
  s.reserve(1 + board_.NumCells());

  // Player to move
  s.push_back(current_player_ == kBlackPlayer ? 'B' : 'W');

  // Board contents in flattened order
  for (std::size_t i = 0; i < board_.NumCells(); ++i) {
    switch (board_.AtIndex(i)) {
      case Stone::kBlack: s.push_back('b'); break;
      case Stone::kWhite: s.push_back('w'); break;
      case Stone::kEmpty: s.push_back('.'); break;
    }
  }
  return s;
}

std::vector<double> GomokuState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
}

bool GomokuState::CheckWinFromLastMove() const {
  if (history_.empty()) return false;
	const auto& last = history_.back();
  Action last_move = last.action;

  const Grid<Stone>::Coord start =
      board_.Unflatten(last_move);
  const Stone stone = board_.At(start);

  SPIEL_CHECK_NE(stone, Stone::kEmpty);

  for (const auto& dir : board_.Directions()) {
    if (!board_.IsCanonical(dir)) continue;

    int count = 1;  // include the starting stone

    // forward direction
    {
      auto c = start;
      while (count < connect_ && board_.Step(c, dir) &&
             board_.At(c) == stone) {
        ++count;
      }
    }

    // backward direction
    {
      auto neg_dir = dir;
      for (int& v : neg_dir) v = -v;

      auto c = start;
      while (count < connect_ && board_.Step(c, neg_dir) &&
             board_.At(c) == stone) {
        ++count;
      }
    }

    if (count >= connect_) {
      return true;
    }
  }
  return false;
}


absl::optional<std::vector<double>> GomokuState::MaybeFinalReturns() const {
	const auto& last = history_.back();
  Action last_move = last.action;
  if (last_move != kInvalidAction) {
    if (CheckWinFromLastMove()) {
      std::vector<double> returns(2, -1.0);
      // winner is the player who made the last move
      returns[CurrentPlayer() ^ 1] = 1.0;
      return returns;
    }
  }

  // draw by full board
  if (move_count_ == board_.NumCells()) {
    return std::vector<double>{0.0, 0.0};
  }

  return absl::nullopt;
}


std::string GomokuState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string GomokuState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

//TODO implement this
void GomokuState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

}

void GomokuState::UndoAction(Player player, Action move) {
	board_.AtIndex(move) = Stone::kEmpty;
  current_player_ = player;
  move_count_ -= 1;
  history_.pop_back();
}

std::unique_ptr<State> GomokuState::Clone() const {
  return std::unique_ptr<State>(new GomokuState(*this));
}

// TODO implement this
std::string GomokuGame::ActionToString(Player player,
                                          Action action_id) const {
  return "";
}

// TODO implement this
std::vector<int> GomokuGame::ObservationTensorShape() const {
	return {15, 15, 3};
}

// TODO implement this
int GomokuGame::MaxGameLength() const {
	return total_size_;
}

int GomokuGame::NumDistinctActions() const {
	return total_size_;
}



GomokuState::GomokuState(std::shared_ptr<const Game> game,
                         const std::string& state_str)
    : State(game),
      size_(static_cast<const GomokuGame&>(*game).Size()),
      dims_(static_cast<const GomokuGame&>(*game).Dims()),
      connect_(static_cast<const GomokuGame&>(*game).Connect()),
      wrap_(static_cast<const GomokuGame&>(*game).Wrap()),
      board_(
        static_cast<std::size_t>(static_cast<const GomokuGame&>(*game).Size()),
        static_cast<std::size_t>(static_cast<const GomokuGame&>(*game).Dims()),
        static_cast<const GomokuGame&>(*game).Wrap()),
      current_player_(kBlackPlayer),
      move_count_(0) {
  if (state_str.empty()) {
    board_.Fill(Stone::kEmpty);
    current_player_ = kBlackPlayer;
    return;
  }
	const std::size_t expected =
    1 + board_.NumCells();  // size^dims

   SPIEL_CHECK_EQ(state_str.size(), expected);
  switch (state_str[0]) {
    case 'W':
      current_player_ = kWhitePlayer;
			break;
    case 'B': 
			current_player_ = kBlackPlayer;
			break;
    default:
      SpielFatalError("Invalid player char in state string");
  }
	for (std::size_t i = 0; i < board_.NumCells(); ++i) {
    char c = state_str[i + 1];
    Stone s;

      switch (c) {
        case 'b':
				  s = Stone::kBlack;
				  break;
        case 'w':
				  s = Stone::kWhite;
				  break;
        case '.':
				  s = Stone::kEmpty;
				  break;
        case ' ': s = Stone::kEmpty;
				  break;
        default:
          SpielFatalError("Invalid board char in state string");
    }

    board_.AtIndex(i) = s;
  }
}



}  // namespace gomoku
}  // namespace open_spiel
