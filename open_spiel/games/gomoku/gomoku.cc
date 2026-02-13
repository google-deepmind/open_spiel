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
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
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
    {
      {"size", GameParameter(kDefaultSize)},
      {"dims", GameParameter(kDefaultDims)},
      {"connect", GameParameter(kDefaultConnect)},
      {"anti", GameParameter(kDefaultAnti)},
      {"wrap", GameParameter(kDefaultWrap)}
    }
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GomokuGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::ostream& operator<<(std::ostream& os, Stone s) {
  switch (s) {
    case Stone::kEmpty: return os << "Empty";
    case Stone::kBlack: return os << "Black";
    case Stone::kWhite: return os << "White";
  }
  return os << "Unknown";
}

int StoneToInt(Stone s) {
  switch (s) {
    case Stone::kEmpty: return -1;
    case Stone::kBlack: return 0;
    case Stone::kWhite: return 1;
  }
  SpielFatalError("Unknown stone.");
  return 0;  // This never happens
}

GomokuGame::GomokuGame(const GameParameters& params)
    : Game(kGameType, params),
      size_(ParameterValue<int>("size")),
      dims_(ParameterValue<int>("dims")),
      connect_(ParameterValue<int>("connect")),
      anti_(ParameterValue<bool>("anti")),
      wrap_(ParameterValue<bool>("wrap")) {
  total_size_ = 1;
  for (int i = 0; i < dims_; ++i) {
    total_size_ *= size_;
  }
  strides_.resize(dims_);
  std::size_t stride = 1;
  for (int d = dims_ - 1; d >= 0; --d) {
    strides_[d] = stride;
    stride *= size_;
  }
  std::mt19937_64 gen(52616);
  zobrist_table_.resize(total_size_);
  for (int i = 0; i < total_size_; ++i) {
    zobrist_table_[i][0] = gen();
    zobrist_table_[i][1] = gen();
  }
  player_to_move_hash_ = gen();
}


std::vector<int> GomokuGame::UnflattenAction(Action action_id) const {
  SPIEL_CHECK_LT(action_id, NumDistinctActions());
  std::vector<int> coord(dims_);
  std::size_t index = action_id;

  for (std::size_t d = 0; d < dims_; ++d) {
    coord[d] = static_cast<int>(index / strides_[d]);
    index %= strides_[d];
  }
  return coord;
}

std::vector<int> GomokuGame::ActionToMove(Action action) const {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, NumDistinctActions());

  std::vector<int> move(dims_);
  std::size_t index = action;

  for (int d = 0; d < dims_; ++d) {
    move[d] = static_cast<int>(index / strides_[d]);
    index %= strides_[d];
  }
  return move;
}

Action GomokuGame::MoveToAction(const std::vector<int>& move) const {
  SPIEL_CHECK_EQ(move.size(), dims_);

  std::size_t action = 0;
  for (int d = 0; d < dims_; ++d) {
    SPIEL_CHECK_GE(move[d], 0);
    SPIEL_CHECK_LT(move[d], size_);
    action += move[d] * strides_[d];
  }

  return static_cast<Action>(action);
}

uint64_t GomokuState::HashValue() const {
  return zobrist_hash_;
}

void GomokuState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_.AtIndex(move), Stone::kEmpty);
  board_.AtIndex(move) = current_player_ == kBlackPlayer
                    ? Stone::kBlack
                    : Stone::kWhite;
  const auto* gomoku = static_cast<const GomokuGame*>(game_.get());
  zobrist_hash_ ^=
    gomoku->ZobristTable()[move][current_player_];
  zobrist_hash_ ^=
    gomoku->PlayerToMoveHash();
  current_player_ = 1 - current_player_;
  move_count_ += 1;
  CheckWinFromLastMove(move);
}

absl::optional<std::vector<Grid<Stone>::Coord>>
GomokuState::FindWinLineFromLastMove(Action last_move) const {
  using Coord = Grid<Stone>::Coord;

  const Coord start = board_.Unflatten(last_move);
  const Stone stone = board_.At(start);

  SPIEL_CHECK_NE(stone, Stone::kEmpty);

  for (const auto& dir : board_.Directions()) {
    if (!board_.IsCanonical(dir)) continue;

    std::vector<Coord> line;
    line.push_back(start);

    // forward
    {
      Coord c = start;
      while (static_cast<int>(line.size()) < connect_ &&
             board_.Step(c, dir) &&
             board_.At(c) == stone) {
        line.push_back(c);
      }
    }

    // backward
    {
      Coord neg_dir = dir;
      for (int& v : neg_dir) v = -v;

      Coord c = start;
      while (static_cast<int>(line.size()) < connect_ &&
             board_.Step(c, neg_dir) &&
             board_.At(c) == stone) {
        line.insert(line.begin(), c);
      }
    }

    if (static_cast<int>(line.size()) >= connect_) {
      line.resize(connect_);  // arbitrary truncation is fine
      return line;
    }
  }

  return absl::nullopt;
}

void GomokuState::CheckWinFromLastMove(Action last_move) {
  auto maybe_line = FindWinLineFromLastMove(last_move);
  if (maybe_line.has_value()) {
    terminal_ = true;
    winning_line_ = *maybe_line;
    // current_player_ flips before this is called.
    if (current_player_ == 0) {
        black_score_ = -1.0;
        white_score_ = 1.0;
      } else {
        black_score_ = 1.0;
        white_score_ = -1.0;
      }
    }
    const auto* gomoku = static_cast<const GomokuGame*>(game_.get());
    if (gomoku->Anti()) {
        black_score_ *= -1;
        white_score_ *= -1;
    }
}

const std::vector<Grid<Stone>::Coord>& GomokuState::WinningLine() const {
  return winning_line_;
}

std::vector<Action> GomokuState::LegalActions() const {
  if (terminal_) return {};

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
  return {black_score_, white_score_};
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

std::vector<int> GomokuGame::ObservationTensorShape() const {
  return {3, total_size_};
}

inline absl::optional<Player> StoneOwner(Stone stone) {
  if (stone == Stone::kEmpty) return absl::nullopt;
  if (stone == Stone::kBlack) return static_cast<Player>(0);
  return static_cast<Player>(1);
}

void GomokuState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  const int num_cells = board_.NumCells();
  SPIEL_CHECK_EQ(values.size(), 3 * num_cells);

  std::fill(values.begin(), values.end(), 0.0f);

  // Plane 0: black stones
  // Plane 1: white stones
  for (int i = 0; i < num_cells; ++i) {
    const Stone stone = board_.AtIndex(i);
    if (stone == Stone::kBlack) {
      values[i] = 1.0f;
    } else if (stone == Stone::kWhite) {
      values[num_cells + i] = 1.0f;
    }
  }

  // Plane 2: player to move
  // Convention: 1.0 if Black to move, 0.0 if White to move
  if (CurrentPlayer() == Player{0}) {
    std::fill(values.begin() + 2 * num_cells,
              values.begin() + 3 * num_cells,
              1.0f);
  }
}

void GomokuState::UndoAction(Player player, Action move) {
  SPIEL_CHECK_FALSE(history_.empty());
  Action old = history_.back().action;
  history_.pop_back();
  board_.AtIndex(old) = Stone::kEmpty;
  current_player_ = 1 - current_player_;
  move_count_ -= 1;
  terminal_ = false;
  winning_line_.clear();
  black_score_ = 0.0;
  white_score_ = 0.0;
  zobrist_hash_ = ComputeZobrist(board_);
}

std::unique_ptr<State> GomokuState::Clone() const {
  return std::unique_ptr<State>(new GomokuState(*this));
}

std::string GomokuGame::ActionToString(Player player,
          Action action_id) const {
  const auto coord = UnflattenAction(action_id);

  std::ostringstream oss;
  for (std::size_t d = 0; d < coord.size(); ++d) {
    if (d > 0) oss << ",";
    oss << coord[d];
  }
  return oss.str();
}


int GomokuGame::MaxGameLength() const {
  return total_size_;
}

int GomokuGame::NumDistinctActions() const {
  return total_size_;
}

uint64_t GomokuState::ComputeZobrist(
  const Grid<Stone>& grid) const {
    const auto* gomoku = static_cast<const GomokuGame*>(game_.get());
    uint64_t h = 0;
    for (int i = 0; i < grid.NumCells(); ++i) {
      Stone s = grid.AtIndex(i);
      if (s != Stone::kEmpty) {
         h ^= gomoku->ZobristTable()[i][StoneToInt(s)];
      }
    }
    if (current_player_ == 1) {
      h ^= gomoku->PlayerToMoveHash();
    }
    return h;
}

uint64_t GomokuState::SymmetricHash() const {
  uint64_t best = ComputeZobrist(board_);

  // --- Basic rotations ---
  for (auto [i, j] : board_.GenRotations()) {
    Grid<Stone> rotated = board_;
    for (int r = 0; r < 3; ++r) {  // 90, 180, 270
      rotated = rotated.ApplyRotation(i, j);
      best = std::min(best, ComputeZobrist(rotated));
    }
  }

  // --- Reflections ---
  if (symmetry_policy_.allow_reflections) {
    for (int axis = 0; axis < dims_; ++axis) {
      Grid<Stone> refl = board_.ApplyReflection(axis);
      best = std::min(best, ComputeZobrist(refl));

      // --- Reflection + rotations ---
      if (symmetry_policy_.allow_reflection_rotations) {
        for (auto [i, j] : board_.GenRotations()) {
          Grid<Stone> rotated = refl;
          for (int r = 0; r < 3; ++r) {
            rotated = rotated.ApplyRotation(i, j);
            best = std::min(best, ComputeZobrist(rotated));
          }
        }
      }
    }
  }

  return best;
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
      move_count_(0),
      initial_stones_(0),
      black_score_(0.0),
      white_score_(0.0) {
  if (state_str.empty()) {
    board_.Fill(Stone::kEmpty);
    current_player_ = kBlackPlayer;
    return;
  }
  const auto* gomoku = static_cast<const GomokuGame*>(game_.get());
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
          initial_stones_++;
          break;
        case 'w':
          s = Stone::kWhite;
          initial_stones_++;
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
  zobrist_hash_ = ComputeZobrist(board_);
}



}  // namespace gomoku
}  // namespace open_spiel
