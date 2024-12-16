// Copyright 2024 DeepMind Technologies Limited
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

#include "open_spiel/games/einstein_wurfelt_nicht/einstein_wurfelt_nicht.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/combinatorics.h"

namespace open_spiel {
namespace einstein_wurfelt_nicht {
namespace {

const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(0, 1.0 / 6),
    std::pair<Action, double>(1, 1.0 / 6),
    std::pair<Action, double>(2, 1.0 / 6),
    std::pair<Action, double>(3, 1.0 / 6),
    std::pair<Action, double>(4, 1.0 / 6),
    std::pair<Action, double>(5, 1.0 / 6)};

// Number of unique directions each cube can take.
constexpr int kNumDirections = 6;

// Direction offsets for black, then white.
constexpr std::array<int, kNumDirections> kDirRowOffsets = {
    {1, 1, 0, -1, -1, 0}};

constexpr std::array<int, kNumDirections> kDirColOffsets = {
    {1, 0, 1, 0, -1, -1}};

// Facts about the game
const GameType kGameType{
    /*short_name=*/"einstein_wurfelt_nicht",
    /*long_name=*/"einstein_wurfelt_nicht",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new EinsteinWurfeltNichtGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

Color PlayerToColor(Player player) {
  SPIEL_CHECK_NE(player, kInvalidPlayer);
  return static_cast<Color>(player);
}

Player ColorToPlayer(Color color) {
  switch (color) {
    case Color::kBlack:
      return kBlackPlayerId;
    case Color::kWhite:
      return kWhitePlayerId;
    default:
      SpielFatalError("No player for this color");
  }
}

Color OpponentColor(Player player) {  // NOLINT
  Color player_color = PlayerToColor(player);
  if (player_color == Color::kBlack) {
    return Color::kWhite;
  } else if (player_color == Color::kWhite) {
    return Color::kBlack;
  } else {
    SpielFatalError("Player should be either black or white");
  }
}

std::string CoordinatesToDirection(int row, int col) {
  std::string direction;
  if (row == col) {
    direction = "diag";
  } else if (row == -1) {
    direction = "up";
  } else if (row == 1) {
    direction = "down";
  } else if (col == 1) {
    direction = "right";
  } else if (col == -1) {
    direction = "left";
  } else {
    std::cout << "r2: " << row << "c2: " << col << std::endl;
    SpielFatalError("Unrecognized cube's movement");
  }
  return direction;
}

}  // namespace

EinsteinWurfeltNichtState::EinsteinWurfeltNichtState(
    std::shared_ptr<const Game> game, int rows, int cols)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kBlackPlayerId),
      turns_(-1),
      rows_(rows),
      cols_(cols) {
  SPIEL_CHECK_GT(rows_, 1);
  SPIEL_CHECK_GT(cols_, 1);
  board_.fill(Cube{Color::kEmpty, -1});

  winner_ = kInvalidPlayer;
  cubes_[0] = cubes_[1] = kNumPlayerCubes;
}

void EinsteinWurfeltNichtState::SetupInitialBoard(Player player,
                                                  Action action) {
  std::vector<int> indices(kNumPlayerCubes);
  std::iota(indices.begin(), indices.end(), 1);
  std::vector<int> cubes_position_order = UnrankPermutation(indices, action);
  int perm_idx = 0;

  // Values in the upper-left corner (black cubes) have a position identified
  // as rows+cols <= 2. Values in the lower-right corner (white cubes) have a
  // position identified as rows+cols >= 6. The rest of the board is empty.
  for (int r = 0; r < kDefaultRows; r++) {
    for (int c = 0; c < kDefaultColumns; c++) {
      if (r + c <= 2 && player == kBlackPlayerId) {
        board_[r * kDefaultColumns + c] =
            Cube{Color::kBlack, cubes_position_order[perm_idx]};
        perm_idx++;
      } else if (r + c >= 6 && player == kWhitePlayerId) {
        board_[r * kDefaultColumns + c] =
            Cube{Color::kWhite, cubes_position_order[perm_idx]};
        perm_idx++;
      }
    }
  }
}

int EinsteinWurfeltNichtState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

int EinsteinWurfeltNichtState::Opponent(int player) const { return 1 - player; }

std::vector<std::vector<int>> EinsteinWurfeltNichtState::AvailableCubesPosition(
    Color player_color) const {
  std::vector<std::pair<int, int>> player_cubes;
  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < cols_; c++) {
      if (board(r, c).color == player_color) {
        if (board(r, c).value == die_roll_) {
          // If there is a cube with the same value as the die,
          // return only this one
          std::vector<std::vector<int>> player_cube;
          player_cube.push_back({board(r, c).value, r, c});
          return player_cube;
        } else {
          player_cubes.push_back({r, c});
        }
      }
    }
  }

  // Initialise lowest/highest cube values to out-of-bound cube's values
  std::vector<int> lowest_cube = {0, 0, 0};   // cube value, r, c
  std::vector<int> highest_cube = {7, 0, 0};  // cube value, r, c
  for (int i = 0; i < player_cubes.size(); ++i) {
    int r = player_cubes[i].first;
    int c = player_cubes[i].second;
    if (board(r, c).value > lowest_cube[0] && board(r, c).value < die_roll_) {
      lowest_cube[0] = board(r, c).value;
      lowest_cube[1] = r;
      lowest_cube[2] = c;
    } else if (board(r, c).value < highest_cube[0] &&
               board(r, c).value > die_roll_) {
      highest_cube[0] = board(r, c).value;
      highest_cube[1] = r;
      highest_cube[2] = c;
    }
  }

  std::vector<std::vector<int>> selected_cubes;
  if (lowest_cube[0] > 0) {
    selected_cubes.push_back(lowest_cube);
  }
  if (highest_cube[0] < 7) {
    selected_cubes.push_back(highest_cube);
  }

  // Legal actions have to be sorted. Sort by row first, then by column
  std::sort(selected_cubes.begin(), selected_cubes.end(),
            [](const std::vector<int>& a, const std::vector<int>& b) {
              if (a[1] != b[1]) return a[1] < b[1];
              return a[2] < b[2];
            });

  return selected_cubes;
}

void EinsteinWurfeltNichtState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LE(action, kNumCubesPermutations - 1);
    turn_history_info_.push_back(TurnHistoryInfo(kChancePlayerId, prev_player_,
                                                 die_roll_, action,
                                                 Cube{Color::kEmpty, -1}));
    if (turns_ == -1) {
      SetupInitialBoard(kBlackPlayerId, action);
      turns_ = 0;
      return;
    } else if (turns_ == 0) {
      SetupInitialBoard(kWhitePlayerId, action);
      turns_++;
      return;
    } else {
      cur_player_ = Opponent(prev_player_);
      prev_player_ = cur_player_;
      die_roll_ = action + 1;
      turns_++;
      return;
    }
  }

  // The die should have been rolled at least once at this point
  SPIEL_CHECK_GE(die_roll_, 1);
  SPIEL_CHECK_LE(die_roll_, 6);

  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, cols_, kNumDirections, 2});
  int r1 = values[0];
  int c1 = values[1];
  int dir = values[2];
  bool capture = values[3] == 1;
  int r2 = r1 + kDirRowOffsets[dir];
  int c2 = c1 + kDirColOffsets[dir];

  SPIEL_CHECK_TRUE(InBounds(r1, c1));
  SPIEL_CHECK_TRUE(InBounds(r2, c2));

  // Remove cubes if captured.
  if (board(r2, c2).color == Color::kBlack) {
    cubes_[ColorToPlayer(Color::kBlack)]--;
  } else if (board(r2, c2).color == Color::kWhite) {
    cubes_[ColorToPlayer(Color::kWhite)]--;
  }

  Cube captured_cube = (capture) ? board(r2, c2) : Cube{Color::kEmpty, -1};
  turn_history_info_.push_back(TurnHistoryInfo(
      cur_player_, prev_player_, die_roll_, action, captured_cube));

  SetBoard(r2, c2, board(r1, c1));
  SetBoard(r1, c1, Cube{Color::kEmpty, -1});

  // Check for winner.
  if ((cur_player_ == 0 && r2 == (rows_ - 1) && c2 == (cols_ - 1)) ||
      (cubes_[ColorToPlayer(Color::kWhite)] == 0)) {
    winner_ = 0;
  } else if ((cur_player_ == 1 && r2 == 0 && c2 == 0) ||
             (cubes_[ColorToPlayer(Color::kBlack)] == 0)) {
    winner_ = 1;
  }

  cur_player_ = NextPlayerRoundRobin(cur_player_, kNumPlayers);
  cur_player_ = kChancePlayerId;
  turns_++;
}

std::string EinsteinWurfeltNichtState::ActionToString(Player player,
                                                      Action action) const {
  std::string action_string = "";

  if (IsChanceNode()) {
    if (turns_ == -1) {
      absl::StrAppend(&action_string,
                      "Placing black cubes on the board - action ", action);
      return action_string;
    } else if (turns_ == 0) {
      absl::StrAppend(&action_string,
                      "Placing white cubes on the board - action ", action);
      return action_string;
    } else if (turns_ >= 0) {
      absl::StrAppend(&action_string, "roll ", action + 1);
      return action_string;
    }
  }

  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, cols_, kNumDirections, 2});
  int r1 = values[0];
  int c1 = values[1];
  int dir = values[2];
  bool capture = values[3] == 1;
  int r2 = kDirRowOffsets[dir];
  int c2 = kDirColOffsets[dir];

  Cube cube = board(r1, c1);
  std::string color = (cube.color == Color::kBlack) ? "B" : "W";

  std::string direction = CoordinatesToDirection(r2, c2);
  absl::StrAppend(&action_string, color);
  absl::StrAppend(&action_string, cube.value);
  absl::StrAppend(&action_string, "-");
  absl::StrAppend(&action_string, direction);
  if (capture) {
    absl::StrAppend(&action_string, "*");
  }
  return action_string;
}

std::vector<Action> EinsteinWurfeltNichtState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  std::vector<Action> movelist;
  if (IsTerminal()) return movelist;
  const Player player = CurrentPlayer();
  Color player_color = PlayerToColor(player);
  std::vector<int> action_bases = {rows_, cols_, kNumDirections, 2};
  std::vector<int> action_values = {0, 0, 0, 0};

  std::vector<std::vector<int>> available_cubes;
  available_cubes = AvailableCubesPosition(player_color);

  for (int i = 0; i < available_cubes.size(); ++i) {
    int r = available_cubes[i][1];
    int c = available_cubes[i][2];
    for (int o = 0; o < kNumDirections / 2; o++) {
      int dir = player * kNumDirections / 2 + o;
      int rp = r + kDirRowOffsets[dir];
      int cp = c + kDirColOffsets[dir];
      if (InBounds(rp, cp)) {
        action_values[0] = r;
        action_values[1] = c;
        action_values[2] = dir;
        if (board(rp, cp).color == Color::kEmpty) {
          action_values[3] = 0;  // no capture
          movelist.push_back(RankActionMixedBase(action_bases, action_values));
        } else {
          action_values[3] = 1;  // capture
          movelist.push_back(RankActionMixedBase(action_bases, action_values));
        }
      }
    }
  }
  return movelist;
}

std::vector<std::pair<Action, double>>
EinsteinWurfeltNichtState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (turns_ <= 0) {
    // First 2 moves corresponds to the initial board setup.
    // There are 6! = 720 possible permutations of the cubes.
    std::vector<std::pair<Action, double>> chance_outcomes;
    double action_prob = 1.0 / kNumCubesPermutations;
    chance_outcomes.reserve(kNumCubesPermutations);

    for (Action i = 0; i < kNumCubesPermutations; ++i) {
      chance_outcomes.emplace_back(i, action_prob);
    }
    return chance_outcomes;
  } else {
    return kChanceOutcomes;
  }
}

bool EinsteinWurfeltNichtState::InBounds(int r, int c) const {
  return (r >= 0 && r < rows_ && c >= 0 && c < cols_);
}

std::string EinsteinWurfeltNichtState::ToString() const {
  std::string W_result = "";

  for (int r = 0; r < kDefaultRows; r++) {
    for (int c = 0; c < kDefaultColumns; c++) {
      if (board_[r * kDefaultColumns + c].color == Color::kBlack) {
        absl::StrAppend(&W_result, "|b");
        absl::StrAppend(&W_result, board_[r * kDefaultColumns + c].value);
        absl::StrAppend(&W_result, "|");
      } else if (board_[r * kDefaultColumns + c].color == Color::kWhite) {
        absl::StrAppend(&W_result, "|w");
        absl::StrAppend(&W_result, board_[r * kDefaultColumns + c].value);
        absl::StrAppend(&W_result, "|");
      } else {
        absl::StrAppend(&W_result, "|__|");
      }
    }
    W_result.append("\n");
  }
  return W_result;
}

bool EinsteinWurfeltNichtState::IsTerminal() const {
  return (winner_ >= 0 || (cubes_[0] == 0 || cubes_[1] == 0));
}

std::vector<double> EinsteinWurfeltNichtState::Returns() const {
  if (winner_ == 0 || cubes_[1] == 0) {
    return {1.0, -1.0};
  } else if (winner_ == 1 || cubes_[0] == 0) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string EinsteinWurfeltNichtState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void EinsteinWurfeltNichtState::ObservationTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  auto value_it = values.begin();

  for (int cube_num = 1; cube_num < kNumPlayerCubes + 1; ++cube_num) {
    for (int player_idx = 0; player_idx < kNumPlayers; ++player_idx) {
      for (int8_t y = 0; y < kDefaultRows; ++y) {
        for (int8_t x = 0; x < kDefaultColumns; ++x) {
          *value_it++ = (board(x, y).value == cube_num &&
                                 board(x, y).color == PlayerToColor(player_idx)
                             ? 1.0
                             : 0.0);
        }
      }
    }
  }
}

void EinsteinWurfeltNichtState::UndoAction(Player player, Action action) {
  const TurnHistoryInfo& thi = turn_history_info_.back();
  SPIEL_CHECK_EQ(thi.player, player);
  SPIEL_CHECK_EQ(action, thi.action);

  if (player != kChancePlayerId) {
    std::vector<int> values =
        UnrankActionMixedBase(action, {rows_, cols_, kNumDirections, 2});
    int r1 = values[0];
    int c1 = values[1];
    int dir = values[2];
    int r2 = r1 + kDirRowOffsets[dir];
    int c2 = c1 + kDirColOffsets[dir];
    Cube captured_cube = thi.captured_cube;

    SetBoard(r1, c1, board(r2, c2));
    if (captured_cube.value != -1) {
      SetBoard(r2, c2, captured_cube);
      if (captured_cube.color == Color::kBlack) {
        cubes_[ColorToPlayer(Color::kBlack)]++;
      } else if (captured_cube.color == Color::kWhite) {
        cubes_[ColorToPlayer(Color::kWhite)]++;
      }
    } else {
      SetBoard(r2, c2, Cube{Color::kEmpty, -1});
    }
  } else {
    for (int r = 0; r < kDefaultRows; r++) {
      for (int c = 0; c < kDefaultColumns; c++) {
        if (turns_ == 1 && board(r, c).color == Color::kWhite) {
          board_[r * kDefaultColumns + c] = Cube{Color::kEmpty, -1};
        } else if (turns_ == 0 && board(r, c).color == Color::kBlack) {
          board_[r * kDefaultColumns + c] = Cube{Color::kEmpty, -1};
        }
      }
    }
  }

  // Undo win status.
  winner_ = kInvalidPlayer;

  turn_history_info_.pop_back();
  history_.pop_back();
  --turns_;
  --move_number_;
}

std::unique_ptr<State> EinsteinWurfeltNichtState::Clone() const {
  return std::unique_ptr<State>(new EinsteinWurfeltNichtState(*this));
}

// Setter function used for debugging and tests. Note: this does not set the
// historical information properly, so Undo likely will not work on states
// set this way!
void EinsteinWurfeltNichtState::SetState(
    int cur_player, int die_roll, const std::array<Cube, k2dMaxBoardSize> board,
    int cubes_black, int cubes_white) {
  cur_player_ = cur_player;
  die_roll_ = die_roll;
  board_ = board;
  cubes_[ColorToPlayer(Color::kBlack)] = cubes_black;
  cubes_[ColorToPlayer(Color::kWhite)] = cubes_white;
}

EinsteinWurfeltNichtGame::EinsteinWurfeltNichtGame(const GameParameters& params)
    : Game(kGameType, params), rows_(kDefaultRows), cols_(kDefaultColumns) {}

int EinsteinWurfeltNichtGame::NumDistinctActions() const {
  return rows_ * cols_ * kNumDirections * 2;
}

}  // namespace einstein_wurfelt_nicht
}  // namespace open_spiel
