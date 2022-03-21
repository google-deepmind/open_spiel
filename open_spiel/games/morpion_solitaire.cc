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
#include "open_spiel/games/morpion_solitaire.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"

namespace open_spiel {
namespace morpion_solitaire {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"morpion_solitaire",
                         /*long_name=*/"Morpion Solitaire",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/1,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MorpionGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// Line methods =============================================================
Line::Line(Point p1, Point p2) { Init(p1, p2); }

// Action encoding (must be changed to support larger boards):
// - 0 - 129 represents lines with direction [0, 1]
// - 130 - 259 represents lines with direction [1, 0]
// - 260 - 359 represents lines with direction [1, -1]
// - 360 - 459 represents lines with direction [1, 1]
Line::Line(Action action) {
  int row;
  int base;
  Point point1;
  Point point2;
  if (action >= 0 && action <= 129) {
    // [0, 1]
    row = action / 10;
    point1 = Point(row, action - row * 10);
    point2 = Point(row, (action - row * 10) + 3);
  } else if (action >= 130 && action <= 259) {
    // [1, 0]
    base = action - 130;
    row = (base) / 13;
    point1 = Point(row, base - row * 13);
    point2 = Point(row + 3, (base - row * 13));
  } else if (action >= 260 && action <= 359) {
    // [1, -1]
    base = action - 260;
    row = (base) / 10;
    point1 = Point(row, base - row * 10);
    point2 = Point(row + 3, (base - row * 10) + 3);
  } else if (action >= 360 && action <= 459) {
    // [1, 1]
    base = action - 360;
    row = (base) / 10;
    point1 = Point(row + 3, base - row * 10);
    point2 = Point(row, (base - row * 10) + 3);
  } else {
    SpielFatalError("action provided does not correspond with a move");
  }
  Init(point1, point2);
}

void Line::Init(Point point1, Point point2) {
  if (point1 < point2) {
    endpoint1_ = point1;
    endpoint2_ = point2;
  } else {
    endpoint1_ = point2;
    endpoint2_ = point1;
  }
  // Categorize line in one of four directions ([0, 1], [1, 1], [1, -1], [1,
  // 0]).
  direction_[0] = static_cast<int>((endpoint2_.x - endpoint1_.x) / 3);
  direction_[1] = static_cast<int>((endpoint2_.y - endpoint1_.y) / 3);
  // Get all points in line (beyond the two initial endpoints) and sort.
  for (int i = 0; i < 4; i++) {
    line_points_.emplace_back(endpoint1_.x + i * direction_[0],
                              endpoint1_.y + i * direction_[1]);
  }
  std::sort(line_points_.begin(), line_points_.end());
}

bool Line::CheckOverlap(Line l) {
  // Only check for overlapping points for lines in the same direction.
  if (direction_ != l.GetDirection()) {
    return false;
  }
  // Check if it's the same line.
  if ((endpoint1_ == l.GetEndpoints()[0]) &&
      (endpoint2_ == l.GetEndpoints()[1])) {
    return false;
  }
  // Check for overlapping points between the two lines.
  std::vector<Point> intersect = {};
  std::vector<Point> l_points = l.GetAllPoints();
  std::set_intersection(l_points.begin(), l_points.end(), line_points_.begin(),
                        line_points_.end(), std::back_inserter(intersect));
  if (!intersect.empty()) {  // Line is overlapping if intersection.size() >=1
                             // in 4D version.
    return true;
  }
  return false;
}

bool Line::operator==(Line other_line) {
  return (endpoint1_ == other_line.GetEndpoints()[0]) &&
         (endpoint2_ == other_line.GetEndpoints()[1]);
}

// Getters
Action Line::GetAction() {
  int dirCode;
  if ((direction_[0] == 0) && (direction_[1] == 1)) {
    dirCode = 1;
  } else if ((direction_[0] == 1) && (direction_[1] == 0)) {
    dirCode = 2;
  } else if ((direction_[0] == 1) && (direction_[1] == 1)) {
    dirCode = 3;
  } else {
    dirCode = 4;
  }
  // Get action encoding from line endpoints
  switch (dirCode) {
    // [0, 1] 0 ... 129
    case 1:
      return endpoint1_.x * 10 + endpoint1_.y;

      // [1, 0] 130 ... 259
    case 2:
      return endpoint1_.x * 13 + endpoint1_.y + 130;

      // [1, 1] 260 ... 359
    case 3:
      return endpoint1_.x * 10 + endpoint1_.y + 260;

      // [1, -1] 360 ... 459
    case 4:
      return (endpoint2_.x - 3) * 10 + endpoint2_.y + 360;

    default:
      SpielFatalError(absl::StrCat("Unhandled case in Line::GetAction()",
                                   ", dirCode = ", dirCode));
  }
}

std::string Line::ToString() const {
  return "(" + endpoint1_.ToString() + " " + endpoint2_.ToString() + ")";
}

std::vector<Point> Line::GetEndpoints() {
  return std::vector<Point>{endpoint1_, endpoint2_};
}

std::array<int, 2> Line::GetDirection() { return direction_; }

std::vector<Point> Line::GetAllPoints() { return line_points_; }

// Morpion State methods ====================================================
void MorpionState::DoApplyAction(Action move) {
  Line newMove = *action_map_.at(move);
  Point newPoint;
  int pos;
  for (Point p : newMove.GetAllPoints()) {
    pos = p.y + (p.x * kNumRows);
    if (board_[pos] == 0) {
      board_[pos] = 1;
      newPoint = p;
      break;
    }
  }
  move_history_.emplace_back(newMove, newPoint);
  num_moves_ += 1;
  current_returns_ += 1;
}

std::vector<Action> MorpionState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  for (Line move : current_valid_moves_) {
    moves.push_back(move.GetAction());
  }
  sort(moves.begin(), moves.end());
  return moves;
}

std::string MorpionState::ActionToString(Player player,
                                         Action action_id) const {
  Line move = *action_map_.at(action_id);
  std::string action_str;
  for (Point p : move.GetAllPoints()) {
    absl::StrAppend(&action_str, p.ToString(), " ");
  }
  return action_str;
}

MorpionState::MorpionState(std::shared_ptr<const Game> game) : State(game) {
  // Initialize 4D starting points and find all possible lines on the board
  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      // Initialize starting points on board
      if ((i == 3 || i == 9) && j > 4 && j < 8) {
        board_[j + (i * kNumRows)] = 1;
      }
      if ((i == 4 || i == 8) && (j == 5 || j == 7)) {
        board_[j + (i * kNumRows)] = 1;
      }
      if ((i == 5 || i == 7) && ((j > 2 && j < 6) || (j > 6 && j < 10))) {
        board_[j + (i * kNumRows)] = 1;
      }
      if (i == 6 && ((j == 3) || (j == 9))) {
        board_[j + (i * kNumRows)] = 1;
      }
      // Get all possible lines on board (460)
      if (j + 3 < kNumCols) {
        all_lines_.emplace_back(Point(i, j), Point(i, j + 3));
      }
      if ((j + 3 < kNumCols) && (i + 3 < kNumRows)) {
        all_lines_.emplace_back(Point(i, j), Point(i + 3, j + 3));
      }
      if (i + 3 < kNumRows) {
        all_lines_.emplace_back(Point(i, j), Point(i + 3, j));
      }
      if ((j >= 3) && (i + 3 < kNumRows)) {
        all_lines_.emplace_back(Point(i, j), Point(i + 3, j - 3));
      }
    }
  }
  // For each line, store in a map of action # -> line object.
  for (Line& line : all_lines_) {
    action_map_[line.GetAction()] = &line;
  }
}

// Generate all valid lines / moves in current board state.
void MorpionState::getAllValidMoves() const {
  current_valid_moves_.clear();
  for (Line l : all_lines_) {
    // Check that exactly one point is empty.
    int count = 0;
    for (Point p : l.GetAllPoints()) {
      if (board_[p.y + (p.x * kNumRows)] == 1) {
        count++;
      }
    }
    if (count != 3) {
      continue;
    }
    // Check that line does not overlap any existing moves / lines.
    bool overlaps = false;
    for (const std::pair<Line, Point>& m : move_history_) {
      overlaps = l.CheckOverlap(m.first);
      if (overlaps) {
        break;
      }
    }
    if (overlaps) {
      continue;
    }
    current_valid_moves_.push_back(l);
  }
}

bool MorpionState::IsTerminal() const {
  getAllValidMoves();
  return current_valid_moves_.empty();
}

std::vector<double> MorpionState::Rewards() const {
  if (move_number_ == 0) {
    return {0.0};
  } else {
    return {1.0};
  }
}

std::vector<double> MorpionState::Returns() const { return {current_returns_}; }

std::string MorpionState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string MorpionState::ToString() const {
  std::string str;
  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      absl::StrAppend(&str, board_[i * kNumRows + j]);
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

std::string MorpionState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void MorpionState::UndoAction(Player player, Action move) {
  std::pair<Line, Point> last_move = move_history_.back();
  board_[last_move.second.x * kNumRows + last_move.second.y] = 0;
  move_history_.pop_back();
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> MorpionState::Clone() const {
  return std::unique_ptr<State>(new MorpionState(*this));
}

MorpionGame::MorpionGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace morpion_solitaire
}  // namespace open_spiel
