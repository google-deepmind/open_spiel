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

#include "open_spiel/games/quoridor.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace open_spiel {
namespace quoridor {
namespace {


// Facts about the game.
const GameType kGameType{
    /*short_name=*/"quoridor",
    /*long_name=*/"Quoridor",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/
    {
        {"board_size",
         GameType::ParameterSpec{GameParameter::Type::kInt, false}},
        {"wall_count",
         GameType::ParameterSpec{GameParameter::Type::kInt, false}},
        {"ansi_color_output",
         GameType::ParameterSpec{GameParameter::Type::kBool, false}},
    }};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new QuoridorGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

std::string Move::ToString() const {
  std::string out = absl::StrCat(
      std::string(1, static_cast<char>('a' + (x / 2))), (y / 2) + 1);
  if (!IsWall()) {
    return out;
  } else if (IsVerticalWall()) {
    return absl::StrCat(out, "v");
  } else if (IsHorizontalWall()) {
    return absl::StrCat(out, "h");
  }
  return "invalid move";
}


QuoridorState::QuoridorState(int board_size, int wall_count,
                             bool ansi_color_output)
    : State((board_size * 2 - 1) * (board_size * 2 - 1),  // Diameter squared.
            kNumPlayers),
      board_size_(board_size),
      board_diameter_(board_size * 2 - 1),
      ansi_color_output_(ansi_color_output) {
  board_.resize(board_diameter_ * board_diameter_, kPlayerNone);
  wall_count_[kPlayer1] = wall_count;
  wall_count_[kPlayer2] = wall_count;
  int start_x = board_size - (board_size % 2);
  player_loc_[kPlayer1] = GetMove(start_x, board_diameter_ - 1);
  player_loc_[kPlayer2] = GetMove(start_x, 0);
  SetPlayer(player_loc_[kPlayer1], kPlayer1, kPlayerNone);
  SetPlayer(player_loc_[kPlayer2], kPlayer2, kPlayerNone);
  end_zone_[kPlayer1] = player_loc_[kPlayer2].y;
  end_zone_[kPlayer2] = player_loc_[kPlayer1].y;
}

Move QuoridorState::ActionToMove(Action action_id) const {
  return GetMove(action_id % board_diameter_,
                 action_id / board_diameter_);
}

std::vector<Action> QuoridorState::LegalActions() const {
  std::vector<Action> moves;

  int max_moves = 5;  // Max pawn moves, including jumps.
  if (wall_count_[current_player_] > 0) {
    max_moves += 2 * (board_size_ - 1) * (board_size_ - 1);  // Max wall moves.
  }
  moves.reserve(max_moves);

  // Pawn moves.
  Move cur = player_loc_[current_player_];
  AddActions(cur, Offset(1, 0), &moves);
  AddActions(cur, Offset(0, 1), &moves);
  AddActions(cur, Offset(-1, 0), &moves);
  AddActions(cur, Offset(0, -1), &moves);

  // Wall placements.
  if (wall_count_[current_player_] > 0) {
    for (int y = 0; y < board_diameter_ - 2; y += 2) {
      for (int x = 0; x < board_diameter_ - 2; x += 2) {
        Move h = GetMove(x, y + 1);
        if (IsValidWall(h)) {
          moves.push_back(h.xy);
        }
        Move v = GetMove(x + 1, y);
        if (IsValidWall(v)) {
          moves.push_back(v.xy);
        }
      }
    }
  }

  std::sort(moves.begin(), moves.end());
  return moves;
}

void QuoridorState::AddActions(Move cur, Offset offset,
                               std::vector<Action> *moves) const {
  SPIEL_CHECK_FALSE(cur.IsWall());

  if (IsWall(cur + offset)) {
    // Hit a wall or edge in this direction.
    return;
  }

  Move forward = cur + offset * 2;
  if (GetPlayer(forward) == kPlayerNone) {
    // Normal single step in this direction.
    moves->push_back(forward.xy);
    return;
  }
  // Other player, so which jumps are valid?

  if (!IsWall(cur + offset * 3)) {
    // A normal jump is allowed. We know that spot is empty.
    moves->push_back((cur + offset * 4).xy);
    return;
  }
  // We are jumping over the other player against a wall, which side jumps are
  // valid?

  Offset left = offset.rotate_left();
  if (!IsWall(forward + left)) {
    moves->push_back((forward + left * 2).xy);
  }
  Offset right = offset.rotate_right();
  if (!IsWall(forward + right)) {
    moves->push_back((forward + right * 2).xy);
  }
}

bool QuoridorState::IsValidWall(Move m) const {
  Offset offset = (m.IsHorizontalWall() ? Offset(1, 0) : Offset(0, 1));

  if (IsWall(m + offset * 0) ||
      IsWall(m + offset * 1) ||
      IsWall(m + offset * 2)) {
    // Already blocked by a wall.
    return false;
  }

  // If this wall doesn't connect two existing walls/edges, then it can't cut
  // any paths. Even connecting to a node where 3 other walls meet, but without
  // connecting them to anything else, can't cut any paths.
  int count = (
      // The 3 walls near the close end.
      (IsWall(m - offset * 2) ||
       IsWall(m - offset + offset.rotate_left()) ||
       IsWall(m - offset + offset.rotate_right())) +
      // The 3 walls near the far end.
      (IsWall(m + offset * 4) ||
       IsWall(m + offset * 3 + offset.rotate_left()) ||
       IsWall(m + offset * 3 + offset.rotate_right())) +
      // The 2 walls in the middle.
      (IsWall(m + offset + offset.rotate_left()) ||
       IsWall(m + offset + offset.rotate_right())));
  if (count <= 1)
    return true;

  // Do a full search to verify both players can get to their respective goals.
  return (SearchEndZone(kPlayer1, m, m + offset * 2) &&
          SearchEndZone(kPlayer2, m, m + offset * 2));
}

bool QuoridorState::SearchEndZone(Player p, Move wall1, Move wall2) const {
  std::vector<bool> mark(board_diameter_ * board_diameter_, false);
  Offset dir(1, 0);  // Direction is arbitrary. Queue will make it fast.
  int goal = end_zone_[p];
  int goal_dir = (goal == 0 ? 1 : -1);  // Sort for shortest dist in a max-heap.
  std::priority_queue<std::pair<int, Move>> queue;  // <distance to goal, move>
  queue.push(std::make_pair(0, player_loc_[p]));
  while (!queue.empty()) {
    // Ignore the distance. It is only for sorting.
    Move c = queue.top().second;
    queue.pop();
    mark[c.xy] = true;
    for (int i = 0; i < 4; i++) {
      Move wall = c + dir;
      Move move = c + dir * 2;
      if (!IsWall(wall) && wall != wall1 && wall != wall2 && !mark[move.xy]) {
        if (move.y == goal)
          return true;
        queue.push(std::make_pair(goal_dir * (goal - move.y), move));
      }
      dir = dir.rotate_left();
    }
  }

  return false;
}

std::string QuoridorState::ActionToString(int player, Action action_id) const {
  return ActionToMove(action_id).ToString();
}

std::string QuoridorState::ToString() const {
  // Generates something like:
  // Board size: 5, walls: 0, 0
  //    a   b   c   d   e
  //  1 . | .   .   .   .
  //      +        ---+---
  //  2 . | . | .   .   .
  //          +
  //  3 .   . | O   @   .
  //   ---+---
  //  4 . | .   .   .   .
  //      +    ---+---
  //  5 . | .   .   .   .

  std::string white = " O ";
  std::string black = " @ ";
  std::string coord = "";
  std::string reset = "";
  if (ansi_color_output_) {
    std::string esc = "\033";
    reset = esc + "[0m";
    coord = esc + "[1;37m";  // bright white
    white = esc + "[1;33m" + " @ " + reset;  // bright yellow
    black = esc + "[1;34m" + " @ " + reset;  // bright blue
  }

  std::ostringstream out;
  out << "Board size: " << board_size_ << ", walls: " << wall_count_[kPlayer1]
      << ", " << wall_count_[kPlayer2] << "\n";

  // Top x coords.
  for (int x = 0; x < board_size_; x++) {
    out << "   " << coord << static_cast<char>('a' + x);
  }
  out << reset << '\n';

  for (int y = 0; y < board_diameter_; y++) {
    if (y % 2 == 0) {
      if (y / 2 + 1 < 10) out <<  " ";
      out << coord << (y / 2 + 1) << reset;  // Leading y coord.
    } else {
      out << "  ";  // Wall lines.
    }

    for (int x = 0; x < board_diameter_; x++) {
      Player p = GetPlayer(GetMove(x, y));
      if (x % 2 == 0 && y % 2 == 0) {
        out << (p == kPlayer1 ? white : p == kPlayer2 ? black : " . ");
      } else if (x % 2 == 1 && y % 2 == 1) {
        out << (p == kPlayerWall ? "+" : " ");
      } else if (x % 2 == 1) {
        out << (p == kPlayerWall ? "|" : " ");
      } else if (y % 2 == 1) {
        out << (p == kPlayerWall ? "---" : "   ");
      }
    }
    out << '\n';
  }
  return out.str();
}

std::vector<double> QuoridorState::Returns() const {
  if (outcome_ == kPlayer1) return {1, -1};
  if (outcome_ == kPlayer2) return {-1, 1};
  if (outcome_ == kPlayerDraw) return {0, 0};
  return {0, 0};  // Unfinished
}

std::string QuoridorState::InformationState(int player) const {
  return HistoryString();
}

std::string QuoridorState::Observation(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void QuoridorState::ObservationAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values->begin(), values->end(), 0.);
  values->resize(board_.size() * (kCellStates + kNumPlayers), 0.);

  auto set_value = [&values, this](int plane, int i, double value) {
    (*values)[i + plane * board_.size()] = value;
  };

  for (int i = 0; i < board_.size(); ++i) {
    if (board_[i] < kCellStates) {
      set_value(static_cast<int>(board_[i]), i, 1.0);
    }
    set_value(kCellStates + kPlayer1, i, wall_count_[kPlayer1]);
    set_value(kCellStates + kPlayer2, i, wall_count_[kPlayer2]);
  }
}

void QuoridorState::DoApplyAction(Action action) {
  SPIEL_CHECK_EQ(board_[action], kPlayerNone);
  SPIEL_CHECK_EQ(outcome_, kPlayerNone);

  Move move = ActionToMove(action);
  SPIEL_CHECK_TRUE(move.IsValid());

  if (move.IsWall()) {
    Offset offset = (move.IsHorizontalWall() ? Offset(1, 0) : Offset(0, 1));
    SetPlayer(move + offset * 0, kPlayerWall, kPlayerNone);
    SetPlayer(move + offset * 1, kPlayerWall, kPlayerNone);
    SetPlayer(move + offset * 2, kPlayerWall, kPlayerNone);
    wall_count_[current_player_] -= 1;
  } else {
    SetPlayer(player_loc_[current_player_], kPlayerNone, current_player_);
    SetPlayer(move, current_player_, kPlayerNone);
    player_loc_[current_player_] = move;

    if (move.y == end_zone_[current_player_]) {
      outcome_ = current_player_;
    }
  }

  moves_made_++;
  if (moves_made_ >= kMaxGameLengthFactor * board_size_ * board_size_) {
    outcome_ = kPlayerDraw;
  }

  current_player_ = (current_player_ == kPlayer1 ? kPlayer2 : kPlayer1);
}

std::unique_ptr<State> QuoridorState::Clone() const {
  return std::unique_ptr<State>(new QuoridorState(*this));
}

QuoridorGame::QuoridorGame(const GameParameters& params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size", kDefaultBoardSize)),
      wall_count_(ParameterValue<int>("wall_count",
                                      board_size_ * board_size_ / 8)),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output", false)) {
}

}  // namespace quoridor
}  // namespace open_spiel
