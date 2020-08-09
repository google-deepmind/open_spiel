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
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/utils/tensor_view.h"

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
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"board_size", GameParameter(kDefaultBoardSize)},
        // A default will be computed from the board_size
        {"wall_count",
         GameParameter(GameParameter::Type::kInt, /*is_mandatory=*/false)},
        {"ansi_color_output", GameParameter(false)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new QuoridorGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

class QuoridorState::SearchState {
  using DistanceAndMove = std::pair<int, Move>;

  class SearchQueue
      : public std::priority_queue<DistanceAndMove,
                                   std::vector<DistanceAndMove>,
                                   std::greater<DistanceAndMove>> {
   public:
    void clear() { c.clear(); }
    void reserve(int capacity) { c.reserve(capacity); }
  };

 public:
  explicit SearchState(int board_diameter) {
    int size = board_diameter * board_diameter;
    mark_.resize(size, false);
    on_shortest_path_.resize(size, false);
    distance_.resize(size, UndefinedDistance());
    queue_.reserve(size);
  }

  bool IsEmpty() const { return queue_.empty(); }

  void ClearSearchQueue() { queue_.clear(); }

  bool Push(int dist, Move move) {
    if (mark_[move.xy] == false) {
      mark_[move.xy] = true;
      queue_.emplace(dist, move);
      return true;
    } else {
      return false;
    }
  }

  Move Pop() {
    Move move = queue_.top().second;
    queue_.pop();
    return move;
  }

  void ResetSearchQueue() {
    std::fill(mark_.begin(), mark_.end(), false);
    queue_.clear();
  }

  void ResetDists() {
    std::fill(distance_.begin(), distance_.end(), UndefinedDistance());
  }

  void SetDist(Move move, int dist) { distance_[move.xy] = dist; }
  int GetDist(Move move) const { return distance_[move.xy]; }
  void SetOnShortestPath(Move move) { on_shortest_path_[move.xy] = true; }
  bool IsOnShortestPath(Move move) const { return on_shortest_path_[move.xy]; }

  static constexpr int UndefinedDistance() { return -1; }

 private:
  SearchQueue queue_;
  std::vector<bool> mark_;     // Whether this position has been pushed before.
  std::vector<int> distance_;  // Distance from player.
  std::vector<bool> on_shortest_path_;  // Is this position on a shortest path?
};

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

QuoridorState::QuoridorState(std::shared_ptr<const Game> game, int board_size,
                             int wall_count, bool ansi_color_output)
    : State(game),
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
  return GetMove(action_id % board_diameter_, action_id / board_diameter_);
}

std::vector<Action> QuoridorState::LegalActions() const {
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
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
    SearchState search_state(board_diameter_);
    SearchShortestPath(kPlayer1, &search_state);
    SearchShortestPath(kPlayer2, &search_state);
    for (int y = 0; y < board_diameter_ - 2; y += 2) {
      for (int x = 0; x < board_diameter_ - 2; x += 2) {
        Move h = GetMove(x, y + 1);
        if (IsValidWall(h, &search_state)) {
          moves.push_back(h.xy);
        }
        Move v = GetMove(x + 1, y);
        if (IsValidWall(v, &search_state)) {
          moves.push_back(v.xy);
        }
      }
    }
  }

  std::sort(moves.begin(), moves.end());
  return moves;
}

void QuoridorState::AddActions(Move cur, Offset offset,
                               std::vector<Action>* moves) const {
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

bool QuoridorState::IsValidWall(Move m, SearchState* search_state) const {
  Offset offset = (m.IsHorizontalWall() ? Offset(1, 0) : Offset(0, 1));

  if (IsWall(m + offset * 0) || IsWall(m + offset * 1) ||
      IsWall(m + offset * 2)) {
    // Already blocked by a wall.
    return false;
  }

  // Any wall that doesn't intersect with a shortest path is clearly legal.
  // Walls that do intersect might still be legal because there's another way
  // around, but that's more expensive to check.
  if (!search_state->IsOnShortestPath(m) &&
      !search_state->IsOnShortestPath(m + offset * 2)) {
    return true;
  }

  // If this wall doesn't connect two existing walls/edges, then it can't cut
  // any paths. Even connecting to a node where 3 other walls meet, but without
  // connecting them to anything else, can't cut any paths.
  int count = (
      // The 3 walls near the close end.
      (IsWall(m - offset * 2) || IsWall(m - offset + offset.rotate_left()) ||
       IsWall(m - offset + offset.rotate_right())) +
      // The 3 walls near the far end.
      (IsWall(m + offset * 4) ||
       IsWall(m + offset * 3 + offset.rotate_left()) ||
       IsWall(m + offset * 3 + offset.rotate_right())) +
      // The 2 walls in the middle.
      (IsWall(m + offset + offset.rotate_left()) ||
       IsWall(m + offset + offset.rotate_right())));
  if (count <= 1) return true;

  // Do a full search to verify both players can get to their respective goals.
  return (SearchEndZone(kPlayer1, m, m + offset * 2, search_state) &&
          SearchEndZone(kPlayer2, m, m + offset * 2, search_state));
}

bool QuoridorState::SearchEndZone(QuoridorPlayer p, Move wall1, Move wall2,
                                  SearchState* search_state) const {
  search_state->ResetSearchQueue();
  Offset dir(1, 0);  // Direction is arbitrary. Queue will make it fast.
  int goal = end_zone_[p];
  int goal_dir = (goal == 0 ? -1 : 1);  // Sort for shortest dist in a min-heap.
  search_state->Push(0, player_loc_[p]);
  while (!search_state->IsEmpty()) {
    Move c = search_state->Pop();
    for (int i = 0; i < 4; ++i) {
      Move wall = c + dir;
      if (!IsWall(wall) && wall != wall1 && wall != wall2) {
        Move move = c + dir * 2;
        if (move.y == goal) {
          return true;
        }
        search_state->Push(goal_dir * (goal - move.y), move);
      }
      dir = dir.rotate_left();
    }
  }

  return false;
}

void QuoridorState::SearchShortestPath(QuoridorPlayer p,
                                       SearchState* search_state) const {
  search_state->ResetSearchQueue();
  search_state->ResetDists();
  Offset dir(1, 0);  // Direction is arbitrary. Queue will make it fast.
  int goal = end_zone_[p];
  int goal_dir = (goal == 0 ? -1 : 1);  // Sort for shortest dist in a min-heap.
  search_state->Push(0, player_loc_[p]);
  search_state->SetDist(player_loc_[p], 0);
  Move goal_found = GetMove(-1, -1);  // invalid

  // A* search for the end-zone, keeping distances to each cell.
  while (!search_state->IsEmpty()) {
    Move c = search_state->Pop();
    int dist = search_state->GetDist(c);
    for (int i = 0; i < 4; ++i) {
      Move wall = c + dir;
      if (!IsWall(wall)) {
        Move move = c + dir * 2;
        if (move.y == goal) {
          search_state->SetDist(move, dist + 1);
          search_state->ClearSearchQueue();  // Break out of the search.
          goal_found = move;
          break;
        }
        if (search_state->Push(dist + 1 + goal_dir * (goal - move.y), move)) {
          search_state->SetDist(move, dist + 1);
        }
      }
      dir = dir.rotate_left();
    }
  }

  // Trace the way back, setting them to be on a shortest path.
  Move current = goal_found;
  int dist = search_state->GetDist(current);
  while (current != player_loc_[p]) {
    for (int i = 0; i < 4; ++i) {
      Move wall = current + dir;
      if (!IsWall(wall)) {
        Move move = current + dir * 2;
        int dist2 = search_state->GetDist(move);
        if (dist2 != search_state->UndefinedDistance() && dist2 + 1 == dist) {
          search_state->SetOnShortestPath(wall);
          current = move;
          dist = dist2;
          break;
        }
      }
      dir = dir.rotate_left();
    }
  }
}

std::string QuoridorState::ActionToString(Player player,
                                          Action action_id) const {
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
    coord = esc + "[1;37m";                  // bright white
    white = esc + "[1;33m" + " @ " + reset;  // bright yellow
    black = esc + "[1;34m" + " @ " + reset;  // bright blue
  }

  std::ostringstream out;
  out << "Board size: " << board_size_ << ", walls: " << wall_count_[kPlayer1]
      << ", " << wall_count_[kPlayer2] << "\n";

  // Top x coords.
  for (int x = 0; x < board_size_; ++x) {
    out << "   " << coord << static_cast<char>('a' + x);
  }
  out << reset << '\n';

  for (int y = 0; y < board_diameter_; ++y) {
    if (y % 2 == 0) {
      if (y / 2 + 1 < 10) out << " ";
      out << coord << (y / 2 + 1) << reset;  // Leading y coord.
    } else {
      out << "  ";  // Wall lines.
    }

    for (int x = 0; x < board_diameter_; ++x) {
      QuoridorPlayer p = GetPlayer(GetMove(x, y));
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

std::string QuoridorState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string QuoridorState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void QuoridorState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(
      values, {kCellStates + kNumPlayers, static_cast<int>(board_.size())},
      true);

  for (int i = 0; i < board_.size(); ++i) {
    if (board_[i] < kCellStates) {
      view[{static_cast<int>(board_[i]), i}] = 1.0;
    }
    view[{kCellStates + kPlayer1, i}] = wall_count_[kPlayer1];
    view[{kCellStates + kPlayer2, i}] = wall_count_[kPlayer2];
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

  ++moves_made_;
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
      board_size_(ParameterValue<int>("board_size")),
      wall_count_(
          ParameterValue<int>("wall_count", board_size_ * board_size_ / 8)),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")) {}

}  // namespace quoridor
}  // namespace open_spiel
