// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/quoridor.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"
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
    /*max_num_players=*/kMaxNumPlayers,
    /*min_num_players=*/kMinNumPlayers,
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
        {"players", GameParameter(kMinNumPlayers, false)},
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
  players_.resize(num_players_);
  // Account for order of turns (order of play is clockwise)
  if (num_players_ == 2) {
    players_[0] = kPlayer1;
    players_[1] = kPlayer2;
  } else if (num_players_ == 3) {
    players_[0] = kPlayer1;
    players_[1] = kPlayer3;
    players_[2] = kPlayer2;
  } else if (num_players_ == 4) {
    players_[0] = kPlayer1;
    players_[1] = kPlayer3;
    players_[2] = kPlayer2;
    players_[3] = kPlayer4;
  }
  wall_count_.resize(num_players_);
  player_loc_.resize(num_players_);
  end_zone_.resize(num_players_);
  for (int i = 0; i < num_players_; ++i) {
    wall_count_[players_[i]] = wall_count;
    InitializePlayer(players_[i]);
  }
}

void QuoridorState::InitializePlayer(QuoridorPlayer p) {
  int center_field = board_size_ - (board_size_ % 2);
  if (p == kPlayer1) {
    player_loc_[p] = GetMove(center_field, board_diameter_ - 1);
    SetPlayer(player_loc_[p], p, kPlayerNone);
    end_zone_[p] = 0;
    return;
  }
  if (p == kPlayer2) {
    player_loc_[p] = GetMove(center_field, 0);
    SetPlayer(player_loc_[p], kPlayer2, kPlayerNone);
    end_zone_[p] = board_diameter_ - 1;
    return;
  }
  if (p == kPlayer3) {
    player_loc_[p] = GetMove(0, center_field);
    SetPlayer(player_loc_[p], p, kPlayerNone);
    end_zone_[p] = board_diameter_ - 1;
    return;
  }
  if (p == kPlayer4) {
    player_loc_[p] = GetMove(board_diameter_ - 1, center_field);
    SetPlayer(player_loc_[p], p, kPlayerNone);
    end_zone_[p] = 0;
    return;
  }
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
    for (int i = 0; i < num_players_; ++i) {
      SearchShortestPath(players_[i], &search_state);
    }
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

  // If no action is possible add 'pass' action to list of moves
  if (moves.empty()) {
    moves.push_back(cur.xy);
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
    // In two-players: A normal jump is allowed. We know that spot is empty.
    // In >2 players, must check.
    if (GetPlayer(cur + offset * 4) == kPlayerNone) {
      moves->push_back((cur + offset * 4).xy);
      return;
    } else {
      return;
    }
  }
  // We are jumping over the other player against a wall, which side jumps are
  // valid?

  Offset left = offset.rotate_left();
  if (!IsWall(forward + left)) {
    if (GetPlayer(forward + left * 2) == kPlayerNone) {
      moves->push_back((forward + left * 2).xy);
    }
  }
  Offset right = offset.rotate_right();
  if (!IsWall(forward + right)) {
    if (GetPlayer(forward + right * 2) == kPlayerNone) {
      moves->push_back((forward + right * 2).xy);
    }
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
  bool pathExists = true;
  for (int i = 0; i < num_players_; ++i) {
    pathExists = pathExists &&
                 SearchEndZone(players_[i], m, m + offset * 2, search_state);
  }
  return pathExists;
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
        int moveCoord;
        if (p == kPlayer1 || p == kPlayer2) {
          moveCoord = move.y;
        } else if (p == kPlayer3 || p == kPlayer4) {
          moveCoord = move.x;
        } else {
          SpielFatalError("Case not handled for player in SearchEndZone.");
        }
        if (moveCoord == goal) {
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
        int moveCoord;
        if (p == kPlayer1 || p == kPlayer2) {
          moveCoord = move.y;
        } else if (p == kPlayer3 || p == kPlayer4) {
          moveCoord = move.x;
        } else {
          SpielFatalError("Case not handled for player in SearchShortestPath");
        }
        if (moveCoord == goal) {
          search_state->SetDist(move, dist + 1);
          search_state->ClearSearchQueue();  // Break out of the search.
          goal_found = move;
          break;
        }
        if (search_state->Push(dist + 1 + goal_dir * (goal - moveCoord),
                               move)) {
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
  //  1 . | .   .   .   . 1
  //      +        ---+---
  //  2 . | . | .   .   . 2
  //          +
  //  3 .   . | O   @   . 3
  //   ---+---
  //  4 . | .   .   .   . 4
  //      +    ---+---
  //  5 . | .   .   .   . 5
  //    a   b   c   d   e

  std::string reset;
  std::array<std::string, 4> colors, coords;
  if (ansi_color_output_) {
    std::string esc = "\033";
    reset = esc + "[0m";
    coords[0] = esc + "[1;33m";
    coords[1] = esc + "[1;34m";
    coords[2] = esc + "[1;35m";
    coords[3] = esc + "[1;36m";
    colors[0] = esc + "[1;33m" + " O " + reset;
    colors[1] = esc + "[1;34m" + " @ " + reset;
    colors[2] = esc + "[1;35m" + " # " + reset;
    colors[3] = esc + "[1;36m" + " % " + reset;
  } else {
    std::string reset = "";
    coords[0] = "";
    coords[1] = "";
    coords[2] = "";
    coords[3] = "";
    colors[0] = " 0 ";
    colors[1] = " @ ";
    colors[2] = " # ";
    colors[3] = " % ";
  }

  std::ostringstream out;
  out << "Board size: " << board_size_ << ", walls: ";
  for (int i = 0; i < num_players_; ++i) {
    out << wall_count_[players_[i]];
    if (i < num_players_ - 1) out << ", ";
  }
  out << "\n";

  // Top x coords.
  for (int x = 0; x < board_size_; ++x) {
    out << "   " << coords[1] << static_cast<char>('a' + x);
  }
  out << reset << '\n';

  for (int y = 0; y < board_diameter_; ++y) {
    if (y % 2 == 0) {
      if (y / 2 + 1 < 10) out << " ";
      out << coords[2] << (y / 2 + 1) << reset;  // Leading y coord.
    } else {
      out << "  ";  // Wall lines.
    }

    for (int x = 0; x < board_diameter_; ++x) {
      QuoridorPlayer p = GetPlayer(GetMove(x, y));
      if (x % 2 == 0 && y % 2 == 0) {
        bool playerFound = false;
        for (int i = 0; i < num_players_; ++i) {
          if (p == players_[i]) {
            out << colors[players_[i]];
            playerFound = true;
          }
        }
        if (!playerFound) {
          out << " . ";
        }
      } else if (x % 2 == 1 && y % 2 == 1) {
        out << (p == kPlayerWall ? "+" : " ");
      } else if (x % 2 == 1) {
        out << (p == kPlayerWall ? "|" : " ");
      } else if (y % 2 == 1) {
        out << (p == kPlayerWall ? "---" : "   ");
      }
    }
    if (y % 2 == 0) {
      if (y / 2 + 1 < 10) out << " ";
      out << coords[3] << (y / 2 + 1) << reset;  // y coord on the right.
    } else {
      out << "  ";  // Wall lines.
    }
    out << '\n';
  }
  // Bottom x coords.
  for (int x = 0; x < board_size_; ++x) {
    out << "   " << coords[0] << static_cast<char>('a' + x);
  }
  out << reset << '\n';
  return out.str();
}

std::vector<double> QuoridorState::Returns() const {
  std::vector<double> res(num_players_, 0.0);
  for (int i = 0; i < num_players_; ++i) {
    if (outcome_ == players_[i]) {
      // If someone as won, set their reward to +1 and all the others to
      // -1 / (num_players - 1).
      std::fill(res.begin(), res.end(), -1.0 / (num_players_ - 1));
      res[i] = 1.0;
      break;
    }
  }
  return res;
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
      values, {NumCellStates() + num_players_, static_cast<int>(board_.size())},
      true);

  for (int i = 0; i < board_.size(); ++i) {
    if (board_[i] < NumCellStates()) {
      view[{static_cast<int>(board_[i]), i}] = 1.0;
    }
    for (int j = 0; j < num_players_; ++j) {
      view[{NumCellStates() + players_[j], i}] = wall_count_[players_[j]];
    }
  }
}

void QuoridorState::DoApplyAction(Action action) {
  // If players is forced to pass it is valid to stay in place, on a field where
  // there is already a player
  if (board_[action] != current_player_) {
    SPIEL_CHECK_EQ(board_[action], kPlayerNone);
  }
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

    int end_zone_coord;
    if (current_player_ == kPlayer1 || current_player_ == kPlayer2) {
      end_zone_coord = move.y;
    } else {
      end_zone_coord = move.x;
    }

    outcome_ = kPlayerNone;
    if (end_zone_coord == end_zone_[current_player_]) {
      outcome_ = current_player_;
    }
  }

  ++moves_made_;
  if (moves_made_ >= kMaxGameLengthFactor * board_size_ * board_size_) {
    outcome_ = kPlayerDraw;
  }

  current_player_index_ += 1;
  if (current_player_index_ == num_players_) current_player_index_ = 0;
  current_player_ = players_[current_player_index_];
}

std::unique_ptr<State> QuoridorState::Clone() const {
  return std::unique_ptr<State>(new QuoridorState(*this));
}

QuoridorGame::QuoridorGame(const GameParameters& params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      wall_count_(
          ParameterValue<int>("wall_count", board_size_ * board_size_ / 8)),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")),
      num_players_(ParameterValue<int>("players")) {}

}  // namespace quoridor
}  // namespace open_spiel
