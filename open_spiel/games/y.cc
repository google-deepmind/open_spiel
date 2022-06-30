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

#include "open_spiel/games/y.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace y_game {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"y",
                         /*long_name=*/"Y Connection Game",
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
                             {"ansi_color_output", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new YGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// The board is represented as a flattened 2d array of the form:
//   1 2 3
// A 0 1 2     0 1 2     0 1 2
// B 3 4 5 <=> 3 4   <=>  3 4
// C 6 7 8     6           6
//
// Neighbors are laid out in this pattern:
//   0   1           0  1
// 5   X   2 <=>  5  X  2
//   4   3        4  3

// Direct neighbors of a cell, clockwise.
constexpr std::array<Move, kMaxNeighbors> neighbor_offsets = {
    Move(0, -1, kMoveOffset), Move(1, -1, kMoveOffset),
    Move(1, 0, kMoveOffset),  Move(0, 1, kMoveOffset),
    Move(-1, 1, kMoveOffset), Move(-1, 0, kMoveOffset),
};

// Precomputed list of neighbors per board_size: [board_size][cell][direction]
std::vector<NeighborList> neighbor_list;

NeighborList gen_neighbors(int board_size) {
  NeighborList out;
  out.resize(board_size * board_size);
  for (int y = 0; y < board_size; y++) {
    for (int x = 0; x < board_size; x++) {
      int xy = x + y * board_size;  // Don't use Move.xy so it works off-board.
      for (int dir = 0; dir < neighbor_offsets.size(); dir++) {
        Move offset = neighbor_offsets[dir];
        out[xy][dir] = Move(x + offset.x, y + offset.y, board_size);
      }
    }
  }
  return out;
}

const NeighborList& get_neighbors(int board_size) {
  if (board_size >= neighbor_list.size()) {
    neighbor_list.resize(board_size + 1);
  }
  if (neighbor_list[board_size].empty()) {
    neighbor_list[board_size] = gen_neighbors(board_size);
  }
  return neighbor_list[board_size];
}

}  // namespace

int Move::Edge(int board_size) const {
  if (!OnBoard()) return 0;

  return (x == 0 ? (1 << 0) : 0) | (y == 0 ? (1 << 1) : 0) |
         (x + y == board_size - 1 ? (1 << 2) : 0);
}

std::string Move::ToString() const {
  if (xy == kMoveUnknown) return "unknown";
  if (xy == kMoveNone) return "none";
  return absl::StrCat(std::string(1, static_cast<char>('a' + x)), y + 1);
}

YState::YState(std::shared_ptr<const Game> game, int board_size,
               bool ansi_color_output)
    : State(game),
      board_size_(board_size),
      neighbors(get_neighbors(board_size)),
      ansi_color_output_(ansi_color_output) {
  board_.resize(board_size * board_size);
  for (int i = 0; i < board_.size(); i++) {
    Move m = ActionToMove(i);
    board_[i] = Cell((m.OnBoard() ? kPlayerNone : kPlayerInvalid), i,
                     m.Edge(board_size));
  }
}

Move YState::ActionToMove(Action action_id) const {
  return Move(action_id % board_size_, action_id / board_size_, board_size_);
}

std::vector<Action> YState::LegalActions() const {
  // Can move in any empty cell.
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  moves.reserve(board_.size() - moves_made_);
  for (int cell = 0; cell < board_.size(); ++cell) {
    if (board_[cell].player == kPlayerNone) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::string YState::ActionToString(Player player, Action action_id) const {
  return ActionToMove(action_id).ToString();
}

std::string YState::ToString() const {
  // Generates something like:
  //  a b c d e f g h i j k
  // 1 O @ O O . @ @ O O @ O
  //  2 . O O . O @ @ . O O
  //   3 . O @ @ O @ O O @
  //    4 O O . @ . @ O O
  //     5 . . . @[@]@ O
  //      6 @ @ @ O O @
  //       7 @ . O @ O
  //        8 . @ @ O
  //         9 @ @ .
  //         10 O .
  //          11 @

  std::string white = "O";
  std::string black = "@";
  std::string empty = ".";
  std::string coord = "";
  std::string reset = "";
  if (ansi_color_output_) {
    std::string esc = "\033";
    reset = esc + "[0m";
    coord = esc + "[1;37m";  // bright white
    empty = reset + ".";
    white = esc + "[1;33m" + "@";  // bright yellow
    black = esc + "[1;34m" + "@";  // bright blue
  }

  std::ostringstream out;

  // Top x coords.
  out << ' ';
  for (int x = 0; x < board_size_; x++) {
    out << ' ' << coord << static_cast<char>('a' + x);
  }
  out << '\n';

  for (int y = 0; y < board_size_; y++) {
    out << std::string(y + ((y + 1) < 10), ' ');  // Leading space.
    out << coord << (y + 1);                      // Leading y coord.

    bool found_last = false;
    for (int x = 0; x < board_size_ - y; x++) {
      Move pos(x, y, board_size_);

      // Spacing and last-move highlight.
      if (found_last) {
        out << coord << ']';
        found_last = false;
      } else if (last_move_ == pos) {
        out << coord << '[';
        found_last = true;
      } else {
        out << ' ';
      }

      // Actual piece.
      Player p = board_[pos.xy].player;
      if (p == kPlayerNone) out << empty;
      if (p == kPlayer1) out << white;
      if (p == kPlayer2) out << black;
    }
    if (found_last) {
      out << coord << ']';
    }
    out << '\n';
  }
  out << reset;
  return out.str();
}

std::vector<double> YState::Returns() const {
  if (outcome_ == kPlayer1) return {1, -1};
  if (outcome_ == kPlayer2) return {-1, 1};
  return {0, 0};  // Unfinished
}

std::string YState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string YState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int PlayerRelative(YPlayer state, Player current) {
  switch (state) {
    case kPlayer1:
      return current == 0 ? 0 : 1;
    case kPlayer2:
      return current == 1 ? 0 : 1;
    case kPlayerNone:
      return 2;
    default:
      SpielFatalError("Unknown player type.");
  }
}

void YState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, static_cast<int>(board_.size())},
                     true);
  for (int i = 0; i < board_.size(); ++i) {
    if (board_[i].player != kPlayerInvalid) {
      view[{PlayerRelative(board_[i].player, player), i}] = 1.0;
    }
  }
}

void YState::DoApplyAction(Action action) {
  SPIEL_CHECK_EQ(board_[action].player, kPlayerNone);
  SPIEL_CHECK_EQ(outcome_, kPlayerNone);

  Move move = ActionToMove(action);
  SPIEL_CHECK_TRUE(move.OnBoard());

  last_move_ = move;
  board_[move.xy].player = current_player_;
  moves_made_++;

  for (const Move& m : neighbors[move.xy]) {
    if (m.OnBoard() && current_player_ == board_[m.xy].player) {
      JoinGroups(move.xy, m.xy);
    }
  }

  if (board_[FindGroupLeader(move.xy)].edge == 0x7) {  // ie all 3 edges.
    outcome_ = current_player_;
  }

  current_player_ = (current_player_ == kPlayer1 ? kPlayer2 : kPlayer1);
}

int YState::FindGroupLeader(int cell) {
  int parent = board_[cell].parent;
  if (parent != cell) {
    do {  // Follow the parent chain up to the group leader.
      parent = board_[parent].parent;
    } while (parent != board_[parent].parent);
    // Do path compression, but only the current one to avoid recursion.
    board_[cell].parent = parent;
  }
  return parent;
}

bool YState::JoinGroups(int cell_a, int cell_b) {
  int leader_a = FindGroupLeader(cell_a);
  int leader_b = FindGroupLeader(cell_b);

  if (leader_a == leader_b)  // Already the same group.
    return true;

  if (board_[leader_a].size < board_[leader_b].size) {
    // Force group a's subtree to be bigger.
    std::swap(leader_a, leader_b);
  }

  // Group b joins group a.
  board_[leader_b].parent = leader_a;
  board_[leader_a].size += board_[leader_b].size;
  board_[leader_a].edge |= board_[leader_b].edge;

  return false;
}

std::unique_ptr<State> YState::Clone() const {
  return std::unique_ptr<State>(new YState(*this));
}

YGame::YGame(const GameParameters& params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")) {}

}  // namespace y_game
}  // namespace open_spiel
