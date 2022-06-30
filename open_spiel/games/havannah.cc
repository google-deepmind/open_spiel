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

#include "open_spiel/games/havannah.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace havannah {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"havannah",
                         /*long_name=*/"Havannah",
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
                             {"swap", GameParameter(false)},
                             {"ansi_color_output", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HavannahGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// The board is represented as a flattened 2d array of the form:
//   1 2 3
// a 0 1 2    0 1       0 1
// b 3 4 5 => 3 4 5 => 3 4 5
// c 6 7 8      7 8     7 8
//
// Neighbors are laid out in this pattern:
//   0   1
// 5   X   2
//   4   3

// Direct neighbors of a cell, clockwise.
constexpr std::array<Move, kMaxNeighbors> neighbor_offsets = {
    Move(-1, -1, kMoveOffset), Move(0, -1, kMoveOffset),
    Move(1, 0, kMoveOffset),   Move(1, 1, kMoveOffset),
    Move(0, 1, kMoveOffset),   Move(-1, 0, kMoveOffset),
};

// Precomputed list of neighbors per board_size: [board_size][cell][direction]
std::vector<NeighborList> neighbor_list;

NeighborList gen_neighbors(int board_size) {
  int diameter = board_size * 2 - 1;
  NeighborList out;
  out.resize(diameter * diameter);
  for (int y = 0; y < diameter; y++) {
    for (int x = 0; x < diameter; x++) {
      int xy = x + y * diameter;  // Don't use Move.xy so it works off-board.
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

// Number of set bits in each 6-bit integer.
// Python code to compute these values: [bin(i).count("1") for i in range(64)]
constexpr int kBitsSetTable64[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
    3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
};

}  // namespace

int Move::Corner(int board_size) const {
  if (!OnBoard()) return 0;

  int m = board_size - 1;
  int e = m * 2;

  if (x == 0 && y == 0) return 1 << 0;
  if (x == m && y == 0) return 1 << 1;
  if (x == e && y == m) return 1 << 2;
  if (x == e && y == e) return 1 << 3;
  if (x == m && y == e) return 1 << 4;
  if (x == 0 && y == m) return 1 << 5;

  return 0;
}

int Move::Edge(int board_size) const {
  if (!OnBoard()) return 0;

  int m = board_size - 1;
  int e = m * 2;

  if (y == 0 && x != 0 && x != m) return 1 << 0;
  if (x - y == m && x != m && x != e) return 1 << 1;
  if (x == e && y != m && y != e) return 1 << 2;
  if (y == e && x != e && x != m) return 1 << 3;
  if (y - x == m && x != m && x != 0) return 1 << 4;
  if (x == 0 && y != m && y != 0) return 1 << 5;

  return 0;
}

std::string Move::ToString() const {
  if (xy == kMoveUnknown) return "unknown";
  if (xy == kMoveNone) return "none";
  return absl::StrCat(std::string(1, static_cast<char>('a' + x)), y + 1);
}

int HavannahState::Cell::NumCorners() const { return kBitsSetTable64[corner]; }
int HavannahState::Cell::NumEdges() const { return kBitsSetTable64[edge]; }

HavannahState::HavannahState(std::shared_ptr<const Game> game, int board_size,
                             bool ansi_color_output, bool allow_swap)
    : State(game),
      board_size_(board_size),
      board_diameter_(board_size * 2 - 1),
      valid_cells_((board_size * 2 - 1) * (board_size * 2 - 1) -
                   board_size * (board_size - 1)),  // diameter^2 - corners
      neighbors_(get_neighbors(board_size)),
      ansi_color_output_(ansi_color_output),
      allow_swap_(allow_swap) {
  board_.resize(board_diameter_ * board_diameter_);
  for (int i = 0; i < board_.size(); i++) {
    Move m = ActionToMove(i);
    board_[i] = Cell((m.OnBoard() ? kPlayerNone : kPlayerInvalid), i,
                     m.Corner(board_size), m.Edge(board_size));
  }
}

Move HavannahState::ActionToMove(Action action_id) const {
  return Move(action_id % board_diameter_, action_id / board_diameter_,
              board_size_);
}

std::vector<Action> HavannahState::LegalActions() const {
  // Can move in any empty cell.
  std::vector<Action> moves;
  if (IsTerminal()) return {};
  moves.reserve(board_.size() - moves_made_);
  for (int cell = 0; cell < board_.size(); ++cell) {
    if (board_[cell].player == kPlayerNone) {
      moves.push_back(cell);
    }
  }
  if (AllowSwap()) {  // The second move is allowed to replace the first one.
    moves.push_back(last_move_.xy);
    absl::c_sort(moves);
  }
  return moves;
}

std::string HavannahState::ActionToString(Player player,
                                          Action action_id) const {
  return ActionToMove(action_id).ToString();
}

bool HavannahState::AllowSwap() const {
  return allow_swap_ && moves_made_ == 1 && current_player_ == kPlayer2;
}

std::string HavannahState::ToString() const {
  // Generates something like:
  //        a b c d e
  //     1 @ O . @ O f
  //    2 . O O @ O @ g
  //   3 . @ @ . . @ O h
  //  4 . @ @ . . . O O i
  // 5 @ . . O . @ @ O .
  //  6 @ O . O O @ @[O]
  //   7 . @ O . O O O
  //    8 @ O @ O O O
  //     9 @ O @ @ @

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
  out << std::string(board_size_ + 3, ' ');
  for (int x = 0; x < board_size_; x++) {
    out << ' ' << coord << static_cast<char>('a' + x);
  }
  out << '\n';

  for (int y = 0; y < board_diameter_; y++) {
    out << std::string(abs(board_size_ - 1 - y) + 1 + ((y + 1) < 10), ' ');
    out << coord << (y + 1);  // Leading y coord.

    bool found_last = false;
    int start_x = (y < board_size_ ? 0 : y - board_size_ + 1);
    int end_x = (y < board_size_ ? board_size_ + y : board_diameter_);
    for (int x = start_x; x < end_x; x++) {
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
    if (y < board_size_ - 1) {  // Trailing x coord.
      out << ' ' << coord << static_cast<char>('a' + board_size_ + y);
    }
    out << '\n';
  }
  out << reset;
  return out.str();
}

std::vector<double> HavannahState::Returns() const {
  if (outcome_ == kPlayer1) return {1, -1};
  if (outcome_ == kPlayer2) return {-1, 1};
  if (outcome_ == kPlayerDraw) return {0, 0};
  return {0, 0};  // Unfinished
}

std::string HavannahState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string HavannahState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int PlayerRelative(HavannahPlayer state, Player current) {
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

void HavannahState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, static_cast<int>(board_.size())},
                     true);
  for (int i = 0; i < board_.size(); ++i) {
    if (board_[i].player < kCellStates) {
      view[{PlayerRelative(board_[i].player, player), i}] = 1.0;
    }
  }
}

void HavannahState::DoApplyAction(Action action) {
  SPIEL_CHECK_EQ(outcome_, kPlayerNone);

  Move move = ActionToMove(action);
  SPIEL_CHECK_TRUE(move.OnBoard());

  if (last_move_ == move) {
    SPIEL_CHECK_TRUE(AllowSwap());
  } else {
    SPIEL_CHECK_EQ(board_[move.xy].player, kPlayerNone);
    moves_made_++;
    last_move_ = move;
  }
  board_[move.xy].player = current_player_;

  bool alreadyjoined = false;  // Useful for finding rings.
  bool skip = false;
  for (const Move& m : neighbors_[move.xy]) {
    if (skip) {
      skip = false;
    } else if (m.OnBoard()) {
      if (current_player_ == board_[m.xy].player) {
        alreadyjoined |= JoinGroups(move.xy, m.xy);

        // Skip the next one. If it is the same group, it is already connected
        // and forms a sharp corner, which we can ignore.
        skip = true;
      }
    }
  }

  const Cell& group = board_[FindGroupLeader(move.xy)];
  if (group.NumEdges() >= 3 || group.NumCorners() >= 2 ||
      (alreadyjoined && CheckRingDFS(move, 0, 3))) {
    outcome_ = current_player_;
  } else if (moves_made_ == valid_cells_) {
    outcome_ = kPlayerDraw;
  }

  current_player_ = (current_player_ == kPlayer1 ? kPlayer2 : kPlayer1);
}

int HavannahState::FindGroupLeader(int cell) {
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

bool HavannahState::JoinGroups(int cell_a, int cell_b) {
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
  board_[leader_a].corner |= board_[leader_b].corner;
  board_[leader_a].edge |= board_[leader_b].edge;

  return false;
}

bool HavannahState::CheckRingDFS(const Move& move, int left, int right) {
  if (!move.OnBoard()) return false;

  Cell& c = board_[move.xy];
  if (current_player_ != c.player) return false;
  if (c.mark) return true;  // Found a ring!

  c.mark = true;
  bool success = false;
  for (int i = left; !success && i <= right; i++) {
    int dir = (i + 6) % 6;  // Normalize.
    success = CheckRingDFS(neighbors_[move.xy][dir], dir - 1, dir + 1);
  }
  c.mark = false;
  return success;
}

std::unique_ptr<State> HavannahState::Clone() const {
  return std::unique_ptr<State>(new HavannahState(*this));
}

HavannahGame::HavannahGame(const GameParameters& params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")),
      allow_swap_(ParameterValue<bool>("swap")) {}

}  // namespace havannah
}  // namespace open_spiel
