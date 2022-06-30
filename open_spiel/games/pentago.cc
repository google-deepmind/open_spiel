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

#include "open_spiel/games/pentago.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace pentago {
namespace {

// Facts about the game.
const GameType kGameType{/*short_name=*/"pentago",
                         /*long_name=*/"Pentago",
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
                             {"ansi_color_output", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PentagoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

struct Move {
  int x, y, xy;  // xy = x + y * kBoardSize
  int r;         // rotation
  int dir;       // which direction to rotate
  int quadrant;  // which quadrant to rotate

  constexpr Move(int x_, int y_, int r_)
      : x(x_),
        y(y_),
        xy(x_ + y_ * kBoardSize),
        r(r_),
        dir(r_ & 1),
        quadrant(r_ >> 1) {}
  constexpr Move(Action a)
      : Move((a / kPossibleRotations) % kBoardSize,
             (a / (kPossibleRotations * kBoardSize)) % kBoardSize,
             a % kPossibleRotations) {}

  Action ToAction() const {
    return ((y * kBoardSize) + x) * kPossibleRotations + r;
  }

  std::string ToString() const {
    return absl::StrCat(std::string(1, static_cast<char>('a' + x)),
                        std::string(1, static_cast<char>('1' + y)),
                        std::string(1, static_cast<char>('s' + r)));
  }
};

// Order the bits such that quadrant rotations are easy.
constexpr int xy_to_bit[kBoardPositions] = {
    0,  1,  2,  15, 16, 9,  // Comment
    7,  8,  3,  14, 17, 10,  // to force
    6,  5,  4,  13, 12, 11,  // clang-format
    29, 30, 31, 22, 23, 24,  // to keep the
    28, 35, 32, 21, 26, 25,  // square spatial
    27, 34, 33, 20, 19, 18,  // alignment.
};

// The bit mask for reading from an xy location.
constexpr uint64_t xym(int xy) { return 1ull << xy_to_bit[xy]; }
constexpr uint64_t xym(int x, int y) { return xym(x + y * kBoardSize); }
constexpr uint64_t xy_bit_mask[kBoardPositions] = {
    xym(0, 0), xym(1, 0), xym(2, 0), xym(3, 0), xym(4, 0), xym(5, 0),
    xym(0, 1), xym(1, 1), xym(2, 1), xym(3, 1), xym(4, 1), xym(5, 1),
    xym(0, 2), xym(1, 2), xym(2, 2), xym(3, 2), xym(4, 2), xym(5, 2),
    xym(0, 3), xym(1, 3), xym(2, 3), xym(3, 3), xym(4, 3), xym(5, 3),
    xym(0, 4), xym(1, 4), xym(2, 4), xym(3, 4), xym(4, 4), xym(5, 4),
    xym(0, 5), xym(1, 5), xym(2, 5), xym(3, 5), xym(4, 5), xym(5, 5),
};

// Helpers for creating the win mask.
constexpr uint64_t pattern(int x, int y, int ox, int oy) {
  return (xym(x + ox * 0, y + oy * 0) |  // Comment
          xym(x + ox * 1, y + oy * 1) |  // to force
          xym(x + ox * 2, y + oy * 2) |  // clang-format
          xym(x + ox * 3, y + oy * 3) |  // to keep
          xym(x + ox * 4, y + oy * 4));  // aligntment.
}
constexpr uint64_t horizontal(int x, int y) { return pattern(x, y, 1, 0); }
constexpr uint64_t vertical(int x, int y) { return pattern(x, y, 0, 1); }
constexpr uint64_t tl_br(int x, int y) { return pattern(x, y, 1, 1); }
constexpr uint64_t bl_tr(int x, int y) { return pattern(x, y, 1, -1); }

// The mask of 5 bits for each of the win conditions.
constexpr uint64_t win_mask[kPossibleWinConditions] = {
    horizontal(0, 0), horizontal(1, 0),  // Row 0
    horizontal(0, 1), horizontal(1, 1),  // Row 1
    horizontal(0, 2), horizontal(1, 2),  // Row 2
    horizontal(0, 3), horizontal(1, 3),  // Row 3
    horizontal(0, 4), horizontal(1, 4),  // Row 4
    horizontal(0, 5), horizontal(1, 5),  // Row 5
    vertical(0, 0), vertical(0, 1),  // Column 0
    vertical(1, 0), vertical(1, 1),  // Column 1
    vertical(2, 0), vertical(2, 1),  // Column 2
    vertical(3, 0), vertical(3, 1),  // Column 3
    vertical(4, 0), vertical(4, 1),  // Column 4
    vertical(5, 0), vertical(5, 1),  // Column 5
    tl_br(0, 0), tl_br(1, 1),  // Center diagonals from top-left to bottom-right
    tl_br(0, 1), tl_br(1, 0),  // Offset diagonals
    bl_tr(0, 5), bl_tr(1, 4),  // Center diagonals from bottom-left to top-right
    bl_tr(0, 4), bl_tr(1, 5),  // Offset diagonals
};

// Rotate a quadrant clockwise or counter-clockwise.
// Pulls a 8-bit segment and rotates it by 2 bits.
uint64_t rotate_quadrant_cw(uint64_t b, int quadrant) {
  uint64_t m = 0xFFull << (quadrant * 9);
  return (b & ~m) | (((b & m) >> 6) & m) | (((b & m) << 2) & m);
}
uint64_t rotate_quadrant_ccw(uint64_t b, int quadrant) {
  uint64_t m = 0xFFull << (quadrant * 9);
  return (b & ~m) | (((b & m) >> 2) & m) | (((b & m) << 6) & m);
}

}  // namespace

PentagoState::PentagoState(std::shared_ptr<const Game> game,
                           bool ansi_color_output)
    : State(std::move(game)), ansi_color_output_(ansi_color_output) {
  board_[0] = 0;
  board_[1] = 0;
}

std::vector<Action> PentagoState::LegalActions() const {
  // Can move in any empty cell, and do all rotations.
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  moves.reserve((kBoardPositions - moves_made_) * kPossibleRotations);
  for (int y = 0; y < kBoardSize; y++) {
    for (int x = 0; x < kBoardSize; x++) {
      if (get(x, y) == kPlayerNone) {
        for (int r = 0; r < kPossibleRotations; r++) {
          moves.push_back(Move(x, y, r).ToAction());
        }
      }
    }
  }
  return moves;
}

std::string PentagoState::ActionToString(Player player,
                                         Action action_id) const {
  return Move(action_id).ToString();
}

std::string PentagoState::ToString() const {
  // Generates something like:
  //     > t     u <
  //     a b c d e f
  // v 1 . . O @ . O v
  // s 2 . . O . . @ v
  //   3 . @ @ . @ O
  //   4 . @ @ . O .
  // z 5 @ . O @ O . w
  // ^ 6 @ O @ O O O ^
  //     > y     x <

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

  // Enable the arrows if/when open_spiel allows unicode in strings.
  // constexpr char const* arrows[] = {"↙", "↗", "↖", "↘", "↗", "↙", "↘", "↖"};
  constexpr char const* arrows[] = {"v", ">", "<", "v", "^", "<", ">", "^"};
  constexpr char const* left[] = {arrows[0], "s", " ", " ", "z", arrows[7]};
  constexpr char const* right[] = {arrows[3], "v", " ", " ", "w", arrows[4]};

  std::ostringstream out;
  out << coord;
  out << "    " << arrows[1] << " t     u " << arrows[2] << "\n";
  out << "    a b c d e f\n";
  for (int y = 0; y < kBoardSize; y++) {
    out << left[y] << " " << (y + 1) << " ";
    for (int x = 0; x < kBoardSize; x++) {
      Player p = get(x, y);
      if (p == kPlayerNone) out << empty;
      if (p == kPlayer1) out << white;
      if (p == kPlayer2) out << black;
      out << " ";
    }
    out << coord << right[y] << "\n";
  }
  out << "    " << arrows[6] << " y     x " << arrows[5] << reset << "\n";
  return out.str();
}

PentagoPlayer PentagoState::get(int i) const {
  return (board_[0] & xy_bit_mask[i]
              ? kPlayer1
              : board_[1] & xy_bit_mask[i] ? kPlayer2 : kPlayerNone);
}

std::vector<double> PentagoState::Returns() const {
  if (outcome_ == kPlayer1) return {1, -1};
  if (outcome_ == kPlayer2) return {-1, 1};
  if (outcome_ == kPlayerDraw) return {0, 0};
  return {0, 0};  // Unfinished
}

std::string PentagoState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string PentagoState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int PlayerRelative(PentagoPlayer state, Player current) {
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

void PentagoState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, kBoardPositions}, true);
  for (int i = 0; i < kBoardPositions; i++) {
    view[{PlayerRelative(get(i), player), i}] = 1.0;
  }
}

void PentagoState::DoApplyAction(Action action) {
  SPIEL_CHECK_EQ(outcome_, kPlayerNone);

  Move move(action);
  SPIEL_CHECK_EQ(get(move.xy), kPlayerNone);

  // Apply the move.
  board_[current_player_] |= xy_bit_mask[move.xy];
  if (move.dir == 0) {
    board_[0] = rotate_quadrant_ccw(board_[0], move.quadrant);
    board_[1] = rotate_quadrant_ccw(board_[1], move.quadrant);
  } else {
    board_[0] = rotate_quadrant_cw(board_[0], move.quadrant);
    board_[1] = rotate_quadrant_cw(board_[1], move.quadrant);
  }
  moves_made_++;

  // Check the win conditions.
  bool p1_won = false;
  bool p2_won = false;
  for (int i = 0; i < kPossibleWinConditions; i++) {
    uint64_t wm = win_mask[i];
    if ((board_[0] & wm) == wm) p1_won = true;
    if ((board_[1] & wm) == wm) p2_won = true;
  }

  // Note that you can rotate such that you cause your opponent to win.
  if (p1_won && p2_won) {
    outcome_ = kPlayerDraw;
  } else if (p1_won) {
    outcome_ = kPlayer1;
  } else if (p2_won) {
    outcome_ = kPlayer2;
  } else if (moves_made_ == kBoardPositions) {
    outcome_ = kPlayerDraw;
  }

  current_player_ = (current_player_ == kPlayer1 ? kPlayer2 : kPlayer1);
}

std::unique_ptr<State> PentagoState::Clone() const {
  return std::unique_ptr<State>(new PentagoState(*this));
}

PentagoGame::PentagoGame(const GameParameters& params)
    : Game(kGameType, params),
      ansi_color_output_(ParameterValue<bool>("ansi_color_output")) {}

}  // namespace pentago
}  // namespace open_spiel
