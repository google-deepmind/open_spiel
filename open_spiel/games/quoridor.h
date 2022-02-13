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

#ifndef OPEN_SPIEL_GAMES_QUORIDOR_H_
#define OPEN_SPIEL_GAMES_QUORIDOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// https://en.wikipedia.org/wiki/Quoridor
//
// Parameters:
//   "board_size"        int     Size of the board (default = 9)
//   "wall_count"        int     How many walls per side (default = size^2/8)
//   "ansi_color_output" bool    Whether to color the output for a terminal.
//   "players"           int     Number of players (default = 2)

namespace open_spiel {
namespace quoridor {

inline constexpr int kDefaultNumPlayers = 2;
inline constexpr int kMinNumPlayers = 2;
inline constexpr int kMaxNumPlayers = 4;
inline constexpr int kDefaultBoardSize = 9;
inline constexpr int kMinBoardSize = 3;
inline constexpr int kMaxBoardSize = 25;
inline constexpr int kMaxGameLengthFactor = 4;

enum QuoridorPlayer : uint8_t {
  kPlayer1,
  kPlayer2,
  kPlayer3,
  kPlayer4,
  kPlayerWall,
  kPlayerNone,
  kPlayerDraw,
};

struct Offset {
  int x, y;

  Offset(int x_, int y_) : x(x_), y(y_) {}

  Offset operator+(const Offset& o) const { return Offset(x + o.x, y + o.y); }
  Offset operator-(const Offset& o) const { return Offset(x - o.x, y - o.y); }
  Offset operator*(const int i) const { return Offset(x * i, y * i); }
  Offset rotate_left() const { return Offset(-y, x); }
  Offset rotate_right() const { return Offset(y, -x); }
};

struct Move {
  int x, y;
  int xy;  // Precomputed x + y * size.
  int size;

  Move() : x(0), y(0), xy(-1), size(-1) {}
  Move(int x_, int y_, int size_)
      : x(x_), y(y_), xy(x_ + (y_ * size_)), size(size_) {}

  std::string ToString() const;

  bool IsValid() const { return x >= 0 && y >= 0 && x < size && y < size; }
  bool IsWall() const { return x & 1 || y & 1; }
  bool IsHorizontalWall() const { return y & 1; }
  bool IsVerticalWall() const { return x & 1; }

  bool operator==(const Move& b) const { return xy == b.xy; }
  bool operator!=(const Move& b) const { return xy != b.xy; }
  bool operator<(const Move& b) const { return xy < b.xy; }

  Move operator+(const Offset& o) const { return Move(x + o.x, y + o.y, size); }
  Move operator-(const Offset& o) const { return Move(x - o.x, y - o.y, size); }
};

// State of an in-play game.
class QuoridorState : public State {
 public:
  QuoridorState(std::shared_ptr<const Game> game, int board_size,
                int wall_count, bool ansi_color_output = false);

  QuoridorState(const QuoridorState&) = default;
  void InitializePlayer(QuoridorPlayer);

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : static_cast<int>(current_player_);
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return outcome_ != kPlayerNone; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  int NumCellStates() const { return num_players_ + 1; }

 protected:
  void DoApplyAction(Action action) override;

  // Turn an action id into a `Move`.
  Move ActionToMove(Action action_id) const;

  Move GetMove(int x, int y) const { return Move(x, y, board_diameter_); }
  bool IsWall(Move m) const {
    return m.IsValid() ? board_[m.xy] == kPlayerWall : true;
  }
  QuoridorPlayer GetPlayer(Move m) const {
    return m.IsValid() ? board_[m.xy] : kPlayerWall;
  }
  void SetPlayer(Move m, QuoridorPlayer p, Player old) {
    SPIEL_CHECK_TRUE(m.IsValid());
    SPIEL_CHECK_EQ(board_[m.xy], old);
    board_[m.xy] = p;
  }

 private:
  // SearchState contains details that are only used in the .cc file.
  // A different technique in the same area is called pimpl (pointer to
  // implementation).
  class SearchState;

  // Helpers for `LegaLActions`.
  void AddActions(Move cur, Offset offset, std::vector<Action>* moves) const;
  bool IsValidWall(Move m, SearchState*) const;
  bool SearchEndZone(QuoridorPlayer p, Move wall1, Move wall2,
                     SearchState*) const;
  void SearchShortestPath(QuoridorPlayer p, SearchState* search_state) const;

  std::vector<QuoridorPlayer> board_;
  std::vector<QuoridorPlayer> players_;
  std::vector<int> wall_count_;
  std::vector<int> end_zone_;
  std::vector<Move> player_loc_;
  QuoridorPlayer current_player_ = kPlayer1;
  int current_player_index_ = 0;
  QuoridorPlayer outcome_ = kPlayerNone;
  int moves_made_ = 0;
  const int board_size_;
  const int board_diameter_;
  const bool ansi_color_output_;
};

// Game object.
class QuoridorGame : public Game {
 public:
  explicit QuoridorGame(const GameParameters& params);

  int NumDistinctActions() const override { return Diameter() * Diameter(); }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new QuoridorState(
        shared_from_this(), board_size_, wall_count_, ansi_color_output_));
  }
  int NumPlayers() const override { return num_players_; }
  int NumCellStates() const { return num_players_ + 1; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {NumCellStates() + num_players_, Diameter(), Diameter()};
  }
  int MaxGameLength() const override {
    // There's no anti-repetition rule, so this could be infinite, but no sane
    // agent would take more moves than placing all the walls and visiting
    // all squares.
    return kMaxGameLengthFactor * board_size_ * board_size_;
  }

 private:
  int Diameter() const { return board_size_ * 2 - 1; }
  const int board_size_;
  const int wall_count_;
  const bool ansi_color_output_ = false;
  const int num_players_;
};

}  // namespace quoridor
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_QUORIDOR_H_
