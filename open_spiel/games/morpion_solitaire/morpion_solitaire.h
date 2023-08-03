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

#ifndef OPEN_SPIEL_GAMES_MORPION_SOLITAIRE_H_
#define OPEN_SPIEL_GAMES_MORPION_SOLITAIRE_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Morpion Solitaire (4D)
// https://en.wikipedia.org/wiki/Join_Five
// http://www.morpionsolitaire.com/
// Parameters: none

namespace open_spiel {
namespace morpion_solitaire {

// Constants.

// There are only 4 possible states for the max move limit (35),
// and 13x13 is the minimal square grid to fit all 4 solutions.
// http://www.morpionsolitaire.com/English/RecordsGrids4T4D.htm
inline constexpr int kNumRows = 13;
inline constexpr int kNumCols = 13;
inline constexpr int kNumPoints = kNumRows * kNumCols;

// Support Classes and Structs
// =============================================================
struct Point {
  int x{}, y{};
  Point() = default;
  Point(int a, int b) {
    this->x = a;
    this->y = b;
  }

  bool operator==(const Point& other_point) const {
    return (x == other_point.x) && (y == other_point.y);
  }

  bool operator<(const Point& other_point) const {
    if (x < other_point.x) {
      return true;
    } else if (x == other_point.x) {
      if (y < other_point.y) {
        return true;
      }
    }
    return false;
  }

  std::string ToString() const { return absl::StrCat("[", x, ",", y, "]"); }
};

class Line {
 public:
  Line(Point p1, Point p2);
  explicit Line(Action action);

  bool operator==(Line other_line);

  // Getters and setters
  std::vector<Point> GetEndpoints();
  std::array<int, 2> GetDirection();
  std::vector<Point> GetAllPoints();
  Action GetAction();
  bool CheckOverlap(Line l);
  std::string ToString() const;

 private:
  void Init(Point point1, Point point2);
  std::array<int, 2>
      direction_{};  // One of 4 line directions (0,0), (1,0), (1,1), (1,-1)
  Point endpoint1_;
  Point endpoint2_;
  std::vector<Point> line_points_;  // Collection of all 4 points on a line
};

// State of an in-play game.
class MorpionState : public State {
 public:
  // Constructors
  MorpionState(const MorpionState&) = default;
  explicit MorpionState(std::shared_ptr<const Game> game);

  MorpionState& operator=(const MorpionState&) = default;

  // Overridden Methods
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : kDefaultPlayerId;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  std::string ObservationString(Player player) const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  void UndoAction(Player player, Action move) override;

 protected:
  void getAllValidMoves() const;
  void DoApplyAction(Action move) override;

 private:
  std::array<int, kNumPoints> board_{};
  std::vector<Line> all_lines_;
  mutable std::vector<Line> current_valid_moves_;
  int num_moves_ = 0;
  double current_returns_{};
  std::vector<std::pair<Line, Point>>
      move_history_;  // Stores both Line and new Point created during move
  std::unordered_map<int, Line*> action_map_;  // Maps action encoding to Line
};

// Game object.
class MorpionGame : public Game {
 public:
  explicit MorpionGame(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new MorpionState(shared_from_this()));
  }

  // Number of distinct actions equals all possible lines drawn on the board.
  // Given 13x13 grid (see above), 4 points per line (4D),
  // For line directions [0, 1], [1, 0]: 10 possible lines (13 - 3) per row and
  // column. For line directions [1, -1], [1, 1]:
  //    - 10 lines (13 - 3) down center diagonal
  //    - 2 x (9 + 8 + .. 1) for other diagonals
  // In total (10 * 13 * 2) + 2 * (10 + (2 * (9 + 8 +... 1))) = 460
  int NumDistinctActions() const override { return 460; }

  // 4D fully solved by enumeration in 2008, with max 35 moves.
  // http://www.morpionsolitaire.com/English/Enumeration.htm
  // http://oeis.org/A204109
  int MaxGameLength() const override { return 35; }

  int NumPlayers() const override { return 1; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return MaxGameLength(); }
};

}  // namespace morpion_solitaire
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MORPION_SOLITAIRE_H_
