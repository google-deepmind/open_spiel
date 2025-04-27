// Copyright 2025 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_HIVE_H_
#define OPEN_SPIEL_GAMES_HIVE_H_

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/hive/hive_board.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// from https://en.wikipedia.org/wiki/Hive_(game):
//
//
//   "The object of Hive is to capture the opponent's queen bee by allowing it
//   to become completely surrounded by other pieces (belonging to either
//   player), while avoiding the capture of one's own queen. Hive differs from
//   other tile-based games in that the tiles, once placed, can then be moved to
//   other positions according to various rules, much like chess pieces."
//
//
// The intent of this Hive implementation is to provide a representation of the
// board state and action space that is compatible for use in the Alpha Zero
// algorithm (or similar).
//
// This becomes particularly tricky as one of the most notable design choices
// in Hive is that it is played on an unbounded surface, with no concept of a
// grid shape or size outside of the total number of tiles present. With the
// tiles being hexagonal in shape, a classic 2D grid representation used in
// 2D convolution complicates things even further.
//
// This implementation aims to minimize the effects of such problems by
// providing bounded grid sizes to reduce computational complexity (most games
// stay within ~6 units of the initial tile in practice). More information can
// be found under the HiveBoard class.
//
// Another important feature is the support of the Universal Hive Protocol (UHP)
// (https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol)
//
// While not a fully compliant UHP-engine implementation (mainly missing the
// required I/O and commands), the board game and state can be serialized to and
// de-serialized from a valid UHP gamestring. This allows the ever-growing
// archive of Hive replay data to be used for model training.
//
//
// Parameters:
//   "board_size"        int     radius of the underlying hexagonal board
//                               (default = 8)
//   "uses_mosquito"     bool    Whether to use the Mosquito expansion tile.
//                               (default = true)
//   "uses_ladybug"      bool    Whether to use the Ladybug expansion tile.
//                               (default = true)
//   "uses_pillbug"      bool    Whether to use the Pillbug expansion tile.
//                               (default = true)
//   "ansi_color_output" bool    Whether to color the output for a terminal.
//                               (default = false)

namespace open_spiel {
namespace hive {

// There are 28 unique tiles and 7 directions a tile can be placed beside (the 6
// hexagonal edges and "above"). So the total action space is 28 * 28 * 7 = 5488
inline constexpr int kNumDistinctActions = 5488 + 1;  // +1 for pass
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumBaseBugTypes = 5;
inline constexpr int kMaxGameLength = 500;
inline constexpr const char* kUHPNotStarted = "NotStarted";
inline constexpr const char* kUHPInProgress = "InProgress";
inline constexpr const char* kUHPWhiteWins = "WhiteWins";
inline constexpr const char* kUHPBlackWins = "BlackWins";
inline constexpr const char* kUHPDraw = "Draw";

// State of an in-play game.
class HiveState : public State {
 public:
  explicit HiveState(std::shared_ptr<const Game> game,
                     int board_size = kDefaultBoardRadius,
                     ExpansionInfo expansions = {},
                     int num_bug_types = kNumBaseBugTypes,
                     bool ansi_color_output = false);

  HiveState(const HiveState&) = default;
  HiveState& operator=(const HiveState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }

  // pretty prints the board state when using ansi_color_output_, and
  // prints the UHP string representation otherwise
  std::string ToString() const override;

  std::string ActionToString(Player player, Action action_id) const override;
  Action StringToAction(Player player,
                        const std::string& move_str) const override;

  bool IsTerminal() const override {
    return WinConditionMet(kPlayerWhite) || WinConditionMet(kPlayerBlack) ||
           MoveNumber() >= game_->MaxGameLength() || force_terminal_;
  }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  // A 3d-tensor where each binary 2d-plane represents the features below.
  // The # of feature planes varies based on the # of expansion tiles in use.
  // Example feature plane indices with all expansion tiles:
  //
  // (0-7):  current player's bugs in play for each of the 8 bug types
  // (8-15): opposing player's bugs in play for each of the 8 bug types
  // (16):   current player's "pinned" bugs
  // (17):   opposing player's "pinned" bugs
  // (18):   current player's valid placement positions
  // (19):   opposing player's valid placement positions
  // (20):   current player's "covered" bugs
  // (21):   opposing player's "covered" bugs
  // (22):   all 0's if it's White's turn, and all 1's if it's Black's turn
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

  // serializes state into a valid Universal Hive Protocol (UHP) game string.
  // UHP provides an interface between other Hive-playing software. Inspired
  // by the Universal Chess Interface protocol used for Chess software.
  // https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol
  // e.g. GameTypeString;GameStateString;TurnString;MoveString1;...;MoveStringN
  std::string Serialize() const override;

  // see above
  std::string UHPGameString() const;
  std::string UHPProgressString() const;
  std::string UHPTurnString() const;
  std::string UHPMovesString() const;

  HiveBoard& Board() { return board_; }
  const HiveBoard& Board() const { return board_; }

  Move ActionToMove(Action action) const;
  Action MoveToAction(Move move) const;
  Action PassAction() const { return NumDistinctActions() - 1; }

  inline bool WinConditionMet(Player player) const {
    return Board().IsQueenSurrounded(OtherColour(PlayerToColour(player)));
  }

 protected:
  void DoApplyAction(Action action) override;

 private:
  // allows any combination of expansion pieces to be used for the observation
  size_t BugTypeToTensorIndex(BugType type) const;

  // an axial coordinate at position (q, r) is stored at index [r][q] after
  // translating the axial coordinate by the length of the radius
  inline std::array<int, 2> AxialToTensorIndex(HivePosition pos) const {
    return {pos.R() + Board().Radius(), pos.Q() + Board().Radius()};
  }

  Player current_player_ = kPlayerWhite;
  HiveBoard board_;
  ExpansionInfo expansions_;
  int num_bug_types_;
  bool ansi_color_output_;
  bool force_terminal_;
};

class HiveGame : public Game {
 public:
  explicit HiveGame(const GameParameters& params);

  std::array<int, 3> ActionsShape() const { return {7, 28, 28}; }
  int NumDistinctActions() const override { return kNumDistinctActions; }
  inline std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<HiveState>(shared_from_this(), board_radius_,
                                       expansions_, num_bug_types_,
                                       ansi_color_output_);
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 0; }

  std::vector<int> ObservationTensorShape() const override {
    return {num_bug_types_ * kNumPlayers  // 2 * the # of bug types in play
                + 2                       // articulation point planes
                + 2                       // placeability planes
                + 2                       // covered planes
                + 1,                      // player turn plane
            2 * board_radius_ + 1,  // dimensions of a sq board from hex board
            2 * board_radius_ + 1};
  }

  int MaxGameLength() const override { return kMaxGameLength; }
  std::unique_ptr<State> DeserializeState(
      const std::string& uhp_string) const override;

  ExpansionInfo GetExpansionInfo() const { return expansions_; }

 private:
  int board_radius_;
  int num_bug_types_;
  bool ansi_color_output_;
  ExpansionInfo expansions_;
};

// helper to construct a game and state from a properly formed UHP string
std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
DeserializeUHPGameAndState(const std::string& uhp_string);

}  // namespace hive
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HIVE_H_
