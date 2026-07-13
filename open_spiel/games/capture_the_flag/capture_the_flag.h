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

#ifndef OPEN_SPIEL_GAMES_CAPTURE_THE_FLAG_H_
#define OPEN_SPIEL_GAMES_CAPTURE_THE_FLAG_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Simultaneous-move adversarial Capture-the-Flag (CTF) on a 2D grid.
//
// Two players occupy opposite ends of a symmetric grid. Each side owns a flag
// at its home base. A player scores a "capture" by stepping onto the
// opponent's flag (picking it up), then returning that flag to their own base
// while their own flag is still home. A flag carrier is vulnerable: if the
// carrier ends a step adjacent to the defender while inside the defender's
// home territory, the carrier is "tagged" and respawns at their own base,
// and the carried flag returns to its home base. Empty-handed players cannot
// tag each other (defense matters only against carriers).
//
// The default grid is symmetric and split into thirds. Player A's home
// territory is the left third, player B's is the right third, the middle
// column is neutral. Players spawn at their bases and the flags start at
// their owners' bases.
//
// Each step:
//   1. Both players submit one of {North, East, South, West, Stay}.
//   2. A chance node resolves move-order initiative (50/50) so the joint
//      action is unambiguous when the two players try to enter the same
//      cell.
//   3. The first-to-resolve player attempts to move (no-op if blocked by
//      bounds, obstacle, or the opponent's cell). Stepping onto the
//      opponent's flag at its home base picks it up. Stepping onto own
//      base while carrying the opponent's flag (and with own flag at home)
//      scores a capture: capture count increments, both flags reset, and
//      the carrier becomes empty-handed.
//   4. The second player resolves the same way against the updated state.
//   5. Tag resolution: for each player carrying the opponent's flag, if
//      the opponent is Manhattan-adjacent (distance == 1) and the carrier
//      is in the defender's home territory, the carrier is tagged. The
//      flag returns to its home base and the carrier respawns at their
//      own base.
//
// Termination:
//   * First player to reach `score_limit` captures wins.
//   * If `horizon >= 0` and the step count hits `horizon` without a winner,
//     the game ends in a draw.
//
// References:
//   * Leibo et al. Multi-agent Reinforcement Learning in Sequential Social
//     Dilemmas. https://arxiv.org/abs/1702.03037
//   * Agapiou et al. (DeepMind). Melting Pot 2.0.
//     https://arxiv.org/abs/2211.13746 (CTF appears as a substrate).
//
// Parameters:
//       "horizon"      int     Maximum number of steps before a draw.
//                              Use -1 to disable the horizon (game ends
//                              only when score_limit is reached).
//                              (default = 1000).
//       "zero_sum"     bool    If true, returns at termination are {+1, -1}
//                              for a winner and 0 each for a draw. If false,
//                              only the winner gets +1; the loser gets 0.
//                              (default = true).
//       "score_limit"  int     Number of captures needed to win.
//                              (default = 1).
//       "grid"         string  Newline-separated grid. Cells:
//                                '.' empty;
//                                '*' obstacle;
//                                'a' Player A's base (flag and spawn);
//                                'b' Player B's base (flag and spawn).
//                              Each grid must have exactly one 'a' and one
//                              'b' cell.

namespace open_spiel {
namespace capture_the_flag {

inline constexpr char kDefaultGrid[] =
    ".......\n"
    ".......\n"
    "a.....b\n"
    ".......\n"
    ".......";

struct Grid {
  int num_rows = 0;
  int num_cols = 0;
  std::vector<std::pair<int, int>> obstacles;
  std::pair<int, int> a_base = {-1, -1};
  std::pair<int, int> b_base = {-1, -1};
};

// Movement actions.
enum MovementType {
  kMoveNorth = 0,
  kMoveEast = 1,
  kMoveSouth = 2,
  kMoveWest = 3,
  kStay = 4
};

inline constexpr int kNumDistinctActions = 5;

// Two chance outcomes encode which player resolves first after each
// simultaneous action.
inline constexpr int kNumChanceOutcomes = 2;
inline constexpr Action kChanceInit0Action = 0;
inline constexpr Action kChanceInit1Action = 1;
enum class ChanceOutcome { kChanceInit0, kChanceInit1 };

// Five observation planes: player A, player B, A's flag, B's flag, obstacle.
inline constexpr int kCellStates = 5;

// Use this special player ID to indicate that no winner has been decided.
inline constexpr int kNoWinnerYetId = -1;


class CaptureTheFlagGame;

class CaptureTheFlagState : public SimMoveState {
 public:
  CaptureTheFlagState(std::shared_ptr<const Game> game, const Grid& grid,
                      int horizon, bool zero_sum, int score_limit);
  CaptureTheFlagState(const CaptureTheFlagState&) = default;

  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;
  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& moves) override;

 private:
  bool InBounds(int r, int c) const;
  bool IsObstacle(int r, int c) const;
  // Apply one player's move to the current state. Pickups and captures
  // happen here; tagging is deferred until both players have moved.
  void ResolveMove(Player player, int move);
  // After both players have resolved their moves for this step, check
  // whether a flag carrier should be tagged.
  void ResolveTags();
  void RespawnPlayer(Player player);
  void ResetFlagToHome(Player flag_owner);
  // True if (r, c) is in `player`'s home territory.
  bool InHomeTerritory(Player player, int r, int c) const;

  int ObservationPlaneForPlayer(int r, int c, Player player) const;
  int ObservationPlaneForFlag(int r, int c, Player flag_owner) const;

  const Grid& grid_;
  const int horizon_;
  const bool zero_sum_;
  const int score_limit_;

  // Set to invalid values; populated in the constructor.
  Player cur_player_ = kInvalidPlayer;
  int total_moves_ = 0;
  int winner_ = kNoWinnerYetId;  // -1 = no winner yet; 0 or 1 if decided.

  std::array<int, 2> player_row_ = {{-1, -1}};
  std::array<int, 2> player_col_ = {{-1, -1}};

  // Flag positions are tracked separately from carriers; a carrier's flag
  // coordinates equal the carrier's coordinates.
  std::array<int, 2> flag_row_ = {{-1, -1}};
  std::array<int, 2> flag_col_ = {{-1, -1}};
  // `holder_[k] = p` means player p is carrying flag k (k = 0 is A's flag,
  // k = 1 is B's flag). -1 means the flag is loose (at its home base).
  std::array<int, 2> flag_holder_ = {{-1, -1}};

  std::array<int, 2> score_ = {{0, 0}};
  std::array<int, 2> moves_ = {{-1, -1}};
  std::vector<double> rewards_ = {0.0, 0.0};
  std::vector<double> returns_ = {0.0, 0.0};
};

class CaptureTheFlagGame : public SimMoveGame {
 public:
  explicit CaptureTheFlagGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return kNumChanceOutcomes; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    // When `horizon_ < 0`, the game can run until score_limit is reached.
    // There is no finite hard cap in that case beyond the score-limit
    // dynamics, so we report a generous default consistent with similar
    // simultaneous-move games (e.g. laser_tag's horizon=-1 behaviour).
    return horizon_ >= 0 ? horizon_ : 1000;
  }
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

 private:
  Grid grid_;
  int horizon_;
  bool zero_sum_;
  int score_limit_;
};

}  // namespace capture_the_flag
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CAPTURE_THE_FLAG_H_
