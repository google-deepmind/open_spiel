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

#ifndef OPEN_SPIEL_GAMES_LASER_TAG_H_
#define OPEN_SPIEL_GAMES_LASER_TAG_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// A fully observable version of the first-person gridworld laser tag game from
// [1,2] with identical mechanics. This implementation includes support for
// both fully observable (the default) and first-person partially observable
// modes. The current grid is "small2" from [2].
//
// When run in partially observable mode each agent has a local field-of-view,
// and sees (by default) 17 spaces in front, 10 to either side, and 2 spaces
// behind (per [2]). Each agent's observation is encoded as a 4x20x21 tensor,
// where each plane encodes the presence of player A, player B, empty, obstacle,
// respectively (same as fully observable tensor observations). The dimensions
// - front, side, and back - of the field of vision can be changed (see
// parameters below).
//
// [1] Leibo et al. Multi-agent Reinforcement Learning in Sequential Social
//     Dilemmas. https://arxiv.org/abs/1702.03037
// [2] Lanctot et al. A Unified Game-Theoretic Approach to Multiagent
//     Reinforcement Learning", https://arxiv.org/abs/1711.00832
//
// Parameters:
//       "horizon"    int     Number of steps per episode. If this is < 0, then
//                            the episode ends after the first tag.
//                            (default = 1000).
//       "zero_sum"   bool    If set, rewards are +1 for a tag and -1 for being
//                            tagged. Otherwise, there there is only positive
//                            reward of +1 per tag. (default = false).
//       "grid"       string  String representation of grid.
//                            Empty spaces are '.', obstacles are '*', spawn
//                            points are 'S' (there must be four of these).
//       "fully_obs"  bool    If set, the environment is full observable.
//                            Otherwise, the environment is partially
//                            observable (default = true)
//       "obs_front"  int     Number of squares each agent sees in front of
//                            themself (only used if fully_obs=false)
//                            (default=17)
//       "obs_back"   int     Number of squares each agent sees behind themself
//                            (only used if fully_obs=false) (default=2)
//       "obs_side"   int     Number of squares each agent sees to either side
//                            of themself (only used if fully_obs=false)
//                            (default=10)

namespace open_spiel {
namespace laser_tag {

inline constexpr char kDefaultGrid[] =
    "S.....S\n"
    ".......\n"
    "..*.*..\n"
    ".**.**.\n"
    "..*.*..\n"
    ".......\n"
    "S.....S";

struct Grid {
  int num_rows;
  int num_cols;
  std::vector<std::pair<int, int>> obstacles;
  std::vector<std::pair<int, int>> spawn_points;
};

// Number of chance outcomes reserved for "initiative" (learning which player's
// action gets resolved first).
inline constexpr int kNumInitiativeChanceOutcomes = 2;

// Reserved chance outcomes for initiative. The ones following these are to
// determine spawn point locations.
inline constexpr Action kChanceInit0Action = 0;
inline constexpr Action kChanceInit1Action = 1;
enum class ChanceOutcome { kChanceInit0, kChanceInit1 };

class LaserTagState : public SimMoveState {
 public:
  explicit LaserTagState(std::shared_ptr<const Game> game, const Grid& grid);
  LaserTagState(const LaserTagState&) = default;

  std::string ActionToString(int player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(int player) const override;
  void ObservationTensor(int player, absl::Span<float> values) const override;
  int CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;

  void Reset(int horizon, bool zero_sum);
  std::vector<Action> LegalActions(int player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& moves) override;

 private:
  void SetField(int r, int c, char v);
  char field(int r, int c) const;
  bool ResolveMove(int player, int move);  // Return true if there was a tag
  bool InBounds(int r, int c) const;
  int observation_plane(int r, int c) const;
  std::vector<int> map_observation_to_grid(int player, int r, int c) const;
  std::string PartialObservationString(int player) const;
  void FullObservationTensor(absl::Span<float> values) const;
  void PartialObservationTensor(int player, absl::Span<float> values) const;

  const Grid& grid_;

  // Fields set to bad values. Use Game::NewInitialState().
  int num_tags_ = 0;
  int cur_player_ = -1;  // Could be chance's turn.
  int total_moves_ = -1;
  int horizon_ = -1;
  bool zero_sum_rewards_ = false;
  bool fully_obs_ = false;
  int obs_front_ = -1;
  int obs_back_ = -1;
  int obs_side_ = -1;
  std::vector<int> needs_respawn_ = {0, 1};
  std::array<int, 2> player_row_ = {{-1, -1}};   // Players' rows.
  std::array<int, 2> player_col_ = {{-1, -1}};   // Players' cols.
  std::array<int, 2> player_facing_ = {{1, 1}};  // Player facing direction.
  std::vector<double> rewards_ = {0, 0};
  std::vector<double> returns_ = {0, 0};
  int ball_row_ = -1;
  int ball_col_ = -1;
  std::array<int, 2> moves_ = {{-1, -1}};  // Moves taken.
  std::vector<char> field_;
};

class LaserTagGame : public SimMoveGame {
 public:
  explicit LaserTagGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return horizon_; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

 private:
  Grid grid_;
  int horizon_;
  bool zero_sum_;
  bool fully_obs_;
  int obs_front_;
  int obs_back_;
  int obs_side_;
};

}  // namespace laser_tag
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LASER_TAG_H_
