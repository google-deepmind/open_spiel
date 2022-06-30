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

#ifndef OPEN_SPIEL_GAMES_MARKOV_SOCCER_H_
#define OPEN_SPIEL_GAMES_MARKOV_SOCCER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// This is the soccer game from the MinimaxQ paper. See
// "Markov Games as a Framework for Reinforcement Learning", Littman '94.
// http://www.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf
//
// Parameters:
//       "horizon"    int     max number of moves before draw  (default = 1000)
//       "grid"       string  String representation of grid.
//                            Empty spaces are '.', possible ball starting
//                            locations are 'O' and player A and B starting
//                            points are 'A' and 'B' respectively.

namespace open_spiel {
namespace markov_soccer {

inline constexpr char kDefaultGrid[] =
    ".....\n"
    "..OB.\n"
    ".AO..\n"
    ".....";

struct Grid {
  int num_rows;
  int num_cols;
  std::pair<int, int> a_start;
  std::pair<int, int> b_start;
  std::vector<std::pair<int, int>> ball_start_points;
};

// Number of chance outcomes reserved for "initiative" (learning which player's
// action gets resolved first).
inline constexpr int kNumInitiativeChanceOutcomes = 2;

// Reserved chance outcomes for initiative. The ones following these are to
// determine spawn point locations.
inline constexpr Action kChanceInit0Action = 0;
inline constexpr Action kChanceInit1Action = 1;
enum class ChanceOutcome { kChanceInit0, kChanceInit1 };

class MarkovSoccerGame;

class MarkovSoccerState : public SimMoveState {
 public:
  explicit MarkovSoccerState(std::shared_ptr<const Game> game,
                             const Grid& grid);
  MarkovSoccerState(const MarkovSoccerState&) = default;

  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return ToString();
  }
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;

  void Reset(int horizon);
  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& moves) override;

 private:
  void SetField(int r, int c, char v);
  char field(int r, int c) const;
  void ResolveMove(Player player, int move);
  bool InBounds(int r, int c) const;
  int observation_plane(int r, int c) const;

  const Grid& grid_;

  // Fields set to bad values. Use Game::NewInitialState().
  int winner_ = -1;
  Player cur_player_ = -1;  // Could be chance's turn.
  int total_moves_ = -1;
  int horizon_ = -1;
  std::array<int, 2> player_row_ = {{-1, -1}};  // Players' rows.
  std::array<int, 2> player_col_ = {{-1, -1}};  // Players' cols.
  int ball_row_ = -1;
  int ball_col_ = -1;
  std::array<int, 2> moves_ = {{-1, -1}};  // Moves taken.
  std::vector<char> field_;
};

class MarkovSoccerGame : public SimMoveGame {
 public:
  explicit MarkovSoccerGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  double UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return horizon_; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

 private:
  Grid grid_;
  int horizon_;
};

}  // namespace markov_soccer
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MARKOV_SOCCER_H_
