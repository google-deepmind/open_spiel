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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_LASER_TAG_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_LASER_TAG_H_

#include <array>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// This is the laser tag game. See
// "A unified Game-Theoretic Approach to Multiagent Reinforcement Learning", Lanctot et al. 2017.
// https://arxiv.org/pdf/1711.00832.pdf
//
// Parameters:
//       "horizon"    int     max number of moves before draw  (default = 1000)

namespace open_spiel {
namespace laser_tag {

class LaserTagGame;

class LaserTagState : public SimMoveState {
 public:
  explicit LaserTagState(const LaserTagGame& parent_game);
  LaserTagState(const LaserTagState&) = default;

  std::string ActionToString(int player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(int player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return ToString();
  }
  void InformationStateAsNormalizedVector(int player,
                                          std::vector<double>* values) const;
  int CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const;

  void Reset(int horizon);
  std::vector<Action> LegalActions(int player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& moves) override;

 private:
  void SetField(int r, int c, char v);
  char field(int r, int c) const;
  void ResolveMove(int player, int move);
  bool InBounds(int r, int c) const;
  int observation_plane(int r, int c) const;

  const LaserTagGame& parent_game_;

  // Fields set to bad values. Use Game::NewInitialState().
  int winner_ = -1;
  int cur_player_ = -1;  // Could be chance's turn.
  int total_moves_ = -1;
  int horizon_ = -1;
  std::array<int, 2> player_row_ = {{-1, -1}};  // Players' rows.
  std::array<int, 2> player_col_ = {{-1, -1}};  // Players' cols.
  std::array<int, 2> player_facing_ = {{1, 1}};  // Player facing direction.
  int ball_row_ = -1;
  int ball_col_ = -1;
  std::array<int, 2> moves_ = {{-1, -1}};  // Moves taken.
  std::vector<char> field_;
};

class LaserTagGame : public SimMoveGame {
 public:
  explicit LaserTagGame(const GameParameters& params);
  int NumDistinctActions() const override { return 5; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 4; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  double UtilitySum() const override { return 0; }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new LaserTagGame(*this));
  }
  std::vector<int> InformationStateNormalizedVectorShape() const override;
  int MaxGameLength() const override { return horizon_; }

 private:
  int horizon_;
};

}  // namespace laser_tag
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_LASER_TAG_H_
