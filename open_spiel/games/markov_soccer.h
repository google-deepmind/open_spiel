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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_MARKOV_SOCCER_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_MARKOV_SOCCER_H_

#include <array>
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

namespace open_spiel {
namespace markov_soccer {

class MarkovSoccerGame;

class MarkovSoccerState : public SimMoveState {
 public:
  explicit MarkovSoccerState(const MarkovSoccerGame& parent_game);
  MarkovSoccerState(const MarkovSoccerState&) = default;

  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return ToString();
  }
  void InformationStateAsNormalizedVector(Player player,
                                          std::vector<double>* values) const;
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const;

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

  const MarkovSoccerGame& parent_game_;

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
  int NumDistinctActions() const override { return 5; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 4; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  double UtilitySum() const override { return 0; }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new MarkovSoccerGame(*this));
  }
  std::vector<int> InformationStateNormalizedVectorShape() const override;
  int MaxGameLength() const override { return horizon_; }

 private:
  int horizon_;
};

}  // namespace markov_soccer
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_MARKOV_SOCCER_H_
