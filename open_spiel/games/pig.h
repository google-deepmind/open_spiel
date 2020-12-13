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

#ifndef OPEN_SPIEL_GAMES_PIG_H_
#define OPEN_SPIEL_GAMES_PIG_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple jeopardy dice game that includes chance nodes.
// See http://cs.gettysburg.edu/projects/pig/index.html for details.
// Also https://en.wikipedia.org/wiki/Pig_(dice_game)
//
// Parameters:
//     "diceoutcomes"  int    number of outcomes of the dice  (default = 6)
//     "horizon"       int    max number of moves before draw (default = 1000)
//     "players"       int    number of players               (default = 2)
//     "winscore"      int    number of points needed to win   (default = 100)

namespace open_spiel {
namespace pig {

class PigGame;

class PigState : public State {
 public:
  PigState(const PigState&) = default;
  PigState(std::shared_ptr<const Game> game, int dice_outcomes, int horizon,
           int win_score);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;

  int dice_outcomes() const { return dice_outcomes_; }
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  // Initialize to bad/invalid values. Use open_spiel::NewInitialState()
  int dice_outcomes_ = -1;  // Number of different dice outcomes (eg, 6).
  int horizon_ = -1;
  int nplayers_ = -1;
  int win_score_ = 0;

  int total_moves_ = -1;    // Total num moves taken during the game.
  Player cur_player_ = -1;  // Player to play.
  int turn_player_ = -1;    // Whose actual turn is it. At chance nodes, we need
                            // to remember whose is playing for next turn.
                            // (cur_player will be the chance player's id.)
  std::vector<int> scores_;  // Score for each player.
  int turn_total_ = -1;
};

class PigGame : public Game {
 public:
  explicit PigGame(const GameParameters& params);

  int NumDistinctActions() const override { return 6; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new PigState(shared_from_this(), dice_outcomes_, horizon_, win_score_));
  }
  int MaxChanceOutcomes() const override { return dice_outcomes_; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return horizon_; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return +1; }
  std::vector<int> ObservationTensorShape() const override;

 private:
  // Number of different dice outcomes, i.e. 6.
  int dice_outcomes_;

  // Maximum number of moves before draw.
  int horizon_;

  // Number of players in this game.
  int num_players_;

  // The amount needed to win.
  int win_score_;
};

}  // namespace pig
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PIG_H_
