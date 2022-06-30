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
// Piglet variant: Instead of increasing the running total by the roll results,
// it is always increased by a fixed step size of 1 upon rolling anything higher
// than a 1. [Note: Internally, this behaviour is modelled with only two chance
// outcomes, rolling a 1 or rolling anything higher than that.]
// Divide winscore by the average dice outcome != 1 (i.e. by diceoutcomes/2 + 1)
// when enabling Piglet to play a game that's roughly equivalent to the
// corresponding Pig game. The main advantage of this variant is thus a greatly
// reduced state space, making the game accessible to tabular methods.
// See also http://cs.gettysburg.edu/~tneller/papers/pig.zip. The original
// Piglet variant described there is played with a fair coin and a winscore
// of 10. This behaviour can be achieved by setting diceoutcomes = 2, winscore =
// 10, piglet = true.
//
// Parameters:
//     "diceoutcomes"  int    number of outcomes of the dice  (default = 6)
//     "horizon"       int    max number of moves before draw (default = 1000)
//     "players"       int    number of players               (default = 2)
//     "winscore"      int    number of points needed to win  (default = 100)
//     "piglet"        bool   is piglet variant enabled?      (default = false)

namespace open_spiel {
namespace pig {

class PigGame;

class PigState : public State {
 public:
  PigState(const PigState&) = default;
  PigState(std::shared_ptr<const Game> game, int dice_outcomes, int horizon,
           int win_score, bool piglet);

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

  int score(const int player_id) const { return scores_[player_id]; }
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
  bool piglet_ = false;

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

  int NumDistinctActions() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new PigState(
        shared_from_this(), dice_outcomes_, horizon_, win_score_, piglet_));
  }
  int MaxChanceOutcomes() const override { return dice_outcomes_; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return horizon_; }

  // Every chance node is preceded by a decision node (roll)
  // -> At most as many chance nodes as decision nodes.
  // -> Up to as many chance nodes as decision nodes, if
  //    every action is "roll" and player never 'falls'.
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

  // Whether Piglet variant is enabled (always move only 1 step forward)
  bool piglet_;
};

}  // namespace pig
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PIG_H_
