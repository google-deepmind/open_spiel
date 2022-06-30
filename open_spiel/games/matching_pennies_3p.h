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

#ifndef OPEN_SPIEL_GAMES_MATCHING_PENNIES_3P_H_
#define OPEN_SPIEL_GAMES_MATCHING_PENNIES_3P_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/normal_form_game.h"

// A three-player matching pennies, described in (J. S. Jordan. Three problems
// in learning mixed-strategy Nash equilibria. Games and Economic Behavior,
// 5:368–386, 1993. Also described in section 1.3 of these notes:
// http://web.stanford.edu/~rjohari/teaching/notes/336_lecture7_2007.pdf
//
// From the notes: "Each player has two actions, H or T. Player 1 wants to match
// the action of player 2; player 2 wants to match the action of player 3; and
// player 3 wants to match the opposite of the action of player 1. Each player
// receives a payoff of 1 if they match as desired, and −1 otherwise. It is
// straightforward to check that this game has a unique Nash equilibrium, where
// all players uniformly randomize. Jordan shows that this Nash equilibrium is
// locally unstable in a strong sense: for any epsilon > 0, and for almost all
// initial empirical distributions that are within (Euclidean) distance epsilon
// of the unique Nash equilibrium, discrete-time fictitious play does not
// converge to the NE; instead, it enters a limit cycle asymptotically as t ->
// infinity".

namespace open_spiel {
namespace matching_pennies_3p {

class MatchingPennies3pState : public NFGState {
 public:
  MatchingPennies3pState(std::shared_ptr<const Game> game);

  std::vector<Action> LegalActions(Player player) const override;
  std::string ActionToString(Player player, Action move_id) const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;

 protected:
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  bool terminal_;
  std::vector<double> returns_;
};

class MatchingPennies3pGame : public NormalFormGame {
 public:
  explicit MatchingPennies3pGame(const GameParameters& params);

  int NumDistinctActions() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new MatchingPennies3pState(shared_from_this()));
  }

  int NumPlayers() const override { return 3; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return +1; }
};

}  // namespace matching_pennies_3p
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MatchingPennies3p_H_
