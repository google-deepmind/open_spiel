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

#ifndef OPEN_SPIEL_GAMES_COORDINATED_MP_H_
#define OPEN_SPIEL_GAMES_COORDINATED_MP_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple game of Coordinate Matching Pennies, a modification of original MP
// that has multiple Nash equilibria lying on a line parametrized with one
// variable for the second player. He must coordinate his actions in the two
// infosets that he has, in such a way that p+q=1 for NE, where p and q are
// probabilities of playing Heads in top and bottom infosets respectively.
//
// For more information on this game (e.g. equilibrium sets, etc.) see
// todo: arxiv link
//

namespace open_spiel {
namespace coordinated_mp {

enum ActionType { kNoAction = -1, kHeads = 0, kTails = 1 };
enum InfosetPosition { kNoInfoset = -1, kTop = 0, kBottom = 1 };

class PenniesObserver;

class PenniesState : public State {
 public:
  explicit PenniesState(std::shared_ptr<const Game> game);
  PenniesState(const PenniesState&) = default;

  Player CurrentPlayer() const override;

  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  friend class PenniesObserver;

  ActionType actionA_ = kNoAction;  // Action of the first player.
  ActionType actionB_ = kNoAction;  // Action of the second player.
  InfosetPosition infoset_ = kNoInfoset;  // The infoset position in the game.
};

class PenniesGame : public Game {
 public:
  explicit PenniesGame(const GameParameters& params);
  int NumDistinctActions() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 2; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -1; };
  double MaxUtility() const override { return 1; };
  double UtilitySum() const override { return 0; }
  int MaxGameLength() const override { return 2; }
  int MaxChanceNodesInHistory() const override { return 1; }

  // New Observation API
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  // Used to implement the old observation API.
  std::shared_ptr<PenniesObserver> default_observer_;
  std::shared_ptr<PenniesObserver> info_state_observer_;
};

}  // namespace coordinated_mp
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COORDINATED_MP_H_
