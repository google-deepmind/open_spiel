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

#ifndef OPEN_SPIEL_GAMES_K_GMP_H_
#define OPEN_SPIEL_GAMES_K_GMP_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/simultaneous_move_game.h"

namespace open_spiel {
namespace k_gmp {

class KGMPGame;

class KGMPState : public SimMoveState {
 public:
  explicit KGMPState(std::shared_ptr<const Game> game);
    KGMPState(const KGMPState&) = default;

  Player CurrentPlayer() const override;

  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
//  void InformationStateTensor(Player player,
//                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
//  void UndoAction(Player player, Action move) override;
//  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions(Player player) const override;
//  std::vector<int> hand() const { return {card_dealt_[CurrentPlayer()]}; }
//  std::unique_ptr<State> ResampleFromInfostate(
//      int player_id, std::function<double()> rng) const override;

//  const std::vector<int>& CardDealt() const { return card_dealt_; }

 protected:
    void DoApplyActions(const std::vector<Action>& actions) override;
    void DoApplyAction(Action move) override;

 private:
//  friend class KuhnObserver;

//  // Whether the specified player made a bet
//  bool DidBet(Player player) const;
//
  const KGMPGame& parent_game_;

  int current_k_gmp_game_num_selected_;
  int k_;
  int n_actions_;

//  // The move history and number of players are sufficient information to
//  // specify the state of the game. We keep track of more information to make
//  // extracting legal actions and utilities easier.
//  // The cost of the additional book-keeping is more complex ApplyAction() and
//  // UndoAction() functions.
//  int first_bettor_;             // the player (if any) who was first to bet
//  std::vector<int> card_dealt_;  // the player (if any) who has each card
  int winner_;                   // winning player, or kInvalidPlayer if the
//                                 // game isn't over yet.
//  int pot_;                      // the size of the pot
//  // How much each player has contributed to the pot, indexed by pid.
//  std::vector<int> ante_;
};

class KGMPGame : public Game {
 public:
  explicit KGMPGame(const GameParameters& params);
  int NumDistinctActions() const override { return std::max(num_actions_, k_); }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 0; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override {return -1.0 * (num_actions_ - 1.0);}
  double MaxUtility() const override {return num_actions_ - 1.0;}
  double UtilitySum() const override { return 0; }
//  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override {return {k_};}
  int MaxGameLength() const override { return 2; }
  int MaxChanceNodesInHistory() const override { return 0; }
//  std::shared_ptr<Observer> MakeObserver(
//      absl::optional<IIGObservationType> iig_obs_type,
//      const GameParameters& params) const override;

  // Used to implement the old observation API.
//  std::shared_ptr<KuhnObserver> default_observer_;
//  std::shared_ptr<KuhnObserver> info_state_observer_;
//  std::shared_ptr<KuhnObserver> public_observer_;
//  std::shared_ptr<KuhnObserver> private_observer_;

   int GetK() const {return k_;}
   int GetNActions() const {return num_actions_;}

private:

            // kgmp
      int k_;
      int num_actions_;
};

// Returns policy that always passes.
//TabularPolicy GetAlwaysPassPolicy(const Game& game);
//
// Returns policy that always bets.
//TabularPolicy GetAlwaysBetPolicy(const Game& game);

}  // namespace k_gmp
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_K_GMP_H_
