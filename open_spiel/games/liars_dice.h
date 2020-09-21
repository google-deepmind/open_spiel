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

#ifndef OPEN_SPIEL_GAMES_LIARS_DICE_H_
#define OPEN_SPIEL_GAMES_LIARS_DICE_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple game that includes chance and imperfect information
// https://en.wikipedia.org/wiki/Liar%27s_dice
//
// Currently only supports a single round and two players.
//
// Parameters:
//   "players"     int    number of players                      (default = 2)
//   "numdice"     int    number of dice per player              (default = 1)
//   "numdiceX"    int    overridden number of dice for player X (default = 1)

namespace open_spiel {
namespace liars_dice {

class LiarsDiceGame;

class LiarsDiceState : public State {
 public:
  explicit LiarsDiceState(std::shared_ptr<const Game> game, int total_num_dice,
                          int max_dice_per_player,
                          const std::vector<int>& num_dice);
  LiarsDiceState(const LiarsDiceState&) = default;

  void Reset(const GameParameters& params);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  void ResolveWinner();

  // Initialized to invalid values. Use Game::NewInitialState().
  Player cur_player_;  // Player whose turn it is.
  int cur_roller_;     // Player currently rolling dice.
  int winner_;
  int loser_;
  int current_bid_;
  int total_num_dice_;
  int total_moves_;
  int calling_player_;  // Player who calls Liar.
  int bidding_player_;  // Player who cast the last bid.
  int max_dice_per_player_;

  // Dice outcomes: first indexed by player, then sorted by outcome.
  std::vector<std::vector<int>> dice_outcomes_;
  std::vector<int> num_dice_;         // How many dice each player has.
  std::vector<int> num_dice_rolled_;  // Number of dice currently rolled.

  // Used to encode the information state.
  std::vector<int> bidseq_;
  std::string bidseq_str_;
};

class LiarsDiceGame : public Game {
 public:
  explicit LiarsDiceGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  double UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override;

  // Returns the maximum among how many dice each player has. For example,
  // if player 1 has 3 dice and player 2 has 2 dice, returns 3.
  int max_dice_per_player() const { return max_dice_per_player_; }

  // Return the total number of dice in the game.
  int total_num_dice() const { return total_num_dice_; }

  // Get the quantity and face of the bid from an integer.
  // The bids starts at 1 and go to total_dice*6+1.
  static std::pair<int, int> GetQuantityFace(int bid, int total_dice);

 private:
  // Number of players.
  int num_players_;

  // Total dice in the game, determines the legal bids.
  int total_num_dice_;

  std::vector<int> num_dice_;  // How many dice each player has.
  int max_dice_per_player_;    // Maximum value in num_dice_ vector.
};

}  // namespace liars_dice
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LIARS_DICE_H_
