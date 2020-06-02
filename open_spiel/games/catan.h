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

#ifndef OPEN_SPIEL_GAMES_CATAN_H_
#define OPEN_SPIEL_GAMES_CATAN_H_

#include <string>
#include <vector>

#include "open_spiel/spiel.h"

#include "open_spiel/games/catan/PyCatanToC-/CatanGameEnv.h"
#include "open_spiel/games/catan/PyCatanToC-/CatanEncoder.h"

// CATAN, formerly The Settlers of Catan
// https://www.catan.com/
//
// Parameters:
//    "players"                     int      number of players in the game              (default=4)
//    "max_turns"                   int      maximal number of turns a game should last (default=72)
//    "random_start_player"         bool     if the start player is random              (default=false)
//    "example_board"               bool     if the example board is used               (default=true)
//    "example_starting_positions"  bool     if the start positions are fixed           (default=false)
//    "harbor_support"              bool     if harbors can be used                     (default=true)
//    "robber_support"              bool     if the robber can be used                  (default=true)
//    "development_card_support"    bool     if development cards can be bought         (default=true)

namespace open_spiel {
namespace catan {

// Game object.
class CatanGame : public Game {
  public:
    explicit CatanGame(const GameParameters& params);
    int NumDistinctActions() const override { return 368; } // hardcoded because CatanGame can not access catan_env_
    std::unique_ptr<State> NewInitialState() const override;
    int NumPlayers() const override { return this->num_players_; }
    double MinUtility() const override { return 0; } // with setup phase players can have a minimum of 2,
                                                     // with the example_starting_positions enabled players can have a minimum of 0
    double MaxUtility() const override { return 10; } // like with the min utility player can reach 8 or 10
    std::shared_ptr<const Game> Clone() const override;
    std::vector<int> ObservationTensorShape() const override {
      return {2576}; // hardcoded because CatanGame can not access catan_env_
    }
    int MaxGameLength() const override { return 100000; } // a game could take forever,
                                                          // with random action selection a game with 4 players takes on average about 1145 actions
                                                          // through the max_turns parameter it is also capped
    int num_players_;
  };

// State of an in-play game.
class CatanState : public State {
 public:
  explicit CatanState(std::shared_ptr<const Game> game, int players, int max_turns,
     bool random_start_player, bool example_board, bool example_starting_positions,
     bool harbor_support, bool robber_support, bool development_card_support);
  explicit CatanState(const CatanState&) = default;
  Player RandomPlayer() const;
  Player CurrentPlayer() const override {
    return this->IsTerminal() ? kTerminalPlayerId : this->current_player_;
  }
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  ActionsAndProbs ChanceOutcomes() const;
  std::string ToString() const override;
  bool IsTerminal() const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  CatanGameEnv catan_env_;
  CatanEncoder enc_;
  Player current_player_ = 0;
  Player previous_player_ = 0; // used to determine the next player or return to the previous in chance nodes
  std::vector<double> prev_state_score_; //saves the last score to calculate the next in the reward function
  int turns_ = 0; // counts the number of player turns, a turn ends if a player selects action 0
  int to_discard_counter_ = 0; // the amount of cards the current player has
                               // still to discard due to a seven
  bool buying_a_development_card = false; //for the chance node to distinguish between chance events
  int setup_phase_counter_ = 0; // for the setup phase to determine when the player have place their second settlement and road
  int max_turns = 72; // to cap the maximum number of turns
};

}  // namespace catan
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CATAN_H_
