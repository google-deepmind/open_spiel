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

#ifndef OPEN_SPIEL_GAMES_BLOTTO_H_
#define OPEN_SPIEL_GAMES_BLOTTO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/normal_form_game.h"

// An implementation of the Blotto: https://en.wikipedia.org/wiki/Blotto_game
// This version supports n >= 2 players. Each player distributes M coins on N
// fields. Each field is won by at most one player: the one with the most
// coins on the specific field; if there is a draw, the field is considered
// drawn (not won by any player), and hence ignored in the scoring. The winner
// is the player with the most won fields: all player have won the same number
// of fields, they each receive 0. Otherwise, the winners share 1 / (number of
// winners) and losers share -1 / (number of losers), reducing to {-1,0,1} in
// the 2-player case.
//
// Parameters:
//   "coins"      int    number of coins each player starts with (default: 10)
//   "fields"     int    number of fields (default: 3)
//   "players"    int    number of players (default: 2)

namespace open_spiel {
namespace blotto {

using ActionMap = std::unordered_map<Action, std::vector<int>>;

class BlottoState : public NFGState {
 public:
  BlottoState(std::shared_ptr<const Game> game, int coins, int fields,
              const ActionMap* action_map,
              const std::vector<Action>* legal_actions_);

  std::vector<Action> LegalActions(Player player) const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;

 protected:
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  int coins_;
  int fields_;
  std::vector<Action> joint_action_;  // The action taken by all the players.
  const ActionMap* action_map_;
  const std::vector<Action>* legal_actions_;
  std::vector<double> returns_;
};

class BlottoGame : public NormalFormGame {
 public:
  explicit BlottoGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BlottoState(shared_from_this(), coins_,
                                                  fields_, action_map_.get(),
                                                  legal_actions_.get()));
  }

  int NumPlayers() const override { return players_; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return +1; }

 private:
  void CreateActionMapRec(int* count, int coins_left,
                          const std::vector<int>& action);

  int num_distinct_actions_;
  int coins_;
  int fields_;
  int players_;
  std::unique_ptr<ActionMap> action_map_;
  std::unique_ptr<std::vector<Action>> legal_actions_;
};

}  // namespace blotto
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BLOTTO_H_
