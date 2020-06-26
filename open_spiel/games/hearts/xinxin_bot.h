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


#ifndef OPEN_SPIEL_GAMES_HEARTS_XINXIN_BOT_H_
#define OPEN_SPIEL_GAMES_HEARTS_XINXIN_BOT_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/hearts.h"
#include "open_spiel/games/hearts/hearts/Hearts.h"
#include "open_spiel/games/hearts/hearts/iiMonteCarlo.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace hearts {

Action GetOpenSpielAction(::hearts::card card);
::hearts::card GetXinxinAction(Action action);

class XinxinBot : public Bot {
 public:
  explicit XinxinBot(int rules, int num_players = 4);

  Action Step(const State& state) override;
  void InformAction(const State& state, Player player_id,
                    Action action) override;
  void Restart() override;
  bool ProvidesForceAction() override { return true; }
  void ForceAction(const State& state, Action action) override;

  static int XinxinRules(GameParameters params);

 private:
  std::unique_ptr<::hearts::SafeSimpleHeartsPlayer> CreatePlayer();

  const int kNumPlayers;
  int num_cards_dealt_;
  ::hearts::tPassDir pass_dir_;
  std::vector<std::vector<::hearts::card>> initial_deal_;

  // A number of pointers to objects need to be created externally, and sent
  // into the xinxin. We use these containers to store them. The vectors are
  // indexed by player number.
  std::unique_ptr<::hearts::HeartsGameState> game_state_;
  std::vector<std::unique_ptr<::hearts::UCT>> xinxin_uct_;
  std::vector<std::unique_ptr<::hearts::iiMonteCarlo>> xinxin_mc_;
  std::vector<std::unique_ptr<::hearts::HeartsPlayout>> xinxin_playouts_;

  void NewDeal(std::vector<std::vector<::hearts::card>>* initial_cards,
               ::hearts::tPassDir pass_dir, int first_player);
  void LogStateMismatchError(const State& state, std::string msg);
};

std::unique_ptr<Bot> MakeXinxinBot(GameParameters params, int num_players);

}  // namespace hearts
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HEARTS_XINXIN_BOT_H_
