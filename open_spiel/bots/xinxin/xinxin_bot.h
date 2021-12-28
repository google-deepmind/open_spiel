// Copyright 2021 DeepMind Technologies Limited
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


#ifndef OPEN_SPIEL_BOTS_XINXIN_XINXIN_BOT_H_
#define OPEN_SPIEL_BOTS_XINXIN_XINXIN_BOT_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/bots/xinxin/hearts/Hearts.h"
#include "open_spiel/bots/xinxin/hearts/iiMonteCarlo.h"
#include "open_spiel/games/hearts.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace hearts {

Action GetOpenSpielAction(::hearts::card card);
::hearts::card GetXinxinAction(Action action);

class XinxinBot : public Bot {
 public:
  explicit XinxinBot(int rules, int uct_num_runs, double uct_c_val,
                     int iimc_num_worlds, bool use_threads);

  Action Step(const State& state) override;
  void InformAction(const State& state, Player player_id,
                    Action action) override;
  void Restart() override;
  void RestartAt(const State& state) override;  // Currently just restarts.
  bool ProvidesForceAction() override { return true; }
  void ForceAction(const State& state, Action action) override;

  static int XinxinRules(GameParameters params);

 private:
  int uct_num_runs_;
  double uct_c_val_;
  int iimc_num_worlds_;
  bool use_threads_;
  std::unique_ptr<::hearts::SafeSimpleHeartsPlayer> CreatePlayer();

  int num_cards_dealt_;
  ::hearts::tPassDir pass_dir_;
  std::vector<std::vector<::hearts::card>> initial_deal_;

  // Keep a copy of the initial state around, to check that RestartAt only takes
  // place from the initial state.
  std::unique_ptr<State> initial_state_;

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

std::unique_ptr<Bot> MakeXinxinBot(GameParameters params, int uct_num_runs = 50,
                                   double uct_c_val = 0.4,
                                   int iimc_num_worlds = 20,
                                   bool use_threads = true);

}  // namespace hearts
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_XINXIN_XINXIN_BOT_H_
