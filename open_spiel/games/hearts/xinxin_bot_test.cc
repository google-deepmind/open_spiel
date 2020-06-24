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
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/games/hearts/hearts/Hearts.h"
#include "open_spiel/games/hearts/xinxin_bot.h"

namespace open_spiel {
namespace {

uint_fast32_t Seed() {
  return absl::ToUnixMicros(absl::Now());
}

void XinxinBot_BasicPlayGame() {
  int kNumPlayers = 4;
  int kNumGames = 5;
  std::mt19937 rng(Seed());
  auto game = open_spiel::LoadGame("hearts");
  std::vector<std::unique_ptr<open_spiel::Bot>> bots;

  int xinxin_index = 0;
  for (int i = 0; i < kNumPlayers; i++) {
    bots.push_back(open_spiel::hearts::MakeXinxinBot(game->GetParameters(),
                                                     game->NumPlayers()));
  }

  for (int i = 0; i < kNumGames; i++) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (open_spiel::Player p = 0; p < bots.size(); ++p) bots[p]->Restart();

    while (!state->IsTerminal()) {
      open_spiel::Player player = state->CurrentPlayer();
      open_spiel::Action action;
      if (state->IsChanceNode()) {
        open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
        action = open_spiel::SampleAction(outcomes, rng).first;
      } else {
        action = bots[player]->Step(*state);
      }

      for (open_spiel::Player p = 0; p < bots.size(); ++p) {
        if (p != player) {
          bots[p]->InformAction(*state, player, action);
        }
      }
      state->ApplyAction(action);
    }
  }
}

void XinxinBot_CardActionTransformationTest() {
  // exhaustively check if action mapping is a bijection
  for (Action action = 0; action < hearts::kNumCards; action++) {
    ::hearts::card card = hearts::GetXinxinAction(action);
    Action transformed = hearts::GetOpenSpielAction(card);
    SPIEL_CHECK_EQ(action, transformed);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::XinxinBot_CardActionTransformationTest();
  open_spiel::XinxinBot_BasicPlayGame();
}

