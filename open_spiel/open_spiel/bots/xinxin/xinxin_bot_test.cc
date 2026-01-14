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

#include "open_spiel/bots/xinxin/xinxin_bot.h"

#include <iostream>
#include <limits>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/bots/xinxin/hearts/Hearts.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace {

uint_fast32_t Seed() { return absl::ToUnixMicros(absl::Now()); }

void XinxinBot_BasicPlayGame() {
  int num_games = 5;
  std::mt19937 rng(Seed());
  auto game = open_spiel::LoadGame("hearts");
  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  std::vector<Bot *> bot_ptrs;

  for (int i = 0; i < hearts::kNumPlayers; i++) {
    bots.push_back(open_spiel::hearts::MakeXinxinBot(game->GetParameters()));
    bot_ptrs.push_back(bots.back().get());
  }

  for (int i = 0; i < num_games; i++) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    EvaluateBots(state.get(), bot_ptrs,
                 absl::Uniform<int>(rng, 0, std::numeric_limits<int>::max()));
    // call xinxinbot with terminal state so that xinxin's internal state's
    // returns can be checked against the OpenSpiel returns
    for (auto bot : bot_ptrs)
      bot->InformAction(*state, kTerminalPlayerId, kInvalidAction);
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
