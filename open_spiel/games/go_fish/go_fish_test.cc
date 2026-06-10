// Copyright 2026 DeepMind Technologies Limited
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

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/go_fish/go_fish.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace go_fish {
namespace {

namespace testing = open_spiel::testing;

void BasicGoFishTests() {
  testing::LoadGameTest("go_fish");
  testing::ChanceOutcomesTest(*LoadGame("go_fish"));
  testing::RandomSimTest(*LoadGame("go_fish"), 100);
  for (Player players = 2; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("go_fish", {{"players", GameParameter(players)}}), 100);
  }
}

void SerializationTests() {
  auto game = LoadGame("go_fish");

  // Default board position.
  std::unique_ptr<State> state = game->NewInitialState();
  std::shared_ptr<State> deserialized_state =
      game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());
}

template <typename T>
void PrintMatrix(const std::vector<std::vector<T>>& m) {
  for (const auto& row : m) {
    for (const auto& x : row) std::cout << x << " ";
    std::cout << std::endl;
  }
}

void ObservationTensorTests() {
  std::shared_ptr<const Game> game = LoadGame("go_fish");
  std::string start =
    "Ask\n0\nc1d1f1g2h1i1:0\nb2d1g1l2m1:0\na4b2c3d2e4f3g1h3i3j4k4l2m3";
  GoFishState state(game, start);
  const GoFishGame* gfg = static_cast<const GoFishGame*>(game.get());
  state.ApplyAction(gfg->AskStringToAction("1,g"));
  state.ApplyAction(gfg->AskStringToAction("1,d"));
  state.ApplyAction(gfg->AskStringToAction("1,h"));
  state.ApplyAction(gfg->FishStringToAction("g"));
  state.ApplyAction(gfg->AskStringToAction("0,b"));
  // do some more actions so drawn_since isn;t always 0
  state.ApplyAction(gfg->FishStringToAction("a"));
  state.ApplyAction(gfg->AskStringToAction("1,i"));
  state.ApplyAction(gfg->FishStringToAction("m"));
  state.ApplyAction(gfg->AskStringToAction("0,a"));
  state.ApplyAction(gfg->FishStringToAction("a"));
  state.ApplyAction(gfg->AskStringToAction("0,m"));

/*
  std::cout << state.ToString() << std::endl;
  std::cout << "PlayerDidAsk" << std::endl;
  PrintMatrix(state.PlayerDidAsk());
  std::cout << "PlayerWasAsked" << std::endl;
  PrintMatrix(state.PlayerWasAsked());
  std::cout << "DrawnSinceWasAsked" << std::endl;
  PrintMatrix(state.DrawnSinceWasAsked());
  std::cout << "PlayerMin" << std::endl;
  PrintMatrix(state.PlayerMin());
  std::cout << v << std::endl;
*/

  auto shape = game->ObservationTensorShape();
  std::vector<float> v(game->ObservationTensorSize());
  state.ObservationTensor(state.CurrentPlayer(),
                                  absl::MakeSpan(v));
  int offset = 0;
  float eps = 1e-6;
  // cards for the player on move
  SPIEL_CHECK_EQ(v[offset++], 2.0/4);  // a
  SPIEL_CHECK_EQ(v[offset++], 2.0/4);  // b
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // d
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // f
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // h
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // i
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // j
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // k
  SPIEL_CHECK_EQ(v[offset++], 2.0/4);  // l
  SPIEL_CHECK_EQ(v[offset++], 2.0/4);  // m
  // phase
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // kDeal
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // kAsk
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // Fish
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // kTerminal
  // pool size
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 34.0/52, eps);  // pool size
  // booked is boolean
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // a
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // b
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // d
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // f
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // h
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // i
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // j
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // k
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // l
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // m
  // values for players ordered by pid

  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 on move
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0/13, eps);  // p0 books
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 6.0/52, eps);  // p0 card count

  // did ask is count of times player asked normalized to suits * ranks
  // was asked is 0 or 1
  // drawn since is normalized to cards * ranks
  // min is normalized to suits
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for a
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // p0 was asked for a
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for a
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min a
                                    //
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for b
  SPIEL_CHECK_EQ(v[offset++], 1.0);  // p0 was asked for b
  // drawn since
  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0/52, eps);
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min b


  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for c
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min c

  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0/52, eps);  // p0 did ask for d
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for d
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for d
  SPIEL_CHECK_EQ(v[offset++], 2.0/ 4);  // player_min d

  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for e
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min e

  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 did ask for f
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for f
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for f
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min f

  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0/52, eps);  // p0 did ask for g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for g
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // player_min g

  SPIEL_CHECK_FLOAT_NEAR(v[offset++], 1.0/52, eps);  // p0 did ask for h
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // p0 was asked for h
  SPIEL_CHECK_EQ(v[offset++], 0.0);  // drawn since was asked for h
  SPIEL_CHECK_EQ(v[offset++], 1.0/4);  // player_min h
}


}  // namespace
}  // namespace go_fish
}  // namespace open_spiel


int main(int argc, char** argv) {
  open_spiel::Init(argv[0], &argc, &argv, true);
  open_spiel::go_fish::BasicGoFishTests();
  open_spiel::go_fish::ObservationTensorTests();
  open_spiel::go_fish::SerializationTests();
}
