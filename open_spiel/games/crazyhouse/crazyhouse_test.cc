// Copyright 2025 George Weinberg
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

#include <string>

#include "open_spiel/games/crazyhouse/crazyhouse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace crazyhouse {
namespace {

namespace testing = open_spiel::testing;


void BasicTests() {
  testing::LoadGameTest("crazyhouse");
  auto game = open_spiel::LoadGame("crazyhouse");
  auto state = game->NewInitialState();

  std::cout << state->ToString() << std::endl;
  // whoo hoo all pass! 
}

void FoolsMateTest() {
  auto game = open_spiel::LoadGame("crazyhouse");
  auto state = game->NewInitialState();
  CrazyhouseState* ch_state = dynamic_cast<CrazyhouseState*>(state.get());
  SPIEL_CHECK_TRUE(ch_state != nullptr);
  auto apply = [&](const std::string& lan) {
    const CrazyhouseBoard& board = ch_state->Board();

    absl::optional<Move> maybe_move = board.ParseLANMove(lan);
    SPIEL_CHECK_TRUE(maybe_move);

    Action action = MoveToAction(*maybe_move, board.BoardSize());
    state->ApplyAction(action);
	std::cout << board.ToFEN() << std::endl;
  };
  apply("f2f4");
  apply("e7e5");
  apply("g2g4");
  apply("d8h4");
  // Checkmate assertions
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // Black wins
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kWhite)], -1);
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kBlack)], 1);
}

  
  void CrazyhouseDropMateTest() {
  auto game = open_spiel::LoadGame("crazyhouse");
  auto state = game->NewInitialState();
  CrazyhouseState* ch_state;

  auto apply = [&](const std::string& lan) {
    ch_state = dynamic_cast<CrazyhouseState*>(state.get());
    SPIEL_CHECK_TRUE(ch_state != nullptr);

    const CrazyhouseBoard& board = ch_state->Board();

    absl::optional<Move> maybe_move = board.ParseLANMove(lan);
    SPIEL_CHECK_TRUE(maybe_move);

    Action action = MoveToAction(*maybe_move, board.BoardSize());
    state->ApplyAction(action);
	std::cout << board.ToFEN() << std::endl;
  };

  // 1. f2f4
  apply("f2f4");
  // ... Ng8f6
  apply("g8f6");

  // 2. g2g4
  apply("g2g4");
  // ... Nxg4
  apply("f6g4");  // see note below

  // 3. Nb1a3
  apply("b1a3");
  // ... P@f2#
  apply("P@f2");


  // Checkmate assertions
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // Black wins
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kWhite)], -1);
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kBlack)], 1);
}


}  // namespace
}  // namespace crazyhouse
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::crazyhouse::FoolsMateTest();
  open_spiel::crazyhouse::BasicTests();
  open_spiel::crazyhouse::CrazyhouseDropMateTest();
}
