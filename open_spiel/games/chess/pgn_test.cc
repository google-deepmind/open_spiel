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

#include <open_spiel/abseil-cpp/absl/random/distributions.h>
#include <algorithms/evaluate_bots.h>
#include "open_spiel/spiel.h"
#include "open_spiel/games/chess/pgn.h"
#include "open_spiel/spiel_bots.h"

#include <iostream>

namespace open_spiel {
namespace pgn {
namespace {

void ChessTest() {
  std::mt19937 rng(123654789);
  auto game = open_spiel::LoadGame("chess");
  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  std::vector<Bot *> bot_ptrs;

  std::unique_ptr<Bot> random1 = MakeUniformRandomBot(0, 123654789);
  std::unique_ptr<Bot> random2 = MakeUniformRandomBot(1, 321456978);

  bot_ptrs.push_back(random1.get());
  bot_ptrs.push_back(random2.get());

  std::unique_ptr<State> state = game->NewInitialState();
  EvaluateBots(state.get(), bot_ptrs,
               absl::Uniform<int>(rng, 0, std::numeric_limits<int>::max()));

  auto chess_state = down_cast<chess::ChessState>(*state);
  auto pgn = ChessPGN(chess_state, "test", 1, "random1", "random2");

  std::cout << pgn;

}

void KriegspielTest() {

  GameParameters params;
  params["50_move_rule"] = GameParameter(true);
  params["threefold_repetition"] = GameParameter(true);

  std::mt19937 rng(123654789);
  auto game = open_spiel::LoadGame("kriegspiel", params);
  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  std::vector<Bot *> bot_ptrs;

  std::unique_ptr<Bot> random1 = MakeUniformRandomBot(0, 123654789);
  std::unique_ptr<Bot> random2 = MakeUniformRandomBot(1, 321456978);

  bot_ptrs.push_back(random1.get());
  bot_ptrs.push_back(random2.get());

  std::unique_ptr<State> state = game->NewInitialState();
  EvaluateBots(state.get(), bot_ptrs,
               absl::Uniform<int>(rng, 0, std::numeric_limits<int>::max()));

  auto kriegspiel_state = down_cast<kriegspiel::KriegspielState>(*state);
  auto pgn = KriegspielPGN(kriegspiel_state, "test", 1, "random1", "random2", chess::Color::kWhite);

  std::cout << pgn;

}

}  // namespace
}  // namespace pgn
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::pgn::ChessTest();
  open_spiel::pgn::KriegspielTest();
}
