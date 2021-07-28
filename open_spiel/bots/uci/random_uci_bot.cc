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
#include <memory>
#include <random>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/games/chess.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/init.h"

ABSL_FLAG(int, seed, 0, "The seed to use.");

namespace open_spiel {
namespace uci {

void RandomUciBot() {
  int seed = absl::GetFlag(FLAGS_seed);
  std::mt19937 rng(seed);
  std::unique_ptr<State> state;
  chess::ChessState* chess_state = nullptr;
  std::shared_ptr<const Game> game = LoadGame("chess");

  for (std::string line; std::getline(std::cin, line);) {
    if (line == "uci") {
      std::cout << "uciok" << std::endl;
    } else if (line == "isready") {
      std::cout << "readyok" << std::endl;
    } else if (line == "ucinewgame") {
      state = game->NewInitialState();
      chess_state = down_cast<chess::ChessState*>(state.get());
    } else if (absl::StartsWith(line, "position fen ")) {
      std::vector<std::string> tokens = absl::StrSplit(line, ' ');
      state = game->NewInitialState(tokens[2]);
      chess_state = down_cast<chess::ChessState*>(state.get());
      if (tokens.size() > 3) {
        SPIEL_CHECK_GT(tokens.size(), 4);
        SPIEL_CHECK_EQ(tokens[3], "moves");
        for (int i = 4; i < tokens.size(); ++i) {
          Action action = chess_state->ParseMoveToAction(tokens[i]);
          state->ApplyAction(action);
        }
      }
    } else if (absl::StartsWith(line, "go movetime ")) {
      std::vector<Action> legal_actions = state->LegalActions();
      int index = absl::Uniform<int>(rng, 0, legal_actions.size());
      Action action = legal_actions[index];
      chess::Move move = ActionToMove(action, chess_state->Board());
      std::cout << move.ToLAN() << std::endl;
    } else if (line == "quit") {
      return;
    } else {
      std::cout << "Unrecognized command: " << line << std::endl;
    }
  }
}

}  // namespace uci
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::Init("", &argc, &argv, false);
  open_spiel::uci::RandomUciBot();
}
