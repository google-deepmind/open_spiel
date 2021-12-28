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
      // This command has following syntax:
      // position fen <FEN> moves <MOVES>
      std::vector<std::string> tokens = absl::StrSplit(line, ' ');
      // Build up the <FEN> which can contain spaces.
      std::stringstream fen;
      int pos = 2;
      bool has_moves = false;
      while (pos < tokens.size()) {
        if (tokens[pos] == "moves") {
          has_moves = true;
          break;
        }
        if (pos > 2) fen << ' ';
        fen << tokens[pos];
        ++pos;
      }

      state = game->NewInitialState(fen.str());
      chess_state = down_cast<chess::ChessState*>(state.get());

      if (has_moves) {
        while (pos < tokens.size()) {
          Action action = chess_state->ParseMoveToAction(tokens[pos]);
          state->ApplyAction(action);
          ++pos;
        }
      }
    } else if (absl::StartsWith(line, "go movetime ")) {
      std::vector<Action> legal_actions = state->LegalActions();
      int index = absl::Uniform<int>(rng, 0, legal_actions.size());
      Action action = legal_actions[index];
      chess::Move move = ActionToMove(action, chess_state->Board());
      std::cout << "bestmove " << move.ToLAN() << std::endl;
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
  absl::ParseCommandLine(argc, argv);
  open_spiel::uci::RandomUciBot();
}
