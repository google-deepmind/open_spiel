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
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

constexpr char kCustom3PlayerAcpcGamedef[] = R"""(
# (Empty lines and lines starting with an '#' are all ignored)

GAMEDEF
nolimit
numPlayers = 3
numRounds = 1
numSuits = 2
numRanks = 4
numHoleCards = 1

# Set per player, so 3 total
stack = 15 15 15
blind = 0 1 0

# Set per round
firstPlayer = 3
numBoardCards = 0

END GAMEDEF
)""";

ABSL_FLAG(std::string, acpc_gamedef, kCustom3PlayerAcpcGamedef,
          "ACPC gamedef.");
ABSL_FLAG(int, num_iters, 2000, "How many iters to run for.");
// Note: reporting exploitability too frequently can be expensive!
ABSL_FLAG(int, report_every, 500, "How often to report exploitability.");

// Example code for using MCCFR on a univeral_poker game loaded from an ACPC
// gamedef (via the wrapper function).
int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "Input ACPC gamedef (raw): " << absl::GetFlag(FLAGS_acpc_gamedef)
            << std::endl;

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::universal_poker::LoadUniversalPokerGameFromACPCGamedef(
          absl::GetFlag(FLAGS_acpc_gamedef));

  // Downcasting to UniversalPokerGame so we can call GetACPCGame(), which isn't
  // on the higher level open_spiel::Game.
  const open_spiel::universal_poker::UniversalPokerGame& game_down_cast =
      open_spiel::down_cast<
          const open_spiel::universal_poker::UniversalPokerGame&>(*game);
  std::cout << "Resulting ACPC gamedef used for universal_poker:\n"
            << game_down_cast.GetACPCGame()->ToString() << std::endl;

  open_spiel::algorithms::ExternalSamplingMCCFRSolver solver(*game);
  std::cerr << "Starting MCCFR on " << game->GetType().short_name << "..."
            << std::endl;

  for (int i = 0; i < absl::GetFlag(FLAGS_num_iters); ++i) {
    solver.RunIteration();
    if (i % absl::GetFlag(FLAGS_report_every) == 0 ||
        i == absl::GetFlag(FLAGS_num_iters) - 1) {
      double exploitability = open_spiel::algorithms::Exploitability(
          *game, *solver.AveragePolicy());
      std::cerr << "Iteration " << i << " exploitability=" << exploitability
                << std::endl;
    }
  }
}
