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

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/higc/referee.h"

ABSL_FLAG(std::string, game, "kuhn_poker", "What game should be played.");
ABSL_FLAG(int, num_matches, 1, "Number of matches to play.");
ABSL_FLAG(std::vector<std::string>, executables, {},
          "Comma-separated list of paths to bot executable files.");
ABSL_FLAG(int, seed, 42, "Seed of the referee.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  open_spiel::higc::Referee ref(absl::GetFlag(FLAGS_game),
                                absl::GetFlag(FLAGS_executables),
                                absl::GetFlag(FLAGS_seed),
                                open_spiel::higc::TournamentSettings{
                                    .timeout_ready = 5000,
                                    .timeout_start = 200,
                                    .timeout_act = 5000,
                                    .timeout_ponder = 200,
                                    .timeout_match_over = 1000,
                                    .time_tournament_over = 60000,
                                    .max_invalid_behaviors = 3,
                                    .disqualification_rate = 0.1,
                                });
  ref.PlayTournament(absl::GetFlag(FLAGS_num_matches));
}
