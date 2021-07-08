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

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/higc/referee.h"

ABSL_FLAG(std::string, bots_dir, "higc/bots",
          "Directory containing the competition bots.");

namespace open_spiel {
namespace higc {
namespace {

void PlaySingleMatchIIGS() {
  std::string bot_first_action = absl::StrCat(absl::GetFlag(FLAGS_bots_dir),
                                              "/test_bot_first_action.sh");
  open_spiel::higc::Referee ref("goofspiel(imp_info=True,points_order=descending)",
                                {bot_first_action, bot_first_action});
  std::unique_ptr<TournamentResults> results = ref.PlayTournament(1);
  SPIEL_CHECK_EQ(results->num_matches(), 1);
  SPIEL_CHECK_TRUE(results->matches[0].terminal->IsTerminal());
  SPIEL_CHECK_EQ(results->matches[0].terminal->HistoryString(),
                 "0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, "
                 "6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11");
}

void PlayWithFailingBots() {
  std::string bot_first_action = absl::StrCat(absl::GetFlag(FLAGS_bots_dir),
                                              "/test_bot_first_action.sh");
  std::vector<std::string> failing_cases = {
      "/test_bot_break_pipe.sh",
      "/test_bot_sleep.sh",
      "/test_bot_ready.sh",
      "/test_bot_start.sh",
  };

  for (const std::string& failing_case : failing_cases) {
    std::cout << "\n\nFailing bot: " << failing_case << std::endl;
    std::string failing_bot = absl::StrCat(absl::GetFlag(FLAGS_bots_dir),
                                           failing_case);
    open_spiel::higc::Referee ref("tic_tac_toe",
                                  {failing_bot, bot_first_action});
    std::unique_ptr<TournamentResults> results = ref.PlayTournament(1);
    SPIEL_CHECK_EQ(results->num_matches(), 0);
    SPIEL_CHECK_EQ(results->corrupted_matches[0], 1);
    SPIEL_CHECK_EQ(results->corrupted_matches[1], 0);
  }
}

void PlayManyRandomMatches(int num_matches = 20) {
  open_spiel::higc::Referee ref(
      "leduc_poker",
      {absl::StrCat(absl::GetFlag(FLAGS_bots_dir), "/random_bot_py.sh"),
       absl::StrCat(absl::GetFlag(FLAGS_bots_dir), "/random_bot_cpp.sh")});
  std::unique_ptr<TournamentResults> results = ref.PlayTournament(num_matches);
  SPIEL_CHECK_EQ(results->num_matches(), num_matches);
  results->PrintCsv(std::cout, /*print_header=*/true);
}


}  // namespace
}  // namespace higc
}  // namespace open_spiel


int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::higc::PlaySingleMatchIIGS();
  open_spiel::higc::PlayWithFailingBots();
  open_spiel::higc::PlayManyRandomMatches();
}
