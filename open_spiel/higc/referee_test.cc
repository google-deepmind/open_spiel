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

#include "open_spiel/higc/referee.h"

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/higc/utils.h"

ABSL_FLAG(std::string, bots_dir, "open_spiel/higc/bots",
          "Directory containing the sources for bots.");
ABSL_FLAG(std::string, build_dir, "open_spiel/higc/bots",
          "Directory containing the binaries for bots.");
// Communication with bots runs asynchronously. Some tests can be flaky and fail
// in testing environments with preemption, see
// https://github.com/deepmind/open_spiel/pull/723
ABSL_FLAG(bool, run_only_blocking, false,
          "Do not run async tests that rely on proper timeout handling. ");

namespace open_spiel {
namespace higc {
namespace {

void SayHelloViaSubprocess() {
  Subprocess s("echo Hello", /*should_block=*/true);
  char buf[5];
  auto bytes_read = read(s.stdout(), &buf, 5);
  SPIEL_CHECK_EQ(bytes_read, 5);
  char expected[5] = {'H', 'e', 'l', 'l', 'o'};
  for (int i = 0; i < 5; ++i) SPIEL_CHECK_EQ(buf[i], expected[i]);
}

void SayHelloViaChannel() {
  // Bot channels are asynchronous -- we read from a different thread.
  std::unique_ptr<BotChannel> channel = MakeBotChannel(0, "echo Hello");
  std::thread read(ReadLineFromChannelStdout, channel.get());
  channel->StartRead(/*time_limit=*/500);
  sleep_ms(1000);
  channel->ShutDown();
  read.join();
  SPIEL_CHECK_EQ(channel->response(), "Hello");
}

void FailViaSubprocess() {
  Subprocess s("exit 1", /*should_block=*/true);
  int status;
  waitpid(s.child_pid(), &status, 0);
  SPIEL_CHECK_EQ(WEXITSTATUS(status), 1);
}

void ImportPythonDependenciesTest() {
  {
    std::cout << "Check that pyspiel can be imported: ";
    Subprocess s("python -c \"import pyspiel\"", /*should_block=*/true);
    int status;
    waitpid(s.child_pid(), &status, 0);
    int exit_code = WEXITSTATUS(status);
    SPIEL_CHECK_EQ(exit_code, 0);
    std::cout << "ok" << std::endl;
  }
  {
    std::cout << "Check that open_spiel python scripts can be imported: ";
    Subprocess s("python -c \"import open_spiel.python.observation\"",
                 /*should_block=*/true);
    int status;
    waitpid(s.child_pid(), &status, 0);
    int exit_code = WEXITSTATUS(status);
    SPIEL_CHECK_EQ(exit_code, 0);
    std::cout << "ok" << std::endl;
  }
}

void PlaySingleMatchIIGS() {
  std::string bot_first_action = absl::StrCat(
      "python ", absl::GetFlag(FLAGS_bots_dir), "/test_bot_first_action.py");
  open_spiel::higc::Referee ref(
      "goofspiel(imp_info=True,points_order=descending)",
      {bot_first_action, bot_first_action},
      /*seed=*/42,
      // Increase times for Python scripts.
      TournamentSettings{
          .timeout_ready = 2000,
          .timeout_start = 500,
      });
  std::unique_ptr<TournamentResults> results = ref.PlayTournament(1);
  SPIEL_CHECK_EQ(results->num_matches(), 1);
  SPIEL_CHECK_TRUE(results->matches[0].terminal->IsTerminal());
  SPIEL_CHECK_EQ(results->matches[0].terminal->HistoryString(),
                 "0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, "
                 "6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11");
}

void PlayWithFailingBots() {
  std::vector<std::string> failing_cases = {
      "/non_existing_bot",           "/test_bot_with_non_exec_flag",
      "/test_bot_break_pipe.sh",     "/test_bot_sleep.sh",
      "/test_bot_ready.sh",          "/test_bot_start.sh",
      "/test_bot_illegal_action.sh",
      //      "/test_bot_buffer_overflow.sh",
  };

  for (int i = 0; i < failing_cases.size(); ++i) {
    const std::string& failing_case = failing_cases[i];
    std::string failing_bot =
        absl::StrCat(absl::GetFlag(FLAGS_bots_dir), failing_case);
    std::cout << "\n\nFailing bot: " << failing_bot << std::endl;

    // Use a single-player game.
    open_spiel::higc::Referee ref(
        "cliff_walking", {failing_bot}, /*seed=*/42,
        /*settings=*/
        TournamentSettings{// Disqualify after the 2nd failing match.
                           .disqualification_rate = 0.5});
    std::unique_ptr<TournamentResults> results = ref.PlayTournament(2);
    SPIEL_CHECK_EQ(results->disqualified[0], true);
    if (i < 4) {
      // No matches are played, if the bot can't even start properly.
      SPIEL_CHECK_EQ(results->num_matches(), 0);
    } else {
      SPIEL_CHECK_EQ(results->num_matches(), 2);
    }
  }
}

void PlayWithSometimesFailingBot() {
  std::string failing_bot =
      absl::StrCat("python ", absl::GetFlag(FLAGS_bots_dir),
                   "/test_bot_fail_after_few_actions.py");
  std::cout << "\n\nFailing bot: " << failing_bot << std::endl;

  // Use a single-player game.
  open_spiel::higc::Referee ref("cliff_walking", {failing_bot}, /*seed=*/42,
                                /*settings=*/
                                TournamentSettings{
                                    // Increase times for Python scripts.
                                    .timeout_ready = 2000,
                                    .timeout_start = 500,
                                    // Disqualify after the 2nd failing match.
                                    .disqualification_rate = 0.5,
                                });
  std::unique_ptr<TournamentResults> results = ref.PlayTournament(2);
  SPIEL_CHECK_EQ(results->disqualified[0], true);
  SPIEL_CHECK_EQ(results->num_matches(), 2);
}

void PonderActTimeout() {
  open_spiel::higc::Referee ref(
      "leduc_poker",
      {absl::StrCat("python ", absl::GetFlag(FLAGS_bots_dir), "/random_bot.py"),
       absl::StrCat(absl::GetFlag(FLAGS_bots_dir), "/test_bot_start.sh")},
      /*seed=*/42,
      // Increase times for Python scripts.
      TournamentSettings{
          .timeout_ready = 2000,
          .timeout_start = 500,
      });
  std::unique_ptr<TournamentResults> results = ref.PlayTournament(1);
  SPIEL_CHECK_EQ(results->num_matches(), 1);
}

void PlayManyRandomMatches(int num_matches = 5) {
  open_spiel::higc::Referee ref(
      "leduc_poker",
      {absl::StrCat("python ", absl::GetFlag(FLAGS_bots_dir), "/random_bot.py"),
       absl::StrCat(absl::GetFlag(FLAGS_build_dir), "/random_bot")},
      /*seed=*/42,
      // Increase times for Python scripts.
      TournamentSettings{
          .timeout_ready = 2000,
          .timeout_start = 500,
      });
  std::unique_ptr<TournamentResults> results = ref.PlayTournament(num_matches);
  SPIEL_CHECK_EQ(results->num_matches(), num_matches);
  results->PrintCsv(std::cout, /*print_header=*/true);
}

void PlayWithManyPlayers() {
  constexpr const int num_bots = 8;
  std::vector<std::string> bots;
  for (int i = 0; i < num_bots; ++i) {
    bots.push_back(absl::StrCat(absl::GetFlag(FLAGS_build_dir), "/random_bot"));
  }
  open_spiel::higc::Referee ref(
      absl::StrCat("goofspiel(players=", num_bots,
                   ",imp_info=True,points_order=descending)"),
      bots,
      /*seed=*/42,
      // Increase times for Python scripts.
      TournamentSettings{
          .timeout_ready = 2000,
          .timeout_start = 500,
      });
  std::unique_ptr<TournamentResults> results = ref.PlayTournament(1);
  SPIEL_CHECK_EQ(results->num_matches(), 1);
}

}  // namespace
}  // namespace higc
}  // namespace open_spiel

// Reroute the SIGPIPE signal here, so the test pass ok.
void signal_callback_handler(int signum) {
  std::cout << "Caught signal SIGPIPE " << signum << std::endl;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  signal(SIGPIPE, signal_callback_handler);

  // General subprocess communication tests.
  // Make sure that we got the right interpreter from virtualenv.
  open_spiel::higc::SayHelloViaSubprocess();
  open_spiel::higc::FailViaSubprocess();
  open_spiel::higc::ImportPythonDependenciesTest();

  // Skip over all the other referee tests.
  if (absl::GetFlag(FLAGS_run_only_blocking)) return;

  open_spiel::higc::SayHelloViaChannel();

  // Actual bot tests.
  open_spiel::higc::PlayWithFailingBots();
  open_spiel::higc::PlayWithSometimesFailingBot();
  open_spiel::higc::PonderActTimeout();
  open_spiel::higc::PlayWithManyPlayers();
  open_spiel::higc::PlaySingleMatchIIGS();
  open_spiel::higc::PlayManyRandomMatches();
}
