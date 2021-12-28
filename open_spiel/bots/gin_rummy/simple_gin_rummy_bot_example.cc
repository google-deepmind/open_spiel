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

#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/bots/gin_rummy/simple_gin_rummy_bot.h"
#include "open_spiel/games/gin_rummy.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel_bots.h"

ABSL_FLAG(std::string, game_string, "gin_rummy",
          "Short name plus optional params.");
ABSL_FLAG(std::string, player0, "simple_gin_rummy_bot",
          "Who controls player 0.");
ABSL_FLAG(std::string, player1, "random", "Who controls player 1.");
ABSL_FLAG(int, num_games, 1, "How many games to play.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for rng.");
ABSL_FLAG(bool, verbose, false, "Log gameplay.");
ABSL_FLAG(bool, show_legals, false, "Sets verbose=true & shows legal actions.");
ABSL_FLAG(bool, log_histories, false, "Log action histories.");
ABSL_FLAG(bool, log_returns, false, "Log returns.");
ABSL_FLAG(bool, log_reach_probs, false, "Log reach probabilities.");
ABSL_FLAG(std::string, path, "/tmp/gin_rummy_logs.txt",
          "Where to output the logs.");

uint_fast32_t Seed() {
  uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

std::unique_ptr<open_spiel::Bot> InitBot(
    std::string type, const open_spiel::Game& game, open_spiel::Player player) {
  if (type == "random") {
    return open_spiel::MakeUniformRandomBot(player, Seed());
  }
  if (type == "simple_gin_rummy_bot") {
    return std::make_unique<open_spiel::gin_rummy::SimpleGinRummyBot>(
        game.GetParameters(), player);
  }
  open_spiel::SpielFatalError(
      "Bad player type. Known types: simple_gin_rummy_bot, random");
}

std::vector<double> PlayGame(const open_spiel::Game& game,
    const std::vector<std::unique_ptr<open_spiel::Bot>>& bots,
    std::mt19937* rng, std::ostream& os, bool verbose, bool show_legals,
    bool log_histories, bool log_returns, bool log_reach_probs) {
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  for (open_spiel::Player p = 0; p < open_spiel::gin_rummy::kNumPlayers; ++p)
    bots[p]->Restart();
  std::vector<double> players_reach(2, 1.0);
  double chance_reach = 1.0;

  while (!state->IsTerminal()) {
    open_spiel::Player player = state->CurrentPlayer();

    if (verbose) os << "Player turn: " << player << std::endl;
    if (show_legals) {
      os << "Legal moves for player " << player << ":" << std::endl;
      for (open_spiel::Action action : state->LegalActions(player))
        os << "  " << state->ActionToString(player, action) << std::endl;
    }

    open_spiel::Action action;
    if (state->IsChanceNode()) {
      std::pair<open_spiel::Action, double> outcome_and_prob =
          open_spiel::SampleAction(state->ChanceOutcomes(), *rng);
      action = outcome_and_prob.first;
      SPIEL_CHECK_PROB(outcome_and_prob.second);
      SPIEL_CHECK_GT(outcome_and_prob.second, 0);
      SPIEL_CHECK_PROB(chance_reach);
      chance_reach *= outcome_and_prob.second;
      if (verbose) {
        os << "Sampled action: " << state->ActionToString(player, action)
           << std::endl;
      }
    } else {
      std::pair<open_spiel::Action, double> outcome_and_prob =
          open_spiel::SampleAction(bots[player]->GetPolicy(*state), *rng);
      action = outcome_and_prob.first;
      SPIEL_CHECK_PROB(outcome_and_prob.second);
      SPIEL_CHECK_GT(outcome_and_prob.second, 0);
      SPIEL_CHECK_PROB(players_reach[player]);
      players_reach[player] *= outcome_and_prob.second;
      if (verbose) {
        os << "Chose action: " << state->ActionToString(player, action)
           << std::endl;
      }
    }
    if (!absl::c_binary_search(state->LegalActions(), action)) {
      std::cerr << "State: " << std::endl << state->ToString() << std::endl
                << "History: " << absl::StrJoin(state->History(), " ")
                << std::endl << "Legal actions: "
                << absl::StrJoin(state->LegalActions(), " ") << std::endl;
      open_spiel::SpielFatalError("Illegal bot action.");
    }
    state->ApplyAction(action);
    if (verbose) os << "State: " << std::endl << state->ToString() << std::endl;
  }
  if (verbose) {
    os << "Returns: " << absl::StrJoin(state->Returns(), ",") << std::endl
       << "History: " << absl::StrJoin(state->History(), " ") << std::endl;
  } else if (log_histories) {
    os << absl::StrJoin(state->History(), " ") << std::endl;
  } else if (log_returns) {
    os << absl::StrJoin(state->Returns(), " ") << " ";
    if (log_reach_probs) {
      os << absl::StrJoin(players_reach, " ") << " " << chance_reach;
    }
    os << std::endl;
  }
  return state->Returns();
}

int main(int argc, char** argv) {
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);
  std::mt19937 rng(Seed());

  std::string game_string = absl::GetFlag(FLAGS_game_string);
  std::cout << "Game string: " << game_string << std::endl;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_string);

  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  bots.push_back(InitBot(absl::GetFlag(FLAGS_player0), *game, 0));
  bots.push_back(InitBot(absl::GetFlag(FLAGS_player1), *game, 1));

  int num_games = absl::GetFlag(FLAGS_num_games);
  bool show_legals = absl::GetFlag(FLAGS_show_legals);
  bool verbose = absl::GetFlag(FLAGS_verbose) || show_legals;
  bool log_histories = absl::GetFlag(FLAGS_log_histories);
  bool log_returns = absl::GetFlag(FLAGS_log_returns);
  bool log_reach_probs = absl::GetFlag(FLAGS_log_reach_probs);
  std::string path = absl::GetFlag(FLAGS_path);

  std::ofstream os(path);
  std::vector<double> overall_returns(2, 0);
  std::vector<int> overall_wins(2, 0);
  int percent = 0;
  int refresh_threshold = 0;
  absl::Time start = absl::Now();
  for (int game_num = 0; game_num < num_games; ++game_num) {
    percent = (100 * (game_num + 1)) / num_games;
    if (percent >= refresh_threshold) {
      // Progress bar.
      std::cout << "\r" << "[" << std::string(percent / 5, '=')
                << std::string(100 / 5 - percent / 5, ' ') << "]" << percent
                << "%" << " [Game " << game_num + 1 << " of " << num_games
                << "]";
      std::cout.flush();
      ++refresh_threshold;
    }
    std::vector<double> returns = PlayGame(*game, bots, &rng, os, verbose,
        show_legals, log_histories, log_returns, log_reach_probs);
    for (int i = 0; i < returns.size(); ++i) {
      double v = returns[i];
      overall_returns[i] += v;
      if (v > 0) overall_wins[i] += 1;
    }
  }
  absl::Time end = absl::Now();
  double seconds = absl::ToDoubleSeconds(end - start);

  std::cout << std::endl << "Number of games played: " << num_games << std::endl
            << "Overall wins: " << absl::StrJoin(overall_wins, ",") << std::endl
            << "Overall returns: " << absl::StrJoin(overall_returns, ",")
            << std::endl << "Seconds: " << seconds << std::endl;
  if (verbose || log_histories || log_returns)
    std::cout << "Game histories logged to " << path << std::endl;
}
