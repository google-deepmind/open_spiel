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

#ifndef OPEN_SPIEL_HIGC_REFEREE_
#define OPEN_SPIEL_HIGC_REFEREE_

#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "open_spiel/higc/channel.h"
#include "open_spiel/higc/subprocess.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace higc {

// Special messages that the bots should submit at appropriate occasions.
// See random bot implementation for explanation.
const char kReadyMessage[] = "ready";
const char kStartMessage[] = "start";
const char kPonderMessage[] = "ponder";
const char kMatchOverMessage[] = "match over";
const char kTournamentOverMessage[] = "tournament over";

struct TournamentSettings {
  // All times are in miliseconds.
  int timeout_ready = 200;
  int timeout_start = 100;
  int timeout_act = 100;
  int timeout_ponder = 50;
  int timeout_match_over = 100;
  int time_tournament_over = 100;

  // Number of invalid responses of a bot that are tolerated within a match.
  // Exceeding this number results in marking the match as corrupted,
  // random actions are selected instead, and the bot is forced to restart.
  // If this happens in too many matches, the bot will be disqualified.
  int max_invalid_behaviors = 1;

  // If the bot corrupts more than this fraction of tournament matches,
  // it is disqualified.
  double disqualification_rate = 0.1;
};

// Store how many errors occurred and of which type, within a match.
struct BotErrors {
  int protocol_error = 0;
  int illegal_actions = 0;
  int ponder_error = 0;
  int time_over = 0;
  int total_errors() const;
  void Reset();
};

struct MatchResult {
  std::unique_ptr<State> terminal;
  std::vector<BotErrors> errors;  // For each bot.
  std::string ToString() const;
};

struct TournamentResults {
  const int num_bots;

  // Match result for each played match.
  std::vector<MatchResult> matches;

  // Incremental computation of match statistics (mean, variance), per bot.
  std::vector<double> returns_mean;
  // For computation of variance, must be normalized first.
  std::vector<double> returns_agg;
  // Average length of a match.
  double history_len_mean = 0.;

  // Summary statistics of how many corrupted matches occurred for each player,
  // i.e. the player did not respond entirely correctly in some played match.
  //
  // A match is marked as corrupted if:
  // 1) There was a protocol error.
  // 2) The number of other errors (illegal_actions, ponder_error, time_over)
  //    exceeded the TournamentSettings::max_invalid_behaviors
  std::vector<int> corrupted_matches;

  // Flag whether a given bot was disqualified.
  // The disqualification criteria are following:
  //
  // 1) The bot could not be properly started.
  // 2) The number of corrupted matches exceeds corruption_threshold,
  //    i.e. num_matches * TournamentSettings::disqualification_rate
  std::vector<bool> disqualified;

  // Number of bot restarts. A restart is forced if a match is corrupted.
  std::vector<int> restarts;

  TournamentResults(int num_bots);
  int num_matches() const { return matches.size(); }
  double returns_var(int pl) const { return returns_agg[pl] / (num_matches()); }
  std::string ToString() const;
  void PrintVerbose(std::ostream&) const;
  void PrintCsv(std::ostream&, bool print_header = false) const;
};

// Referee that communicates with the bots and provides them with observations
// of the current state of the game.
class Referee {
  std::string game_name_;
  std::shared_ptr<const Game> game_;
  std::vector<std::string> bot_commands_;
  std::mt19937 rng_;
  std::ostream& log_;
  TournamentSettings settings_;
  std::shared_ptr<Observer> public_observer_;
  std::shared_ptr<Observer> private_observer_;
  std::unique_ptr<Observation> public_observation_;
  std::unique_ptr<Observation> private_observation_;

  std::vector<BotErrors> errors_;
  std::vector<std::unique_ptr<BotChannel>> channels_;
  std::vector<std::unique_ptr<std::thread>> threads_stdout_;
  std::vector<std::unique_ptr<std::thread>> threads_stderr_;

 public:
  Referee(const std::string& game_name,
          const std::vector<std::string>& bot_commands, int seed = 42,
          TournamentSettings settings = TournamentSettings(),
          std::ostream& log = std::cout);
  ~Referee() { ShutDownPlayers(); }
  std::unique_ptr<TournamentResults> PlayTournament(int num_matches);
  //  bool StartedSuccessfully() const;

  int num_bots() const { return bot_commands_.size(); }
  const TournamentSettings& settings() const { return settings_; }

 private:
  int total_errors(int pl) const { return errors_[pl].total_errors(); }
  // Did the player corrupt the current match?
  bool corrupted_match_due(int pl) const;

  std::unique_ptr<State> PlayMatch();
  std::vector<bool> StartPlayers();
  void ShutDownPlayers();
  void RestartPlayer(int pl);

  void ResetErrorTracking();
  void TournamentOver();

  bool StartPlayer(int pl);
  void ShutDownPlayer(int pl);
  bool CheckResponse(const std::string& expected_response, int pl);
  std::vector<bool> CheckResponses(const std::string& expected_response);
  void WaitForPonderingBots(const std::vector<bool>& is_acting);
  void WaitForActingBots(const std::vector<bool>& is_acting);
  void WaitForBots(const std::vector<bool>& is_acting, bool mask);
};

}  // namespace higc
}  // namespace open_spiel

#endif  // OPEN_SPIEL_HIGC_REFEREE_
