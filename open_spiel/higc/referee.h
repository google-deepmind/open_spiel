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

#ifndef OPEN_SPIEL_HIGC_REFEREE_
#define OPEN_SPIEL_HIGC_REFEREE_

#include <thread>
#include <mutex>

#include "open_spiel/spiel.h"
#include "open_spiel/higc/subprocess.hpp"

namespace open_spiel {
namespace higc {

// All times are in miliseconds.
struct TournamentSettings {
  int timeout_ready = 200;
  int timeout_start = 100;
  int timeout_act = 100;
  int timeout_ponder = 50;
  int timeout_match_over = 100;
  int time_tournament_over = 100;
  int max_invalid_behaviors = 1;
  double disqualification_rate = 0.1;
};

// Store how many errors occured and of which type.
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
  std::vector<BotErrors> errors;
};

struct TournamentResults {
  const int num_bots;
  // For each match.
  std::vector<MatchResult> matches;

  // Incremental computation of match statistics (mean, variance), per bot.
  std::vector<double> returns_mean;
  std::vector<double> returns_agg;  // For computation of variance,
                                    // must be normalized first.
  // Average length of a match.
  double history_len_mean = 0.;
  // Summary statistics of how many corrupted matches occurred for each player,
  // i.e. the player did not respond entirely correctly in the match.
  std::vector<int> corrupted_matches;
  // Flag whether a given player was disqualified.
  std::vector<bool> disqualified;
  // Number of bot restarts.
  std::vector<int> restarts;

  TournamentResults(int num_bots);
  int num_matches() const { return matches.size(); }
  void PrintVerbose(std::ostream&);
  void PrintCsv(std::ostream&, bool print_header = false);
};

// Communication channel with the bot.
class BotChannel {
 public:
  BotChannel(int bot_index, std::unique_ptr<subprocess::popen> popen)
      : bot_index_(bot_index), popen_(std::move(popen)) {}
  int in() { return popen_->stdin(); }
  int out() { return popen_->stdout(); }
  int err() { return popen_->stderr(); };

  void StartRead(int time_limit);
  void CancelReadBlocking();
  void ShutDown();

  bool has_read() const { return !response_.empty(); }
  bool is_time_out() const { return time_out_; }
  std::string response() const { return response_; }

 private:
  int bot_index_;
  std::unique_ptr<subprocess::popen> popen_;
  std::string response_;    // A complete line response.
  std::string buf_;         // Incomplete response buffer.
  bool time_out_ = false;

  std::atomic<bool> shutdown_ = false;
  std::atomic<bool> wait_for_message_ = true;
  int time_limit_ = 0;
  bool cancel_read_ = false;
  std::mutex mx_read;

  // Reading thread loops.
  friend void ReadLineFromChannelStdout(BotChannel* c);
  friend void ReadLineFromChannelStderr(BotChannel* c);
};

// Referee that communicates with the bots and provides them with observations
// of the current state of the game.
class Referee {
  std::string game_name_;
  std::shared_ptr<const Game> game_;
  std::vector<std::string> executables_;
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
          const std::vector<std::string>& executables,
          int seed = 42,
          TournamentSettings settings = TournamentSettings(),
          std::ostream& log = std::cout);
  ~Referee() { ShutDownPlayers(); }

  std::unique_ptr<TournamentResults> PlayTournament(int num_matches);

  int num_bots() const { return executables_.size(); }
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
};

}  // namespace higc
}  // namespace open_spiel

#endif  // OPEN_SPIEL_HIGC_REFEREE_
