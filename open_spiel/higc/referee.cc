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

#include <unistd.h>

#include <exception>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "open_spiel/abseil-cpp/absl/strings/escaping.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/higc/utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace higc {

// Start all players and wait for ready messages from all them simultaneously.
std::vector<bool> Referee::StartPlayers() {
  SPIEL_CHECK_EQ(game_->NumPlayers(), num_bots());

  // Launch players and create communication channels.
  log_ << "Starting players." << std::endl;
  for (int pl = 0; pl < num_bots(); ++pl) {
    const std::string& bot_command = bot_commands_[pl];
    log_ << "Bot#" << pl << ": " << bot_command << std::endl;
    errors_.push_back(BotErrors());
    channels_.push_back(MakeBotChannel(pl, bot_command));
    // Read from bot's stdout/stderr in separate threads.
    threads_stdout_.push_back(std::make_unique<std::thread>(
        ReadLineFromChannelStdout, channels_.back().get()));
    threads_stderr_.push_back(std::make_unique<std::thread>(
        ReadLineFromChannelStderr, channels_.back().get()));
  }

  // Send setup information.
  for (int pl = 0; pl < num_bots(); ++pl) {
    BotChannel* chn = channels_[pl].get();
    chn->Write(game_name_ + "\n");
    chn->Write(std::to_string(pl) + "\n");
    chn->StartRead(settings_.timeout_ready);
  }

  sleep_ms(
      settings_.timeout_ready);  // Blocking sleep to give time to the bots.
  return CheckResponses(kReadyMessage);
}

// Start a single player and wait for a ready message.
bool Referee::StartPlayer(int pl) {
  // Launch players and create communication channels.
  log_ << "Starting player " << pl << " only." << std::endl;
  const std::string& bot_command = bot_commands_[pl];
  log_ << "Bot#" << pl << ": " << bot_command << std::endl;
  channels_[pl] = MakeBotChannel(pl, bot_command);
  // Read from bot's stdout/stderr in separate threads.
  threads_stdout_[pl] = std::make_unique<std::thread>(ReadLineFromChannelStdout,
                                                      channels_.back().get());
  threads_stderr_[pl] = std::make_unique<std::thread>(ReadLineFromChannelStderr,
                                                      channels_.back().get());

  BotChannel* chn = channels_[pl].get();
  chn->Write(game_name_ + "\n");
  chn->Write(std::to_string(pl) + "\n");
  chn->StartRead(settings_.timeout_ready);

  sleep_ms(settings_.timeout_ready);  // Blocking sleep to give time to the bot.
  return CheckResponse(kReadyMessage, pl);
}

// Shut down all the players.
void Referee::ShutDownPlayers() {
  for (std::unique_ptr<BotChannel>& chn : channels_) chn->ShutDown();
  for (std::unique_ptr<std::thread>& th : threads_stdout_) th->join();
  for (std::unique_ptr<std::thread>& th : threads_stderr_) th->join();
  channels_.clear();
  threads_stdout_.clear();
  threads_stderr_.clear();
  errors_.clear();
}

// Shut down a single player.
void Referee::ShutDownPlayer(int pl) {
  log_ << "Shutting down player " << pl << " only." << std::endl;
  channels_[pl]->ShutDown();
  threads_stdout_[pl]->join();
  threads_stderr_[pl]->join();
  channels_[pl] = nullptr;
  threads_stdout_[pl] = nullptr;
  threads_stderr_[pl] = nullptr;
  errors_[pl].Reset();
}

std::unique_ptr<State> Referee::PlayMatch() {
  SPIEL_CHECK_EQ(num_bots(), game_->NumPlayers());
  std::unique_ptr<State> state = game_->NewInitialState();

  std::vector<int> player_order(num_bots());
  std::vector<bool> is_acting(num_bots(), false);
  bool only_ponder = false;  // Whether all bots only ponder (i.e chance node)
  for (int i = 0; i < num_bots(); ++i) player_order[i] = i;

  // Check start of match message.
  for (int pl = 0; pl < num_bots(); ++pl) {
    BotChannel* chn = channels_[pl].get();
    chn->StartRead(settings_.timeout_start);
  }
  sleep_ms(settings_.timeout_start);
  CheckResponses(kStartMessage);

  while (!state->IsTerminal()) {
    log_ << "\nHistory: " << absl::StrJoin(state->History(), " ") << std::endl;

    only_ponder = state->IsChanceNode();
    // Cache whether player is acting.
    for (int pl = 0; pl < num_bots(); ++pl) {
      is_acting[pl] = state->IsPlayerActing(pl);
    }
    // Make sure no player is preferred when we communicate with it.
    std::shuffle(player_order.begin(), player_order.end(), rng_);

    // Send players' observation and possibly a set of legal actions
    // available to the players.
    for (int pl : player_order) {
      BotChannel* chn = channels_[pl].get();
      public_observation_->SetFrom(*state, pl);
      private_observation_->SetFrom(*state, pl);
      std::string public_tensor = public_observation_->Compress();
      std::string private_tensor = private_observation_->Compress();

      // Send observations.
      absl::string_view public_string(
          reinterpret_cast<char* const>(public_tensor.data()),
          public_tensor.size());
      chn->Write(absl::Base64Escape(public_string));
      chn->Write(" ");
      absl::string_view private_string(
          reinterpret_cast<char* const>(private_tensor.data()),
          private_tensor.size());
      chn->Write(absl::Base64Escape(private_string));
      // Send actions.
      if (is_acting[pl]) {
        std::vector<Action> legal_actions = state->LegalActions(pl);
        for (Action a : legal_actions) {
          chn->Write(" ");
          chn->Write(std::to_string(a));
        }
      }
      chn->Write("\n");
    }

    // Start waiting for response within the time limits.
    for (int pl : player_order) {
      BotChannel* chn = channels_[pl].get();
      chn->StartRead(is_acting[pl] ? settings_.timeout_act
                                   : settings_.timeout_ponder);
    }

    // Wait for ponder messages.
    WaitForPonderingBots(is_acting);
    for (int pl = 0; pl < num_bots(); ++pl) {
      if (is_acting[pl]) continue;
      BotChannel* chn = channels_[pl].get();
      std::string response = chn->response();
      if (response != kPonderMessage) {
        log_ << "Bot#" << pl << " ponder bad response: '" << response << "'"
             << std::endl;
        errors_[pl].ponder_error++;
        if (chn->is_time_out()) {
          log_ << "Bot#" << pl << " ponder also timed out." << std::endl;
          errors_[pl].time_over++;
        }
      } else {
        log_ << "Bot#" << pl << " ponder ok." << std::endl;
      }
    }

    // Wait for response(s) from acting player(s).
    // If (all) response(s) arrive before the time limit,
    // we don't have to wait to apply the action(s).
    WaitForActingBots(is_acting);

    // Parse submitted actions based on the bot responses.
    std::vector<Action> bot_actions(num_bots(), kInvalidAction);
    for (int pl = 0; pl < num_bots(); ++pl) {
      if (!is_acting[pl]) continue;  // Ponders have been already processed.

      BotChannel* chn = channels_[pl].get();
      std::vector<Action> legal_actions = state->LegalActions(pl);

      if (chn->comm_error() != 0) {
        log_ << "Bot#" << pl
             << " act communication error: " << chn->comm_error() << std::endl;
        errors_[pl].protocol_error++;
      } else if (chn->is_time_out()) {
        log_ << "Bot#" << pl << " act timed out. " << std::endl;
        errors_[pl].time_over++;
      } else if (!chn->has_read()) {
        log_ << "Bot#" << pl << " act no response. " << std::endl;
        errors_[pl].protocol_error++;
      } else {
        std::string response = chn->response();
        log_ << "Bot#" << pl << " act response: '" << response << "'"
             << std::endl;

        int action = -1;
        bool success = absl::SimpleAtoi(response, &action);
        if (!success) {
          log_ << "Bot#" << pl << " act invalid action. " << std::endl;
          errors_[pl].protocol_error++;
        } else if (std::find(legal_actions.begin(), legal_actions.end(),
                             action) == legal_actions.end()) {
          log_ << "Bot#" << pl << " act illegal action. " << std::endl;
          errors_[pl].illegal_actions++;
        } else {
          log_ << "Bot#" << pl << " act ok. " << std::endl;
          if (errors_[pl].total_errors() > settings_.max_invalid_behaviors) {
            log_ << "Bot#" << pl << " act randomly (exceeded illegal behaviors)"
                 << std::endl;
          } else {
            bot_actions[pl] = action;
          }
        }
      }

      if (bot_actions[pl] == kInvalidAction) {  // Pick a random action.
        log_ << "Picking random action for Bot#" << pl << std::endl;
        std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
        int random_idx = dist(rng_);
        bot_actions[pl] = legal_actions[random_idx];
      }
    }
    log_ << "Submitting actions:";
    for (Action a : bot_actions) log_ << ' ' << a;
    log_ << std::endl;

    // Apply actions.
    if (state->IsChanceNode()) {
      ActionsAndProbs actions_and_probs = state->ChanceOutcomes();
      std::uniform_real_distribution<double> dist;
      const auto& [chance_action, prob] =
          SampleAction(actions_and_probs, dist(rng_));
      log_ << "Chance action: " << chance_action << " with prob " << prob
           << std::endl;
      state->ApplyAction(chance_action);
    } else if (state->IsSimultaneousNode()) {
      state->ApplyActions(bot_actions);
    } else {
      state->ApplyAction(bot_actions[state->CurrentPlayer()]);
    }
  }

  std::vector<double> returns = state->Returns();

  log_ << "\nMatch over!" << std::endl;
  log_ << "History: " << absl::StrJoin(state->History(), " ") << std::endl;

  for (int pl = 0; pl < num_bots(); ++pl) {
    int score = returns[pl];
    channels_[pl]->Write(absl::StrCat(kMatchOverMessage, " ",
                                      score, "\n"));
    channels_[pl]->StartRead(settings_.timeout_match_over);
  }

  for (int pl = 0; pl < num_bots(); ++pl) {
    log_ << "Bot#" << pl << " returns " << returns[pl] << std::endl;
    log_ << "Bot#" << pl << " protocol errors " << errors_[pl].protocol_error
         << std::endl;
    log_ << "Bot#" << pl << " illegal actions " << errors_[pl].illegal_actions
         << std::endl;
    log_ << "Bot#" << pl << " ponder errors " << errors_[pl].ponder_error
         << std::endl;
    log_ << "Bot#" << pl << " time overs " << errors_[pl].time_over
         << std::endl;
  }

  sleep_ms(settings_.timeout_match_over);
  CheckResponses(kMatchOverMessage);

  return state;
}

// Response that we do not recover from.
class UnexpectedBotResponse : std::exception {};

std::vector<bool> Referee::CheckResponses(
    const std::string& expected_response) {
  std::vector<bool> response_ok;
  response_ok.reserve(num_bots());
  for (int pl = 0; pl < num_bots(); ++pl) {
    response_ok.push_back(CheckResponse(expected_response, pl));
  }
  return response_ok;
}

bool Referee::CheckResponse(const std::string& expected_response, int pl) {
  BotChannel* chn = channels_[pl].get();
  chn->CancelReadBlocking();
  std::string response = chn->response();
  if (response != expected_response) {
    log_ << "Bot#" << pl << " did not respond '" << expected_response << "'"
         << std::endl;
    log_ << "Bot#" << pl << " response was: '" << response << "'" << std::endl;
    if (chn->comm_error() < 0) {
      log_ << "Bot#" << pl
           << " also had a communication error: " << chn->comm_error()
           << std::endl;
    }
    errors_[pl].protocol_error++;
    if (chn->is_time_out()) {
      errors_[pl].time_over++;
      log_ << "Bot#" << pl << " also timed out." << std::endl;
    }
    return false;
  } else {
    log_ << "Bot#" << pl << " " << expected_response << " ok." << std::endl;
    return true;
  }
}

void Referee::TournamentOver() {
  for (int pl = 0; pl < num_bots(); ++pl) {
    channels_[pl]->Write(absl::StrCat(kTournamentOverMessage, "\n"));
  }
  log_ << "Waiting for tournament shutdown (" << settings_.time_tournament_over
       << "ms)" << std::endl;
  sleep_ms(settings_.time_tournament_over);
  // Do not check the final message.
}

void Referee::ResetErrorTracking() {
  for (BotErrors& e : errors_) e.Reset();
}

bool Referee::corrupted_match_due(int pl) const {
  return errors_[pl].total_errors() > settings_.max_invalid_behaviors ||
         errors_[pl].protocol_error > 0;
}

void Referee::RestartPlayer(int pl) {
  ShutDownPlayer(pl);
  StartPlayer(pl);
}

Referee::Referee(const std::string& game_name,
                 const std::vector<std::string>& bot_commands, int seed,
                 TournamentSettings settings, std::ostream& log)
    : game_name_(game_name),
      game_(LoadGame(game_name)),
      bot_commands_(bot_commands),
      rng_(seed),
      log_(log),
      settings_(settings),
      public_observer_(game_->MakeObserver(kPublicObsType, {})),
      private_observer_(game_->MakeObserver(kPrivateObsType, {})),
      public_observation_(
          std::make_unique<Observation>(*game_, public_observer_)),
      private_observation_(
          std::make_unique<Observation>(*game_, private_observer_)) {
  SPIEL_CHECK_FALSE(bot_commands_.empty());
  SPIEL_CHECK_EQ(game_->NumPlayers(), num_bots());
  SPIEL_CHECK_LT(settings_.timeout_ponder, settings_.timeout_act);
}

std::unique_ptr<TournamentResults> Referee::PlayTournament(int num_matches) {
  auto results = std::make_unique<TournamentResults>(num_bots());
  std::vector<bool> start_ok = StartPlayers();
  bool all_ok = true;
  for (int pl = 0; pl < num_bots(); ++pl) {
    all_ok = all_ok && start_ok[pl];
    if (!start_ok[pl]) results->disqualified[pl] = true;
  }
  if (!all_ok) {
    log_ << "Could not start all players correctly, "
            "cannot play the tournament."
         << std::endl;
    return results;
  }

  const int corruption_threshold =
      num_matches * settings().disqualification_rate;
  int match;
  for (match = 0; match < num_matches; ++match) {
    log_ << "\n";
    for (int j = 0; j < 80; ++j) log_ << '-';
    log_ << "\nPlaying match " << match + 1 << " / " << num_matches
         << std::endl;
    for (int j = 0; j < 80; ++j) log_ << '-';
    log_ << std::endl;

    ResetErrorTracking();
    std::unique_ptr<State> state = PlayMatch();
    std::vector<double> returns = state->Returns();

    // Update mean,var statistics.
    results->history_len_mean +=
        (state->FullHistory().size() - results->history_len_mean) /
        (match + 1.);
    for (int pl = 0; pl < num_bots(); ++pl) {
      double delta = returns[pl] - results->returns_mean[pl];
      results->returns_mean[pl] += delta / (match + 1.);
      double delta2 = returns[pl] - results->returns_mean[pl];
      results->returns_agg[pl] += delta * delta2;
    }
    // Disqualifications update.
    bool tournament_over = false;
    for (int pl = 0; pl < num_bots(); ++pl) {
      if (!corrupted_match_due(pl)) continue;
      log_ << "Bot#" << pl << " exceeded illegal behaviors in match " << match
           << std::endl;
      results->corrupted_matches[pl]++;

      if (results->corrupted_matches[pl] > corruption_threshold) {
        log_ << "Bot#" << pl << " is disqualified!" << std::endl;
        results->disqualified[pl] = true;
        tournament_over = true;
      } else {
        log_ << "Bot#" << pl << " is going to restart!" << std::endl;
        ++results->restarts[pl];
        RestartPlayer(pl);
      }
    }

    results->matches.push_back(
        MatchResult{.terminal = std::move(state), .errors = errors_});

    if (tournament_over) {
      break;
    }
  }

  log_ << "\n";
  for (int j = 0; j < 80; ++j) log_ << '-';
  log_ << "\nTournament is over!" << std::endl;
  for (int j = 0; j < 80; ++j) log_ << '-';
  log_ << std::endl;

  results->PrintVerbose(log_);
  TournamentOver();
  log_ << "Shutting down players." << std::endl;
  ShutDownPlayers();

  return results;
}

void Referee::WaitForBots(const std::vector<bool>& is_acting, bool mask) {
  int num_bots_to_wait_for = 0;
  for (int pl = 0; pl < is_acting.size(); ++pl) {
    if (is_acting[pl] == mask) num_bots_to_wait_for++;
  }
  if (num_bots_to_wait_for == 0) return;

  while (true) {
    sleep_ms(1);

    int arrived_bots = 0;
    for (int pl = 0; pl < is_acting.size(); ++pl) {
      if (is_acting[pl] == mask && channels_[pl]->is_waiting_for_referee()) {
        arrived_bots++;
      }
    }
    if (arrived_bots == num_bots_to_wait_for) return;
  }
}

void Referee::WaitForPonderingBots(const std::vector<bool>& is_acting) {
  WaitForBots(is_acting, /*mask=*/false);
}

void Referee::WaitForActingBots(const std::vector<bool>& is_acting) {
  WaitForBots(is_acting, /*mask=*/true);
}

// bool Referee::StartedSuccessfully() const {
//   for (int pl = 0; pl < num_bots(); ++pl) {
//     if (channels_[pl]->exit_status() != -1) return false;
//   }
//   return true;
// }

void BotErrors::Reset() {
  protocol_error = 0;
  illegal_actions = 0;
  ponder_error = 0;
  time_over = 0;
}

int BotErrors::total_errors() const {
  return protocol_error + illegal_actions + ponder_error + time_over;
}

TournamentResults::TournamentResults(int num_bots)
    : num_bots(num_bots),
      returns_mean(num_bots, 0.),
      returns_agg(num_bots, 0.),
      corrupted_matches(num_bots, 0),
      disqualified(num_bots, false),
      restarts(num_bots, 0) {}

void TournamentResults::PrintVerbose(std::ostream& os) const {
  os << "In total played " << num_matches() << " matches." << std::endl;
  os << "Average length of a match was " << history_len_mean << " actions."
     << std::endl;
  os << "\nCorruption statistics:" << std::endl;
  for (int pl = 0; pl < num_bots; ++pl) {
    os << "Bot#" << pl << ": " << corrupted_matches[pl] << '\n';
  }

  os << "\nReturns statistics:" << std::endl;
  for (int pl = 0; pl < num_bots; ++pl) {
    double mean = returns_mean[pl];
    double var = returns_var(pl);
    os << "Bot#" << pl << " mean: " << mean << " var: " << var << std::endl;
  }
}

std::string TournamentResults::ToString() const {
  std::stringstream ss;
  PrintVerbose(ss);
  return ss.str();
}

void TournamentResults::PrintCsv(std::ostream& os, bool print_header) const {
  if (print_header) {
    os << "history,";
    for (int pl = 0; pl < num_bots; ++pl) {
      os << "returns[" << pl << "],"
         << "protocol_error[" << pl << "],"
         << "illegal_actions[" << pl << "],"
         << "ponder_error[" << pl << "],"
         << "time_over[" << pl << "]";
    }
    os << std::endl;
  }
  for (const MatchResult& match : matches) {
    os << absl::StrJoin(match.terminal->History(), " ");
    for (int pl = 0; pl < num_bots; ++pl) {
      os << ',' << match.terminal->Returns()[pl] << ','
         << match.errors[pl].protocol_error << ','
         << match.errors[pl].illegal_actions << ','
         << match.errors[pl].ponder_error << ',' << match.errors[pl].time_over;
    }
    os << std::endl;
  }
}

std::string MatchResult::ToString() const {
  std::string out = "History: " + terminal->HistoryString();
  out += "\nReturns: ";
  std::vector<double> r = terminal->Returns();
  for (int i = 0; i < r.size(); ++i) {
    out += std::to_string(r[i]) + " ";
  }
  out += "\nErrors:  ";
  for (int i = 0; i < errors.size(); ++i) {
    out += std::to_string(errors[i].total_errors()) + " ";
  }
  return out;
}

}  // namespace higc
}  // namespace open_spiel
