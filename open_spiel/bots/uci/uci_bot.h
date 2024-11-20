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

#ifndef OPEN_SPIEL_BOTS_UCI_BOT_H_
#define OPEN_SPIEL_BOTS_UCI_BOT_H_

#include <cstdio>   // for size_t, needed by ::getline
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/chess/chess.h"

// **IMPORTANT NOTE** The basic test currently hangs, so consider this bot
// currently experimental. The original authors claimed to have verified it with
// external engines:
// https://github.com/deepmind/open_spiel/pull/496#issuecomment-791578615 See
// https://github.com/deepmind/open_spiel/issues/681 for details.
namespace open_spiel {
namespace uci {

using Options = std::map<std::string, std::string>;

enum class SearchLimitType {
  kMoveTime,
  kNodes,
  kDepth,
  kMate,
};

class UCIBot : public Bot {
 public:
  // Search limit value is the argument sent to either "go movetime",
  // "go depth", or "go nodes".
  UCIBot(const std::string& bot_binary_path, int search_limit_value,
         bool ponder, const Options& options,
         SearchLimitType search_limit_type = SearchLimitType::kMoveTime,
         bool use_game_history_for_position = false);
  ~UCIBot() override;

  Action Step(const State& state) override;

  std::pair<Action, std::string> StepVerbose(const State& state) override;

  void Restart() override;
  void RestartAt(const State& state) override;

  void InformAction(const State& state, Player player_id,
                    Action action) override;

  void Write(const std::string& msg) const;
  // Always blocks until a line is read.
  std::string ReadLine();

  void Position(const std::string& fen,
                const std::vector<std::string>& moves = {});

 private:
  void StartProcess(const std::string& bot_binary_path);
  void Uci();
  void SetOption(const std::string& name, const std::string& value);
  void UciNewGame();
  void IsReady();
  std::pair<std::string, absl::optional<std::string>> Go(
      absl::optional<std::string*> info_string = absl::nullopt);
  void GoPonder();
  void PonderHit();
  std::pair<std::string, absl::optional<std::string>> Stop();
  void Quit();
  std::pair<std::string, absl::optional<std::string>> ReadBestMove(
      absl::optional<std::string*> info_string = absl::nullopt);
  void PositionFromState(const chess::ChessState& state,
                         const std::vector<std::string>& extra_moves = {});

  pid_t pid_ = -1;
  int output_fd_ = -1;
  SearchLimitType search_limit_type_;
  int search_limit_value_;
  std::string search_limit_string_;
  absl::optional<std::string> ponder_move_ = absl::nullopt;
  bool was_ponder_hit_ = false;

  bool ponder_;
  bool use_game_history_for_position_ = false;

  // Input stream member variables for the bot.
  FILE* input_stream_ = nullptr;
  char* input_stream_buffer_ = nullptr;
  size_t input_stream_buffer_size_ = 0;
};

/**
 * @param bot_binary_path Path to the uci engine executable that is going to be
 * run in a new process.
 * @param move_time Time limit per move in millis. Right now chess lacks any
 * kind of time control so it is needed to provide at least this. Without any
 * time control, the uci engine behaviour is undefined (e.g. Ethereal searches
 * to depth 1, but Stockfish searches until explicitly stopped)
 * @param ponder Boolean indicating whether this bot should make the uci engine
 * ponder (think even when it's opponent's turn). In some engines, this should
 * be accompanied with an options (see param options) so that the engine can
 * adapt time control.
 * @param options Additional options to set in the engine. There might be
 * different options available for each engine.
 * @return unique_ptr to a UCIBot
 */
std::unique_ptr<Bot> MakeUCIBot(
    const std::string& bot_binary_path, int search_limit_value,
    bool ponder = false, const Options& options = {},
    SearchLimitType search_limit_type = SearchLimitType::kMoveTime,
    bool use_game_history_for_position = false);

}  // namespace uci
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_UCI_BOT_H_
