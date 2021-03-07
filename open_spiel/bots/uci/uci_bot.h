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


#ifndef OPEN_SPIEL_BOTS_STOCKFISH_UCI_BOT_H_
#define OPEN_SPIEL_BOTS_STOCKFISH_UCI_BOT_H_

#include "open_spiel/spiel_bots.h"

#include "open_spiel/games/chess.h"

namespace open_spiel {
namespace uci {

using Options = std::map<std::string, std::string>;

class UCIBot : public Bot {
 public:
  explicit UCIBot(const std::string &path,
                  int move_time,
                  bool ponder,
                  const Options &options);
  ~UCIBot() override;

  Action Step(const State& state) override;
  void Restart() override;
  void RestartAt(const State& state) override;

  void InformAction(const State &state, Player player_id, Action action) override;

 private:

  void StartProcess(const std::string& path);
  void Uci();
  void SetOption(const std::string& name, const std::string& value);
  void UciNewGame();
  void IsReady();
  void Position(const std::string& fen,
                const std::vector<std::string> &moves = {});
  std::pair<std::string, std::optional<std::string>> Go();
  void GoPonder();
  void PonderHit();
  std::pair<std::string, std::optional<std::string>> Stop();
  void Quit();
  std::pair<std::string, std::optional<std::string>> ReadBestMove();

  void Write(const std::string& msg) const;
  std::string Read(bool wait) const;

  pid_t pid_ = -1;
  int input_fd_ = -1;
  int output_fd_ = -1;
  int move_time_;
  std::optional<std::string> ponder_move_ = std::nullopt;
  bool was_ponder_hit_ = false;

  bool ponder_;
};

std::unique_ptr<Bot> MakeUCIBot(const std::string &path,
                                int move_time,
                                bool ponder = false,
                                const Options &options = {});

}  // namespace uci
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_STOCKFISH_UCI_BOT_H_
