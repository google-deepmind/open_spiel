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


#ifndef OPEN_SPIEL_BOTS_STOCKFISH_STOCKFISH_BOT_H_
#define OPEN_SPIEL_BOTS_STOCKFISH_STOCKFISH_BOT_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel_bots.h"
#include "open_spiel/games/chess.h"
#include "open_spiel/bots/stockfish/stockfish/src/types.h"

namespace open_spiel {
namespace stockfish {

class StockfishBot : public Bot {
 public:
  explicit StockfishBot(int argc, char **argv, int move_time);

  Action Step(const State& state) override;
  void InformAction(const State& state, Player player_id,
                    Action action) override;
  void Restart() override;
  void RestartAt(const State& state) override;

 private:
  static chess::Move MoveFromStockfishMove(Move move, const chess::StandardChessBoard &board);

  int move_time_;
};

std::unique_ptr<Bot> MakeStockfishBot(int argc, char **argv, int move_time);

}  // namespace stockfish
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_STOCKFISH_STOCKFISH_BOT_H_
