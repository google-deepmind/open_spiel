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
#include "open_spiel/bots/uci/uci_bot.h"

namespace open_spiel {
namespace stockfish {

enum StockfishAnalysisContempt {
  kBoth,
  kOff,
  kWhite,
  kBlack
};

class StockfishOptionsBuilder {
 public:
  StockfishOptionsBuilder() { Reset(); };

  StockfishOptionsBuilder *WithDebugLogFile(const std::string &debug_log_file);
  StockfishOptionsBuilder *WithContempt(int contempt);
  StockfishOptionsBuilder *WithAnalysisContempt(
      StockfishAnalysisContempt analysis_contempt);
  StockfishOptionsBuilder *WithThreads(int threads);
  StockfishOptionsBuilder *WithHash(int hash);
  StockfishOptionsBuilder *WithPonder(bool ponder);
  StockfishOptionsBuilder *WithMultiPV(int multi_pv);
  StockfishOptionsBuilder *WithSkillLevel(int skill_level);
  StockfishOptionsBuilder *WithMoveOverHead(int move_overhead);
  StockfishOptionsBuilder *WithSlowMover(int slow_mover);
  StockfishOptionsBuilder *WithNodesTime(int nodes_time);
  StockfishOptionsBuilder *WithUCI_Chess960(bool chess960);
  StockfishOptionsBuilder *WithUCI_AnalyseMode(bool analyse_mode);
  StockfishOptionsBuilder *WithUCI_LimitStrength(bool limit_strength);
  StockfishOptionsBuilder *WithUCI_Elo(int elo);
  StockfishOptionsBuilder *WithUCI_ShowWDL(bool show_wdl);
  StockfishOptionsBuilder *WithSyzygyPath(const std::string &syzygy_path);
  StockfishOptionsBuilder *WithSyzygyProbeDepth(int syzygy_probe_depth);
  StockfishOptionsBuilder *WithSyzygy50MoveRule(bool syzygy_50_move_rule);
  StockfishOptionsBuilder *WithSyzygyProbeLimit(int syzygy_probe_limit);
  StockfishOptionsBuilder *WithUseNNUE(bool use_nnue);
  StockfishOptionsBuilder *WithEvalFile(const std::string &eval_file);

  uci::Options Build() {
    return options_;
  }

  void Reset() {
    options_ = {};
  }

 private:
  uci::Options options_;
};

std::unique_ptr<StockfishOptionsBuilder> MakeStockfishOptionsBuilder();

std::unique_ptr<Bot> MakeStockfishBot(int move_time,
                                      bool ponder = false,
                                      const uci::Options &options = {});

}  // namespace stockfish
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_STOCKFISH_STOCKFISH_BOT_H_
