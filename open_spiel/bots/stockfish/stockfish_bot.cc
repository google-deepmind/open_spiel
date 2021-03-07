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

#include "open_spiel/bots/stockfish/stockfish/src/types.h"
#include "open_spiel/bots/stockfish/stockfish_bot.h"

namespace open_spiel {
namespace stockfish {

StockfishOptionsBuilder *StockfishOptionsBuilder::WithDebugLogFile(
    const std::string &debug_log_file) {
  options_["Log File"] = debug_log_file;
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithContempt(int contempt) {
  SPIEL_CHECK_GE(contempt, -100);
  SPIEL_CHECK_LE(contempt, 100);
  options_["Contempt"] = std::to_string(contempt);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithAnalysisContempt(
    StockfishAnalysisContempt analysis_contempt) {
  std::string str_val;
  switch (analysis_contempt) {
    case StockfishAnalysisContempt::kBoth:
      str_val = "Both";
      break;
    case StockfishAnalysisContempt::kOff:
      str_val = "Off";
      break;
    case StockfishAnalysisContempt::kWhite:
      str_val = "White";
      break;
    case StockfishAnalysisContempt::kBlack:
      str_val = "Black";
      break;
    default:
      SpielFatalError("Unknown stockfish analysis contempt");

  }
  options_["Analysis Contempt"] = str_val;
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithThreads(int threads) {
  SPIEL_CHECK_GE(threads, 1);
  SPIEL_CHECK_LE(threads, 512);
  options_["Threads"] = std::to_string(threads);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithHash(int hash) {
  SPIEL_CHECK_GE(hash, 1);
  SPIEL_CHECK_LE(hash, 33554432);
  options_["Hash"] = std::to_string(hash);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithPonder(bool ponder) {
  options_["Ponder"] = ponder ? "true" : "false";
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithMultiPV(int multi_pv) {
  SPIEL_CHECK_GE(multi_pv, 1);
  SPIEL_CHECK_LE(multi_pv, 500);
  options_["MultiPV"] = std::to_string(multi_pv);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithSkillLevel(
    int skill_level) {
  SPIEL_CHECK_GE(skill_level, 0);
  SPIEL_CHECK_LE(skill_level, 20);
  options_["Skill Level"] = std::to_string(skill_level);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithMoveOverHead(
    int move_overhead) {
  SPIEL_CHECK_GE(move_overhead, 0);
  SPIEL_CHECK_LE(move_overhead, 5000);
  options_["Move Overhead"] = std::to_string(move_overhead);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithSlowMover(
    int slow_mover) {
  SPIEL_CHECK_GE(slow_mover, 10);
  SPIEL_CHECK_LE(slow_mover, 1000);
  options_["Slow Mover"] = std::to_string(slow_mover);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithNodesTime(
    int nodes_time) {
  SPIEL_CHECK_GE(nodes_time, 0);
  SPIEL_CHECK_LE(nodes_time, 10000);
  options_["nodestime"] = std::to_string(nodes_time);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithUCI_Chess960(
    bool chess960) {
  options_["UCI_Chess960"] = chess960 ? "true" : "false";
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithUCI_AnalyseMode(
    bool analyse_mode) {
  options_["UCI_AnalyseMode"] = analyse_mode ? "true" : "false";
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithUCI_LimitStrength(
    bool limit_strength) {
  options_["UCI_LimitStrength"] = limit_strength ? "true" : "false";
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithUCI_Elo(int elo) {
  SPIEL_CHECK_GE(elo, 1350);
  SPIEL_CHECK_LE(elo, 2850);
  options_["UCI_Elo"] = std::to_string(elo);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithUCI_ShowWDL(
    bool show_wdl) {
  options_["UCI_ShowWDL"] = show_wdl ? "true" : "false";
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithSyzygyPath(
    const std::string &syzygy_path) {
  options_["SyzygyPath"] = syzygy_path;
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithSyzygyProbeDepth(
    int syzygy_probe_depth) {
  SPIEL_CHECK_GE(syzygy_probe_depth, 1);
  SPIEL_CHECK_LE(syzygy_probe_depth, 100);
  options_["SyzygyProbeDepth"] = std::to_string(syzygy_probe_depth);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithSyzygy50MoveRule(
    bool syzygy_50_move_rule) {
  options_["Syzygy50MoveRule"] = syzygy_50_move_rule ? "true" : "false";
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithSyzygyProbeLimit(
    int syzygy_probe_limit) {
  SPIEL_CHECK_GE(syzygy_probe_limit, 0);
  SPIEL_CHECK_LE(syzygy_probe_limit, 7);
  options_["SyzygyProbeLimit"] = std::to_string(syzygy_probe_limit);
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithUseNNUE(bool use_nnue) {
  options_["Use NNUE"] = use_nnue ? "true" : "false";
  return this;
}

StockfishOptionsBuilder *StockfishOptionsBuilder::WithEvalFile(
    const std::string &eval_file) {
  options_["EvalFile"] = eval_file;
  return this;
}

std::unique_ptr<StockfishOptionsBuilder> MakeStockfishOptionsBuilder() {
return std::make_unique<StockfishOptionsBuilder>();
};

std::unique_ptr<Bot> stockfish::MakeStockfishBot(int move_time,
                                                 bool ponder,
                                                 const uci::Options &options) {
  return uci::MakeUCIBot("stockfish", move_time, ponder, options);
}

}  // namespace stockfish
}  // namespace open_spiel
