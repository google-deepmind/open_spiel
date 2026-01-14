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

#include "open_spiel/game_transforms/efg_writer.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "open_spiel/spiel.h"

namespace open_spiel {

EFGWriter::EFGWriter(const Game& game, const std::string filename,
                     bool action_names, bool separate_infostate_numbers)
    : game_(game),
      filename_(filename),
      action_names_(action_names),
      separate_infostate_numbers_(separate_infostate_numbers),
      // Node indices start at 1.
      chance_node_counter_(1),
      terminal_node_counter_(1) {
  const auto& info = game_.GetType();
  SPIEL_CHECK_EQ(info.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(info.reward_model, GameType::RewardModel::kTerminal);
  SPIEL_CHECK_NE(info.chance_mode, GameType::ChanceMode::kSampledStochastic);
}

void EFGWriter::Write() {
  std::ofstream efg_file(filename_);
  efg_file << "EFG 2 R";
  GameParameters params = game_.GetParameters();
  efg_file << " \"" << game_.ToString() << "\" { ";
  for (int i = 1; i <= game_.NumPlayers(); i++) {
    // EFG player index starts at 1.
    efg_file << '"' << "Player " << i << "\" ";
    infostate_numbers_.push_back(std::map<std::string, int>());
  }
  efg_file << "}\n";

  // Get the root state.
  Write(efg_file, *game_.NewInitialState());
  efg_file.close();
}

void EFGWriter::Write(std::ostream& f, const State& state) {
  if (state.IsTerminal()) {
    f << "t \"\" ";
    f << terminal_node_counter_;
    terminal_node_counter_++;
    f << " \"\" ";
    f << "{ ";
    for (auto r : state.Returns()) {
      f << r << " ";
    }
    f << "}\n";
    return;
  } else if (state.IsChanceNode()) {
    f << "c \"\" ";
    f << chance_node_counter_;
    chance_node_counter_++;
    f << " \"\" ";
    f << "{ ";
    for (auto action_and_probs : state.ChanceOutcomes()) {
      if (action_names_) {
        f << '"' << state.ActionToString(action_and_probs.first) << "\" ";
      } else {
        f << '"' << action_and_probs.first << "\" ";
      }
      f << std::setprecision(10) << action_and_probs.second << " ";
    }
    f << "} 0\n";
  } else {
    int p = state.CurrentPlayer();
    f << "p \"\" " << p + 1 << " ";  // EFG player index starts at 1.

    std::string key = state.InformationStateString();
    int idx = state.CurrentPlayer();
    if (!separate_infostate_numbers_) idx = 0;  // Only use one map.

    if (infostate_numbers_[idx].find(key) == infostate_numbers_[idx].end()) {
      infostate_numbers_[idx][key] = infostate_numbers_[idx].size();
    }
    f << infostate_numbers_[idx][key] + 1;  // Infostate numbering starts at 1.
    f << " \"\" { ";
    for (auto action : state.LegalActions()) {
      if (action_names_) {
        f << '"' << state.ActionToString(action) << "\" ";
      } else {
        f << '"' << action << "\" ";
      }
    }
    f << "} 0\n";
  }
  for (auto action : state.LegalActions()) {
    Write(f, *state.Child(action));
  }
}

}  // namespace open_spiel
