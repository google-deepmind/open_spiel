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

#include "open_spiel/algorithms/nfg_writer.h"

#include <fstream>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/normal_form_game.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
const std::string GameToNFGString(const Game& game) {
  // NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
  // { "Player 1" "Player 2" } { 3 2 }
  // 1 1 0 2 0 2 1 1 0 3 2 0
  const auto* nfg = dynamic_cast<const NormalFormGame*>(&game);
  if (nfg == nullptr) {
    SpielFatalError("Must be a normal-form game");
  }

  int num_players = nfg->NumPlayers();
  std::vector<std::vector<Action>> legal_actions(num_players);
  std::unique_ptr<State> initial_state = nfg->NewInitialState();
  for (Player player = 0; player < num_players; ++player) {
    legal_actions[player] = initial_state->LegalActions(player);
  }

  // Line 1.
  std::string nfg_text =
      absl::StrCat("NFG 1 R \"OpenSpiel export of ", nfg->ToString(), "\"\n");

  // Line 2.
  absl::StrAppend(&nfg_text, "{");
  for (Player p = 0; p < num_players; ++p) {
    absl::StrAppend(&nfg_text, " \"Player ", p, "\"");
  }
  absl::StrAppend(&nfg_text, " } {");
  for (Player p = 0; p < num_players; ++p) {
    absl::StrAppend(&nfg_text, " ", legal_actions[p].size());
  }
  absl::StrAppend(&nfg_text, " }\n\n");

  // Now the payoffs.
  for (auto flat_joint_action : initial_state->LegalActions()) {
    std::vector<double> returns =
        initial_state->Child(flat_joint_action)->Returns();
    for (Player p = 0; p < returns.size(); ++p) {
      absl::StrAppendFormat(&nfg_text, "%.15g ", returns[p]);
    }
    absl::StripAsciiWhitespace(&nfg_text);
    absl::StrAppend(&nfg_text, "\n");
  }

  return nfg_text;
}

}  // namespace open_spiel
