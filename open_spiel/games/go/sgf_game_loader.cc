// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/go/sgf_game_loader.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/go/go.h"
#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/sgf_reader.h"
#include "open_spiel/utils/status.h"

// A simple SGF game reader for Go games.
//
// Supports a subset of SGF properties seen below. This is mainly used as a way
// to start the game from a particular state.
//
// One current restriction is that is there are any setup moves (AB or AW) then
// the resulting state will not be serializable. This is because the setup moves
// are applied directly to the board and not recorded in the game history.
//
// SGF spec here:
// - https://homepages.cwi.nl/~aeb/go/misc/sgf.html
// - https://www.red-bean.com/sgf/ff1_3/ff3.html

namespace open_spiel {
namespace go {

namespace {

std::vector<std::string> kSupportedProperties = {
    "AB", "AW", "AP", "B",  "BR", "CA", "DT", "FF", "ID",
    "GM", "GC", "HA", "PW", "PB", "PC", "RE", "RO", "SZ",
    "ST", "TM", "KM", "VW", "W",  "WB", "WR"};

VirtualPoint SgfActionStrToVirtualPoint(const std::string& sgf_action_str) {
  // first convert from two letters to letter and number
  // e.g. "pd" -> "p4"
  SPIEL_CHECK_EQ(sgf_action_str.length(), 2);
  char letter =
      (sgf_action_str[0] < 'i' ? sgf_action_str[0] : sgf_action_str[0] + 1);
  int number = sgf_action_str[1] - 'a' + 1;
  std::string action_str;
  action_str.push_back(letter);
  absl::StrAppend(&action_str, number);
  return MakePoint(action_str);
}

Action ConvertAction(const GoState& state, const std::string& sgf_action_str) {
  VirtualPoint point = SgfActionStrToVirtualPoint(sgf_action_str);
  const GoBoard& board = state.board();
  return board.VirtualActionToAction(point);
}

bool LegalsContains(const std::vector<Action>& legal_actions, Action action) {
  auto iter = std::find(legal_actions.begin(), legal_actions.end(), action);
  return iter != legal_actions.end();
}
}  // namespace

bool CheckCaptureKoPassConnect(GoColor color,
                               const std::vector<Player>& player_history,
                               const SgfNode& node) {
  if (player_history.size() < 2) {
    return false;
  }
  // must be last move of the game
  if (!node.children.empty()) {
    return false;
  }
  Player last_player = player_history.back();
  Player second_last_player = player_history[player_history.size() - 2];
  return (last_player == ColorToPlayer(color) &&
          second_last_player == ColorToPlayer(OppColor(color)));
}

void AddStones(GoState* go_state, GoColor color,
               const std::vector<std::string>& property_values) {
  GoBoard* board = go_state->mutable_board();
  for (const std::string& property_value : property_values) {
    VirtualPoint point = SgfActionStrToVirtualPoint(property_value);
    SPIEL_CHECK_TRUE(board->PlayMove(point, color));
  }
}

bool ProcessRootNodeSetup(
    absl::flat_hash_map<std::string, std::vector<std::string>>* root_node_map,
    GoState* go_state) {
  bool applied_setup = false;
  if (root_node_map->contains("AB")) {
    AddStones(go_state, GoColor::kBlack, root_node_map->at("AB"));
    applied_setup = true;
  }
  if (root_node_map->contains("AW")) {
    AddStones(go_state, GoColor::kWhite, root_node_map->at("AW"));
    applied_setup = true;
  }
  return applied_setup;
}

absl::flat_hash_map<std::string, std::vector<std::string>> ReadRootNode(
    const SgfNode& node) {
  absl::flat_hash_map<std::string, std::vector<std::string>> root_node;
  for (const SgfProperty& property : node.properties) {
    root_node[property.name] = property.values;
  }
  return root_node;
}

std::vector<GameAndState> LoadGames(const std::vector<SgfNode>& nodes) {
  std::vector<GameAndState> games_and_states;

  for (const SgfNode& node : nodes) {
    absl::flat_hash_map<std::string, std::vector<std::string>> root_node_map =
        ReadRootNode(node);

    float komi = kDefaultKomi;
    if (root_node_map.contains("KM")) {
      SPIEL_CHECK_TRUE(absl::SimpleAtof(root_node_map["KM"][0], &komi));
    }
    int board_size = kDefaultBoardSize;
    if (root_node_map.contains("SZ")) {
      SPIEL_CHECK_TRUE(absl::SimpleAtoi(root_node_map["SZ"][0], &board_size));
    }
    int handicap = kDefaultHandicap;
    if (root_node_map.contains("HA")) {
      SPIEL_CHECK_TRUE(absl::SimpleAtoi(root_node_map["HA"][0], &handicap));
    }
    std::shared_ptr<const Game> game =
        LoadGame("go", {{"komi", GameParameter(komi)},
                        {"board_size", GameParameter(board_size)},
                        {"handicap", GameParameter(handicap)}});

    std::unique_ptr<State> state = game->NewInitialState();
    GoState* go_state = static_cast<GoState*>(state.get());
    bool setup_moves_applied = ProcessRootNodeSetup(&root_node_map, go_state);

    // If there are no AB or AW properties in the header, we currently do not
    // support game moves after this. So, we check: if setup moves have been
    // applied, there should be no child nodes.
    if (setup_moves_applied) {
      games_and_states.push_back({game, std::move(state)});
      SPIEL_CHECK_TRUE(node.children.empty());
      break;
    }

    // Each child at this level is an independent game.
    for (const SgfNode& child_node : node.children) {
      std::unique_ptr<State> state = game->NewInitialState();
      std::vector<Player> player_history;
      const SgfNode* current_node = &child_node;
      GoState* go_state = static_cast<GoState*>(state.get());

      while (current_node != nullptr) {
        for (const SgfProperty& property : current_node->properties) {
          std::vector<Action> legal_actions = state->LegalActions();
          SPIEL_CHECK_FALSE(state->IsTerminal());
          if (property.name == "B" || property.name == "W") {
            // Player move.
            SPIEL_CHECK_EQ(property.values.size(), 1);
            GoColor color =
                (property.name == "B" ? GoColor::kBlack : GoColor::kWhite);
            if (state->CurrentPlayer() != ColorToPlayer(color)) {
              // If it's not that player's turn, it can be an capture ko, pass,
              // or connect sequence at the end of the game. Sometimes it is
              // valid to have two moves from the same player in a row in an SGF
              // file. See the "cleaning up" section of this page:
              // https://www.red-bean.com/sgf/ff1_3/style.html
              if (CheckCaptureKoPassConnect(color, player_history,
                                            *current_node)) {
                Action pass_action =
                    static_cast<const GoGame*>(game.get())->PassAction();
                player_history.push_back(state->CurrentPlayer());
                state->ApplyAction(pass_action);
                legal_actions = state->LegalActions();
              } else {
                SpielFatalError(
                    absl::StrCat("Expected black to play, but got player ",
                                 state->CurrentPlayer(), ". Context:\n\n",
                                 property.name, " ", property.values[0]));
              }
            }
            player_history.push_back(ColorToPlayer(color));
            SPIEL_CHECK_EQ(state->CurrentPlayer(), ColorToPlayer(color));
            Action action = ConvertAction(*go_state, property.values[0]);
            SPIEL_CHECK_TRUE(LegalsContains(legal_actions, action));
            state->ApplyAction(action);
          } else if (property.name == "AB" || property.name == "AW") {
            // Set stones directly onto the board.
            GoColor color =
                (property.name == "AB" ? GoColor::kBlack : GoColor::kWhite);
            AddStones(go_state, color, property.values);
          } else {
            SpielFatalError(
                absl::StrCat("Unsupported property: ", property.name));
          }
        }

        current_node = current_node->children.empty()
                           ? nullptr
                           : &current_node->children[0];
      }

      games_and_states.push_back({game, std::move(state)});
    }
  }

  return games_and_states;
}

VectorOfGamesAndStates LoadGamesFromSGFFile(const std::string& sgf_filename) {
  return LoadGamesFromSGFString(file::ReadContentsFromFile(sgf_filename, "r"));
}

VectorOfGamesAndStates LoadGamesFromSGFString(const std::string& sgf_string) {
  StatusWithValue<std::vector<SgfNode>> status_with_nodes =
      ReadSgfString(sgf_string);
  if (!status_with_nodes.ok()) {
    SpielFatalError(status_with_nodes.message());
  }
  std::vector<SgfNode> nodes = status_with_nodes.value();
  return LoadGames(nodes);
}

}  // namespace go
}  // namespace open_spiel
