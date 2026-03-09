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

#include "open_spiel/games/go/sgf_reader.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/go/go.h"
#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

// A simple SGF game reader for Go games.
//
// Supports a subset of SGF properties seen below. This is mainly used as a way
// to start the game from a particular state.
//
// One restriction is that there cannot both have (B or W) properties in the
// same variation if there are also (AB or AW) properties. In other words, each
// variation can only have one of (B or W) properties or (AB or AW) properties,
// but not both.
//
// SGF spec here:
// - https://homepages.cwi.nl/~aeb/go/misc/sgf.html
// - https://www.red-bean.com/sgf/ff1_3/ff3.html

namespace open_spiel {
namespace go {

namespace {

constexpr int kContextWidth = 10;

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

std::string SGFReader::GetContext(int index) const {
  int start_index = std::max(0, index - kContextWidth);
  int end_index = std::min<int>(sgf_string_.length(), index + kContextWidth);
  return sgf_string_.substr(start_index, end_index - start_index);
}

std::string SGFReader::GetCurrentGame() const {
  std::string game = sgf_string_.substr(current_game_index_);
  int end_index = game.find(')');
  return game.substr(0, end_index + 1);
}

void SGFReader::CheckCharAtIndex(int index, char expected) const {
  SPIEL_CHECK_LT(index, sgf_string_.length());
  if (sgf_string_[index] != expected) {
    std::string error_message = "Expected character ";
    absl::StrAppend(&error_message, std::string("") + expected);
    absl::StrAppend(&error_message, " but got ");
    absl::StrAppend(&error_message, std::string("") + sgf_string_[index]);
    absl::StrAppend(&error_message, " at index ");
    absl::StrAppend(&error_message, index);
    absl::StrAppend(&error_message, " in SGF string. \nContext:\n\n");
    absl::StrAppend(&error_message, GetContext(index));
    SpielFatalError(error_message);
  }
}

bool SGFReader::CheckCaptureKoPassConnect(
    GoColor color, const std::vector<Player>& player_history) const {
  if (player_history.size() < 2) {
    return false;
  }
  // must be last move of the game
  if (sgf_string_[index_] != ')') {
    return false;
  }
  Player last_player = player_history.back();
  Player second_last_player = player_history[player_history.size() - 2];
  return (last_player == ColorToPlayer(color) &&
          second_last_player == ColorToPlayer(OppColor(color)));
}

void SGFReader::SkipWhitespace() {
  while (index_ < sgf_string_.length() &&
         absl::ascii_isspace(sgf_string_[index_])) {
    ++index_;
  }
}

std::string SGFReader::ReadPropertyName() {
  SkipWhitespace();
  for (const std::string& property_name : kSupportedProperties) {
    if (absl::StartsWith(absl::string_view(sgf_string_).substr(index_),
                         property_name) &&
        index_ + property_name.length() < sgf_string_.length() &&
        sgf_string_[index_ + property_name.length()] == '[') {
      index_ += property_name.length();
      return property_name;
    }
  }
  return "";
}

std::vector<std::string> SGFReader::ReadPropertyValues() {
  std::vector<std::string> property_values;
  while (index_ < sgf_string_.length() && sgf_string_[index_] == '[') {
    CheckCharAtIndex(index_, '[');
    ++index_;
    std::string property_value = "";
    while (index_ < sgf_string_.length() && sgf_string_[index_] != ']') {
      property_value += sgf_string_[index_];
      ++index_;
    }
    CheckCharAtIndex(index_, ']');
    ++index_;
    property_values.push_back(property_value);
  }
  return property_values;
}

absl::flat_hash_map<std::string, std::string> SGFReader::ReadRootNode() {
  SkipWhitespace();
  CheckCharAtIndex(index_, ';');
  ++index_;
  absl::flat_hash_map<std::string, std::string> root_node;
  std::string property_name = ReadPropertyName();
  while (!property_name.empty()) {
    std::vector<std::string> property_value = ReadPropertyValues();
    if (property_name == "AB" || property_name == "AW") {
      root_node[property_name] = absl::StrJoin(property_value, ",");
    } else {
      SPIEL_CHECK_EQ(property_value.size(), 1);
      root_node[property_name] = property_value[0];
    }
    SkipWhitespace();
    property_name = ReadPropertyName();
  }
  return root_node;
}

std::vector<PropertyValuesPair> SGFReader::ReadNextNode() {
  SkipWhitespace();
  std::vector<PropertyValuesPair> property_names_and_values;
  if (index_ >= sgf_string_.length() || sgf_string_[index_] == ')') {
    return property_names_and_values;
  }
  CheckCharAtIndex(index_, ';');
  ++index_;
  while (index_ < sgf_string_.length() && sgf_string_[index_] != ';' &&
         sgf_string_[index_] != ')') {
    std::string property_name = ReadPropertyName();
    std::vector<std::string> property_values = ReadPropertyValues();
    property_names_and_values.push_back({property_name, property_values});
    SkipWhitespace();
  }
  return property_names_and_values;
}

void SGFReader::AddStones(
    GoState* go_state, GoColor color,
    const std::vector<std::string>& property_values) const {
  GoBoard* board = go_state->mutable_board();
  for (const std::string& property_value : property_values) {
    VirtualPoint point = SgfActionStrToVirtualPoint(property_value);
    SPIEL_CHECK_TRUE(board->PlayMove(point, color));
  }
}

void SGFReader::ProcessRootNodeSetup(
    absl::flat_hash_map<std::string, std::string>* root_node,
    GoState* go_state) const {
  if (root_node->contains("AB")) {
    AddStones(go_state, GoColor::kBlack,
              absl::StrSplit(root_node->at("AB"), ','));
    root_node->erase("AB");
  }
  if (root_node->contains("AW")) {
    AddStones(go_state, GoColor::kWhite,
              absl::StrSplit(root_node->at("AW"), ','));
    root_node->erase("AW");
  }
}

std::vector<GameAndState> SGFReader::ReadNextGames() {
  SkipWhitespace();
  if (index_ >= sgf_string_.length()) {
    return {};
  }

  current_game_index_ = index_;
  CheckCharAtIndex(index_, '(');
  ++index_;

  int parenthesis_depth = 1;

  absl::flat_hash_map<std::string, std::string> root_node = ReadRootNode();

  float komi = kDefaultKomi;
  if (root_node.contains("KM")) {
    SPIEL_CHECK_TRUE(absl::SimpleAtof(root_node["KM"], &komi));
  }
  int board_size = kDefaultBoardSize;
  if (root_node.contains("SZ")) {
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(root_node["SZ"], &board_size));
  }
  int handicap = kDefaultHandicap;
  if (root_node.contains("HA")) {
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(root_node["HA"], &handicap));
  }
  std::shared_ptr<const Game> game =
      LoadGame("go", {{"komi", GameParameter(komi)},
                      {"board_size", GameParameter(board_size)},
                      {"handicap", GameParameter(handicap)}});

  SkipWhitespace();
  std::vector<GameAndState> games_and_states;

  while (parenthesis_depth > 0) {
    std::unique_ptr<State> state = game->NewInitialState();
    std::vector<Player> player_history;

    // Root node could have AB and AW properties. Process these first.
    if (root_node.contains("AB") || root_node.contains("AW")) {
      ProcessRootNodeSetup(&root_node, static_cast<GoState*>(state.get()));
    }

    if (sgf_string_[index_] == '(') {
      parenthesis_depth++;
      index_++;
      SkipWhitespace();
    }

    GoState* go_state = static_cast<GoState*>(state.get());

    std::vector<PropertyValuesPair> property_names_and_values = ReadNextNode();
    while (!property_names_and_values.empty()) {
      for (const auto& property_name_and_values : property_names_and_values) {
        std::vector<Action> legal_actions = state->LegalActions();
        SPIEL_CHECK_FALSE(state->IsTerminal());
        if (property_name_and_values.first == "B" ||
            property_name_and_values.first == "W") {
          // Player move.
          SPIEL_CHECK_EQ(property_name_and_values.second.size(), 1);
          GoColor color =
              (property_name_and_values.first == "B" ? GoColor::kBlack
                                                     : GoColor::kWhite);
          if (state->CurrentPlayer() != ColorToPlayer(color)) {
            // If it's not that player's turn, it can be an capture ko, pass, or
            // connect sequence at the end of the game. Sometimes it is valid
            // to have two moves from the same player in a row in an SGF file.
            // See the "cleaning up" section of this page:
            // https://www.red-bean.com/sgf/ff1_3/style.html
            if (CheckCaptureKoPassConnect(color, player_history)) {
              Action pass_action =
                  static_cast<const GoGame*>(game.get())->PassAction();
              player_history.push_back(state->CurrentPlayer());
              state->ApplyAction(pass_action);
              legal_actions = state->LegalActions();
            } else {
              SpielFatalError(absl::StrCat(
                  "Expected black to play, but got player ",
                  state->CurrentPlayer(), ". Context:\n\n",
                  property_name_and_values.first, " ",
                  property_name_and_values.second[0], "\n", GetCurrentGame()));
            }
          }
          player_history.push_back(ColorToPlayer(color));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), ColorToPlayer(color));
          Action action =
              ConvertAction(*go_state, property_name_and_values.second[0]);
          SPIEL_CHECK_TRUE(LegalsContains(legal_actions, action));
          state->ApplyAction(action);
        } else if (property_name_and_values.first == "AB" ||
                   property_name_and_values.first == "AW") {
          // Set stones directly onto the board.
          GoColor color =
              (property_name_and_values.first == "AB" ? GoColor::kBlack
                                                      : GoColor::kWhite);
          AddStones(go_state, color, property_name_and_values.second);
        } else {
          SpielFatalError(absl::StrCat("Unsupported property: ",
                                       property_name_and_values.first));
        }
      }

      SkipWhitespace();
      property_names_and_values = ReadNextNode();
    }
    CheckCharAtIndex(index_, ')');
    ++index_;
    SPIEL_CHECK_GT(parenthesis_depth, 0);
    parenthesis_depth--;
    SkipWhitespace();

    games_and_states.push_back({game, std::move(state)});

    // Check if we have reached the end. If so, break.
    if (index_ >= sgf_string_.length()) {
      break;
    } else if (sgf_string_[index_] == ')') {
      SPIEL_CHECK_GT(parenthesis_depth, 0);
      parenthesis_depth--;
      index_++;
    }
  }

  SPIEL_CHECK_EQ(parenthesis_depth, 0);
  return games_and_states;
}

VectorOfGamesAndStates LoadGamesFromSGFFile(const std::string& sgf_filename) {
  return LoadGamesFromSGFString(file::ReadContentsFromFile(sgf_filename, "r"));
}

VectorOfGamesAndStates LoadGamesFromSGFString(const std::string& sgf_string) {
  SGFReader sgf_reader(sgf_string);
  VectorOfGamesAndStates all_games_and_states;
  VectorOfGamesAndStates games_and_states = sgf_reader.ReadNextGames();
  while (!games_and_states.empty()) {
    while (!games_and_states.empty()) {
      all_games_and_states.push_back(std::move(games_and_states.back()));
      games_and_states.pop_back();
    }
    games_and_states = sgf_reader.ReadNextGames();
  }
  return all_games_and_states;
}

}  // namespace go
}  // namespace open_spiel
