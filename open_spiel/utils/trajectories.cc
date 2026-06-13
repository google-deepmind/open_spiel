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

#include "open_spiel/utils/trajectories.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace trajectories {

Trajectory::Trajectory(const nlohmann::json& json) { ConstructFromJson(json); }

Trajectory::Trajectory(const std::string& json_str) {
  ConstructFromString(json_str);
}

Trajectory::Trajectory(const State* final_state) {
  std::string game_string = final_state->GetGame()->ToString();
  const std::vector<State::PlayerAction>& history = final_state->FullHistory();
  header_.game_string = game_string;
  header_.terminal = final_state->IsTerminal();
  header_.returns = final_state->Returns();

  // Note: we reload the game here (i.e. instead of reusing the game object
  // within final_state) on purpose, to ensure that the game can be
  // reconstructed from the game string extracted above and that the history
  // still corresponds to legal moves in the newly constructed game.
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::unique_ptr<State> state = game->NewInitialState();

  auto history_it = history.begin();
  while (history_it != history.end()) {
    if (state->IsSimultaneousNode()) {
      std::vector<Player> active_players;
      for (Player p = 0; p < game->NumPlayers(); ++p) {
        if (!state->LegalActions(p).empty()) {
          active_players.push_back(p);
        }
      }

      std::vector<Action> joint_action(game->NumPlayers(), kInvalidAction);
      for (Player p : active_players) {
        if (history_it == history.end()) {
          SpielFatalError("Unexpected end of history in simultaneous node.");
        }
        if (history_it->player != p) {
          SpielFatalError(absl::StrCat("Expected action for player ", p,
                                       " but got player ", history_it->player));
        }
        joint_action[p] = history_it->action;
        ++history_it;
      }

      for (Player p : active_players) {
        std::vector<Action> legal = state->LegalActions(p);
        if (std::find(legal.begin(), legal.end(), joint_action[p]) ==
            legal.end()) {
          SpielFatalError(absl::StrCat("Game: ", game_string, ". Player ", p,
                                       " action ", joint_action[p],
                                       " not found in legal actions ",
                                       absl::StrJoin(legal, ",")));
        }
      }

      transitions_.push_back({
          .player = kSimultaneousPlayerId,
          .action = kInvalidAction,
          .joint_action = std::make_unique<std::vector<Action>>(joint_action),
      });
      state->ApplyActions(joint_action);
    } else {
      if (history_it == history.end()) {
        SpielFatalError("Unexpected end of history.");
      }
      Player player = history_it->player;
      Action action = history_it->action;
      ++history_it;

      if (player != state->CurrentPlayer()) {
        SpielFatalError(absl::StrCat("Expected player ", state->CurrentPlayer(),
                                     " but got player ", player));
      }

      std::vector<Action> legal_actions = state->LegalActions();
      if (std::find(legal_actions.begin(), legal_actions.end(), action) ==
          legal_actions.end()) {
        SpielFatalError(absl::StrCat("Game: ", game_string, ". Action ", action,
                                     " not found in legal actions ",
                                     absl::StrJoin(legal_actions, ",")));
      }
      transitions_.push_back({.player = player, .action = action});
      state->ApplyAction(action);
    }
  }
}

void Trajectory::ConstructFromJson(const nlohmann::json& json) {
  if (json.contains("header")) {
    auto header_json = json["header"];
    header_.game_string = header_json.value("game_string", "");
    header_.terminal = header_json.value("terminal", false);
    if (header_json.contains("returns")) {
      header_.returns = header_json["returns"].get<std::vector<double>>();
    }
    if (header_json.contains("meta_data")) {
      header_.meta_data = header_json.value("meta_data", "");
    }
  } else {
    SpielFatalError("Trajectory JSON must contain header");
  }
  if (json.contains("transitions")) {
    for (const auto& transition_json : json["transitions"]) {
      Transition transition;
      transition.player = transition_json.value("player", kInvalidPlayer);
      transition.action = transition_json.value("action", kInvalidAction);
      if (transition_json.contains("legal_actions")) {
        transition.legal_actions = std::make_unique<std::vector<Action>>(
            transition_json["legal_actions"].get<std::vector<Action>>());
      }
      if (transition_json.contains("chance_outcomes")) {
        auto chance_json = transition_json["chance_outcomes"];
        auto chance_outcomes = std::make_unique<ActionsAndProbs>();
        for (const auto& item : chance_json) {
          if (item.is_array() && item.size() == 2) {
            chance_outcomes->push_back(
                {item[0].get<Action>(), item[1].get<double>()});
          } else {
            SpielFatalError(
                "chance_outcomes must be an array of [action, probability] "
                "pairs");
          }
        }
        transition.chance_outcomes = std::move(chance_outcomes);
      }
      if (transition_json.contains("joint_action")) {
        transition.joint_action = std::make_unique<std::vector<Action>>(
            transition_json["joint_action"].get<std::vector<Action>>());
      }
      transitions_.push_back(std::move(transition));
    }
  } else {
    SpielFatalError("Trajectory JSON must contain transitions");
  }
}

void Trajectory::ConstructFromString(const std::string& json_str) {
  try {
    auto json = nlohmann::json::parse(json_str);
    ConstructFromJson(json);
  } catch (const nlohmann::json::exception& e) {
    SpielFatalError(std::string("Failed to parse trajectory JSON: ") +
                    e.what());
  }
}

std::unique_ptr<State> Trajectory::ReconstructFinalState() const {
  return ReconstructHistory(nullptr);
}

std::vector<std::unique_ptr<State>> Trajectory::ReconstructAllStates() const {
  std::vector<std::unique_ptr<State>> states;
  ReconstructHistory(&states);
  return states;
}

std::unique_ptr<State> Trajectory::ReconstructHistory(
    std::vector<std::unique_ptr<State>>* states) const {
  if (header_.game_string.empty()) {
    return {};
  }
  std::shared_ptr<const Game> game = LoadGame(header_.game_string);
  std::unique_ptr<State> current_state = game->NewInitialState();
  if (states != nullptr) {
    states->push_back(current_state->Clone());
  }

  for (const auto& transition : transitions_) {
    if (transition.player == kSimultaneousPlayerId) {
      if (transition.joint_action == nullptr) {
        SpielFatalError("Simultaneous transition missing joint_action.");
      }
      for (Player p = 0; p < game->NumPlayers(); ++p) {
        std::vector<Action> legal = current_state->LegalActions(p);
        if (legal.empty()) continue;
        Action action = (*transition.joint_action)[p];
        if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
          SpielFatalError(
              absl::StrCat("Action ", action, " not legal for player ", p));
        }
      }
      current_state->ApplyActions(*transition.joint_action);
      if (states != nullptr) {
        states->push_back(current_state->Clone());
      }
    } else {
      auto legal_actions = current_state->LegalActions();
      if (std::find(legal_actions.begin(), legal_actions.end(),
                    transition.action) == legal_actions.end()) {
        SpielFatalError(absl::StrCat("Action ", transition.action,
                                     " not found in legal actions ",
                                     absl::StrJoin(legal_actions, ",")));
      }
      current_state->ApplyAction(transition.action);
      if (states != nullptr) {
        states->push_back(current_state->Clone());
      }
    }
  }
  return current_state;
}



std::string Trajectory::ToString() const {
  nlohmann::json json;
  nlohmann::json header_json;
  header_json["game_string"] = header_.game_string;
  header_json["terminal"] = header_.terminal;
  header_json["returns"] = header_.returns;
  if (!header_.meta_data.empty()) {
    header_json["meta_data"] = header_.meta_data;
  }
  json["header"] = header_json;

  nlohmann::json transitions = nlohmann::json::array();
  for (const auto& transition : transitions_) {
    nlohmann::json trans;
    trans["player"] = transition.player;
    trans["action"] = transition.action;
    if (transition.legal_actions != nullptr) {
      trans["legal_actions"] = *transition.legal_actions;
    }
    if (transition.chance_outcomes != nullptr) {
      nlohmann::json chance = nlohmann::json::array();
      for (const auto& pair : *transition.chance_outcomes) {
        chance.push_back({pair.first, pair.second});
      }
      trans["chance_outcomes"] = chance;
    }
    if (transition.joint_action != nullptr) {
      trans["joint_action"] = *transition.joint_action;
    }
    transitions.push_back(trans);
  }
  json["transitions"] = transitions;
  return json.dump();
}

}  // namespace trajectories
}  // namespace open_spiel
