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

#include "open_spiel/spiel.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/usage_logging.h"

namespace open_spiel {
namespace {

constexpr const int kSerializationVersion = 1;
constexpr const char* kSerializeMetaSectionHeader = "[Meta]";
constexpr const char* kSerializeGameSectionHeader = "[Game]";
constexpr const char* kSerializeGameRNGStateSectionHeader = "[GameRNGState]";
constexpr const char* kSerializeStateSectionHeader = "[State]";

// Returns the available parameter keys, to be used as a utility function.
std::string ListValidParameters(
    const GameParameters& param_spec) {
  std::vector<std::string> available_keys;
  available_keys.reserve(param_spec.size());
  for (const auto& item : param_spec) {
    available_keys.push_back(item.first);
  }
  std::sort(available_keys.begin(), available_keys.end());
  return absl::StrJoin(available_keys, ", ");
}

// Check on supplied parameters for game creation.
// Issues a SpielFatalError if any are missing, of the wrong type, or
// unexpectedly present.
void ValidateParams(const GameParameters& params,
                    const GameParameters& param_spec) {
  // Check all supplied parameters are supported and of the right type.
  for (const auto& param : params) {
    const auto it = param_spec.find(param.first);
    if (it == param_spec.end()) {
      SpielFatalError(absl::StrCat(
          "Unknown parameter '", param.first,
          "'. Available parameters are: ", ListValidParameters(param_spec)));
    }
    if (it->second.type() != param.second.type()) {
      SpielFatalError(absl::StrCat(
          "Wrong type for parameter ", param.first,
          ". Expected type: ", GameParameterTypeToString(it->second.type()),
          ", got ", GameParameterTypeToString(param.second.type()), " with ",
          param.second.ToString()));
    }
  }
  // Check we aren't missing any mandatory parameters.
  for (const auto& param : param_spec) {
    if (param.second.is_mandatory() && !params.count(param.first)) {
      SpielFatalError(absl::StrCat("Missing parameter ", param.first));
    }
  }
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const StateType& type) {
  switch (type) {
    case StateType::kMeanField: {
      os << "MEAN_FIELD";
      break;
    }
    case StateType::kChance: {
      os << "CHANCE";
      break;
    }
    case StateType::kDecision: {
      os << "DECISION";
      break;
    }
    case StateType::kTerminal: {
      os << "TERMINAL";
      break;
    }
  }
  return os;
}

StateType State::GetType() const {
  if (IsChanceNode()) {
    return StateType::kChance;
  } else if (IsTerminal()) {
    return StateType::kTerminal;
  } else if (CurrentPlayer() == kMeanFieldPlayerId) {
    return StateType::kMeanField;
  } else {
    return StateType::kDecision;
  }
}

bool GameType::ContainsRequiredParameters() const {
  for (const auto& key_val : parameter_specification) {
    if (key_val.second.is_mandatory()) {
      return true;
    }
  }
  return false;
}

GameRegisterer::GameRegisterer(const GameType& game_type, CreateFunc creator) {
  RegisterGame(game_type, creator);
}

std::shared_ptr<const Game> GameRegisterer::CreateByName(
    const std::string& short_name, const GameParameters& params) {
  auto iter = factories().find(short_name);
  if (iter == factories().end()) {
    SpielFatalError(absl::StrCat("Unknown game '", short_name,
                                 "'. Available games are:\n",
                                 absl::StrJoin(RegisteredNames(), "\n")));

  } else {
    ValidateParams(params, iter->second.first.parameter_specification);
    return (iter->second.second)(params);
  }
}

std::vector<std::string> GameRegisterer::RegisteredNames() {
  std::vector<std::string> names;
  for (const auto& key_val : factories()) {
    names.push_back(key_val.first);
  }
  return names;
}

std::vector<GameType> GameRegisterer::RegisteredGames() {
  std::vector<GameType> games;
  for (const auto& key_val : factories()) {
    games.push_back(key_val.second.first);
  }
  return games;
}

bool GameRegisterer::IsValidName(const std::string& short_name) {
  return factories().find(short_name) != factories().end();
}

void GameRegisterer::RegisterGame(const GameType& game_type,
                                  GameRegisterer::CreateFunc creator) {
  factories()[game_type.short_name] = std::make_pair(game_type, creator);
}

bool IsGameRegistered(const std::string& short_name) {
  return GameRegisterer::IsValidName(short_name);
}

std::vector<std::string> RegisteredGames() {
  return GameRegisterer::RegisteredNames();
}

std::vector<GameType> RegisteredGameTypes() {
  return GameRegisterer::RegisteredGames();
}

std::shared_ptr<const Game> DeserializeGame(const std::string& serialized) {
  std::pair<std::string, std::string> game_and_rng_state =
      absl::StrSplit(serialized, kSerializeGameRNGStateSectionHeader);

  // Remove the trailing "\n" from the game section.
  if (game_and_rng_state.first.length() > 0 &&
      game_and_rng_state.first.back() == '\n') {
    game_and_rng_state.first.pop_back();
  }
  std::shared_ptr<const Game> game = LoadGame(game_and_rng_state.first);

  if (game_and_rng_state.second.length() > 0) {
    // Game is implicitly stochastic.
    // Remove the trailing "\n" from the RNG state section.
    if (game_and_rng_state.second.back() == '\n') {
      game_and_rng_state.second.pop_back();
    }
    game->SetRNGState(game_and_rng_state.second);
  }
  return game;
}

std::shared_ptr<const Game> LoadGame(const std::string& game_string) {
  return LoadGame(GameParametersFromString(game_string));
}

std::shared_ptr<const Game> LoadGame(const std::string& short_name,
                                     const GameParameters& params) {
  std::shared_ptr<const Game> result =
      GameRegisterer::CreateByName(short_name, params);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat("Unable to create game: ", short_name));
  }
  return result;
}

std::shared_ptr<const Game> LoadGame(GameParameters params) {
  auto it = params.find("name");
  if (it == params.end()) {
    SpielFatalError(absl::StrCat("No 'name' parameter in params: ",
                                 GameParametersToString(params)));
  }
  std::string name = it->second.string_value();
  params.erase(it);
  std::shared_ptr<const Game> result =
      GameRegisterer::CreateByName(name, params);
  if (result == nullptr) {
    SpielFatalError(absl::StrCat("Unable to create game: ", name));
  }
  LogUsage();
  return result;
}

State::State(std::shared_ptr<const Game> game)
    : game_(game),
      num_distinct_actions_(game->NumDistinctActions()),
      num_players_(game->NumPlayers()),
      move_number_(0) {}

void NormalizePolicy(ActionsAndProbs* policy) {
  const double sum = absl::c_accumulate(
      *policy, 0.0, [](double& a, auto& b) { return a + b.second; });
  absl::c_for_each(*policy, [sum](auto& o) { o.second /= sum; });
}

std::pair<Action, double> SampleAction(const ActionsAndProbs& outcomes,
                                       absl::BitGenRef rng) {
  return SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0));
}
std::pair<Action, double> SampleAction(const ActionsAndProbs& outcomes,
                                       double z) {
  SPIEL_CHECK_GE(z, 0);
  SPIEL_CHECK_LT(z, 1);

  // Special case for one-item lists.
  if (outcomes.size() == 1) {
    SPIEL_CHECK_FLOAT_EQ(outcomes[0].second, 1.0);
    return outcomes[0];
  }

  // First do a check that this is indeed a proper discrete distribution.
  double sum = 0;
  for (const std::pair<Action, double>& outcome : outcomes) {
    double prob = outcome.second;
    SPIEL_CHECK_PROB(prob);
    sum += prob;
  }
  SPIEL_CHECK_FLOAT_EQ(sum, 1.0);

  // Now sample an outcome.
  sum = 0;
  for (const std::pair<Action, double>& outcome : outcomes) {
    double prob = outcome.second;
    if (sum <= z && z < (sum + prob)) {
      return outcome;
    }
    sum += prob;
  }

  // If we get here, something has gone wrong
  std::cerr << "Chance sampling failed; outcomes:" << std::endl;
  for (const std::pair<Action, double>& outcome : outcomes) {
    std::cerr << outcome.first << "  " << outcome.second << std::endl;
  }
  SpielFatalError(
      absl::StrCat("Internal error: failed to sample an outcome; z=", z));
}

std::string State::Serialize() const {
  // This simple serialization doesn't work for the following games:
  // - games with sampled chance nodes, since the history doesn't give us enough
  //   information to reconstruct the state.
  // - Mean field games, since this base class does not store the history of
  //   state distributions passed in UpdateDistribution() (and it would be
  //   very expensive to do so for games with many possible states and a long
  //   time horizon).
  // If you wish to serialize states in such games, you must implement custom
  // serialization and deserialization for the state.
  SPIEL_CHECK_NE(game_->GetType().chance_mode,
                 GameType::ChanceMode::kSampledStochastic);
  SPIEL_CHECK_NE(game_->GetType().dynamics, GameType::Dynamics::kMeanField);
  return absl::StrCat(absl::StrJoin(History(), "\n"), "\n");
}

Action State::StringToAction(Player player,
                             const std::string& action_str) const {
  for (const Action action : LegalActions()) {
    if (action_str == ActionToString(player, action)) return action;
  }
  SpielFatalError(
      absl::StrCat("Couldn't find an action matching ", action_str));
}

void State::ApplyAction(Action action_id) {
  // history_ needs to be modified *after* DoApplyAction which could
  // be using it.

  // Cannot apply an invalid action.
  SPIEL_CHECK_NE(action_id, kInvalidAction);
  Player player = CurrentPlayer();
  DoApplyAction(action_id);
  history_.push_back({player, action_id});
  ++move_number_;
}

void State::ApplyActionWithLegalityCheck(Action action_id) {
  std::vector<Action> legal_actions = LegalActions();
  if (absl::c_find(legal_actions, action_id) == legal_actions.end()) {
    Player cur_player = CurrentPlayer();
    SpielFatalError(
        absl::StrCat("Current player ", cur_player, " calling ApplyAction ",
                     "with illegal action (", action_id, "): ",
                     ActionToString(cur_player, action_id)));
  }
  ApplyAction(action_id);
}

void State::ApplyActions(const std::vector<Action>& actions) {
  // history_ needs to be modified *after* DoApplyActions which could
  // be using it.
  DoApplyActions(actions);
  history_.reserve(history_.size() + actions.size());
  for (int player = 0; player < actions.size(); ++player) {
    history_.push_back({player, actions[player]});
  }
  ++move_number_;
}

void State::ApplyActionsWithLegalityChecks(const std::vector<Action>& actions) {
  for (Player player = 0; player < actions.size(); ++player) {
    std::vector<Action> legal_actions = LegalActions(player);
    if (!legal_actions.empty() &&
        absl::c_find(legal_actions, actions[player]) == legal_actions.end()) {
      SpielFatalError(
          absl::StrCat("Player ", player, " calling ApplyAction ",
                       "with illegal action (", actions[player], "): ",
                       ActionToString(player, actions[player])));
    }
  }
  ApplyActions(actions);
}

std::vector<int> State::LegalActionsMask(Player player) const {
  int length = (player == kChancePlayerId) ? game_->MaxChanceOutcomes()
                                           : num_distinct_actions_;
  std::vector<int> mask(length, 0);
  for (int action : LegalActions(player)) mask[action] = 1;
  return mask;
}

std::vector<std::unique_ptr<State>> Game::NewInitialStates() const {
  std::vector<std::unique_ptr<State>> states;
  if (GetType().dynamics == GameType::Dynamics::kMeanField &&
      NumPlayers() >= 2) {
    states.reserve(NumPlayers());
    for (int p = 0; p < NumPlayers(); ++p) {
      states.push_back(NewInitialStateForPopulation(p));
    }
    return states;
  }
  states.push_back(NewInitialState());
  return states;
}

std::unique_ptr<State> Game::DeserializeState(const std::string& str) const {
  // This does not work for games with sampled chance nodes and for mean field
  //  games. See comments in State::Serialize() for the explanation. If you wish
  //  to serialize states in such games, you must implement custom serialization
  //  and deserialization for the state.
  SPIEL_CHECK_NE(game_type_.chance_mode,
                 GameType::ChanceMode::kSampledStochastic);
  SPIEL_CHECK_NE(game_type_.dynamics,
                 GameType::Dynamics::kMeanField);

  std::unique_ptr<State> state = NewInitialState();
  if (str.length() == 0) {
    return state;
  }
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  for (int i = 0; i < lines.size(); ++i) {
    if (lines[i].empty()) continue;
    if (state->IsSimultaneousNode()) {
      std::vector<Action> actions;
      for (int p = 0; p < state->NumPlayers(); ++p, ++i) {
        SPIEL_CHECK_LT(i, lines.size());
        Action action = static_cast<Action>(std::stol(lines[i]));
        actions.push_back(action);
      }
      state->ApplyActions(actions);
      // Must decrement i here, otherwise it is incremented too many times.
      --i;
    } else {
      Action action = static_cast<Action>(std::stol(lines[i]));
      state->ApplyAction(action);
    }
  }
  return state;
}

std::string SerializeGameAndState(const Game& game, const State& state) {
  std::string str = "";

  // Meta section.
  absl::StrAppend(&str,
                  "# Automatically generated by OpenSpiel "
                  "SerializeGameAndState\n");
  absl::StrAppend(&str, kSerializeMetaSectionHeader, "\n");
  absl::StrAppend(&str, "Version: ", kSerializationVersion, "\n");
  absl::StrAppend(&str, "\n");

  // Game section.
  absl::StrAppend(&str, kSerializeGameSectionHeader, "\n");
  absl::StrAppend(&str, game.Serialize(), "\n");

  // State section.
  absl::StrAppend(&str, kSerializeStateSectionHeader, "\n");
  absl::StrAppend(&str, state.Serialize(), "\n");

  return str;
}

std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
DeserializeGameAndState(const std::string& serialized_state) {
  std::vector<std::string> lines = absl::StrSplit(serialized_state, '\n');

  enum Section { kInvalid = -1, kMeta = 0, kGame = 1, kState = 2 };
  std::vector<std::string> section_strings = {"", "", ""};
  Section cur_section = kInvalid;

  for (int i = 0; i < lines.size(); ++i) {
    if (lines[i].length() == 0 || lines[i].at(0) == '#') {
      // Skip comments and blank lines.
    } else if (lines[i] == kSerializeMetaSectionHeader) {
      SPIEL_CHECK_EQ(cur_section, kInvalid);
      cur_section = kMeta;
    } else if (lines[i] == kSerializeGameSectionHeader) {
      SPIEL_CHECK_EQ(cur_section, kMeta);
      cur_section = kGame;
    } else if (lines[i] == kSerializeStateSectionHeader) {
      SPIEL_CHECK_EQ(cur_section, kGame);
      cur_section = kState;
    } else {
      SPIEL_CHECK_NE(cur_section, kInvalid);
      absl::StrAppend(&section_strings[cur_section], lines[i], "\n");
    }
  }

  // Remove the trailing "\n" from the game and state sections.
  if (section_strings[kGame].length() > 0 &&
      section_strings[kGame].back() == '\n') {
    section_strings[kGame].pop_back();
  }
  if (section_strings[kState].length() > 0 &&
      section_strings[kState].back() == '\n') {
    section_strings[kState].pop_back();
  }

  // We currently just ignore the meta section.
  std::shared_ptr<const Game> game = DeserializeGame(section_strings[kGame]);
  std::unique_ptr<State> state =
      game->DeserializeState(section_strings[kState]);

  return std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>(
      game, std::move(state));
}

std::ostream& operator<<(std::ostream& stream, GameType::Dynamics value) {
  switch (value) {
    case GameType::Dynamics::kSimultaneous:
      return stream << "Simultaneous";
    case GameType::Dynamics::kSequential:
      return stream << "Sequential";
    case GameType::Dynamics::kMeanField:
      return stream << "MeanField";
    default:
      SpielFatalError(absl::StrCat("Unknown dynamics: ", value));
  }
}

std::istream& operator>>(std::istream& stream, GameType::Dynamics& var) {
  std::string str;
  stream >> str;
  if (str == "Simultaneous") {
    var = GameType::Dynamics::kSimultaneous;
  } else if (str == "Sequential") {
    var = GameType::Dynamics::kSequential;
  } else if (str == "MeanField") {
    var = GameType::Dynamics::kMeanField;
  } else {
    SpielFatalError(absl::StrCat("Unknown dynamics ", str, "."));
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, GameType::ChanceMode value) {
  switch (value) {
    case GameType::ChanceMode::kDeterministic:
      return stream << "Deterministic";
    case GameType::ChanceMode::kExplicitStochastic:
      return stream << "ExplicitStochastic";
    case GameType::ChanceMode::kSampledStochastic:
      return stream << "SampledStochastic";
    default:
      SpielFatalError("Unknown mode.");
  }
}

std::ostream& operator<<(std::ostream& stream, const State& state) {
  return stream << state.ToString();
}

std::istream& operator>>(std::istream& stream, GameType::ChanceMode& var) {
  std::string str;
  stream >> str;
  if (str == "Deterministic") {
    var = GameType::ChanceMode::kDeterministic;
  } else if (str == "ExplicitStochastic") {
    var = GameType::ChanceMode::kExplicitStochastic;
  } else if (str == "SampledStochastic") {
    var = GameType::ChanceMode::kSampledStochastic;
  } else {
    SpielFatalError(absl::StrCat("Unknown chance mode ", str, "."));
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, GameType::Information value) {
  switch (value) {
    case GameType::Information::kOneShot:
      return stream << "OneShot";
    case GameType::Information::kPerfectInformation:
      return stream << "PerfectInformation";
    case GameType::Information::kImperfectInformation:
      return stream << "ImperfectInformation";
    default:
      SpielFatalError("Unknown value.");
  }
}

std::istream& operator>>(std::istream& stream, GameType::Information& var) {
  std::string str;
  stream >> str;
  if (str == "OneShot") {
    var = GameType::Information::kOneShot;
  } else if (str == "PerfectInformation") {
    var = GameType::Information::kPerfectInformation;
  } else if (str == "ImperfectInformation") {
    var = GameType::Information::kImperfectInformation;
  } else {
    SpielFatalError(absl::StrCat("Unknown information ", str, "."));
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, GameType::Utility value) {
  switch (value) {
    case GameType::Utility::kZeroSum:
      return stream << "ZeroSum";
    case GameType::Utility::kConstantSum:
      return stream << "ConstantSum";
    case GameType::Utility::kGeneralSum:
      return stream << "GeneralSum";
    case GameType::Utility::kIdentical:
      return stream << "Identical";
    default:
      SpielFatalError("Unknown value.");
  }
}

std::istream& operator>>(std::istream& stream, GameType::Utility& var) {
  std::string str;
  stream >> str;
  if (str == "ZeroSum") {
    var = GameType::Utility::kZeroSum;
  } else if (str == "ConstantSum") {
    var = GameType::Utility::kConstantSum;
  } else if (str == "GeneralSum") {
    var = GameType::Utility::kGeneralSum;
  } else if (str == "Identical") {
    var = GameType::Utility::kIdentical;
  } else {
    SpielFatalError(absl::StrCat("Unknown utility ", str, "."));
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, GameType::RewardModel value) {
  switch (value) {
    case GameType::RewardModel::kRewards:
      return stream << "Rewards";
    case GameType::RewardModel::kTerminal:
      return stream << "Terminal";
    default:
      SpielFatalError("Unknown value.");
  }
}

std::istream& operator>>(std::istream& stream, GameType::RewardModel& var) {
  std::string str;
  stream >> str;
  if (str == "Rewards") {
    var = GameType::RewardModel::kRewards;
  } else if (str == "Terminal") {
    var = GameType::RewardModel::kTerminal;
  } else {
    SpielFatalError(absl::StrCat("Unknown reward model ", str, "."));
  }
  return stream;
}

std::string Game::Serialize() const {
  std::string str = ToString();
  if (GetType().chance_mode == GameType::ChanceMode::kSampledStochastic) {
    absl::StrAppend(&str, "\n", kSerializeGameRNGStateSectionHeader, "\n",
                    GetRNGState());
  }
  return str;
}

std::string Game::ToString() const {
  GameParameters params = game_parameters_;
  params["name"] = GameParameter(game_type_.short_name);
  return GameParametersToString(params);
}

std::string GameTypeToString(const GameType& game_type) {
  std::string str = "";

  absl::StrAppend(&str, "short_name: ", game_type.short_name, "\n");
  absl::StrAppend(&str, "long_name: ", game_type.long_name, "\n");

  absl::StrAppend(&str, "dynamics: ",
                  open_spiel::internal::SpielStrCat(game_type.dynamics), "\n");

  absl::StrAppend(&str, "chance_mode: ",
                  open_spiel::internal::SpielStrCat(game_type.chance_mode),
                  "\n");

  absl::StrAppend(&str, "information: ",
                  open_spiel::internal::SpielStrCat(game_type.information),
                  "\n");

  absl::StrAppend(&str, "utility: ",
                  open_spiel::internal::SpielStrCat(game_type.utility), "\n");

  absl::StrAppend(&str, "reward_model: ",
                  open_spiel::internal::SpielStrCat(game_type.reward_model),
                  "\n");

  absl::StrAppend(&str, "max_num_players: ", game_type.max_num_players, "\n");
  absl::StrAppend(&str, "min_num_players: ", game_type.min_num_players, "\n");

  absl::StrAppend(
      &str, "provides_information_state_string: ",
      game_type.provides_information_state_string ? "true" : "false", "\n");
  absl::StrAppend(
      &str, "provides_information_state_tensor: ",
      game_type.provides_information_state_tensor ? "true" : "false", "\n");

  absl::StrAppend(&str, "provides_observation_string: ",
                  game_type.provides_observation_string ? "true" : "false",
                  "\n");
  absl::StrAppend(&str, "provides_observation_tensor: ",
                  game_type.provides_observation_tensor ? "true" : "false",
                  "\n");
  absl::StrAppend(
      &str, "provides_factored_observation_string: ",
      game_type.provides_factored_observation_string ? "true" : "false", "\n");

  // Check that there are no newlines in the serialized params.
  std::string serialized_params =
      SerializeGameParameters(game_type.parameter_specification);
  SPIEL_CHECK_TRUE(!absl::StrContains(serialized_params, "\n"));
  absl::StrAppend(&str, "parameter_specification: ", serialized_params);

  return str;
}

GameType GameTypeFromString(const std::string& game_type_str) {
  std::map<std::string, std::string> game_type_values;
  std::vector<std::string> parts = absl::StrSplit(game_type_str, '\n');

  SPIEL_CHECK_EQ(parts.size(), 15);

  for (const auto& part : parts) {
    std::pair<std::string, std::string> pair =
        absl::StrSplit(part, absl::MaxSplits(": ", 1));
    game_type_values.insert(pair);
  }

  GameType game_type = GameType();
  game_type.short_name = game_type_values.at("short_name");
  game_type.long_name = game_type_values.at("long_name");

  std::istringstream(game_type_values.at("dynamics")) >> game_type.dynamics;
  std::istringstream(game_type_values.at("chance_mode")) >>
      game_type.chance_mode;
  std::istringstream(game_type_values.at("information")) >>
      game_type.information;
  std::istringstream(game_type_values.at("utility")) >> game_type.utility;
  std::istringstream(game_type_values.at("reward_model")) >>
      game_type.reward_model;

  SPIEL_CHECK_TRUE(absl::SimpleAtoi(game_type_values.at("max_num_players"),
                                    &(game_type.max_num_players)));
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(game_type_values.at("min_num_players"),
                                    &(game_type.min_num_players)));

  game_type.provides_information_state_string =
      game_type_values.at("provides_information_state_string") == "true";
  game_type.provides_information_state_tensor =
      game_type_values.at("provides_information_state_tensor") == "true";

  game_type.provides_observation_string =
      game_type_values.at("provides_observation_string") == "true";
  game_type.provides_observation_tensor =
      game_type_values.at("provides_observation_tensor") == "true";
  game_type.provides_factored_observation_string =
      game_type_values.at("provides_factored_observation_string") == "true";

  game_type.parameter_specification =
      DeserializeGameParameters(game_type_values.at("parameter_specification"));
  return game_type;
}

std::vector<float> State::ObservationTensor(Player player) const {
  // We add this player check, to prevent errors if the game implementation
  // lacks that check (in particular as this function is the one used in
  // Python). This can lead to doing this check twice.
  // TODO(author2): Do we want to prevent executing this twice for games
  // that implement it?
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::vector<float> observation(game_->ObservationTensorSize());
  ObservationTensor(player, absl::MakeSpan(observation));
  return observation;
}

void State::ObservationTensor(Player player, std::vector<float>* values) const {
  // Retained for backwards compatibility.
  values->resize(game_->ObservationTensorSize());
  ObservationTensor(player, absl::MakeSpan(*values));
}

std::vector<float> State::InformationStateTensor(Player player) const {
  // We add this player check, to prevent errors if the game implementation
  // lacks that check (in particular as this function is the one used in
  // Python). This can lead to doing this check twice.
  // TODO(author2): Do we want to prevent executing this twice for games
  // that implement it?
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::vector<float> info_state(game_->InformationStateTensorSize());
  InformationStateTensor(player, absl::MakeSpan(info_state));
  return info_state;
}

void State::InformationStateTensor(Player player,
                                   std::vector<float>* values) const {
  // Retained for backwards compatibility.
  values->resize(game_->InformationStateTensorSize());
  InformationStateTensor(player, absl::MakeSpan(*values));
}

bool State::PlayerAction::operator==(const PlayerAction& other) const {
  return player == other.player && action == other.action;
}

int State::MeanFieldPopulation() const {
  if (GetGame()->GetType().dynamics != GameType::Dynamics::kMeanField) {
    SpielFatalError(
        "MeanFieldPopulation() does not make sense for games that are not mean "
        "field games.");
  }
  return 0;
}

std::ostream& operator<<(std::ostream& os, const State::PlayerAction& action) {
  os << absl::StreamFormat("PlayerAction(player=%i,action=%i)", action.player,
                           action.action);
  return os;
}

std::vector<std::string> ActionsToStrings(const State& state,
                                          const std::vector<Action>& actions) {
  std::vector<std::string> out;
  out.reserve(actions.size());
  for (Action action : actions) out.push_back(state.ActionToString(action));
  return out;
}

std::string ActionsToString(const State& state,
                            const std::vector<Action>& actions) {
  return absl::StrCat(
      "[", absl::StrJoin(ActionsToStrings(state, actions), ", "), "]");
}

void SpielFatalErrorWithStateInfo(const std::string& error_msg,
                                  const Game& game,
                                  const State& state) {
  // A fatal error wrapper designed to return useful debugging information.
  const std::string& info = SerializeGameAndState(game, state);
  SpielFatalError(absl::StrCat(error_msg, "Serialized state:\n", info));
}

}  // namespace open_spiel
