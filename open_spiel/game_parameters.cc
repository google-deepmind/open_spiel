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

#include "open_spiel/game_parameters.h"

#include <iostream>
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

std::string GameParameter::ToReprString() const {
  switch (type_) {
    case Type::kInt:
      return absl::StrCat("GameParameter(int_value=", int_value(), ")");
    case Type::kDouble:
      return absl::StrCat("GameParameter(double_value=", double_value(), ")");
    case Type::kString:
      return absl::StrCat("GameParameter(string_value='", string_value(), "')");
    case Type::kBool:
      return absl::StrCat(
          "GameParameter(bool_value=", bool_value() ? "True" : "False", ")");
    case Type::kUnset:
      return absl::StrCat("GameParameter()");
    case Type::kGame:
      return absl::StrCat("GameParameter(game_value=",
                          GameParametersToString(game_value()));
    default:
      SpielFatalError("Unknown type.");
      return "This will never return.";
  }
}

std::string GameParameter::ToString() const {
  switch (type_) {
    case Type::kInt:
      return absl::StrCat(int_value());
    case Type::kDouble:
      return absl::StrCat(double_value());
    case Type::kString:
      return string_value();
    case Type::kBool:
      return bool_value() ? std::string("True") : std::string("False");
    case Type::kUnset:
      return absl::StrCat("unset");
    case Type::kGame:
      return GameParametersToString(game_value());
    default:
      SpielFatalError("Unknown type.");
      return "This will never return.";
  }
}

inline std::string GameParametersToString(const GameParameters& game_params) {
  std::string str;
  if (game_params.count("name")) str = game_params.at("name").string_value();
  str.push_back('(');
  bool first = true;
  for (auto key_val : game_params) {
    if (key_val.first != "name") {
      if (!first) str.push_back(',');
      str.append(key_val.first);
      str.append("=");
      str.append(key_val.second.ToString());
      first = false;
    }
  }
  str.push_back(')');
  return str;
}

GameParameter GameParameterFromString(const std::string& str) {
  if (str == "True" || str == "true")
    return GameParameter(true);
  else if (str == "False" || str == "false")
    return GameParameter(false);
  else if (str.find_first_not_of("+-0123456789") == std::string::npos)
    return GameParameter(stoi(str));
  else if (str.find_first_not_of("+-0123456789.") == std::string::npos)
    return GameParameter(stod(str));
  else if (str.back() == ')')
    return GameParameter(GameParametersFromString(str));
  else
    return GameParameter(str);
}

GameParameters GameParametersFromString(const std::string& game_string) {
  GameParameters params;
  int first_paren = game_string.find('(');
  if (first_paren == std::string::npos) {
    params["name"] = GameParameter(game_string);
    return params;
  }
  params["name"] = GameParameter(game_string.substr(0, first_paren));
  int start = first_paren + 1;
  int parens = 1;
  int equals = -1;
  for (int i = start; i < game_string.length(); ++i) {
    if (game_string[i] == '(') {
      ++parens;
    } else if (game_string[i] == ')') {
      --parens;
    } else if (game_string[i] == '=' && parens == 1) {
      equals = i;
    }
    if ((game_string[i] == ',' && parens == 1) ||
        (game_string[i] == ')' && parens == 0 && i > start + 1)) {
      params[game_string.substr(start, equals - start)] =
          GameParameterFromString(
              game_string.substr(equals + 1, i - equals - 1));
      start = i + 1;
      equals = -1;
    }
  }
  return params;
}

std::string GameParameterTypeToString(const GameParameter::Type& type) {
  switch (type) {
    case GameParameter::Type::kUnset:
      return "kUnset";
    case GameParameter::Type::kInt:
      return "kInt";
    case GameParameter::Type::kDouble:
      return "kDouble";
    case GameParameter::Type::kString:
      return "kString";
    case GameParameter::Type::kBool:
      return "kBool";
    case GameParameter::Type::kGame:
      return "kGame";
    default:
      SpielFatalError("Invalid GameParameter");
  }
}

}  // namespace open_spiel
