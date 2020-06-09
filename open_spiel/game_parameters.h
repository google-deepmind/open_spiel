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

#ifndef OPEN_SPIEL_GAME_PARAMETERS_H_
#define OPEN_SPIEL_GAME_PARAMETERS_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// A GameParameter can be used in 2 contexts:
// - when defining the parameters for a game, with their optional default value
//   and whether they are mandatory or not.
// - when specifying in Python a parameter value.
//
class GameParameter;

using GameParameters = std::map<std::string, GameParameter>;
std::string GameParametersToString(const GameParameters& game_params);
GameParameters GameParametersFromString(const std::string& game_string);

class GameParameter {
 public:
  enum class Type { kUnset = -1, kInt, kDouble, kString, kBool, kGame };

  explicit GameParameter(Type type = Type::kUnset, bool is_mandatory = false)
      : is_mandatory_(is_mandatory), type_(type) {}

  explicit GameParameter(int value, bool is_mandatory = false)
      : is_mandatory_(is_mandatory), int_value_(value), type_(Type::kInt) {}

  explicit GameParameter(double value, bool is_mandatory = false)
      : is_mandatory_(is_mandatory),
        double_value_(value),
        type_(Type::kDouble) {}

  explicit GameParameter(std::string value, bool is_mandatory = false)
      : is_mandatory_(is_mandatory),
        string_value_(value),
        type_(Type::kString) {}

  explicit GameParameter(bool value, bool is_mandatory = false)
      : is_mandatory_(is_mandatory), bool_value_(value), type_(Type::kBool) {}

  explicit GameParameter(std::map<std::string, GameParameter> value,
                         bool is_mandatory = false)
      : is_mandatory_(is_mandatory),
        game_value_(std::move(value)),
        type_(Type::kGame) {}

  bool has_int_value() const { return type_ == Type::kInt; }
  bool has_double_value() const { return type_ == Type::kDouble; }
  bool has_string_value() const { return type_ == Type::kString; }
  bool has_bool_value() const { return type_ == Type::kBool; }
  bool has_game_value() const { return type_ == Type::kGame; }
  Type type() const { return type_; }

  bool is_mandatory() const { return is_mandatory_; }

  // A readable string format, for display purposes; does not distinguish
  // types in ambiguous cases, e.g. string True vs boolean True.
  std::string ToString() const;

  // An unambiguous string representation, including type information.
  // Used for __repr__ in the Python interface.
  std::string ToReprString() const;

  // Everything necessary to reconstruct the parameter in string form:
  // type/value/is_mandatory.
  std::string Serialize(const std::string& delimiter = "/") const;

  int int_value() const {
    SPIEL_CHECK_TRUE(type_ == Type::kInt);
    return int_value_;
  }

  double double_value() const {
    SPIEL_CHECK_TRUE(type_ == Type::kDouble);
    return double_value_;
  }

  const std::string& string_value() const {
    SPIEL_CHECK_TRUE(type_ == Type::kString);
    return string_value_;
  }

  bool bool_value() const {
    SPIEL_CHECK_TRUE(type_ == Type::kBool);
    return bool_value_;
  }

  const std::map<std::string, GameParameter>& game_value() const {
    SPIEL_CHECK_TRUE(type_ == Type::kGame);
    return game_value_;
  }

 private:
  bool is_mandatory_;

  int int_value_;
  double double_value_;
  std::string string_value_;
  bool bool_value_;
  std::map<std::string, GameParameter> game_value_;
  Type type_;
};

std::string GameParameterTypeToString(const GameParameter::Type& type);

// Game Parameters and Game Parameter Serialization/Deserialization form:
// param_name=type/value/is_mandatory|param_name_2=type2/value2/is_mandatory2
// assumes none of the delimeters appears in the string values
std::string SerializeGameParameters(
  const GameParameters& game_params,
  const std::string& name_delimiter = "=",
  const std::string& parameter_delimeter = "|");
GameParameters DeserializeGameParameters(
  const std::string& data,
  const std::string& name_delimiter = "=",
  const std::string& parameter_delimeter = "|");                                        
GameParameter DeserializeGameParameter(const std::string& data,
                                       const std::string& delimiter = "/");

inline bool IsParameterSpecified(const GameParameters& table,
                                 const std::string& key) {
  return table.find(key) != table.end();
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_PARAMETERS_H_
