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

#ifndef OPEN_SPIEL_GAME_PARAMETERS_H_
#define OPEN_SPIEL_GAME_PARAMETERS_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/heterogenous_lookup.h"
#include "open_spiel/abseil-cpp/absl/container/btree_map.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"

namespace open_spiel {

// A GameParameter can be used in 3 contexts:
// - when defining the parameters for a game, with their optional default value
//   and whether they are mandatory or not.
// - when specifying in Python a parameter value.
// - when passing a parametrization of additional Observer fields.
//
class GameParameter;

// NOTE: the usage of std::map< std::string, GameParameter>
// (alias: GameParameters) as a member of GameParamter as type is
// undefined behaviour in c++17. This is due to GameParameter remaining
// an incomplete type at this point of its class definition. Incomplete
// value types as template type argument are only allowed for certain
// containers in the std library (e.g. shared/unique_ptr pre c++17 and
// certain containers such as vector or list as of c++17).
// 'map' has not yet defined behaviour for incomplete value types.
// This forces us to wrap the value type in a unique_ptr
//using GameParameters =
//    absl::flat_hash_map<std::string, GameParameter, internal::StringHasher,
//                        internal::StringEq>;
using GameParameters =
    absl::btree_map<std::string, std::shared_ptr<const GameParameter>,
                    internal::StringCmp>;
std::string GameParametersToString(const GameParameters& game_params);
GameParameters GameParametersFromString(std::string_view game_string);

/// use SFINAE to filter out any passed types that are not `GameParameters`
/// NOTE: we use a templated function here to avoid duplicating code for
/// const l-values and pure rvalues which can be treated the same way in the
/// function body (but a const ref only declaration would force a copy on
/// r-values)
template <typename T1, typename T2,
          typename = std::enable_if_t<
              internal::all_of_v<GameParameters, internal::remove_cvref_t<T1>,
                                 internal::remove_cvref_t<T2>>>>
bool GameParametersEquality(T1 &&parameters1, T2 &&parameters2) {
  return (parameters1.size() == parameters2.size()) and
         std::all_of(parameters1.begin(), parameters1.end(),
                     [&, end_it = parameters2.end()](const auto &key_val) {
                       const auto &[key, parameter_sptr] = key_val;
                       auto search_it = parameters2.find(key);
                       if (search_it == end_it) {
                         // key of parameter set 1 is not present in parameter
                         // set 2
                         // --> the two sets are not equal
                         return false;
                       }
                       // key is present in the second set --> compare the
                       // values of both
                       // (we need to deref because they are shared ptrs)
                       return *(search_it->second) == *parameter_sptr;
                     });
}

inline constexpr const char* kDefaultNameDelimiter = "=";
inline constexpr const char* kDefaultParameterDelimiter = "|||";
inline constexpr const char* kDefaultInternalDelimiter = "***";

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

  explicit GameParameter(std::string_view value, bool is_mandatory = false)
      : is_mandatory_(is_mandatory),
        string_value_(value),
        type_(Type::kString) {}

  // Allows construction of a `GameParameter` from a string literal. This method
  // is not subsumed by the previous method, even if value can be converted to a
  // std::string, because the [C++ standard][iso] requires that the *standard
  // conversion sequence* (see §13.3.3.1.1)
  // `(const char[]) -> const char* -> bool` take precedence over the
  // *user-defined conversion sequence*
  // `(const char[]) -> const char* -> std::string` defined in the standard
  // library.
  // [iso]: http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2011/n3242.pdf
  explicit GameParameter(const char* value, bool is_mandatory = false)
      : is_mandatory_(is_mandatory),
        string_value_(value),
        type_(Type::kString) {}

  explicit GameParameter(bool value, bool is_mandatory = false)
      : is_mandatory_(is_mandatory), bool_value_(value), type_(Type::kBool) {}

  explicit GameParameter(GameParameters value,
                         bool is_mandatory = false)
      : is_mandatory_(is_mandatory),
        game_value_(std::move(value)),
        type_(Type::kGame) {}
//
//  // Copy constructor
//  GameParameter(const GameParameter &other)
//      : is_mandatory_(other.is_mandatory_), int_value_(other.int_value_),
//        double_value_(other.double_value_), string_value_(other.string_value_),
//        bool_value_(other.bool_value_), game_value_(other.game_value_), type_(other.type_) {
//
////    CopyGameParameters(other.game_value_, game_value_);
//  }
//
//  // Move constructor
//  GameParameter(GameParameter &&other) noexcept
//      : is_mandatory_(other.is_mandatory_), int_value_(other.int_value_),
//        double_value_(other.double_value_),
//        string_value_(std::move(other.string_value_)),
//        bool_value_(other.bool_value_),
//        game_value_(std::move(other.game_value_)), type_(other.type_) {}
//
//  // Copy assignment operator
//  GameParameter &operator=(const GameParameter &other) {
//    if (this != &other) {
//      is_mandatory_ = other.is_mandatory_;
//      int_value_ = other.int_value_;
//      double_value_ = other.double_value_;
//      string_value_ = other.string_value_;
//      bool_value_ = other.bool_value_;
//      game_value_ = other.game_value_;
//      type_ = other.type_;
//      // copy the game parameters manually
////      game_value_.clear();
////      CopyGameParameters(other.game_value_, game_value_);
//    }
//    return *this;
//  }
//
//  // Move assignment operator
//  GameParameter &operator=(GameParameter &&other) noexcept {
//    if (this != &other) {
//      is_mandatory_ = other.is_mandatory_;
//      int_value_ = other.int_value_;
//      double_value_ = other.double_value_;
//      string_value_ = std::move(other.string_value_);
//      bool_value_ = other.bool_value_;
//      game_value_ = std::move(other.game_value_);
//      type_ = other.type_;
//    }
//    return *this;
//  }
//  ~GameParameter() = default;

  static GameParameter FromString(std::string_view str);

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
  // type <delimiter> value <delimiter> is_mandatory.
  std::string
  Serialize(std::string_view delimiter = kDefaultInternalDelimiter) const;

  static std::string TypeToString(const GameParameter::Type &type);

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

  const GameParameters& game_value() const {
    SPIEL_CHECK_TRUE(type_ == Type::kGame);
    return game_value_;
  }

  // Access values via param.value<T>().
  // There are explicit specializations of this function that call the
  // ***_value() functions above, however they are defined in game_parameters.cc
  // to avoid compilation problems on some older compilers.
  template <typename T>
  T value() const;

  template <typename T>
  T value_with_default(T default_value) const;

  bool operator==(const GameParameter& rhs) const {
    switch (type_) {
      case Type::kInt:
        return rhs.has_int_value() && int_value_ == rhs.int_value();
      case Type::kDouble:
        return rhs.has_double_value() && double_value_ == rhs.double_value();
      case Type::kString:
        return rhs.has_string_value() && string_value_ == rhs.string_value();
      case Type::kBool:
        return rhs.has_bool_value() && bool_value_ == rhs.bool_value();
      case Type::kGame:
        return rhs.has_game_value() && game_value_ == rhs.game_value();
      case Type::kUnset:
        return rhs.type_ == Type::kUnset;
    }
    std::cerr << "Unrecognized parameter type in operator=="
              << ", returning false." << std::endl;
    return false;
  }
  bool operator!=(const GameParameter& rhs) const { return !(*this == rhs); }

 private:
  bool is_mandatory_;

  // Default initializations are required here. This is because some games mark
  // parameters as not mandatory and also do not specify default values when
  // registering the game type... instead, setting the documented defaults upon
  // game creation (often due to missing information at registration time).
  // This causes a problem when inspecting the game types themselves, even after
  // the game is created via Game::GetType(), which returns the type as it was
  // when it was registered. These initial values are used for those cases.
  int int_value_ = 0;
  double double_value_ = 0.0;
  std::string string_value_ = "";
  bool bool_value_ = false;
  GameParameters game_value_ = {};
  Type type_;
};

template <typename... Args>
std::shared_ptr<const GameParameter> MakeGameParameter(Args &&...args) {
  return std::make_shared<const GameParameter>(std::forward<Args>(args)...);
}

struct from_string_tag {};

// resolve ambiguity with the constructor (string_view, bool=defaulted)
// by tag dispatch (no-op construction)
inline std::shared_ptr<const GameParameter>
MakeGameParameter(std::string_view str, from_string_tag) {
  return std::make_shared<const GameParameter>(GameParameter::FromString(str));
}

// Game Parameters and Game Parameter Serialization/Deserialization form:
// param_name=type/value/is_mandatory|param_name_2=type2/value2/is_mandatory2
// assumes none of the delimeters appears in the string values
std::string SerializeGameParameters(
    const GameParameters& game_params,
    std::string_view name_delimiter = kDefaultNameDelimiter,
    std::string_view parameter_delimeter = kDefaultParameterDelimiter);
GameParameters DeserializeGameParameters(
    std::string_view data,
    std::string_view name_delimiter = kDefaultNameDelimiter,
    std::string_view parameter_delimeter = kDefaultParameterDelimiter);
GameParameter DeserializeGameParameter(
    std::string_view data,
    std::string_view delimiter = kDefaultInternalDelimiter);

inline bool IsParameterSpecified(const GameParameters& table,
                                 const std::string& key) {
  return table.find(key) != table.end();
}

template <typename T>
T ParameterValue(const GameParameters& params, std::string_view key,
                 absl::optional<T> default_value = absl::nullopt) {
  auto iter = params.find(key);
  if (iter == params.end()) {
    if (!default_value.has_value()) {
      SpielFatalError(absl::StrCat("Cannot find parameter and no default "
                                   "value passed for key: ", key));
    }

    return *default_value;
  } else {
    return iter->second->value<T>();
  }
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_PARAMETERS_H_
