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

#ifndef THIRD_PARTY_OPEN_SPIEL_SPIEL_UTILS_H_
#define THIRD_PARTY_OPEN_SPIEL_SPIEL_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"

// Code that is not part of the API, but is widely useful in implementations

namespace open_spiel {

// Generic ostream operator<< overloads for std:: containers. They have to be
// defined here before call sites because we cannot rely on argument-dependent
// lookup here since that requires putting these overloads into std::, which is
// not allowed (only template specializations on std:: template classes may be
// added to std::, and this is not one of them).
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& v) {
  stream << "[";
  for (const auto& element : v) {
    stream << element << " ";
  }
  stream << "]";
  return stream;
}

// SpielStrOut(out, a, b, c) is equivalent to:
//    out << a << b << c;
// It is useful mostly to enable absl::StrAppend and absl::StrCat, below.
template <typename Out, typename T>
void SpielStrOut(Out& out, const T& arg) {
  out << arg;
}

template <typename Out, typename T, typename... Args>
void SpielStrOut(Out& out, const T& arg1, Args&&... args) {
  out << arg1;
  SpielStrOut(out, std::forward<Args>(args)...);
}

namespace internal {
// Builds a string from pieces:
//
//  SpielStrCat(1, " + ", 1, " = ", 2) --> "1 + 1 = 2"
//
// Converting the parameters to strings is done using the stream operator<<.
// This is only kept around to be used in the SPIEL_CHECK_* macros and should
// not be called by any code outside of this file.
template <typename... Args>
std::string SpielStrCat(Args&&... args) {
  std::ostringstream out;
  SpielStrOut(out, std::forward<Args>(args)...);
  return out.str();
}

}  // namespace internal

using Action = int64_t;

// Floating point comparisons use this as a multiplier on the larger of the two
// numbers as the threshold.
constexpr float FloatingPointDefaultThresholdRatio() { return 1e-5; }

// Useful functions for parsing the command-line for arguments of the form
// --name=value.

// Returns (true, value) if command-line argument is found, or (false, "")
// otherwise.
std::pair<bool, std::string> ParseCmdLineArg(int argc, char** argv,
                                             const std::string& name);

// Returns the value of the command-line argument if found, otherwise returns
// the default value.
std::string ParseCmdLineArgDefault(int argc, char** argv,
                                   const std::string& name,
                                   const std::string& default_value);

// Helpers used to convert actions represented as integers in mixed bases.
// E.g. RankActionMixedBase({2, 3, 6}, {1, 1, 1}) = 1*18 + 1*6 + 1 = 25,
// and UnrankActioMixedBase(25, {2, 3, 6}, &digits) sets digits to {1, 1, 1}.
// For the rank, both vectors must be the same size. For the unrank, the digits
// must already have size equal to bases.size().
Action RankActionMixedBase(const std::vector<int>& bases,
                           const std::vector<int>& digits);

void UnrankActionMixedBase(Action action, const std::vector<int>& bases,
                           std::vector<int>* digits);

// Helper function to determine the next player in a round robin.
int NextPlayerRoundRobin(int player, int nplayers);

// Helper function to determine the previous player in a round robin.
int PreviousPlayerRoundRobin(int player, int nplayers);

// Returns whether the absolute difference between floating point values a and
// b is less than or equal to FloatingPointThresholdRatio() * max(|a|, |b|).
template <typename T>
bool Near(T a, T b) {
  static_assert(std::is_floating_point<T>::value,
                "Near() is only for floating point args.");
  return fabs(a - b) <=
         (std::max(fabs(a), fabs(b)) * FloatingPointDefaultThresholdRatio());
}

// Returns whether |a - b| <= epsilon.
template <typename T>
bool Near(T a, T b, T epsilon) {
  static_assert(std::is_floating_point<T>::value,
                "Near() is only for floating point args.");
  return fabs(a - b) <= epsilon;
}

// Macros to check for error conditions.
// These trigger SpielFatalError if the condition is violated.

#define SPIEL_CHECK_OP(x_exp, op, y_exp)                             \
  do {                                                               \
    auto x = x_exp;                                                  \
    auto y = y_exp;                                                  \
    if (!((x)op(y)))                                                 \
      open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
          __FILE__, ":", __LINE__, " ", #x_exp " " #op " " #y_exp,   \
          "\n" #x_exp, " = ", x, ", " #y_exp " = ", y));             \
  } while (false)

#define SPIEL_CHECK_FN2(x_exp, y_exp, fn)                                 \
  do {                                                                    \
    auto x = x_exp;                                                       \
    auto y = y_exp;                                                       \
    if (!fn(x, y))                                                        \
      open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat(      \
          __FILE__, ":", __LINE__, " ", #fn "(" #x_exp ", " #y_exp ")\n", \
          #x_exp " = ", x, ", " #y_exp " = ", y));                        \
  } while (false)

#define SPIEL_CHECK_FN3(x_exp, y_exp, z_exp, fn)                         \
  do {                                                                   \
    auto x = x_exp;                                                      \
    auto y = y_exp;                                                      \
    auto z = z_exp;                                                      \
    if (!fn(x, y, z))                                                    \
      open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat(     \
          __FILE__, ":", __LINE__, " ",                                  \
          #fn "(" #x_exp ", " #y_exp ", " #z_exp ")\n", #x_exp " = ", x, \
          ", " #y_exp " = ", y, ", " #z_exp " = ", z));                  \
  } while (false)

#define SPIEL_CHECK_GE(x, y) SPIEL_CHECK_OP(x, >=, y)
#define SPIEL_CHECK_GT(x, y) SPIEL_CHECK_OP(x, >, y)
#define SPIEL_CHECK_LE(x, y) SPIEL_CHECK_OP(x, <=, y)
#define SPIEL_CHECK_LT(x, y) SPIEL_CHECK_OP(x, <, y)
#define SPIEL_CHECK_EQ(x, y) SPIEL_CHECK_OP(x, ==, y)
#define SPIEL_CHECK_NE(x, y) SPIEL_CHECK_OP(x, !=, y)

// Checks that x and y are equal to the default dynamic threshold proportional
// to max(|x|, |y|).
#define SPIEL_CHECK_FLOAT_EQ(x, y) SPIEL_CHECK_FN2(x, y, Near)

// Checks that x and y are epsilon apart or closer.
#define SPIEL_CHECK_FLOAT_NEAR(x, y, epsilon) \
  SPIEL_CHECK_FN3(x, y, epsilon, Near)

#define SPIEL_CHECK_TRUE(x)                                      \
  while (!(x))                                                   \
  open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
      __FILE__, ":", __LINE__, " CHECK_TRUE(", #x, ")"))

#define SPIEL_CHECK_FALSE(x)                                     \
  while (x)                                                      \
  open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
      __FILE__, ":", __LINE__, " CHECK_FALSE(", #x, ")"))

// When an error is encountered, OpenSpiel code should call SpielFatalError()
// which will forward the message to the current error handler.
// The default error handler outputs the error message to stderr, and exits
// the process with exit code 1.

// When called from Python, a different error handled is used, which returns
// RuntimeException to the caller, containing the error message.

// Report a runtime error.
[[noreturn]] void SpielFatalError(const std::string& error_msg);

// Specify a new error handler.
using ErrorHandler = void (*)(const std::string&);
void SetErrorHandler(ErrorHandler error_handler);

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_SPIEL_UTILS_H_
