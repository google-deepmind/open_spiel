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

#ifndef OPEN_SPIEL_SPIEL_UTILS_H_
#define OPEN_SPIEL_SPIEL_UTILS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_real_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"

// Code that is not part of the API, but is widely useful in implementations.

namespace open_spiel {

// Generic ostream operator<< overloads for std:: containers. They have to be
// defined here before call sites because we cannot rely on argument-dependent
// lookup here since that requires putting these overloads into std::, which is
// not allowed (only template specializations on std:: template classes may be
// added to std::, and this is not one of them).

// Make sure that arbitrary structures can be printed out.
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::unique_ptr<T>& v);
template <typename T, typename U>
std::ostream& operator<<(std::ostream& stream, const std::pair<T, U>& v);
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& v);
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& stream, const std::array<T, N>& v);
template <typename T>
std::ostream& operator<<(std::ostream& stream, const absl::optional<T>& v);
std::ostream& operator<<(std::ostream& stream, const absl::nullopt_t& v);
template <typename T>
std::ostream& operator<<(std::ostream& stream, absl::Span<T> v);

// Actual template implementations.
template <typename T>
std::ostream& operator<<(std::ostream& stream, absl::Span<T> v) {
  stream << "[";
  for (const auto& element : v) {
    stream << element << " ";
  }
  stream << "]";
  return stream;
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& v) {
  return stream << absl::MakeSpan(v);
}
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& stream, const std::array<T, N>& v) {
  stream << "[";
  for (const auto& element : v) {
    stream << element << " ";
  }
  stream << "]";
  return stream;
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::unique_ptr<T>& v) {
  return stream << *v;
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, const absl::optional<T>& v) {
  return stream << *v;
}
template <typename T, typename U>
std::ostream& operator<<(std::ostream& stream, const std::pair<T, U>& v) {
  stream << "(" << v.first << "," << v.second << ")";
  return stream;
}

namespace internal {
// SpielStrOut(out, a, b, c) is equivalent to:
//    out << a << b << c;
// It is used to enable SpielStrCat, below.
template <typename Out, typename T>
void SpielStrOut(Out& out, const T& arg) {
  out << arg;
}

template <typename Out, typename T, typename... Args>
void SpielStrOut(Out& out, const T& arg1, Args&&... args) {
  out << arg1;
  SpielStrOut(out, std::forward<Args>(args)...);
}

// Builds a string from pieces:
//
//  SpielStrCat(1, " + ", 1, " = ", 2) --> "1 + 1 = 2"
//
// Converting the parameters to strings is done using the stream operator<<.
// This is only kept around to be used in the SPIEL_CHECK_* macros and should
// not be called by any code outside of this file. Prefer absl::StrCat instead.
// It is kept here due to support for more types, including char.
template <typename... Args>
std::string SpielStrCat(Args&&... args) {
  std::ostringstream out;
  SpielStrOut(out, std::forward<Args>(args)...);
  return out.str();
}

}  // namespace internal

using Player = int;
using Action = int64_t;

// Floating point comparisons use this as a multiplier on the larger of the two
// numbers as the threshold.
inline constexpr float FloatingPointDefaultThresholdRatio() { return 1e-5; }

// Helpers used to convert actions represented as integers in mixed bases.
// E.g. RankActionMixedBase({2, 3, 6}, {1, 1, 1}) = 1*18 + 1*6 + 1 = 25,
// and UnrankActioMixedBase(25, {2, 3, 6}, &digits) sets digits to {1, 1, 1}.
// For the rank, both vectors must be the same size. For the unrank, the digits
// must already have size equal to bases.size().
Action RankActionMixedBase(const std::vector<int>& bases,
                           const std::vector<int>& digits);

std::vector<int> UnrankActionMixedBase(Action action,
                                       const std::vector<int>& bases);

// Helper function to determine the next player in a round robin.
int NextPlayerRoundRobin(Player player, int nplayers);

// Helper function to determine the previous player in a round robin.
int PreviousPlayerRoundRobin(Player player, int nplayers);

// Finds a file by looking up a number of directories. For example: if levels is
// 3 and filename is my.txt, it will look for ./my.txt, ../my.txt, ../../my.txt,
// and ../../../my.txt, return the first file found or absl::nullopt if not
// found.
absl::optional<std::string> FindFile(const std::string& filename, int levels);

// Normalizes the span.
void Normalize(absl::Span<double> weights);

// Format in decimal format, with at most 15 places for the fractional part,
// adding ".0" for integer values, and removing any additional trailing zeroes
// after the first decimal place.
std::string FormatDouble(double value);

// Converts a bool to either "true" or "false".
std::string BoolToStr(bool b);

// Converts a vector of pairs to a string.
template <class A, class B>
std::string VectorOfPairsToString(const std::vector<std::pair<A, B>>& vec,
                                  const std::string& delimiter,
                                  const std::string& pair_delimiter);

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

template <typename T>
bool AllNear(const std::vector<T>& vector1, const std::vector<T>& vector2,
             T epsilon) {
  if (vector1.size() != vector2.size()) {
    return false;
  }
  for (int i = 0; i < vector1.size(); ++i) {
    if (!Near(vector1[i], vector2[i], epsilon)) {
      return false;
    }
  }
  return true;
}

// Macros to check for error conditions.
// These trigger SpielFatalError if the condition is violated.
// These macros are always executed. If you want to use checks
// only for debugging, use SPIEL_DCHECK_*

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
#define SPIEL_CHECK_PROB(x) \
  SPIEL_CHECK_GE(x, 0);     \
  SPIEL_CHECK_LE(x, 1);     \
  SPIEL_CHECK_FALSE(std::isnan(x) || std::isinf(x))

// Checks that x and y are equal to the default dynamic threshold proportional
// to max(|x|, |y|).
#define SPIEL_CHECK_FLOAT_EQ(x, y) \
  SPIEL_CHECK_FN2(static_cast<float>(x), static_cast<float>(y), \
                  open_spiel::Near)

// Checks that x and y are epsilon apart or closer.
#define SPIEL_CHECK_FLOAT_NEAR(x, y, epsilon)                   \
  SPIEL_CHECK_FN3(static_cast<float>(x), static_cast<float>(y), \
                  static_cast<float>(epsilon), open_spiel::Near)

#define SPIEL_CHECK_TRUE(x)                                      \
  while (!(x))                                                   \
  open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
      __FILE__, ":", __LINE__, " CHECK_TRUE(", #x, ")"))

#define SPIEL_CHECK_FALSE(x)                                     \
  while (x)                                                      \
  open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
      __FILE__, ":", __LINE__, " CHECK_FALSE(", #x, ")"))

#if !defined(NDEBUG)

// Checks that are executed in Debug / Testing build type,
// and turned off for Release build type.
#define SPIEL_DCHECK_OP(x_exp, op, y_exp) SPIEL_CHECK_OP(x_exp, op, y_exp)
#define SPIEL_DCHECK_FN2(x_exp, y_exp, fn) SPIEL_CHECK_FN2(x_exp, y_exp, fn)
#define SPIEL_DCHECK_FN3(x_exp, y_exp, z_exp, fn) \
  SPIEL_CHECK_FN3(x_exp, y_exp, z_exp, fn)
#define SPIEL_DCHECK_GE(x, y) SPIEL_CHECK_GE(x, y)
#define SPIEL_DCHECK_GT(x, y) SPIEL_CHECK_GT(x, y)
#define SPIEL_DCHECK_LE(x, y) SPIEL_CHECK_LE(x, y)
#define SPIEL_DCHECK_LT(x, y) SPIEL_CHECK_LT(x, y)
#define SPIEL_DCHECK_EQ(x, y) SPIEL_CHECK_EQ(x, y)
#define SPIEL_DCHECK_NE(x, y) SPIEL_CHECK_NE(x, y)
#define SPIEL_DCHECK_PROB(x) SPIEL_DCHECK_PROB(x)
#define SPIEL_DCHECK_FLOAT_EQ(x, y) SPIEL_CHECK_FLOAT_EQ(x, y)
#define SPIEL_DCHECK_FLOAT_NEAR(x, y, epsilon) \
  SPIEL_CHECK_FLOAT_NEAR(x, y, epsilon)
#define SPIEL_DCHECK_TRUE(x) SPIEL_CHECK_TRUE(x)
#define SPIEL_DCHECK_FALSE(x) SPIEL_CHECK_FALSE(x)

#else  // defined(NDEBUG)

// Turn off checks for the (optimized) Release build type.
#define SPIEL_DCHECK_OP(x_exp, op, y_exp)
#define SPIEL_DCHECK_FN2(x_exp, y_exp, fn)
#define SPIEL_DCHECK_FN3(x_exp, y_exp, z_exp, fn)
#define SPIEL_DCHECK_GE(x, y)
#define SPIEL_DCHECK_GT(x, y)
#define SPIEL_DCHECK_LE(x, y)
#define SPIEL_DCHECK_LT(x, y)
#define SPIEL_DCHECK_EQ(x, y)
#define SPIEL_DCHECK_NE(x, y)
#define SPIEL_DCHECK_PROB(x)
#define SPIEL_DCHECK_FLOAT_EQ(x, y)
#define SPIEL_DCHECK_FLOAT_NEAR(x, y, epsilon)
#define SPIEL_DCHECK_TRUE(x)
#define SPIEL_DCHECK_FALSE(x)

#endif  // !defined(NDEBUG)

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

// A ProbabilitySampler that samples uniformly from a distribution.
class UniformProbabilitySampler {
 public:
  UniformProbabilitySampler(int seed, double min = 0., double max = 1.)
      : seed_(seed), rng_(seed_), dist_(min, max), min_(min), max_(max) {}

  UniformProbabilitySampler(double min = 0., double max = 1.)
      : rng_(seed_), dist_(min, max), min_(min), max_(max) {}

  // When copying, we reinitialize the sampler to have the initial seed.
  UniformProbabilitySampler(const UniformProbabilitySampler& other)
      : seed_(other.seed_),
        rng_(other.seed_),
        dist_(other.min_, other.max_),
        min_(other.min_),
        max_(other.max_) {}

  double operator()() { return dist_(rng_); }

 private:
  // Set the seed as the number of nanoseconds
  const int seed_ = absl::ToInt64Nanoseconds(absl::Now() - absl::UnixEpoch());
  std::mt19937 rng_;
  absl::uniform_real_distribution<double> dist_;

  const double min_;
  const double max_;
};

// Utility functions intended to be used for casting
// from a Base class to a Derived subclass.
// These functions handle various use cases, such as pointers and const
// references. For shared or unique pointers you can get the underlying pointer.
// When you use debug mode, a more expensive dynamic_cast is used and it checks
// whether the casting has been successful. In optimized builds only static_cast
// is used when possible.

// use like this: down_cast<T*>(foo);
template <typename To, typename From>
inline To down_cast(From* f) {
#if !defined(NDEBUG)
  if (f != nullptr && dynamic_cast<To>(f) == nullptr) {
    std::string from = typeid(From).name();
    std::string to = typeid(From).name();
    SpielFatalError(
        absl::StrCat("Cast failure: could not cast a pointer from '", from,
                     "' to '", to, "'"));
  }
#endif
  return static_cast<To>(f);
}

// use like this: down_cast<T&>(foo);
template <typename To, typename From>
inline To down_cast(From& f) {
  typedef typename std::remove_reference<To>::type* ToAsPointer;
#if !defined(NDEBUG)
  if (dynamic_cast<ToAsPointer>(&f) == nullptr) {
    std::string from = typeid(From).name();
    std::string to = typeid(From).name();
    SpielFatalError(
        absl::StrCat("Cast failure: could not cast a reference from '", from,
                     "' to '", to, "'"));
  }
#endif
  return *static_cast<ToAsPointer>(&f);
}

// Creates a sampler from a std::function<double()> conforming to the
// probabilities received. absl::discrete_distribution requires a URBG as a
// source of randomness (as opposed to a std::function<double()>) so cannot
// be used directly.
class SamplerFromRng {
 public:
  explicit SamplerFromRng(std::function<double()> rng) : rng_(std::move(rng)) {}

  int operator()(absl::Span<const double> probs);

 private:
  std::function<double()> rng_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_SPIEL_UTILS_H_
