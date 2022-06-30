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

#include "open_spiel/spiel_utils.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"


namespace open_spiel {

int NextPlayerRoundRobin(Player player, int nplayers) {
  if (player + 1 < nplayers) {
    return player + 1;
  } else {
    return 0;
  }
}

// Helper function to determine the previous player in a round robin.
int PreviousPlayerRoundRobin(Player player, int nplayers) {
  if (player - 1 >= 0) {
    return player - 1;
  } else {
    return nplayers - 1;
  }
}

// Used to convert actions represented as integers in mixed bases.
Action RankActionMixedBase(const std::vector<int>& bases,
                           const std::vector<int>& digits) {
  SPIEL_CHECK_EQ(bases.size(), digits.size());
  SPIEL_CHECK_GT(digits.size(), 0);

  Action action = 0;
  int one_plus_max = 1;
  for (int i = digits.size() - 1; i >= 0; --i) {
    SPIEL_CHECK_GE(digits[i], 0);
    SPIEL_CHECK_LT(digits[i], bases[i]);
    SPIEL_CHECK_GT(bases[i], 1);
    action += digits[i] * one_plus_max;
    one_plus_max *= bases[i];
    SPIEL_CHECK_LT(action, one_plus_max);
  }

  return action;
}

std::vector<int> UnrankActionMixedBase(Action action,
                                       const std::vector<int>& bases) {
  std::vector<int> digits(bases.size());
  for (int i = digits.size() - 1; i >= 0; --i) {
    SPIEL_CHECK_GT(bases[i], 1);
    digits[i] = action % bases[i];
    action /= bases[i];
  }
  SPIEL_CHECK_EQ(action, 0);
  return digits;
}

absl::optional<std::string> FindFile(const std::string& filename, int levels) {
  std::string candidate_filename = filename;
  for (int i = 0; i <= levels; ++i) {
    if (i == 0) {
      std::ifstream file(candidate_filename.c_str());
      if (file.good()) {
        return candidate_filename;
      }
    } else {
      candidate_filename = "../" + candidate_filename;
      std::ifstream file(candidate_filename.c_str());
      if (file.good()) {
        return candidate_filename;
      }
    }
  }
  return absl::nullopt;
}

std::string FormatDouble(double value) {
  // We cannot use StrCat as that would default to exponential notation
  // sometimes. For example, the default format of 10^-9 is the string
  // "1e-9". For that reason, we use StrFormat with %f explicitly, and add
  // the .0 if necessary (to clarify that it's a double value).
  std::string double_str = absl::StrFormat("%.15f", value);
  size_t idx = double_str.find('.');

  if (double_str.find('.') == std::string::npos) {  // NOLINT
    absl::StrAppend(&double_str, ".0");
  } else {
    // Remove the extra trailing zeros, if there are any.
    while (double_str.length() > idx + 2 && double_str.back() == '0') {
      double_str.pop_back();
    }
  }
  return double_str;
}

void SpielDefaultErrorHandler(const std::string& error_msg) {
  std::cerr << "Spiel Fatal Error: " << error_msg << std::endl
            << std::endl
            << std::flush;
  std::exit(1);
}

ErrorHandler error_handler = SpielDefaultErrorHandler;

void SetErrorHandler(ErrorHandler new_error_handler) {
  error_handler = new_error_handler;
}

void SpielFatalError(const std::string& error_msg) {
  error_handler(error_msg);
  // The error handler should not return. If it does, we will abort the process.
  std::cerr << "Error handler failure - exiting" << std::endl;
  std::exit(1);
}

std::ostream& operator<<(std::ostream& stream, const absl::nullopt_t& v) {
  return stream << "(nullopt)";
}

void Normalize(absl::Span<double> weights) {
  SPIEL_CHECK_FALSE(weights.empty());
  const double normalizer = absl::c_accumulate(weights, 0.);
  SPIEL_CHECK_FALSE(std::isnan(normalizer));
  const double uniform_prob = 1.0 / weights.size();
  absl::c_for_each(weights, [&](double& w) {
    w = (normalizer == 0.0 ? uniform_prob : w / normalizer);
  });
}

std::string BoolToStr(bool b) { return b ? "true" : "false"; }

template <class A, class B>
std::string VectorOfPairsToString(std::vector<std::pair<A, B>>& vec,
                                  const std::string& delimiter,
                                  const std::string& pair_delimiter) {
  std::string str;
  for (int i = 0; i < vec.size(); ++i) {
    absl::StrAppend(&str, vec[i].first, pair_delimiter, vec[i].second);
    if (i != vec.size() - 1) {
      absl::StrAppend(&str, delimiter);
    }
  }
  return str;
}

int SamplerFromRng::operator()(absl::Span<const double> probs) {
  const float cutoff = rng_();
  float sum = 0.0f;
  for (int i = 0; i < probs.size(); ++i) {
    sum += probs[i];
    if (cutoff < sum) {
      return i;
    }
  }

  // To be on the safe side, cover case cutoff == 1.0 and sum < 1
  for (int i = probs.size() - 1; i >= 0; --i) {
    if (probs[i] > 0.0) return i;
  }

  SpielFatalError("SamplerFromRng: not a probability distribution.");
}

}  // namespace open_spiel
