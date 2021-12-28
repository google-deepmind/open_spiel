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

#ifndef OPEN_SPIEL_UTILS_STATS_H_
#define OPEN_SPIEL_UTILS_STATS_H_

#include <algorithm>
#include <cmath>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {

// Track the count, min max, avg and standard deviation.
class BasicStats {
 public:
  BasicStats() { Reset(); }

  // Reset all the stats to 0.
  void Reset() {
    num_ = 0;
    min_ = std::numeric_limits<double>::max();
    max_ = std::numeric_limits<double>::min();
    sum_ = 0;
    sum_sq_ = 0;
  }

  // Merge two BasicStats. Useful for merging per thread stats before printing.
  BasicStats& operator+=(const BasicStats& o) {
    num_ += o.num_;
    sum_ += o.sum_;
    sum_sq_ += o.sum_sq_;
    min_ = std::min(min_, o.min_);
    max_ = std::max(max_, o.max_);
    return *this;
  }

  void Add(double val) {
    min_ = std::min(min_, val);
    max_ = std::max(max_, val);
    sum_ += val;
    sum_sq_ += val * val;
    num_ += 1;
  }

  int64_t Num() const { return num_; }
  double Min() const { return (num_ == 0 ? 0 : min_); }
  double Max() const { return (num_ == 0 ? 0 : max_); }
  double Avg() const { return (num_ == 0 ? 0 : sum_ / num_); }
  double StdDev() const {
    if (num_ <= 1) return 0;
    // Numerical precision can cause variance to be negative, leading to NaN's.
    double variance = (sum_sq_ - sum_ * sum_ / num_) / (num_ - 1);
    return variance <= 0 ? 0 : std::sqrt(variance);
  }

  json::Object ToJson() const {
    return {
        {"num", Num()},
        {"min", Min()},
        {"max", Max()},
        {"avg", Avg()},
        {"std_dev", StdDev()},
    };
  }

 private:
  int64_t num_;
  double min_;
  double max_;
  double sum_;
  double sum_sq_;
};

// Track the occurences for `count` buckets. You need to decide how to map your
// data into the buckets. Mainly useful for scalar values.
class HistogramNumbered {
 public:
  explicit HistogramNumbered(int num_buckets) : counts_(num_buckets, 0) {}
  void Reset() { absl::c_fill(counts_, 0); }
  void Add(int bucket_id) { counts_[bucket_id] += 1; }
  json::Array ToJson() const { return json::CastToArray(counts_); }

 private:
  std::vector<int> counts_;
};

// Same as HistogramNumbered, but each bucket has a name associated with it
// and is returned in the json output. Mainly useful for categorical values.
class HistogramNamed {
 public:
  explicit HistogramNamed(std::vector<std::string> names)
      : counts_(names.size(), 0), names_(names) {}
  void Reset() { absl::c_fill(counts_, 0); }
  void Add(int bucket_id) { counts_[bucket_id] += 1; }
  json::Object ToJson() const {
    return {
        {"counts", json::CastToArray(counts_)},
        {"names", json::CastToArray(names_)},
    };
  }

 private:
  std::vector<int> counts_;
  std::vector<std::string> names_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_STATS_H_
