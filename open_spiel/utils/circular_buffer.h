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

#ifndef OPEN_SPIEL_UTILS_CIRCULAR_BUFFER_H_
#define OPEN_SPIEL_UTILS_CIRCULAR_BUFFER_H_

#include <nop/serializer.h>
#include <nop/utility/stream_reader.h>
#include <nop/utility/stream_writer.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// A simple circular buffer of fixed size.
template <class T>
class CircularBuffer {
 public:
  explicit CircularBuffer(int max_size)
      : max_size_(max_size), total_added_(0) {}

  // Add one element, replacing the oldest once it's full.
  void Add(const T& value) {
    if (data_.size() < max_size_) {
      data_.push_back(value);
    } else {
      data_[total_added_ % max_size_] = value;
    }
    total_added_ += 1;
  }

  // Return `num` elements without replacement.
  std::vector<T> Sample(std::mt19937* rng, int num) {
    std::vector<T> out;
    out.reserve(num);
    std::sample(data_.begin(), data_.end(), std::back_inserter(out), num, *rng);
    return out;
  }

  // Return the full buffer.
  const std::vector<T>& Data() const { return data_; }

  // Access a single element from the buffer.
  const T& operator[](int i) const { return data_[i]; }

  // How many elements are in the buffer.
  int Size() const { return data_.size(); }

  // Is the buffer empty?
  bool Empty() const { return data_.empty(); }

  // How many elements have ever been added to the buffer.
  int64_t TotalAdded() const { return total_added_; }

  // Serialize the data of the buffer to a file.
  void SaveBuffer(const std::string& path) const {
    nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{path};
    serializer.Write(max_size_);
    serializer.Write(total_added_);
    serializer.Write(data_);
  }

  // Populate the buffer with data from a saved buffer's file.
  void LoadBuffer(const std::string& path) {
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};

    // Ensure this buffer's max size equals the max size of the saved buffer.
    int max_size;
    deserializer.Read(&max_size);
    if (max_size != max_size_) {
      SpielFatalError(absl::StrFormat("Cannot load data from a buffer with max"
                                      "size %d into a buffer with max size %d.",
                                      max_size,
                                      max_size_));
    }

    deserializer.Read(&total_added_);
    deserializer.Read(&data_);
  }

 private:
  const int max_size_;
  int64_t total_added_;
  std::vector<T> data_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_CIRCULAR_BUFFER_H_
