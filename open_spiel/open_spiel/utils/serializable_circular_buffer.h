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

#ifndef OPEN_SPIEL_UTILS_SERIALIZABLE_CIRCULAR_BUFFER_H_
#define OPEN_SPIEL_UTILS_SERIALIZABLE_CIRCULAR_BUFFER_H_

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
#include "open_spiel/utils/circular_buffer.h"

namespace open_spiel {

// A serializable circular buffer of fixed size.
template <class T>
class SerializableCircularBuffer : public CircularBuffer<T> {
 public:
  explicit SerializableCircularBuffer(int max_size)
      : CircularBuffer<T>(max_size) {}

  // Serialize the data of the buffer to a file.
  void SaveBuffer(const std::string& path) const {
    nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{path};
    serializer.Write(this->max_size_);
    serializer.Write(this->total_added_);
    serializer.Write(this->data_);
  }

  // Populate the buffer with data from a saved buffer's file.
  void LoadBuffer(const std::string& path) {
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};

    // Ensure this buffer's max size equals the max size of the saved buffer.
    int max_size;
    deserializer.Read(&max_size);
    if (max_size != this->max_size_) {
      SpielFatalError(absl::StrFormat("Cannot load data from a buffer with max"
                                      "size %d into a buffer with max size %d.",
                                      max_size,
                                      this->max_size_));
    }

    deserializer.Read(&(this->total_added_));
    deserializer.Read(&(this->data_));
  }
};
}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_SERIALIZABLE_CIRCULAR_BUFFER_H_
