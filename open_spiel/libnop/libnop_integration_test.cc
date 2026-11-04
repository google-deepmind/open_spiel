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

#include <nop/serializer.h>
#include <nop/structure.h>
#include <nop/utility/stream_writer.h>

#include <array>
#include <iostream>
#include <sstream>
#include <string>

namespace {

// Libnop example taken from
// https://github.com/google/libnop/blob/master/README.md

// Contrived template type with private members.
template <typename T>
struct UserDefined {
 public:
  UserDefined() = default;
  UserDefined(std::string label, std::vector<T> vector)
      : label_{std::move(label)}, vector_{std::move(vector)} {}

  const std::string label() const { return label_; }
  const std::vector<T>& vector() const { return vector_; }

 private:
  std::string label_;
  std::vector<T> vector_;

  NOP_STRUCTURE(UserDefined, label_, vector_);
};

void TestSerialization() {
  using Writer = nop::StreamWriter<std::stringstream>;
  nop::Serializer<Writer> serializer;

  serializer.Write(UserDefined<int>{"ABC", {1, 2, 3, 4, 5}});

  using ArrayType = std::array<UserDefined<float>, 2>;
  serializer.Write(
      ArrayType{{{"ABC", {1, 2, 3, 4, 5}}, {"XYZ", {3.14, 2.72, 23.14}}}});

  const std::string data = serializer.writer().stream().str();
  std::cout << "Wrote " << data.size() << " bytes." << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  TestSerialization();
}
