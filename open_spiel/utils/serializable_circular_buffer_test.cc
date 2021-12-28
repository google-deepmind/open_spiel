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

#include "open_spiel/utils/serializable_circular_buffer.h"

#include <nop/structure.h>

#include <random>
#include <string>
#include <vector>
#include <utility>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace {

const char* kSimpleSerializationFilename = "simple_buffer_data.nop";
const char* kComplexSerializationFilename = "complex_buffer_data.nop";

struct TestStruct {
  std::vector<Action> action_vector;
  std::vector<float> float_vector;
  std::vector<std::pair<Action, double>> actions_and_probs;
  double double_value;

  bool operator==(const TestStruct& other_test_struct) const {
    return action_vector == other_test_struct.action_vector &&
           float_vector == other_test_struct.float_vector &&
           actions_and_probs == other_test_struct.actions_and_probs &&
           double_value == other_test_struct.double_value;
  }

  NOP_STRUCTURE(TestStruct,
                action_vector,
                float_vector,
                actions_and_probs,
                double_value);
};

void TestSerializableCircularBuffer() {
  SerializableCircularBuffer<int> buffer(4);
  std::mt19937 rng;
  std::vector<int> sample;

  SPIEL_CHECK_TRUE(buffer.Empty());
  SPIEL_CHECK_EQ(buffer.Size(), 0);

  buffer.Add(13);
  SPIEL_CHECK_FALSE(buffer.Empty());
  SPIEL_CHECK_EQ(buffer.Size(), 1);
  SPIEL_CHECK_EQ(buffer.TotalAdded(), 1);
  SPIEL_CHECK_EQ(buffer[0], 13);

  sample = buffer.Sample(&rng, 1);
  SPIEL_CHECK_EQ(sample.size(), 1);
  SPIEL_CHECK_EQ(sample[0], 13);

  buffer.Add(14);
  buffer.Add(15);
  buffer.Add(16);

  SPIEL_CHECK_EQ(buffer.Size(), 4);
  SPIEL_CHECK_EQ(buffer.TotalAdded(), 4);

  sample = buffer.Sample(&rng, 2);
  SPIEL_CHECK_EQ(sample.size(), 2);
  SPIEL_CHECK_GE(sample[0], 13);
  SPIEL_CHECK_LE(sample[0], 16);
  SPIEL_CHECK_GE(sample[1], 13);
  SPIEL_CHECK_LE(sample[1], 16);

  buffer.Add(17);
  buffer.Add(18);

  SPIEL_CHECK_EQ(buffer.Size(), 4);
  SPIEL_CHECK_EQ(buffer.TotalAdded(), 6);

  sample = buffer.Sample(&rng, 1);
  SPIEL_CHECK_EQ(sample.size(), 1);
  SPIEL_CHECK_GE(sample[0], 15);
  SPIEL_CHECK_LE(sample[0], 18);
}

void TestSimpleSerializableCircularBufferSerialization() {
  SerializableCircularBuffer<int> original_buffer(6);
  original_buffer.Add(1);
  original_buffer.Add(2);
  original_buffer.Add(3);
  original_buffer.Add(4);
  original_buffer.Add(5);
  original_buffer.Add(6);
  original_buffer.SaveBuffer(kSimpleSerializationFilename);

  SerializableCircularBuffer<int> new_buffer(6);
  new_buffer.LoadBuffer(kSimpleSerializationFilename);

  SPIEL_CHECK_EQ(original_buffer.Size(), new_buffer.Size());
  SPIEL_CHECK_EQ(original_buffer.TotalAdded(), new_buffer.TotalAdded());
  SPIEL_CHECK_TRUE(original_buffer.Data() == new_buffer.Data());
}

void TestComplexSerializableCircularBufferSerialization() {
  TestStruct struct1 = {.action_vector = {1, 2, 3},
                        .float_vector = {1.0f, 2.0f, 3.0f},
                        .actions_and_probs = {{1, 1.0}, {2, 2.0}, {3, 3.0}},
                        .double_value = 1.23};
  TestStruct struct2 = {.action_vector = {4, 5, 6},
                        .float_vector = {4.0f, 5.0f, 6.0f},
                        .actions_and_probs = {{4, 4.0}, {5, 5.0}, {6, 6.0}},
                        .double_value = 4.56};
  TestStruct struct3 = {.action_vector = {7, 8, 9},
                        .float_vector = {7.0f, 8.0f, 9.0f},
                        .actions_and_probs = {{7, 7.0}, {8, 8.0}, {9, 9.0}},
                        .double_value = 7.89};

  SerializableCircularBuffer<TestStruct> original_buffer(3);
  original_buffer.Add(struct1);
  original_buffer.Add(struct2);
  original_buffer.Add(struct3);
  original_buffer.SaveBuffer(kComplexSerializationFilename);

  SerializableCircularBuffer<TestStruct> new_buffer(3);
  new_buffer.LoadBuffer(kComplexSerializationFilename);

  SPIEL_CHECK_EQ(original_buffer.Size(), new_buffer.Size());
  SPIEL_CHECK_EQ(original_buffer.TotalAdded(), new_buffer.TotalAdded());
  SPIEL_CHECK_TRUE(original_buffer.Data() == new_buffer.Data());
}

void EndCircularBufferTest() {
  // Remove the files created in the serialization tests.
  file::Remove(kSimpleSerializationFilename);
  file::Remove(kComplexSerializationFilename);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestSerializableCircularBuffer();
  open_spiel::TestSimpleSerializableCircularBufferSerialization();
  open_spiel::TestComplexSerializableCircularBufferSerialization();
  open_spiel::EndCircularBufferTest();
}
