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

#include "open_spiel/utils/tensor_view.h"

#include <array>
#include <vector>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestTensorView() {
  std::vector<float> values;

  values.resize(6);
  TensorView<2> view2(absl::MakeSpan(values), {2, 3}, true);
  SPIEL_CHECK_EQ(view2.size(), 6);
  SPIEL_CHECK_EQ(values.size(), 6);
  SPIEL_CHECK_EQ(view2.rank(), 2);
  SPIEL_CHECK_EQ(view2.shape(), (std::array<int, 2>{2, 3}));
  SPIEL_CHECK_EQ(view2.shape(0), 2);
  SPIEL_CHECK_EQ(view2.shape(1), 3);

  // All 0 initialized
  for (int i = 0; i < values.size(); ++i) {
    SPIEL_CHECK_EQ(values[i], 0);
    values[i] = i + 1;
  }

  // Index correctly
  for (int a = 0, i = 0; a < view2.shape(0); ++a) {
    for (int b = 0; b < view2.shape(1); ++b, ++i) {
      SPIEL_CHECK_EQ(view2.index({a, b}), i);
      SPIEL_CHECK_EQ((view2[{a, b}]), i + 1);
      view2[{a, b}] = -i;
    }
  }

  // Index correctly
  for (int i = 0; i < values.size(); ++i) {
    SPIEL_CHECK_EQ(values[i], -i);
  }

  // Clear works
  view2.clear();

  for (int i = 0; i < values.size(); ++i) {
    SPIEL_CHECK_EQ(values[i], 0);
    values[i] = i + 1;
  }

  // Works for more dimensions
  values.resize(24);
  TensorView<3> view3(absl::MakeSpan(values), {4, 2, 3}, true);
  SPIEL_CHECK_EQ(view3.size(), 24);
  SPIEL_CHECK_EQ(values.size(), 24);
  SPIEL_CHECK_EQ(view3.rank(), 3);
  SPIEL_CHECK_EQ(view3.shape(), (std::array<int, 3>{4, 2, 3}));
  SPIEL_CHECK_EQ(view3.shape(0), 4);
  SPIEL_CHECK_EQ(view3.shape(1), 2);
  SPIEL_CHECK_EQ(view3.shape(2), 3);

  // All 0 initialized
  for (int i = 0; i < values.size(); ++i) {
    SPIEL_CHECK_EQ(values[i], 0);
    values[i] = i + 1;
  }

  // Index correctly
  for (int a = 0, i = 0; a < view3.shape(0); ++a) {
    for (int b = 0; b < view3.shape(1); ++b) {
      for (int c = 0; c < view3.shape(2); ++c, ++i) {
        SPIEL_CHECK_EQ(view3.index({a, b, c}), i);
        SPIEL_CHECK_EQ((view3[{a, b, c}]), i + 1);
        view3[{a, b, c}] = -i;
      }
    }
  }

  // Index correctly
  for (int i = 0; i < values.size(); ++i) {
    SPIEL_CHECK_EQ(values[i], -i);
  }

  // Works for a single dimension
  values.resize(8);
  TensorView<1> view1(absl::MakeSpan(values), {8}, true);
  SPIEL_CHECK_EQ(view1.size(), 8);
  SPIEL_CHECK_EQ(values.size(), 8);
  SPIEL_CHECK_EQ(view1.rank(), 1);
  SPIEL_CHECK_EQ(view1.shape(), (std::array<int, 1>{8}));
  SPIEL_CHECK_EQ(view1.shape(0), 8);

  // All 0 initialized
  for (int i = 0; i < values.size(); ++i) {
    SPIEL_CHECK_EQ(values[i], 0);
    values[i] = i + 1;
  }

  // Index correctly
  for (int a = 0; a < view1.shape(0); ++a) {
    SPIEL_CHECK_EQ(view1.index({a}), a);
    SPIEL_CHECK_EQ(view1[{a}], a + 1);
    view1[{a}] = -a;
  }

  // Keeps the previous values.
  TensorView<2> view_keep(absl::MakeSpan(values), {2, 4}, false);
  SPIEL_CHECK_EQ(view_keep.size(), 8);
  SPIEL_CHECK_EQ(values.size(), 8);
  SPIEL_CHECK_EQ(view_keep.rank(), 2);
  SPIEL_CHECK_EQ(view_keep.shape(), (std::array<int, 2>{2, 4}));
  SPIEL_CHECK_EQ(view_keep.shape(0), 2);
  SPIEL_CHECK_EQ(view_keep.shape(1), 4);

  // Index correctly
  for (int a = 0, i = 0; a < view_keep.shape(0); ++a) {
    for (int b = 0; b < view_keep.shape(1); ++b, ++i) {
      SPIEL_CHECK_EQ(view_keep.index({a, b}), i);
      SPIEL_CHECK_EQ((view_keep[{a, b}]), -i);
    }
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::TestTensorView(); }
