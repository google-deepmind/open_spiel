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

#include "open_spiel/utils/functional.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void TestZip() {
  std::vector<Action> actions = {1, 2, 3};
  std::vector<double> probs = {0.1, 0.2, 0.3};
  std::vector<std::pair<Action, double>> action_probs;
  Zip(actions.begin(), actions.end(), probs.begin(), action_probs);

  std::vector<std::pair<Action, double>> expected_action_probs = {
    {1, 0.1}, {2, 0.2}, {3, 0.3}};
  SPIEL_CHECK_EQ(action_probs, expected_action_probs);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestZip();
}
