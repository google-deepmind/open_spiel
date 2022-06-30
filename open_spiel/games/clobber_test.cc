// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/clobber.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace clobber {
namespace {

namespace testing = open_spiel::testing;

double ValueAt(const std::vector<float>& v, const std::vector<int>& shape,
               int plane, int x, int y) {
  return v[plane * shape[1] * shape[2] + y * shape[2] + x];
}

void BasicClobberTests() {
  testing::LoadGameTest("clobber");
  testing::NoChanceOutcomesTest(*LoadGame("clobber"));

  // Test game simulations on competition boards:
  testing::RandomSimTest(*LoadGame("clobber"), 100);
  testing::RandomSimTest(*LoadGame("clobber(rows=8,columns=8)"), 50);
  testing::RandomSimTest(*LoadGame("clobber(rows=10,columns=10)"), 30);
}

void TerminalReturnsTests() {
  std::shared_ptr<const Game> clobber2x2 =
      LoadGame("clobber(rows=2,columns=2)");
  std::shared_ptr<const Game> clobber4x4 =
      LoadGame("clobber(rows=4,columns=4)");
  std::shared_ptr<const Game> clobber5x6 =
      LoadGame("clobber(rows=5,columns=6)");

  ClobberState end_state1(clobber2x2, 2, 2, "0xxxx");
  SPIEL_CHECK_EQ(end_state1.IsTerminal(), true);
  SPIEL_CHECK_EQ(end_state1.Returns(), (std::vector<double>{-1.0, 1.0}));

  ClobberState end_state2(clobber2x2, 2, 2, "1oooo");
  SPIEL_CHECK_EQ(end_state2.IsTerminal(), true);
  SPIEL_CHECK_EQ(end_state2.Returns(), (std::vector<double>{1.0, -1.0}));

  ClobberState end_state3(clobber2x2, 2, 2, "1x.x.");
  SPIEL_CHECK_EQ(end_state3.IsTerminal(), true);
  SPIEL_CHECK_EQ(end_state3.Returns(), (std::vector<double>{1.0, -1.0}));

  ClobberState end_state4(clobber2x2, 2, 2, "0o..x");
  SPIEL_CHECK_EQ(end_state4.IsTerminal(), true);
  SPIEL_CHECK_EQ(end_state4.Returns(), (std::vector<double>{-1.0, 1.0}));

  ClobberState end_state5(clobber4x4, 4, 4, "0o..xo.......x..o");
  SPIEL_CHECK_EQ(end_state5.IsTerminal(), true);
  SPIEL_CHECK_EQ(end_state5.Returns(), (std::vector<double>{-1.0, 1.0}));

  ClobberState ongoing_state(clobber5x6, 5, 6,
                             "0ox..ox..oxoxox..ox..oxoxoxoxox");
  SPIEL_CHECK_EQ(ongoing_state.IsTerminal(), false);
}

void ObservationTensorTests() {
  std::shared_ptr<const Game> clobber8x8 =
      LoadGame("clobber(rows=8,columns=8)");
  std::unique_ptr<State> clobber_state = clobber8x8->NewInitialState();
  auto shape = clobber8x8->ObservationTensorShape();
  auto v = clobber_state->ObservationTensor(clobber_state->CurrentPlayer());

  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, 4, 4), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, 5, 6), 1.0);

  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, 7, 2), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, 4, 6), 1.0);

  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, 2, 2), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, 1, 6), 0.0);

  std::vector<Action> legal_actions = clobber_state->LegalActions();
  bool action_performed = false;
  for (Action action : legal_actions) {
    if (clobber_state->ActionToString(action) == "a1b1") {
      clobber_state->ApplyAction(action);
      action_performed = true;
      break;
    }
  }

  if (!action_performed) {
    return;
  }

  clobber_state->ObservationTensor(clobber_state->CurrentPlayer(),
                                   absl::MakeSpan(v));

  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, 7, 2), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, 4, 6), 1.0);

  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, 4, 4), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, 5, 6), 1.0);

  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, 2, 2), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, 0, 7), 1.0);
}

}  // namespace
}  // namespace clobber
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::clobber::BasicClobberTests();
  open_spiel::clobber::TerminalReturnsTests();
  open_spiel::clobber::ObservationTensorTests();
}
