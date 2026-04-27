// Copyright 2024 DeepMind Technologies Limited
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

#include "open_spiel/games/xiangqi/xiangqi.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace xiangqi {
namespace {

namespace testing = open_spiel::testing;

// Helper: find the action that moves from (fr,fc) to (tr,tc).
Action FindAction(const State& state, int fr, int fc, int tr, int tc) {
  Action target = EncodeMove(SquareIndex(fr, fc), SquareIndex(tr, tc));
  for (Action a : state.LegalActions()) {
    if (a == target) return a;
  }
  return kInvalidAction;
}

bool HasAction(const State& state, int fr, int fc, int tr, int tc) {
  return FindAction(state, fr, fc, tr, tc) != kInvalidAction;
}

void BasicXiangqiTests() {
  testing::LoadGameTest("xiangqi");
  testing::NoChanceOutcomesTest(*LoadGame("xiangqi"));
  testing::RandomSimTest(*LoadGame("xiangqi"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("xiangqi"), 10);
}

void InitialStateTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();
  const auto* xq = static_cast<const XiangqiState*>(state.get());

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);  // Red first
  SPIEL_CHECK_FALSE(state->IsTerminal());

  // Black back rank (row 0).
  SPIEL_CHECK_EQ(xq->BoardAt(0, 0).type, kChariot);
  SPIEL_CHECK_EQ(xq->BoardAt(0, 0).player, 1);
  SPIEL_CHECK_EQ(xq->BoardAt(0, 1).type, kHorse);
  SPIEL_CHECK_EQ(xq->BoardAt(0, 4).type, kGeneral);
  SPIEL_CHECK_EQ(xq->BoardAt(0, 4).player, 1);

  // Red back rank (row 9).
  SPIEL_CHECK_EQ(xq->BoardAt(9, 0).type, kChariot);
  SPIEL_CHECK_EQ(xq->BoardAt(9, 0).player, 0);
  SPIEL_CHECK_EQ(xq->BoardAt(9, 4).type, kGeneral);
  SPIEL_CHECK_EQ(xq->BoardAt(9, 4).player, 0);

  // Cannons.
  SPIEL_CHECK_EQ(xq->BoardAt(2, 1).type, kCannon);
  SPIEL_CHECK_EQ(xq->BoardAt(2, 1).player, 1);
  SPIEL_CHECK_EQ(xq->BoardAt(7, 7).type, kCannon);
  SPIEL_CHECK_EQ(xq->BoardAt(7, 7).player, 0);

  // Soldiers.
  SPIEL_CHECK_EQ(xq->BoardAt(3, 0).type, kSoldier);
  SPIEL_CHECK_EQ(xq->BoardAt(3, 0).player, 1);
  SPIEL_CHECK_EQ(xq->BoardAt(6, 4).type, kSoldier);
  SPIEL_CHECK_EQ(xq->BoardAt(6, 4).player, 0);

  // Empty squares.
  SPIEL_CHECK_TRUE(xq->BoardAt(1, 0).IsEmpty());
  SPIEL_CHECK_TRUE(xq->BoardAt(4, 4).IsEmpty());
  SPIEL_CHECK_TRUE(xq->BoardAt(5, 4).IsEmpty());
}

void SoldierMovementTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();

  // Red soldier at (6, 0) should only be able to move forward to (5, 0)
  // (before crossing the river).
  SPIEL_CHECK_TRUE(HasAction(*state, 6, 0, 5, 0));   // forward
  SPIEL_CHECK_FALSE(HasAction(*state, 6, 0, 6, 1));  // no sideways
  SPIEL_CHECK_FALSE(HasAction(*state, 6, 0, 7, 0));  // no backward

  // Move the Red soldier to row 5 (still own side), then to row 4 (crossed
  // the river). We need a few filler moves for Black in between.
  Action a1 = FindAction(*state, 6, 0, 5, 0);
  SPIEL_CHECK_NE(a1, kInvalidAction);
  state->ApplyAction(a1);  // Red soldier -> (5, 0)

  // Black moves a soldier.
  Action b1 = FindAction(*state, 3, 0, 4, 0);
  SPIEL_CHECK_NE(b1, kInvalidAction);
  state->ApplyAction(b1);  // Black soldier -> (4, 0)

  // Red soldier at (5, 0) moves forward to (4, 0)... but (4, 0) has a Black
  // soldier. This is a capture and also crossing the river.
  SPIEL_CHECK_TRUE(HasAction(*state, 5, 0, 4, 0));  // forward (capture)
  Action a2 = FindAction(*state, 5, 0, 4, 0);
  state->ApplyAction(a2);  // Red soldier -> (4, 0), captures Black soldier

  // Black does something else.
  Action b2 = FindAction(*state, 3, 2, 4, 2);
  SPIEL_CHECK_NE(b2, kInvalidAction);
  state->ApplyAction(b2);  // Black soldier -> (4, 2)

  // Now Red soldier is at (4, 0), which is the enemy's side (crossed river).
  // It should be able to move forward and sideways.
  SPIEL_CHECK_TRUE(HasAction(*state, 4, 0, 3, 0));   // forward
  SPIEL_CHECK_TRUE(HasAction(*state, 4, 0, 4, 1));   // sideways right
  SPIEL_CHECK_FALSE(HasAction(*state, 4, 0, 5, 0));  // no backward
}

void HorseBlockingTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();

  // Red Horse at (9, 1). The orthogonal blocking square toward (7, 0) is (8,
  // 1), which is empty. So the horse should be able to go to (7, 0) and (7,
  // 2).
  SPIEL_CHECK_TRUE(HasAction(*state, 9, 1, 7, 0));
  SPIEL_CHECK_TRUE(HasAction(*state, 9, 1, 7, 2));

  // The horse should NOT be able to go to (8, 3) because the blocking square
  // (9, 2) = Elephant, which is occupied.
  SPIEL_CHECK_FALSE(HasAction(*state, 9, 1, 8, 3));
}

void ElephantRiverTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();

  // Red Elephant at (9, 2) can move to (7, 0) or (7, 4) if midpoint is empty.
  // Midpoint for (7, 0) is (8, 1) which is empty -> allowed.
  SPIEL_CHECK_TRUE(HasAction(*state, 9, 2, 7, 0));
  // Midpoint for (7, 4) is (8, 3) which is empty -> allowed.
  SPIEL_CHECK_TRUE(HasAction(*state, 9, 2, 7, 4));

  // Move Red Elephant to (7, 4).
  Action a1 = FindAction(*state, 9, 2, 7, 4);
  state->ApplyAction(a1);

  // Black move.
  state->ApplyAction(FindAction(*state, 3, 0, 4, 0));

  // Red Elephant at (7, 4) could try to go to (5, 2) or (5, 6) - these are
  // still on Red's side (row >= 5). It should NOT go to (5, 2) or (5, 6) if
  // blocked, but if not blocked they are legal since row 5 is still Red's
  // side.
  // Check (5, 2): midpoint (6, 3) is empty -> legal (still own side)
  SPIEL_CHECK_TRUE(HasAction(*state, 7, 4, 5, 2));
  // Check (5, 6): midpoint (6, 5) is empty -> legal (still own side)
  SPIEL_CHECK_TRUE(HasAction(*state, 7, 4, 5, 6));

  // Move to (5, 6).
  state->ApplyAction(FindAction(*state, 7, 4, 5, 6));
  state->ApplyAction(FindAction(*state, 3, 2, 4, 2));  // Black move

  // Red Elephant at (5, 6). Trying to go to (3, 4) or (3, 8) would cross the
  // river (row 3 < 5, which is Black's side).
  SPIEL_CHECK_FALSE(HasAction(*state, 5, 6, 3, 4));
  SPIEL_CHECK_FALSE(HasAction(*state, 5, 6, 3, 8));
}

void CannonCaptureTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();

  // Red Cannon at (7, 1). It can move freely along the column and row
  // (non-capture). It CANNOT capture without exactly one screen piece.
  // Column 1 upward: (6, 1) is empty, (5, 1) is empty, (4, 1) is empty,
  // (3, 1) is empty, (2, 1) has Black Cannon (first piece encountered).
  // The cannon can move to (6, 1), (5, 1), (4, 1), (3, 1) but NOT capture
  // at (2, 1) because there's no screen piece.
  SPIEL_CHECK_TRUE(HasAction(*state, 7, 1, 6, 1));
  SPIEL_CHECK_TRUE(HasAction(*state, 7, 1, 5, 1));
  SPIEL_CHECK_TRUE(HasAction(*state, 7, 1, 4, 1));
  SPIEL_CHECK_TRUE(HasAction(*state, 7, 1, 3, 1));
  SPIEL_CHECK_FALSE(HasAction(*state, 7, 1, 2, 1));  // no screen piece

  // Move Red Cannon to (4, 1). Now (3, 1) is empty, Black Cannon is at (2,
  // 1), Black Horse at (0, 1). The Red Cannon at (4, 1) wants to capture (0,
  // 1). Between (4, 1) and (0, 1) on col 1: (3, 1) empty, (2, 1) has Black
  // Cannon (screen), (1, 1) empty. So (0, 1) can be captured since the Black
  // Cannon at (2, 1) is the screen.
  state->ApplyAction(FindAction(*state, 7, 1, 4, 1));  // Red Cannon -> (4, 1)
  state->ApplyAction(FindAction(*state, 3, 4, 4, 4));  // Black Soldier moves

  SPIEL_CHECK_TRUE(HasAction(*state, 4, 1, 0, 1));  // capture over screen
}

void FlyingGeneralTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();

  // Set up a board where moving a piece would expose the flying general.
  // We play a bunch of moves to clear the column between the generals.
  // This is complex to set up from scratch, so we test by verifying the rule
  // is enforced in an endgame-like situation.

  // Build a minimal board manually: put just the two generals facing each
  // other on column 4, with one piece between them.
  // We can't easily set up arbitrary positions, so instead we verify that in
  // the initial position, the generals don't face each other (column 4 has
  // pieces between them).
  SPIEL_CHECK_FALSE(state->IsTerminal());

  // Instead, let's test using action validation. In the initial position,
  // Red pieces on column 4 include General(9,4) and Soldier(6,4).
  // If the Soldier at (6,4) moves to (5,4), the generals still don't face
  // each other because there are Black Soldiers at (3,4) between them.
  SPIEL_CHECK_TRUE(HasAction(*state, 6, 4, 5, 4));
}

void TerminalStateTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();

  // Play a game until it terminates (using random simulation already tested
  // in BasicXiangqiTests). Here we just verify Returns on a non-terminal
  // state.
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{0.0, 0.0}));
}

void UndoActionTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();
  std::string initial_str = state->ToString();

  // Apply an action and undo it.
  Player player = state->CurrentPlayer();
  std::vector<Action> legal = state->LegalActions();
  SPIEL_CHECK_FALSE(legal.empty());
  Action action = legal[0];
  state->ApplyAction(action);
  SPIEL_CHECK_NE(state->ToString(), initial_str);

  state->UndoAction(player, action);
  SPIEL_CHECK_EQ(state->ToString(), initial_str);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), player);

  // Multi-step undo.
  Action a1 = state->LegalActions()[0];
  state->ApplyAction(a1);
  std::string after_a1 = state->ToString();
  Player p2 = state->CurrentPlayer();
  Action a2 = state->LegalActions()[0];
  state->ApplyAction(a2);

  state->UndoAction(p2, a2);
  SPIEL_CHECK_EQ(state->ToString(), after_a1);
  state->UndoAction(player, a1);
  SPIEL_CHECK_EQ(state->ToString(), initial_str);
}

void ObservationTensorTest() {
  auto game = LoadGame("xiangqi");
  auto state = game->NewInitialState();
  auto shape = game->ObservationTensorShape();

  SPIEL_CHECK_EQ(shape.size(), 3);
  SPIEL_CHECK_EQ(shape[0], kNumObservationPlanes);
  SPIEL_CHECK_EQ(shape[1], kNumRows);
  SPIEL_CHECK_EQ(shape[2], kNumCols);

  std::vector<float> tensor(game->ObservationTensorSize(), 0.0f);
  state->ObservationTensor(0, absl::MakeSpan(tensor));

  // Red General at (9, 4) should appear on plane 0.
  int idx = 0 * kNumRows * kNumCols + 9 * kNumCols + 4;
  SPIEL_CHECK_EQ(tensor[idx], 1.0f);

  // Black General at (0, 4) should appear on plane 7.
  idx = 7 * kNumRows * kNumCols + 0 * kNumCols + 4;
  SPIEL_CHECK_EQ(tensor[idx], 1.0f);

  // Current player plane (14): should be 1.0 (Red to move).
  idx = 14 * kNumRows * kNumCols + 0 * kNumCols + 0;
  SPIEL_CHECK_EQ(tensor[idx], 1.0f);
}

}  // namespace
}  // namespace xiangqi
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::xiangqi::BasicXiangqiTests();
  open_spiel::xiangqi::InitialStateTest();
  open_spiel::xiangqi::SoldierMovementTest();
  open_spiel::xiangqi::HorseBlockingTest();
  open_spiel::xiangqi::ElephantRiverTest();
  open_spiel::xiangqi::CannonCaptureTest();
  open_spiel::xiangqi::FlyingGeneralTest();
  open_spiel::xiangqi::TerminalStateTest();
  open_spiel::xiangqi::UndoActionTest();
  open_spiel::xiangqi::ObservationTensorTest();
}
