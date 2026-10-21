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

#include "open_spiel/games/konane/konane.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/json/include/nlohmann/json.hpp"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace konane {
namespace {

namespace testing = open_spiel::testing;

int CountStones(const State& state) {
  const auto& konane_state = down_cast<const KonaneState&>(state);
  const std::vector<int> shape = state.GetGame()->ObservationTensorShape();
  int count = 0;
  for (int row = 0; row < shape[1]; row++) {
    for (int column = 0; column < shape[2]; column++) {
      if (konane_state.BoardAt(row, column) != CellState::kEmpty) count++;
    }
  }
  return count;
}

std::vector<std::string> ActionStrings(const State& state) {
  std::vector<std::string> strings;
  for (Action action : state.LegalActions()) {
    strings.push_back(state.ActionToString(action));
  }
  return strings;
}

void BasicKonaneTests() {
  testing::LoadGameTest("konane");
  testing::NoChanceOutcomesTest(*LoadGame("konane"));
  testing::RandomSimTest(*LoadGame("konane"), 20);
  testing::RandomSimTest(*LoadGame("konane(rows=6,columns=6)"), 20);
  testing::RandomSimTest(*LoadGame("konane(rows=4,columns=10)"), 20);
}

// Boards whose longest side is at most four allow only single jumps. The jump
// count is a mixed-radix digit and its base must exceed 1, so these sizes used
// to fail an internal check as soon as any jump was enumerated.
void SmallBoardTest() {
  for (const std::string& size :
       {"rows=2,columns=2", "rows=2,columns=4", "rows=3,columns=3",
        "rows=3,columns=4", "rows=4,columns=3", "rows=4,columns=4"}) {
    testing::RandomSimTest(*LoadGame(absl::StrCat("konane(", size, ")")), 5);
  }

  // A 4x4 board reaches a real jump, which is what used to crash.
  std::shared_ptr<const Game> game = LoadGame("konane(rows=4,columns=4)");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(state->LegalActions()[0]);  // Black removes a4.
  state->ApplyAction(state->LegalActions()[0]);  // White removes an adjacent.
  SPIEL_CHECK_FALSE(state->LegalActions().empty());
  const Action jump = state->LegalActions()[0];
  SPIEL_CHECK_EQ(state->ActionToString(jump).size(), 4);
  SPIEL_CHECK_LT(jump, game->NumDistinctActions());
  state->ApplyAction(jump);
}

// Plane 0 holds the observing player's stones, plane 1 the opponent's and
// plane 2 the empty squares, so the perspective flips between the players.
void ObservationTensorTest() {
  std::shared_ptr<const Game> game = LoadGame("konane");
  std::unique_ptr<State> state = game->NewInitialState();
  const std::vector<int> shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(shape, (std::vector<int>{3, 8, 8}));

  for (Player player : {Player{0}, Player{1}}) {
    std::vector<float> tensor = state->ObservationTensor(player);
    SPIEL_CHECK_EQ(tensor.size(), game->ObservationTensorSize());

    const int plane_size = shape[1] * shape[2];
    for (int row = 0; row < shape[1]; row++) {
      for (int column = 0; column < shape[2]; column++) {
        const int offset = row * shape[2] + column;
        const CellState cell =
            down_cast<const KonaneState&>(*state).BoardAt(row, column);
        const int expected =
            cell == CellState::kEmpty
                ? 2
                : (cell == CellState::kBlack ? player : 1 - player);
        for (int plane = 0; plane < shape[0]; plane++) {
          SPIEL_CHECK_EQ(tensor[plane * plane_size + offset],
                         plane == expected ? 1.0 : 0.0);
        }
      }
    }
  }

  // Exactly one plane is set per square, and the opening board has no holes.
  std::vector<float> tensor = state->ObservationTensor(0);
  SPIEL_CHECK_EQ(std::accumulate(tensor.begin(), tensor.end(), 0.0f), 64.0f);
  SPIEL_CHECK_EQ(std::accumulate(tensor.begin() + 128, tensor.end(), 0.0f),
                 0.0f);

  // After a removal the vacated square moves into the empty plane.
  state->ApplyAction(state->LegalActions()[0]);  // Black removes a8.
  tensor = state->ObservationTensor(0);
  SPIEL_CHECK_EQ(std::accumulate(tensor.begin() + 128, tensor.end(), 0.0f),
                 1.0f);
}

// Black opens from a corner or the centre; on the standard board that is
// a8, d5, e4 and h1 -- the black squares among the corners and the centre.
void OpeningMovesTest() {
  std::shared_ptr<const Game> game = LoadGame("konane");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(ActionStrings(*state),
                 (std::vector<std::string>{"a8", "d5", "e4", "h1"}));

  // After a8 is removed, White may only take the two white stones next to it.
  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_EQ(ActionStrings(*state),
                 (std::vector<std::string>{"b8", "a7"}));
}

// The two removals leave adjacent holes, and a stone may not jump over a hole,
// so exactly one jump is available on the standard opening.
void FirstJumpTest() {
  std::shared_ptr<const Game> game = LoadGame("konane");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(state->LegalActions()[0]);  // Black removes a8.
  state->ApplyAction(state->LegalActions()[0]);  // White removes b8.
  SPIEL_CHECK_EQ(state->ToString().substr(0, 10), "8..xoxoxo\n");

  // a6 jumps up over a7 into a8; c8 cannot jump left because b8 is empty.
  SPIEL_CHECK_EQ(ActionStrings(*state), (std::vector<std::string>{"a6a8"}));

  state->ApplyAction(state->LegalActions()[0]);
  SPIEL_CHECK_EQ(state->ToString().substr(0, 10), "8x.xoxoxo\n");
  SPIEL_CHECK_EQ(state->ObservationString(0).substr(0, 17),
                 "Player to move: 1");
}

// Play a full deterministic game, always taking the longest chain on offer,
// and check that every turn captures exactly one stone per jump.
void ChainedJumpTest() {
  std::shared_ptr<const Game> game = LoadGame("konane");
  std::unique_ptr<State> state = game->NewInitialState();
  int longest_chain = 0;

  while (!state->IsTerminal()) {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_FALSE(actions.empty());
    // num_jumps is the least significant digit of the action, so the highest
    // ranked action is the longest chain from the last eligible stone.
    const Action action = actions.back();
    const std::string action_string = state->ActionToString(action);
    const int stones_before = CountStones(*state);
    state->ApplyAction(action);
    const int captured = stones_before - CountStones(*state);

    if (action_string.size() == 2) {
      SPIEL_CHECK_EQ(captured, 1);  // An opening removal.
    } else {
      // A chain of n jumps moves 2n squares and captures n stones.
      const int distance = std::abs(action_string[0] - action_string[2]) +
                           std::abs(action_string[1] - action_string[3]);
      SPIEL_CHECK_EQ(distance, 2 * captured);
      longest_chain = std::max(longest_chain, captured);
    }
  }

  SPIEL_CHECK_GE(longest_chain, 2);  // Multi-jumps really do occur.
  SPIEL_CHECK_EQ(state->Returns()[0] + state->Returns()[1], 0.0);
  SPIEL_CHECK_TRUE(state->Returns()[0] == 1.0 || state->Returns()[0] == -1.0);
}

// The player with no legal move loses.
// Every legal action must survive a trip through its struct, in both the
// opening removals and ordinary jumping play.
void ActionStructRoundTripTest() {
  std::shared_ptr<const Game> game = LoadGame("konane(rows=6,columns=6)");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(93);
  int removals_seen = 0;
  int jumps_seen = 0;
  while (!state->IsTerminal()) {
    const KonaneState* konane_state =
        down_cast<const KonaneState*>(state.get());
    for (Action action : state->LegalActions()) {
      std::unique_ptr<ActionStruct> action_struct =
          konane_state->ActionToStruct(state->CurrentPlayer(), action);
      const auto* move =
          SafeActionCast<KonaneActionStruct>(*action_struct);
      if (move->direction.empty()) {
        ++removals_seen;
        SPIEL_CHECK_EQ(move->num_jumps, 0);
      } else {
        ++jumps_seen;
        SPIEL_CHECK_GE(move->num_jumps, 1);
      }
      const std::vector<Action> actions =
          konane_state->StructToActions(*action_struct);
      SPIEL_CHECK_EQ(actions.size(), 1);
      SPIEL_CHECK_EQ(actions[0], action);
    }
    const std::vector<Action> legal = state->LegalActions();
    std::uniform_int_distribution<int> dis(0, legal.size() - 1);
    state->ApplyAction(legal[dis(rng)]);
  }
  SPIEL_CHECK_GT(removals_seen, 0);
  SPIEL_CHECK_GT(jumps_seen, 0);
}

// A position must survive a trip through its JSON, including whether it is
// terminal, and both players must observe the same thing.
void StateStructRoundTripTest() {
  std::shared_ptr<const Game> game = LoadGame("konane(rows=4,columns=4)");
  std::unique_ptr<State> state = game->NewInitialState();
  // The board starts full, with stones alternating along each row.
  SPIEL_CHECK_EQ(
      state->ToJson(),
      "{\"board\":[\"x\",\"o\",\"x\",\"o\",\"o\",\"x\",\"o\","
      "\"x\",\"x\",\"o\",\"x\",\"o\",\"o\",\"x\",\"o\","
      "\"x\"],\"columns\":4,\"current_player\":\"x\","
      "\"num_moves\":0,\"rows\":4}");

  std::mt19937 rng(17);
  while (true) {
    const KonaneState* konane_state =
        down_cast<const KonaneState*>(state.get());
    std::unique_ptr<StateStruct> state_struct = konane_state->ToStruct();
    SPIEL_CHECK_EQ(state_struct->ToJson(), konane_state->ToJson());

    // Konane is a perfect information game.
    SPIEL_CHECK_EQ(konane_state->ToObservationStruct(0)->ToJson(),
                   konane_state->ToObservationStruct(1)->ToJson());
    SPIEL_CHECK_EQ(konane_state->ToObservationStruct(0)->ToJson(),
                   state_struct->ToJson());

    std::unique_ptr<State> restored =
        game->NewInitialState(nlohmann::json::parse(state_struct->ToJson()));
    SPIEL_CHECK_EQ(restored->ToString(), state->ToString());
    SPIEL_CHECK_EQ(restored->ToJson(), state_struct->ToJson());
    SPIEL_CHECK_EQ(restored->IsTerminal(), state->IsTerminal());
    SPIEL_CHECK_EQ(restored->CurrentPlayer(), state->CurrentPlayer());
    if (state->IsTerminal()) {
      SPIEL_CHECK_EQ(restored->Returns(), state->Returns());
      break;
    }
    SPIEL_CHECK_EQ(restored->LegalActions(), state->LegalActions());
    const std::vector<Action> legal = state->LegalActions();
    std::uniform_int_distribution<int> dis(0, legal.size() - 1);
    state->ApplyAction(legal[dis(rng)]);
  }
}

// The struct API can drive the game: applying an action struct is the same as
// applying the action it names.
void ApplyActionStructTest() {
  std::shared_ptr<const Game> game = LoadGame("konane(rows=4,columns=4)");
  std::unique_ptr<State> state = game->NewInitialState();
  const Action removal = state->LegalActions().front();
  std::unique_ptr<ActionStruct> removal_struct =
      down_cast<const KonaneState*>(state.get())
          ->ActionToStruct(state->CurrentPlayer(), removal);

  std::unique_ptr<State> by_struct = game->NewInitialState();
  SPIEL_CHECK_TRUE(by_struct->ApplyActionStruct(*removal_struct).ok());
  state->ApplyAction(removal);
  SPIEL_CHECK_EQ(by_struct->ToString(), state->ToString());

  // The same removal cannot be played twice: that square is empty now.
  SPIEL_CHECK_FALSE(by_struct->ValidateActionStruct(*removal_struct).ok());
}

void TerminalTest() {
  std::shared_ptr<const Game> game = LoadGame("konane(rows=2,columns=2)");
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(state->LegalActions()[0]);  // Black removes a2.
  state->ApplyAction(state->LegalActions()[0]);  // White removes b2.
  // Only two stones are left on a 2x2 board, so Black cannot move.
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{-1.0, 1.0}));
}

}  // namespace
}  // namespace konane
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::konane::BasicKonaneTests();
  open_spiel::konane::SmallBoardTest();
  open_spiel::konane::ObservationTensorTest();
  open_spiel::konane::OpeningMovesTest();
  open_spiel::konane::FirstJumpTest();
  open_spiel::konane::ChainedJumpTest();
  open_spiel::konane::TerminalTest();
  open_spiel::konane::ActionStructRoundTripTest();
  open_spiel::konane::StateStructRoundTripTest();
  open_spiel::konane::ApplyActionStructTest();
}
