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

#include "open_spiel/games/rbc.h"

#include "open_spiel/games/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace rbc {
namespace {

namespace testing = open_spiel::testing;

void TestPassMove() {
  auto game = LoadGame("rbc");
  std::unique_ptr<State> s = game->NewInitialState();
  SPIEL_CHECK_EQ(s->ToString(), chess::kDefaultStandardFEN);

  // First move
  SPIEL_CHECK_EQ(s->ActionToString(Player{0}, 0), "Sense a1");
  SPIEL_CHECK_EQ(s->StringToAction("Sense a1"), 0);
  s->ApplyAction(0);  // Sense phase
  SPIEL_CHECK_EQ(s->ActionToString(Player{0}, chess::kPassAction), "pass");
  SPIEL_CHECK_EQ(s->StringToAction("pass"), chess::kPassAction);
  s->ApplyAction(chess::kPassAction);  // Move phase
  std::string black_fen = chess::kDefaultStandardFEN;
  std::replace(black_fen.begin(), black_fen.end(), 'w', 'b');  // Switch sides.
  SPIEL_CHECK_EQ(s->ToString(), black_fen);

  // Second move
  SPIEL_CHECK_EQ(s->ActionToString(Player{1}, 0), "Sense a1");
  s->ApplyAction(0);  // Sense phase
  SPIEL_CHECK_EQ(s->ActionToString(Player{1}, chess::kPassAction), "pass");
  s->ApplyAction(chess::kPassAction);  // Move phase
  std::string white_fen = chess::kDefaultStandardFEN;
  std::replace(white_fen.begin(), white_fen.end(), '1', '2');  // Update clock.
  SPIEL_CHECK_EQ(s->ToString(), white_fen);
}

void TestRepetitionDraw() {
  auto game = LoadGame("rbc");
  auto state = game->NewInitialState();
  auto rbc_state = down_cast<RbcState*>(state.get());
  for (int i = 0; i < 2 * 2; ++i) {  // 2 players, 2 repetitions.
    SPIEL_CHECK_FALSE(state->IsTerminal());
    SPIEL_CHECK_FALSE(rbc_state->IsRepetitionDraw());
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(chess::kPassAction);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(rbc_state->IsRepetitionDraw());
}

// Helper function. Get the named tensor.
SpanTensor GetSpanTensor(Observation& obs, const std::string& name) {
  for (SpanTensor span_tensor : obs.tensors()) {
    if (span_tensor.info().name() == name) return span_tensor;
  }
  SpielFatalError(absl::StrCat("SpanTensor '", name, "' was not found!"));
}

void TestIllegalMovesFlag() {
  // Setup test code.
  auto game = LoadGame("rbc");
  Observation observation(*game, game->MakeObserver(kDefaultObsType, {}));
  SpanTensor illegal_move_span = GetSpanTensor(observation, "illegal_move");
  SPIEL_CHECK_EQ(illegal_move_span.info().size(), 2);  // Binary observation.
  auto CHECK_OBSERVATION = [&](const State& s, bool illegal) {
    observation.SetFrom(s, kDefaultPlayerId);
    SPIEL_CHECK_EQ(illegal_move_span.at(0), !illegal);
    SPIEL_CHECK_EQ(illegal_move_span.at(1), illegal);
  };

  {  // No move has been made.
    auto state = game->NewInitialState();
    CHECK_OBSERVATION(*state, /*illegal=*/false);
  }
  {  // Legal pawn move.
    auto state = game->NewInitialState();
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("a2a4"));
    CHECK_OBSERVATION(*state, /*illegal=*/false);
  }
  {  // Illegal pawn attack.
    auto state = game->NewInitialState();
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("a2b3"));
    CHECK_OBSERVATION(*state, /*illegal=*/true);
  }
  {  // Illegal pawn forward move.
    auto state = game->NewInitialState();
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("a2a4"));
    CHECK_OBSERVATION(*state, /*illegal=*/false);
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("a7a5"));
    CHECK_OBSERVATION(*state, /*illegal=*/false);
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("a4a5"));
    CHECK_OBSERVATION(*state, /*illegal=*/true);
  }
  {  // Allow castling when king is in check.
    auto state = game->NewInitialState(
        "rnbqkb1r/pppppp1p/6p1/8/8/3n1NPB/PPP1PP1P/RNBQK2R w KQkq - 0 1");
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("e1g1"));
    CHECK_OBSERVATION(*state, /*illegal=*/false);
  }
  {  // Allow castling through a square controlled by the enemy.
    auto state = game->NewInitialState(
        "rnbqkb1r/pppppp1p/6p1/8/8/5NPB/PPPnPP1P/RNBQK2R w KQkq - 0 1");
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("e1g1"));
    CHECK_OBSERVATION(*state, /*illegal=*/false);
  }
  {  // Allow castling when king will be in check after castling.
    auto state = game->NewInitialState(
        "rnbqkb1r/pppppp1p/6p1/8/8/5nPB/PPP1PP1P/RNBQK2R w KQkq - 0 1");
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("e1g1"));
    CHECK_OBSERVATION(*state, /*illegal=*/false);
  }
  {  // Illegal castling:
    // There is an opponent piece between the king and the rook.
    auto state = game->NewInitialState(
        "rnbqkb1r/pppppp1p/6p1/8/8/6PB/PPP1PP1P/RNBQK1nR w KQkq - 0 1");
    state->ApplyAction(state->StringToAction("Sense a1"));
    state->ApplyAction(state->StringToAction("e1g1"));
    CHECK_OBSERVATION(*state, /*illegal=*/true);
  }
}

void TestKingCaptureEndsTheGame() {
  auto game = LoadGame("rbc");
  auto state = game->NewInitialState(
      "rnbqk1nr/pppp1ppp/4p3/8/4P3/3P1Pb1/PPP3PP/RNBQKBNR b KQkq - 0 1");
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(state->StringToAction("Sense a1"));
  state->ApplyAction(state->StringToAction("g3e1"));
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(Player{0}), 1);
}

void TestMakeKingMoveInCheck() {
  // Make a move that leaves the king in check.
  auto game = LoadGame("rbc");
  auto state = game->NewInitialState(
      "rnbqk1nr/pppp1ppp/4p3/8/4P3/3P1Pb1/PPP3PP/RNBQKBNR w KQkq - 0 1");
  state->ApplyAction(state->StringToAction("Sense a1"));
  state->ApplyAction(state->StringToAction("e1f2"));
}

void TestPawnBreachingMoveTwoSquares() {
  auto game = LoadGame("rbc");
  auto state = game->NewInitialState(
      "rnbqk1nr/pppp1ppp/4p3/8/1b6/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
  state->ApplyAction(state->StringToAction("Sense a1"));
  state->ApplyAction(state->StringToAction("b2b4"));
  // Pawn moved only one square.
  SPIEL_CHECK_EQ(
      state->ToString(),
      "rnbqk1nr/pppp1ppp/4p3/8/1b6/1P2P3/P1PP1PPP/RNBQKBNR b KQkq - 0 1");
  // And the move was marked as illegal.
  SPIEL_CHECK_TRUE(down_cast<RbcState*>(state.get())->illegal_move_attempted());
}

void BasicRbcTests(int board_size) {
  GameParameters params;
  params["board_size"] = GameParameter(board_size);

  testing::LoadGameTest("rbc");
  testing::NoChanceOutcomesTest(*LoadGame("rbc", params));
  testing::RandomSimTest(*LoadGame("rbc", params), 100);
  testing::RandomSimTestWithUndo(*LoadGame("rbc", params), 1);
}

}  // namespace
}  // namespace rbc
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::rbc::TestPassMove();
  open_spiel::rbc::TestRepetitionDraw();
  open_spiel::rbc::TestIllegalMovesFlag();
  open_spiel::rbc::TestKingCaptureEndsTheGame();
  open_spiel::rbc::TestMakeKingMoveInCheck();
  open_spiel::rbc::TestPawnBreachingMoveTwoSquares();

  open_spiel::rbc::BasicRbcTests(/*board_size=*/8);
  open_spiel::rbc::BasicRbcTests(/*board_size=*/4);
}
