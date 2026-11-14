// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/altior/altior.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace altior {
namespace {

namespace testing = open_spiel::testing;

using Placement = std::vector<std::pair<std::string, std::string>>;
using Moves = std::vector<std::string>;
using Values = std::vector<double>;

// Cell index of a square such as "c3" (file a-g, rank 1-7; rank 7 is row 0).
int Cell(const std::string& name) {
  return (kNumRows - (name[1] - '0')) * kNumCols + (name[0] - 'a');
}

// Builds a state from {square, discs} pairs, discs bottom first ('i'/'o').
std::unique_ptr<AltiorState> MakeState(
    const Placement& placement, Player to_move = kIvory,
    const std::string& game_string = "altior") {
  Board board;
  for (const auto& [square, discs] : placement) {
    board[Cell(square)] = Stack::FromString(discs);
  }
  return std::make_unique<AltiorState>(LoadGame(game_string), board, to_move);
}

const AltiorState& AsAltior(const State& state) {
  return down_cast<const AltiorState&>(state);
}

std::string StackAt(const State& state, const std::string& square) {
  return AsAltior(state).StackAt(Cell(square)).ToString();
}

// Sorted action strings of the legal moves.
Moves LegalMoves(const State& state) {
  Moves moves;
  for (Action action : state.LegalActions()) {
    moves.push_back(state.ActionToString(action));
  }
  std::sort(moves.begin(), moves.end());
  return moves;
}

bool IsLegal(const State& state, const std::string& move) {
  Moves moves = LegalMoves(state);
  return std::find(moves.begin(), moves.end(), move) != moves.end();
}

// True if any legal move starts from `square`.
bool CanMoveFrom(const State& state, const std::string& square) {
  for (const std::string& move : LegalMoves(state)) {
    if (move.substr(0, 2) == square) return true;
  }
  return false;
}

void Play(State* state, const std::string& move) {
  for (Action action : state->LegalActions()) {
    if (state->ActionToString(action) == move) {
      state->ApplyAction(action);
      return;
    }
  }
  SpielFatalError(absl::StrCat("Illegal move: ", move));
}

void PlayAll(State* state, const Moves& moves) {
  for (const std::string& move : moves) Play(state, move);
}

// One Ivory and one Onyx single shuffling out and back to the start.
Moves ShuffleCycle() { return {"a7-b7", "g1-f1", "b7-a7", "f1-g1"}; }

void BasicAltiorTests() {
  testing::LoadGameTest("altior");
  testing::NoChanceOutcomesTest(*LoadGame("altior"));
  testing::RandomSimTest(*LoadGame("altior"), 100);
  testing::RandomSimTestWithUndo(*LoadGame("altior"), 10);
  testing::RandomSimTest(*LoadGame("altior(max_game_length=20)"), 20);
}

void InitialSetupTest() {
  std::shared_ptr<const Game> game = LoadGame("altior");
  std::unique_ptr<State> state = game->NewInitialState();
  const AltiorState& altior = AsAltior(*state);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kIvory);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 196);
  SPIEL_CHECK_EQ(game->MaxGameLength(), kDefaultMaxGameLength);
  int ivory = 0;
  int onyx = 0;
  for (int cell = 0; cell < kNumCells; ++cell) {
    const Stack& stack = altior.StackAt(cell);
    if (cell == Cell("d4")) {
      SPIEL_CHECK_TRUE(stack.empty());
      continue;
    }
    SPIEL_CHECK_EQ(stack.height, 1);
    // Checkerboard: Ivory on every cell where row + col is even.
    SPIEL_CHECK_EQ(stack.Top(), ((cell / 7 + cell % 7) % 2 == 0) ? 0 : 1);
    (stack.Top() == kIvory ? ivory : onyx)++;
  }
  SPIEL_CHECK_EQ(ivory, 24);
  SPIEL_CHECK_EQ(onyx, 24);
  SPIEL_CHECK_EQ(StackAt(*state, "a7"), "i");
  SPIEL_CHECK_EQ(StackAt(*state, "b7"), "o");
  SPIEL_CHECK_EQ(StackAt(*state, "g1"), "i");
  SPIEL_CHECK_EQ(altior.RepetitionCount(), 1);
  // Count taken from the reference TypeScript engine.
  SPIEL_CHECK_EQ(state->LegalActions().size(), 80);
}

void MovementDistanceAndDirectionTest() {
  // A two-stack moves exactly two cells in each of the four directions.
  auto state = MakeState({{"c3", "ii"}, {"g1", "o"}});
  SPIEL_CHECK_TRUE(LegalMoves(*state) ==
                   Moves({"c3-a3", "c3-c1", "c3-c5", "c3-e3"}));
  // A three-stack jumps over the removed center (d2 -> d5), and sideways.
  state = MakeState({{"d2", "iii"}, {"g7", "o"}});
  SPIEL_CHECK_TRUE(LegalMoves(*state) == Moves({"d2-a2", "d2-d5", "d2-g2"}));
  // A four-stack on a4 can only reach e4, across the center.
  state = MakeState({{"a4", "oiii"}, {"g7", "o"}});
  SPIEL_CHECK_TRUE(LegalMoves(*state) == Moves({"a4-e4"}));
  // A single steps one cell, up and left included.
  state = MakeState({{"c3", "i"}, {"g7", "o"}});
  SPIEL_CHECK_TRUE(LegalMoves(*state) ==
                   Moves({"c3-b3", "c3-c2", "c3-c4", "c3-d3"}));
}

void JumpingTest() {
  // Intervening stacks of either color do not block a two-stack.
  auto state = MakeState({{"b2", "ii"}, {"b3", "o"}, {"c2", "i"}, {"g7", "o"}});
  SPIEL_CHECK_TRUE(IsLegal(*state, "b2-b4"));
  SPIEL_CHECK_TRUE(IsLegal(*state, "b2-d2"));
  Play(state.get(), "b2-b4");
  SPIEL_CHECK_EQ(StackAt(*state, "b4"), "ii");
  SPIEL_CHECK_EQ(StackAt(*state, "b3"), "o");  // Jumped stacks are untouched.
  SPIEL_CHECK_EQ(StackAt(*state, "c2"), "i");
  SPIEL_CHECK_EQ(StackAt(*state, "b2"), "");
}

void BoardEdgeAndCenterTest() {
  // Corner stacks may not leave the board.
  auto state = MakeState({{"a7", "ii"}, {"g1", "o"}});
  SPIEL_CHECK_TRUE(LegalMoves(*state) == Moves({"a7-a5", "a7-c7"}));
  // A single may not land on the removed center from any side.
  state = MakeState(
      {{"d5", "i"}, {"c4", "i"}, {"e4", "i"}, {"d3", "i"}, {"g1", "o"}});
  for (const std::string& move : LegalMoves(*state)) {
    SPIEL_CHECK_NE(move.substr(3), "d4");
  }
  SPIEL_CHECK_EQ(state->LegalActions().size(), 4 * 3);
  // A two-stack may not land on it either (d6 -> d4), nor jump to it.
  state = MakeState({{"d6", "ii"}, {"g1", "o"}});
  SPIEL_CHECK_TRUE(LegalMoves(*state) == Moves({"d6-b6", "d6-f6"}));
  // The center is shown as unplayable and the action is named but not legal.
  const State& base_state = *state;
  SPIEL_CHECK_EQ(base_state.ActionToString(Cell("d6") * 4 + kDown), "d6-?");
  SPIEL_CHECK_EQ(StackAt(*state, "d4"), "");
}

void MergeOrderAndControlTest() {
  // The moving stack goes on top; both stacks keep their internal order.
  auto state = MakeState({{"b3", "oi"}, {"d3", "io"}, {"g1", "o"}});
  Play(state.get(), "b3xd3");
  SPIEL_CHECK_EQ(StackAt(*state, "d3"), "iooi");
  SPIEL_CHECK_EQ(StackAt(*state, "b3"), "");
  // Control follows the top disc: an Onyx single lands on an Ivory stack.
  state = MakeState({{"a1", "o"}, {"a2", "ii"}, {"g7", "i"}}, kOnyx);
  Play(state.get(), "a1xa2");
  SPIEL_CHECK_EQ(StackAt(*state, "a2"), "iio");
  SPIEL_CHECK_EQ(AsAltior(*state).StackAt(Cell("a2")).Top(), kOnyx);
  // Ivory no longer controls it; Onyx does, once Ivory has moved.
  SPIEL_CHECK_FALSE(CanMoveFrom(*state, "a2"));
  Play(state.get(), "g7-g6");
  SPIEL_CHECK_TRUE(IsLegal(*state, "a2-a5"));
  Play(state.get(), "a2-a5");
  SPIEL_CHECK_EQ(StackAt(*state, "a5"), "iio");
  // An Ivory two-stack with an Onyx disc below lands on an Onyx single.
  state = MakeState({{"c1", "oi"}, {"e1", "o"}, {"g7", "o"}});
  Play(state.get(), "c1xe1");
  SPIEL_CHECK_EQ(StackAt(*state, "e1"), "ooi");
}

void HeightLimitTest() {
  // Three onto two makes exactly five: legal, and the result is frozen.
  auto state =
      MakeState({{"d2", "iii"}, {"g2", "oo"}, {"a7", "o"}, {"a1", "i"}});
  SPIEL_CHECK_TRUE(IsLegal(*state, "d2xg2"));
  Play(state.get(), "d2xg2");
  SPIEL_CHECK_EQ(StackAt(*state, "g2"), "ooiii");
  Play(state.get(), "a7-a6");
  SPIEL_CHECK_FALSE(state->IsTerminal());  // Ivory still has a1.
  SPIEL_CHECK_FALSE(CanMoveFrom(*state, "g2"));
  // Three onto three would make six: illegal.
  state = MakeState({{"d2", "iii"}, {"g2", "ooo"}, {"a7", "o"}});
  SPIEL_CHECK_TRUE(LegalMoves(*state) == Moves({"d2-a2", "d2-d5"}));
  // Nothing may land on a five-stack, and a five-stack never moves, even if
  // its top disc belongs to the player to move.
  state = MakeState({{"c2", "i"}, {"c3", "ooooi"}, {"g7", "o"}});
  SPIEL_CHECK_FALSE(IsLegal(*state, "c2xc3"));
  SPIEL_CHECK_FALSE(CanMoveFrom(*state, "c3"));
  SPIEL_CHECK_EQ(state->LegalActions().size(), 3);  // c2 -> b2, d2, c1.
}

void FourStackWinTest() {
  // A four-stack landing four cells away on an empty cell wins at once.
  auto state = MakeState({{"a4", "oiii"}, {"g7", "o"}});
  Play(state.get(), "a4-e4");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kTerminalPlayerId);
  SPIEL_CHECK_TRUE(state->LegalActions().empty());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kFourStackWin);
  SPIEL_CHECK_EQ(AsAltior(*state).winner(), kIvory);
  SPIEL_CHECK_TRUE(state->Returns() == Values({1.0, -1.0}));
  SPIEL_CHECK_EQ(state->PlayerReturn(kOnyx), -1.0);
  // Onyx can win too.
  state = MakeState({{"a4", "iooo"}, {"g7", "i"}}, kOnyx);
  Play(state.get(), "a4-e4");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->Returns() == Values({-1.0, 1.0}));
}

void FalseWinTest() {
  // A four-stack onto an occupied cell makes a frozen five, not a win.
  auto state = MakeState({{"a4", "oiii"}, {"e4", "o"}, {"g7", "o"}});
  Play(state.get(), "a4xe4");
  SPIEL_CHECK_EQ(StackAt(*state, "e4"), "ooiii");
  SPIEL_CHECK_FALSE(state->IsTerminal());
  // A three-stack onto a single builds a four, but not onto an empty cell.
  state = MakeState({{"d2", "iii"}, {"g2", "o"}, {"a7", "o"}});
  Play(state.get(), "d2xg2");
  SPIEL_CHECK_EQ(StackAt(*state, "g2"), "oiii");
  SPIEL_CHECK_FALSE(state->IsTerminal());
  // So do two two-stacks merging.
  state = MakeState({{"a1", "ii"}, {"c1", "oi"}, {"g7", "o"}});
  Play(state.get(), "a1xc1");
  SPIEL_CHECK_EQ(StackAt(*state, "c1"), "oiii");
  SPIEL_CHECK_FALSE(state->IsTerminal());
  // A three-stack onto an empty cell is an ordinary move.
  state = MakeState({{"d2", "iii"}, {"a7", "o"}});
  Play(state.get(), "d2-d5");
  SPIEL_CHECK_FALSE(state->IsTerminal());
  // An opponent-controlled four-stack cannot be used by the player to move.
  state = MakeState({{"a4", "iiio"}, {"g7", "i"}});
  SPIEL_CHECK_FALSE(CanMoveFrom(*state, "a4"));
}

void ImmobilityTest() {
  // Neither player can move: draw.
  auto state = MakeState({{"a7", "iiiii"}, {"g1", "ooooo"}});
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kBothImmobile);
  SPIEL_CHECK_TRUE(state->Returns() == Values({0.0, 0.0}));
  SPIEL_CHECK_TRUE(state->LegalActions().empty());
  // The same draw reached by a move that freezes the last mobile stacks.
  state = MakeState({{"a7", "i"}, {"b7", "iiii"}, {"g1", "ooooo"}});
  Play(state.get(), "a7xb7");
  SPIEL_CHECK_EQ(StackAt(*state, "b7"), "iiiii");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kBothImmobile);
  SPIEL_CHECK_TRUE(state->Returns() == Values({0.0, 0.0}));
  // Only the player to move is stuck: they lose (official rulebook).
  const Placement stuck_ivory = {
      {"a7", "i"}, {"b7", "ooooo"}, {"a6", "ooooo"}, {"g2", "o"}};
  state = MakeState(stuck_ivory, kIvory);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() ==
                   EndReason::kOpponentImmobile);
  SPIEL_CHECK_TRUE(state->Returns() == Values({-1.0, 1.0}));
  // The same loss reached by the opponent's move.
  state = MakeState(stuck_ivory, kOnyx);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  Play(state.get(), "g2-g1");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(AsAltior(*state).winner(), kOnyx);
  SPIEL_CHECK_TRUE(state->Returns() == Values({-1.0, 1.0}));
  // A position where both players can move is not terminal.
  state = MakeState({{"a7", "i"}, {"c3", "i"}, {"g1", "o"}});
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void ThreefoldRepetitionTest() {
  auto state = MakeState({{"a7", "i"}, {"g1", "o"}});
  const Moves cycle = ShuffleCycle();
  PlayAll(state.get(), cycle);
  // The start position has now occurred twice: not yet a draw.
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 2);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  PlayAll(state.get(), Moves(cycle.begin(), cycle.begin() + 3));
  SPIEL_CHECK_FALSE(state->IsTerminal());
  Play(state.get(), cycle[3]);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kRepetition);
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 3);
  SPIEL_CHECK_TRUE(state->Returns() == Values({0.0, 0.0}));
  SPIEL_CHECK_TRUE(state->LegalActions().empty());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kTerminalPlayerId);
}

void TranspositionTest() {
  // Positions are compared by content, not by the move order reaching them.
  auto state = MakeState({{"a7", "i"}, {"a1", "i"}, {"g1", "o"}, {"g7", "o"}});
  PlayAll(state.get(), {"a7-b7", "g1-f1", "a1-b1", "g7-f7"});
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 1);
  const std::string position = state->ToString();
  // Back to the start (second occurrence there), then the same position again
  // by playing the same four moves in a different order.
  PlayAll(state.get(), {"b7-a7", "f1-g1", "b1-a1", "f7-g7"});
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 2);
  PlayAll(state.get(), {"a1-b1", "g7-f7", "a7-b7", "g1-f1"});
  SPIEL_CHECK_EQ(state->ToString(), position);
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 2);
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void CloneAndUndoPreserveRepetitionTest() {
  auto state = MakeState({{"a7", "i"}, {"g1", "o"}});
  PlayAll(state.get(), ShuffleCycle());
  std::unique_ptr<State> clone = state->Clone();
  SPIEL_CHECK_EQ(AsAltior(*clone).RepetitionCount(), 2);
  PlayAll(state.get(), ShuffleCycle());
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // The clone is unaffected by moves made on the original.
  SPIEL_CHECK_FALSE(clone->IsTerminal());
  SPIEL_CHECK_EQ(clone->History().size(), 4);
  // Undoing the draw reopens the game and rolls the counts back.
  state->UndoAction(kOnyx, state->History().back());
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kOnyx);
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 2);
  // Replaying that move reproduces the draw.
  Play(state.get(), "f1-g1");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kRepetition);
  // The clone reaches the same draw independently.
  PlayAll(clone.get(), ShuffleCycle());
  SPIEL_CHECK_TRUE(clone->IsTerminal());
  SPIEL_CHECK_EQ(clone->ToString(), state->ToString());
}

// A game from the standard start that ends by repetition at ply 13. The
// reference TypeScript engine reaches the same draw at the same ply.
void RepetitionFromStandardStartTest() {
  const Moves kLine = {"c1xc2", "f7xg7", "g5xg4", "e6xe7", "d2xe2",
                       "g7-g5", "d6-e6", "g5-g7", "e6-d6", "g7-g5",
                       "d6-e6", "g5-g7", "e6-d6"};
  std::shared_ptr<const Game> game = LoadGame("altior");
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<std::string> snapshots = {state->ToString()};
  for (const std::string& move : kLine) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    Play(state.get(), move);
    snapshots.push_back(state->ToString());
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kRepetition);
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 3);

  // Serialization replays the history and so rebuilds the repetition counts.
  auto restored = DeserializeGameAndState(SerializeGameAndState(*game, *state));
  SPIEL_CHECK_TRUE(restored.second->IsTerminal());
  SPIEL_CHECK_EQ(restored.second->ToString(), state->ToString());
  SPIEL_CHECK_EQ(AsAltior(*restored.second).RepetitionCount(), 3);

  // Undoing every move restores every earlier position and its count.
  for (int ply = kLine.size(); ply > 0; --ply) {
    state->UndoAction(ply % 2 == 1 ? kIvory : kOnyx, state->History().back());
    snapshots.pop_back();
    SPIEL_CHECK_EQ(state->ToString(), snapshots.back());
    SPIEL_CHECK_FALSE(state->IsTerminal());
  }
  SPIEL_CHECK_EQ(AsAltior(*state).RepetitionCount(), 1);
  // Replaying reproduces the draw: the counts were fully rolled back.
  PlayAll(state.get(), kLine);
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kRepetition);
}

void MoveLimitTest() {
  // The move limit is a safeguard, reported apart from repetition.
  auto state = MakeState({{"a7", "i"}, {"g1", "o"}}, kIvory,
                         "altior(max_game_length=3)");
  PlayAll(state.get(), {"a7-b7", "g1-f1"});
  SPIEL_CHECK_FALSE(state->IsTerminal());
  Play(state.get(), "b7-a7");
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kMoveLimit);
  SPIEL_CHECK_TRUE(state->Returns() == Values({0.0, 0.0}));
  state->UndoAction(kIvory, state->History().back());
  SPIEL_CHECK_FALSE(state->IsTerminal());
  // A win on the last permitted ply is still a win.
  state = MakeState({{"a4", "oiii"}, {"g7", "o"}}, kIvory,
                    "altior(max_game_length=1)");
  Play(state.get(), "a4-e4");
  SPIEL_CHECK_TRUE(AsAltior(*state).end_reason() == EndReason::kFourStackWin);
}

// Plays `plies` random moves (stopping early at a terminal state).
void PlayRandomPlies(State* state, std::mt19937* rng, int plies) {
  for (int i = 0; i < plies && !state->IsTerminal(); ++i) {
    std::vector<Action> actions = state->LegalActions();
    state->ApplyAction(actions[std::uniform_int_distribution<int>(
        0, actions.size() - 1)(*rng)]);
  }
}

void CloneIndependenceTest() {
  std::shared_ptr<const Game> game = LoadGame("altior");
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(42);
  PlayRandomPlies(state.get(), &rng, 10);
  const std::string before = state->ToString();
  std::unique_ptr<State> clone = state->Clone();
  SPIEL_CHECK_EQ(clone->ToString(), before);
  PlayRandomPlies(clone.get(), &rng, 10);
  SPIEL_CHECK_EQ(state->ToString(), before);
  SPIEL_CHECK_EQ(state->History().size(), 10);
  SPIEL_CHECK_EQ(clone->History().size(), 20);
  PlayRandomPlies(state.get(), &rng, 3);
  SPIEL_CHECK_NE(state->ToString(), clone->ToString());
}

void SerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("altior");
  std::mt19937 rng(7);
  for (int plies : {0, 1, 17, 60}) {
    std::unique_ptr<State> state = game->NewInitialState();
    PlayRandomPlies(state.get(), &rng, plies);
    std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>> restored =
        DeserializeGameAndState(SerializeGameAndState(*game, *state));
    SPIEL_CHECK_EQ(restored.second->ToString(), state->ToString());
    SPIEL_CHECK_TRUE(restored.second->History() == state->History());
    SPIEL_CHECK_TRUE(restored.second->LegalActions() == state->LegalActions());
    SPIEL_CHECK_EQ(AsAltior(*restored.second).RepetitionCount(),
                   AsAltior(*state).RepetitionCount());
    SPIEL_CHECK_EQ(restored.first->MaxGameLength(), game->MaxGameLength());
  }
  // Non-default parameters survive the round trip.
  std::shared_ptr<const Game> short_game =
      LoadGame("altior(max_game_length=5)");
  std::unique_ptr<State> state = short_game->NewInitialState();
  PlayRandomPlies(state.get(), &rng, 3);
  auto restored =
      DeserializeGameAndState(SerializeGameAndState(*short_game, *state));
  SPIEL_CHECK_EQ(restored.first->MaxGameLength(), 5);
}

void UndoTest() {
  std::shared_ptr<const Game> game = LoadGame("altior");
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<std::string> snapshots = {state->ToString()};
  std::mt19937 rng(11);
  for (int i = 0; i < 25 && !state->IsTerminal(); ++i) {
    PlayRandomPlies(state.get(), &rng, 1);
    snapshots.push_back(state->ToString());
  }
  while (!state->History().empty()) {
    const Action last = state->History().back();
    const Player mover = state->History().size() % 2 == 1 ? kIvory : kOnyx;
    state->UndoAction(mover, last);
    snapshots.pop_back();
    SPIEL_CHECK_EQ(state->ToString(), snapshots.back());
    SPIEL_CHECK_EQ(state->MoveNumber(), state->History().size());
  }
  // Undoing a merge restores the order of every disc, and a win is undone.
  auto merge = MakeState({{"b3", "oi"}, {"d3", "io"}, {"a4", "oiii"}});
  Play(merge.get(), "b3xd3");
  merge->UndoAction(kIvory, merge->History().back());
  SPIEL_CHECK_EQ(StackAt(*merge, "b3"), "oi");
  SPIEL_CHECK_EQ(StackAt(*merge, "d3"), "io");
  Play(merge.get(), "a4-e4");
  SPIEL_CHECK_TRUE(merge->IsTerminal());
  merge->UndoAction(kIvory, merge->History().back());
  SPIEL_CHECK_FALSE(merge->IsTerminal());
  SPIEL_CHECK_EQ(StackAt(*merge, "a4"), "oiii");
  SPIEL_CHECK_EQ(StackAt(*merge, "e4"), "");
  SPIEL_CHECK_EQ(merge->CurrentPlayer(), kIvory);
  // Undoing an immobility result.
  auto stuck = MakeState(
      {{"a7", "i"}, {"b7", "ooooo"}, {"a6", "ooooo"}, {"g2", "o"}}, kOnyx);
  Play(stuck.get(), "g2-g1");
  SPIEL_CHECK_TRUE(stuck->IsTerminal());
  stuck->UndoAction(kOnyx, stuck->History().back());
  SPIEL_CHECK_FALSE(stuck->IsTerminal());
  SPIEL_CHECK_EQ(StackAt(*stuck, "g2"), "o");
}

// Observation tensor for `player` (the base-class overload returning a vector).
std::vector<float> Tensor(const State& state, Player player) {
  return state.ObservationTensor(player);
}

// Offset of `plane` for cell `cell` in a flattened observation tensor.
int Index(int plane, int cell) { return plane * kNumCells + cell; }

float PlaneSum(const std::vector<float>& tensor, int plane) {
  return std::accumulate(tensor.begin() + Index(plane, 0),
                         tensor.begin() + Index(plane + 1, 0), 0.0f);
}

void ObservationTest() {
  std::shared_ptr<const Game> game = LoadGame("altior");
  SPIEL_CHECK_TRUE(game->ObservationTensorShape() ==
                   std::vector<int>({kNumObservationPlanes, 7, 7}));
  SPIEL_CHECK_EQ(game->ObservationTensorSize(), kNumObservationPlanes * 49);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<float> tensor = Tensor(*state, kIvory);
  SPIEL_CHECK_EQ(tensor.size(), kNumObservationPlanes * 49);
  SPIEL_CHECK_TRUE(tensor == Tensor(*state, kOnyx));
  SPIEL_CHECK_EQ(state->ObservationString(kIvory), state->ToString());
  SPIEL_CHECK_EQ(PlaneSum(tensor, 0), 24);  // Ivory bottom discs.
  SPIEL_CHECK_EQ(PlaneSum(tensor, kOnyxPlaneOffset), 24);
  for (int depth = 1; depth < kMaxHeight; ++depth) {
    SPIEL_CHECK_EQ(PlaneSum(tensor, depth), 0);
    SPIEL_CHECK_EQ(PlaneSum(tensor, kOnyxPlaneOffset + depth), 0);
  }
  SPIEL_CHECK_EQ(PlaneSum(tensor, kEmptyPlane), 0);
  SPIEL_CHECK_EQ(PlaneSum(tensor, kUnplayablePlane), 1);
  SPIEL_CHECK_EQ(tensor[Index(kUnplayablePlane, Cell("d4"))], 1);
  SPIEL_CHECK_EQ(PlaneSum(tensor, kToMovePlaneOffset + kIvory), 49);
  SPIEL_CHECK_EQ(PlaneSum(tensor, kToMovePlaneOffset + kOnyx), 0);
  SPIEL_CHECK_EQ(tensor[Index(0, Cell("a7"))], 1);
  SPIEL_CHECK_EQ(tensor[Index(kOnyxPlaneOffset, Cell("b7"))], 1);

  // The whole stack is visible: disc order matters, not only the top disc.
  auto io = MakeState({{"c3", "io"}, {"a7", "i"}, {"g1", "o"}});
  auto oi = MakeState({{"c3", "oi"}, {"a7", "i"}, {"g1", "o"}});
  const std::vector<float> io_tensor = Tensor(*io, kIvory);
  const std::vector<float> oi_tensor = Tensor(*oi, kIvory);
  SPIEL_CHECK_TRUE(io_tensor != oi_tensor);
  SPIEL_CHECK_EQ(io_tensor[Index(0, Cell("c3"))], 1);
  SPIEL_CHECK_EQ(io_tensor[Index(kOnyxPlaneOffset + 1, Cell("c3"))], 1);
  SPIEL_CHECK_EQ(oi_tensor[Index(kOnyxPlaneOffset, Cell("c3"))], 1);
  SPIEL_CHECK_EQ(oi_tensor[Index(1, Cell("c3"))], 1);
  SPIEL_CHECK_EQ(PlaneSum(oi_tensor, kEmptyPlane), 49 - 1 - 3);
  // Terminal states have no player to move.
  auto won = MakeState({{"a4", "oiii"}, {"g7", "o"}});
  Play(won.get(), "a4-e4");
  const std::vector<float> terminal = Tensor(*won, kOnyx);
  SPIEL_CHECK_EQ(PlaneSum(terminal, kToMovePlaneOffset + kIvory), 0);
  SPIEL_CHECK_EQ(PlaneSum(terminal, kToMovePlaneOffset + kOnyx), 0);
  SPIEL_CHECK_EQ(terminal[Index(3, Cell("e4"))], 1);
}

// A complete game from the reference TypeScript engine (seeded random play).
// Each entry is a move and the number of legal moves before it was played.
void ReferenceEngineGameTest() {
  const std::vector<std::pair<std::string, int>> kGame = {
      {"g7xg6", 80}, {"f5xg5", 77}, {"d2xe2", 77}, {"e4xe3", 72},
      {"f2xf3", 72}, {"g2xg3", 69}, {"e1xe2", 68}, {"b5xb4", 69},
      {"c3xb3", 60}, {"d7xe7", 63}, {"e2xe5", 56}, {"d1xc1", 60},
      {"d6-d7", 49}, {"d3xe3", 60}, {"e5-e1", 48}};
  std::unique_ptr<State> state = LoadGame("altior")->NewInitialState();
  for (const auto& [move, num_legal] : kGame) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->LegalActions().size(), num_legal);
    Play(state.get(), move);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(AsAltior(*state).winner(), kIvory);
  SPIEL_CHECK_EQ(StackAt(*state, "e1"), "ioii");
}

}  // namespace
}  // namespace altior
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::altior::BasicAltiorTests();
  open_spiel::altior::InitialSetupTest();
  open_spiel::altior::MovementDistanceAndDirectionTest();
  open_spiel::altior::JumpingTest();
  open_spiel::altior::BoardEdgeAndCenterTest();
  open_spiel::altior::MergeOrderAndControlTest();
  open_spiel::altior::HeightLimitTest();
  open_spiel::altior::FourStackWinTest();
  open_spiel::altior::FalseWinTest();
  open_spiel::altior::ImmobilityTest();
  open_spiel::altior::ThreefoldRepetitionTest();
  open_spiel::altior::TranspositionTest();
  open_spiel::altior::CloneAndUndoPreserveRepetitionTest();
  open_spiel::altior::RepetitionFromStandardStartTest();
  open_spiel::altior::MoveLimitTest();
  open_spiel::altior::CloneIndependenceTest();
  open_spiel::altior::SerializationTest();
  open_spiel::altior::UndoTest();
  open_spiel::altior::ObservationTest();
  open_spiel::altior::ReferenceEngineGameTest();
}
