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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/games/fox_and_geese/fox_and_geese.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/status.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace fox_and_geese {
namespace {

namespace testing = open_spiel::testing;

int Cell(int row, int col) { return row * kNumCols + col; }

Action Move(int from_row, int from_col, int to_row, int to_col) {
  return Cell(from_row, from_col) * kNumCells + Cell(to_row, to_col);
}

bool Contains(const std::vector<Action>& actions, Action action) {
  return std::find(actions.begin(), actions.end(), action) != actions.end();
}

// Builds a state from a seven-row picture drawn with the symbols used by
// ToString(): 'f' fox, 'g' goose, '.' empty, ' ' off the cross.
std::unique_ptr<State> MakeState(const std::shared_ptr<const Game>& game,
                                 const std::vector<std::string>& rows,
                                 const std::string& current_player,
                                 int continue_jump_from = -1) {
  SPIEL_CHECK_EQ(static_cast<int>(rows.size()), kNumRows);
  FoxAndGeeseStateStruct state_struct;
  state_struct.board.reserve(kNumCells);
  for (const std::string& row : rows) {
    SPIEL_CHECK_EQ(static_cast<int>(row.size()), kNumCols);
    for (char symbol : row) {
      state_struct.board.push_back(std::string(1, symbol));
    }
  }
  state_struct.current_player = current_player;
  state_struct.continue_jump_from = continue_jump_from;
  return down_cast<const FoxAndGeeseGame*>(game.get())
      ->NewInitialState(state_struct);
}

void GameParametersTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");
  SPIEL_CHECK_EQ(game->NumDistinctActions(), kNumActions);
  SPIEL_CHECK_EQ(game->NumPlayers(), kNumPlayers);
  SPIEL_CHECK_EQ(game->MaxGameLength(), kMaxGameLength);
  SPIEL_CHECK_EQ(game->MinUtility(), -1.0);
  SPIEL_CHECK_EQ(game->MaxUtility(), 1.0);
  SPIEL_CHECK_TRUE(game->UtilitySum().has_value());
  SPIEL_CHECK_EQ(game->UtilitySum().value(), 0.0);

  const auto* fg_game = down_cast<const FoxAndGeeseGame*>(game.get());
  SPIEL_CHECK_EQ(fg_game->NumFoxes(), kDefaultNumFoxes);
  SPIEL_CHECK_EQ(fg_game->NumGeese(), kDefaultNumGeese);

  std::shared_ptr<const Game> game17 =
      LoadGame("fox_and_geese(num_geese=17)");
  SPIEL_CHECK_EQ(
      down_cast<const FoxAndGeeseGame*>(game17.get())->NumGeese(), 17);

  SPIEL_CHECK_TRUE(IsSupportedNumFoxes(1));
  SPIEL_CHECK_FALSE(IsSupportedNumFoxes(2));
  SPIEL_CHECK_TRUE(IsSupportedNumGeese(13));
  SPIEL_CHECK_TRUE(IsSupportedNumGeese(15));
  SPIEL_CHECK_TRUE(IsSupportedNumGeese(17));
  SPIEL_CHECK_FALSE(IsSupportedNumGeese(14));
  SPIEL_CHECK_TRUE(GeeseMayMoveBackward(13));
  SPIEL_CHECK_FALSE(GeeseMayMoveBackward(15));
  SPIEL_CHECK_FALSE(GeeseMayMoveBackward(17));
}

// The three traditional layouts are nested: 15 and 17 add geese along the
// fox's row without disturbing the 13-goose position.
void InitialStateTest() {
  const std::string kExpected13 =
      "  ...  \n"
      "  ...  \n"
      ".......\n"
      "...f...\n"
      "ggggggg\n"
      "  ggg  \n"
      "  ggg  ";
  const std::string kExpected15 =
      "  ...  \n"
      "  ...  \n"
      ".......\n"
      "g..f..g\n"
      "ggggggg\n"
      "  ggg  \n"
      "  ggg  ";
  const std::string kExpected17 =
      "  ...  \n"
      "  ...  \n"
      ".......\n"
      "gg.f.gg\n"
      "ggggggg\n"
      "  ggg  \n"
      "  ggg  ";

  const std::vector<std::pair<std::string, std::string>> configs = {
      {"fox_and_geese", kExpected13},
      {"fox_and_geese(num_geese=15)", kExpected15},
      {"fox_and_geese(num_geese=17)", kExpected17}};

  for (const auto& config : configs) {
    std::shared_ptr<const Game> game = LoadGame(config.first);
    std::unique_ptr<State> state = game->NewInitialState();
    SPIEL_CHECK_EQ(state->ToString(), config.second);
    // The geese move first.
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
    SPIEL_CHECK_FALSE(state->IsTerminal());

    const auto* fg_state = down_cast<const FoxAndGeeseState*>(state.get());
    SPIEL_CHECK_EQ(fg_state->BoardAt(kNumRows / 2, kNumCols / 2),
                   CellState::kFox);
    const int num_geese =
        down_cast<const FoxAndGeeseGame*>(game.get())->NumGeese();
    SPIEL_CHECK_EQ(fg_state->NumGeeseRemaining(), num_geese);

    int foxes = 0, geese = 0, empty = 0, out_of_bounds = 0;
    for (CellState cell : fg_state->Board()) {
      if (cell == CellState::kFox) ++foxes;
      if (cell == CellState::kGoose) ++geese;
      if (cell == CellState::kEmpty) ++empty;
      if (cell == CellState::kOutOfBounds) ++out_of_bounds;
    }
    SPIEL_CHECK_EQ(foxes, 1);
    SPIEL_CHECK_EQ(geese, num_geese);
    SPIEL_CHECK_EQ(out_of_bounds, kNumCells - kPlayableCells);
    SPIEL_CHECK_EQ(foxes + geese + empty, kPlayableCells);
  }

  // Opening move counts for the geese.
  SPIEL_CHECK_EQ(
      LoadGame("fox_and_geese")->NewInitialState()->LegalActions().size(), 10);
  SPIEL_CHECK_EQ(LoadGame("fox_and_geese(num_geese=15)")
                     ->NewInitialState()
                     ->LegalActions()
                     .size(),
                 12);
}

// The board's lines are all four orthogonals plus diagonals drawn only through
// the five points (1,3), (3,1), (3,3), (3,5) and (5,3). Isolating the fox with
// every neighbour empty and no jump available makes its legal move count equal
// to the degree of the point it stands on.
void BoardGeometryTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");

  // (3,3) is a diagonal point: four orthogonal plus four diagonal lines.
  std::unique_ptr<State> center = MakeState(game,
                                            {"  ...  ",
                                             "  ...  ",
                                             ".......",
                                             "...f...",
                                             ".......",
                                             "  g.g  ",
                                             "  ggg  "},
                                            "f");
  SPIEL_CHECK_EQ(center->LegalActions().size(), 8);
  SPIEL_CHECK_TRUE(Contains(center->LegalActions(), Move(3, 3, 2, 2)));
  SPIEL_CHECK_TRUE(Contains(center->LegalActions(), Move(3, 3, 4, 4)));

  // (0,2) has two orthogonal lines and one diagonal, the latter only because
  // (1,3) is a diagonal point.
  std::unique_ptr<State> corner = MakeState(game,
                                            {"  f..  ",
                                             "  ...  ",
                                             ".......",
                                             ".......",
                                             ".......",
                                             "  g.g  ",
                                             "  ggg  "},
                                            "f");
  SPIEL_CHECK_EQ(corner->LegalActions().size(), 3);
  SPIEL_CHECK_TRUE(Contains(corner->LegalActions(), Move(0, 2, 0, 3)));
  SPIEL_CHECK_TRUE(Contains(corner->LegalActions(), Move(0, 2, 1, 2)));
  SPIEL_CHECK_TRUE(Contains(corner->LegalActions(), Move(0, 2, 1, 3)));
  // (1,1) is off the cross, so no diagonal runs there.
  SPIEL_CHECK_FALSE(Contains(corner->LegalActions(), Move(0, 2, 1, 1)));
}

// Jumping is never compulsory, so steps stay legal alongside a capture.
void OptionalCaptureTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");
  std::unique_ptr<State> state = MakeState(game,
                                           {"  ...  ",
                                            "  ...  ",
                                            "..g....",
                                            "...fg..",
                                            "...gg..",
                                            "  gg.  ",
                                            "  ggg  "},
                                           "f");
  std::vector<Action> actions = state->LegalActions();
  SPIEL_CHECK_EQ(actions.size(), 5);
  // The capture over (3,4).
  SPIEL_CHECK_TRUE(Contains(actions, Move(3, 3, 3, 5)));
  // Ordinary steps remain available.
  SPIEL_CHECK_TRUE(Contains(actions, Move(3, 3, 2, 3)));
  SPIEL_CHECK_TRUE(Contains(actions, Move(3, 3, 3, 2)));
  SPIEL_CHECK_TRUE(std::is_sorted(actions.begin(), actions.end()));

  const auto* fg_state = down_cast<const FoxAndGeeseState*>(state.get());
  const int before = fg_state->NumGeeseRemaining();
  state->ApplyAction(Move(3, 3, 3, 5));
  fg_state = down_cast<const FoxAndGeeseState*>(state.get());
  SPIEL_CHECK_EQ(fg_state->NumGeeseRemaining(), before - 1);
  SPIEL_CHECK_EQ(fg_state->BoardAt(3, 4), CellState::kEmpty);
  SPIEL_CHECK_EQ(fg_state->BoardAt(3, 5), CellState::kFox);
  SPIEL_CHECK_EQ(fg_state->BoardAt(3, 3), CellState::kEmpty);
  // No further jump from (3,5), so the turn passes to the geese.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

// After a jump the fox keeps the move while another jump is available, and
// gives it up by playing kEndTurnAction.
void JumpChainTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");
  std::unique_ptr<State> state = MakeState(game,
                                           {"  ...  ",
                                            "  .f.  ",
                                            "...g...",
                                            ".......",
                                            "...g...",
                                            "  g.g  ",
                                            "  ggg  "},
                                           "f");
  SPIEL_CHECK_EQ(state->LegalActions().size(), 8);
  SPIEL_CHECK_FALSE(Contains(state->LegalActions(), kEndTurnAction));

  state->ApplyAction(Move(1, 3, 3, 3));
  // Still the fox's move, and the only choices are the continuation or a stop.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::vector<Action> mid = state->LegalActions();
  SPIEL_CHECK_EQ(mid.size(), 2);
  SPIEL_CHECK_TRUE(Contains(mid, Move(3, 3, 5, 3)));
  SPIEL_CHECK_TRUE(Contains(mid, kEndTurnAction));

  std::unique_ptr<State> stopped = state->Clone();
  stopped->ApplyAction(kEndTurnAction);
  SPIEL_CHECK_EQ(stopped->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(
      down_cast<const FoxAndGeeseState*>(stopped.get())->NumGeeseRemaining(),
      6);

  state->ApplyAction(Move(3, 3, 5, 3));
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(
      down_cast<const FoxAndGeeseState*>(state.get())->NumGeeseRemaining(), 5);
}

// The thirteen-goose game places no restriction on the geese. The fifteen- and
// seventeen-goose games forbid them from retreating.
void GeeseDirectionTest() {
  const std::vector<std::string> rows = {"  ...  ", "  ...  ", "...g...",
                                         ".......", "...f...", "  ggg  ",
                                         "  ...  "};

  std::unique_ptr<State> free_geese =
      MakeState(LoadGame("fox_and_geese"), rows, "g");
  SPIEL_CHECK_TRUE(Contains(free_geese->LegalActions(), Move(2, 3, 3, 3)));
  SPIEL_CHECK_TRUE(Contains(free_geese->LegalActions(), Move(2, 3, 1, 3)));
  SPIEL_CHECK_TRUE(Contains(free_geese->LegalActions(), Move(2, 3, 2, 2)));

  std::unique_ptr<State> restricted =
      MakeState(LoadGame("fox_and_geese(num_geese=15)"), rows, "g");
  // Backwards is away from the fox, i.e. towards the arm the geese started in.
  SPIEL_CHECK_FALSE(Contains(restricted->LegalActions(), Move(2, 3, 3, 3)));
  SPIEL_CHECK_TRUE(Contains(restricted->LegalActions(), Move(2, 3, 1, 3)));
  SPIEL_CHECK_TRUE(Contains(restricted->LegalActions(), Move(2, 3, 2, 2)));
}

void TerminalTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");

  // The fox stands on (0,2) with its three lines blocked and every landing
  // square behind them occupied, so it cannot move: the geese win.
  std::unique_ptr<State> trapped = MakeState(game,
                                             {"  fgg  ",
                                              "  ggg  ",
                                              "..g.g..",
                                              ".......",
                                              ".......",
                                              "  ...  ",
                                              "  ...  "},
                                             "f");
  SPIEL_CHECK_TRUE(trapped->IsTerminal());
  SPIEL_CHECK_TRUE(trapped->LegalActions().empty());
  SPIEL_CHECK_EQ(down_cast<const FoxAndGeeseState*>(trapped.get())->outcome(),
                 1);
  SPIEL_CHECK_EQ(trapped->Returns(), std::vector<double>({-1.0, 1.0}));

  // Fewer than kMinGeeseToTrapFox geese remain, so the fox has won.
  std::unique_ptr<State> attrition = MakeState(game,
                                               {"  ...  ",
                                                "  ...  ",
                                                ".......",
                                                "...f...",
                                                ".......",
                                                "  g.g  ",
                                                "  ..g  "},
                                               "f");
  SPIEL_CHECK_TRUE(attrition->IsTerminal());
  SPIEL_CHECK_EQ(
      down_cast<const FoxAndGeeseState*>(attrition.get())->NumGeeseRemaining(),
      kMinGeeseToTrapFox - 1);
  SPIEL_CHECK_EQ(down_cast<const FoxAndGeeseState*>(attrition.get())->outcome(),
                 0);
  SPIEL_CHECK_EQ(attrition->Returns(), std::vector<double>({1.0, -1.0}));
}

// Undo has to restore the moved piece, any captured goose, and the chain.
void UndoTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");
  std::unique_ptr<State> state = MakeState(game,
                                           {"  ...  ",
                                            "  .f.  ",
                                            "...g...",
                                            ".......",
                                            "...g...",
                                            "  g.g  ",
                                            "  ggg  "},
                                           "f");
  const std::string before = state->ToString();
  const std::vector<Action> legal_before = state->LegalActions();
  const int geese_before =
      down_cast<const FoxAndGeeseState*>(state.get())->NumGeeseRemaining();

  const Player player = state->CurrentPlayer();
  const Action jump = Move(1, 3, 3, 3);
  state->ApplyAction(jump);
  SPIEL_CHECK_NE(state->ToString(), before);

  state->UndoAction(player, jump);
  SPIEL_CHECK_EQ(state->ToString(), before);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), player);
  SPIEL_CHECK_EQ(state->LegalActions(), legal_before);
  SPIEL_CHECK_EQ(
      down_cast<const FoxAndGeeseState*>(state.get())->NumGeeseRemaining(),
      geese_before);
  SPIEL_CHECK_TRUE(state->History().empty());

  // Undoing a plain step from the opening position.
  std::unique_ptr<State> opening = game->NewInitialState();
  const std::string opening_str = opening->ToString();
  const Action step = opening->LegalActions().front();
  const Player mover = opening->CurrentPlayer();
  opening->ApplyAction(step);
  opening->UndoAction(mover, step);
  SPIEL_CHECK_EQ(opening->ToString(), opening_str);
  SPIEL_CHECK_EQ(opening->CurrentPlayer(), mover);
}

void SerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");
  const auto* fg_game = down_cast<const FoxAndGeeseGame*>(game.get());

  std::unique_ptr<State> state = MakeState(game,
                                           {"  ...  ",
                                            "  .f.  ",
                                            "...g...",
                                            ".......",
                                            "...g...",
                                            "  g.g  ",
                                            "  ggg  "},
                                           "f");
  // Part way through a jump chain, so continue_jump_from is not -1.
  state->ApplyAction(Move(1, 3, 3, 3));
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  std::unique_ptr<StateStruct> as_struct = state->ToStruct();
  const auto* typed = down_cast<FoxAndGeeseStateStruct*>(as_struct.get());
  SPIEL_CHECK_EQ(typed->current_player, "f");
  SPIEL_CHECK_EQ(typed->continue_jump_from, Cell(3, 3));

  std::unique_ptr<State> restored = fg_game->NewInitialState(*typed);
  SPIEL_CHECK_EQ(restored->ToString(), state->ToString());
  SPIEL_CHECK_EQ(restored->CurrentPlayer(), state->CurrentPlayer());
  SPIEL_CHECK_EQ(restored->LegalActions(), state->LegalActions());
  SPIEL_CHECK_TRUE(Contains(restored->LegalActions(), kEndTurnAction));

  // A state that is not mid-chain round trips with continue_jump_from unset.
  std::unique_ptr<State> opening = game->NewInitialState();
  std::unique_ptr<StateStruct> opening_struct = opening->ToStruct();
  const auto* opening_typed =
      down_cast<FoxAndGeeseStateStruct*>(opening_struct.get());
  SPIEL_CHECK_EQ(opening_typed->current_player, "g");
  SPIEL_CHECK_EQ(opening_typed->continue_jump_from, -1);
  std::unique_ptr<State> opening_restored =
      fg_game->NewInitialState(*opening_typed);
  SPIEL_CHECK_EQ(opening_restored->ToString(), opening->ToString());
  SPIEL_CHECK_EQ(opening_restored->LegalActions(), opening->LegalActions());
}

// Mirrors TestStateStruct in tic_tac_toe_test.cc. The board is a flat array of
// all 49 grid slots, including the 16 that lie off the cross.
void TestStateStruct() {
  auto game = LoadGame("fox_and_geese");
  auto state = game->NewInitialState();
  auto* fg_state = down_cast<FoxAndGeeseState*>(state.get());
  auto state_struct = fg_state->ToStruct();
  SPIEL_CHECK_EQ(state_struct->ToJson(), fg_state->ToJson());
  std::string state_json =
      "{\"board\":["
      "\" \",\" \",\".\",\".\",\".\",\" \",\" \","
      "\" \",\" \",\".\",\".\",\".\",\" \",\" \","
      "\".\",\".\",\".\",\".\",\".\",\".\",\".\","
      "\".\",\".\",\".\",\"f\",\".\",\".\",\".\","
      "\"g\",\"g\",\"g\",\"g\",\"g\",\"g\",\"g\","
      "\" \",\" \",\"g\",\"g\",\"g\",\" \",\" \","
      "\" \",\" \",\"g\",\"g\",\"g\",\" \",\" \""
      "],\"continue_jump_from\":-1,\"current_player\":\"g\"}";
  SPIEL_CHECK_EQ(state_struct->ToJson(), state_json);
  SPIEL_CHECK_EQ(nlohmann::json::parse(state_json).dump(),
                 FoxAndGeeseStateStruct(state_json).ToJson());
}

// The game is perfect information, so each player observes the whole state.
void TestObservationStruct() {
  auto game = LoadGame("fox_and_geese");
  auto state = game->NewInitialState();
  state->ApplyAction(Move(4, 0, 3, 0));  // A goose advances on the left.
  auto* fg_state = down_cast<FoxAndGeeseState*>(state.get());
  for (Player player = 0; player < kNumPlayers; ++player) {
    auto obs_struct = fg_state->ToObservationStruct(player);
    SPIEL_CHECK_EQ(obs_struct->ToJson(), fg_state->ToJson());
    SPIEL_CHECK_EQ(
        nlohmann::json::parse(obs_struct->ToJson()).dump(),
        FoxAndGeeseObservationStruct(obs_struct->ToJson()).ToJson());
  }
}

// Mirrors TestActionStruct in tic_tac_toe_test.cc.
void TestActionStruct() {
  auto game = LoadGame("fox_and_geese");
  auto state = game->NewInitialState();
  auto* fg_state = down_cast<FoxAndGeeseState*>(state.get());

  // A goose steps from (4,0) to (3,0), one of the ten opening moves.
  Action action_id = Move(4, 0, 3, 0);
  auto action_struct = fg_state->ActionToStruct(1, action_id);
  std::string action_json =
      "{\"end_turn\":false,\"from_col\":0,\"from_row\":4,\"to_col\":0,"
      "\"to_row\":3}";
  SPIEL_CHECK_EQ(action_struct->ToJson(), action_json);

  // Test ApplyActionStruct.
  auto state2 = game->NewInitialState();
  Status status = state2->ApplyActionStruct(*action_struct);
  SPIEL_CHECK_TRUE(status.ok());
  SPIEL_CHECK_EQ(state2->ToString(),
                 "  ...  \n"
                 "  ...  \n"
                 ".......\n"
                 "g..f...\n"
                 ".gggggg\n"
                 "  ggg  \n"
                 "  ggg  ");

  // Test ValidateActionStruct with a valid action.
  auto state3 = game->NewInitialState();
  SPIEL_CHECK_TRUE(state3->ValidateActionStruct(*action_struct).ok());

  // Test ValidateActionStruct with an invalid action: (0,2) holds no piece.
  auto empty_origin = fg_state->ActionToStruct(1, Move(0, 2, 0, 3));
  SPIEL_CHECK_FALSE(state3->ValidateActionStruct(*empty_origin).ok());

  // Test JSON parsing.
  SPIEL_CHECK_EQ(nlohmann::json::parse(action_json).dump(),
                 FoxAndGeeseActionStruct(action_json).ToJson());

  // Test StructToActions.
  std::vector<Action> expected_actions = {action_id};
  SPIEL_CHECK_EQ(expected_actions, fg_state->StructToActions(*action_struct));

  // The reserved end-of-turn action carries no coordinates.
  auto end_struct = fg_state->ActionToStruct(0, kEndTurnAction);
  auto* end_typed = down_cast<FoxAndGeeseActionStruct*>(end_struct.get());
  SPIEL_CHECK_TRUE(end_typed->end_turn);
  SPIEL_CHECK_EQ(fg_state->StructToActions(*end_typed),
                 std::vector<Action>({kEndTurnAction}));
}

// Captures are rendered distinctly from steps so that playthroughs are
// readable.
void ActionToStringTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");
  std::unique_ptr<State> state = MakeState(game,
                                           {"  ...  ",
                                            "  ...  ",
                                            "..g....",
                                            "...fg..",
                                            "...gg..",
                                            "  gg.  ",
                                            "  ggg  "},
                                           "f");
  const std::string capture = state->ActionToString(0, Move(3, 3, 3, 5));
  const std::string step = state->ActionToString(0, Move(3, 3, 3, 2));
  SPIEL_CHECK_NE(capture, step);
  SPIEL_CHECK_EQ(capture.back(), 'x');
  SPIEL_CHECK_NE(step.back(), 'x');
  SPIEL_CHECK_EQ(state->ActionToString(0, kEndTurnAction), "end turn");
}

void ObservationTensorTest() {
  std::shared_ptr<const Game> game = LoadGame("fox_and_geese");
  SPIEL_CHECK_EQ(game->ObservationTensorShape(),
                 std::vector<int>({kCellStates, kNumRows, kNumCols}));
  SPIEL_CHECK_EQ(game->ObservationTensorSize(),
                 kCellStates * kNumRows * kNumCols);

  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<float> values(game->ObservationTensorSize());
  state->ObservationTensor(0, absl::MakeSpan(values));
  // Exactly one plane is set per point of the grid.
  float total = 0.0;
  for (float value : values) total += value;
  SPIEL_CHECK_EQ(total, static_cast<float>(kNumCells));
}

void BasicFoxAndGeeseTests() {
  testing::LoadGameTest("fox_and_geese");
  testing::NoChanceOutcomesTest(*LoadGame("fox_and_geese"));
  testing::RandomSimTest(*LoadGame("fox_and_geese"), 20);
  testing::RandomSimTest(*LoadGame("fox_and_geese(num_geese=15)"), 20);
  testing::RandomSimTest(*LoadGame("fox_and_geese(num_geese=17)"), 20);
  testing::RandomSimTestWithUndo(*LoadGame("fox_and_geese"), 10);
}

}  // namespace
}  // namespace fox_and_geese
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::fox_and_geese::BasicFoxAndGeeseTests();
  open_spiel::fox_and_geese::GameParametersTest();
  open_spiel::fox_and_geese::InitialStateTest();
  open_spiel::fox_and_geese::BoardGeometryTest();
  open_spiel::fox_and_geese::OptionalCaptureTest();
  open_spiel::fox_and_geese::JumpChainTest();
  open_spiel::fox_and_geese::GeeseDirectionTest();
  open_spiel::fox_and_geese::TerminalTest();
  open_spiel::fox_and_geese::UndoTest();
  open_spiel::fox_and_geese::SerializationTest();
  open_spiel::fox_and_geese::TestStateStruct();
  open_spiel::fox_and_geese::TestObservationStruct();
  open_spiel::fox_and_geese::TestActionStruct();
  open_spiel::fox_and_geese::ActionToStringTest();
  open_spiel::fox_and_geese::ObservationTensorTest();
}
