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

#include "open_spiel/games/siguo_junqi/siguo_junqi.h"

#include <algorithm>
#include <array>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace siguo_junqi {
namespace {

namespace testing = open_spiel::testing;

using Board = std::array<Piece, kNumNodes>;

void Put(Board* board, int node, Player owner, PieceType type) {
  (*board)[node] = {owner, type};
}

// A board where every player has a flag in their left headquarters.
Board BoardWithFlags() {
  Board board;
  for (Player p = 0; p < kNumPlayers; ++p) {
    Put(&board, ZoneNode(p, 1, 2), p, kFlag);
  }
  return board;
}

std::unique_ptr<SiguoJunqiState> StateWithBoard(const Game& game,
                                                const Board& board,
                                                Player player) {
  std::unique_ptr<State> state = game.NewInitialState();
  auto* junqi_state = static_cast<SiguoJunqiState*>(state.release());
  junqi_state->SetBoardForTesting(board, player);
  return std::unique_ptr<SiguoJunqiState>(junqi_state);
}

std::vector<std::string> DestinationNames(const SiguoJunqiState& state,
                                          int node) {
  std::vector<std::string> names;
  for (int destination : state.Destinations(node)) {
    names.push_back(NodeName(destination));
  }
  std::sort(names.begin(), names.end());
  return names;
}

Action MoveAction(int from, int to) {
  return kNumDeployActions + from * kNumNodes + to;
}

void BasicSiguoJunqiTests() {
  testing::LoadGameTest("siguo_junqi");
  testing::NoChanceOutcomesTest(*LoadGame("siguo_junqi"));
  testing::RandomSimTest(*LoadGame("siguo_junqi"), 20);
  testing::RandomSimTest(
      *LoadGame("siguo_junqi(information_mode=partner_visible)"), 5);
  testing::RandomSimTest(
      *LoadGame("siguo_junqi(information_mode=all_visible)"), 5);
  testing::RandomSimTest(
      *LoadGame("siguo_junqi(flying_bombs=true,mine_survives_attack=false,"
                "marshal_death_reveals_flag=false,no_collision_draw_plies=0)"),
      5);
}

void TopologyTest() {
  int camps = 0;
  int headquarters = 0;
  int rail = 0;
  for (int node = 0; node < kNumNodes; ++node) {
    camps += NodeTypeOf(node) == NodeType::kCamp;
    headquarters += NodeTypeOf(node) == NodeType::kHeadquarters;
    rail += IsRailNode(node);
  }
  SPIEL_CHECK_EQ(camps, 4 * 5);
  SPIEL_CHECK_EQ(headquarters, 4 * 2);
  SPIEL_CHECK_EQ(rail, 4 * 16 + 9);
  SPIEL_CHECK_EQ(NodeName(ZoneNode(0, 1, 2)), "S12");
  SPIEL_CHECK_EQ(NodeName(HubNode(2, 2)), "C22");
  SPIEL_CHECK_EQ(ZoneOf(ZoneNode(3, 6, 5)), 3);
  SPIEL_CHECK_EQ(ZoneOf(HubNode(1, 1)), kInvalidPlayer);
}

void BattleTest() {
  SPIEL_CHECK_TRUE(ResolveBattle(kGeneral, kColonel, true) ==
                   BattleResult::kAttackerWins);
  SPIEL_CHECK_TRUE(ResolveBattle(kColonel, kGeneral, true) ==
                   BattleResult::kDefenderWins);
  SPIEL_CHECK_TRUE(ResolveBattle(kMajor, kMajor, true) ==
                   BattleResult::kBothRemoved);
  SPIEL_CHECK_TRUE(ResolveBattle(kBomb, kFieldMarshal, true) ==
                   BattleResult::kBothRemoved);
  SPIEL_CHECK_TRUE(ResolveBattle(kEngineer, kBomb, true) ==
                   BattleResult::kBothRemoved);
  SPIEL_CHECK_TRUE(ResolveBattle(kBomb, kLandmine, true) ==
                   BattleResult::kBothRemoved);
  SPIEL_CHECK_TRUE(ResolveBattle(kEngineer, kLandmine, true) ==
                   BattleResult::kAttackerWins);
  SPIEL_CHECK_TRUE(ResolveBattle(kFieldMarshal, kLandmine, true) ==
                   BattleResult::kDefenderWins);
  SPIEL_CHECK_TRUE(ResolveBattle(kFieldMarshal, kLandmine, false) ==
                   BattleResult::kBothRemoved);
  SPIEL_CHECK_TRUE(ResolveBattle(kEngineer, kFlag, true) ==
                   BattleResult::kAttackerWins);
  SPIEL_CHECK_TRUE(ResolveBattle(kBomb, kFlag, true) ==
                   BattleResult::kBothRemoved);
}

void RailMovementTest() {
  std::shared_ptr<const Game> game = LoadGame("siguo_junqi");
  int start = ZoneNode(0, 5, 1);

  // A non-engineer slides straight along the railway, through the hub into
  // North's column, or around the arc into West's column, but never turns.
  Board board = BoardWithFlags();
  Put(&board, start, 0, kColonel);
  auto state = StateWithBoard(*game, board, 0);
  std::vector<std::string> expected = {
      "C11", "C21", "C31", "N25", "N35", "N45", "N55", "N65", "S21",
      "S31", "S41", "S52", "S61", "W25", "W35", "W45", "W55", "W65"};
  std::sort(expected.begin(), expected.end());
  SPIEL_CHECK_EQ(DestinationNames(*state, start), expected);

  // A piece on the hub corner stops the straight line but not the arc, which
  // bypasses the corner point.
  Put(&board, HubNode(1, 1), 1, kMajor);
  state = StateWithBoard(*game, board, 0);
  expected = {"C11", "S21", "S31", "S41", "S52", "S61",
              "W25", "W35", "W45", "W55", "W65"};
  std::sort(expected.begin(), expected.end());
  SPIEL_CHECK_EQ(DestinationNames(*state, start), expected);

  // An ally in the way blocks without being attackable.
  Put(&board, HubNode(1, 1), 2, kMajor);
  state = StateWithBoard(*game, board, 0);
  expected = {"S21", "S31", "S41", "S52", "S61",
              "W25", "W35", "W45", "W55", "W65"};
  std::sort(expected.begin(), expected.end());
  SPIEL_CHECK_EQ(DestinationNames(*state, start), expected);

  // An engineer turns freely and reaches every railway point plus the camp.
  board = BoardWithFlags();
  Put(&board, start, 0, kEngineer);
  state = StateWithBoard(*game, board, 0);
  SPIEL_CHECK_EQ(state->Destinations(start).size(), 4 * 16 + 9 - 1 + 1);

  // Bombs only turn with flying_bombs.
  board = BoardWithFlags();
  Put(&board, start, 0, kBomb);
  state = StateWithBoard(*game, board, 0);
  SPIEL_CHECK_EQ(state->Destinations(start).size(), 18);
  std::shared_ptr<const Game> flying =
      LoadGame("siguo_junqi(flying_bombs=true)");
  state = StateWithBoard(*flying, board, 0);
  SPIEL_CHECK_EQ(state->Destinations(start).size(), 4 * 16 + 9);
}

void RoadMovementTest() {
  std::shared_ptr<const Game> game = LoadGame("siguo_junqi");
  // Camps have diagonal roads.
  Board board = BoardWithFlags();
  int camp = ZoneNode(0, 4, 3);
  Put(&board, camp, 0, kCaptain);
  auto state = StateWithBoard(*game, board, 0);
  std::vector<std::string> expected = {"S32", "S33", "S34", "S42",
                                       "S44", "S52", "S53", "S54"};
  SPIEL_CHECK_EQ(DestinationNames(*state, camp), expected);

  // Pieces in camps cannot be attacked; occupied camps cannot be entered.
  Put(&board, ZoneNode(0, 3, 2), 1, kLieutenant);
  Put(&board, ZoneNode(0, 3, 3), 1, kLieutenant);
  state = StateWithBoard(*game, board, 0);
  expected = {"S33", "S34", "S42", "S44", "S52", "S53", "S54"};
  SPIEL_CHECK_EQ(DestinationNames(*state, camp), expected);

  // Pieces in a headquarters, landmines and flags never move.
  board = BoardWithFlags();
  Put(&board, ZoneNode(0, 1, 4), 0, kGeneral);
  Put(&board, ZoneNode(0, 2, 3), 0, kLandmine);
  state = StateWithBoard(*game, board, 0);
  SPIEL_CHECK_TRUE(state->Destinations(ZoneNode(0, 1, 4)).empty());
  SPIEL_CHECK_TRUE(state->Destinations(ZoneNode(0, 2, 3)).empty());
  SPIEL_CHECK_TRUE(state->Destinations(ZoneNode(0, 1, 2)).empty());
}

void EliminationTest() {
  std::shared_ptr<const Game> game = LoadGame("siguo_junqi");
  Board board = BoardWithFlags();
  Put(&board, ZoneNode(1, 2, 2), 0, kEngineer);
  Put(&board, ZoneNode(0, 6, 3), 0, kColonel);
  Put(&board, ZoneNode(1, 6, 4), 1, kLieutenant);
  Put(&board, ZoneNode(2, 6, 4), 2, kLieutenant);
  // West has only a flag, so it cannot move.
  auto state = StateWithBoard(*game, board, 0);

  // South captures East's flag: East is eliminated and its pieces removed.
  state->ApplyAction(MoveAction(ZoneNode(1, 2, 2), ZoneNode(1, 1, 2)));
  SPIEL_CHECK_TRUE(state->IsEliminated(1));
  SPIEL_CHECK_TRUE(state->PieceAt(ZoneNode(1, 6, 4)).empty());
  SPIEL_CHECK_EQ(state->PieceAt(ZoneNode(1, 1, 2)).owner, 0);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);

  // After North moves, West has no legal move and is eliminated, so team 0
  // wins.
  state->ApplyAction(MoveAction(ZoneNode(2, 6, 4), ZoneNode(2, 6, 3)));
  SPIEL_CHECK_TRUE(state->IsEliminated(3));
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>({1, -1, 1, -1}));
}

void FlagRevealTest() {
  std::shared_ptr<const Game> game = LoadGame("siguo_junqi");
  Board board = BoardWithFlags();
  for (Player p = 0; p < kNumPlayers; ++p) {
    Put(&board, ZoneNode(p, 6, 3), p, kCaptain);
  }
  Put(&board, HubNode(1, 1), 0, kFieldMarshal);
  Put(&board, HubNode(1, 2), 1, kBomb);
  auto state = StateWithBoard(*game, board, 1);
  SPIEL_CHECK_EQ(state->ObservationString(1).find("SF"), std::string::npos);
  state->ApplyAction(MoveAction(HubNode(1, 2), HubNode(1, 1)));
  SPIEL_CHECK_TRUE(state->PieceAt(HubNode(1, 1)).empty());
  SPIEL_CHECK_TRUE(state->FlagRevealed(0));
  SPIEL_CHECK_FALSE(state->FlagRevealed(1));
  SPIEL_CHECK_EQ(state->MovesSinceCollision(), 0);
  SPIEL_CHECK_NE(state->ObservationString(1).find("SF"), std::string::npos);
  SPIEL_CHECK_NE(
      state->InformationStateString(3).find("S flag revealed at S12"),
      std::string::npos);
}

void NoCollisionDrawTest() {
  std::shared_ptr<const Game> game =
      LoadGame("siguo_junqi(no_collision_draw_plies=2)");
  Board board = BoardWithFlags();
  for (Player p = 0; p < kNumPlayers; ++p) {
    Put(&board, ZoneNode(p, 6, 3), p, kCaptain);
  }
  auto state = StateWithBoard(*game, board, 0);
  state->ApplyAction(MoveAction(ZoneNode(0, 6, 3), ZoneNode(0, 5, 3)));
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(MoveAction(ZoneNode(1, 6, 3), ZoneNode(1, 5, 3)));
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>(kNumPlayers, 0.0));
}

void DeploymentTest() {
  std::shared_ptr<const Game> game = LoadGame("siguo_junqi");
  std::mt19937 rng(1234);
  for (int sim = 0; sim < 50; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    for (int i = 0; i < kNumPlayers * kPiecesPerPlayer; ++i) {
      std::vector<Action> actions = state->LegalActions();
      SPIEL_CHECK_FALSE(actions.empty());
      std::uniform_int_distribution<int> dist(0, actions.size() - 1);
      state->ApplyAction(actions[dist(rng)]);
    }
    const auto& junqi_state = static_cast<const SiguoJunqiState&>(*state);
    SPIEL_CHECK_FALSE(junqi_state.IsDeploymentPhase());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
    for (Player p = 0; p < kNumPlayers; ++p) {
      std::array<int, kNumPieceTypes> counts = {};
      for (int row = 1; row <= kZoneRows; ++row) {
        for (int col = 1; col <= kZoneCols; ++col) {
          int node = ZoneNode(p, row, col);
          const Piece& piece = junqi_state.PieceAt(node);
          if (NodeTypeOf(node) == NodeType::kCamp) {
            SPIEL_CHECK_TRUE(piece.empty());
            continue;
          }
          SPIEL_CHECK_EQ(piece.owner, p);
          ++counts[piece.type];
          if (piece.type == kFlag) {
            SPIEL_CHECK_TRUE(NodeTypeOf(node) == NodeType::kHeadquarters);
          }
          if (piece.type == kLandmine) SPIEL_CHECK_LE(row, 2);
          if (piece.type == kBomb) SPIEL_CHECK_LT(row, kZoneRows);
        }
      }
      for (int type = 0; type < kNumPieceTypes; ++type) {
        SPIEL_CHECK_EQ(counts[type], kPieceCounts[type]);
      }
    }
  }
}

// Deploys every player's pieces with the first legal action, except player 0
// who uses the last one.
std::unique_ptr<State> DeployedState(const Game& game, bool vary_player0) {
  std::unique_ptr<State> state = game.NewInitialState();
  while (static_cast<const SiguoJunqiState&>(*state).IsDeploymentPhase()) {
    std::vector<Action> actions = state->LegalActions();
    bool last = vary_player0 && state->CurrentPlayer() == 0;
    state->ApplyAction(last ? actions.back() : actions.front());
  }
  return state;
}

void InformationStateTest() {
  for (std::string mode : {"hidden", "partner_visible"}) {
    std::shared_ptr<const Game> game =
        LoadGame("siguo_junqi", {{"information_mode", GameParameter(mode)}});
    std::unique_ptr<State> a = DeployedState(*game, false);
    std::unique_ptr<State> b = DeployedState(*game, true);
    SPIEL_CHECK_NE(a->InformationStateString(0), b->InformationStateString(0));
    SPIEL_CHECK_EQ(a->InformationStateString(1), b->InformationStateString(1));
    SPIEL_CHECK_EQ(a->InformationStateString(3), b->InformationStateString(3));
    SPIEL_CHECK_EQ(a->ObservationTensor(1), b->ObservationTensor(1));
    if (mode == "hidden") {
      SPIEL_CHECK_EQ(a->InformationStateString(2),
                     b->InformationStateString(2));
      SPIEL_CHECK_EQ(a->ObservationString(2), b->ObservationString(2));
    } else {
      SPIEL_CHECK_NE(a->InformationStateString(2),
                     b->InformationStateString(2));
      SPIEL_CHECK_NE(a->ObservationString(2), b->ObservationString(2));
    }
  }
}

void StructTest() {
  std::shared_ptr<const Game> game = LoadGame("siguo_junqi");
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action action : state->LegalActions()) {
    SPIEL_CHECK_EQ(state->StructToActions(*state->ActionToStruct(action)),
                   std::vector<Action>{action});
  }
  state = DeployedState(*game, false);
  for (Action action : state->LegalActions()) {
    SPIEL_CHECK_EQ(state->StructToActions(*state->ActionToStruct(action)),
                   std::vector<Action>{action});
  }
  auto state_struct = state->ToStruct();
  const auto& contents =
      static_cast<const SiguoJunqiStateStruct&>(*state_struct);
  SPIEL_CHECK_EQ(contents.phase, "play");
  SPIEL_CHECK_EQ(contents.board.size(), kNumPlayers * kPiecesPerPlayer);
  SPIEL_CHECK_NE(contents.board.at("S12"), "S:?");
  auto observation = state->ToObservationStruct(1);
  const auto& observed =
      static_cast<const SiguoJunqiObservationStruct&>(*observation);
  SPIEL_CHECK_EQ(observed.board.at("S12"), "S:?");
  SPIEL_CHECK_NE(observed.board.at("E12"), "E:?");
  SiguoJunqiStateStruct round_trip(state_struct->ToJson());
  SPIEL_CHECK_TRUE(round_trip.board == contents.board);
}

}  // namespace
}  // namespace siguo_junqi
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::siguo_junqi::TopologyTest();
  open_spiel::siguo_junqi::BattleTest();
  open_spiel::siguo_junqi::RailMovementTest();
  open_spiel::siguo_junqi::RoadMovementTest();
  open_spiel::siguo_junqi::EliminationTest();
  open_spiel::siguo_junqi::FlagRevealTest();
  open_spiel::siguo_junqi::NoCollisionDrawTest();
  open_spiel::siguo_junqi::DeploymentTest();
  open_spiel::siguo_junqi::InformationStateTest();
  open_spiel::siguo_junqi::StructTest();
  open_spiel::siguo_junqi::BasicSiguoJunqiTests();
}
