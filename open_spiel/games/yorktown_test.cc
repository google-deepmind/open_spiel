// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/yorktown.h"

#include <memory>
#include <string>

#include "open_spiel/games/yorktown/yorktown_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace yorktown {
namespace {

namespace testing = open_spiel::testing;

void BasicTests() {
  testing::LoadGameTest("yorktown");
  testing::NoChanceOutcomesTest(*LoadGame("yorktown"));
  testing::RandomSimTest(*LoadGame("yorktown"), 1);
    testing::RandomSimTestWithUndo(*LoadGame("yorktown"), 1);
}

void testTesting(){
  std::shared_ptr<const Game> game = LoadGame("yorktown");
  std::cout << "Number of Players " << game->NumPlayers() << std::endl;
  YorktownState state(
      game, "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHaa__aa__aaaa__aa__aaSTPQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 20"); 
  std::cout << "State to String "<< state.ToString() << std::endl;
  std::cout << "IsTerminal "<<state.IsTerminal() << std::endl;
  std::cout << "Current Player "<< state.CurrentPlayer() << std::endl;
  std::cout << "InformationStateString Red " << state.InformationStateString(0) << std::endl;
  std::cout << "InformationStateString Blue " << state.InformationStateString(1) << std::endl;
  std::cout << state.Board().DebugString() << std::endl;
  
}

void MoveGenerationTests() {
  std::shared_ptr<const Game> game = LoadGame("yorktown");
  YorktownState state(
      game, "febMBEFEEfbgiBHIBEDBGjdddHCGJGDHDLIFKDDHaa__aa__aaaa__aa__aastpQNSQPTSUPWPVRPXPURnqONNQSNVPtNQRRTYup r 20"); 
  int legalMoves = state.LegalActions().size();
  
  std::cout << state.Board().DebugString() << std::endl;
  std::cout << "How many legal actions " <<state.LegalActions().size() << std::endl;
  for(Action a : state.LegalActions()){
    std::cout << state.ActionToString(0, a) << std::endl;
  }
  
  SPIEL_CHECK_EQ(legalMoves, 10);
}

void CheckUndo(const std::string strados, const std::string LAN, const std::string afterMove) {
  std::cout << "Check Undo" << std::endl; 
  std::shared_ptr<const Game> game = LoadGame("yorktown");
  YorktownState state(game, strados);
  Player player = state.CurrentPlayer();
  absl::optional<Move> maybe_move = state.Board().ParseLANMove(LAN);
  SPIEL_CHECK_TRUE(maybe_move);
  Action action = MoveToAction(*maybe_move);
  state.ApplyAction(action);
  SPIEL_CHECK_EQ(state.Board().ToStraDos3(), afterMove);
  state.UndoAction(player, action);
  SPIEL_CHECK_EQ(state.Board().ToStraDos3(), strados);
}


void UndoTests() {
  // Move
  CheckUndo("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHaa__aa__aaaa__aa__aaSTQQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0",
  "a4a5",
  "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGaHDLIFKDDHDa__aa__aaaa__aa__aaSTQQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP b 1");

  // Capture
  CheckUndo("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGaHDLIFKDDHDa__aa__aaSa__aa__aaaTQQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 2",
            "i4i7",
            "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGaHDLIFKDaHDa__aa__aaSa__aa__aaaTQQNSQPtSUPWPVRPXPURNQONNQSNVPTNQRRTYUP b 3");

  // Capture a knwon piece
  CheckUndo("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGaHDLIFKDDHDa__aa__aaSa__aa__aaaTQQNSQPtSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 2",
            "i4i7",
            "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGaHDLIFKDaHDa__aa__aaSa__aa__aaaTQQNSQPtSUPWPVRPXPURNQONNQSNVPTNQRRTYUP b 3");
}

void TerminalReturnTests() {
  std::cout << "Test TerminalStates" << std::endl;
  std::shared_ptr<const Game> game = LoadGame("yorktown");
  YorktownState state_without_flag(
      game, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa r 20");
  SPIEL_CHECK_EQ(state_without_flag.IsTerminal(), true);
  SPIEL_CHECK_EQ(state_without_flag.Returns(), (std::vector<double>{-1.0, 1.0}));

  YorktownState state_without_flag2(
      game, "MQaaaaaaaaaLaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaY b 20");
  
  std::cout << state_without_flag2.Board().DebugString() << std::endl;
  
  SPIEL_CHECK_EQ(state_without_flag2.IsTerminal(), false);
  
  absl::optional<Move> maybe_move = state_without_flag2.Board().ParseLANMove("b1a1");
  SPIEL_CHECK_TRUE(maybe_move);
  Action action = MoveToAction(*maybe_move);
  state_without_flag2.ApplyAction(action);

  std::cout << state_without_flag2.Board().DebugString() << std::endl;
  
  SPIEL_CHECK_EQ(state_without_flag2.IsTerminal(), true);
  SPIEL_CHECK_EQ(state_without_flag2.Returns(), (std::vector<double>{-1.0, 1.0}));

  YorktownState state_without_flag4(
      game, "YLaaaaaaaaaQaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaM r 20");
  
  std::cout << state_without_flag4.Board().DebugString() << std::endl;
  
  SPIEL_CHECK_EQ(state_without_flag4.IsTerminal(), false);
  
  maybe_move = state_without_flag4.Board().ParseLANMove("b1a1");
  SPIEL_CHECK_TRUE(maybe_move);
  action = MoveToAction(*maybe_move);
  state_without_flag4.ApplyAction(action);
  
  std::cout << state_without_flag4.Board().DebugString() << std::endl;
  
  SPIEL_CHECK_EQ(state_without_flag4.IsTerminal(), true);
  SPIEL_CHECK_EQ(state_without_flag4.Returns(), (std::vector<double>{1.0, -1.0}));


  YorktownState state_without_flag3(
      game, "MaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaO b 20");
  SPIEL_CHECK_EQ(state_without_flag3.IsTerminal(), true);
  SPIEL_CHECK_EQ(state_without_flag3.Returns(), (std::vector<double>{1.0, -1.0}));

  YorktownState no_moving_pieces(game, "MBLaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaNY b 20");
  SPIEL_CHECK_EQ(no_moving_pieces.IsTerminal(), true);
  SPIEL_CHECK_EQ(no_moving_pieces.Returns(), (std::vector<double>{1.0, -1.0}));
  
}

float ValueAt(const std::vector<float>& v, const std::vector<int>& shape,
              int plane, int x, int y) {
  return v[plane * shape[1] * shape[2] + y * shape[2] + x];
}

float ValueAt(const std::vector<float>& v, const std::vector<int>& shape,
              int plane, const std::string& square) {
  Square sq = *SquareFromString(square);
  return ValueAt(v, shape, plane, sq.x, sq.y);
}

void InformationStateTensorTestsWithProbability() {
  std::shared_ptr<const Game> game = LoadGame("yorktown");
  YorktownState state(
      game, "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHaa__aa__aaaa__aa__aaSTPQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 20"); 
  
  std::cout << state.Board().DebugString() << std::endl;
  
  auto shape = game->InformationStateTensorShape();
  std::vector<float> v(game->InformationStateTensorSize());
  state.InformationStateTensor(state.CurrentPlayer(),
                                  absl::MakeSpan(v));


  // For each piece type, check one square that's supposed to be occupied, and
  // one that isn't.
  std::cout << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
  for(auto k = 0; k < shape[0]; ++k){
    std::cout << "---> " << k << std::endl;
    for(auto i = 0; i < shape[1];++i){
      for(auto j = 0; j < shape[2];++j){
        std::cout << std::setprecision(3) << v[k*100+i*10+j] << " ";
      }
    std::cout << "" << std::endl;
   }
   std::cout << "" << std::endl;
  }


  // Flags.
  std::cout << "Check Flag position" << std::endl;
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "d1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "e1"), 0.0);
  SPIEL_CHECK_FLOAT_EQ(ValueAt(v, shape, 1, "e:"), 1.0/40.0);
  SPIEL_CHECK_FLOAT_EQ(ValueAt(v, shape, 1, "d9"), 0.025);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e5"), 0.0);

  absl::optional<Move> maybe_move = state.Board().ParseLANMove("a4a5");
  SPIEL_CHECK_TRUE(maybe_move);
  Action action = MoveToAction(*maybe_move);
  state.ApplyAction(action);
  maybe_move = state.Board().ParseLANMove("a7a6");
  SPIEL_CHECK_TRUE(maybe_move);
  action = MoveToAction(*maybe_move);
  state.ApplyAction(action);

  state.InformationStateTensor(state.CurrentPlayer(),
                                  absl::MakeSpan(v));
 
  std::cout << "Check Flag position after Move" << std::endl;
  SPIEL_CHECK_FLOAT_EQ(ValueAt(v, shape, 1, "e9"), 1.0/39.0);
  SPIEL_CHECK_FLOAT_EQ(ValueAt(v, shape, 1, "d:"), 1.0/39.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e5"), 0.0);

}

void InformationStateTensorTests() {
  std::shared_ptr<const Game> game = LoadGame("yorktown");
  YorktownState state(game);
  auto shape = game->InformationStateTensorShape();
  std::vector<float> v(game->InformationStateTensorSize());
  state.InformationStateTensor(state.CurrentPlayer(),
                                  absl::MakeSpan(v));

  // For each piece type, check one square that's supposed to be occupied, and
  // one that isn't.
  
  std::cout << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
  for(auto k = 0; k < shape[0]; ++k){
    std::cout << "---> " << k << std::endl;
    for(auto i = 0; i < shape[1];++i){
      for(auto j = 0; j < shape[2];++j){
        std::cout << std::setprecision(3) << v[k*100+i*10+j] << " ";
      }
    std::cout << "" << std::endl;
   }
   std::cout << "" << std::endl;
  }

  // Flags.
  std::cout << "Check Flag position" << std::endl;
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "d1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e5"), 0.0);

  absl::optional<Move> maybe_move = state.Board().ParseLANMove("a4a5");
  SPIEL_CHECK_TRUE(maybe_move);
  Action action = MoveToAction(*maybe_move);
  state.ApplyAction(action);
  maybe_move = state.Board().ParseLANMove("a7a6");
  SPIEL_CHECK_TRUE(maybe_move);
  action = MoveToAction(*maybe_move);
  state.ApplyAction(action);

  state.InformationStateTensor(state.CurrentPlayer(),
                                  absl::MakeSpan(v));
 
  std::cout << "Check Flag position after Move" << std::endl;
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e9"), 0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "d:"), 0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e5"), 0.0);

}

void CloneTests() {
  std::shared_ptr<const Game> game = LoadGame("yorktown");
  YorktownState state(
      game, "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHaa__aa__aaaa__aa__aaSTPQNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 20"); 
  
  std::cout << state.ToString() << std::endl;
  std::unique_ptr<State> clone = state.Clone();
  std::cout << clone->ToString() << std::endl;

  

  
}

}  // namespace
}  // namespace yorktown
}  // namespace open_spiel

int main(int argc, char** argv) {
  //open_spiel::yorktown::BasicTests();
  open_spiel::yorktown::testTesting();
  open_spiel::yorktown::MoveGenerationTests();
  open_spiel::yorktown::UndoTests();
  open_spiel::yorktown::TerminalReturnTests();
  open_spiel::yorktown::InformationStateTensorTestsWithProbability();
  open_spiel::yorktown::CloneTests();
  //open_spiel::yorktown::InformationStateTensorTests();
}
