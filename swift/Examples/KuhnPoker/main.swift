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

/// This file implements an example program using the OpenSpiel library

import TensorFlow
import OpenSpiel

let game = KuhnPoker()
var gameState = game.initialState

print("About to play a Kuhn Poker game!")

while !gameState.isTerminal {
  print("""
      Step \(gameState.history.count):
        State: \(gameState)
        Current player: \(gameState.currentPlayer)
      """)
  switch gameState.currentPlayer {
  case .chance:
    let outcomes = gameState.chanceOutcomes
    let sampledOutcome = sampleChanceOutcome(outcomes)
    print("  --> \(gameState.currentPlayer) sampled outcome \(sampledOutcome)")
    gameState.apply(sampledOutcome)
  case .player, .simultaneous:
    let legalActions = gameState.legalActions
    print("Legal actions: \(legalActions)")
    let action = legalActions.randomElement()!
    print("  --> \(gameState.currentPlayer) taking action \(action)")
    gameState.apply(action)
  default:
    fatalError("Unexpected current player: \(gameState.currentPlayer)")
  }
}

print(gameState)
print("""
      Game over!
       - Player 0 score: \(gameState.utility(for: .player(0)))
       - Player 1 score: \(gameState.utility(for: .player(1)))
      """)
