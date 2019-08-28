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

import XCTest
import OpenSpiel

/// Check chance outcomes in a state and all child states.
func checkChanceOutcomes<State: StateProtocol>(state: State) {
  if state.isTerminal {
    return
  }

  if state.currentPlayer == .chance {
    let legalActions = state.legalActions
    let legalActionSet = Set(legalActions)
    let chanceOutcomes = state.chanceOutcomes

    var sum = 0.0
    for (outcome, probability) in chanceOutcomes {
      XCTAssert(legalActionSet.contains(outcome))
      XCTAssert(probability > 0.0 && probability <= 1.0)
      sum += probability
    }
    XCTAssertEqual(sum, 1.0, accuracy: 0.00000001)
  }

  for action in state.legalActions {
    var clone = state
    clone.apply(action)
    checkChanceOutcomes(state: clone)
  }
}

/// Check that chance outcomes are valid and consistent.
/// Performs an exhaustive search of the game tree, so should only be
/// used for smallish games.
func checkChanceOutcomes<Game: GameProtocol>(game: Game) {
  checkChanceOutcomes(state: game.initialState)
}
