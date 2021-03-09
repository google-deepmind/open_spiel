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
