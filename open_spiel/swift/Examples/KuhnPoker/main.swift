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
