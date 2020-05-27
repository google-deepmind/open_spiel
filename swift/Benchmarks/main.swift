/// Benchmarks for OpenSpiel.
///
/// To run these from SwiftPM, execute:
///
///    swift run -c release -Xswiftc -cross-module-optimization Benchmarks
///

import Benchmark
import OpenSpiel

func playRandomGame<T: GameProtocol>(_ game: T) {
  // TODO: Add custom benchmark metric tracking the number of turns done,
  // which would be a far more fair comparison. For additional context,
  // see: https://github.com/google/swift-benchmark/issues/11

  var actionLimit = 1000  // Prevents some games from looping infinitely.
  var state = game.initialState
  while !state.isTerminal && actionLimit > 0 {
    state.apply(state.legalActions.randomElement()!)
    actionLimit -= 1
  }
}

benchmark("random game: TicTacToe") {
  playRandomGame(TicTacToe())
}

benchmark("random game: Kuhn Poker") {
  playRandomGame(KuhnPoker())
}

benchmark("random game: Breakthrough") {
  playRandomGame(Breakthrough())
}

benchmark("random game: Texas Hold'em") {
  playRandomGame(TexasHoldem(playerCount: 3))
}

Benchmark.main()
