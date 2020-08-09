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

benchmark("random game: FastBreakthrough") {
  playRandomGame(FastBreakthrough())
}

benchmark("random game: Texas Hold'em") {
  playRandomGame(TexasHoldem(playerCount: 3))
}

Benchmark.main()
