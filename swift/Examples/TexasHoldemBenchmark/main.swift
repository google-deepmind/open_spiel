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

/// This file executes repeated random play over the Texas Hold'em game.
///
/// It is used for profiling and benchmarking purposes.
///
/// On saeta@'s desktop machine, performance numbers are as follows (be sure to
/// build with optimizations on!):
///   - Random games: 115 +/- 51 usec per game
///   - Fixed games:  4.7 +/- 0.4 usec per game

import Dispatch
import Foundation
import TensorFlow
import OpenSpiel

// BEGIN GOOGLE-INTERNAL
import base_swift_base

Google.initialize()
// END GOOGLE-INTERNAL

func time(_ benchmarkName: String, _ f: () -> ()) {
  // Warm up
  f()

  var times = [Double]()

  // Play 10000 games.
  for _ in 1...10000 {
    let start = DispatchTime.now()
    f()
    let end = DispatchTime.now()
    let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
    let milliseconds = nanoseconds / 1e6
    times.append(milliseconds)
  }

  let meanMs = times.reduce(0, +) / Double(times.count)
  // Separate stddev calculation to 2 lines to ease typechecking. :-(
  let sumSquaredDifferences = times.map { pow($0 - meanMs, 2) }.reduce(0, +)
  let stddev = sqrt(sumSquaredDifferences / Double(times.count - 1))
  print("\(benchmarkName)    mean: \(meanMs) ms +/- \(stddev)")
}


/// Play games at random to see
///
/// Note: this benchmark is dominated by the overheads of random number generation
/// and constructing collection manipulations (specifically around chance nodes, as
/// there are a lot more cards than player actions, so these tend to dominate the
/// runtime).
func playRandomGame() {
  let game = TexasHoldem(playerCount: 2)
  var state = game.initialState
  while !state.isTerminal {
    state.apply(state.legalActions.randomElement()!)
  }
}

/// Play a fixed game to benchmark the overheads of the state mechanisms itself.
///
/// Note: based on profiling, the vast majority of the time is actually spent in computeBestHand
/// with the rest of the time spent in either array reallocation or swift_beginAccess (these
/// dynamic safety checks can be disabled with additional optimization flags).
func playFixed2PlayerGame() {
  let game = TexasHoldem(playerCount: 2)
  var state = game.initialState
  // Deal 4 cards (this deck happens to be relatively poorly shuffled...)
  state.apply(.card(Card(suit: .clubs, rank: .two)))
  state.apply(.card(Card(suit: .clubs, rank: .three)))
  state.apply(.card(Card(suit: .clubs, rank: .four)))
  state.apply(.card(Card(suit: .clubs, rank: .five)))

  // Pre-flop
  state.apply(.call)
  state.apply(.call)

  // Deal flop
  state.apply(.card(Card(suit: .clubs, rank: .six)))
  state.apply(.card(Card(suit: .clubs, rank: .seven)))
  state.apply(.card(Card(suit: .clubs, rank: .eight)))

  // Bid
  state.apply(.raise(1))
  state.apply(.raise(1))
  state.apply(.call)

  // Turn & bid
  state.apply(.card(Card(suit: .clubs, rank: .nine)))
  state.apply(.call)
  state.apply(.raise(2))
  state.apply(.call)

  // River & bid
  state.apply(.card(Card(suit: .clubs, rank: .ten)))
  state.apply(.allIn)
  state.apply(.call)
}

print("About to start benchmarks...")
time("Random 2-player Texas Hold'em games", playRandomGame)
time("Fixed 2-player Texas Hold'em games", playFixed2PlayerGame)
print("Done!")
