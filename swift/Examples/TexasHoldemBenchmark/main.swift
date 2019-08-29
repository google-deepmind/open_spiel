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

import Dispatch
import Foundation
import TensorFlow
import OpenSpiel


func time(_ benchmarkName: String, _ f: () -> ()) {
  // Warm up
  f()

  var times = [Double]()

  // Play 100 games.
  for _ in 1...1000 {
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


func playRandomGame() {
  let game = TexasHoldem(playerCount: 2)
  var state = game.initialState
  while !state.isTerminal {
    state.apply(state.legalActions.randomElement()!)
  }
}

print("About to start playing random games...")
time("Random 2-player Texas Hold'em games", playRandomGame)
print("Done!")
