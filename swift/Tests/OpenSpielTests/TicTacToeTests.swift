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

final class TicTacToeTests: XCTestCase {
  func testSimpleGame() throws {
    let game = TicTacToe()
    var state = game.initialState
    XCTAssertEqual(state.currentPlayer, .player(0))
    var allActions = Set((0..<3).flatMap { x in (0..<3).map { y in TicTacToe.Action(x: x, y: y) }})
    XCTAssertEqual(Set(state.legalActions), allActions)
    state.apply(TicTacToe.Action(x: 1, y: 1))
    allActions.remove(TicTacToe.Action(x: 1, y: 1))
    XCTAssertEqual(Set(state.legalActions), allActions)
    XCTAssertEqual(state.currentPlayer, .player(1))
    state.apply(TicTacToe.Action(x: 0, y: 1))
    allActions.remove(TicTacToe.Action(x: 0, y: 1))
    XCTAssertEqual(Set(state.legalActions), allActions)
    XCTAssertEqual(state.currentPlayer, .player(0))
    state.apply(TicTacToe.Action(x: 2, y: 0))
    state.apply(TicTacToe.Action(x: 0, y: 0))
    state.apply(TicTacToe.Action(x: 0, y: 2))
    XCTAssertTrue(state.isTerminal)
    XCTAssertEqual(state.winner, .player(0))
    XCTAssertEqual(state.utility(for: .player(0)), 1)
    XCTAssertEqual(state.utility(for: .player(1)), -1)
  }

  func testTieGame() throws {
    // constructs a board that looks as follows:
    //
    //    o x x
    //    x o o
    //    o x x
    //
    // which results in a tie game.
    let game = TicTacToe()
    var state = game.initialState
    state.apply(TicTacToe.Action(x: 1, y: 0))  // x
    state.apply(TicTacToe.Action(x: 0, y: 0))  // o
    state.apply(TicTacToe.Action(x: 0, y: 1))  // x
    state.apply(TicTacToe.Action(x: 1, y: 1))  // o
    state.apply(TicTacToe.Action(x: 1, y: 2))  // x
    state.apply(TicTacToe.Action(x: 0, y: 2))  // o
    state.apply(TicTacToe.Action(x: 2, y: 0))  // x
    state.apply(TicTacToe.Action(x: 2, y: 1))  // o
    state.apply(TicTacToe.Action(x: 2, y: 2))  // x
    XCTAssertTrue(state.isTerminal)
    XCTAssertEqual(state.winner, nil)
    XCTAssertEqual(state.utility(for: .player(0)), 0)
    XCTAssertEqual(state.utility(for: .player(1)), 0)
  }
}

extension TicTacToeTests {
  static var allTests = [
    ("testSimpleGame", testSimpleGame),
    ("testTieGame", testTieGame),
  ]
}
