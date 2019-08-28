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

final class BreakthroughTests: XCTestCase {

  func testBoardLocations() throws {
    let game = Breakthrough()
    XCTAssertTrue(game.isValid(location: Breakthrough.BoardLocation(x: 0, y: 0)))
    XCTAssertTrue(game.isValid(location: Breakthrough.BoardLocation(x: 3, y: 6)))
    XCTAssertFalse(game.isValid(location: Breakthrough.BoardLocation(x: 7, y: 8)))
    XCTAssertFalse(game.isValid(location: Breakthrough.BoardLocation(x: 9, y: 9)))
  }

  func testBoardLocationMovement() throws {
    XCTAssertEqual(Breakthrough.BoardLocation(x: 0, y: 0),
                   Breakthrough.BoardLocation(x: 1, y: 1).move(in: .left, for: .black))
    XCTAssertEqual(Breakthrough.BoardLocation(x: 1, y: 1),
                   Breakthrough.BoardLocation(x: 0, y: 0).move(in: .right, for: .white))
  }

  func testStateInitialization() throws {
    let smallGame = Breakthrough(boardHeight: 5, boardWidth: 2)
    let state = smallGame.initialState
    XCTAssertEqual(state[0, 0], .white)
    XCTAssertEqual(state[0, 1], .white)
    XCTAssertEqual(state[0, 2], nil)
    XCTAssertEqual(state[0, 3], .black)
    XCTAssertEqual(state[0, 4], .black)
    XCTAssertEqual(state[1, 0], .white)
    XCTAssertEqual(state[1, 1], .white)
    XCTAssertEqual(state[1, 2], nil)
    XCTAssertEqual(state[1, 3], .black)
    XCTAssertEqual(state[1, 4], .black)

    XCTAssertEqual(state.currentPlayer, Breakthrough.BreakthroughPlayer.black.player)
    XCTAssertEqual(Breakthrough.BreakthroughPlayer(state.currentPlayer), .black)
  }

  func testSimpleGame() throws {
    let smallGame = Breakthrough(boardHeight: 5, boardWidth: 2)
    var state = smallGame.initialState

    XCTAssertEqual(state.currentPlayer, .player(0))
    XCTAssertEqual(Set(state.legalActions), Set([
      Breakthrough.Action(location: Breakthrough.BoardLocation(x: 0, y: 3),
                          direction: .forward),
      Breakthrough.Action(location: Breakthrough.BoardLocation(x: 0, y: 3),
                          direction: .right),
      Breakthrough.Action(location: Breakthrough.BoardLocation(x: 1, y: 3),
                          direction: .forward),
      Breakthrough.Action(location: Breakthrough.BoardLocation(x: 1, y: 3),
                          direction: .left),
    ]))
    state.apply(Breakthrough.Action(location: Breakthrough.BoardLocation(x: 0, y: 3),
                                    direction: .forward))
    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssertEqual(state.history.count, 1)
    state.apply(Breakthrough.Action(location: Breakthrough.BoardLocation(x: 1, y: 1),
                                    direction: .left))
    XCTAssertEqual(state.currentPlayer, .player(0))
    XCTAssertEqual(state.history.count, 2)
    state.apply(Breakthrough.Action(location: Breakthrough.BoardLocation(x: 1, y: 3),
                                    direction: .left))
    state.apply(Breakthrough.Action(location: Breakthrough.BoardLocation(x: 1, y: 0),
                                    direction: .forward))
    state.apply(Breakthrough.Action(location: Breakthrough.BoardLocation(x: 0, y: 2),
                                    direction: .right))
    state.apply(Breakthrough.Action(location: Breakthrough.BoardLocation(x: 0, y: 1),
                                    direction: .forward))
    state.apply(Breakthrough.Action(location: Breakthrough.BoardLocation(x: 1, y: 1),
                                    direction: .forward))
    XCTAssertEqual(state.history.count, 7)
    XCTAssertEqual(state.currentPlayer, .terminal)
    XCTAssertEqual(state.winner, .black)
    XCTAssertEqual(state.utility(for: .player(0)), 1)
    XCTAssertEqual(state.utility(for: .player(1)), -1)
  }
}

extension BreakthroughTests {
  static var allTests = [
    ("testBoardLocations", testBoardLocations),
    ("testBoardLocationMovement", testBoardLocationMovement),
    ("testStateInitialization", testStateInitialization),
    ("testSimpleGame", testSimpleGame),
  ]
}
