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

final class TexasHoldemTests: XCTestCase {
  func testRandomGameplay() throws {
    let game = TexasHoldem(playerCount: 2)
    for _ in 1...100 {
      playRandomGame(game)
    }
  }

  func testRaiseLogic() {
    let game = TexasHoldem(playerCount: 4, smallBlind: 30, bigBlind: 70)
    var state = game.initialState
    XCTAssertEqual(state.pot, 100)
    dealCards(&state, expected: 8)

    XCTAssertEqual(state.currentPlayer, .player(3))
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5), .raise(1), .raise(2)])

    state.apply(.raise(1))

    XCTAssertEqual(state.pot, 200)
    XCTAssertEqual(state.money, [10000, 9970, 9930, 9900])
    XCTAssertEqual(state.currentPlayer, .player(0))

    state.apply(.raise(1))

    XCTAssertEqual(state.pot, 200)
    XCTAssertEqual(state.money, [9800, 9970, 9930, 9900])
    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5), .raise(1), .raise(2)])
  }

  func testSimpleTwoPlayerGame() {
    let game = TexasHoldem(playerCount: 2)
    var state = game.initialState
    XCTAssertEqual(state.money, [9900, 9950])
    verifyNoDuplicateCards(state)

    dealCards(&state, expected: 4)

    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(1), .raise(2)])
    XCTAssert(!state.isTerminal)
    verifyNoDuplicateCards(state)

    state.apply(.call)

    XCTAssertEqual(state.currentPlayer, .player(0))
    XCTAssertEqual(state.legalActions, [.call, .allIn, .raise(1), .raise(2)])
    XCTAssert(!state.isTerminal)
    XCTAssertEqual(state.money, [9900, 9900])
    XCTAssertEqual(state.pot, 200)

    state.apply(.call)

    XCTAssertEqual(state.currentPlayer, .chance)
    XCTAssertEqual(state.round, .postFlop)
    XCTAssert(!state.isTerminal)
    verifyNoDuplicateCards(state)

    dealCards(&state, expected: 3)

    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssertEqual(state.legalActions, [.call, .allIn, .raise(1), .raise(2)])
    verifyNoDuplicateCards(state)

    state.apply(.raise(1))

    XCTAssertEqual(state.pot, 400)
    XCTAssertEqual(state.currentPlayer, .player(0))
    XCTAssertEqual(state.money, [9900, 9700])
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5), .raise(1), .raise(2)])
    XCTAssert(!state.isTerminal)

    state.apply(.call)

    XCTAssertEqual(state.pot, 600)
    XCTAssertEqual(state.currentPlayer, .chance)
    XCTAssertEqual(state.round, .postTurn)
    XCTAssert(!state.isTerminal)

    dealCards(&state, expected: 1)

    XCTAssertEqual(state.currentPlayer, .player(1))

    state.apply(.call)  // Check

    XCTAssertEqual(state.currentPlayer, .player(0))
    XCTAssertEqual(state.legalActions, [.call, .allIn, .raise(0.5), .raise(1), .raise(2)])

    state.apply(.raise(0.5))

    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5), .raise(1), .raise(2)])
    XCTAssertEqual(state.pot, 900)
    XCTAssertEqual(state.money, [9400, 9700])
    XCTAssertEqual(state.betLevel, 600)
    XCTAssert(!state.isTerminal)

    state.apply(.call)

    XCTAssertEqual(state.round, .postRiver)
    XCTAssertEqual(state.currentPlayer, .chance)
    XCTAssertEqual(state.pot, 1200)
    XCTAssertEqual(state.money, [9400, 9400])
    XCTAssertEqual(state.betLevel, 600)

    dealCards(&state, expected: 1)

    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssert(!state.isTerminal)

    state.apply(.raise(200))
    state.apply(.fold)

    XCTAssertEqual(state.currentPlayer, .terminal)
    XCTAssert(state.isTerminal)
    XCTAssertEqual(state.utility(for: .player(0)), -600)
    XCTAssertEqual(state.utility(for: .player(1)), 600)
  }

  // Testcase discovered through random play.
  func testAllIn() {
    let game = TexasHoldem(playerCount: 2)
    var state = game.initialState
    dealCards(&state, expected: 4)

    state.apply(.call)
    state.apply(.allIn)
    XCTAssertEqual(state.legalActions, [.fold, .call], "State: \(state)")
  }

  // Testcase discovered through random play.
  func testRaisesAndAllIn() {
    let game = TexasHoldem(playerCount: 2)
    var state = game.initialState
    dealCards(&state, expected: 4)

    state.apply(.raise(2))
    state.apply(.raise(2))
    state.apply(.raise(2))
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5)])
  }

  func testRevealWinnerComputation() {
    let game = TexasHoldem(playerCount: 2)
    var state = game.initialState
    // Fast forward the state to the showdown to test a double call.
    state.money = [9600, 9600]
    state.betLevel = 400
    state.pot = 800
    state.round = .postRiver
    state.currentPlayer = .player(1)
    state.communityCards = [
      Card(suit: .spades, rank: .three),
      Card(suit: .hearts, rank: .three),
      Card(suit: .diamonds, rank: .three),
      Card(suit: .spades, rank: .four),
      Card(suit: .spades, rank: .five),
    ]
    state.privateCards = [
      TexasHoldem.PrivateCards(
        first: Card(suit: .diamonds, rank: .five),
        second: Card(suit: .hearts, rank: .queen)),
      TexasHoldem.PrivateCards(
        first: Card(suit: .clubs, rank: .four),
        second: Card(suit: .clubs, rank: .three)),
    ]
    XCTAssert(!state.isTerminal)

    state.apply(.call)
    state.apply(.call)

    XCTAssertEqual(state.currentPlayer, .terminal)
    XCTAssert(state.isTerminal)
    XCTAssertEqual(state.utility(for: .player(0)), -400)
    XCTAssertEqual(state.utility(for: .player(1)), 400)
  }

  func testRevealWinnerComputationTie() {
    let game = TexasHoldem(playerCount: 2)
    var state = game.initialState
    // Fast forward the state to the showdown to test a double call.
    state.money = [9600, 9600]
    state.pot = 800
    state.round = .postRiver
    state.currentPlayer = .player(1)
    state.betLevel = 400
    state.communityCards = [
      Card(suit: .spades, rank: .three),
      Card(suit: .hearts, rank: .three),
      Card(suit: .diamonds, rank: .four),
      Card(suit: .spades, rank: .four),
      Card(suit: .spades, rank: .five),
    ]
    state.privateCards = [
      TexasHoldem.PrivateCards(
        first: Card(suit: .diamonds, rank: .five),
        second: Card(suit: .hearts, rank: .three)),
      TexasHoldem.PrivateCards(
        first: Card(suit: .clubs, rank: .five),
        second: Card(suit: .clubs, rank: .three)),
    ]
    XCTAssert(!state.isTerminal)

    state.apply(.call)
    state.apply(.call)

    XCTAssertEqual(state.currentPlayer, .terminal)
    XCTAssert(state.isTerminal)
    XCTAssertEqual(state.utility(for: .player(0)), 0)
    XCTAssertEqual(state.utility(for: .player(1)), 0)
  }

  func testAllFoldAfterRaise() {
    let game = TexasHoldem(playerCount: 4)
    var state = game.initialState
    dealCards(&state)
    XCTAssertEqual(state.currentPlayer, .player(3))
    state.apply(.raise(1))
    XCTAssertEqual(state.currentPlayer, .player(0))
    XCTAssertEqual(state.money, [10000, 9950, 9900, 9750])
    // Everyone immediately folds.
    state.apply(.fold)
    state.apply(.fold)
    state.apply(.fold)
    XCTAssertEqual(state.currentPlayer, .terminal)
    XCTAssertEqual(state.money, [10000, 9950, 9900, 10150])
  }

  func testLegalActions() {
    let game = TexasHoldem(playerCount: 4)
    var state = game.initialState
    dealCards(&state)
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(1), .raise(2)])

    state.pot = 300
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5), .raise(1), .raise(2)])

    state.round = .postRiver
    state.pot = 150
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(2)])

    state.pot = 300
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(1), .raise(2)])

    state.pot = 500
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5), .raise(1), .raise(2)])

    state.pot = 6000
    XCTAssertEqual(state.legalActions, [.fold, .call, .allIn, .raise(0.5), .raise(1)])
  }

  private func playRandomGame(_ game: TexasHoldem) {
    var state = game.initialState
    var stepCount = 0
    while !state.isTerminal && stepCount <= game.maxGameLength {
      stepCount += 1
      assert(state.legalActions.count > 0, "\(state)\n\n\(state.legalActions)")
      let action = state.legalActions.randomElement()!
      switch action {
      case .raise:
        state.apply(.raise(2))
      default:
        state.apply(action)
      }
    }
    XCTAssert(state.isTerminal, "The state should be terminal.")
    XCTAssertLessThan(
      stepCount,
      game.maxGameLength,
      "Game did not terminate in time.")
  }

  private func dealCards(_ state: inout TexasHoldem.State, expected: Int? = nil) {
    var numCardsDelt = 0
    while state.currentPlayer == .chance {
      numCardsDelt += 1
      assert(state.legalActions.count > 0, "\(state.legalActions)\n\(state)")
      state.apply(state.legalActions.randomElement()!)
    }
    if let expected = expected {
      XCTAssertEqual(numCardsDelt, expected)
    }
  }

  private func verifyNoDuplicateCards(_ state: TexasHoldem.State) {
    var seenCards = Array(repeating: false, count: 52)
    func check(_ card: Card) {
      XCTAssert(!seenCards[Int(card.index)])
      seenCards[Int(card.index)] = true
    }

    for case let card? in state.deck {
      check(card)
    }
    for privateCards in state.privateCards {
      if let card = privateCards.first {
        check(card)
      }
      if let card = privateCards.second {
        check(card)
      }
    }
    for card in state.communityCards {
      check(card)
    }
    XCTAssert(seenCards.allSatisfy { $0 }, "\(seenCards) \(state)")
  }
}

extension TexasHoldemTests {
  static var allTests = [
    ("testAllIn", testAllIn),
    ("testRaisesAndAllIn", testRaisesAndAllIn),
    ("testRandomGameplay", testRandomGameplay),
    ("testSimpleTwoPlayerGame", testSimpleTwoPlayerGame),
    ("testRevealWinnerComputation", testRevealWinnerComputation),
    ("testRevealWinnerComputationTie", testRevealWinnerComputationTie),
    ("testAllFoldAfterRaise", testAllFoldAfterRaise),
  ]
}
