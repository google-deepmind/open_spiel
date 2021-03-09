import XCTest
import OpenSpiel

final class LeducPokerTests: XCTestCase {
  func testCardPairing() throws {
    let a = LeducPoker.Card(value: 1)
    let b = LeducPoker.Card(value: 0)
    XCTAssertEqual(a.rank, b.rank)
    XCTAssertNotEqual(a.suit, b.suit)
  }

  func testCheckChanceOutcomes() throws {
    let game = LeducPoker(playerCount: 3)
    checkChanceOutcomes(game: game)
  }

  func testHandRanking() throws {
    let game = LeducPoker(playerCount: 2)
    var state = game.initialState
    // In the standard 2-player Leduc poker, there are 6 cards, corresponding to Jack, Queen, & King
    // in the 2 (unranked) suits.

    // Construct a scenario where player 1 has 2 Queens, and Player 2 has a King & a Queen.
    state.privateCards = [LeducPoker.Card(value: 3), LeducPoker.Card(value: 5)]
    state.publicCard = LeducPoker.Card(value: 2)
    // 6^2 + 1 (numCards^2 + pairRank)
    XCTAssertEqual(37, state.rankHand(for: .player(0)))
    // 6*2 + 1 (numCards * higherRank + lowerRank)
    XCTAssertEqual(13, state.rankHand(for: .player(1)))

    // Change public card to a jack, and re-evaluate.
    state.publicCard = LeducPoker.Card(value: 0)
    // 6*2 + 0
    XCTAssertEqual(12, state.rankHand(for: .player(1)))
    // 6*1 + 0
    XCTAssertEqual(6, state.rankHand(for: .player(0)))

    // Compute winners
    state.resolveWinner()
    XCTAssertEqual(1, state.winnerCount)
    XCTAssertEqual([false, true], state.winners)
  }

  func testSimpleSequence() throws {
    let game = LeducPoker(playerCount: 2)
    var state = game.initialState

    XCTAssertEqual(.chance, state.currentPlayer)
    state.apply(.card(LeducPoker.Card(value: 3)))
    XCTAssertEqual(.chance, state.currentPlayer)
    state.apply(.card(LeducPoker.Card(value: 0)))
    XCTAssertEqual(.player(0), state.currentPlayer)

    state.apply(.raise)
    XCTAssertEqual(.player(1), state.currentPlayer)
    state.apply(.call)

    XCTAssertEqual(.chance, state.currentPlayer)
    state.apply(.card(LeducPoker.Card(value: 1)))
    XCTAssertEqual(.player(0), state.currentPlayer)
    state.apply(.call)
    XCTAssertEqual(.player(1), state.currentPlayer)
    state.apply(.raise)
    XCTAssertEqual(.player(0), state.currentPlayer)
    state.apply(.call)

    XCTAssertEqual(.terminal, state.currentPlayer)
    XCTAssertEqual([false, true], state.winners)
    XCTAssertEqual(1, state.winnerCount)
  }
}

extension LeducPokerTests {
  static var allTests = [
    ("testCardPairing", testCardPairing),
    ("testHandRanking", testHandRanking),
    ("testCheckChanceOutcomes", testCheckChanceOutcomes),
    ("testSimpleSequence", testSimpleSequence),
  ]
}
