import XCTest
import OpenSpiel

final class KuhnPokerTests: XCTestCase {
  func testCheckChanceOutcomes() throws {
    let game = KuhnPoker(playerCount: 3)
    checkChanceOutcomes(game: game)
  }
}

extension KuhnPokerTests {
  static var allTests = [
    ("testCheckChanceOutcomes", testCheckChanceOutcomes),
  ]
}
