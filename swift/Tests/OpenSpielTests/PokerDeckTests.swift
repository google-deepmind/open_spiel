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

final class PokerDeckTests: XCTestCase {
  func testCardRoundTrips() throws {
    var seen = Set<Card>()
    seen.reserveCapacity(52)
    for index in 0..<52 {
      let orig = Card(index: UInt8(index))!
      XCTAssert(!seen.contains(orig), "Already seen \(orig)")
      XCTAssert(seen.insert(orig).inserted)
      // Test roundtripping.
      let altInit = Card(suit: orig.suit, rank: orig.rank)
      XCTAssertEqual(orig, altInit)
      XCTAssertEqual(orig.suit, altInit.suit)
      XCTAssertEqual(orig.rank, altInit.rank)
    }
  }

  func testHandRanking() throws {
    XCTAssertLessThan(PokerHandRank.highCard(.seven, .five, .four, .three, .two),
                      PokerHandRank.highCard(.eight, .five, .four, .three, .two))
    XCTAssertLessThan(PokerHandRank.highCard(.seven, .six, .four, .three, .two),
                      PokerHandRank.highCard(.eight, .five, .four, .three, .two))
    XCTAssertLessThan(PokerHandRank.highCard(.ace, .five, .four, .three, .two),
                      PokerHandRank.pair(.two, kickers: .ace, .king, .queen))
    XCTAssertLessThan(PokerHandRank.pair(.two, kickers: .ace, .king, .queen),
                      PokerHandRank.pair(.ace, kickers: .four, .three, .two))
    XCTAssertLessThan(PokerHandRank.pair(.ace, kickers: .king, .queen, .jack),
                      PokerHandRank.twoPair(.three, .two, kicker: .four))
    XCTAssertLessThan(PokerHandRank.twoPair(.ace, .king, kicker: .queen),
                      PokerHandRank.threeOfAKind(.two, kickers: .four, .three))
    XCTAssertLessThan(PokerHandRank.threeOfAKind(.ace, kickers: .king, .queen),
                      PokerHandRank.straight(.five))
    XCTAssertLessThan(PokerHandRank.straight(.ace), PokerHandRank.flush(.six))
    XCTAssertLessThan(PokerHandRank.flush(.ace), PokerHandRank.fullHouse(.two, .three))
    XCTAssertLessThan(PokerHandRank.fullHouse(.two, .ace), PokerHandRank.fullHouse(.three, .two))
    XCTAssertLessThan(PokerHandRank.fullHouse(.ace, .king),
                      PokerHandRank.fourOfAKind(.two, kicker: .three))
    XCTAssertLessThan(PokerHandRank.fourOfAKind(.two, kicker: .three),
                      PokerHandRank.fourOfAKind(.two, kicker: .four))
    XCTAssertLessThan(PokerHandRank.fourOfAKind(.two, kicker: .three),
                      PokerHandRank.fourOfAKind(.four, kicker: .two))
    XCTAssertLessThan(PokerHandRank.fourOfAKind(.two, kicker: .ace),
                      PokerHandRank.fourOfAKind(.king, kicker: .jack))
    XCTAssertLessThan(PokerHandRank.fourOfAKind(.ace, kicker: .king),
                      PokerHandRank.straightFlush(.five))
    XCTAssertLessThan(PokerHandRank.straightFlush(.king), PokerHandRank.straightFlush(.ace))
  }

  func testHandScoring() throws {
    XCTAssertEqual(
      PokerHand((.clubs, .two),
                (.diamonds, .two),
                (.spades, .three),
                (.hearts, .five),
                (.hearts, .queen)).rank,
      .pair(.two, kickers: .queen, .five, .three))

    XCTAssertEqual(
      PokerHand((.clubs, .two),
                (.clubs, .five),
                (.clubs, .eight),
                (.clubs, .queen),
                (.clubs, .king)).rank,
      .flush(.king))

    XCTAssertEqual(
      PokerHand((.hearts, .eight),
                (.spades, .five),
                (.clubs, .seven),
                (.diamonds, .nine),
                (.hearts, .six)).rank,
      .straight(.nine))

    XCTAssertEqual(
      PokerHand((.spades, .three),
                (.spades, .five),
                (.spades, .ace),
                (.spades, .four),
                (.spades, .two)).rank,
      .straightFlush(.five))

    XCTAssertEqual(
      PokerHand((.hearts, .jack),
                (.hearts, .king),
                (.hearts, .ace),
                (.hearts, .queen),
                (.hearts, .ten)).rank,
      .straightFlush(.ace))

    XCTAssertEqual(
      PokerHand((.hearts, .king),
                (.spades, .king),
                (.diamonds, .king),
                (.hearts, .jack),
                (.clubs, .jack)).rank,
      .fullHouse(.king, .jack))

    XCTAssertEqual(
      PokerHand((.clubs, .five),
                (.diamonds, .two),
                (.hearts, .seven),
                (.clubs, .two),
                (.spades, .five)).rank,
      .twoPair(.five, .two, kicker: .seven))

    XCTAssertEqual(
      PokerHand((.clubs, .nine),
                (.spades, .ace),
                (.hearts, .nine),
                (.spades, .nine),
                (.diamonds, .nine)).rank,
      .fourOfAKind(.nine, kicker: .ace))
    XCTAssertEqual(
      PokerHand((.clubs, .ace),
                (.hearts, .ace),
                (.diamonds, .jack),
                (.spades, .two),
                (.hearts, .two)).rank,
      .twoPair(.ace, .two, kicker: .jack))
  }

  func testTexasHoldemRanking() {
    XCTAssertEqual(
      PokerHandRank(texasHoldem: makeCards("s3", "d5", "h8", "s9", "sJ", "sQ", "cA")),
      .highCard(.ace, .queen, .jack, .nine, .eight))
    XCTAssertEqual(
      PokerHandRank(texasHoldem: makeCards("s3", "d5", "s8", "s9", "sJ", "sQ", "cA")),
      .flush(.queen))
    XCTAssertEqual(
      PokerHandRank(texasHoldem: makeCards("dJ", "d5", "s8", "hJ", "sJ", "sQ", "c5")),
      .fullHouse(.jack, .five))
    XCTAssertEqual(
      PokerHandRank(texasHoldem: makeCards("s8", "d5", "c8", "s9", "dT", "h7", "h6")),
      .straight(.ten))
    XCTAssertEqual(
      PokerHandRank(texasHoldem: makeCards("s3", "d5", "s8", "s9", "sJ", "sQ", "sT")),
      .straightFlush(.queen))
    XCTAssertEqual(
      PokerHandRank(texasHoldem: makeCards("sK", "d5", "c8", "d7", "hK", "sQ", "h7")),
      .twoPair(.king, .seven, kicker: .queen))
    XCTAssertEqual(
      PokerHandRank(texasHoldem: makeCards("s2", "d2", "h2", "c2", "sJ", "sQ", "sA")),
      .fourOfAKind(.two, kicker: .ace))
  }
}

extension PokerDeckTests {
  static var allTests = [
    ("testCardRoundTrips", testCardRoundTrips),
    ("testHandRanking", testHandRanking),
    ("testHandScoring", testHandScoring),
    ("testTexasHoldemRanking", testTexasHoldemRanking),
  ]
}

fileprivate func makeCards(_ cards: String...) -> [Card] {
  return cards.map { str in
    precondition(str.count == 2, "Card \(str) is not valid.")
    let suit: Suit
    let rank: Rank

    switch str.first! {
      case "c":
        suit = .clubs
      case "d":
        suit = .diamonds
      case "h":
        suit = .hearts
      case "s":
        suit = .spades
      default:
        fatalError("Invalid card (unknown suit): \(str)")
    }

    switch str.last! {
      case "2":
        rank = .two
      case "3":
        rank = .three
      case "4":
        rank = .four
      case "5":
        rank = .five
      case "6":
        rank = .six
      case "7":
        rank = .seven
      case "8":
        rank = .eight
      case "9":
        rank = .nine
      case "T":
        rank = .ten
      case "J":
        rank = .jack
      case "Q":
        rank = .queen
      case "K":
        rank = .king
      case "A":
        rank = .ace
      default:
        fatalError("Invalid card (unknown rank): \(str)")
    }
    return Card(suit: suit, rank: rank)
  }
}
