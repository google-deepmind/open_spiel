/// A full 52 card deck used for poker.

/// The suit of a card.
public enum Suit: UInt8, CaseIterable {
  case clubs, diamonds, hearts, spades
}

extension Suit: CustomStringConvertible {
  public var description: String {
    switch self {
    case .clubs:
      return "c"
    case .diamonds:
      return "d"
    case .hearts:
      return "h"
    case .spades:
      return "s"
    }
  }
}

/// The rank of the card.
public enum Rank: UInt8, CaseIterable {
  case two = 2, three, four, five, six, seven, eight, nine, ten, jack, queen, king, ace
}

extension Rank: CustomStringConvertible {
  public var description: String {
    switch self {
    case .two: return "2"
    case .three: return "3"
    case .four: return "4"
    case .five: return "5"
    case .six: return "6"
    case .seven: return "7"
    case .eight: return "8"
    case .nine: return "9"
    case .ten: return "T"
    case .jack: return "J"
    case .queen: return "Q"
    case .king: return "K"
    case .ace: return "A"
    }
  }
}

/// A card in a standard playing deck.
public struct Card: Equatable, Hashable {
  /// The rank-major ordering of cards.
  public let index: UInt8

  /// Initialize a Card based on suit and rank.
  public init(suit: Suit, rank: Rank) {
    self.index = ((rank.rawValue - 2) << 2) + suit.rawValue
  }

  /// Initialize a card based on the index.
  ///
  /// This initializer returns nil if the index is not less than 52.
  public init?(index: UInt8) {
    guard index < 52 else { return nil }
    self.index = index
  }

  /// The suit of the card.
  public var suit: Suit {
    Suit(rawValue: index & 3)!
  }

  /// The rank of the card.
  public var rank: Rank {
    Rank(rawValue: (index >> 2) + 2)!
  }
}

extension Card: CustomStringConvertible {
  public var description: String {
    "\(suit)\(rank)"
  }
}

/// This enum captures the salient details of the quality of a hand.
///
/// It is used to compare two hands.
public enum PokerHandRank: Equatable {
  case highCard(Rank, Rank, Rank, Rank, Rank)
  case pair(Rank, kickers: Rank, Rank, Rank)
  case twoPair(Rank, Rank, kicker: Rank)
  case threeOfAKind(Rank, kickers: Rank, Rank)
  case straight(Rank)
  case flush(Rank)
  case fullHouse(Rank, Rank)
  case fourOfAKind(Rank, kicker: Rank)  // four of a kind has a kicker
  case straightFlush(Rank)

  // Because a normal conformance to Comparable would result in a very branchy implementation of
  // func <, we instead compute a non-overlapping int score that orders all hands, which is used
  // for comparison.
  //
  // The score is the combination of at most 5 ranks. 5 bits are used for each rank, leaving 7 bits
  // for the hand type. (Only 5 are used.)
  //
  // To simplify construction of the score, we stuff the rank into the lowest 5 bits. To support
  // four of a kind, we reserve another 5 bits for the rank of the kicker. The top 6 bits (only 5
  // are used) encode the case.
  public var score: UInt32 {
    switch self {
    case let .highCard(rank1, rank2, rank3, rank4, rank5):
      return UInt32(rank1.rawValue) << 20 | UInt32(rank2.rawValue) << 15 |
          UInt32(rank3.rawValue) << 10 | UInt32(rank4.rawValue) << 5 | UInt32(rank5.rawValue)
    case let .pair(rank1, kickers: rank2, rank3, rank4):
      return UInt32(1 << 26) | UInt32(rank1.rawValue) << 20 | UInt32(rank2.rawValue) << 10 |
          UInt32(rank3.rawValue) << 5 | UInt32(rank4.rawValue)
    case let .twoPair(rank1, rank2, kicker: rank3):
      return UInt32((2 << 26) | UInt32(rank1.rawValue) << 15 | UInt32(rank2.rawValue) << 10 |
                    UInt32(rank3.rawValue))
    case let .threeOfAKind(rank1, kickers: rank2, rank3):
      return UInt32((3 << 26) | UInt32(rank1.rawValue) << 15 | UInt32(rank2.rawValue) << 5 |
                    UInt32(rank3.rawValue))
    case let .straight(rank):
      return UInt32((4 << 26) | UInt32(rank.rawValue))
    case let .flush(rank):
      return UInt32((5 << 26) | UInt32(rank.rawValue))
    case let .fullHouse(triplesRank, pairRank):
      return UInt32((6 << 26) | UInt32(triplesRank.rawValue) << 5 | UInt32(pairRank.rawValue))
    case let .fourOfAKind(rank, kicker: kickerRank):
      return UInt32((7 << 26) | UInt32(rank.rawValue << 5) | UInt32(kickerRank.rawValue))
    case let .straightFlush(rank):
      return UInt32((8 << 26) | UInt32(rank.rawValue))
    }
  }
}

extension PokerHandRank: Comparable {
  public static func < (lhs: PokerHandRank, rhs: PokerHandRank) -> Bool {
    return lhs.score < rhs.score
  }
}

/// A 5-card poker hand.
///
/// Used for comparing between hands. It is hand-written out to store values inline.
public struct PokerHand {
  public let card1: Card
  public let card2: Card
  public let card3: Card
  public let card4: Card
  public let card5: Card
  public let rank: PokerHandRank

  public init(_ cards: (Suit, Rank)...) {
    self.init(cards: cards.map(Card.init))
  }
  public init(cards: [Card]) {
    precondition(cards.count == 5, "\(cards) has invalid size.")
    let sorted = cards.sorted{ $0.index < $1.index }
    rank = PokerHand.computePokerHandRank(sortedCards: sorted)
    card1 = sorted[0]
    card2 = sorted[1]
    card3 = sorted[2]
    card4 = sorted[3]
    card5 = sorted[4]
  }

  public var cards: [Card] {
    [card1, card2, card3, card4, card5]
  }

  private static func computePokerHandRank(sortedCards cards: [Card]) -> PokerHandRank {
    var isFlush = true
    var isStraight = true
    var fourOfAKind: Rank? // Non-nil iff there is a four-of-a-kind.
    var threeOfAKind: Rank? // Non-nil iff there is a three-of-a-kind.
    var pair: Rank? // Non-nil iff there is a pair.
    var secondPair: Rank?

    // Check for a 4-of-a-kind.
    for i in 0...1 {
      if cards[(i+1)..<(i+4)].allSatisfy({ $0.rank == cards[i].rank }) {
        fourOfAKind = cards[i].rank
      }
    }
    // Check for a three-of-a-kind.
    for i in 0...2 {
      if cards[(i+1)..<(i+3)].allSatisfy({ $0.rank == cards[i].rank }) {
        threeOfAKind = cards[i].rank
      }
    }
    // Look for a pair that's not a three of a kind. Pick the highest rank pair.
    // Note: we don't have to worry about four-of-a-kind's confusing pair logic, because pairs don't
    // affect scoring 4-of-a-kind's.
    for i in 0...3 {
      if cards[i].rank != threeOfAKind && cards[i].rank == cards[i+1].rank {
        if pair != nil {
          secondPair = pair
        }
        pair = cards[i].rank
      }
    }

    // Check for straights & flushes
    for i in 1..<5 {
      if cards[i].suit != cards[0].suit {
        isFlush = false
      }
      // Compute straights, special casing aces.
      if cards[i].rank.rawValue != (cards[i-1].rank.rawValue + 1) &&
         !(i == 4 && cards[0].rank == .two && cards[4].rank == .ace) {
        isStraight = false
      }
    }
    if isStraight && isFlush {
      if cards[0].rank == .ten {
        return .straightFlush(.ace)
      }
      if cards[4].rank == .ace && cards[0].rank == .two {
        return .straightFlush(cards[3].rank)
      }
      return .straightFlush(cards[4].rank)
    }
    if isFlush {
      return .flush(cards[4].rank)
    }
    if isStraight {
      if cards[4].rank == .ace && cards[0].rank == .two {
        return .straight(cards[3].rank)
      }
      return .straight(cards[4].rank)
    }
    if let fourOfAKind = fourOfAKind {
      return .fourOfAKind(fourOfAKind,
                          kicker: cards.reversed().first { $0.rank != fourOfAKind }!.rank)
    }
    if let threeOfAKind = threeOfAKind {
      if let pair = pair {
        return .fullHouse(threeOfAKind, pair)
      }
      let kickers = cards.reversed().filter { $0.rank != threeOfAKind }
      return .threeOfAKind(threeOfAKind, kickers: kickers[0].rank, kickers[1].rank)
    }
    if let pair = pair {
      if let secondPair = secondPair {
        return .twoPair(
          pair, secondPair,
          kicker: cards.reversed().first { $0.rank != pair && $0.rank != secondPair }!.rank)
      }
      let kickers = cards.reversed().filter { $0.rank != pair }
      return .pair(pair, kickers: kickers[0].rank, kickers[1].rank, kickers[2].rank)
    }

    return .highCard(cards[4].rank, cards[3].rank, cards[2].rank, cards[1].rank, cards[0].rank)
  }
}

extension PokerHandRank {
  /// This initializer takes all 7 cards (5 community cards + 2 private cards) and computes the best
  /// poker hand rank.
  public init(texasHoldem cards: [Card]) {
    precondition(cards.count == 7, "Cards (\(cards)) must contain exactly 7 cards.")
    // Sort the cards in ascending order.
    let sorted = cards.sorted { $0.index < $1.index }
    // Look for #'s-of-a-kind & flushes.
    var suitCount = Array(repeating: 0, count: Suit.allCases.count)
    var rankCount = Array(repeating: 0, count: Rank.allCases.count + 2)
    for card in sorted {
      suitCount[Int(card.suit.rawValue)] += 1
      rankCount[Int(card.rank.rawValue)] += 1
    }
    let flushSuit = suitCount.firstIndex { $0 > 4 }.map { Suit(rawValue: UInt8($0)) }
    let straightRank = PokerHandRank.findStraight(in: sorted)
    // Check for straight flush / royal flush
    if straightRank != nil, let flushSuit = flushSuit {
      if let straightFlushRank = PokerHandRank.findStraight(
          in: sorted.filter { $0.suit == flushSuit }
      ) {
        self = .straightFlush(straightFlushRank)
        return
      }
    }
    // Iterate through the array looking for n-of-a-kinds. We iterate in forward order, and look at
    // the last element(s) of the resulting arrays for the highest ranks.
    var fourOfAKinds = [Rank]()
    var threeOfAKinds = [Rank]()
    var pairs = [Rank]()
    for (rawRank, count) in rankCount.enumerated() {
      if rawRank == 0 || rawRank == 1 { continue }  // invalid ranks
      let rank = Rank(rawValue: UInt8(rawRank))!
      switch count {
      case 0: break
      case 1: break
      case 2:
        pairs.append(rank)
      case 3:
        threeOfAKinds.append(rank)
      case 4:
        fourOfAKinds.append(rank)
      default:
        fatalError("Unexpected count (\(count)) for rank \(rank).")
      }
    }
    if let fourOfAKindRank = fourOfAKinds.last {
      let kicker = sorted.reversed().first { $0.rank != fourOfAKindRank }!
      self = .fourOfAKind(fourOfAKindRank, kicker: kicker.rank)
      return
    }
    // Check for full house.
    if let tripleRank = threeOfAKinds.last, let pairRank = pairs.last {
      self = .fullHouse(tripleRank, pairRank)
      return
    }
    // Check for flush.
    if let flushSuit = flushSuit {
      self = .flush(sorted.reversed().first { $0.suit == flushSuit }!.rank)
      return
    }
    // Check for straight.
    if let straightRank = straightRank {
      self = .straight(straightRank)
      return
    }
    if let threeOfAKind = threeOfAKinds.last {
      let kickers = sorted.reversed().filter { $0.rank != threeOfAKind }
      self = .threeOfAKind(threeOfAKind, kickers: kickers[0].rank, kickers[1].rank)
      return
    }
    if pairs.count > 1 {
      pairs.reverse()
      let kicker = sorted.reversed().first { $0.rank != pairs[0] && $0.rank != pairs[1] }!.rank
      self = .twoPair(pairs[0], pairs[1],
                      kicker: kicker)
      return
    }
    if let pairRank = pairs.last {
      let kickers = sorted.reversed().filter { $0.rank != pairRank }
      self = .pair(pairRank, kickers: kickers[0].rank, kickers[1].rank, kickers[2].rank)
      return
    }
    self = .highCard(sorted[6].rank, sorted[5].rank, sorted[4].rank, sorted[3].rank,
                     sorted[2].rank)
  }

  private static func findStraight(in cards: [Card]) -> Rank? {
    var consecutiveRanks = 1
    var currentRank = cards[0].rank
    var straightRank: Rank?
    for card in cards {
      if card.rank == currentRank { continue }
      if card.rank.rawValue == currentRank.rawValue + 1 {
        consecutiveRanks += 1
        if consecutiveRanks > 4 {
          straightRank = card.rank
        }
      } else {
        consecutiveRanks = 1
      }
      currentRank = card.rank
    }
    return straightRank
  }
}
