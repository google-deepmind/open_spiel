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

/// Leduc poker is a simplified form of poker used in game theory research.
///
/// There are 2 * (num players + 1) cards in 2 suits. Players receive their private cards, and do a
/// first round of betting. Then a community card is chosen, and a second round of betting occurs.
///
/// Scoring values pairs highly based on rank, and then card ranks. Ties are possible, as the suits
/// are unranked.
///
/// Below is a sequence of a standard 2-player Leduc Poker game:
///  0. The deck consists of Jack, Queen, and King of 2 suits (referred to J0, J1, Q0, Q1, K0, K1).
///     Players ante up into the pot.
///  1. chance chooses a card for player 0: Q1
///  2. chance chooses a card for player 1: J0
///  3. player 0 chooses to raise.
///  4. player 1 chooses to call.
///  5. chance chooses the community card: J1
///  6. player 0 chooses to call.
///  7. player 1 chooses to raise.
///  8. player 0 calls.
///  9. Player 1 wins the entire contents of the pot.
///
public struct LeducPoker: GameProtocol {
  public static let info = GameInfo(
    shortName: "leduc_poker",
    longName: "Leduc Poker",
    dynamics: .sequential,
    chanceMode: .explicitStochastic,
    information: .imperfect,
    utility: .zeroSum,
    rewardModel: .terminal,
    maxPlayers: 10,
    minPlayers: 2,
    providesInformationStateString: true,
    providesInformationStateTensor: true
  )

  // Representations of cards in Leduc poker. Assumes 2 suits of equivalent value.
  public struct Card: Hashable {
    var value: Int
    public init(value: Int) {
      self.value = value
    }
    public var suit: Bool {
      value & 1 == 0
    }
    public var rank: Int {
      value / 2  // Integer division intended!
    }
  }

  public enum Action: Hashable {
    // Chance can choose a card.
    case card(Card)
    // Players can either fold, call, or raise.
    case fold
    case call
    case raise
  }

  public enum Round {
    case one
    case two
  }

  /// The State type captures all the progress of the game.
  ///
  /// In order to make it efficient to apply updates to the state, the state struct materializes a
  /// number of values that track the progress of the game, even though in theory everything could
  /// be re-computed by looking at the history.
  public struct State: StateProtocol {
    public typealias Action = LeducPoker.Action
    public let game: LeducPoker

    /// The current player, starting in the chance state.
    public var currentPlayer: Player = .chance
    /// Number of calls this round (total, not per player).
    public var callCount = 0
    // Number of raises made in the round (not per player).
    public var raiseCount = 0
    /// Round number (Moves monotonically from .one to .two).
    public var round = Round.one
    /// The current 'level' of the bet.
    public var stakes = 1
    /// The number of winning players. (Only defined after resolveWinner() has been called.)
    public var winnerCount: Int?
    /// The number of chips in the pot.
    public var pot: Int
    /// The public card (revealed after round 1).
    public var publicCard: Card?
    /// Number of cards remaining; not equal to deck.size().
    public var deckSize: Int
    /// How many private cards have currently been dealt.
    public var privateCardsDealt = 0
    /// Number players still in (not folded).
    public var remainingPlayers: Int

    /// Each player's single private card, indexed by playerID. (Initially empty.)
    public var privateCards: [Card] = []
    /// How much money each player has, indexed by playerID.
    public var money: [Double]
    /// How much each player has contributed to the pot, indexed by playerID.
    public var ante: [Int]
    /// Flag for whether the player has folded, indexed by playerID.
    public var folded: [Bool]
    /// Is this player a winner? Indexed by playerID.
    public var winners: [Bool]

    /// Cards by value (0-6 for standard 2-player game, nil if no longer in the deck.)
    var deck: [Card?]

    /// Sequence of actions for each round. Needed to report information state.
    var round1Sequence: [Action] = []
    var round2Sequence: [Action] = []

    /// History of the game.
    public var history: [Action] = []

    /// The return/reward for a given player (non-zero only when the game is over).
    public func utility(for player: Player) -> Double {
      if !isTerminal { return 0 }
      guard case let .player(playerID) = player else {
        preconditionFailure("Invalid player \(player) for LeducPoker.State.utility(for:)")
      }
      return money[playerID] - game.startingMoney
    }

    public func informationStateString(for player: Player) -> String {
      guard case let .player(playerID) = player else {
        preconditionFailure("Invalid player \(player) for LeducPoker.State.informationState")
      }
      guard playerID < game.playerCount else {
        preconditionFailure("Invalid playerID \(playerID).")
      }

      let metadataStr = "[Round \(round)][Player: \(currentPlayer)][Pot: \(pot)]"
      let moneyStr = money.map { String($0) }.joined(separator: " ")
      let privateStr = String(describing: privateCards[playerID])
      let round1Str = round1Sequence.map { String(describing: $0) }.joined(separator: " ")
      let round2Str = round2Sequence.map { String(describing: $0) }.joined(separator: " ")
      let publicCardStr = String(describing: publicCard)
      let roundsStr = "[Round 1: \(round1Str)][Public card: \(publicCardStr)][Round 2: \(round2Str)]"
      return "\(metadataStr)[Money: \(moneyStr)][Private: \(privateStr)]\(roundsStr)"
    }

    public func informationStateTensor(for player: Player) -> [Double] {
      guard case let .player(playerID) = player else {
        preconditionFailure("Invalid player \(player) for LeducPoker.State.informationState")
      }

      // Layout of observation:
      //  my player number: num_player bits
      //  my card: deckSize bits
      //  public card: deckSize bits
      //  first round sequence: (max round seq length)*2 bits
      //  second round sequence: (max round seq length)*2 bits

      var state = Array<Double>(repeating: 0, count: game.informationStateTensorShape[0])

      var offset = 0

      // Mark who I am.
      state[playerID] = 1
      offset += game.playerCount

      // Mark my private card.
      if !privateCards.isEmpty {
        state[offset + privateCards[playerID].value] = 1
      }
      offset += game.totalCards

      // Mark the public card
      if let publicCard = publicCard {
        state[offset + publicCard.value] = 1
      }
      offset += game.totalCards

      for round in [round1Sequence, round2Sequence] {
        for (i, action) in round.enumerated() {
          assert(offset + i + 1 < state.count,
                 "Offset: \(offset), i: \(i), state.count: \(state.count), state: \(state).")
          if action == .call {
            // Encode call as 10.
            state[offset + (2 * i)] = 1
          } else {
            // Encode raise as 01.
            state[offset + (2 * i) + 1] = 0
          }
        }
        // Increment the offset.
        offset += game.maxGameLength
      }

      return state
    }
  }

  public let playerCount: Int
  // Only 2 suits in the deck.
  let suitCount = 2
  let firstRaiseAmount = 2
  let secondRaiseAmount = 4
  let totalRaisesPerRound = 2
  let maxRaises = 2
  let startingMoney = 100.0
  let ante = 1
  public var allActions: [Action] { [.call, .fold, .raise] }
  let totalCards: Int
  public var minUtility: Double {
    // In poker, the utility is defined as the money a player has at the end of
    // the game minus the money the player had before starting the game.
    // The most any single player can lose is the maximum number of raises per
    // round times the amounts of each of the raises, plus the original chip they
    // put in to play.
    -1 * Double(totalRaisesPerRound * firstRaiseAmount +
                totalRaisesPerRound * secondRaiseAmount + 1)
  }
  public var maxUtility: Double {
    // In poker, the utility is defined as the money a player has at the end of
    // the game minus then money the player had before starting the game.
    // The most a player can win *per opponent* is the most each player can put
    // into the pot, which is the raise amounts on each round times the maximum
    // number raises, plus the original chip they put in to play.
    Double((playerCount - 1) * (totalRaisesPerRound * firstRaiseAmount +
                                totalRaisesPerRound * secondRaiseAmount + 1))
  }
  /// The total utility for all players.
  /// Leduc poker is a zero-sum game.
  public var utilitySum: Double? { 0 }

  public var maxGameLength: Int {
    2 * (2 + (playerCount - 1) * 2 + (playerCount - 2))
  }
  public var initialState: State {
    State(game: self)
  }
  public var informationStateTensorShape: [Int] {
    [playerCount + (totalCards * 2) + (maxGameLength * 2)]
  }

  public init(playerCount: Int = 2) {
    precondition(playerCount >= 2, "Player count (\(playerCount)) must be >= 2")
    precondition(playerCount <= 10, "Player count (\(playerCount)) must be <= 10")
    self.playerCount = playerCount
    totalCards = (playerCount + 1) * suitCount
  }
}

extension LeducPoker.State {
  init(game: LeducPoker) {
    self.game = game
    self.deckSize = (game.playerCount + 1) * game.suitCount
    self.pot = game.ante * game.playerCount
    self.remainingPlayers = game.playerCount
    self.winners = Array(repeating: false, count: game.playerCount)
    self.deck = (0..<self.deckSize).map(LeducPoker.Card.init)
    self.money = Array(repeating: game.startingMoney - Double(game.ante), count: game.playerCount)
    self.ante = Array(repeating: game.ante, count: game.playerCount)
    self.folded = Array(repeating: false, count: game.playerCount)
  }

  public mutating func apply(_ action: Action) {
    history.append(action)
    // Compute the book-keeping state based on the action.
    switch action {
    case let .card(card):
      precondition(currentPlayer == .chance,
                   "Invalid action \(action) for player \(currentPlayer)")
      precondition(card.value >= 0 && card.value < deck.count,
                   "Invalid card: \(card). deck.count: \(deck.count)")
      precondition(deck[card.value] != nil, "Invalid card: \(card). deck: \(deck)")
      deck[card.value] = nil
      deckSize -= 1
      if privateCardsDealt < game.playerCount {
        assert(round == .one, "Internal game state error.")
        privateCards.append(card)
        privateCardsDealt += 1
        if privateCardsDealt == game.playerCount {
          currentPlayer = .player(0)  // When all private cards are dealt, move to player 0.
        }
        return
      }
      // Round 2: A single public card.
      assert(round == .two, "Internal game state error.")
      publicCard = card
      currentPlayer = nextPlayer // Let's bet!
      return

    case .fold:
      guard case let .player(playerID) = currentPlayer else {
        fatalError("Invalid action \(action) for current player \(currentPlayer)")
      }
      folded[playerID] = true
      remainingPlayers -= 1

    case .call:
      guard case let .player(playerID) = currentPlayer else {
        fatalError("Invalid action \(action) for current player \(currentPlayer)")
      }
      assert(stakes >= ante[playerID], "Player ID \(playerID) overbet? \(stakes) > \(ante)")
      let amount = stakes - ante[playerID]
      ante(currentPlayer, amount: amount)
      callCount += 1
      append(action: action)

    case .raise:
      guard case let .player(playerID) = currentPlayer else {
        fatalError("Invalid action \(action) for current player \(currentPlayer)")
      }
      precondition(raiseCount <= game.maxRaises, "raiseCount \(raiseCount) too high!")
      raiseCount += 1

      // First, this player must match the current stakes.
      let callAmount = stakes - ante[playerID]
      assert(callAmount >= 0, "Call amount: \(callAmount)")
      if callAmount > 0 {
        ante(currentPlayer, amount: callAmount)
      }
      // Now, raise the stakes.
      var raiseAmount: Int
      switch round {
      case .one: raiseAmount = game.firstRaiseAmount
      case .two: raiseAmount = game.secondRaiseAmount
      }
      stakes += raiseAmount
      ante(currentPlayer, amount: raiseAmount)
      callCount = 0
      append(action: action)
    }
    if isTerminal {
      resolveWinner()
    } else if isReadyForNextRound {
      startNextRound()
    } else {
      currentPlayer = nextPlayer
    }
  }

  public var legalActionsMask: [Bool] {
    switch currentPlayer {
    case let .player(playerID):
      return game.allActions.map {
        switch $0 {
          // Can always call / check
          case .call: return true
          // Can't just randomly fold; only allow fold when under pressure.
          case .fold: return stakes > ante[playerID]
          // Can raise if game hasn't reached `maxRaises`.
          case .raise: return raiseCount < game.maxRaises
          case .card: preconditionFailure("invalid action for non-chance player")
        }
      }
    default:
      preconditionFailure("legalActionsMask is valid only for actual players")
    }
  }

  public var chanceOutcomes: [Action: Double] {
    precondition(currentPlayer == .chance)
    var outcomes: [Action: Double] = [:]
    let p = 1.0 / Double(deckSize)
    for case let card? in deck {
      outcomes[.card(card)] = p
    }
    return outcomes
  }

  var nextPlayer: Player {
    var curRealPlayer: Int
    switch currentPlayer {
    case .chance: curRealPlayer = -1
    case let .player(playerID): curRealPlayer = playerID
    case .terminal: return .terminal
    case .invalid: return .invalid
    case .simultaneous: fatalError("Internal error in LeducPoker. \(self)")
    }
    for i in 1..<game.playerCount {
      let playerID = (curRealPlayer + i) % game.playerCount
      assert(playerID >= 0, "Player: \(playerID)")
      assert(playerID < game.playerCount, "Player: \(playerID)")
      if !folded[playerID] {
        return .player(playerID)
      }
    }
    fatalError("No valid next player. \(self)")
  }

  public var isTerminal: Bool {
    remainingPlayers == 1 || (round == .two && isReadyForNextRound)
  }

  var isReadyForNextRound: Bool {
    (raiseCount == 0 && callCount == remainingPlayers) ||
    (raiseCount > 0 && callCount == remainingPlayers - 1)
  }

  mutating func startNextRound() {
    precondition(round == .one, "Can only call next round when in the 1st round!")
    round = .two
    raiseCount = 0
    callCount = 0
    currentPlayer = .chance  // Public card.
  }

  // Append round specific information.
  mutating func append(action: Action) {
    switch round {
      case .one: round1Sequence.append(action)
      case .two: round2Sequence.append(action)
    }
  }

  mutating func ante(_ player: Player, amount: Int) {
    guard case let .player(playerID) = player else {
      fatalError("Cannot ante player \(player).")
    }
    pot += amount
    ante[playerID] += amount
    money[playerID] -= Double(amount)
  }
  // Padded betting sequence?

  public func rankHand(for player: Player) -> Int {
    guard case let .player(pId) = player else {
      fatalError("Invalid rank for player: \(player)")
    }
    guard let publicCard = publicCard else {
      fatalError("Cannot rank hand before public card is set!")
    }
    let hand = [publicCard, privateCards[pId]].sorted(by: { $0.value < $1.value })
    if hand[0].rank == hand[1].rank {
      // Pair! Offset by deckSize^2 to rank higher than every singles combo.
      return deck.count * deck.count + hand[0].rank
    }
    // Otherwise, card value dominates. No high / low suits, so go based off of card rank.
    // Note: assumes cards are sorted!
    return hand[1].rank * deck.count + hand[0].rank
  }

  public mutating func resolveWinner() {
    precondition(winnerCount == nil, "winnerCount is not nil: \(winnerCount!)")
    currentPlayer = .terminal
    if remainingPlayers == 1 {
      // Only one left? They get the pot!
      for playerID in 0..<game.playerCount {
        if !folded[playerID] {
          winnerCount = 1
          winners[playerID] = true
          money[playerID] += Double(pot)
          pot = 0
          return
        }
      }
    }
    // Otherwise, showdown!
    precondition(publicCard != nil, "Invalid state: public card is nil when resolving winner.")
    var bestHandRank = -1
    winnerCount = 0
    for playerID in 0..<game.playerCount {
      let rank = rankHand(for: .player(playerID))
      if rank > bestHandRank {
        // Beat the current best hand! Clear the winners list, then add playerID.
        winners.resetToFalse()
        winners[playerID] = true
        winnerCount = 1
        bestHandRank = rank
      } else if rank == bestHandRank {
        // Tied with the best hand rank, so this player is a winner as well.
        winnerCount! += 1
        winners[playerID] = true
      }
    }
    for playerID in 0..<game.playerCount {
      if winners[playerID] {
        money[playerID] += Double(pot) / Double(winnerCount!)
      }
    }
    pot = 0
  }
}

extension LeducPoker.Action: CustomStringConvertible {
  public var description: String {
    switch self {
      case let .card(card): return "\(card)"
      case .fold: return "fold"
      case .call: return "call"
      case .raise: return "raise"
    }
  }
}

extension LeducPoker.State: CustomStringConvertible {
  public var description: String {
    "TODO LeducPoker.State.description!"
  }
}

extension Array where Element == Bool {
  // Helper function to in-place reset the winners array to all false.
  mutating func resetToFalse() {
    for i in indices {
      self[i] = false
    }
  }
}
