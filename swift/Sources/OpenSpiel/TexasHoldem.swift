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

/// Texas Hold'em Poker is the "classic" poker.
///
/// This poker variant is played with a full 52-card deck. This implementation
/// supports between 2 and 6 players. Player 0 is always the dealer. When there
/// are only 2 players, player 0 (the dealer) puts in the big blind.
public struct TexasHoldem: GameProtocol {
  public static let info = GameInfo(
    shortName: "texas_holdem",
    longName: "Texas Hold'em",
    dynamics: .sequential,
    chanceMode: .explicitStochastic,
    information: .imperfect,
    utility: .zeroSum,
    rewardModel: .terminal,
    maxPlayers: 6,
    minPlayers: 2,
    providesInformationState: true,
    providesInformationStateAsNormalizedVector: true
  )

  /// The available actions in the Texas Holdem game.
  public enum Action: Hashable {
    /// Chance can choose a card.
    case card(Card)

    /// A player can fold (i.e. abandon their hand).
    case fold
    /// A player can call (if no bet has been made, this is also known as a check).
    case call
    /// A player can raise the bet level.
    ///
    /// Note: raises are with respect to a fraction of the pot. For example, if the pot contains
    /// 200 units, a 1x raise would increase the pot value 400.
    case raise(Double)
    /// Bet all the player's remaining money.
    case allIn
  }

  /// Texas Holdem has a number of phases of a game, centering around a set of
  /// four betting rounds, followed by a final reveal round.
  public enum Round: Int {
    /// Pre-Flop is the betting round after the private cards have been delt,
    /// but before the first 3 community cards are revealed.
    case preFlop
    /// Post-flop is the round of betting after the flop.
    case postFlop
    /// Post-Turn is the round after 4 total community cards have been revealed.
    ///
    /// Note: the minimum bet size is now 2x the big blind.
    case postTurn
    /// Post-River: the final betting round.
    case postRiver

    /// This is the minimum bet size (in multiples of the big blind) for a given round.
    var minimumBetMultiple: Int {
      switch self {
      case .preFlop, .postFlop: return 1
      case .postTurn, .postRiver: return 2
      }
    }
  }

  public struct PrivateCards {
    public var first: Card?
    public var second: Card?
    public var hasBeenRevealed = false
    fileprivate init() {}
    public init(first: Card, second: Card) {
      self.first = first
      self.second = second
    }
  }

  public struct State: StateProtocol {
    public typealias Action = TexasHoldem.Action
    /// The game configuration.
    public let game: TexasHoldem
    /// The current player, starting in the chance state.
    public var currentPlayer: Player = .chance
    /// History of the game.
    public var history: [Action] = []
    /// The number of non-folded players.
    public var nonFoldedCount: Int
    /// Current size of the pot.
    public var pot = 0
    /// The current bet level.
    public var betLevel: Int
    /// Counter of the number of actions since the last raise. Used to determine
    /// when to advance to the next round.
    public var actionsSinceRaise = 0
    /// The phase of the game we're in.
    public var round: Round = .preFlop
    /// The cards in the deck (nil if they are no longer in the deck).
    public var deck: [Card?]
    /// Community cards.
    ///
    /// They are maintained in the order they appear, with the exception that
    /// the flop is always sorted to slightly reduce the state space.
    public var communityCards: [Card] = []
    /// The amount of money each player has, indexed by playerID.
    public var money: [Int]
    /// True iff playerID index has folded, false otherwise, indexed by playerID.
    public var folded: [Bool]
    /// The private cards, indexed by player id. They are always sorted by
    /// Card index.
    public var privateCards: [PrivateCards]
    /// Determines whether the game is over or not.
    public var isTerminal: Bool { currentPlayer == .terminal }
  }

  public var initialState: TexasHoldem.State {
    State(game: self)
  }

  public let playerCount: Int

  /// In this formulation, players begin with 10000 chips, and must bet in whole chips.
  public var initialMoney: Int { 10000 }
  public let smallBlind: Int
  public let bigBlind: Int
  public var minUtility: Double { 0 }
  public var maxUtility: Double {
    Double(playerCount * initialMoney)
  }
  /// The total utility for all players.
  /// Texas Hold'em poker is a zero-sum game.
  public var utilitySum: Double? { 0 }

  public var maxGameLength: Int {
    // The maximum game length is when everyone iteratively bets the minimum bet size (bigBlind),
    // plus a maximum number of calls & folds.
    ((playerCount * initialMoney) / bigBlind) + playerCount * 3
  }

  /// The bet discretization are all fractions of the pot that the player can bet.
  ///
  /// For example, a bet discretization could be: [0.5, 1, 2]. If the pot is initially 100, and
  /// player 2 raises by 100 (pot is now 200), a 1x pot-size raise from player 3 results in a total
  /// bet of 400.
  public let betDiscretization: [Double]

  /// All available player actions.
  ///
  /// Beware, the size of the raise chosen can be variable, and the minimum amount changes over the
  /// course of the game.
  public var allActions: [TexasHoldem.Action] {
    [.fold, .call, .allIn] + betDiscretization.map { .raise($0) }
  }

  public var informationStateNormalizedVectorShape: [Int] {
    []  // TODO: IMPLEMENT ME!
  }

  public init(
    playerCount: Int,
    betDiscretization: [Double] = [0.5, 1, 2],
    smallBlind: Int = 50,
    bigBlind: Int = 100
  ) {
    self.playerCount = playerCount
    self.betDiscretization = betDiscretization
    self.smallBlind = smallBlind
    self.bigBlind = bigBlind
  }
}

extension TexasHoldem.State {
  public init(game: TexasHoldem) {
    self.game = game
    self.money = Array(repeating: game.initialMoney, count: game.playerCount)
    self.folded = Array(repeating: false, count: game.playerCount)
    self.privateCards = Array(
      repeating: TexasHoldem.PrivateCards(),
      count: game.playerCount)
    self.deck = (0..<52).map(Card.init)
    self.betLevel = game.bigBlind
    self.nonFoldedCount = game.playerCount
    // Set blinds.
    self.money[1] -= game.smallBlind
    let bigBlindPlayer = game.playerCount == 2 ? 0 : 2
    self.money[bigBlindPlayer] -= game.bigBlind
    pot += (game.smallBlind + game.bigBlind)
  }

  /// The legal actions to be made in this state.
  public var legalActionsMask: [Bool] {
    switch currentPlayer {
    case .chance:
      fatalError("legalActionsMask not available when currentPlayer == .chance!")
    case let .player(playerID):
      assert(!folded[playerID])
      if money[playerID] == 0 {
        var mask = Array(repeating: false, count: game.allActions.count)
        mask[1] = true  // .call
        return mask
      }
      let allowedRaiseAmounts = game.betDiscretization.map { potFraction -> Bool in
        let chipAmount = Int(potFraction * Double(pot))
        return chipAmount > minimumBetAmount && chipAmount < game.initialMoney
      }
      if (game.initialMoney - money[playerID]) == betLevel {
        let base = [false, true, true]  // .fold disallowed, .call and .allIn allowed.
        return base + allowedRaiseAmounts
      }
      return [true, true, true] + allowedRaiseAmounts
    default:
      fatalError("Invalid current player \(currentPlayer).")
    }
  }

  public mutating func apply(_ action: TexasHoldem.Action) {
    history.append(action)
    switch action {
    case let .card(card):
      precondition(currentPlayer == .chance)
      precondition(deck[Int(card.index)] != nil)
      deck[Int(card.index)] = nil  // remove the card from the deck.
      switch round {
      case .preFlop:
        dealPrivateCard(card)
        return
      case .postFlop:
        assert(communityCards.count < 3)
        communityCards.append(card)
        if communityCards.count == 3 {
          communityCards.sort { $0.index < $1.index }
          startNextRound()
        }
      case .postTurn, .postRiver:
        communityCards.append(card)
        startNextRound()
      }
    case .fold:
      guard case let .player(playerID) = currentPlayer else {
        fatalError("Invalid fold action for \(currentPlayer)")
      }
      folded[playerID] = true
      actionsSinceRaise += 1
      nonFoldedCount -= 1
      advanceToNextPlayer()
    case .call:
      call()
    case let .raise(fraction):
      let amount = Int(Double(pot) * fraction)
      betLevel += amount
      actionsSinceRaise = 0
      call()
    case .allIn:
      betLevel = game.initialMoney
      actionsSinceRaise = 0
      call()
    }
  }

  /// The probability associated with the corresponding actions.
  public var chanceOutcomes: [TexasHoldem.Action: Double] {
    assert(currentPlayer == .chance)
    let chanceActions = deck.compactMap { $0.map(Action.card) }
    let p = 1.0 / Double(chanceActions.count)
    var outcomes = [TexasHoldem.Action: Double]()
    for action in chanceActions {
      outcomes[action] = p
    }
    return outcomes
  }

  /// The return / reward for a given player (non-zero only when the game is over).
  public func utility(for player: Player) -> Double {
    if !isTerminal { return 0 }
    guard case let .player(playerID) = player else {
      preconditionFailure(
        "Invalid player \(player) passed to TexasHoldem.State.utility(for:)")
    }
    precondition(
      playerID < game.playerCount,
      "Invalid player \(player). Only \(game.playerCount) players.")
    return Double(money[playerID] - game.initialMoney)
  }
  /// Computes a string-based information state for a given player in the current state.
  public func informationState(for player: Player) -> String {
    fatalError("IMPLEMENT ME!")  // TODO
  }
  /// Computes a normalized vector representing the information state.
  public func informationStateAsNormalizedVector(for player: Player) -> [Double] {
    fatalError("UMIMPLEMENTED!") // TODO
  }
}

// Helper functions.
extension TexasHoldem.State {
  /// The minimum amount legally allowed at this phase of the game.
  public var minimumBetAmount: Int {
    round.minimumBetMultiple * game.bigBlind
  }

  private mutating func dealPrivateCard(_ card: Card) {
    for i in 0..<game.playerCount {
      if privateCards[i].first == nil {
        privateCards[i].first = card
        return
      }
      if privateCards[i].second == nil {
        privateCards[i].second = card
        // Reorder the cards if necessary.
        if privateCards[i].first!.index > privateCards[i].second!.index {
          (privateCards[i].first, privateCards[i].second) =
            (privateCards[i].second, privateCards[i].first)
        }
        // We've dealt our final private card. Let's play!
        if i == game.playerCount - 1 {
          switch game.playerCount {
          case 2:
            currentPlayer = .player(1)
          case 3:
            currentPlayer = .player(0)
          default:
            currentPlayer = .player(3)
          }
        }
        return
      }
    }
  }

  // Trues up the current player to the betLevel, adjusting the pot and money.
  private mutating func call() {
    guard case let .player(playerID) = currentPlayer else {
      fatalError("Call action for \(currentPlayer)")
    }
    let newMoney = game.initialMoney - betLevel
    let addedMoney = money[playerID] - newMoney
    assert(addedMoney >= 0, "\(addedMoney) >= 0; \(self)")
    pot += addedMoney
    money[playerID] = newMoney
    actionsSinceRaise += 1
    advanceToNextPlayer()
  }

  private mutating func startNextRound() {
    for index in 1..<game.playerCount {
      if !folded[index] {
        currentPlayer = .player(index)
        return
      }
    }
    preconditionFailure("No non-folded player?!?! \(self)")
  }

  private mutating func advanceToNextPlayer() {
    guard case let .player(playerID) = currentPlayer else {
      fatalError("Invalid current player \(currentPlayer)")
    }
    if nonFoldedCount == 1 {
      // We have a winner!
      currentPlayer = .terminal
      for id in 0..<game.playerCount where !folded[id] {
        assert(pot != 0, "Internal corruption!")
        money[id] += pot
        pot = 0
      }
      return
    }
    // Check to see if we're done with this round.
    if actionsSinceRaise == nonFoldedCount {
      if round == .postRiver {
        computeWinner()
        return
      }
      // Everyone's squared up; advance to the next round.
      actionsSinceRaise = 0
      round = TexasHoldem.Round(rawValue: round.rawValue + 1)!
      currentPlayer = .chance
      return
    }
    // Determine the next non-folded player.
    for offset in 1..<game.playerCount {
      let index = (playerID + offset) % game.playerCount
      if folded[index] {
        continue
      }
      currentPlayer = .player(index)
      return
    }
    // There should always be at least 2 non-folded players.
    fatalError("Corrupted internal state:\n\(self)")
  }

  private mutating func computeWinner() {
    var bestHands: [PokerHandRank?] = Array(repeating: nil, count: game.playerCount)
    for index in 0..<game.playerCount {
      if folded[index] { continue }
      bestHands[index] = computeBestHand(for: index)
    }
    let bestScore = bestHands.map { $0?.score ?? 0 }.max()
    let winnerIndices = (0..<game.playerCount).filter {
      bestHands[$0]?.score == bestScore
    }
    let perPlayerWinnings = pot / winnerIndices.count
    for index in winnerIndices {
      money[index] += perPlayerWinnings
    }
    pot = 0
    currentPlayer = .terminal
  }

  private func computeBestHand(for playerIndex: Int) -> PokerHandRank {
    return PokerHandRank(texasHoldem: communityCards +
                         [privateCards[playerIndex].first!, privateCards[playerIndex].second!])
  }
}

extension TexasHoldem.PrivateCards: CustomStringConvertible {
  public var description: String {
    func strForCard(_ card: Card?) -> String {
      if let card = card {
        return card.description
      }
      return "_"
    }
    return "(\(strForCard(first)), \(strForCard(second)))"
  }
}

extension TexasHoldem.State: CustomStringConvertible {
  public var description: String {
    func describePlayer(_ playerID: Int) -> String {
      return """
        \(money[playerID])\(folded[playerID] ? " [folded]" : "") - \
        \(privateCards[playerID])
        """
    }

    let nonChanceActions = history.filter {
      switch $0 {
      case .card:
        return false
      default:
        return true
      }
    }

    return """
      Round: \(round) Current player: \(currentPlayer). Players: \
      \((0..<game.playerCount).map(describePlayer)); Community Cards: \
      \(communityCards) Pot: \(pot). History: \(nonChanceActions)
      """
  }
}
