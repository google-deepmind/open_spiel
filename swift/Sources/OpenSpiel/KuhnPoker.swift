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

/// A simple game that includes chance and imperfect information
/// http://en.wikipedia.org/wiki/Kuhn_poker
///
/// For more information on this game (e.g. equilibrium sets, etc.) see
/// http://poker.cs.ualberta.ca/publications/AAAI05.pdf
///
/// The multiplayer (n>2) version is the one described in
/// http://mlanctot.info/files/papers/aamas14sfrd-cfr-kuhn.pdf
///
/// Parameters:
///     "playerCount"       int    number of players               (default = 2)
public struct KuhnPoker: GameProtocol {
  public static let info = GameInfo(
    shortName: "kuhn_poker",
    longName: "Kuhn Poker",
    dynamics: .sequential,
    chanceMode: .explicitStochastic,
    information: .imperfect,
    utility: .zeroSum,
    rewardModel: .terminal,
    maxPlayers: 10,
    minPlayers: 2,
    providesInformationState: true,
    providesInformationStateAsNormalizedVector: true
  )

  public enum Action: Hashable {
    case card(Int)
    case pass
    case bet
  }

  public struct State: StateProtocol {
    public typealias Action = KuhnPoker.Action
    public let game: KuhnPoker
    public var history: [Action] = []
    var firstBettor: Player = .invalid
    var cardDealt: [Player]
    var winner: Player = .invalid
    var pot: Int
    public func utility(for player: Player) -> Double {
      if !isTerminal { return 0 }
      let bet = didBet(player) ? 2 : 1
      return Double(player == winner ? (pot - bet) : -bet)
    }
    /// Information state is card then bets, e.g. 1pb.
    public func informationState(for player: Player) -> String {
      switch player {
      case let .player(playerID) where playerID < game.playerCount:
        if history.count <= playerID { return "" }
        var str = history[playerID].description
        if history.count > game.playerCount {
          str.append(history[game.playerCount...].map { $0.description }.joined(separator: ""))
        }
        return str
      default:
        fatalError()
      }
    }

    public func informationStateAsNormalizedVector(for player: Player) -> [Double] {
      // Initialize the vector with zeros.
      var values = Array<Double>(repeating: 0, count: 6 * game.playerCount - 1)
      switch player {
      case let .player(playerID) where playerID < game.playerCount:
        // The current player
        values[playerID] = 1
        // The player's card, if one has been dealt.
        if history.count > playerID, case let .card(cardID) = history[playerID] {
          values[game.playerCount + cardID] = 1
        }
        // Betting sequence.
        for i in game.playerCount..<history.count {
          if case let .card(cardID) = history[i] {
            values[1 + 2 * i + cardID] = 1
          }
        }
      default:
        fatalError()
      }
      return values
    }
  }

  public init(playerCount: Int = 2) {
    self.playerCount = playerCount
  }

  public let playerCount: Int
  public var allActions: [Action] { [.pass, .bet] }

  public var minUtility: Double { -2.0 }
  public var maxUtility: Double {
    return Double(playerCount - 1) * 2
  }
  /// The total utility for all players.
  /// Kuhn poker is a zero-sum game.
  public var utilitySum: Double? { 0 }

  public var maxGameLength: Int {
    return playerCount * 2 - 1
  }

  public var initialState: State {
    return State(game: self)
  }

  public var informationStateNormalizedVectorShape: [Int] {
    return [6 * playerCount - 1]
  }
}

public extension KuhnPoker.State {
  init(game: KuhnPoker) {
    self.cardDealt = Array(repeating: .invalid, count: game.playerCount + 1)
    self.pot = game.playerCount
    self.game = game
  }

  var currentPlayer: Player {
    if isTerminal {
      return .terminal
    } else {
      return history.count < game.playerCount ? .chance : .player(history.count % game.playerCount)
    }
  }

  var isTerminal: Bool {
    return winner != .invalid
  }

  mutating func apply(_ action: Action) {
    // Additional book-keeping
    switch action {
    case let .card(cardID):
      // Give card cardID to player history.count (currentPlayer will return
      // Player.chance, so we use this instead).
      cardDealt[cardID] = .player(history.count)
    case .bet:
      if firstBettor == .invalid {
        firstBettor = currentPlayer
      }
      pot += 1
    case .pass:
      break
    }
    history.append(action)

    // Check for the game being over.
    let actions = history.count - game.playerCount
    switch firstBettor {
    case .invalid:
      // Nobody bet; the winner is the person with the highest card dealt,
      // which is either the highest or the next-highest card.
      // Losers lose 1, winner wins 1 * (game.playerCount - 1)
      if actions == game.playerCount {
        winner = cardDealt[game.playerCount]
        if winner == .invalid {
          winner = cardDealt[game.playerCount - 1]
        }
      }
    case let .player(firstBettorID) where actions == game.playerCount + firstBettorID:
      // There was betting; so the winner is the person with the highest card
      // who stayed in the hand.
      // Check players in turn starting with the highest card.
      winner = cardDealt.last { $0 != .invalid && didBet($0) } ?? winner
    default:
      break
    }
  }

  var legalActionsMask: [Bool] {
    precondition(currentPlayer != .chance)
    return [true, true]
  }

  var chanceOutcomes: [Action: Double] {
    precondition(currentPlayer == .chance)
    var outcomes: [Action: Double] = [:]
    let p = 1.0 / Double(game.playerCount + 1 - history.count)
    for (cardID, player) in cardDealt.enumerated() where player == .invalid {
      outcomes[.card(cardID)] = p
    }
    return outcomes
  }
}

extension KuhnPoker.Action: CustomStringConvertible {
  public var description: String {
    switch self {
    case let .card(cardID): return "\(cardID)"
    case .pass: return "p"
    case .bet: return "b"
    }
  }
}

extension KuhnPoker.State: CustomStringConvertible {
  public var description: String {
    return history.map { $0.description }.joined(separator: " ")
  }
}

extension KuhnPoker.State {
  /// Whether the specified player has made a bet
  public func didBet(_ player: Player) -> Bool {
    switch (player, firstBettor) {
    case let (.player(playerID), .player(firstBettorID)) where playerID == firstBettorID:
      return true
    case (let .player(playerID), let .player(firstBettorID)) where playerID > firstBettorID:
      return history[game.playerCount + playerID] == .bet
    case (let .player(playerID), .player):
      return history[game.playerCount * 2 + playerID] == .bet
    default: return false
    }
  }
}
