import TensorFlow

/// Simple game of Noughts and Crosses:
/// https://en.wikipedia.org/wiki/Tic-tac-toe
///
/// Parameters: none

public class TicTacToe: GameProtocol {
  public static let info = GameInfo(
    shortName: "tic_tac_toe",
    longName: "Tic Tac Toe",
    dynamics: .sequential,
    chanceMode: .deterministic,
    information: .perfect,
    utility: .zeroSum,
    rewardModel: .terminal,
    maxPlayers: 2,
    minPlayers: 2,
    providesInformationStateString: true,
    providesInformationStateTensor: true
  )

  /// An action represents the squares that can be marked.
  public struct Action: Hashable {
    let x: Int
    let y: Int
    public init(x: Int, y: Int) {
      self.x = x
      self.y = y
    }
  }

  /// A State represents a time during the game.
  public struct State: StateProtocol {
    public let game: TicTacToe
    public var history: [Action] = []
    public var board: [Player?]
    public var currentPlayer: Player = .player(0)
    public var winner: Player?
    public func utility(for player: Player) -> Double {
      if winner == nil { return 0 }
      if winner == player {
        return 1
      } else {
        return -1
      }
    }

    public func informationStateString(for player: Player) -> String {
      history.map { String(describing: $0) }.joined(separator: " ")
    }

    public func informationStateTensor(for player: Player) -> [Double] {
      var informationState: [Double] = Array(
        repeating: 0,
        count: game.informationStateTensorShape[0])
      let viewSize = game.boardSize * game.boardSize
      for i in 0..<game.boardSize {
        for j in 0..<game.boardSize {
          let coordinate = TicTacToe.Action(x: i, y: j)
          let viewOffset = game.boardSize * i + j
          switch self[coordinate] {
          case nil:
            informationState[viewOffset] = 1
          case .player(0):
            informationState[1 * viewSize + viewOffset] = 1
          case .player(1):
            informationState[2 * viewSize + viewOffset] = 1
          default:
            fatalError("Unknown player value at \(coordinate): \(self[coordinate]!). \(self)")
          }
        }
      }
      return informationState
    }
  }

  public var minUtility: Double { -1.0 }
  public var maxUtility: Double { 1.0 }
  /// The total utility for all players.
  /// Tic-tac-toe is a zero-sum game.
  public var utilitySum: Double? { 0 }

  public var boardSize: Int { 3 }
  public var playerCount: Int { 2 }
  public lazy var allActions: [Action] = {
    var actions: [Action] = []
    for x in 0..<boardSize {
      for y in 0..<boardSize {
        actions.append(Action(x: x, y: y))
      }
    }
    return actions
  }()
  public var informationStateTensorShape: [Int] { [3 * boardSize * boardSize] }
  public var maxGameLength: Int { boardSize * boardSize }
  public var initialState: State { State(self) }
  public init() {}
}

extension TicTacToe.State {
  init(_ game: TicTacToe) {
    self.game = game
    self.board = Array(repeating: nil, count: game.boardSize * game.boardSize)
  }

  public var isTerminal: Bool {
    currentPlayer == .terminal
  }

  subscript(_ action: TicTacToe.Action) -> Player? {
    get {
      board[action.x * game.boardSize + action.y]
    }
    set {
      board[action.x * game.boardSize + action.y] = newValue
    }
  }

  public mutating func apply(_ action: TicTacToe.Action) {
    precondition(self[action] == nil, "Invalid action: \(action); \(self)")
    history.append(action)
    self[action] = currentPlayer
    checkWinCondition()
    if !isTerminal {
      currentPlayer = currentPlayer.otherPlayer
    }
  }

  public var legalActionsMask: [Bool] {
    var mask: [Bool] = []
    for i in 0..<game.boardSize {
      for j in 0..<game.boardSize {
        let action = TicTacToe.Action(x: i, y: j)
        mask.append(self[action] == nil)
      }
    }
    return mask
  }

  mutating func checkWinCondition() {
    for i in 0..<game.boardSize {
      if check(row: i) || check(column: i) {
        // Winner!
        winner = currentPlayer
        currentPlayer = .terminal
        return
      }
    }
    if checkDiagonal(forward: true) || checkDiagonal(forward: false) {
      // Winner!
      winner = currentPlayer
      currentPlayer = .terminal
      return
    }
    if history.count == (game.boardSize * game.boardSize) {
      currentPlayer = .terminal
    }
  }

  func check(row: Int) -> Bool {
    for i in 0..<game.boardSize {
      let coordinate = TicTacToe.Action(x: i, y: row)
      if self[coordinate] != currentPlayer {
        return false
      }
    }
    return true
  }
  func check(column: Int) -> Bool {
    for i in 0..<game.boardSize {
      let coordinate = TicTacToe.Action(x: column, y: i)
      if self[coordinate] != currentPlayer {
        return false
      }
    }
    return true
  }
  func checkDiagonal(forward: Bool) -> Bool {
    for i in 0..<game.boardSize {
      var y: Int
      if forward {
        y = i
      } else {
        y = game.boardSize - i - 1
      }
      let coordinate = TicTacToe.Action(x: i, y: y)
      if self[coordinate] != currentPlayer {
        return false
      }
    }
    return true
  }
}

extension TicTacToe.State: CustomStringConvertible {
  public var description: String {
    var description = "TicTacToe: [current: \(currentPlayer)]\n"
    for i in 0..<3 {
      for j in 0..<3 {
        let coordinate = TicTacToe.Action(x: i, y: j)
        description.append(" \(self[coordinate].ticTacToeDescription)")
      }
      description.append("\n")
    }
    return description
  }
}

fileprivate extension Player {
  var otherPlayer: Player {
    switch self {
    case .player(0): return .player(1)
    case .player(1): return .player(0)
    default: fatalError("Cannot switch to other player from \(self).")
    }
  }
}

fileprivate extension Optional where Wrapped == Player {
  var ticTacToeDescription: String {
    switch self {
    case .none: return "."
    case let .some(player):
      switch player {
      case .player(0):
        return "x"
      case .player(1):
        return "o"
      default:
        fatalError("Unknown player \(player).")
      }
    }
  }
}
