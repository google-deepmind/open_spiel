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

/// Breakthrough is a 2-player "chess-like" boardgame used in game theory research.
///
/// To find out more, check out: https://en.wikipedia.org/wiki/Breakthrough_%28board_game%29.
///
/// In this implementation, player 0 (black) starts, and goes from high y-coordinate values to low-y
/// coordinates. Player 1 (white) starts from low-y coordinate values and is going to high-y
/// coordiante values.
public class Breakthrough: GameProtocol {
  public static let info = GameInfo(
    shortName: "breakthrough",
    longName: "Breakthrough",
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

  /// Represents the two player agents.
  public enum BreakthroughPlayer {
    case white
    case black

    public init?(_ player: Player) {
      switch player {
        case .player(0): self = .black
        case .player(1): self = .white
        default: return nil
      }
    }

    /// The corresponding OpenSpiel.Player.
    public var player: Player {
      switch self {
      case .white: return .player(1)
      case .black: return .player(0)
      }
    }

    /// The y direction this player's pawns move.
    public var moveYDirection: Int {
      switch self {
      case .white: return 1
      case .black: return -1
      }
    }

    /// The other player.
    public var otherPlayer: Self {
      switch self {
      case .white: return .black
      case .black: return .white
      }
    }
  }

  /// A BoardLocation represents a location on a Breakthrough board.
  ///
  /// This implementation packs a board location into a 16-bit value, which allows Actions to be 1
  /// word. Note: this limits the maximum board size to be 255x255. (Normal board size is 8x8.)
  public struct BoardLocation: Hashable, Equatable {
    let x: UInt8
    let y: UInt8
    public init(x: UInt8, y: UInt8) {
      self.x = x
      self.y = y
    }
  }

  /// A direction represents the 3 possible moves a pawn can make.
  public enum Direction: Hashable, Equatable, CaseIterable {
    case forward
    case left
    case right

    var deltaX: Int {
      switch self {
      case .forward: return 0
      case .left: return -1
      case .right: return 1
      }
    }
  }

  /// An action a player can make.
  public struct Action: Hashable {
    /// location defines the location of the pawn to be moved.
    public var location: BoardLocation
    /// direction describes the direction of the movement.
    public var direction: Direction

    public init(location: BoardLocation, direction: Direction) {
      self.location = location
      self.direction = direction
    }

    /// The board location the pawn should move to.
    public func targetLocation(_ player: Player) -> BoardLocation? {
      guard case let btPlayer? = BreakthroughPlayer(player) else { return nil }
      return location.move(in: direction, for: btPlayer)
    }
  }

  /// A State represents a specific game and turn instance.
  ///
  /// It contains a history of all actions played as part of the game, as well as some computed
  /// state to make certain operations more efficient.
  public struct State: StateProtocol {
    public let game: Breakthrough
    public var currentPlayer = BreakthroughPlayer.black.player
    public var board: [BreakthroughPlayer?]
    public var winner: BreakthroughPlayer? = nil
    public var history: [Game.Action] = []
    public init(game: Breakthrough) {
      self.game = game
      board = Array(repeating: nil, count: Int(game.boardHeight * game.boardWidth))
      for i in 0..<game.boardWidth {
        self[i, 0] = .white
        self[i, 1] = .white
        self[i, game.boardHeight - 1] = .black
        self[i, game.boardHeight - 2] = .black
      }
    }
    public func utility(for player: Player) -> Double {
      if !isTerminal { return 0 }
      if winner!.player == player {
        return 1
      } else {
        return -1
      }
    }

    public func informationStateString(for player: Player) -> String {
      String(describing: self)
    }

    public func informationStateTensor(for player: Player) -> [Double] {
      var state = Array<Double>(repeating: 0,
                                count: game.informationStateTensorShape.reduce(1, *))
      // Note: this implementation is intended to produce the same information state vectors as the
      // original C++ implementation.
      let planeSize = Int(game.boardWidth * game.boardHeight)
      for i in 0..<game.boardWidth {
        for j in 0..<game.boardHeight {
          let planeOffset = Int(i * game.boardHeight + j)
          switch self[i, j] {
          case .black: state[planeOffset] = 1
          case .white: state[planeSize + planeOffset] = 1
          case nil: state[2 * planeSize + planeOffset] = 1
          }
        }
      }
      return state
    }
  }

  public let boardHeight, boardWidth: UInt8
  public var minUtility: Double { -1.0 }
  public var maxUtility: Double { 1.0 }
  /// The total utility for all players.
  /// Breakthrough is a zero-sum game.
  public var utilitySum: Double? { 0 }

  public var initialState: State {
    State(game: self)
  }
  public var playerCount: Int { 2 }
  public lazy var allActions: [Action] = {
    var actions: [Action] = []
    for x in 0..<boardWidth {
      for y in 0..<boardHeight {
        let location = BoardLocation(x: x, y: y)
        for direction in Direction.allCases {
          actions.append(Action(location: location, direction: direction))
        }
      }
    }
    return actions
  }()
  public var informationStateTensorShape: [Int] {
    [3, /* # of cell states */
    Int(boardHeight),
    Int(boardWidth)]
  }
  public var maxGameLength: Int {
    Int(boardWidth * (boardHeight - 2)) + 1
  }

  /// Initializes an instance of the Breakthrough game family.
  ///
  /// Note the restrictions on board sizes (minimum: 4x2, maximum: 255x255).
  public init(boardHeight: Int = 8, boardWidth: Int = 8) {
    precondition(boardHeight >= 4 && boardHeight < 255)
    precondition(boardWidth >= 2 && boardWidth < 255)
    self.boardHeight = UInt8(boardHeight)
    self.boardWidth = UInt8(boardWidth)
  }
  public func isValid(location: BoardLocation) -> Bool {
    return location.x < boardWidth && location.y < boardHeight
  }
}

public extension Breakthrough.State {
  mutating func apply(_ action: Breakthrough.Action) {
    // Validate that this is a valid action.
    precondition(game.isValid(location: action.location), "Invalid location for action: \(action).")
    precondition(self[action.location] == currentBTPlayer,
                 "No pawn at location \(action.location). \(self).")
    guard case let targetLocation? = action.targetLocation(currentPlayer) else {
      preconditionFailure("Invalid move \(action). \(self)")
    }
    precondition(game.isValid(location: targetLocation),
                 "Invalid location on board \(targetLocation). \(self)")
    precondition(self[targetLocation] != currentBTPlayer,
                 "Cannot move a pawn onto a spot already occupied by another pawn.")
    if action.direction == .forward {
      precondition(self[targetLocation] == nil,
                   "Cannot move forward into non-empty spot \(action). \(self)")
    }
    // Add the move into the history.
    history.append(action)
    // Update materialized state.
    self[action.location] = nil
    self[targetLocation] = currentBTPlayer
    if targetLocation.y == winYCoordinate {
      winner = currentBTPlayer!
      currentPlayer = .terminal
    } else {
      currentPlayer = currentBTPlayer!.otherPlayer.player
    }
  }
  var currentBTPlayer: Breakthrough.BreakthroughPlayer? {
    Breakthrough.BreakthroughPlayer(currentPlayer)
  }
  
  var legalActions: [Game.Action] {
    var actions = [Game.Action]()
    actions.reserveCapacity(Int(game.boardWidth * game.boardHeight))
    let curBTPlayer = currentBTPlayer!
    for i in 0..<game.boardWidth {
      for j in 0..<game.boardHeight {
        let boardLoc = Breakthrough.BoardLocation(x: i, y: j)
        // Skip all board locations that don't have a pawn that corresponds to the current player.
        if self[boardLoc] != curBTPlayer { continue }
        // Iterate across all possible directions.
        for direction in Breakthrough.Direction.allCases {
          // Compute the moved board location & verify it's still on the board.
          guard case let movedBoardLoc? = boardLoc.move(in: direction, for: curBTPlayer),
              game.isValid(location: movedBoardLoc) else { continue }
          // If the moved board location is already occupied by one of our own pawns, it's
          // not an available move.
          if self[movedBoardLoc] == curBTPlayer { continue }
          // If we're trying to move forward and it's not empty, it's not a valid move.
          if direction == .forward && self[movedBoardLoc] != nil { continue }
          // Append the available actions to the possible actions list.
          actions.append(Game.Action(location: boardLoc, direction: direction))
        }
      }
    }
    return actions
  }

  var legalActionsMask: [Bool] {
    let directionCount = Breakthrough.Direction.allCases.count
    var mask: [Bool] = Array(
      repeating: false, count: Int(game.boardWidth) * Int(game.boardHeight) * directionCount)
    let curBTPlayer = currentBTPlayer!
    for i in 0..<game.boardWidth {
      for j in 0..<game.boardHeight {
        let boardLoc = Breakthrough.BoardLocation(x: i, y: j)
        // Skip all board locations that don't have a pawn that corresponds to the current player.
        if self[boardLoc] != curBTPlayer { continue }
        // Iterate across all possible directions.
        for (k, direction) in Breakthrough.Direction.allCases.enumerated() {
          // Compute the moved board location & verify it's still on the board.
          guard case let movedBoardLoc? = boardLoc.move(in: direction, for: curBTPlayer),
              game.isValid(location: movedBoardLoc) else { continue }
          // If the moved board location is already occupied by one of our own pawns, it's
          // not an available move.
          if self[movedBoardLoc] == curBTPlayer { continue }
          // If we're trying to move forward and it's not empty, it's not a valid move.
          if direction == .forward && self[movedBoardLoc] != nil { continue }
          // Append the available actions to the possible actions list.
          let index = Int(i) * Int(game.boardHeight) * directionCount + Int(j) * directionCount + k
          mask[index] = true
        }
      }
    }
    return mask
  }

  var isTerminal: Bool {
    winner != nil
  }

  // Make it easy to access board squares.
  subscript(x: UInt8, y: UInt8) -> Breakthrough.BreakthroughPlayer? {
    get {
      board[Int(x) * Int(game.boardHeight) + Int(y)]
    }
    set {
      board[Int(x) * Int(game.boardHeight) + Int(y)] = newValue
    }
  }

  subscript(loc: Breakthrough.BoardLocation) -> Breakthrough.BreakthroughPlayer? {
    get {
      self[loc.x, loc.y]
    }
    set {
      self[loc.x, loc.y] = newValue
    }
  }

  var winYCoordinate: UInt8 {
    switch currentBTPlayer {
    case .black: return 0
    case .white: return game.boardHeight - 1
    case nil: fatalError("Invalid current player: \(currentPlayer).")
    }
  }
}

extension Breakthrough.BreakthroughPlayer: CustomStringConvertible {
  public var description: String {
    switch self {
    case .white: return "w"
    case .black: return "b"
    }
  }
}

extension Breakthrough.State: CustomStringConvertible {
  public var description: String {
    var description = """
    Breakthrough State:
      Board size: \(game.boardWidth) x \(game.boardHeight)
      Winner: \(String(describing: winner)); Current Player: \(String(describing: currentBTPlayer))


    """
    for j in (0..<game.boardHeight).reversed() {
      description.append("    \(j)")
      for i in 0..<game.boardWidth {
        description.append(" \(self[UInt8(i), UInt8(j)]?.description ?? " ")")
      }
      description.append("\n")
    }
    description.append("      ")
    description.append((0..<game.boardWidth).map { String(describing: $0) }.joined(separator: " "))
    description.append("\n\n")
    return description
  }
}

extension Breakthrough.Direction: CustomStringConvertible {
  public var description: String {
    switch self {
    case .forward: return "forward"
    case .left: return "left"
    case .right: return "right"
    }
  }
}

public extension Breakthrough.BoardLocation {

  init?(x: Int, y: Int) {
    guard x < 256 && x >= 0 else { return nil }
    guard y < 256 && y >= 0 else { return nil }
    self.x = UInt8(x)
    self.y = UInt8(y)
  }

  func move(in direction: Breakthrough.Direction,
            for player: Breakthrough.BreakthroughPlayer) -> Breakthrough.BoardLocation? {
    // Perform the translation in
    let newX = Int(self.x) + direction.deltaX
    let newY = Int(self.y) + player.moveYDirection
    return Breakthrough.BoardLocation(x: newX, y: newY)
  }
}
