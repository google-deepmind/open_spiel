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
/// coordiante values. Additionally, this currently only supports a fixed-size 8x8 board.
public class FastBreakthrough: GameProtocol {
  public static let info = GameInfo(
    shortName: "fastBreakthrough",
    longName: "Fast Breakthrough",
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
  public enum BreakthroughPlayer: UInt8 {
    case white = 0
    case black = 1

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
  public struct BoardLocation: Hashable, Equatable, Comparable {
    let x: UInt8
    let y: UInt8
    public init(x: UInt8, y: UInt8) {
      self.x = x
      self.y = y
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
      if lhs.x < rhs.x { return true }
      if lhs.x == rhs.x { return lhs.y < rhs.y }
      return false
    }
  }

  /// A direction represents the 3 possible moves a pawn can make.
  public enum Direction: Hashable, Equatable, CaseIterable, Comparable {
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
  public struct Action: Hashable, Comparable {
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

    public static func < (lhs: Self, rhs: Self) -> Bool {
      if lhs.location < rhs.location { return true }
      if lhs.location == rhs.location { return lhs.direction < rhs.direction }
      return false
    }
  }

  /// Represents an 8x8 game board as 2 UInt64 values.
  public struct Board: Equatable, Hashable {
    /// The bit is set if the position is occupied, false otherwise.
    var occupiedMask: UInt64
    /// If the corresponding bit in `occupiedMask` is set, then the corresponding bit in `pawnColor` indicates
    /// the color of the pawn at the given location.
    var pawnColor: UInt64

    /// Accesses the board at position `(x, y)`, returning the player corresponding to the pawn occupying the
    /// square, or nil if the square is empty.
    public subscript(x: UInt8, y: UInt8) -> BreakthroughPlayer? {
      get {
        let shift = bitShiftForCoordinate(x: x, y: y)
        let mask = UInt64(1) << shift
        if occupiedMask & mask == 0 {
          return nil
        }
        return pawnColor & mask == 0 ? .white : .black
      }
      set {
        let shift = bitShiftForCoordinate(x: x, y: y)
        let mask = UInt64(1) << shift
        guard let player = newValue else {
          // Clear the occupied bit & return.
          occupiedMask = occupiedMask & ~mask
          return
        }
        occupiedMask |= mask
        // We do the following shenanigans to avoid taking a branch.
        pawnColor &= ~mask  // Mask off the bit.
        pawnColor |= (UInt64(player.rawValue) << shift)
      }
    }

    private func bitShiftForCoordinate(x: UInt8, y: UInt8) -> UInt64 {
        assert(x < 8, "\(x)")
        assert(y < 8, "\(y)")
        return UInt64(y << 3 + x)
    }

    static public func >> (board: Self, shift: UInt64) -> Self {
      return Board(occupiedMask: board.occupiedMask >> shift, pawnColor: board.pawnColor >> shift)
    }

    static public func << (board: Self, shift: UInt64) -> Self {
      return Board(occupiedMask: board.occupiedMask << shift, pawnColor: board.pawnColor << shift)
    }
  }

  /// A State represents a specific game and turn instance.
  ///
  /// It contains a history of all actions played as part of the game, as well as some computed
  /// state to make certain operations more efficient.
  public struct State: StateProtocol {
    public let game: FastBreakthrough
    public var currentPlayer = BreakthroughPlayer.black.player
    public var board: Board
    public var winner: BreakthroughPlayer? = nil
    public var history: [Game.Action] = []
    public init(game: FastBreakthrough) {
      self.game = game
      board = Board()
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

  public let boardHeight: UInt8 = 8
  public let boardWidth: UInt8 = 8
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

  /// Initializes an instance of the FastBreakthrough game family.
  ///
  /// Note future work may generalize this to support a variety of different board sizes.
  /// This is left as future work.
  public init() {}

  /// Returns true iff `location` is valid for this size game.
  public func isValid(location: BoardLocation) -> Bool {
    return location.x < boardWidth && location.y < boardHeight
  }

}

public extension FastBreakthrough.State {
  mutating func apply(_ action: FastBreakthrough.Action) {
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
  var currentBTPlayer: FastBreakthrough.BreakthroughPlayer? {
    FastBreakthrough.BreakthroughPlayer(currentPlayer)
  }
  
  var legalActions: [Game.Action] {
    // print("Board: \(board)")
    // print("Computing legal actions...\npawncolor:    \(board.pawnColor.bitString)\noccupiedmask: \(board.occupiedMask.bitString)")
    let curBTPlayer = currentBTPlayer!
    let playerColorBits = curBTPlayer.rawValue == 0 ? UInt64(0) : ~UInt64(0)  // bitmask setting the player's bit everywhere.
    // print("player color: \(playerColorBits.bitString)")
    let currentPawnLocations = (board.pawnColor ^ ~playerColorBits) & board.occupiedMask
    // print("current locs: \(currentPawnLocations.bitString)")
    // Use vector operations to efficiently compute the valid actions.
    let forwardMovesAvailable, leftMovesAvailable, rightMovesAvailable: UInt64

    // Hope the branch predictor does okay here...
    if curBTPlayer == .white {
      // White is moving from "low-y" to "low-y". As a result, we use left-shift operators to move forward.
      // Note: we don't worry about the "last row" case, as that's a game-over state already.
      
      // Compute forward options; in the forward board, we need squares to be unoccupied.
      let forward = board >> 8
      forwardMovesAvailable = currentPawnLocations & ~forward.occupiedMask
      // Compute left & right options; here we just need squares to be on the board & not occupied by self.
      let left = board >> 7
      leftMovesAvailable = currentPawnLocations & ~((left.pawnColor ^ ~playerColorBits) & left.occupiedMask) & ~Game.Board.kLeftMask
      let right = board >> 9
      rightMovesAvailable = currentPawnLocations & ~((right.pawnColor ^ ~playerColorBits) & right.occupiedMask) & ~Game.Board.kRightMask
    } else {
      assert(curBTPlayer == .black)
      // Black is moving from "low-y" to "high-y", and so we use right-shift operators to move forward.
      // Note: we don't worry about the "last row" case, as that's a game-over state already.

      // Compute forward options.
      let forward = board << 8
      forwardMovesAvailable = currentPawnLocations & ~forward.occupiedMask
      // Compute left & right options; here we just need squares to be on the board & not occupied by self.
      let left = board << 9
      leftMovesAvailable = currentPawnLocations & ~((left.pawnColor ^ ~playerColorBits) & left.occupiedMask) & ~Game.Board.kLeftMask
      let right = board << 7
      rightMovesAvailable = currentPawnLocations & ~((right.pawnColor ^ ~playerColorBits) & right.occupiedMask) & ~Game.Board.kRightMask
    }

    // Turn bitsets into logical actions.
    var actions = [Game.Action]()
    actions.reserveCapacity(
      forwardMovesAvailable.nonzeroBitCount +
      leftMovesAvailable.nonzeroBitCount +
      rightMovesAvailable.nonzeroBitCount)
    
    func addActions(mask: UInt64, direction: Game.Direction) {
      var index = 0
      var mask = mask
      while mask != 0 {
        let lowestBitOffset = mask.trailingZeroBitCount
        index += lowestBitOffset
        let location = Game.BoardLocation(x: UInt8(index & 7), y: UInt8((index & (7 << 3)) >> 3))
        actions.append(.init(location: location, direction: direction))
        mask = mask >> (lowestBitOffset + 1)
        index += 1
      }
    }
    addActions(mask: forwardMovesAvailable, direction: .forward)
    addActions(mask: leftMovesAvailable, direction: .left)
    addActions(mask: rightMovesAvailable, direction: .right)
    return actions
  }

  var legalActionsMask: [Bool] {
    print("SLOW PATH! PLEASE FILE A BUG TO OPTIMIZE ME!")
    let directionCount = FastBreakthrough.Direction.allCases.count
    var mask: [Bool] = Array(
      repeating: false, count: Int(game.boardWidth) * Int(game.boardHeight) * directionCount)
    let curBTPlayer = currentBTPlayer!
    for i in 0..<game.boardWidth {
      for j in 0..<game.boardHeight {
        let boardLoc = FastBreakthrough.BoardLocation(x: i, y: j)
        // Skip all board locations that don't have a pawn that corresponds to the current player.
        if self[boardLoc] != curBTPlayer { continue }
        // Iterate across all possible directions.
        for (k, direction) in FastBreakthrough.Direction.allCases.enumerated() {
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
  subscript(x: UInt8, y: UInt8) -> FastBreakthrough.BreakthroughPlayer? {
    get {
      board[x, y]
    }
    set {
      board[x, y] = newValue
    }
  }

  subscript(loc: FastBreakthrough.BoardLocation) -> FastBreakthrough.BreakthroughPlayer? {
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

extension FastBreakthrough.BreakthroughPlayer: CustomStringConvertible {
  public var description: String {
    switch self {
    case .white: return "w"
    case .black: return "b"
    }
  }
}

extension FastBreakthrough.State: CustomStringConvertible {
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

extension FastBreakthrough.Board {
  /// Initializes an empty board.
  public init() {
    occupiedMask = 0
    pawnColor = 0
  }

  // TODO: update these in order to support resizable boards.
  static var kBottomMask: UInt64 { 255 }
  static var kTopMask: UInt64 { UInt64(255) << 55 }
  static var kLeftMask: UInt64 {
    // TODO: verify that these get constant-folded away.
    return
      UInt64(1) << 0 |
      UInt64(1) << 8 |
      UInt64(1) << 16 |
      UInt64(1) << 24 |
      UInt64(1) << 32 |
      UInt64(1) << 40 |
      UInt64(1) << 48 |
      UInt64(1) << 56
  }

  static var kRightMask: UInt64 {
    // TODO: verify that these get constant folded away.
    return
      UInt64(1) << 7 |
      UInt64(1) << 15 |
      UInt64(1) << 23 |
      UInt64(1) << 31 |
      UInt64(1) << 39 |
      UInt64(1) << 47 |
      UInt64(1) << 55 |
      UInt64(1) << 63
  }
}

extension FastBreakthrough.Board: CustomStringConvertible {
  public var description: String {
    var description = "\n"
    for j in (UInt8(0)..<8).reversed() {
      description.append("   \(j)")
      for i in UInt8(0)..<8 {
        description.append(" \(self[i, j]?.description ?? " ")")
      }
      description.append("\n")
    }
    description.append("     ")
    description.append((0..<8).map { String(describing: $0) }.joined(separator: " "))
    description.append("\n")
    return description
  }
}

extension FastBreakthrough.Direction: CustomStringConvertible {
  public var description: String {
    switch self {
    case .forward: return "forward"
    case .left: return "left"
    case .right: return "right"
    }
  }
}

public extension FastBreakthrough.BoardLocation {

  init?(x: Int, y: Int) {
    guard x < 256 && x >= 0 else { return nil }
    guard y < 256 && y >= 0 else { return nil }
    self.x = UInt8(x)
    self.y = UInt8(y)
  }

  func move(in direction: FastBreakthrough.Direction,
            for player: FastBreakthrough.BreakthroughPlayer) -> FastBreakthrough.BoardLocation? {
    // Perform the translation in
    let newX = Int(self.x) + direction.deltaX
    let newY = Int(self.y) + player.moveYDirection
    return FastBreakthrough.BoardLocation(x: newX, y: newY)
  }
}

fileprivate extension UInt64 {
  var bitString: String {
    var str = ""
    for i in UInt64(0)..<64 {
      if i % 8 == 0 && i > 0 {
        str.append("_")
      }
      str.append(self & (UInt64(1) << i) == 0 ? "0" : "1")
    }
    return "0b\(String(str.reversed()))"
  }
}
