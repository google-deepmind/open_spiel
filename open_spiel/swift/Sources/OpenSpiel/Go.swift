/// The board game Go, https://en.wikipedia.org/wiki/Go_(game)

import Foundation

/// Go board implementation based on third_party/open_spiel/games/go/go_board.h.
public struct Go {
  /// For simplicity and speed, we store the board in terms of a "virtual board",
  /// with a border of guard stones around all sides of the board.
  /// This allows us to skip bounds checking.
  /// We support boards up to size 19.
  static let maxBoardSize = 19
  static let virtualBoardSize = maxBoardSize + 2
  static let virtualBoardPoints = virtualBoardSize * virtualBoardSize

  /// A point on the board, used to index various structures in the GoBoard.
  public struct Point: Equatable, Hashable {
    public static let invalid = Point(0)
    public static let pass = Point(Go.virtualBoardPoints + 1)

    /// Returns the Point corresponding to the provided coordinates, e.g. "d4" or "f10".
    public static func fromString(_ s: String) -> Point {
      let s = s.lowercased()

      if s == passString {
        return Point.pass
      }
      guard s.count >= 2 && s.count <= 3 else {
        return Point.invalid
      }

      let scalars = s.unicodeScalars

      var col = Int(scalars[0].value) - 97  // 'a' == 97
      guard scalars[0] != "i" else {
        return Point.invalid
      }
      if scalars[0] >= "i" {
        col -= 1
      }
      var row = Int(scalars[1].value) - 48  // '0' == 48
      if s.count == 3 {
        row *= 10
        row += Int(scalars[2].value) - 48
      }
      return Point.from2D((row - 1, col))
    }

    /// Builds a Point from a 2D representation.
    public static func from2D(_ rowCol: (Int, Int)) -> Point {
      let (row, col) = rowCol
      guard row >= 0 && row < Go.maxBoardSize && col >= 0 && col < Go.maxBoardSize else {
        return invalid
      }
      return Point((row + 1) * Go.virtualBoardSize + col + 1)
    }

    // Turns a Point into a 2D representation.
    public var coordinates: (row: Int, col: Int)? {
      guard self != Point.invalid && self != Point.pass else {
        return nil
      }

      let row = Int(point) / Go.virtualBoardSize
      let col = Int(point) % Go.virtualBoardSize
      return (row - 1, col - 1)
    }

    fileprivate init(_ p: Int) {
      point = UInt16(p)
    }

    /// The point is encoded as row * boardSize + col, but embedded in a "virtual" board that
    /// surrounds the real board on all sides.
    /// The encoding for a point described by (row, col) on a real board is therefore:
    ///   (row + 1) * Go.virtualBoardSize + col + 1
    /// This ensures that all directly adjacent neighbours are always on the virtual board, so we
    /// can skip bounds checking when walking the neighbours of a point.
    fileprivate let point: UInt16

    private static let invalidString = "invalid_point"
    private static let passString = "pass"
  }

  /// Color of a point on the board.
  public enum Color {
    case black, white, empty, border

    /// Returns the color of the opposing player to this one.
    public var opponent: Color {
      switch self {
      case .black:
        return .white
      case .white:
        return .black
      default:
        return self
      }
    }

    /// Returns a single letter representation of this color.
    public var char: Character {
      switch self {
      case .black:
        return "X"
      case .white:
        return "O"
      case .empty:
        return "+"
      case .border:
        return "#"
      }
    }
  }

  /// A fast Go board that supports playing according to standard Chinese rules.
  public struct Board {
    /// Creates a new empty board, valid sizes between 1 and 19.
    public init?(size: Int) {
      guard size >= 1 && size <= maxBoardSize else { return nil }
      self.boardSize = size
      clear()
    }

    /// Clears the board, as if a new board had been created.
    public mutating func clear() {
      for i in 0 ..< board.count {
        let p = Point(i)
        if isInBoardArea(point: p) {
          board[p] = Vertex(color: .empty, chainHead: p, chainNext: p)
          chains[p] = Chain(forBoardPoint: ())
        } else {
          board[p] = Vertex(color: .border, chainHead: p, chainNext: p)
          chains[p] = Chain(forBorderPoint: ())
        }
      }

      // Already checked legal board sizes in init, so force unwrap is safe.
      for p in boardPoints(boardSize)! {
        forEachNeighbour(of: p) { n in
          if isEmpty(point: n) {
            withChain(for: p) { $0.add(liberty: n) }
          }
        }
      }
    }

    public func color(of: Point) -> Color { return board[of].color }

    public func isEmpty(point: Point) -> Bool { return color(of: point) == .empty }

    /// Plays a stone on the specified point for the specified player.
    ///
    /// Returns true if the move was legal and has been applied. For performance reasons, this
    /// method does not perform all the checks of `isLegalMove`, so if you are in doubt whether a
    /// move is legal you should check with that method first.
    public mutating func play(point: Point, for c: Color) -> Bool {
      guard point != .pass else { return true }
      guard color(of: point) == .empty else { return false }

      // Preparation for ko checking.
      var playedInEnemyEye = true
      forEachNeighbour(of: point) { n in
        let s = color(of: n)
        if s == c || s == .empty {
          playedInEnemyEye = false
        }
      }

      mergeChains(for: c, around: point)
      setStone(on: point, to: c)
      removeLibertyFromChains(neighbouring: point)
      let stonesCaptured = captureDeadChains(for: c, around: point)

      if playedInEnemyEye && stonesCaptured == 1 {
        lastKoPoint = lastCaptures[0]
      } else {
        lastKoPoint = .invalid
      }

      if chain(for: point).pseudoLibertyCount <= 0 {
        fatalError("newly placed stone \(point) for \(c) must be alive, on: \(self)")
      }

      return true
    }

    /// Whether player `c` is allowed to place a stone on point `p`.
    ///
    /// If `isLegalMove` returns true, `playMove` will also succeed and return true.
    public func isLegal(point: Point, for c: Color) -> Bool {
      guard point != .pass else { return true }
      guard isInBoardArea(point: point) && isEmpty(point: point) && point != lastKoPoint else {
        return false
      }
      if chain(for: point).pseudoLibertyCount > 0 { return true }

      // For all checks below, the newly placed stone is completely surrounded by enemy and friendly
      // stones.

      // Allow to play if the placed stones connects to a group that still has at least one other
      // liberty after connecting.
      var hasLiberty = false
      forEachNeighbour(of: point) { n in
        hasLiberty = hasLiberty || (color(of: n) == c && !chain(for: n).inAtari())
      }
      if hasLiberty { return true }

      // Allow to play if the placed stone will kill at least one group.
      var killsGroup = false
      forEachNeighbour(of: point) { n in
        killsGroup = killsGroup || (color(of: n) == c.opponent && chain(for: n).inAtari())
      }
      if killsGroup { return true }

      return false
    }

    /// Whether the point is inside the area of this board.
    ///
    /// It is only legal to play on points that are on the board.
    public func isInBoardArea(point: Point) -> Bool {
      guard let (row, col) = point.coordinates else { return false }
      return row >= 0 && row < boardSize && col >= 0 && col < boardSize
    }

    /// Count of pseudo-liberties for the chain that `p` is a part of.
    ///
    /// This value is exact chains that only contain a single stone and for larger chains if they
    /// only have 0 and 1 liberties, but only an approximation when a chain has 2 or more liberties.
    public func pseudoLibertyCount(for p: Point) -> Int {
      if chain(for: p).pseudoLibertyCount == 0 {
        return 0
      }
      if chain(for: p).inAtari() {
        return 1
      }
      return Int(chain(for: p).pseudoLibertyCount)
    }

    /// Whether the chain that p is a part of is in Atari.
    public func inAtari(_ p: Point) -> Bool { return chain(for: p).inAtari() }

    private mutating func setStone(on p: Point, to c: Color) {
      board[p].color = c
    }

    /// Merges all chains around point p that belong to player c, including point p.
    private mutating func mergeChains(for c: Color, around p: Point) {
      // Find the largest neighbouring chain for point p.
      var largestChainHead = Point.invalid
      var largestChainCount = Int16(0)
      forEachNeighbour(of: p) { n in
        if color(of: n) == c {
          let ch = chain(for: n)
          if ch.stoneCount > largestChainCount {
            largestChainCount = ch.stoneCount
            largestChainHead = chainHead(for: n)
          }
        }
      }

      // If there were no neighbouring chains, just make a new one and we are done.
      if largestChainCount == 0 {
        initNewChain(for: p)
        return
      }

      // Otherwise, merge all other chains into the largest one.
      forEachNeighbour(of: p) { n in
        if color(of: n) == c {
          let head = chainHead(for: n)
          if head != largestChainHead {
            let target = chain(for: n)
            withChain(for: largestChainHead) { $0.merge(with: target) }

            // Set all stones in the smaller chain to be part of the larger chain.
            var cur = n
            repeat {
              board[cur].chainHead = largestChainHead
              cur = board[cur].chainNext
            } while (cur != n)

            // Connect the 2 linked lists representing the stones in the two chains.
            let a = board[largestChainHead].chainNext
            let b = board[n].chainNext
            board[n].chainNext = a
            board[largestChainHead].chainNext = b
          }
        }
      }

      // Finally, add the point into the merged chain.
      board[p].chainNext = board[largestChainHead].chainNext
      board[largestChainHead].chainNext = p
      board[p].chainHead = largestChainHead
      withChain(for: largestChainHead) { $0.stoneCount += 1 }

      // Add any empty neighbouring points as liberties.
      forEachNeighbour(of: p) { n in
        if isEmpty(point: n) {
          withChain(for: largestChainHead) { $0.add(liberty: n) }
        }
      }
    }

    /// Creates a new chain that only contains point p.
    private mutating func initNewChain(for p: Point) {
      board[p].chainHead = p
      board[p].chainNext = p

      var newChain = Chain(forBoardPoint: ())
      newChain.stoneCount += 1

      forEachNeighbour(of: p) { n in
        if isEmpty(point: n) {
          newChain.add(liberty: n)
        }
      }

      chains[p] = newChain
    }

    private mutating func removeLibertyFromChains(neighbouring p: Point) {
      forEachNeighbour(of: p) { n in
        withChain(for: n) { $0.remove(liberty: p) }
      }
    }

    /// Captures and removes all dead chains from the board, returning the count of stones captured.
    private mutating func captureDeadChains(for c: Color, around p: Point) -> Int {
      var stonesCaptured = 0
      var captureIndex = 0

      forEachNeighbour(of: p) { n in
        if color(of: n) == c.opponent && chain(for: n).pseudoLibertyCount == 0 {
          lastCaptures[captureIndex] = chainHead(for: n)
          captureIndex += 1

          stonesCaptured += Int(chain(for: n).stoneCount)
          removeChain(for: n)
        }
      }

      for i in captureIndex ..< lastCaptures.count {
        lastCaptures[i] = .invalid
      }

      return stonesCaptured
    }

    /// Removes a chain from the board.
    private mutating func removeChain(for p: Point) {
      let thisChainHead = chainHead(for: p)
      var cur = p
      repeat {
        let next = board[cur].chainNext

        setStone(on: cur, to: .empty)
        initNewChain(for: cur)

        forEachNeighbour(of: cur) { n in
          if chainHead(for: n) != thisChainHead || isEmpty(point: n) {
            withChain(for: n) { $0.add(liberty: cur) }
          }
        }

        cur = next
      } while cur != p
    }

    /// Head of the chain that `p` is a part of.
    ///
    /// Each chain has exactly one head that can be used to uniquely identify it. Chain heads may
    /// change over successive `play()` calls.
    private func chainHead(for p: Point) -> Point { return board[p].chainHead }

    /// Chain that `p` is a part of.
    private func chain(for p: Point) -> Chain { return chains[chainHead(for: p)] }

    /// Calls the provided closure with the Chain that `p` is a part of.
    private mutating func withChain<T>(for p: Point, fn: (inout Chain) throws -> T) rethrows -> T {
      return try fn(&chains[chainHead(for: p)])
    }

    /// Helper structure to contain all information for a specific point on the board.
    private struct Vertex {
      var color: Color = .border
      var chainHead = Point.invalid
      var chainNext = Point.invalid
    }

    /// A chain of directly (horizontal/vertical, not diagional) connected stones of one color.
    ///
    /// In Go, a chain of stones lives and dies together, so tracking the liberties (directly
    /// adjacent empty points) on a chain level simplifies calculations.
    ///
    /// In this case we don't track the full set of liberties, but a simplified concept of "pseudo"
    /// liberties (https://groups.google.com/d/msg/computer-go-archive/-MzbYeiuFzs/Z9gH42ACzMoJ).
    /// Instead of counting each empty neighbouring point exactly once, we count it once for every
    /// stone that it is adjacent to. This does not allow us to compute the exact liberty count, but
    /// does tell is if a stone is an Atari or captured, and is much faster than keeping track of
    /// the exact set.
    ///
    /// Stones in a chain are stored as a linked list, using the `chainHead` and `chainNext` members
    /// in each Vertex struct. The canonical place to store information for a chain is
    /// `chains[chainHead]`.
    struct Chain {
      var libertyVertexSumSquared: Int32 = 0
      var libertyVertexSum: Int16 = 0
      var stoneCount: Int16 = 0
      var pseudoLibertyCount: Int16 = 0
      /// Create an initial no-liberty state, for normal points on the board.
      init(forBoardPoint: Void) {
        stoneCount = 0
        pseudoLibertyCount = 0
        libertyVertexSum = 0
        libertyVertexSumSquared = 0
      }

      /// Create an initial "infinite" liberty state, for the virtual border around the board.
      init(forBorderPoint: Void) {
        stoneCount = 0
        // Need to have values big enough that they never overflow even if all liberties are
        // removed, but small enough they never overflow.
        pseudoLibertyCount = Int16.max / 2
        libertyVertexSum = Int16.max / 2
        libertyVertexSumSquared = Int32.max / 2

      }

      /// Merges the liberties of another chain into this chain.
      mutating func merge(with other: Chain) {
        stoneCount += other.stoneCount
        pseudoLibertyCount += other.pseudoLibertyCount
        libertyVertexSum += other.libertyVertexSum
        libertyVertexSumSquared += other.libertyVertexSumSquared
      }

      /// Whether this chain is in Atari, i.e. it has exactly one liberty left.
      func inAtari() -> Bool {
        return Int32(pseudoLibertyCount) * libertyVertexSumSquared == Int32(libertyVertexSum)
          * Int32(libertyVertexSum)
      }

      /// Adds an adjacent point as a liberty of this chain.
      mutating func add(liberty: Point) {
        let value = Int16(liberty.point)
        pseudoLibertyCount += 1
        libertyVertexSum += value
        libertyVertexSumSquared += Int32(value) * Int32(value)
      }

      /// Removes an adjacent point as a liberty of this chain.
      mutating func remove(liberty: Point) {
        let value = Int16(liberty.point)
        pseudoLibertyCount -= 1
        libertyVertexSum -= value
        libertyVertexSumSquared -= Int32(value) * Int32(value)
      }
    }

    /// An array of points on the board, with a virtual border of guard stones around the board.
    ///
    /// Using a fixed size board with a boarder of guard stones allows us to remove branching and
    /// skip bounds checks during operations on the hot path, significantly speeding up important
    /// methods. It also allows us to directly use Go.Point instances to index the board.
    struct BoardArray<T> {
      init(repeating: T) {
        storage = [T](repeating: repeating, count: virtualBoardPoints)
      }

      subscript(_ p: Point) -> T {
        get { return storage[Int(p.point)] }
        set(newValue) { storage[Int(p.point)] = newValue }
      }

      var count: Int {
        return storage.count
      }

      private var storage: [T]
    }

    private var board = BoardArray(repeating: Vertex())
    private var chains = BoardArray(repeating: Chain(forBorderPoint: ()))

    /// Chain heads of the most recently captured chains.
    ///
    /// A point in Go has exactly 4 direct neighbours, so we can capture at most 4 distinct chains
    /// in a single move.
    private var lastCaptures = [Point](repeating: Point.invalid, count: 4)

    private let boardSize: Int
    private var lastKoPoint = Point.invalid
  }

  /// Returns a vector that contains all points that are on a board of the specified size.
  public static func boardPoints(_ boardSize: Int) -> [Point]? {
    guard boardSize >= 1 && boardSize <= maxBoardSize else {
      return nil
    }

    var points: [Point] = []
    points.reserveCapacity(boardSize * boardSize)

    for row in 0..<boardSize {
      for col in 0..<boardSize {
        points.append(Point.from2D((row, col)))
      }
    }

    return points
  }
}

/// Formats a Point to a human readable string which can be parsed by MakePoint.
extension Go.Point: CustomStringConvertible {
  public var description: String {
    switch self {
    case Go.Point.invalid:
      return Go.Point.invalidString
    case Go.Point.pass:
      return Go.Point.passString
    default:
      let (r, c) = coordinates!
      // Go column encoding skips the letter i to avoid confusion with j.
      let col = Character(UnicodeScalar(Int(97) + c + (c >= 8 ? 1 : 0))!)
      return "\(col)\(r + 1)"
    }
  }
}

/// Prints the stones on the Go board in a human readable way.
extension Go.Board: CustomStringConvertible {
  public var description: String {
    var s = "Board\n"
    for row in 0 ..< boardSize {
      s += String(format: "%2d ", row + 1)
      for col in 0 ..< boardSize {
        s.append(color(of: Go.Point.from2D((row, col))).char)
      }
      s += "\n"
    }
    let columns = "ABCDEFGHJKLMNOPQRST"
    s += "   " + columns.prefix(boardSize)
    return s
  }
}

// Calls fn with all 4 direct neighbours of p.
private func forEachNeighbour(of p: Go.Point, fn: (Go.Point) -> Void) {
  let intPoint = Int(p.point)
  fn(Go.Point(intPoint + Go.virtualBoardSize))
  fn(Go.Point(intPoint + 1))
  fn(Go.Point(intPoint - 1))
  fn(Go.Point(intPoint - Go.virtualBoardSize))
}

/// Helper for Point.fromString.
extension String.UnicodeScalarView {
  subscript(i: Int) -> Unicode.Scalar {
    return self[index(startIndex, offsetBy: i)]
  }
}
