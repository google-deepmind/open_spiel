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

import Foundation

public class GridMaze: GameProtocol {

  // Maximum set of possible actions in any grid position in the maze
  public enum Action: Int, CustomStringConvertible, CaseIterable, Hashable {
    case LEFT
    case UP
    case DOWN
    case RIGHT

    public var description: String {
      switch self {
      case .LEFT: return "<"
      case .UP: return "A"
      case .DOWN: return "V"
      case .RIGHT: return ">"
      }
    }
  }

  public struct GameState: StateProtocol {
    public let game: GridMaze
    public var history = [GridMaze.Action]()
    public var currentPlayer: Player { return isTerminal ? .terminal : .player(0) }
    public var isTerminal: Bool { return gridCell.isTerminal }
    public var gridCell: GridCell  // Cell in the maze where we are currently positioned
    var utility = 0.0
  }

  // *** Section 1: Required by GameProtocol
  // Many of the things could instead of fatalError return nil if that was supported by GameProtocol
  // Crashing or -Double.infinity are only current alteratives
  public var minUtility: Double { fatalError("Cannot be calcuated for GridMaze") }
  /// TODO: Ok? It is not known
  public var maxUtility: Double { fatalError("Cannot be calcuated for GridMaze") }
  /// TODO: Ok? It is not known
  public var utilitySum: Double? { return nil }  // Only known for very specific (and trivial) mazes
  public var playerCount: Int { return 1 }  // Only one player navigating (and terminal player does not count)
  public var maxGameLength: Int { fatalError("Cannot be calcuated for GridMaze") }
  /// TODO: Ok? It is not known
  public static var info: GameInfo {
    fatalError("Cannot be (easily) caluated for GridMaze")
    // TODO
    //      return GameInfo(
    //        shortName: "grid_maze",
    //        longName: "Grid Maze",
    //        dynamics: .simultaneous,  /// TODO: "Every player acts at each stage." Equivalent to .sequential as there is only one player
    //        chanceMode: TODO,  /// TODO: Can change dynamically (even after init) with JumpSpecifications != 1
    //        information: .perfect,  /// TODO: Depends on API user probes environment
    //        utility: .generalSum,
    //        rewardModel: .rewards, /// TODO: Can change dynamically (even after init)
    //        maxPlayers: 1,
    //        minPlayers: 1,
    //        providesInformationState: true,
    //        providesInformationStateAsNormalizedVector: true
    //      )
  }

  public var initialState: GameState {
    let slist = maze.flatMap({ $0 }).filter({ $0.isInitial })
    if slist.count == 1 {
      return GameState(gridCell: slist[0])
    } else {
      fatalError(
        "GridMaze misconfigured: One (and only one) initial state is required and supported")
    }
  }

  public var allActions: [Action] { return GridMaze.Action.allCases }

  // TODO
  // Spiel docs refer to ImperfectInformationState.informationStateAsNormalizedVector`
  // cannot find ImperfectInformationState so
  // "see the documentation of that function for details of the data layout." not possible
  // Full inforation state is position (of robot) in maze, either [1] if we
  // flatMap maze and use position index, alt [1, 1] if we use (row,col)?
  // What should we return, TicTacToe returns [3*boardSize*boardSize]
  public var informationStateNormalizedVectorShape: [Int] { [1] }

  // *** Section 2: Native things for GridMaze
  public var maze: [[GridCell]] = [[GridCell]]()

  public var rowCount: Int
  public var colCount: Int

  //----
  public init(
    rowCount: Int, colCount: Int,
    cellLeftSide: GridCell = BOUNCE(reward: -1.0),
    cellTopSide: GridCell = BOUNCE(reward: -1.0),
    cellBottomSide: GridCell = BOUNCE(reward: -1.0),
    cellRightSide: GridCell = BOUNCE(reward: -1.0),
    cellTopLeft: GridCell = BOUNCE(reward: -1.0),
    cellTopRight: GridCell = BOUNCE(reward: -1.0),
    cellBottomLeft: GridCell = BOUNCE(reward: -1.0),
    cellBottomRight: GridCell = BOUNCE(reward: -1.0),
    cellAllOthers: GridCell = SPACE(reward: -1.0)
  ) {
    precondition(rowCount > 0 && colCount > 0, "Row and column count must be larger than 0")

    self.rowCount = rowCount
    self.colCount = colCount

    // Allocate and assign state entries for each cell in grid
    // TODO: Can we use init(repeating: ..., count: ...) with [[GridCell]] in an elegant way?
    maze.reserveCapacity(rowCount)
    for ridx in 0..<rowCount {
      maze.append([GridCell]())
      for cidx in 0..<colCount {
        maze[ridx].append(cellAllOthers)
        self[ridx, cidx] = cellAllOthers  // Sets state in cell element
      }
    }

    for i in 0..<rowCount {
      self[i, 0] = cellLeftSide
      self[i, 0] = cellLeftSide
      self[i, colCount - 1] = cellRightSide
    }
    for i in 0..<colCount {
      self[0, i] = cellTopSide
      self[rowCount - 1, i] = cellBottomSide
    }

    self[0, 0] = cellTopLeft
    self[0, colCount - 1] = cellTopRight
    self[rowCount - 1, 0] = cellBottomLeft
    self[rowCount - 1, colCount - 1] = cellBottomRight
  }

  //---
  public subscript(row: Int, col: Int) -> GridCell {
    get { return self.maze[row][col] }
    set {
      self.maze[row][col] = newValue
      self.maze[row][col].game = self
      self.maze[row][col].row = row
      self.maze[row][col].col = col
      // TODO: Do full maze validation, e.g. to avoid chained JumpSpecifications or landing at !isVisitable cells
    }
  }
}

/// Extensions required by Spiel framework
extension GridMaze.GameState {

  init(gridCell: GridCell) {
    self.gridCell = gridCell
    self.game = gridCell.game!
  }

  public var legalActionsMask: [Bool] {
    if gridCell.isTerminal {  // No actions available from terminal states
      return [Bool]()
    }

    var mask: [Bool] = Array(repeating: false, count: GridMaze.Action.allCases.count)
    mask
      = zip(mask, GridMaze.Action.allCases).map {
        gridCell.getTargetWOJumpSpecification(takingAction: $0.1).canAttemptToBeEntered
      }
    return mask
  }

  public func utility(for player: Player) -> Double { return utility }

  public mutating func apply(_ action: GridMaze.Action) {
    let gridCell2 = gridCell.getTargetWJumpSpecification(takingAction: action)
    gridCell = gridCell2
    precondition(
      gridCell.canAttemptToBeEntered && gridCell.canBeEntered,
      "Action: \(action) is illegal from position: [\(gridCell.row!),\(gridCell.col!)]")
    utility += gridCell.rewardOnEnter
    history.append(action)
  }

  // TODO: How do we work with this and JumpSpecification? Is that what we're supposed to do?
  public var chanceOutcomes: [GridMaze.Action: Double] {
    /// From Spiel:
    /// Note: what is returned here depending on the game's chanceMode (in
    /// its GameInfo):
    ///   - Option 1. `.explicitStochastic`. All chance node outcomes are returned along
    ///     with their respective probabilities. Then State.apply(...) is deterministic.
    ///   - Option 2. `.sampledStochastic`. Return a dummy single action here with
    ///     probability 1, and then `State.apply(_:)` does the real sampling. In this
    ///     case, the game has to maintain its own RNG.
    fatalError("Method not implemented")
  }

  // TODO: Correct understanding?
  public func informationStateAsNormalizedVector(for player: Player) -> [Double] {
    return [Double(game.colCount * gridCell.row! + gridCell.col!)]  // Each position is a unique info state
  }

  // TODO: Correct understanding?
  public func informationState(for player: Player) -> String {
    return GridMaze.GameState.informationStateImpl(gridCell: gridCell)
  }

  fileprivate static func informationStateImpl(gridCell: GridCell) -> String {
    return String(format: "%d:%d", gridCell.row!, gridCell.col!)  // Do this so developer gets a readable position
  }
}

/// ********************************************************************

//----
// JumpSpecification allows stochastic behavior configuration for grid cells
public typealias JumpSpecificationProbability = (js: JumpSpecification, prob: Double)
public enum JumpSpecification: Equatable, CustomStringConvertible {
  case Welcome  // I am happy to welcome you to my space
  case BounceBack  // You cannot enter my space, bounce back (e.g. Wall)
  case Relative(Int, Int)  // Teleport (extend to also support a function argument)
  case Absolute(Int, Int)  // Teleport (extend to also support a function argument)

  public var description: String {
    switch self {
    case .Welcome: return "Welcome"
    case .BounceBack: return "BounceBack"
    case let .Relative(row, col): return String(format: "Relative (%d,%d)", row, col)
    case let .Absolute(row, col): return String(format: "Absolute (%d,%d)", row, col)
    }
  }
}

//----
// Defines behavior for a cell in the GridMaze
public struct GridCell {
  var oneWordDescription: String
  var game: GridMaze?  // Set by GridMaze.subscript
  var row, col: Int?  // Position of GridCell, set by GridMaze.subscript (optional because not assigned in init)

  var rewardOnEnter: Double  // If you enter, if went elsewhere due to JumpSpecification then that target cells R is used

  // TODO: Need to read these from public scope, can we statically limit write to private scope?
  public var canAttemptToBeEntered: Bool  // True if it is legal to take an action towards entering this cell
  public var canBeEntered: Bool {  // False if you can never actually arrive at this cell, i.e. does not have a welcoming JS)
    return entryJumpProbabilities.count == 0
      || entryJumpProbabilities.contains { $0.js == .Welcome && $0.prob > 0 }
  }

  public var isInitial: Bool
  public var isTerminal: Bool

  // Stoachstic behavior for cell
  // When attempting to enter cell, this specification can specify alternative behavior than just entering cell
  // (i.e compare to the the wind in Frozen Lake environment)
  public var entryJumpProbabilities = [JumpSpecificationProbability]() {
    willSet {
      let probSum = newValue.reduce(0) { $0 + $1.prob }
      guard newValue.count == 0 || probSum == 1.0
      else {  // All probabilities must sum to 1.0
        fatalError("Jump probabilities don't sum to 1.0")
      }
    }
  }

  public var hasProbabilisticTarget: Bool {
    return entryJumpProbabilities.filter { $0.js != .Welcome }.count > 0
  }

  init(
    oneWordDescription: String,
    reward: Double,
    entryJumpProbabilities: [JumpSpecificationProbability] = [],
    isInitial: Bool,
    isTerminal: Bool,
    canAttemptToBeEntered: Bool
  ) {
    self.oneWordDescription = oneWordDescription
    self.rewardOnEnter = reward
    self.isInitial = isInitial
    self.isTerminal = isTerminal
    self.canAttemptToBeEntered = canAttemptToBeEntered
    self.entryJumpProbabilities = entryJumpProbabilities
  }

  // Not considering JumpSpecifications, which cell would takaingAction lead to
  // (used to find out the JS that might be applied)
  func getTargetWOJumpSpecification(takingAction: GridMaze.Action) -> GridCell {
    // Find new co-ordinates wrapping if needed
    var newRow = row!
    var newCol = col!
    switch takingAction {
    case .LEFT:
      newCol -= 1
      if col == 0 {
        newCol = game!.colCount - 1
      }
    case .UP:
      newRow -= 1
      if row == 0 {
        newRow = game!.rowCount - 1
      }
    case .DOWN:
      newRow += 1
      if row == game!.rowCount - 1 {
        newRow = 0
      }
    case .RIGHT:
      newCol += 1
      if col == game!.colCount - 1 {
        newCol = 0
      }
    }

    // Returned state may !isVisitable, needs to checked by caller
    return game![newRow, newCol]
  }

  // Find target cell takaingAction, applying any stochastic JumpSpecification behavior
  func getTargetWJumpSpecification(takingAction: GridMaze.Action) -> GridCell {
    let targetCell = getTargetWOJumpSpecification(takingAction: takingAction)
    if !targetCell.hasProbabilisticTarget {
      return targetCell
    }

    let probabilities = targetCell.entryJumpProbabilities.map { $0.prob }
    let probabilityIndex = randomIndexFromProbabilities(probabilityList: probabilities)
    let targetCell2 = calculateTargeCellJumpSpecification(
      originCell: self,
      targetCell: targetCell,
      js: targetCell.entryJumpProbabilities[probabilityIndex].js)
    return targetCell2
  }

  // If js is stocahstically selected, return target cell js could end up in
  func calculateTargeCellJumpSpecification(
    originCell: GridCell,  // Neeeded for bounce-back
    targetCell: GridCell,
    js: JumpSpecification
  ) -> GridCell {

    var rv = targetCell
    switch js {
    case .Welcome:
      rv = targetCell
    case .BounceBack:
      rv = originCell
    case let .Absolute(row, col):
      rv = game![row, col]
    case let .Relative(row, col):
      rv = game![targetCell.row! + row, targetCell.col! + col]
    }
    // Only support 1-step jumps. I.e, cannot jump to a state having JSs specifying additional jumps
    precondition(
      js == .Welcome || !rv.hasProbabilisticTarget,
      "At cell (\(originCell.row!),\(originCell.col!)). Target cell (\(rv.row!),\(rv.col!)) was not purely welcoming"
    )
    return rv
  }

  public typealias ProbRewardCell = (prob: Double, reward: Double, cell: GridCell)

  public func probeActionTargets(takingAction: GridMaze.Action) -> [(ProbRewardCell)] {
    precondition(
      canBeEntered && canAttemptToBeEntered && !isTerminal,
      "Cannot probeActionWithProbabilities from this cell (\(row!),\(col!))")
    let targetCell = getTargetWOJumpSpecification(takingAction: takingAction)
    precondition(
      targetCell.canAttemptToBeEntered,
      "Target cell from (\(row!),\(col!)) taking action: \(takingAction) is not visitable")
    if targetCell.entryJumpProbabilities.count == 0 {
      return [(1.0, targetCell.rewardOnEnter, targetCell)]
    }

    let result: [ProbRewardCell] = targetCell.entryJumpProbabilities.map {
      let tc = calculateTargeCellJumpSpecification(
        originCell: self,
        targetCell: targetCell,
        js: $0.js)
      return ($0.prob, tc.rewardOnEnter, tc)  // TODO: Swift question, is single element () implicitly type convertible to [()]
    }
    return result
  }
}

// ***************************************************************************
// Create factory functions for common Cell Types to use in GridMazes

public func SPACE(reward: Double = -1) -> GridCell {
  return GridCell(
    oneWordDescription: "SPACE", reward: reward,
    isInitial: false, isTerminal: false,
    canAttemptToBeEntered: true)
}

public func START(reward: Double = -1) -> GridCell {
  return GridCell(
    oneWordDescription: "START", reward: reward,
    isInitial: true, isTerminal: false,
    canAttemptToBeEntered: true)
}

public func GOAL(reward: Double) -> GridCell {
  return GridCell(
    oneWordDescription: "GOAL", reward: reward,
    isInitial: false, isTerminal: true,
    canAttemptToBeEntered: true)
}

public func HOLE(reward: Double) -> GridCell {
  return GridCell(
    oneWordDescription: "HOLE", reward: reward,
    isInitial: false, isTerminal: true,
    canAttemptToBeEntered: true)
}

public func WALL(reward: Double = -Double.infinity) -> GridCell {
  return GridCell(
    oneWordDescription: "WALL", reward: reward,
    isInitial: false, isTerminal: false,
    canAttemptToBeEntered: false)
}

public func BOUNCE(reward: Double = -1) -> GridCell {
  return GridCell(
    oneWordDescription: "BOUNCE", reward: reward,
    entryJumpProbabilities: [(prob: 1.0, js: .BounceBack)],
    isInitial: false, isTerminal: false,
    canAttemptToBeEntered: true)
}

// ***************************************************************************
// GridMaze printing functions

extension GridMaze {
  public typealias QTableType = [String: [(action: GridMaze.Action, qvalue: Double)]]
  public typealias VTableType = [String: Double]
  public typealias PTableType = [String: [GridMaze.Action]]

  public func printMazeAndTable(
    header: String,  // header string
    vtable: VTableType? = nil,
    qtable: QTableType? = nil,
    ptable: PTableType? = nil,
    printFullFloat: Bool = false
  ) {

    // Print header
    if header.count > 0 { print(header) }

    // Print transition probabilities (JumpSpecifications) in top section (i.e. not as part of cell in grid)
    var transitionProbabilitiesFound = false
    for (ri, r) in maze.enumerated() {
      for (ci, cell) in r.enumerated() {
        let jsps: [JumpSpecificationProbability] = cell.entryJumpProbabilities
        if jsps.count > 0 {
          if (jsps.count == 1 && jsps[0].js == JumpSpecification.Welcome) {
            break
          }
          if !transitionProbabilitiesFound {  // Print header
            print(
              "\nTransition probabilities (non-stochastic transitions are not printed (i.e as expected based on action)"
            )
          }

          print("[\(ri),\(ci)]: ", terminator: "")
          for (i, jsp) in jsps.enumerated() {
            if (i > 0) { print("       ", terminator: "") }
            print("\(String(format: "Probability: %0.2f", jsp.prob)), Type: \(jsp.js.description)")
            transitionProbabilitiesFound = true
          }
        }
      }
    }
    if transitionProbabilitiesFound {
      print()
    }

    // Get cell definitions strings
    let cellStrs = getCellDefinitionStrs(printFullFloat: printFullFloat)

    // Get v-, q-, and policy-value strings
    let (vtableStrs, qtableStrs, ptableStrs) = getTableAndPolicyStrs(
      vtable: vtable,
      qtable: qtable,
      ptable: ptable,
      printFullFloat: printFullFloat)

    // Get the maximum column width needed to support maze def, q-/v-/policy-table printing
    let colCount = cellStrs[0].count
    var colMaxWidth = 2
    for i in 0..<colCount {
      let maxWidth = getLargestColSize(
        col: i, maze: cellStrs, vtable: vtableStrs, qtable: qtableStrs, policy: ptableStrs)
      if maxWidth > colMaxWidth {
        colMaxWidth = maxWidth
      }
    }

    // Print the columns on first row
    print("        ", terminator: "")  // Row index pre-space
    for i in 0..<colCount {
      print(strCenter(str: String(format: "%02d", i), len: colMaxWidth), terminator: "")
      print("  ", terminator: "")
    }
    print()

    // Print grid cell rows with maze definition, tables, and policy parts
    for (ri, r) in cellStrs.enumerated() {
      print(String(format: "%02d      ", ri), terminator: "")  // row index
      // Print cell definition
      for c in r {
        let ccenter = strCenter(str: c, len: colMaxWidth)
        print("\(ccenter)  ", terminator: "")
      }
      print()
      // Potentially print vtable
      if let vts = vtableStrs?[ri] {
        for (ci, c) in vts.enumerated() {
          let ccenter = strCenter(str: c, len: colMaxWidth)
          if (ci == 0) {
            print("VTable  ", terminator: "")
          }
          print("\(ccenter)  ", terminator: "")
        }
        print()
      }
      // Potentially print qtable
      if let qts = qtableStrs?[ri] {
        for (ci, c) in qts.enumerated() {
          let ccenter1 = strCenter(str: c[0], len: colMaxWidth)
          if ci == 0 {
            print("QTable  ", terminator: "")
          }
          print("\(ccenter1)  ", terminator: "")
        }
        print()
        for (ci, c) in qts.enumerated() {
          let ccenter1 = strCenter(str: c[1], len: colMaxWidth)
          if ci == 0 {
            print("        ", terminator: "")
          }
          print("\(ccenter1)  ", terminator: "")
        }
        print()
      }
      // Potentially print policy
      if let ps = ptableStrs?[ri] {
        for (ci, c) in ps.enumerated() {
          let ccenter = strCenter(str: c, len: colMaxWidth)
          if (ci == 0) {
            print("Policy  ", terminator: "")
          }
          print("\(ccenter)  ", terminator: "")
        }
        print()
      }
      print()
    }
  }

  func getCellDefinitionStrs(printFullFloat: Bool) -> [[String]] {
    var cellDescriptions: [[String]] = []

    for row in maze {
      var rowStrs = [String]()
      var str = ""
      for cell in row {
        var jsStr = ""  // Add a note about the existance of a JS if one exists
        if cell.entryJumpProbabilities.count > 1
          || (
            cell.entryJumpProbabilities.count == 1 && cell.entryJumpProbabilities[0].js != .Welcome
          )
        {
          jsStr = "(JS)"
        }
        str
          = "\(cell.oneWordDescription)\(jsStr):\(double2Str(printFullFloat: printFullFloat, value: cell.rewardOnEnter))"
        if cell.isTerminal {
          str += ":T"
        } else if cell.isInitial {
          str += ":I"
        }
        rowStrs.append(str)
      }
      cellDescriptions.append(rowStrs)
    }
    return cellDescriptions
  }

  func getTableAndPolicyStrs(
    vtable: VTableType? = nil,
    qtable: QTableType? = nil,
    ptable: PTableType? = nil,
    printFullFloat: Bool
  ) -> ([[String]]?, [[[String]]]?, [[String]]?) {

    var vtableResult: [[String]]? = nil
    if vtable != nil {
      vtableResult = []
    }
    var qtableResult: [[[String]]]? = nil
    if qtable != nil {
      qtableResult = []
    }
    var ptableResult: [[String]]? = nil
    if ptable != nil {
      ptableResult = []
    }

    for (ri, r) in maze.enumerated() {
      var vtableStrs = [String]()
      var qtableStrs = [[String]]()
      var ptableStrs = [String]()
      for ci in 0..<r.count {
        let informationState = GridMaze.GameState.informationStateImpl(gridCell: maze[ri][ci])

        // QTable
        // TODO: This is ugly, what is the beatiful Swift way to write this?
        var qs1 = "*"
        var qs2 = "*"
        if maze[ri][ci].canAttemptToBeEntered && maze[ri][ci].canBeEntered
          && !maze[ri][ci].isTerminal
        {
          qs1 = ""
          qs2 = ""
          var q = [(action: GridMaze.Action, qvalue: Double)]()

          if let qtmp = qtable?[informationState] {
            q = qtmp
          } else {
            let gs = GameState(gridCell: maze[ri][ci])
            let qvaluePart = 1.0 / Double(gs.legalActions.count)
            q = gs.legalActions.map { ($0, qvaluePart) }
          }
          _
            = q.map {
              if $0 == .UP || $0 == .DOWN {
                if qs1.count > 0 { qs1 += " " }
                qs1 += "\($0.description):\(double2Str(printFullFloat: printFullFloat, value: $1))"
              } else {
                if qs2.count > 0 { qs2 += " " }
                qs2 += "\($0.description):\(double2Str(printFullFloat: printFullFloat, value: $1))"
              }
            }
        }
        if qtable != nil {
          qtableStrs.append([qs1, qs2])
        }

        // VTable
        var vs = "*"
        if maze[ri][ci].canAttemptToBeEntered && maze[ri][ci].canBeEntered {
          vs = "0"
          if let v = vtable?[informationState] {
            vs = double2Str(printFullFloat: printFullFloat, value: v)
          }
        }
        if vtable != nil {
          vtableStrs.append(vs)
        }

        // PTable
        var ps = "*"
        if maze[ri][ci].canAttemptToBeEntered && maze[ri][ci].canBeEntered
          && !maze[ri][ci].isTerminal
        {
          ps = "?"
          if let list = ptable?[informationState] {
            ps = list.reduce("?") { str, elem in "\(str)\(elem)," }
            if ps.suffix(1) == "," {
              ps = String("\(ps.dropFirst().dropLast())")
            }
          }
        }
        if ptable != nil {
          ptableStrs.append(ps)
        }
      }  // End-for iterating through GridCells in maze

      qtableResult?.append(qtableStrs)
      vtableResult?.append(vtableStrs)
      ptableResult?.append(ptableStrs)
    }
    return (vtableResult, qtableResult, ptableResult)
  }

  // Format value to print nicely in tables
  func double2Str(printFullFloat: Bool, value: Double) -> String {
    if printFullFloat {
      return String(format: "%f", value)
    }

    // Print as an integer
    if value.truncatingRemainder(dividingBy: 1) == 0 {
      return String(format: "%d", Int(value.rounded(.toNearestOrAwayFromZero)))
    }

    // Deal with inifinity (may very well exist, for rewards of cells that can never be entered)
    if value == Double.infinity {
      return "inf"
    }
    if value == -Double.infinity {
      return "-inf"
    }

    // Value is >=10 or <=-10, then skip the decimals
    if value >= 10.0 || value <= -10.0 {
      return String(format: "%d", Int(value.rounded(.toNearestOrAwayFromZero)))
    }

    // Print 1 decimal if >=5 or <=5, otherwise limit to 2
    var r = ""
    if value >= 5.0 || value <= -5.0 {
      r = String(format: "%0.1f", value)  // Drop second decimal, it's 0 or neglectible
    } else {
      r = String(format: "%0.2f", value)
    }

    // If a decimal point exist, remove all trailig 0s
    if r.contains(".") {
      while r.suffix(1) == "0" {
        r = String(r.dropLast(1))
      }
      if r.suffix(1) == "." {  // If trimming leaved '.' as trailer then remove that also
        r = String(r.dropLast(1))
      }
    }

    // Remove a leading zero
    if r.starts(with: "0") {
      return String(r.dropFirst(1))
    }
    // Remove a leading zero after '-'
    if r.starts(with: "-0") {
      return "-" + r.dropFirst(2)
    }
    return r
  }

  // Given all values, what is the longest string
  // This value is used to pad all other values to get symmetric table
  func getLargestColSize(
    col: Int, maze: [[String]], vtable: [[String]]?, qtable: [[[String]]]?, policy: [[String]]?
  ) -> Int {
    let mcols = maze.map { $0[col] }
    let vcols = vtable?.map { $0[col] } ?? []
    let qcols1 = qtable?.map { $0[col][0] } ?? []
    let qcols2 = qtable?.map { $0[col][1] } ?? []
    let pcols = policy?.map { $0[col] } ?? []
    return (mcols + vcols + qcols1 + qcols2 + pcols).max { $0.count < $1.count }!.count  // at least maze is non-empty
  }
}

// **************************************************************************
// Miscellaneous functions needed

//----
// TODO: Does a function like this existin in Swift library
fileprivate func randomIndexFromProbabilities(probabilityList: [Double]) -> Int {
  let sum = probabilityList.reduce(0, +)
  let randomNumber = Double.random(in: 0.0..<sum)

  var ladder = 0.0
  for (idx, probability) in probabilityList.enumerated() {
    ladder += probability
    if randomNumber < ladder {
      return idx
    }
  }
  // If everything falls through, then pick last one
  return (probabilityList.count - 1)
}

//----
// TODO: Does a function like this existin in Swift library
//  (i.e shift content of str to the center of len (and pad with " ")
fileprivate func strCenter(str: String, len: Int) -> String {
  precondition(len >= str.count, "Cannot center text if it's larger than space")
  var r = str
  let space = len - r.count
  let even = space / 2
  let rem = space % 2

  if rem > 0 {
    r = " \(r)"
  }
  for _ in 0..<even {
    r = " \(r) "
  }
  return r
}
