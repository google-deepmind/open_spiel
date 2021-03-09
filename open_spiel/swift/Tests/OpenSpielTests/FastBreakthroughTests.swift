import XCTest
import OpenSpiel

final class FastBreakthroughTests: XCTestCase {

  func testBoardLocations() throws {
    let game = FastBreakthrough()
    XCTAssertTrue(game.isValid(location: FastBreakthrough.BoardLocation(x: 0, y: 0)))
    XCTAssertTrue(game.isValid(location: FastBreakthrough.BoardLocation(x: 3, y: 6)))
    XCTAssertFalse(game.isValid(location: FastBreakthrough.BoardLocation(x: 7, y: 8)))
    XCTAssertFalse(game.isValid(location: FastBreakthrough.BoardLocation(x: 9, y: 9)))
  }

  func testBoardLocationMovement() throws {
    XCTAssertEqual(FastBreakthrough.BoardLocation(x: 0, y: 0),
                   FastBreakthrough.BoardLocation(x: 1, y: 1).move(in: .left, for: .black))
    XCTAssertEqual(FastBreakthrough.BoardLocation(x: 1, y: 1),
                   FastBreakthrough.BoardLocation(x: 0, y: 0).move(in: .right, for: .white))
  }

  func testBoardSubscript() {
      var board = FastBreakthrough.Board()
      XCTAssertNil(board[0, 0])
      board[0, 0] = .black
      XCTAssertEqual(board[0, 0], .black, "\(board)")

      board[0, 0] = .white
      XCTAssertEqual(board[0, 0], .white, "\(board)")

      board[0, 0] = nil
      XCTAssertNil(board[0, 0])

      XCTAssertNil(board[5, 6])
      board[5, 6] = .white
      XCTAssertEqual(.white, board[5, 6])
      board[5, 6] = nil
      XCTAssertNil(board[5, 6])

      XCTAssertNil(board[7, 7])
      board[7, 7] = .black
      XCTAssertEqual(.black, board[7, 7])
      board[7, 7] = nil
      XCTAssertNil(board[7, 7])
  }

  func testStateInitialization() throws {
    let smallGame = FastBreakthrough()
    let state = smallGame.initialState
    for i in UInt8(0)..<8 {
      XCTAssertEqual(state[i, 0], .white)
      XCTAssertEqual(state[i, 1], .white)
      XCTAssertEqual(state[i, 2], nil)
      XCTAssertEqual(state[i, 3], nil)
      XCTAssertEqual(state[i, 4], nil)
      XCTAssertEqual(state[i, 5], nil)
      XCTAssertEqual(state[i, 6], .black)
      XCTAssertEqual(state[i, 7], .black)
    }
    XCTAssertEqual(state.currentPlayer, FastBreakthrough.BreakthroughPlayer.black.player)
    XCTAssertEqual(FastBreakthrough.BreakthroughPlayer(state.currentPlayer), .black)
  }

  func testSimpleGame() throws {
    // let smallGame = FastBreakthrough(boardHeight: 5, boardWidth: 2)
    let smallGame = FastBreakthrough()
    var state = smallGame.initialState

    func debug() {
    //   print("\(state)")  // Uncomment me to watch the game progress.
    }
    func prettyPrintActions(_ actions: [FastBreakthrough.Action]) -> String {
    let sorted = actions.sorted()
    var output = "[\n"
    for action in sorted {
        output.append("  \(action)\n")
    }
    output.append("]")
    return output
    }

    XCTAssertEqual(state.currentPlayer, .player(0))

    do {
      let initialActions: [FastBreakthrough.Action] = Array(UInt8(0)..<8).flatMap { x -> [FastBreakthrough.Action] in
        var actions = [FastBreakthrough.Action]()
        if x != 0 {
          actions.append(.init(location: .init(x: x, y: 6), direction: .left))
        }
        if x != 7 {
          actions.append(.init(location: .init(x: x, y: 6), direction: .right))
        }
        actions.append(.init(location: .init(x: x, y: 6), direction: .forward))
        return actions
      }
      XCTAssertEqual(Set(state.legalActions), Set(initialActions), "Want: \(prettyPrintActions(initialActions)), got: \(prettyPrintActions(state.legalActions))")
    }
    state.apply(.init(location: .init(x: 0, y: 6), direction: .forward))
    debug()

    do {
      let initialWhiteActions = Array(UInt8(0)..<8).flatMap { x -> [FastBreakthrough.Action] in
        var actions = [FastBreakthrough.Action]()
        if x != 0 {
          actions.append(.init(location: .init(x: x, y: 1), direction: .left))
        }
        if x != 7 {
          actions.append(.init(location: .init(x: x, y: 1), direction: .right))
        }
        actions.append(.init(location: .init(x: x, y: 1), direction: .forward))
        return actions
      }
      XCTAssertEqual(Set(state.legalActions), Set(initialWhiteActions))
    }
    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssertEqual(state.history.count, 1)
    state.apply(.init(location: .init(x: 1, y: 1), direction: .forward))
    debug()
    XCTAssertEqual(state.currentPlayer, .player(0))
    XCTAssertEqual(state.history.count, 2)
    state.apply(.init(location: .init(x: 0, y: 5), direction: .forward))
    debug()
    XCTAssertEqual(state.currentPlayer, .player(1))
    XCTAssertEqual(state.history.count, 3)
    
    state.apply(.init(location: .init(x: 1, y: 2), direction: .forward))
    debug()
    state.apply(.init(location: .init(x: 0, y: 4), direction: .right))
    debug()
    state.apply(.init(location: .init(x: 0, y: 1), direction: .forward))
    debug()
    state.apply(.init(location: .init(x: 1, y: 3), direction: .left))
    debug()
    state.apply(.init(location: .init(x: 1, y: 0), direction: .forward))
    debug()
    state.apply(.init(location: .init(x: 0, y: 2), direction: .right))
    debug()
    state.apply(.init(location: .init(x: 7, y: 1), direction: .forward))
    debug()
    state.apply(.init(location: .init(x: 1, y: 1), direction: .left))
    debug()
    XCTAssertEqual(state.history.count, 11)
    XCTAssertEqual(state.currentPlayer, .terminal)
    XCTAssertEqual(state.winner, .black)
    XCTAssertEqual(state.utility(for: .player(0)), 1)
    XCTAssertEqual(state.utility(for: .player(1)), -1)
  }

  // TODO: Test all legal action rules.
}

extension FastBreakthroughTests {
  static var allTests = [
    ("testBoardLocations", testBoardLocations),
    ("testBoardSubscript", testBoardSubscript),
    ("testBoardLocationMovement", testBoardLocationMovement),
    ("testStateInitialization", testStateInitialization),
    ("testSimpleGame", testSimpleGame),
  ]
}
