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

import XCTest
import OpenSpiel

// Generates a go board from the given string, setting X to black stones and O
// to white stones. The first character of the first line is mapped to A1, the
// second character to B1, etc, as below:
//     ABCDEFGH
//   1 ++++XO++
//   2 XXXXXO++
//   3 OOOOOO++
//   4 ++++++++
func createBoard(_ initialStones: String, boardSize: Int = 19) -> Go.Board? {
  guard var board = Go.Board(size: boardSize) else { return nil }

  var row = 0
  for line in initialStones.split(separator: "\n") {
    var col = 0
    var stonesStarted = false
    for c in line {
      if c == " " {
        precondition(!stonesStarted, "whitespace is only allowed at the start of the line")
        continue
      } else if c == "X" {
        stonesStarted = true
        precondition(board.play(point: Go.Point.from2D((row, col)), for: .black))
      } else if c == "O" {
        stonesStarted = true
        precondition(board.play(point: Go.Point.from2D((row, col)), for: .white))
      } else if c == "+" {
        stonesStarted = true
      } else {
        fatalError("unexpected character '\(c)' in '\(line)'")
      }
      col += 1
    }
    if stonesStarted {
      row += 1
    }
  }

  return board
}

final class GoPointTests: XCTestCase {
  static var allTests = [
    ("special_points", testSpecialPoints),
    ("roundtrip", testRoundtrip),
    ("board_points", testBoardPoints),
  ]

  let boardSize = 19

  func testSpecialPoints() {
    XCTAssertEqual(Go.Point.invalid, Go.Point.fromString("a"))
    XCTAssertEqual(Go.Point.pass, Go.Point.fromString("pass"))
  }

  func testRoundtrip() {
    for row in 0..<boardSize {
      for col in 0..<boardSize {
        let p = Go.Point.from2D((row, col))
        let (gotRow, gotCol) = p.coordinates!
        XCTAssertEqual(gotRow, row, "Unexpected row for \(p.description)")
        XCTAssertEqual(gotCol, col, "Unexpected column for \(p.description)")

        let restoredPoint = Go.Point.fromString(p.description)
        XCTAssertEqual(restoredPoint, p, "Failed to recover point at \(row) x \(col)")
      }
    }

    XCTAssertEqual("a1", Go.Point.from2D((0, 0)).description)
    XCTAssertEqual("t19", Go.Point.from2D((18, 18)).description)

    let (gotRow, gotCol) = Go.Point.fromString("a1").coordinates!
    XCTAssertEqual(gotRow, 0)
    XCTAssertEqual(gotCol, 0)

    let (gotRow2, gotCol2) = Go.Point.fromString("t19").coordinates!
    XCTAssertEqual(gotRow2, 18)
    XCTAssertEqual(gotCol2, 18)

    let (gotRow3, gotCol3) = Go.Point.fromString("d3").coordinates!
    XCTAssertEqual(gotRow3, 2)
    XCTAssertEqual(gotCol3, 3)

    // Handle invalid points.
    for invalidPoint in ["ab2", "z3", "a20", "d0", "f-5", "foobar", "&*82"] {
      XCTAssertEqual(Go.Point.fromString(invalidPoint), Go.Point.invalid)
    }
  }

  func testBoardPoints() {

    // We only support board sizes up to 19.
    for boardSize in 1...19 {
      let points = Set(Go.boardPoints(boardSize)!)
      XCTAssertEqual(
        points.count,
        boardSize * boardSize,
        "unexpected number of points for \(boardSize): \(points)"
      )
    }

    for invalidSize in [-19, -1, 0, 20, 21] {
      XCTAssertEqual(Go.boardPoints(invalidSize), nil)
    }
  }
}

class GoBoardTests: XCTestCase {
  static var allTests = [
    ("startsEmpty", testStartsEmpty),
    ("playMove", testPlayMove),
    ("isLegal", testIsLegalMove),
    ("libertyCount", testLibertyCount),
    ("isLegalMoveSurrounded", testIsLegalMoveSurrounded),
    ("isLegalMoveSurrounded", testIsLegalMoveSurrounded),
    ("isLegalMoveSurroundedCapture", testIsLegalMoveSurroundedCapture),
    ("isLegalMoveSuicide", testIsLegalMoveSuicide),
    ("IsLegalMoveSuicideAfterCapture", testIsLegalMoveSuicideAfterCapture),
    ("CaptureSingleStone", testCaptureSingleStone),
    ("CaptureGroup", testCaptureGroup),
    ("KoLegalitiy", testKoLegalitiy),
    ("pseudoLibertyCount", testPseudoLibertyCount),
  ]

  let boardSize = 19

  func testStartsEmpty() {
    let board = Go.Board(size: boardSize)!
    for p in Go.boardPoints(boardSize)! {
      XCTAssertEqual(board.color(of: p), .empty)
    }
  }

  func testPlayMove() {
    var board = Go.Board(size: boardSize)!
    XCTAssertTrue(board.play(point: Go.Point.fromString("d4"), for: .black))
    XCTAssertEqual(board.color(of: Go.Point.fromString("d4")), .black)
  }

  func testIsLegalMove() {
    var board = Go.Board(size: boardSize)!

    // Pass is always legal.
    XCTAssertTrue(board.isLegal(point: .pass, for: .black))

    // Can play on empty.
    XCTAssertTrue(board.isLegal(point: Go.Point.fromString("a1"), for: .black))
    XCTAssertTrue(board.play(point: Go.Point.fromString("a1"), for: .black))

    // Can't play on top.
    XCTAssertFalse(board.isLegal(point: Go.Point.fromString("a1"), for: .white));
  }

  func testLibertyCount() {
    var board = Go.Board(size: boardSize)!

    // A point in a corner should start out with two liberties.
    let a1 = Go.Point.fromString("a1")
    XCTAssertTrue(board.play(point: a1, for: .black))
    XCTAssertEqual(board.pseudoLibertyCount(for: a1), 2, board.description)

    // If the opponent plays next to it, it should lose one liberty.
    let a2 = Go.Point.fromString("a2")
    XCTAssertTrue(board.play(point: a2, for: .white))
    XCTAssertEqual(board.pseudoLibertyCount(for: a1), 1, board.description)
    // The opponent stone should also have one liberty fewer.
    XCTAssertEqual(board.pseudoLibertyCount(for: a2), 2, board.description)
  }

  func testIsLegalMoveSurrounded() {
    let board = createBoard(
      """
                            +++++
                            ++O++
                            +O+O+
                            ++O++
                            +++++
                            """
    )!
    // Can play into a fully surrounded spot if it connects with own group..
    XCTAssertTrue(board.isLegal(point: Go.Point.fromString("c3"), for: .white))
    // .. but not if it's the opponents group.
    XCTAssertFalse(board.isLegal(point: Go.Point.fromString("c3"), for: .black), board.description)
  }

  func testIsLegalMoveSurroundedCapture() {
    // Unless that group doesn't have any liberties itself:
    let board = createBoard(
      """
                            ++X++
                            +XOX+
                            XO+OX
                            +XOX+
                            ++X++
                            """
    )!
    XCTAssertFalse(board.isLegal(point: Go.Point.fromString("c3"), for: .white))
    // But it is legal to play there to capture the group.
    XCTAssertTrue(board.isLegal(point: Go.Point.fromString("c3"), for: .black))
  }

  func testIsLegalMoveSuicide() {
    let board = createBoard(
      """
                            +++XO+X+X+
                            ++XOOOOX++
                            ++XOOOX+++
                            ++XOOOX+++
                            +++XXXO+++
                            ++++++++++
                            """
    )!

    // White can't suicide..
    XCTAssertFalse(board.isLegal(point: Go.Point.fromString("f1"), for: .white))
    // .. but black can capture.
    XCTAssertTrue(board.isLegal(point: Go.Point.fromString("f1"), for: .black))
  }

  func testIsLegalMoveSuicideAfterCapture() {
    var board = createBoard(
      """
                            OOO++XO++
                            OXXOOOXX+
                            X+XO+OX++
                            ++XOOOX++
                            ++XXXXX++
                            +++++++++
                            """
    )!

    // Capture the white group in the corner.
    XCTAssertTrue(board.play(point: Go.Point.fromString("d1"), for: .black))
    XCTAssertEqual(1, board.pseudoLibertyCount(for: Go.Point.fromString("c1")))
    XCTAssertEqual(2, board.pseudoLibertyCount(for: Go.Point.fromString("d1")))

    XCTAssertTrue(board.play(point: Go.Point.fromString("b1"), for: .white))
    XCTAssertEqual(0, board.pseudoLibertyCount(for: Go.Point.fromString("c1")))
    XCTAssertEqual(2, board.pseudoLibertyCount(for: Go.Point.fromString("b1")))

    XCTAssertTrue(board.play(point: Go.Point.fromString("a1"), for: .black))

    XCTAssertEqual(0, board.pseudoLibertyCount(for: Go.Point.fromString("c1")));
    XCTAssertEqual(1, board.pseudoLibertyCount(for: Go.Point.fromString("b1")))
    XCTAssertTrue(board.inAtari(Go.Point.fromString("b1")));
    XCTAssertFalse(board.inAtari(Go.Point.fromString("d1")));
    XCTAssertFalse(board.inAtari(Go.Point.fromString("c2")));
    XCTAssertFalse(board.isLegal(point: Go.Point.fromString("c1"), for: .white))
  }

  func testCaptureSingleStone() {
    var board = createBoard(
      """
                            +++++
                            +OOO+
                            +OXO+
                            +O+O+
                            +++++
                            """
    )!
    XCTAssertTrue(board.play(point: Go.Point.fromString("c4"), for: .white))
    XCTAssertEqual(Go.Color.empty, board.color(of: Go.Point.fromString("c3")))
  }

  func testCaptureGroup() {
    var board = createBoard(
      """
                            OOX
                            OXX
                            OX+
                            +X+
                            """
    )!
    XCTAssertTrue(board.play(point: Go.Point.fromString("a4"), for: .black))
    XCTAssertEqual(Go.Color.empty, board.color(of: Go.Point.fromString("a1")))
    XCTAssertEqual(Go.Color.empty, board.color(of: Go.Point.fromString("a2")))
    XCTAssertEqual(Go.Color.empty, board.color(of: Go.Point.fromString("a3")))
  }

  func testKoLegalitiy() {
    var board = createBoard(
      """
                            ++++++
                            ++XO++
                            +XO+O+
                            ++XO++
                            ++++++
                            """
    )!
    // Capturing the ko the first time is legal..
    XCTAssertTrue(board.isLegal(point: Go.Point.fromString("d3"), for: .black))
    XCTAssertTrue(board.play(point: Go.Point.fromString("d3"), for: .black))
    // .. but immediate recapture is not.
    XCTAssertFalse(board.isLegal(point: Go.Point.fromString("c3"), for: .white))

    // After a move somewhere else, the ko can be retaken.
    XCTAssertTrue(board.play(point: Go.Point.fromString("f16"), for: .white))
    XCTAssertTrue(board.isLegal(point: Go.Point.fromString("c3"), for: .white))
  }

  func testPseudoLibertyCount() {
    var board = createBoard(
      """
                            +++++
                            +XX++
                            +O+X+
                            +XX++
                            +++++
                            """
    )!
    // Lone stone starts out with 4 liberties.
    XCTAssertEqual(4, board.pseudoLibertyCount(for: Go.Point.fromString("d3")));

    // An adjacent enemy stones remove liberties..
    XCTAssertTrue(board.play(point: Go.Point.fromString("c3"), for: .white))
    XCTAssertEqual(3, board.pseudoLibertyCount(for: Go.Point.fromString("d3")))

    XCTAssertTrue(board.play(point: Go.Point.fromString("e3"), for: .white))
    XCTAssertEqual(2, board.pseudoLibertyCount(for: Go.Point.fromString("d3")))
    XCTAssertFalse(board.inAtari(Go.Point.fromString("d3")))

    // .. but when the enemey stone is captured the liberties are returned.
    XCTAssertTrue(board.play(point: Go.Point.fromString("a3"), for: .black))
    XCTAssertEqual(3, board.pseudoLibertyCount(for: Go.Point.fromString("d3")))

    // And they can be taken again.
    XCTAssertTrue(board.play(point: Go.Point.fromString("d2"), for: .white))
    XCTAssertEqual(2, board.pseudoLibertyCount(for: Go.Point.fromString("d3")))
    XCTAssertFalse(board.inAtari(Go.Point.fromString("d3")))
  }
}
