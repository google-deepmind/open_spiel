import XCTest

#if !os(macOS)
  public func allTests() -> [XCTestCaseEntry] {
    return [
      testCase(BreakthroughTests.allTests),
      testCase(FastBreakthroughTests.allTests),
      testCase(KuhnPokerTests.allTests),
      testCase(LeducPokerTests.allTests),
      testCase(PokerDeckTests.allTests),
      testCase(TexasHoldemTests.allTests),
      testCase(ExploitabilityTests.allTests),
      testCase(TicTacToeTests.allTests),
      testCase(GoPointTests.allTests),
      testCase(GoBoardTests.allTests),
      testCase(ExploitabilityDescentTests.allTests)
    ]
  }
#endif
