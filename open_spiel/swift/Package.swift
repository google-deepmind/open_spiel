// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "OpenSpiel",
  platforms: [.macOS(.v10_13)],
  products: [
    .library(
      name: "OpenSpiel",
      targets: ["OpenSpiel"]),
    .executable(
      name: "GridMazeExample",
      targets: ["GridMazeExample"]),
    .executable(
      name: "KuhnPokerExample",
      targets: ["KuhnPokerExample"]),
    .executable(
      name: "TexasHoldemBenchmark",
      targets: ["TexasHoldemBenchmark"]),
  ],
  dependencies: [
    .package(name: "Benchmark", url: "https://github.com/google/swift-benchmark.git", from: "0.1.0"),
  ],
  targets: [
    .target(
      name: "OpenSpiel",
      dependencies: [],
      path: "swift/Sources/OpenSpiel"),
    .testTarget(
      name: "OpenSpielTests",
      dependencies: ["OpenSpiel"],
      path: "swift/Tests/OpenSpielTests"),
    .target(
      name: "GridMazeExample",
      dependencies: ["OpenSpiel"],
      path: "swift/Examples/GridMaze"),
    .target(
      name: "KuhnPokerExample",
      dependencies: ["OpenSpiel"],
      path: "swift/Examples/KuhnPoker"),
    .target(
      name: "TexasHoldemBenchmark",
      dependencies: ["OpenSpiel"],
      path: "swift/Examples/TexasHoldemBenchmark"),
    .target(
      name: "Benchmarks",
      dependencies: ["OpenSpiel", "Benchmark"],
      path: "swift/Benchmarks"),
  ]
)
