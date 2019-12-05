# Swift OpenSpiel

The `swift/` folder contains a port of OpenSpiel to use
[Swift for TensorFlow](https://github.com/tensorflow/swift). This Swift port
explores using a single programming language for the entire OpenSpiel
environment, from game implementations to the algorithms and deep learning
models.

This Swift port is intended for serious research use. As the Swift for
TensorFlow platform matures and gains additional capabilities (e.g. distributed
training), expect the kinds of algorithm that are expressible and tractable to
train to grow significantly.

Contributions welcome for both additional games, and algorithms! If you run into
issues (or would like to share your successes), please do reach out to the Swift
for TensorFlow community at
[`swift@tensorflow.org`](https://groups.google.com/a/tensorflow.org/forum/#!forum/swift).

## Building

To use Swift OpenSpiel, simply download a recent Swift for TensorFlow toolchain
by following the
[installation instructions](https://github.com/tensorflow/swift/blob/master/Installation.md)
(available for macOS and Linux currently). Currently, OpenSpiel builds with the
latest stable toolchains.

Once you have installed the Swift for TensorFlow toolchain, you can build and
test Swift OpenSpiel like a normal Swift package. For example, on the command
line:

```bash
cd swift
swift build  # builds the OpenSpiel library
swift test   # runs all unit tests
```

A simple example of using the OpenSpiel package:
```
import OpenSpiel

let game = TicTacToe()
var state = game.initialState
state.apply(TicTacToe.Action(x: 1, y: 1))
print(state)
```

## Using XCode

To use OpenSpiel as a dependency for an XCode project, you need to use the [Swift Package Manager](https://swift.org/package-manager/) and use it to generate an XCode project. Create an executable package called `foo`:
```
mkdir foo
cd foo
swift package init --type executable
```
Now open the file `Package.swift` that was generated, and add OpenSpiel as a dependency. The contents are now:
```
// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "foo",
    dependencies: [
        .package(url: "https://github.com/deepmind/open_spiel.git", .branch("master")),
    ],
    targets: [
        .target(
            name: "foo",
            dependencies: ["OpenSpiel"]),
        .testTarget(
            name: "fooTests",
            dependencies: ["foo"]),
    ]
)
```
An XCode project can be generated from this package:
```
swift package generate-xcodeproj
open foo.xcodeproj
```
Set the build system to the Legacy Build System (File → Project Settings → Build System) required by [Swift for Tensorflow](https://github.com/tensorflow/swift/blob/master/Installation.md#installation), and you are ready to build using XCode.

## A tour through the code

*   `Spiel.swift` contains the primary abstractions common to all games, such as
    the `GameProtocol` and the `StateProtocol`.
*   There are a number of games each implemented in their own files. There are
    perfect information games, such as TicTacToe and Breakthrough, and there are
    imperfect information games, such as KuhnPoker and LeducPoker.
*   Available algorithms include TabularExploitability, and Exploitability
    Descent.

## Join the community!

If you have any questions about Swift for TensorFlow (or would like to tell the
community about something you did, or research you've published), please join
our mailing list
[`swift@tensorflow.org`](https://groups.google.com/a/tensorflow.org/forum/#!forum/swift).
