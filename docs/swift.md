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
