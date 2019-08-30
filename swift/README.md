# OpenSpiel + Swift for TensorFlow

This directory is a port of OpenSpiel using
[Swift for TensorFlow](https://github.com/tensorflow/swift). Swift OpenSpiel
explores using a single programming language for the entire OpenSpiel
environment, from game implementations to algorithms and deep learning models.

Swift OpenSpiel is intended for serious research use. As the Swift for
TensorFlow platform matures and gains additional capabilities (e.g. distributed
training), significantly more algorithms will become expressible and tractable
to train.

All contributions are welcome: both games and algorithms! If you run into
issues (or would like to share your successes), please reach out to the Swift
for TensorFlow community at
[`swift@tensorflow.org`](https://groups.google.com/a/tensorflow.org/forum/#!forum/swift).


## Building

To use Swift OpenSpiel, download a recent Swift for TensorFlow toolchain following these
[installation instructions](https://github.com/tensorflow/swift/blob/master/Installation.md)
(available for macOS and Ubuntu currently). Swift OpenSpiel currently builds
with the latest stable toolchains.

Using the toolchain, build and test Swift OpenSpiel like a normal SwiftPM package:

```bash
cd swift
swift build # Build the OpenSpiel library.
swift test  # Run tests.
```

## A tour through the code

* `Spiel.swift` contains common game abstractions, including `GameProtocol` and
  `StateProtocol`.
* Games are implemented in separate subdirectories. Perfect information games
  include `TicTacToe` and `Breakthrough`. Imperfect information games include
  `KuhnPoker` and `LeducPoker`.
* Available algorithms include tabular exploitability and exploitability
  descent.

## Join the community!

If you have any questions about Swift for TensorFlow (or would like to share
your work or research with the community), please join our mailing list
[`swift@tensorflow.org`](https://groups.google.com/a/tensorflow.org/forum/#!forum/swift).

