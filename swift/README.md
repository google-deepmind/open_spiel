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

BEGIN GOOGLE-INTERNAL

If you're interested in using Swift OpenSpiel within Google, please reach out to
<tf-swift-team@google.com> so we can appropriately support you!

END GOOGLE-INTERNAL

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

BEGIN GOOGLE-INTERNAL

## Building within Google

Within Google, there are two ways to build Swift OpenSpiel. We currently
recommend building with an open source Swift for TensorFlow toolchain on your
local desktop, as this enables fast incremental development. Alternatively, you
can use `blaze`.

### Using a local toolchain (open source style)

To build Swift OpenSpiel using a local toolchain (assuming you are running on
gLinux), follow these steps.

 1. Copy a Swift for TensorFlow toolchain to your corp desktop (or
    CloudTop VM):
    ```
    fileutil cp /placer/prod/home/kokoro-dedicated/build_artifacts/prod/s4tf/toolchain-ubuntu-docker/release-rodete/268/20190523-061232/swift-tensorflow-DEVELOPMENT-rodete.tar.gz .
    ```

    > Note: you can find newer toolchain versions by going to [Fusion](https://fusion.corp.google.com/projectanalysis/summary/KOKORO/prod:s4tf%2Ftoolchain-ubuntu-docker%2Frelease-rodete) and clicking on the "Placer" link for a recent
    green build.

 2. Untar the file:
    ```
    tar -zxf swift-tensorflow-DEVELOPMENT-rodete.tar.gz
    ```

 3. Add `usr/bin` to your `PATH` environment variable.
    ```
    export PATH=$PATH:`pwd`/usr/bin
    ```

 4. Go to the OpenSpiel source code:
    ```
    g4d -f openspiel
    cd third_party/open_spiel/swift
    ```

 5. Build and test!
    ```
    swift build
    swift test
    ```

### Using the Google3 / Forge Toolchain

To build Swift OpenSpiel using the Google3 / Forge toolchain, simply create a
citc client and use `blaze` normally:

```
g4d -f openspiel
blaze build //third_party/open_spiel/swift/...
blaze test //third_party/open_spiel/swift/...
```

END GOOGLE-INTERNAL
