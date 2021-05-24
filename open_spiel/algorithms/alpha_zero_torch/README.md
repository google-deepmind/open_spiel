# C++ LibTorch-based AlphaZero

This is a C++ implementation of the AlphaZero algorithm based on LibTorch,
similar to the C++ TF-based AlphaZero.

To build and use this implementation, you must set the optional global variable
`OPEN_SPIEL_BUILD_WITH_LIBTORCH` to `ON` when installing dependencies and
building OpenSpiel.

Then, to get started, see `examples/alpha_zero_torch_example.cc`.

Important note: this implementation was a user contribution (see
[this PR](https://github.com/deepmind/open_spiel/pull/319)), and is not
regularly tested nor maintained by the core team. This means that, at any time,
it may not build or work as originally intended due to a change that will not
have been caught by our tests. Hence, if bugs occur, please open an issue to let
us know so we can fix them.

This code was structured in a similar way to the TF-based C++ AlphaZero, using
several of the same components. If you have any questions, feel free to ask the
original author Christian Jans directly by following up on the PR linked above.
The PR also includes some results of experiments run using this implementation
that may be useful.
