# C++ LibTorch-based AlphaZero

This is a C++ implementation of the AlphaZero algorithm based on LibTorch,
similar to the C++ TF-based AlphaZero.

To build and use this implementation, you must set the optional global variables
`OPEN_SPIEL_BUILD_WITH_LIBTORCH` and `OPEN_SPIEL_BUILD_WITH_LIBNOP` to `ON` when
installing dependencies and building OpenSpiel.

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

## Setting up LibTorch AlphaZero

1.  In [global_variables.sh](../../scripts/global_variables.sh), find the
    `OPEN_SPIEL_BUILD_WITH_LIBNOP` variable and set its value to `"ON"`.
2.  In [global_variables.sh](../../scripts/global_variables.sh), find the
    `OPEN_SPIEL_BUILD_WITH_LIBTORCH` variable and set its value to `"ON"`.
3.  In [global_variables.sh](../../scripts/global_variables.sh), find the
    `OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL` variable and set its value to
    the LibTorch version URL compatible with your OS and hardware (see the
    comments in global_variables.sh for the URLs):
4.  Download libnop and the specified version of LibTorch by running:
    ```bash
    $ ./install.sh
    ```
5.  Build OpenSpiel to compile LibTorch-dependent and libnop-dependent code
    (such as LibTorch AlphaZero).
    ```bash
    $ ./open_spiel/scripts/build_and_run_tests.sh
    ```
**Note:** If you are building from CentOS and/or encounter missing symbol errors (e.g. undefined reference to `memcpy@GLIBC_2.14`, `lgamma@GLIBC_2.23`, etc.), see solution steps described in [this issue]( https://github.com/deepmind/open_spiel/issues/619#issuecomment-854126238).

## Starting LibTorch AlphaZero Training

Starting training from scratch can be done by running
`alpha_zero_torch_example`:
```sh
$ ./build/examples/alpha_zero_torch_example --game=tic_tac_toe --path=/home/me/az_example/
```
Run with the `--help` flag to see a complete list of flags and a brief
description of each.

## Resuming LibTorch AlphaZero Training

Training can be resumed from the most recent checkpoint by providing the path to
the `config.json` (which is created during the initial training run) as a
positional argument:
```sh
$ ./build/examples/alpha_zero_torch_example /home/me/az_example/config.json
```

## Playing a Trained LibTorch AlphaZero

A trained LibTorch AlphaZero can be played by running
`alpha_zero_torch_game_example`:
```sh
$ ./build/examples/alpha_zero_torch_game_example --game=tic_tac_toe --player1=az --player2=mcts --az_path=/home/me/az_example/ --az_checkpoint=-1
```
Run with the `--help` flag to see a complete list of flags and a brief
description of each.
