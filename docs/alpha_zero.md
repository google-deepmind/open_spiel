# AlphaZero

OpenSpiel includes three implementations of AlphaZero, two based on Tensorflow
(one in Python and one in C++ using Tensorflow C++ API), with a shared model
written in TensorFlow. The other based on C++ Libtorch-base. This document
covers mostly the TF-based implementation and common components. For the
Libtorch-based implementation,
[see here](https://github.com/deepmind/open_spiel/tree/master/open_spiel/algorithms/alpha_zero_torch).

**Disclaimer**: this is not the code that was used for the Go challenge matches
or the AlphaZero paper results. It is a re-implementation for illustrative
purposes, and although it can handle games like Connect Four, it is not designed
to scale to superhuman performance in Go or Chess.

## Background

AlphaZero is an algorithm for training an agent to play perfect information
games from pure self-play. It uses Monte Carlo Tree Search (MCTS) with the prior
and value given by a neural network to generate training data for that neural
network.

Links to relevant articles/papers:

-   [AlphaGo Zero: Starting from scratch](https://deepmind.com/blog/article/alphago-zero-starting-scratch)
    has an open access link to the AlphaGo Zero nature paper that describes the
    model in detail.
-   [AlphaZero: Shedding new light on chess, shogi, and Go](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go)
    has an open access link to the AlphaZero science paper that describes the
    training regime and generalizes to more games.

## Overview:

The Python and C++ implementations are conceptually fairly similar, and have
roughly the same components: [actors](#actors) that generate data through
self-play using [MCTS](#mcts) with an [evaluator](#mcts-evaluator) that uses a
[neural network](#model), a [learner](#learner) that updates the network based
on those games, and [evaluators](#evaluators) playing vs standard MCTS to gauge
progress. Both [write checkpoints](#output) that can be [played](#playing-vs-checkpoints)
independently of the training setup, and logs that can be [analyzed](#analysis)
programmatically.

The Python implementation uses one process per actor/evaluator, doesn't support
batching for inference and does all inference and training on the cpu. The C++
implementation, by contrast, uses threads, a shared cache, supports batched
inference, and can do both inference and training on GPUs. As such the C++
implementation can take advantage of additional hardware and can train
significantly faster.

### Model

The model defined in
[open_spiel/python/algorithms/alpha_zero/model.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/model.py) is used by
both the python and C++ implementations. The C++ version wraps the exported
tensorflow graph in
[open_spiel/algorithms/alpha_zero/vpnet.h](https://github.com/deepmind/open_spiel/blob/master/open_spiel/algorithms/alpha_zero/vpnet.h), and supports both
inference and training.

The model defines three architectures in decreasing complexity:

-   resnet: same as the AlphaGo/AlphaZero paper when set with width 256 and
    depth 20.
-   conv2d: same as the resnet except uses a conv+batchnorm+relu instead of the
    residual blocks.
-   mlp: same as conv2d except uses dense layers instead of conv, and drops
    batch norm.

The model is parameterized by the size of the observations and number of actions
for the game you specify, so can play any 2-player game. The conv2d and resnet
models are restricted to games with a 2d representation (ie a 3d observation
tensor).

The models are all parameterized with a width and depth:

-   The depth is the number of blocks in the torso, where the definition of a
    block varies by model. For a resnet it's a resblock which is two conv2ds,
    batch norms and relus, and an addition. For conv2d it's a conv2d, a batch
    norm and a relu. For mlp it's a dense plus relu.
-   The width is the number of filters for any conv2d and the number of hidden
    units for any dense layer.

The networks all give two outputs: a value and a policy, which are used by the
MCTS evaluator.

### MCTS

Monte Carlo Tree Search (MCTS) is a general search algorithm used to play many
games, but first found success playing Go back in ~2005. It builds a tree
directed by random rollouts, and does usually uses UCT to direct the
exploration/exploitation tradeoff. For our use case we replace random rollouts
with a value network. Instead of a uniform prior we use a policy network.
Instead of UCT we use PUCT.

We have implementations of MCTS in
[C++](https://github.com/deepmind/open_spiel/blob/master/open_spiel/algorithms/mcts.h) and
[python](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/mcts.py).

### MCTS Evaluator

Both MCTS implementations above have a configurable evaluator that returns the
value and prior policy of a given node. For standard MCTS the value is given by
random rollouts, and the prior policy is uniform. For AlphaZero the value and
prior are given by a neural network evaluation. The AlphaZero evaluator takes a
model, so can be used during training or with a trained checkpoint for play with
[open_spiel/python/examples/mcts.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/mcts.py).

### Actors

The main script launches a set of actor processes (Python) or threads (C++). The
actors create two MCTS instances with a shared evaluator and model, and play
self-play games, passing the trajectories to the learner via a queue. The more
actors the faster it can generate training data, assuming you have sufficient
compute to actually run them. Too many actors for your hardware will mean longer
for individual games to finish and therefore your data could be more out of date
with respect to the up to date checkpoint/weights.

### Learner

The learner pulls trajectories from the actors and stores them in a fixed size
FIFO replay buffer. Once the replay buffer has enough new data, it does an
update step sampling from the replay buffer. It then saves a checkpoint and
updates all the actor's models. It also updates a `learner.jsonl` file with some
stats.

### Evaluators

The main script also launches a set of evaluator processes/threads. They
continually play games against a standard MCTS+Solver to give an idea of how
training is progressing. The MCTS opponents can be scaled in strength based on
the number of simulations they are given per move, so more levels means stronger
but slower opponents.

### Output

When running the algorithm a directory must be specified and all output goes
there.

Due to the parallel nature of the algorithm writing logs to stdout/stderr isn't
very useful, so each actor/learner/evaluator writes its own log file to the
configured directory.

Checkpoints are written after every update step, mostly overwriting the latest
one at `checkpoint--1` but every `checkpoint_freq` is saved at
`checkpoint-<step>`.

The config file is written to `config.json`, to make the experiment more
repeatable.

The learner also writes machine readable logs in the
[jsonlines](http://jsonlines.org/) format to `learner.jsonl`, which can be read
with the analysis library.

## Usage:

### Python

The code lives at [open_spiel/python/algorithms/alpha_zero/](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/).

The simplest example trains a tic_tac_toe agent for a set number of training
steps:

```bash
python3 open_spiel/python/examples/tic_tac_toe_alpha_zero.py
```

Alternatively you can train on an arbitrary game with many more options:

```bash
python3 open_spiel/python/examples/alpha_zero.py --game connect_four --nn_model mlp --actors 10
```

### C++

The code lives at [open_spiel/algorithms/alpha_zero/](https://github.com/deepmind/open_spiel/blob/master/open_spiel/algorithms/alpha_zero/)
with an example executable at
[open_spiel/examples/alpha_zero_example.cc](https://github.com/deepmind/open_spiel/blob/master/open_spiel/examples/alpha_zero_example.cc).

Compiling it is now possible with the help of the
[tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc) project. TensorflowCC
allows the usage of the TensorFlow C++ API from outside the Tensorflow source
directory.

For build instructions, please see
[open_spiel/algorithms/alpha_zero/README.md](https://github.com/deepmind/open_spiel/blob/master/open_spiel/algorithms/alpha_zero/README.md).

Although targets are built successfully, there are still some runtime issues.
[OpenSpiel Issue #172](https://github.com/deepmind/open_spiel/issues/172) has
some information that may help figure out how to fix them. Contributions are
welcome.


### Analysis

There's an analysis library at
[open_spiel/python/algorithms/alpha_zero/analysis.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/analysis.py) which
reads the `config.json` and `learner.jsonl` from an experiment (either python or
C++), and graphs losses, value accuracy, evaluation results, actor speed, game
lengths, etc. It should be reasonable to turn this into a colab.

### Playing vs checkpoints

The checkpoints are compatible between python and C++, and can be loaded by the
model. You can try playing against one directly with
[open_spiel/python/examples/mcts.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/mcts.py):

```bash
python3 open_spiel/python/examples/mcts.py --game=tic_tac_toe --player1=human --player2=az --az_path <path to your checkpoint directory>
```
