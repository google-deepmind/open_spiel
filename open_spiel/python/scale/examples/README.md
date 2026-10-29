# Examples

## Getting started

On a CPU laptop, after the install below (the pure-JAX path is enough for steps
1 to 3), from outside the OpenSpiel checkout (for example `cd /tmp`). Each step
prints the line shown under it (numbers vary with the platform; wall times on 12
cores):

1. Train two tiny tic-tac-toe agents (about 30 s and 15 s):

       python -m open_spiel.python.scale.examples.ppo_tic_tac_toe --checkpoint_dir=/tmp/ttt_a
       python -m open_spiel.python.scale.examples.ppo_tic_tac_toe --num_updates=20 --seed=1 --checkpoint_dir=/tmp/ttt_b

   expect `artifact: /tmp/ttt_a@150`, then `artifact: /tmp/ttt_b@20`.

2. Play the two checkpoints against each other, sampling from their
   policies, and save every game (about 7 s):

       python -m open_spiel.python.scale.examples.play_checkpoints --a=/tmp/ttt_a@150 --b=/tmp/ttt_b@20 --rule=sample --games=1024 --out=/tmp/a_vs_b

   expect `a's mean payoff per game: +0.6250 +/- 0.0212 (95% CI +0.5834 ..
   +0.6666); b's: -0.6250` and `wrote 1024 games to
   /tmp/a_vs_b/trajectories.jsonl (game_trajectories format 1) ...`.

3. Evaluate a checkpoint against a uniform-random player (about 7 s):

       python -m open_spiel.python.scale.examples.play_checkpoints --a=/tmp/ttt_a@150 --b=uniform --games=1024 --out=/tmp/a_vs_uniform

   expect `a's mean payoff per game: +0.6221 +/- 0.0174 (95% CI +0.5879 ..
   +0.6563); b's: -0.6221`.

4. Optionally replay a saved game in OpenSpiel (each step above already
   replays all of them and prints `replayed 1024 saved games in pyspiel:
   every move legal; 1024 with the same returns`):

       python -c "import json, pyspiel; r = json.loads(open('/tmp/a_vs_b/trajectories.jsonl').readline()); s = pyspiel.load_game(r['game_string']).new_initial_state(); [s.apply_action(a) for a in r['history']]; print(r['seats'], r['history'], s.returns())"

   expect a line like `['a', 'b'] [4, 1, 5, 0, 3] [1.0, -1.0]`.

Each line of `trajectories.jsonl` is one game: `history` holds OpenSpiel
action ids (chance outcomes included) that replay from
`pyspiel.load_game(game_string).new_initial_state()`, `seats` says which
side sat in each seat and `returns` is the result; `summary.json` holds the
entrants and the head-to-head. The docstring of `play_checkpoints.py`
documents every field. A side can also be `<dir>` (its latest step);
without `--rule` each artifact plays greedily, so two greedy tic-tac-toe
agents play only two distinct games.

## Overview

The starting point: install, run the quickstart, then the other examples.
Every script runs on a laptop CPU and uses only the framework's public
modules. Each prints what it measured; each takes absl flags, so `--help`
lists them and the tiny settings below make a smoke test.

| Script | What it shows |
|---|---|
| `quickstart.py` | **Start here.** The named config `ppo_tic_tac_toe` trained through the command-line trainer's entry point, with a checkpoint you can resume, then its score against a uniform-random player. |
| `jax_games_vs_openspiel.py` | Load a JAX game by its OpenSpiel name, step it, check it move for move against OpenSpiel (`pyspiel`), and play thousands of games at once with `vmap` inside one compiled `while_loop`. |
| `ppo_tic_tac_toe.py` | PPO self-play on tic-tac-toe built from the framework's parts, then a win rate against a uniform-random player. |
| `mmd_kuhn_poker.py` | Magnetic mirror descent on Kuhn poker with the exact exploitability printed as it trains (a custom metric sink), against the exact entropy-regularised equilibrium it converges to. |
| `alphazero_tic_tac_toe.py` | AlphaZero-style MCTS self-play training, then the raw network and the search over it against a random player. |
| `play_checkpoints.py` | Two checkpoints (or a checkpoint and a uniform-random player) played against each other in seat-swapped pairs: the head-to-head with its standard error, and every game saved with OpenSpiel action ids and replayed in `pyspiel`. |

Everything here runs locally on a CPU, with no cloud account.

## Install

**Pure-JAX path, no C++ build.** `quickstart.py`, `ppo_tic_tac_toe.py`,
`mmd_kuhn_poker.py`, `alphazero_tic_tac_toe.py` and `play_checkpoints.py`
run with only the pinned JAX dependencies and the checkout on `PYTHONPATH`,
from the root of the OpenSpiel checkout:

    python3 -m venv ~/venv && source ~/venv/bin/activate
    pip install --upgrade pip setuptools
    source open_spiel/scripts/python_extra_deps.sh python3
    echo "$OPEN_SPIEL_PYTHON_JAX_DEPS" | xargs pip install
    export PYTHONPATH=$PWD:$PYTHONPATH

Without `pyspiel`, `play_checkpoints.py` plays, scores and saves every game
but skips the replay in OpenSpiel (and says so), and
`jax_games_vs_openspiel.py` needs `pyspiel` to compare against. For those,
and for Getting started step 4, use the full install.

**Full install.** OpenSpiel from source with its pinned JAX dependencies,
from the root of the OpenSpiel checkout (Python 3.11 to 3.13):

    ./install.sh
    python3 -m venv ~/venv && source ~/venv/bin/activate
    pip install --upgrade pip setuptools
    pip install .                      # builds OpenSpiel: about 5 minutes on 16 cores
    source open_spiel/scripts/python_extra_deps.sh python3
    echo "$OPEN_SPIEL_PYTHON_JAX_DEPS" | xargs pip install

Run the examples from outside the checkout (`cd /tmp`), so Python imports the
installed package (or the checkout on `PYTHONPATH`). The framework needs 64-bit
JAX (`jax_enable_x64`): each example calls
`open_spiel.python.scale.enable_x64()` first thing in `main`, and so must your
own code. Do not set `JAX_ENABLE_X64=0`.

## Run

    python -m open_spiel.python.scale.examples.quickstart
    python -m open_spiel.python.scale.examples.jax_games_vs_openspiel
    python -m open_spiel.python.scale.examples.ppo_tic_tac_toe
    python -m open_spiel.python.scale.examples.mmd_kuhn_poker
    python -m open_spiel.python.scale.examples.alphazero_tic_tac_toe
    python -m open_spiel.python.scale.examples.play_checkpoints --a=/tmp/ttt_a@150 --b=uniform --games=1024 --out=/tmp/a_vs_uniform  # after Getting started step 1

Each example (and the trainer, `open_spiel/python/scale:train`) is also a
Bazel target, run with the JAX dependency group, for example:

    bazel run --config=jax //open_spiel/python/scale/examples:ppo_tic_tac_toe -- --num_updates=20

and the test, which runs each example with tiny flags in its own process:

    python -m open_spiel.python.scale.examples.examples_test

(it is an `absltest`). It takes several minutes (each example compiles and
trains for real), so in OpenSpiel's Bazel and ctest runs it is a long test,
tagged `manual`: run only when named, or under ctest with
`OPEN_SPIEL_RUN_MANUAL_TESTS=ON`.

The outputs below came from a fresh environment with the pinned
dependencies. Wall times are whole-process times (imports and compilation
included), measured on 16 cores of a shared x86 CPU machine; a laptop takes
a few times longer. Numbers vary with the seed, the platform and the
versions.

### quickstart (about 20 s on 16 cores)

Defaults: 20 updates of 512 games x 32 plies, 512 seat-swapped pairs of
evaluation games; the checkpoint and metrics go to `/tmp/quickstart`
(`--out_dir` picks another directory), and running the same command again
resumes from them. It does not need `pyspiel`. On a fresh 16-core VM following
the install above, it took 16 s.

    training PPO on tic-tac-toe against a random opponent (20 updates of 512 x 32 plies) into /tmp/quickstart
    vs uniform random, 512 seat-swapped deal pairs: mean payoff +0.660 +/- 0.020 (points share 83.0%); one short run, an indication rather than a benchmark
    done in 16 s; checkpoint and metrics in /tmp/quickstart

### jax_games_vs_openspiel (about 15 s)

Defaults: tic-tac-toe, Connect Four, Kuhn and Leduc poker, 50 random
playouts each, 4096 games in the batched rollout.

    tic_tac_toe: 50 random playouts agree with OpenSpiel at all 430 states (2.1 s)
    tic_tac_toe: 4096 random games in one jitted vmap loop: 0.023 s (compile 1.1 s); mean returns [+0.313, -0.313]
    connect_four: 50 random playouts agree with OpenSpiel at all 1111 states (5.4 s)
    ...
    leduc_poker: 50 random playouts agree with OpenSpiel at all 390 states (1.3 s)
    leduc_poker: 4096 random games in one jitted vmap loop: 0.040 s (compile 1.2 s); mean returns [-0.011, +0.011]

A JAX game's tensors may use their own layout, so observation and
information-state tensors are compared through the game's converter from a
pyspiel state (`observation_from_os`, `information_state_from_os`).

With the pinned dependencies, these games agree with OpenSpiel at every state of
the check (`--games=...`, 20 playouts each): `tic_tac_toe`, `connect_four`,
`kuhn_poker`, `leduc_poker`, `othello`, `breakthrough`, `gomoku`,
`phantom_ttt`, `dark_hex`, `bargaining` and `lines_of_action`. The JAX games
package registers more games than these, and not every one matches its
OpenSpiel namesake (some differ in action count or ending, and a few have no
OpenSpiel game of that name); this script is the tool for checking any
other: it stops at the first difference and names the game, playout, step
and quantity. Random games of `lines_of_action` run to hundreds of moves,
so give it a smaller batch (`--batch=256` took about a minute here; the
default 4096 takes many minutes).

### ppo_tic_tac_toe (about 40 s)

Defaults: 150 updates of 256 games x 32 plies, a 128x128 MLP, 2048
seat-swapped pairs of evaluation games.

    untrained network vs uniform: win 0.339  draw 0.178  loss 0.483  (win as first player 0.536, as second 0.143; mean payoff -0.144 +- 0.025)
    trained 150 updates in 27 s
    trained network vs uniform: win 0.680  draw 0.187  loss 0.133  (win as first player 0.944, as second 0.417; mean payoff +0.548 +- 0.018)

### mmd_kuhn_poker (about 2.5 minutes)

Defaults: 100 updates of 256 games x 16 plies, Adam at 1e-3 annealed to 0,
entropy temperature 0.2 on unnormalised advantages (so the temperature is in
payoff units). The exact exploitability is computed after every update,
which is most of the time. The uniform policy's exploitability is
11/24 = 0.4583. Wall times here are on 12 cores of the same machine under
load.

    uniform policy: exploitability 0.4583
    regularised equilibrium at temperature 0.2: exploitability 0.0436 (where MMD at this temperature converges to)
    update     0  exploitability 0.6495
    update     1  exploitability 0.5415
    update    10  exploitability 0.1538
    update    20  exploitability 0.0841
    update    30  exploitability 0.0520
    update    40  exploitability 0.0638
    update    50  exploitability 0.0480
    update    60  exploitability 0.0461
    ...
    update   100  exploitability 0.0586
    trained 100 updates in 148 s
    final policy: exploitability 0.0586 (NashConv 0.1173, 12 information states)
    exploitability 0.6495 -> 0.0586
    last 50 updates: exploitability mean 0.0515 (min 0.0297, max 0.0757); regularised equilibrium 0.0436
    final policy vs regularised equilibrium: largest probability difference 0.066

Why the curve levels off above zero: MMD with a fixed entropy temperature
converges (last iterate) to the entropy-regularised equilibrium at that
temperature, the agent logit quantal response equilibrium, in which every
information state plays the softmax of its action values divided by the
temperature. Its exploitability is positive and grows with the temperature;
the script computes it exactly (tabular MMD on Kuhn's 12 information
states, no network) and prints it as the reference line. For Kuhn it is
0.0189 at temperature 0.1, 0.0436 at 0.2, 0.0690 at 0.3 and 0.1124 at 0.5.

Seeds 0, 1 and 2 averaged 0.052, 0.056 and 0.069 over their last 50
updates (ranges 0.030-0.076, 0.034-0.093, 0.039-0.086) and ended at 0.059,
0.063 and 0.059, a little above the reference 0.0436: each update's
sampling noise keeps the policy jittering around the equilibrium, and a
jittering policy is on average more exploitable than the equilibrium
itself. Numbers also move with the CPU thread count, which changes the
order of floating-point sums.

Exploitability reaches zero only as the temperature goes to zero, and the
step size must shrink with it. The recipe's own setting, entropy 0.05 on
normalised advantages (a temperature of about 0.04 in payoff units, whose
regularised equilibrium has exploitability about 0.005), made this run
oscillate between 0.04 and 0.26 after update 20; exact tabular MMD at
temperature 0.05 behaves the same way with a large step (swinging between
0.12 and 0.21) and converges to its equilibrium's 0.0072 with a step ten
times smaller. Annealing the temperature linearly from 0.2 to 0.01 over the
default 100 updates reached 0.023-0.030 midway but ended at 0.06-0.11, so
the example keeps a fixed temperature; `--temperature` changes it (the
exact reference needs 0.1 or more).

### alphazero_tic_tac_toe (about 60 s)

Defaults: 20 iterations of 128 self-play games at 32 simulations per move,
512 seat-swapped pairs of evaluation games.

    iteration  20  loss 0.2885  (policy 0.2684, value 0.0401)  average game length 8.80
    trained 20 iterations in 27 s
    untrained network vs uniform: win 0.323  draw 0.185  loss 0.492  (mean payoff -0.169 +- 0.051)
    trained network vs uniform:   win 0.721  draw 0.089  loss 0.190  (mean payoff +0.530 +- 0.041)
    search + trained vs uniform:  win 0.805  draw 0.107  loss 0.088  (mean payoff +0.717 +- 0.034)

## Where to go next

The command-line trainer runs MMD on Kuhn poker from a named config (the
recipe's own settings: entropy 0.05 on normalised advantages, longer runs):
`python -m open_spiel.python.scale.train --config=mmd_kuhn_poker
--checkpoint_dir=/tmp/kuhn` (see the package's README.md for configs,
checkpoints and resuming).

## Run on Google Cloud (optional)

An add-on, never required: the quickstart's config trains on a Compute
Engine VM in your own project in a few minutes. You need a project with the
Compute Engine API on, an existing bucket, and `gcloud`:

    gcloud auth login
    gcloud auth application-default login
    python3 -m open_spiel.python.scale.launch.cloud --project=my-project --bucket=my-bucket \
        --dry_run -- --config=ppo_tic_tac_toe

Drop `--dry_run` to launch: the launcher checks its prerequisites and prints the
fix for anything missing, creates one SPOT VM labelled with the run's id,
trains, syncs checkpoints to the bucket and deletes the VM when the run ends (or
at `--max_run_hours`); if the final checkpoint upload fails, it keeps the VM,
whose disk still holds the checkpoint, and prints how to recover.
`--down=<run_id>` deletes a run's VM; it never deletes anything it did not label
and never creates or deletes a bucket. Settings can live in a file
(`launch/cloud_example.yaml`); the docstring of `launch/cloud.py` lists every
default and the roles it needs.
