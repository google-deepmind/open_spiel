# Reinforcement learning for games in JAX

Reinforcement learning for games in JAX: one training loop (every run is a
league), one decide, one checkpoint format, components that plug together by
config, and a supervisor that keeps a run going through preemptions on
whatever machines you give it.

## Run it locally

Everything in this section runs on your own machine's CPU, with no cloud
account.

**Pure-JAX path, no C++ build.** The quickstart, the tic-tac-toe examples,
training from a named config and the tests of the JAX code need only the
pinned JAX dependencies and the OpenSpiel checkout on `PYTHONPATH`. From the
root of the checkout:

    python3 -m venv ~/venv && source ~/venv/bin/activate
    pip install --upgrade pip setuptools
    source open_spiel/scripts/python_extra_deps.sh python3
    echo "$OPEN_SPIEL_PYTHON_JAX_DEPS" | xargs pip install
    export PYTHONPATH=$PWD:$PYTHONPATH

What needs `pyspiel` (OpenSpiel's C++ build, the full install below):
replaying saved games in OpenSpiel (`examples.play_checkpoints` plays and
saves games without it and says it skipped the replay),
`examples.jax_games_vs_openspiel` and the other comparisons against
OpenSpiel's games, serving (`serve`: a pyspiel state in, a move out), and
the tests that compare with or replay in OpenSpiel (reported SKIPPED with
pyspiel named when it is absent).

**Full install.** OpenSpiel from source (see `docs/install.md`), then its
pinned JAX dependencies, from the root of the OpenSpiel checkout:

    ./install.sh
    python3 -m venv ~/venv && source ~/venv/bin/activate
    pip install --upgrade pip setuptools
    pip install .                      # builds OpenSpiel: about 5 minutes on 16 cores
    source open_spiel/scripts/python_extra_deps.sh python3
    echo "$OPEN_SPIEL_PYTHON_JAX_DEPS" | xargs pip install

**Quickstart** (about 20 s on 16 cores; a laptop takes a few times longer).
Run it from outside the checkout:

    cd /tmp && python3 -m open_spiel.python.scale.examples.quickstart

It trains PPO on tic-tac-toe against a uniform random player, then prints
its score and where the checkpoint went (`--out_dir` picks another
directory than `/tmp/quickstart`; the same command again resumes):

    vs uniform random, 512 seat-swapped deal pairs: mean payoff +0.660 +/- 0.020 (points share 83.0%); one short run, an indication rather than a benchmark
    done in 16 s; checkpoint and metrics in /tmp/quickstart

**More examples**, each one command (outputs and times in
examples/README.md): `python3 -m open_spiel.python.scale.examples.jax_games_vs_openspiel`
(JAX games checked move for move against OpenSpiel, about 15 s),
`open_spiel.python.scale.examples.ppo_tic_tac_toe` (about 40 s),
`open_spiel.python.scale.examples.mmd_kuhn_poker` (exact exploitability as it trains, about
2.5 minutes) and `open_spiel.python.scale.examples.alphazero_tic_tac_toe` (about 60 s).

**Train from a named config** (running the same command again resumes):

    python3 -m open_spiel.python.scale.train --config=ppo_tic_tac_toe --checkpoint_dir=/tmp/ppo_ttt

or, with Bazel, `bazel run --config=jax //open_spiel/python/scale:train -- --config=ppo_tic_tac_toe`
(every example in `examples/` is a target too).

The named configs (`configs/`, `python3 -m open_spiel.python.scale.train --config=<name>`):

| Config | What |
|---|---|
| `mmd_kuhn_poker_jax` | MMD self-play on the built-in Kuhn poker; needs no game package |
| `mmd_kuhn_poker` | MMD self-play on Kuhn poker |
| `mmd_leduc_poker` | MMD self-play on Leduc poker |
| `mmd_gin_rummy` | MMD self-play on gin rummy, scored every 100 updates against the scripted rule bot |
| `ppo_tic_tac_toe` | PPO against a uniform-random seat on tic-tac-toe |
| `az_connect_four` | AlphaZero on Connect Four |
| `bc_connect_four` | behaviour cloning of a scripted Connect Four player |
| `psro_connect_four` | PSRO with PPO best responses on Connect Four |

Each is a full-size run; the trainer's flags shrink one to a smoke test, for
example gin rummy (about 2.5 minutes on 12 CPU cores, most of it compiling
the training step and the rule-bot evaluation):

    python3 -m open_spiel.python.scale.train --config=mmd_gin_rummy --checkpoint_dir=/tmp/gin \
        --num_updates=2 --num_envs=8 --num_steps=16 --set=eval.deals=4

**Tests.** Each test file runs on its own, for example
`python3 -m open_spiel.python.scale.core.tests.rng_test`. Under Bazel, `bazel test
--config=jax //open_spiel/...` runs the fast tests; the long ones (over a
minute each) are tagged `manual` and run when their targets are named.
Under ctest, build OpenSpiel with `OPEN_SPIEL_ENABLE_JAX=ON` and run
`ctest -R python/` in the build directory: the long ones are added with
`-DOPEN_SPIEL_RUN_MANUAL_TESTS=ON` (with every other manual test).

## Run on Google Cloud (optional)

An add-on, never required: the same configs train on a Compute Engine VM in
your own project in a few minutes. You need a project with the Compute
Engine API on, an existing bucket, and `gcloud`:

    gcloud auth login
    gcloud auth application-default login
    python3 -m open_spiel.python.scale.launch.cloud --project=my-project --bucket=my-bucket \
        --dry_run -- --config=ppo_tic_tac_toe

Drop `--dry_run` to launch. The launcher checks its prerequisites and prints
the fix for anything missing, creates one SPOT VM labelled with the run's
id, trains, writes checkpoints and metrics to the bucket, and deletes the VM
when the run ends. If the final checkpoint upload fails, it keeps the VM (the
checkpoint is still on it) and prints how to recover. `--down=<run_id>` deletes a run's VM. It never creates or
deletes a bucket and never deletes anything it did not label. Settings can
live in a file (`launch/cloud_example.yaml`).

## 64-bit JAX

The package computes in 64 bits (int64 counters, f64 decisions), so it needs
`jax_enable_x64` on. Importing it changes nothing, so other JAX code in the
same process keeps its own setting. The trainer, the league runner's `run()`,
the launcher and the policy server turn x64 on themselves. Code that uses
the components directly calls `open_spiel.python.scale.enable_x64()` once, before it creates
arrays; a component used without it raises `x64.NotEnabledError`.

## Games

The package has ONE built-in game, `kuhn_poker_jax` (envs/kuhn_jax.py: Kuhn poker
in plain JAX). It needs nothing beyond the JAX dependencies, and the config
`mmd_kuhn_poker_jax` trains it.

Every other game adapter in envs/ (Kuhn and Leduc poker, tic-tac-toe,
Connect Four, gin rummy, Go, Hex, ...) runs on OpenSpiel's pure-JAX games
package, `open_spiel.python.jax.games`, which installs with OpenSpiel. A test
that needs an optional package that is not installed (pyspiel, tensorboardX,
wandb) is reported SKIPPED with the package named (conftest.py).

## Layout

| Folder | What |
|---|---|
| core/ | the contracts (Env, Policy, decide, TrajectoryBatch, Learner), the Arena |
| envs/ | game adapters (envs/registry.py is the one game table); envs/kuhn_jax.py is the built-in game |
| encoders/, networks/, policies/ | InfoState encoders, the network catalog, policies (networks, search, routed, tabular) |
| learners/ | PG (ppo, mmd, registered variants), BC, AlphaZero; losses, advantages, schedules |
| league/ | the league loop: learner kinds, samplers, controllers (PSRO, PBT, ...), artifact store |
| eval/, belief/, serve/, sources/ | evaluators, belief samplers, serving, trajectory sources |
| ckpt/ | artifacts (a manifest plus params), resume checkpoints, provenance |
| launch/ | running training: the Backend interface, the supervisor, the LOCAL and GCP backends, the one-command cloud launcher |
| metrics.py | the metric writer and the MetricSink interface (JSONL, TensorBoard, W&B) |
| plugins.py | how a deployment adds its own sinks, loaders, learner kinds, configs |
| testing/ | contract checkers every component's tests use |
| examples/ | the quickstart and four examples (start at examples/README.md) |

## Running somewhere

A run is one training run with an id; it executes as attempts, and a
preempted attempt is replaced by one that resumes from the checkpoint. The
supervisor keeps that state in the run's own storage, so running the same
command again continues the run (or reports it done; `--retry` continues a
run that failed).

A backend is configured by one YAML file. The local backend runs attempts as
subprocesses of this machine; `local.yaml` holds three fields:

    backend: local
    root: /tmp/runs           # runs live under <root>/<run_id> (relative: to this file)
    python: python3            # the interpreter that runs the trainer

    python -m open_spiel.python.scale.launch.run --backend_config=local.yaml --run_id=kuhn-a \
        -- --config=mmd_kuhn_poker_jax --seed=0

The GCP backend runs each attempt on a Compute Engine VM through gcloud:
project, storage bucket, image, network, service account, labels, the code to
ship, setup lines, and an ordered menu of machines (SPOT, STANDARD or
FLEX_START, with zones, and the platform JAX must see on each). Every field
is required and nothing defaults to any project; `${VAR}` reads the
environment. Two complete, tested examples are in launch/examples/:
`gcp_cpu.yaml` and `gcp_gpu.yaml` (GPU machines on an image with the NVIDIA
driver; its setup also installs the CUDA build of jax, and a trainer that
does not see a GPU on a GPU machine stops before it trains). Both ship the
package folder and the JAX games package under their import names
(`${PACKAGE_ROOT}`, `${PACKAGE_PATH}` and `${GAMES_PACKAGE_PATH}` are built
in) and install the tested dependency pins plus gcsfs, for the gs://
checkpoint dir. Before anything is created, a dry run included, the launcher
refuses a run whose game needs a game package the config does not ship. Set PROJECT, BUCKET and SERVICE_ACCOUNT, then, with
the path of `launch/examples/gcp_cpu.yaml` in your checkout:

    python -m open_spiel.python.scale.launch.run --backend_config=PATH/TO/gcp_cpu.yaml \
        --run_id=kuhn-cpu-1 --dry_run -- --config=mmd_kuhn_poker_jax

`--dry_run` prints every gcloud command and startup script and creates
nothing. Other platforms implement launch/backend.py.

For one run on Google Cloud with no backend file, use the cloud launcher
(above).

## Extending

Metrics go to every registered sink (`metrics.register_sink`, or
`--metric_sinks=tensorboard,wandb`). A deployment registers its own sinks,
learner kinds, PG loss variants, configs, network families, checkpoint
loaders, in-loop evals and ExIt eval instruments from a plugin module named
in `$OPEN_SPIEL_SCALE_PLUGINS` or `--plugins` (plugins.py lists the registries).
Every environment variable the package reads starts with `OPEN_SPIEL_SCALE_`.

## Acknowledgements

This package builds on the JAX ecosystem, in particular JAX, Flax, Optax,
Chex and mctx, and draws on ideas from many open-source JAX reinforcement
learning and games libraries. See Awesome JAX
(https://github.com/n2cholas/awesome-jax) for a broader list of the
community's work.

## Licence

Part of OpenSpiel, under its Apache-2.0 licence (the `LICENSE` file at the
repository root).
