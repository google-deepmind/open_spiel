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
examples/README.md):
`python3 -m open_spiel.python.scale.examples.jax_games_vs_openspiel`
(JAX games checked move for move against OpenSpiel, about 15 s),
`open_spiel.python.scale.examples.ppo_tic_tac_toe` (about 40 s),
`open_spiel.python.scale.examples.mmd_kuhn_poker` (exact exploitability as
it trains, about 2.5 minutes) and
`open_spiel.python.scale.examples.alphazero_tic_tac_toe` (about 60 s).

**Train from a named config** (running the same command again resumes):

    python3 -m open_spiel.python.scale.train --config=ppo_tic_tac_toe --checkpoint_dir=/tmp/ppo_ttt

or, with Bazel (every example in `examples/` is a target too):

    bazel run --config=jax //open_spiel/python/scale:train -- --config=ppo_tic_tac_toe

The named configs (`configs/`; train one with
`python3 -m open_spiel.python.scale.train --config=<name>`):

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
Under ctest, after the full install above, from the root of the checkout
in the same venv:

    source open_spiel/scripts/python_extra_deps.sh python3
    echo "$OPEN_SPIEL_PYTHON_TEST_DEPS" | xargs pip install
    export OPEN_SPIEL_ENABLE_JAX=ON
    export PYTHONPATH=$PWD:$PWD/build/python:$PYTHONPATH
    mkdir -p build && cd build
    cmake -DPython3_EXECUTABLE=$(which python3) ../open_spiel
    make -j$(nproc) pyspiel
    ctest -j$(nproc) --output-on-failure -R python/

`$OPEN_SPIEL_PYTHON_TEST_DEPS` (pytest) is needed: the tests run under it
and `open_spiel/scripts/find_jax.sh` imports it. `OPEN_SPIEL_ENABLE_JAX=ON`
must be an environment variable: OpenSpiel's CMake takes the setting from
the environment, whose default is `AUTO`, over a `-DOPEN_SPIEL_ENABLE_JAX=ON`
flag, and under `AUTO` a failed `find_jax.sh` import leaves this package's
tests out of ctest with no error (`ctest -N -R python/` lists what is
registered). The long tests are added with `-DOPEN_SPIEL_RUN_MANUAL_TESTS=ON`
on the cmake line (with every other manual test).

## Play the pretrained gin agent

A gin rummy agent trained with this package (the named config
`gin_mmd_from_scratch`, step 30000) is distributed as an artifact folder anyone
can read, with no account and no cloud tools; point `CKPT` at a copy. It plays
Oklahoma gin, the game string `gin_rummy(oklahoma=true)`. Playing it needs the
full install above (pyspiel).

    CKPT=/path/to/artifact   # a local directory, gs://<bucket>/<path>, or
                             # https://storage.googleapis.com/<bucket>/<path>

The folder is an artifact in this package's own format (`ckpt.artifact`), one
step:

    $CKPT/30000/manifest.json   what rebuilds the agent: the game and its
                                rules hash, the encoder (gin_card_tokens),
                                the architecture (the card-token transformer
                                with the scripted endgame routed) and the
                                params digest
    $CKPT/30000/params/         the params, an orbax folder

A `https://storage.googleapis.com/<bucket>/<path>` URL is read anonymously
over plain HTTPS: the artifact is copied into `~/.cache` (the environment
variable `OPEN_SPIEL_SCALE_ARTIFACT_CACHE` names another folder) and loaded from
there, its params checked against the digest in its manifest. `$CKPT@30000`
names the step (`$CKPT` alone plays the latest). A `gs://` URL works too: with
Google credentials it is read through etils epath, and without them it is read
anonymously through `gcsfs` (`pip install gcsfs`). A local directory is read
as it is.

**Against a bot**, in the terminal: the agent plays the scripted gin rule bot
on 64 seat-swapped deals, and every game is saved and replayed in OpenSpiel.
The gin env draws the first mover at a chance node that OpenSpiel's
`gin_rummy` does not have, so each saved history is OpenSpiel's: the seat
that moved first is OpenSpiel's player 0 (each record's `seats`), and the
env's own history is kept beside it (`env_history`):

    python3 -m open_spiel.python.scale.examples.play_checkpoints --a="$CKPT@30000" \
        --b=arch:gin_rummy.rule_bot --games=64 --out=/tmp/gin_vs_rule_bot

**Against you**, in OpenSpiel's game GUI (`open_spiel/python/gui`,
documented in `docs/gui.md`). The play_checkpoints path above needs only
this package; the GUI path needs a GUI with controller plugins (its
`--controllers` flag and the `open_spiel.gui.controllers` entry point) and
Node.js 22.22.2 or later, with `npm`, to build the GUI's client once. From
the root of the OpenSpiel checkout, build the client, then start the server
with this package's "checkpoint" controller, by module name (or with
`--controllers auto` through the entry point OpenSpiel declares for it).
After `pip install .` (the full install above, not an editable install) the
installed server does not find the client in the checkout, so
`--client-dir` points at it:

    (cd open_spiel/gui && npm ci && npm run build)
    python3 -m open_spiel.python.gui --controllers open_spiel.python.scale.serve.controller \
        --client-dir "$PWD/open_spiel/gui/dist"

Then open http://127.0.0.1:8080, choose gin rummy with `oklahoma=true`, play
one seat yourself and give the other seat the controller "Checkpoint", its
`checkpoint` setting `$CKPT@30000` written out. The seat shows as preparing
while the agent loads and compiles (tens of seconds); after that it moves at
once. The agent sees only its own seat's view of the game.

**Equal deadwood after layoffs is an undercut**, as the agent was trained.
OpenSpiel releases up to 2.0.2 score that hand 0-0; with such a release,
play_checkpoints' replay in OpenSpiel still checks every move of a game that
ends that way and reports it, by its index, as this known scoring difference
rather than a mismatch.

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
checkpoint is still on it) and prints how to recover. `--down=<run_id>`
deletes a run's VM. It never creates or deletes a bucket and never deletes
anything it did not label. Settings can live in a file
(`launch/cloud_example.yaml`).

## 64-bit JAX

The package computes in 64 bits (int64 counters, f64 decisions), so it needs
`jax_enable_x64` on. Importing it changes nothing, so other JAX code in the
same process keeps its own setting. The trainer, the league runner's `run()`,
the launcher and the policy server turn x64 on themselves. Code that uses
the components directly calls `open_spiel.python.scale.enable_x64()` once,
before it creates arrays; a component used without it raises
`x64.NotEnabledError`.

## Games

The package has ONE built-in game, `kuhn_poker_jax` (envs/kuhn_jax.py: Kuhn
poker in plain JAX). It needs nothing beyond the JAX dependencies, and the
config `mmd_kuhn_poker_jax` trains it.

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
refuses a run whose game needs a game package the config does not ship. Set
PROJECT, BUCKET and SERVICE_ACCOUNT, then, with the path of
`launch/examples/gcp_cpu.yaml` in your checkout:

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
repository root). Published checkpoints (trained weights) are data, not code:
they are released separately under the Creative Commons Attribution 4.0
International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/).
