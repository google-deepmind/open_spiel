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
| `gin_mmd_from_scratch` | MMD self-play from scratch on Oklahoma gin rummy (recipe of the pretrained checkpoint below) |
| `hearts_move_transformer_self_play` | four-seat self-play on hearts with the card move transformer and the `ppo_magnet` loss ("Train and play hearts and euchre" below) |
| `euchre_move_transformer_self_play` | the same recipe on euchre |
| `ppo_tic_tac_toe` | PPO against a uniform-random seat on tic-tac-toe |
| `az_connect_four` | AlphaZero on Connect Four |
| `bc_connect_four` | behaviour cloning of a scripted Connect Four player |
| `psro_connect_four` | PSRO with PPO best responses on Connect Four |

Each is a full-size run; the trainer's flags shrink one to a smoke test, for
example gin rummy (about 2.5 minutes on 12 CPU cores, most of it compiling
the training step and the rule-bot evaluation):

    python3 -m open_spiel.python.scale.train --config=mmd_gin_rummy --checkpoint_dir=/tmp/gin \
        --num_updates=2 --num_envs=8 --num_steps=16 --set=eval.deals=4

**Tests.** The tests run under pytest, so install the test dependencies
first, in the same venv:

    source open_spiel/scripts/python_extra_deps.sh python3
    echo "$OPEN_SPIEL_PYTHON_TEST_DEPS" | xargs pip install

Each test file then runs on its own, for example
`python3 -m open_spiel.python.scale.core.tests.rng_test`. Under Bazel, `bazel test
--config=jax //open_spiel/...` runs the fast tests; the long ones (over a
minute each) are tagged `manual` and run when their targets are named.
Under ctest, after the full install above, from the root of the checkout
in the same venv (this package's tests only: `-R python/scale`, about 130
tests; `-R python/` runs every OpenSpiel Python test, which takes much
longer):

    export OPEN_SPIEL_ENABLE_JAX=ON
    export PYTHONPATH=$PWD:$PWD/build/python:$PYTHONPATH
    mkdir -p build && cd build
    cmake -DPython3_EXECUTABLE=$(which python3) ../open_spiel
    make -j$(nproc) pyspiel
    ctest -j$(nproc) --output-on-failure -R python/scale

`$OPEN_SPIEL_PYTHON_TEST_DEPS` (pytest, installed above) is needed: the
tests run under it and `open_spiel/scripts/find_jax.sh` imports it.
`OPEN_SPIEL_ENABLE_JAX=ON`
must be an environment variable: OpenSpiel's CMake takes the setting from
the environment, whose default is `AUTO`, over a `-DOPEN_SPIEL_ENABLE_JAX=ON`
flag, and under `AUTO` a failed `find_jax.sh` import leaves this package's
tests out of ctest with no error (`ctest -N -R python/` lists what is
registered). The long tests are added with `-DOPEN_SPIEL_RUN_MANUAL_TESTS=ON`
on the cmake line (with every other manual test).

## Play the pretrained gin agent

A gin rummy agent trained with this package (the named config
`gin_mmd_from_scratch`, step 30000) is distributed as an artifact folder anyone
can read, with no account and no cloud tools; point `CKPT` at it (or at a local
copy). It plays Oklahoma gin, the game string `gin_rummy(oklahoma=true)`.
Playing it needs the full install above (pyspiel).

    CKPT=https://storage.googleapis.com/openspiel-data/checkpoints/scale/gin_rummy/v1
    # A local copy or gs://openspiel-data/checkpoints/scale/gin_rummy/v1 also works.

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
documented in `docs/gui.md`); "Play in the GUI" below does it in one command,
and the two ways here set it up by hand. The play_checkpoints path above needs only
this package; the GUI path needs a GUI with controller plugins (its
`--controllers` flag and the `open_spiel.gui.controllers` entry point) and
Node.js 22.22.2 or later, with `npm`, to build the GUI's client once
(`./install.sh` does not install Node.js; `docs/gui.md` shows how, with
nvm). From
the root of the OpenSpiel checkout, build the client, then start the server
with this package's "checkpoint" controller, by module name (or with
`--controllers auto` through the entry point OpenSpiel declares for it).
After `pip install .` (the full install above, not an editable install) the
installed server does not find the client in the checkout, so
`--client-dir` points at it:

    (cd open_spiel/gui && npm ci && npm run build)
    python3 -m open_spiel.python.gui.play --game 'gin_rummy(oklahoma=true)' \
        --bot open_spiel.python.scale.serve.remote_bot:make_bot --checkpoint "$CKPT@30000" \
        --state full_state

Once the bot has loaded and compiled (tens of seconds), open the printed
`http://127.0.0.1:8080/game/gin_rummy?...` address to play (see
`../gui/REMOTE_BOTS.md`). Alternatively, to run the in-process `"checkpoint"`
controller plugin directly inside the GUI server (`--client-dir` points at the
built client after a non-editable `pip install .`):

    python3 -m open_spiel.python.gui --controllers open_spiel.python.scale.serve.controller \
        --client-dir "$PWD/open_spiel/gui/dist"

Then open http://127.0.0.1:8080, choose gin rummy with `oklahoma=true`, play
one seat yourself and give the other seat the controller "Checkpoint", its
`checkpoint` setting `$CKPT@30000` written out. The agent sees only its own
seat's view of the game.

**As a Remote bot**, served by its own process to any OpenSpiel GUI (one
with the "Remote bot" controller and its reference bot server,
`open_spiel/python/gui/REMOTE_BOTS.md`), on this machine or another. Start
the bot server with this package's factory; it loads and compiles the agent
before it listens:

    OPEN_SPIEL_SCALE_REMOTE_BOT_CHECKPOINT="$CKPT@30000" \
    python3 -m open_spiel.python.gui.remote_bot_server --port 9001 \
        --bot open_spiel.python.scale.serve.remote_bot:make_bot

Check: it prints `Serving the open_spiel.python.scale.serve.remote_bot:make_bot bot at
http://127.0.0.1:9001`, and http://localhost:9001/info shows
`"state_modes": ["full_state"]` and `"games": ["gin_rummy"]`. Then start the
GUI (`python3 -m open_spiel.python.gui --port 8080`, with
`--move-budget-seconds 60` for a slow machine), open gin rummy with
`oklahoma=true`, and in Setup › Players give the agent's seat the
controller **Remote bot**, URL `http://localhost:9001`, State `full_state`,
Autoplay on. A seat's Config may name another artifact,
`{"checkpoint": "<dir or URL>@<step>"}`, loaded on its first move. The
bot is sent the whole state and plays from its seat's view only, as above;
a move on which the agent's rules and OpenSpiel's differ is still legal,
and its notice ("engine/agent rules differ ...") shows beside the seat's
controller to that seat's viewer and omniscient viewers (GUI contract 9;
the GUI's analysis mode passes a full-state seat's notice on, and a
stricter mode, a match, may drop it). The notice names only the seat's own
actions; anything more stays in the bot server's log.

**Equal deadwood after layoffs is an undercut**, as the agent was trained.
OpenSpiel releases up to 2.0.2 score that hand 0-0; with such a release,
play_checkpoints' replay in OpenSpiel still checks every move of a game that
ends that way and reports it, by its index, as this known scoring difference
rather than a mismatch.

## Play in the GUI

One command plays this package's gin agent in OpenSpiel's game GUI as a
Remote bot (`open_spiel/python/gui/REMOTE_BOTS.md`): it starts the bot
server with the agent's factory and a checkpoint, starts the GUI server, and
prints the address of a gin rummy game whose seat 1 is the agent.

It needs, from the root of the OpenSpiel checkout:

- the full install above (OpenSpiel's C++ toolchain via `./install.sh`, then
  `pip install .`, so `pyspiel` imports),
- this package's JAX dependencies (`$OPEN_SPIEL_PYTHON_JAX_DEPS`, above),
- the GUI client built once, with Node.js 22.22.2 or later and `npm`
  (`docs/gui.md` shows how to install them with nvm),
- `gcsfs` only for a `gs://` checkpoint read without Google credentials.

    (cd open_spiel/gui && npm ci && npm run build)
    python3 -m open_spiel.python.gui.play --game 'gin_rummy(oklahoma=true)' \
        --bot open_spiel.python.scale.serve.remote_bot:make_bot --checkpoint <dir>@<step> \
        --checkpoint-env OPEN_SPIEL_SCALE_REMOTE_BOT_CHECKPOINT \
        --state full_state --client-dir "$PWD/open_spiel/gui/dist"

`--state full_state` sends the bot the whole state: this agent's bot takes
only that (it replays the hand into its env, where the policy sees only its
seat's view), and the launcher picks it by itself when the flag is left
out. `<dir>@<step>` is an artifact folder and its step, as above
(`$CKPT@30000`). The launcher passes it to the bot server in its
environment, so the agent loads before the game opens (you set nothing
else); `--checkpoint-env` names the one variable to set,
`OPEN_SPIEL_SCALE_REMOTE_BOT_CHECKPOINT`, the one this package's bot reads.
`--client-dir` is needed after a non-editable `pip install .`; an editable install (`pip install -e .`) finds
the client without it. Check: once the agent has loaded and compiled (a minute or so)
it prints `Play http://127.0.0.1:8080/game/gin_rummy?session=<id>&viewer=0`;
open it and you are seat 0 against the agent. Ctrl-C stops both servers.

## Train and play hearts and euchre

Hearts and euchre have one recipe: four-seat self-play of one shared policy,
the card move transformer (`networks/card_move_transformer.py`: one token per
card over the card planes `encoders/hearts_card_planes.py` [56, 52] and
`encoders/euchre_card_planes.py` [64, 24], moves as pointer logits) and the
`ppo_magnet` loss (`learners/ppo_magnet.py`: clipped PPO with a KL to the
behaviour policy and a KL to a uniform magnet). A hearts seat's reward is its
score against the other three; a euchre seat's is its partnership's. The
named configs spell it out, one per game, with each knob's value in their
module docs.

**Train your own.** The full run is sized for one GPU: 6,000 updates for
hearts, 3,000 for euchre; on one H100 about 23 s and 17 s per update, so
about 38 h and 14 h, using about 21 GB and 13 GB of device memory. It needs
the CUDA build of JAX at the pinned version, in the same venv; check that
JAX sees the GPU before a long run (on a CPU the trainer runs, very
slowly, and logs `"platform": "cpu"`):

    pip install "jax[cuda12]==$(python3 -c 'import jax; print(jax.__version__)')"
    python3 -c "import jax; print(jax.devices())"    # a CudaDevice

Then (running the same command again resumes):

    python3 -m open_spiel.python.scale.train --config=hearts_move_transformer_self_play \
        --checkpoint_dir=/tmp/hearts_mt
    python3 -m open_spiel.python.scale.train --config=euchre_move_transformer_self_play \
        --checkpoint_dir=/tmp/euchre_mt

A released checkpoint's folder states, in its model card, the exact command
that trained it.

**Pretrained agents.** A hearts agent (step 6000) and a euchre agent
(step 3000) trained with these two configs are distributed as artifact
folders anyone can read, with no account and no cloud tools, like the gin
agent above, each with its model card:

    CKPT=https://storage.googleapis.com/openspiel-data/checkpoints/scale/hearts/v1
    # euchre: https://storage.googleapis.com/openspiel-data/checkpoints/scale/euchre/v1

`$CKPT@6000` names the step (`$CKPT` alone plays the latest); a local copy
or `gs://` URL works too, or pass `/tmp/hearts_mt@<step>` from a run of
your own.

A smoke run on a CPU (the full-size network, two updates; about 90 s on 8
cores):

    python3 -m open_spiel.python.scale.train --config=hearts_move_transformer_self_play \
        --checkpoint_dir=/tmp/hearts_smoke --num_updates=2 --num_envs=8 \
        --num_steps=16 --set=learner.learner.minibatch_size=64 \
        --set=learner.learner.sgd_steps=2 --set=eval.0.deals=4

(euchre: the same with `--config=euchre_move_transformer_self_play`). Each run
writes an artifact every `keep_every` updates and at its last update;
`<dir>@<step>` names one, as for gin above, and `<dir>` alone the latest (the
smoke run's is step 2, `/tmp/hearts_smoke@2`).

**Play against it.** In the terminal, the agent in one seat against uniform
random play in the other three, every deal with the agent in each seat (the
number of games is a multiple of 4, at least 8; `--a=/tmp/hearts_mt@<step>`
plays a run of your own):

    python3 -m open_spiel.python.scale.examples.play_checkpoints --a="$CKPT@6000" \
        --b=uniform --games=64 --out=/tmp/hearts_vs_uniform

It prints each side's mean payoff with its 95% interval. A hearts seat's
payoff is 26 minus the points it took (higher is better; a uniform-random
seat averages about 19.5). A euchre seat's is its partnership's points for
the hand: 1, 2 or 4 when it scores, the same negated when the other side
does. Euchre hands are noisy: use `--games=512` to tell an agent from random
play. A second artifact as `--b` plays the other three seats. Hearts' saved
games
are replayed in OpenSpiel move for move. Euchre is played by this package's
env, which follows OpenSpiel's euchre as corrected on its master branch (a
959-entry information state tensor); a released OpenSpiel's euchre (935
entries) has other rules, so with one installed pass `--noverify_pyspiel`,
and serving euchre in the GUI needs OpenSpiel built from master. To check
which you have (both report the same version number):
`python3 -c "import pyspiel; print(pyspiel.load_game('euchre').information_state_tensor_size())"`
prints 959 for the corrected euchre.

In OpenSpiel's game GUI, against the agent on the other three seats: start
the GUI server with this package's in-process "checkpoint" controller (as
for gin, "Against you" above), from the root of the checkout:

    (cd open_spiel/gui && npm ci && npm run build)
    python3 -m open_spiel.python.gui --controllers open_spiel.python.scale.serve.controller \
        --client-dir "$PWD/open_spiel/gui/dist"

open http://127.0.0.1:8080, choose `hearts` (or `euchre`), keep seat 0 for
yourself and give seats 1, 2 and 3 the controller "Checkpoint", each with
its `checkpoint` setting `$CKPT@6000` written out (or
`/tmp/hearts_mt@<step>`). The Remote bot works too: the one-command
launcher `gui.play` seats its `--bot` on every seat but yours
(`--human-seat`, seat 0 by default), so one bot fills seats 1, 2 and 3:

    python3 -m open_spiel.python.gui.play --game hearts \
        --bot open_spiel.python.scale.serve.remote_bot:make_bot \
        --checkpoint "$CKPT" \
        --state full_state --client-dir "$PWD/open_spiel/gui/dist"

Pass `--checkpoint /tmp/hearts_mt@<step>` for a run of your own, or
`--game euchre` with the euchre URL above.

A checkpoint loads in a fresh process with no import of the
recipe's modules: the encoder, network and loss registries import them when
an artifact or config names them.

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
repository root). Pretrained checkpoint artifacts distributed under
`https://storage.googleapis.com/openspiel-data/checkpoints/scale/` are
licensed under CC-BY 4.0 (each artifact folder carries its own `LICENSE`
and `README.md` model card).
