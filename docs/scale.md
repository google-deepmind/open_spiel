# scale: reinforcement learning for OpenSpiel games

`open_spiel.python.scale` is a reinforcement-learning framework for OpenSpiel games, built
on JAX: an arena that plays batches of games on the accelerator, learners
(PPO and other policy-gradient methods, MMD, AlphaZero-style search with
`mctx`), leagues and population training, checkpoints, evaluation, and a
launcher that runs training on your own machine or on a Google Cloud VM.

`open_spiel.python.jax.games` holds JAX implementations of OpenSpiel games that run inside `jit`
and `vmap`. They are JAX environments, not registered `pyspiel` games: the
`pyspiel` games in `open_spiel.python.games` are unchanged.

It is separate from the standalone agents in `open_spiel.python.jax` (`dqn`,
`nfsp`, `policy_gradient` and the others), which are built on
`rl_environment` and are unchanged.

## Run it locally

Everything in this section runs on your own machine's CPU, with no cloud
account.

### Install

**Pure-JAX path, no C++ build.** The quickstart, the tic-tac-toe examples,
training from a named config and the tests of the JAX code need only the
pinned JAX dependencies and the OpenSpiel checkout on `PYTHONPATH`. From the
root of the checkout:

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install --upgrade pip setuptools
source open_spiel/scripts/python_extra_deps.sh python3
echo "$OPEN_SPIEL_PYTHON_JAX_DEPS" | xargs pip install
export PYTHONPATH=$PWD:$PYTHONPATH
```

What needs `pyspiel` (OpenSpiel's C++ build, the full install below):
replaying saved games in OpenSpiel (`open_spiel.python.scale.examples.play_checkpoints`
plays and saves games without it and says it skipped the replay),
`open_spiel.python.scale.examples.jax_games_vs_openspiel` and the other comparisons
against OpenSpiel's games, serving (`open_spiel.python.scale.serve`: a `pyspiel` state
in, a move out), and the tests that compare with or replay in OpenSpiel
(reported as skipped, naming `pyspiel`, when it is absent).

**Full install.** Install OpenSpiel from source (see
[Installation](install.md)), then the pinned JAX dependencies, from the
root of the checkout:

```bash
./install.sh
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install --upgrade pip setuptools
pip install .
source open_spiel/scripts/python_extra_deps.sh python3
echo "$OPEN_SPIEL_PYTHON_JAX_DEPS" | xargs pip install
```

### Quickstart (about 20 seconds on 16 cores)

```bash
cd /tmp && python3 -m open_spiel.python.scale.examples.quickstart
```

This trains PPO on tic-tac-toe against a uniformly random opponent on the
CPU, prints the trained policy's score against that opponent (mean payoff
per game and points share), and writes the checkpoint and metrics to
`/tmp/quickstart` (`--out_dir` picks another directory). The score is an indication from one short run, not a
benchmark result. Four more examples, with their outputs and run times,
are in the package's examples folder, `open_spiel/python/scale/examples`.

### Training

```bash
python3 -m open_spiel.python.scale.train --config=ppo_tic_tac_toe \
    --checkpoint_dir=/tmp/ppo_tic_tac_toe
```

With Bazel, the same trainer (and each example) is a target:
`bazel run --config=jax //open_spiel/python/scale:train -- --config=ppo_tic_tac_toe`.

`--config` names a config in `open_spiel.python.scale.configs` or a path to a Python file
whose `get_config()` returns a run config. Running the same command again
resumes from the checkpoint. The named configs are `mmd_kuhn_poker_jax`,
`mmd_kuhn_poker`, `mmd_leduc_poker`, `mmd_gin_rummy` (MMD self-play on gin
rummy, scored against the scripted rule bot), `ppo_tic_tac_toe`,
`az_connect_four`, `bc_connect_four` and `psro_connect_four`, each a
full-size run. The trainer's flags shrink any of them to a smoke test, for
example:

```bash
python3 -m open_spiel.python.scale.train --config=mmd_gin_rummy --checkpoint_dir=/tmp/gin \
    --num_updates=2 --num_envs=8 --num_steps=16 --set=eval.deals=4
```

### Tests

Each test file runs on its own, for example
`python3 -m open_spiel.python.scale.core.tests.rng_test`. Under `ctest`, build OpenSpiel
with `OPEN_SPIEL_ENABLE_JAX=ON`: the fast tests run by default, and the long
ones (over a minute each) are registered with
`-DOPEN_SPIEL_RUN_MANUAL_TESTS=ON`.

## Running on Google Cloud (optional)

An add-on, never required: the same configs train on a Compute Engine VM in
your own project in a few minutes. You need a project with the Compute
Engine API on, an existing bucket, and the `gcloud` CLI:

```bash
gcloud auth login
gcloud auth application-default login
python3 -m open_spiel.python.scale.launch.cloud --project=my-project \
    --bucket=my-bucket --dry_run -- --config=ppo_tic_tac_toe
```

Drop `--dry_run` to launch. The VM is deleted when the run ends; if the final
checkpoint upload fails, the launcher keeps it (the checkpoint is still on
it) and prints how to recover. `--down=<run_id>` deletes a run's VM. The
launcher checks its prerequisites first (the `gcloud` CLI, application
default credentials, the Compute Engine API, the bucket) and prints the
command that fixes each missing one. It only ever deletes VMs it labelled for
its own runs, and it never creates or deletes buckets. New projects have no
GPU quota; request it before using `--accelerator`. Settings can also live in
a YAML file; the package ships an example with placeholders,
`open_spiel/python/scale/launch/cloud_example.yaml`.
