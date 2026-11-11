# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""PPO self-play on tic-tac-toe, scored against a uniform-random player.

What it shows:
  * building a training run from the framework's parts: a policy-gradient
    learner (`learners.pg.PGConfig` with the clipped PPO loss), an MLP from
    the network catalog, and a one-learner self-play run (`league.runner.
    one_slot`), trained by the trainer every learner shares,
    `league.runner.run` (what the command-line trainer `train` calls);
  * evaluating the trained policy head to head against a uniform-random
    player with `eval.match.Evaluator`, which plays seat-swapped pairs of
    games and reports every mean with its standard error.

The learner plays both seats against itself; it never sees the random
player while training. Before training the network plays no better than
chance; after the default 150 updates (under a minute on a CPU) it wins
most games against the random player and loses almost none.

  python -m open_spiel.python.scale.examples.ppo_tic_tac_toe
  python -m open_spiel.python.scale.examples.ppo_tic_tac_toe \
      --num_updates=10 --eval_games=256
  python -m open_spiel.python.scale.examples.ppo_tic_tac_toe \
      --checkpoint_dir=/tmp/ttt

With --checkpoint_dir the run checkpoints there and publishes the trained
policy as an artifact, `<checkpoint_dir>@<num_updates>`, which
`play_checkpoints.py` loads (running the same command again resumes the
finished run instead of training afresh).
"""

import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import uniform

_NUM_UPDATES = flags.DEFINE_integer("num_updates", 150, "Training updates.")
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", 256, "Games played in parallel per update."
)
_NUM_STEPS = flags.DEFINE_integer("num_steps", 32, "Plies per row per update.")
_EVAL_GAMES = flags.DEFINE_integer(
    "eval_games", 2048, "Seat-swapped pairs of games against the random player."
)
_SEED = flags.DEFINE_integer(
    "seed", 0, "Seed of the run and of the evaluation."
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "If set, checkpoint the run here and publish the trained policy "
    "as an artifact (local or gs://).",
)

GAME = "tic_tac_toe"
# The JAX game's observation tensor, ravelled: its own egocentric layout
# [row, column, plane] with planes (the seat's marks, the opponent's marks, a
# plane filled with the seat id), not OpenSpiel's (empty, o, x) planes.
ENCODER = "tic_tac_toe_observation_tensor"
OBS_SHAPE = [27]
NUM_ACTIONS = 9


def make_config(
    num_updates: int, num_envs: int, num_steps: int, seed: int
) -> runner.LeagueRunConfig:
  """Returns a PPO self-play run on tic-tac-toe.

  Args:
    num_updates: The number of updates.
    num_envs: The parallel envs.
    num_steps: The steps per update.
    seed: The seed.

  Returns:
    The run config.
  """
  learner = pg.PGConfig(
      loss="ppo",
      lr=1e-3,
      lr_end=0.0,
      anneal_lr=True,
      num_updates=num_updates,
      update_epochs=4,
      num_minibatches=4,
      gamma=1.0,
      gae_lambda=0.95,
      pg=losses.PGLossConfig(clip_eps=0.2, vf_coef=0.5, clip_value=False),
      ent_coef=schedules.Schedule("constant", 0.01),
      kl_coef=schedules.Schedule("constant", 0.0),
  )
  network = catalog.mlp([128, 128], "relu", "shared")
  return runner.one_slot(
      "example_ppo_tic_tac_toe",
      GAME,
      kinds.PG(learner),
      num_envs=num_envs,
      num_steps=num_steps,
      num_updates=num_updates,
      self_play=True,
      keep_every=num_updates,
      seed=seed,
      policy_kwargs=config_common.network_policy_kwargs(
          GAME, ENCODER, OBS_SHAPE, NUM_ACTIONS, 2, network
      ),
  )


def score_vs_uniform(
    env, policy, params, games: int, seed: int
) -> dict[str, Any]:
  """Returns the win, draw and loss rates of `policy` against random play.

  `policy` plays greedily.

  Args:
    env: The env.
    policy: The policy.
    params: Its params.
    games: The number of games.
    seed: The seed.

  Returns:
    {win, draw, loss, mean, se, win_first, win_second}.
  """
  me = match.Entrant("learner", policy, params, types.ARGMAX)
  rnd = match.Entrant(
      "uniform", uniform.UniformPolicy(GAME, NUM_ACTIONS, 2), {}, types.SAMPLE
  )
  res = match.Evaluator(env).head_to_head(me, rnd, games, jax.random.key(seed))
  # [games, 2]: the learner in seat 0, then seat 1.
  payoff = np.asarray(res.a_payoff)
  return dict(
      win=float(np.mean(payoff > 0)),
      draw=float(np.mean(payoff == 0)),
      loss=float(np.mean(payoff < 0)),
      mean=res.mean.mean,
      se=res.mean.se,
      win_first=float(np.mean(payoff[:, 0] > 0)),
      win_second=float(np.mean(payoff[:, 1] > 0)),
  )


def report(label: str, s: dict[str, Any]) -> None:
  """Prints a score_vs_uniform result.

  Args:
    label: The line's label.
    s: The score_vs_uniform result.
  """
  print(
      f"{label}: win {s['win']:.3f}  draw {s['draw']:.3f}  loss "
      f"{s['loss']:.3f}  (win as first player {s['win_first']:.3f}, as second "
      f"{s['win_second']:.3f}; "
      f"mean payoff {s['mean']:+.3f} +- {1.96 * s['se']:.3f})",
      flush=True,
  )


def main(argv):
  """Trains, evaluates and prints the results.

  Args:
    argv: The command line; no positional arguments.

  Raises:
    app.UsageError: there are positional arguments.
    RuntimeError: the run wrote no artifact at its last update.
  """
  # The framework needs 64-bit JAX; on before anything is built.
  x64.enable_x64()
  if len(argv) > 1:
    raise app.UsageError(f"unexpected arguments {argv[1:]}")
  cfg = make_config(
      _NUM_UPDATES.value, _NUM_ENVS.value, _NUM_STEPS.value, _SEED.value
  )
  env = env_registry.make(GAME)
  t0 = time.time()
  res = runner.run(cfg, checkpoint_dir=_CHECKPOINT_DIR.value)
  train_s = time.time() - t0
  policy, params = res.learner.policy(), res.state.params
  untrained = policy.init_params(jax.random.key(_SEED.value))
  report(
      "untrained network vs uniform",
      score_vs_uniform(env, policy, untrained, _EVAL_GAMES.value, _SEED.value),
  )
  last = res.metrics[-1]
  logging.info("last update metrics: %s", last)
  print(f"trained {len(res.metrics)} updates in {train_s:.0f} s", flush=True)
  report(
      "trained network vs uniform",
      score_vs_uniform(env, policy, params, _EVAL_GAMES.value, _SEED.value + 1),
  )
  if _CHECKPOINT_DIR.value:
    # A learner's artifact step is its update count.
    if not artifact.exists(_CHECKPOINT_DIR.value, _NUM_UPDATES.value):
      raise RuntimeError(
          f"no artifact {_CHECKPOINT_DIR.value}@{_NUM_UPDATES.value}"
      )
    print(f"artifact: {_CHECKPOINT_DIR.value}@{_NUM_UPDATES.value}", flush=True)


if __name__ == "__main__":
  app.run(main)
