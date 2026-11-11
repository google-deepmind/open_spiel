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

r"""AlphaZero-style search training on tic-tac-toe, then evaluation.

What it shows:
  * `learners.az.AZConfig` and the `configs.recipes.alphazero` recipe: MCTS
    self-play (mctx under the hood) whose visit counts and game outcomes train
    a policy-and-value network, through the same trainer as every other
    learner (`league.runner.run`);
  * `learners.az.build`, which returns the search policy
    (`policies.search.SearchPolicy`) over the trained network, so the same
    network can be played raw or with search on top;
  * head-to-head evaluation against a uniform-random player
    (`eval.match.Evaluator`).

It prints the win, draw and loss rates against the random player of the
untrained network, the trained network played greedily, and the search over
the trained network. The default 20 iterations take under a minute on a CPU.

  python -m open_spiel.python.scale.examples.alphazero_tic_tac_toe
  python -m open_spiel.python.scale.examples.alphazero_tic_tac_toe \\
      --num_iterations=2 --num_simulations=4
"""

import time

from absl import app
from absl import flags
import jax
import numpy as np

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import az
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import uniform

_NUM_ITERATIONS = flags.DEFINE_integer(
    "num_iterations",
    20,
    "Training iterations (self-play games, then training).",
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 128, "Self-play games per iteration."
)
_NUM_SIMULATIONS = flags.DEFINE_integer(
    "num_simulations", 32, "MCTS simulations per move."
)
_EVAL_GAMES = flags.DEFINE_integer(
    "eval_games", 512, "Seat-swapped pairs of games against the random player."
)
_SEED = flags.DEFINE_integer(
    "seed", 0, "Seed of the run and of the evaluation."
)

GAME = "tic_tac_toe"


def make_az_config(
    num_iterations: int, batch_size: int, num_simulations: int, seed: int
) -> az.AZConfig:
  """Returns the AlphaZero settings for tic-tac-toe.

  Args:
    num_iterations: The training iterations.
    batch_size: The batch size.
    num_simulations: The MCTS simulations per move.
    seed: The seed.

  Returns:
    The AZConfig.
  """
  return az.AZConfig(
      env_name=GAME,
      encoder="tic_tac_toe_observation_tensor",
      obs_shape=[27],
      network=az.jsonable(
          catalog.mlp([128, 128], "relu", "shared", value_tanh=True)
      ),
      sampler="perfect_information",
      seed=seed,
      num_simulations=num_simulations,
      dirichlet_alpha=1.0,
      dirichlet_epsilon=0.25,
      temperature=1.0,
      temperature_drop_move=4,
      temperature_post_drop=1e-3,
      batch_size=batch_size,
      num_iterations=num_iterations,
      num_epochs=4,
      minibatch_size=256,
      replay_buffer_size=4,
      lr=1e-3,
      lr_end=1e-4,
      weight_decay=1e-4,
  )


def score_vs_uniform(env, policy, params, games: int, seed: int) -> str:
  """Returns the win / draw / loss rates of `policy` against random play.

  `policy` plays greedily.

  Args:
    env: The env.
    policy: The policy.
    params: Its params.
    games: The number of games.
    seed: The seed.

  Returns:
    The rates and the mean payoff, as a line of text.
  """
  me = match.Entrant("player", policy, params, types.ARGMAX)
  rnd = match.Entrant(
      "uniform",
      uniform.UniformPolicy(GAME, env.num_actions, 2),
      {},
      types.SAMPLE,
  )
  res = match.Evaluator(env).head_to_head(me, rnd, games, jax.random.key(seed))
  payoff = np.asarray(res.a_payoff)
  return (
      f"win {np.mean(payoff > 0):.3f}  draw {np.mean(payoff == 0):.3f}  "
      f"loss {np.mean(payoff < 0):.3f}  (mean payoff {res.mean.mean:+.3f} +- "
      f"{1.96 * res.mean.se:.3f})"
  )


def main(argv):
  """Trains, evaluates and prints the results.

  Args:
    argv: The command line; no positional arguments.

  Raises:
    app.UsageError: there are positional arguments.
  """
  # The framework needs 64-bit JAX; on before anything is built.
  x64.enable_x64()
  if len(argv) > 1:
    raise app.UsageError(f"unexpected arguments {argv[1:]}")
  cfg = make_az_config(
      _NUM_ITERATIONS.value,
      _BATCH_SIZE.value,
      _NUM_SIMULATIONS.value,
      _SEED.value,
  )
  t0 = time.time()
  res = runner.run(
      recipes.alphazero(
          "example_alphazero_tic_tac_toe", cfg, keep_every=cfg.num_iterations
      )
  )
  train_s = time.time() - t0
  for i, m in enumerate(res.metrics):
    print(
        f"iteration {i + 1:3d}  loss {m['loss_total']:.4f}  (policy "
        f"{m['loss_policy']:.4f}, value {m['loss_value']:.4f})  "
        f"average game length {m['avg_game_length']:.2f}",
        flush=True,
    )
  print(f"trained {len(res.metrics)} iterations in {train_s:.0f} s", flush=True)

  parts = az.build(cfg)
  net, search = parts.base, parts.self_play
  trained = res.state.params
  untrained = net.init_params(jax.random.key(_SEED.value))
  n, s = _EVAL_GAMES.value, _SEED.value
  print(
      "untrained network vs uniform: "
      + score_vs_uniform(parts.env, net, untrained, n, s),
      flush=True,
  )
  print(
      "trained network vs uniform:   "
      + score_vs_uniform(parts.env, net, trained, n, s + 1),
      flush=True,
  )
  print(
      "search + trained vs uniform:  "
      + score_vs_uniform(
          parts.env, search, search.make_params(trained), n, s + 2
      ),
      flush=True,
  )


if __name__ == "__main__":
  app.run(main)
