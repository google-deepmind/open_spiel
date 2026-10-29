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

"""Standard algorithms as ready-made run configs.

Each recipe returns a `league.runner.LeagueRunConfig` with one learner slot
under a fixed sampler rule (`league.runner.one_slot`), so a plain run and a
population run share one trainer, one checkpoint format and one resume path.

  cfg = recipes.mmd_self_play(
      "my_kuhn", "kuhn_poker_jax", "kuhn_poker_jax_observation_tensor",
      [7], 2, num_updates=200, num_steps=16, num_envs=256)

  mmd_self_play       magnetic mirror descent (Sokota et al. 2023) in self-play
  ppo_vs_uniform      PPO (Schulman et al. 2017) against a uniform-random seat
  behaviour_cloning   a student imitating a teacher that plays itself
  alphazero           AlphaZero (Silver et al. 2018) search self-play
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import az
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog


def mmd_self_play(
    name: str,
    env: str,
    encoder: str,
    obs_shape,
    num_actions: int,
    num_updates: int,
    *,
    num_steps: int,
    num_envs: int = 1024,
) -> runner.LeagueRunConfig:
  """Magnetic mirror descent in self-play.

  A separate-torso MLP (256x2, relu) trained with the MMD loss: KL 0.2 to a
  magnet refreshed every update, entropy 0.05, ratio clip 0.1, value
  coefficient 0.5, a value-target EMA of 0.925, Adam at a constant 2.5e-4,
  gradient norm 0.5, gamma 0.99, GAE lambda 0.95, 4 epochs of 4 minibatches.
  Every seat is a perspective seat. The recipe is for two-player games: the
  network's policy is built for two seats.

  Args:
    name: the run name.
    env: a registered env name.
    encoder: the encoder name the network reads (its PolicySpec.obs_format).
    obs_shape: the encoder's output shape.
    num_actions: the env's action count.
    num_updates: the run length in updates.
    num_steps: plies per update.
    num_envs: Arena rows per update.

  Returns:
    The runner.LeagueRunConfig.

  Raises:
    ValueError: num_steps < 1.
  """
  if int(num_steps) < 1:
    raise ValueError(f"num_steps must be >= 1, got {num_steps}")
  learner = pg.PGConfig(
      loss="mmd",
      lr=2.5e-4,
      lr_end=2.5e-4,
      anneal_lr=False,
      num_updates=num_updates,
      max_grad_norm=0.5,
      adam_eps=1e-5,
      use_adamw=False,
      update_epochs=4,
      num_minibatches=4,
      shuffle="flat",
      gamma=0.99,
      gae_lambda=0.95,
      pg=losses.PGLossConfig(
          clip_eps=0.1, vf_coef=0.5, clip_value=False, kl_init_to_uniform=False
      ),
      ent_coef=schedules.Schedule("constant", 0.05),
      kl_coef=schedules.Schedule("constant", 0.2),
      reference_update_every=1,
      value_ema=True,
      value_ema_beta=0.925,
  )
  return runner.one_slot(
      name,
      env,
      kinds.PG(learner),
      num_envs=int(num_envs),
      num_steps=int(num_steps),
      num_updates=num_updates,
      policy_kwargs=common.network_policy_kwargs(
          env,
          encoder,
          obs_shape,
          num_actions,
          2,
          catalog.mlp([256, 256], "relu", "separate"),
      ),
      self_play=True,
      keep_every=200,
  )


def uniform_opponent(
    env: str, num_actions: int, num_players: int = 2
) -> refs.FixedOpponent:
  """A fixed opponent that plays uniformly at random over legal actions.

  Args:
    env: a registered env name.
    num_actions: the env's action count.
    num_players: the env's seat count.

  Returns:
    The refs.FixedOpponent.
  """
  return refs.FixedOpponent(
      "uniform",
      refs.PolicyRef(
          arch_name="uniform",
          arch_kwargs=dict(
              game=env,
              num_actions=int(num_actions),
              num_players=int(num_players),
          ),
      ),
      types.SAMPLE,
  )


def ppo_vs_uniform(
    name: str,
    env: str,
    encoder: str,
    obs_shape,
    num_actions: int,
    num_updates: int,
    width: int,
    *,
    num_steps: int = 32,
    num_envs: int = 4096,
) -> runner.LeagueRunConfig:
  """PPO against a uniform-random seat.

  A shared-trunk MLP (one hidden layer of `width`, swish, bf16 compute)
  trained with clipped PPO: clip 0.2, entropy 0.01, value coefficient 0.5, a
  value-target EMA of 0.925, AdamW (weight decay 1e-6) with the learning rate
  annealed linearly from 3e-4 to 0 over the run, gradient norm 0.5, 4 epochs
  of 8 minibatches over num_envs rows x num_steps plies. The learner sits at
  seat row % 2 and the uniform opponent at the other: the recipe is for
  two-player games.

  Args:
    name: the run name.
    env: a registered env name.
    encoder: the encoder name the network reads (its PolicySpec.obs_format).
    obs_shape: the encoder's output shape.
    num_actions: the env's action count.
    num_updates: the run length in updates.
    width: the hidden layer's width.
    num_steps: plies per update.
    num_envs: Arena rows per update.

  Returns:
    The runner.LeagueRunConfig.
  """
  learner = pg.PGConfig(
      loss="ppo",
      lr=3e-4,
      lr_end=0.0,
      anneal_lr=True,
      num_updates=num_updates,
      max_grad_norm=0.5,
      adam_eps=1e-5,
      use_adamw=True,
      weight_decay=1e-6,
      update_epochs=4,
      num_minibatches=8,
      shuffle="flat",
      gamma=0.99,
      gae_lambda=0.95,
      pg=losses.PGLossConfig(clip_eps=0.2, vf_coef=0.5, clip_value=False),
      ent_coef=schedules.Schedule("constant", 0.01),
      kl_coef=schedules.Schedule("constant", 0.0),
      value_ema=True,
      value_ema_beta=0.925,
  )
  return runner.one_slot(
      name,
      env,
      kinds.PG(learner),
      num_envs=int(num_envs),
      num_steps=int(num_steps),
      num_updates=num_updates,
      policy_kwargs=common.network_policy_kwargs(
          env,
          encoder,
          obs_shape,
          num_actions,
          2,
          catalog.mlp([width], "swish", "shared", compute_dtype="bfloat16"),
      ),
      self_play=False,
      opponents=(uniform_opponent(env, num_actions),),
      keep_every=200,
  )


def behaviour_cloning(
    name: str,
    env: str,
    student: Mapping[str, Any],
    teacher: refs.PolicyRef,
    *,
    batch_size: int,
    num_iterations: int,
    lr: float,
    lr_end: float,
    weight_decay: float,
    max_grad_norm: float,
    vf_coef: float,
    loss_type: str,
    random_opening_plies: int,
    max_game_length: int,
    checkpoint_freq: int,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    student_init: Any = None,
    notes: str = "",
) -> runner.LeagueRunConfig:
  """Behaviour cloning of a teacher that plays itself.

  The teacher plays every seat greedily, except that the first
  random_opening_plies plies of a game are uniform at random. The student
  learns the teacher's argmax (loss_type "argmax": one-hot cross-entropy) or
  its full distribution ("distill": soft cross-entropy), plus a value MSE to
  the actor's final return on finished games, with one AdamW step per update
  on the whole batch and a learning rate linear from lr to lr_end over
  num_iterations.

  Each update collects batch_size whole games of at most max_game_length
  decisions (num_envs = batch_size, num_steps = max_game_length, first hand
  only), every seat a perspective seat.

  Args:
    name: the run name.
    env: a registered env name.
    student: arch_kwargs of the student "network" policy
      (common.network_policy_kwargs).
    teacher: the teacher policy.
    batch_size: games per update.
    num_iterations: the run length in updates.
    lr: the initial learning rate.
    lr_end: the final learning rate.
    weight_decay: AdamW's weight decay.
    max_grad_norm: the gradient norm clip.
    vf_coef: the value loss coefficient.
    loss_type: "argmax" or "distill".
    random_opening_plies: uniform-random plies at the start of each game.
    max_game_length: the most decisions a game may take.
    checkpoint_freq: updates between kept artifacts.
    env_kwargs: env options.
    student_init: a policy reference whose params warm-start the student.
    notes: free text recorded with the run.

  Returns:
    The runner.LeagueRunConfig.

  Raises:
    ValueError: an unknown loss_type.
  """
  if loss_type not in ("argmax", "distill"):
    raise ValueError(
        f"loss_type must be 'argmax' or 'distill', got {loss_type!r}"
    )
  learner = bc.BCConfig(
      loss=losses.DistillLossConfig(
          policy="ce_action" if loss_type == "argmax" else "ce_target",
          value="mse",
          value_coef=vf_coef,
      ),
      action_label="target_argmax",
      learning_rate=lr,
      lr_end=lr_end,
      total_steps=num_iterations,
      weight_decay=weight_decay,
      max_grad_norm=max_grad_norm,
  )
  spec = kinds.BC(
      teacher=teacher,
      learner=learner,
      beta=schedules.constant(1.0),
      student_init=student_init,
      opening_plies=random_opening_plies,
      first_hand_only=True,
      gamma=1.0,
      value_target="return",
      return_lookahead=False,
  )
  return runner.one_slot(
      name,
      env,
      spec,
      num_envs=batch_size,
      num_steps=max_game_length,
      num_updates=num_iterations,
      policy_kwargs=dict(student),
      learner_rule=types.ARGMAX,
      self_play=True,
      keep_every=checkpoint_freq,
      env_kwargs=env_kwargs,
      notes=notes,
  )


def alphazero(
    name: str, cfg: Any, keep_every: int = 10
) -> runner.LeagueRunConfig:
  """AlphaZero search self-play from a learners.az.AZConfig.

  Every seat plays the SearchPolicy and is a perspective seat; each update
  plays one whole game per row (num_steps = the env's max game length + 1),
  then trains on the replay. The run length, batch size and seed come from
  cfg; keep_every is the number of updates between kept artifacts (the last
  update always keeps one).

  Args:
    name: the run name.
    cfg: the learners.az.AZConfig.
    keep_every: updates between kept artifacts.

  Returns:
    The runner.LeagueRunConfig.

  Raises:
    TypeError: cfg is not an AZConfig.
  """
  if not isinstance(cfg, az.AZConfig):
    raise TypeError(
        f"{name}: alphazero takes a learners.az.AZConfig, got "
        f"{type(cfg).__name__}"
    )
  env = env_registry.make(cfg.env_name)
  return runner.one_slot(
      name,
      cfg.env_name,
      kinds.AZ(cfg),
      num_envs=cfg.batch_size,
      num_steps=env.max_len + 1,
      num_updates=cfg.num_iterations,
      learner_rule=types.DecisionRule("sample", cfg.temperature),
      self_play=True,
      keep_every=keep_every,
      seed=cfg.seed,
  )
