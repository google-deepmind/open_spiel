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
  ppo_magnet_config   the PGConfig of the "ppo_magnet" loss (PPO-clip, a
                      behaviour KL and an annealed magnet KL) with fixed-size
                      SGD minibatches
  ppo_magnet_self_play  one shared network on every seat of an N-player game,
                      trained with a ppo_magnet_config learner
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
from open_spiel.python.scale.learners import ppo_magnet
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


#: ppo_magnet_config's defaults: one SGD minibatch holds at most this many
#: plies; a forward call runs on at most FORWARD_CHUNK rows (execution only).
PPO_MAGNET_MINIBATCH = 4096
PPO_MAGNET_FORWARD_CHUNK = 1024


def ppo_magnet_config(
    *,
    num_updates: int,
    sgd_steps: int,
    value_scale: float,
    minibatch_size: int = PPO_MAGNET_MINIBATCH,
    forward_chunk_rows: int = PPO_MAGNET_FORWARD_CHUNK,
    rescale_step: int = 0,
    rescale_time_scale: float = 1.0,
    schedule_step_offset: int = 0,
    value_rows: str = "",
    field_players: int = 0,
    max_decision_share: Optional[float] = None,
    loss_config: Optional[ppo_magnet.PPOMagnetConfig] = None,
) -> pg.PGConfig:
  """The PGConfig of the ppo_magnet loss (learners.ppo_magnet).

  loss       PPO-clip 0.2 on raw advantages + 0.1 KL(pi || behaviour) +
             temperature x KL(pi || uniform over legal) + 1.0 value MSE, no
             entropy bonus (PPOMagnetConfig defaults; loss_config replaces
             them); GAE(0.5) advantages and TD(0.8) returns at gamma 1,
             rewards over value_scale.
  optimizer  one epoch, clip_by_global_norm 0.267, Adam (b 0.9 / 0.999,
             eps 1e-8), step size clip(0.5 / (n + 1)**1.1, 5e-6, 1e-4).
  SGD        the update's trained plies are shuffled and dealt over
             max(sgd_steps, ceil(plies / minibatch_size)) steps of at most
             minibatch_size plies.
  schedules  n is the update counter (plus schedule_step_offset); with
             rescale_step S > 0 both the step size and the temperature read
             n = t * rescale_time_scale from update S on (a run continued
             at S under a stretched clock, in one run). The step size there
             must already be at its floor (it is, for S = 3000 and a time
             scale of 14): it is the Schedule's step-size anneal from S - 1
             to S down to the floor, held after.
  credit     field_players P > 0: advantages and value targets from each
             seat's reward relative to the field (learners.ppo_magnet; the
             batch must hold every seat of each table, ppo_magnet_self_play).

  Args:
    num_updates: the run length in updates.
    sgd_steps: the fewest SGD steps per update.
    value_scale: the reward divisor of the value targets.
    minibatch_size: the most plies per SGD step.
    forward_chunk_rows: the most rows per learner forward (execution only).
    rescale_step: 0, or the update the clock stretch starts at.
    rescale_time_scale: the stretch.
    schedule_step_offset: added to the update counter of every schedule.
    value_rows: "" or the "module:attr" of the value-only plies.
    field_players: 0, or the seats of the field credit (PPOMagnetConfig).
    max_decision_share: PGConfig.max_decision_share.
    loss_config: the PPOMagnetConfig; None is its defaults with the rescale
      and field fields set from the arguments.

  Raises:
    ValueError: a loss config whose rescale or field credit disagrees, or a
      rescale whose step size is not at its floor from rescale_step on.

  Returns:
    The pg.PGConfig.
  """
  if loss_config is None:
    loss_config = ppo_magnet.PPOMagnetConfig(
        rescale_step=int(rescale_step),
        rescale_time_scale=float(rescale_time_scale),
        field_players=int(field_players),
    )
  if not isinstance(loss_config, ppo_magnet.PPOMagnetConfig):
    raise TypeError(
        f"loss_config must be a PPOMagnetConfig, got {type(loss_config)}"
    )
  got = (
      loss_config.rescale_step,
      float(loss_config.rescale_time_scale),
      loss_config.field_players,
  )
  want = (int(rescale_step), float(rescale_time_scale), int(field_players))
  if got != want:
    raise ValueError(
        "the loss config's (rescale_step, rescale_time_scale, "
        f"field_players) {got} != the recipe's {want}"
    )
  lr = _ppo_magnet_lr(int(rescale_step), float(rescale_time_scale))
  return pg.PGConfig(
      loss=ppo_magnet.LOSS,
      lr=1e-4,
      lr_end=1e-4,
      anneal_lr=False,
      lr_sched=lr,
      num_updates=int(num_updates),
      max_grad_norm=0.267,
      adam_eps=1e-8,
      use_adamw=False,
      weight_decay=0.0,
      update_epochs=1,
      num_minibatches=1,
      shuffle="flat",
      gamma=1.0,
      gae_lambda=0.5,
      ent_coef=schedules.constant(0.0),
      kl_coef=schedules.constant(0.0),
      loss_config=loss_config,
      schedule_step_offset=int(schedule_step_offset),
      value_scale=float(value_scale),
      value_rows=value_rows,
      minibatch_size=int(minibatch_size),
      sgd_steps=int(sgd_steps),
      forward_chunk_rows=int(forward_chunk_rows),
      max_decision_share=max_decision_share,
  )


def _ppo_magnet_lr(
    rescale_step: int, rescale_time_scale: float
) -> schedules.Schedule:
  """The step size clip(0.5 / (n + 1)**1.1, 5e-6, 1e-4), n stretched from S.

  Args:
    rescale_step: 0, or S (ppo_magnet_config).
    rescale_time_scale: the stretch.

  Returns:
    The Schedule.

  Raises:
    ValueError: the stretched step size is above its floor at S.
  """
  start, power, floor, ceil = 0.5, 1.1, 5e-6, 1e-4
  if not rescale_step:
    return schedules.Schedule(
        "inverse_power", start, power=power, floor=floor, ceil=ceil
    )
  if start / (rescale_step * rescale_time_scale + 1.0) ** power > floor:
    raise ValueError(
        f"the step size at update {rescale_step} under time scale"
        f" {rescale_time_scale} is above its floor: only a stretch that lands"
        " on the floor is supported"
    )
  before = min(
      ceil, max(floor, start / float(rescale_step) ** power)
  )  # n = S - 1
  return schedules.Schedule(
      "inverse_power",
      start,
      power=power,
      floor=floor,
      ceil=ceil,
      anneal_start=rescale_step - 1,
      anneal_end=rescale_step,
      anneal_final_frac=floor / before,
  )


def ppo_magnet_self_play(
    name: str,
    env: str,
    learner: pg.PGConfig,
    policy_kwargs: Mapping[str, Any],
    *,
    num_envs: int,
    num_steps: int,
    num_updates: int,
    seed: int,
    keep_every: int,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    init: Optional[refs.PolicyRef] = None,
    parents: tuple[str, ...] = (),
    eval: Any = None,  # pylint: disable=redefined-builtin
    metadata: Optional[Mapping[str, str]] = None,
    notes: str = "",
) -> runner.LeagueRunConfig:
  """One shared network on every seat, trained by a ppo_magnet learner.

  Pure self-play: every seat of every row is the current learner, acting
  from its own InfoState and sampled at temperature 1, and every seat is a
  perspective seat (num_players x num_envs batch rows); no pool, snapshot
  or fixed opponent. Warm-started from `init` when given.

  Args:
    name: the run name.
    env: a registered env name.
    learner: a PGConfig of loss "ppo_magnet" (ppo_magnet_config).
    policy_kwargs: the network policy's arch_kwargs
      (config_common.network_policy_kwargs).
    num_envs: Arena rows per update.
    num_steps: plies per update.
    num_updates: the run length in updates.
    seed: the run seed.
    keep_every: updates between kept artifacts.
    env_kwargs: the env's options.
    init: the warm start, or None.
    parents: the warm start's lineage.
    eval: the in-run evals.
    metadata: extra metadata.
    notes: the run notes.

  Returns:
    The runner.LeagueRunConfig.

  Raises:
    ValueError: the learner's loss is not ppo_magnet, or its field credit
      names another seat count than the env's.
  """
  if learner.loss != ppo_magnet.LOSS:
    raise ValueError(
        f"{name}: ppo_magnet_self_play takes a {ppo_magnet.LOSS!r} "
        f"learner, got loss {learner.loss!r}"
    )
  field = (
      learner.loss_config.field_players
      if learner.loss_config is not None
      else 0
  )
  if field and field != int(policy_kwargs["num_players"]):
    raise ValueError(
        f"{name}: field credit over {field} seats, the game has "
        f"{policy_kwargs['num_players']}"
    )
  return runner.one_slot(
      name,
      env,
      kinds.PG(learner, init=init, parents=tuple(parents)),
      num_envs=int(num_envs),
      num_steps=int(num_steps),
      num_updates=int(num_updates),
      policy_kwargs=dict(policy_kwargs),
      learner_rule=types.SAMPLE,
      self_play=True,
      keep_every=int(keep_every),
      seed=int(seed),
      env_kwargs=env_kwargs,
      eval=eval,
      metadata=metadata,
      notes=notes,
  )
