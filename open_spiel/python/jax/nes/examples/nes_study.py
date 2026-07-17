import json
import time
from typing import Any
import jax
import chex
import jax.numpy as jnp
import numpy as np
from absl import app, flags, logging

import pyspiel
from open_spiel.python.jax.nes import (
  nes,
  networks,
  games,
  samplers,
  utils,
  equilibria_utils as eu,
)
from open_spiel.python.egt.utils import game_payoffs_array

Objective = samplers.Objective


"""Reproduction experiments for the NES paper.

Experiments follow the reproduction plan:
  1. Training convergence on 8x8
  2. In-distribution accuracy (8x8)
  3. Zero-shot on 4x4, 16x16, 32x32
  4. MWME vs. MRE vs. ε variants
  5. CCE vs. CE duals
  6. 2x2 canonical games
"""

# | # | Test                                 | What It Proves                                   |
# | - | ------------------------------------ | ------------------------------------------------ |
# | 1 | Training convergence on `8×8`        | The unsupervised loss works                      |
# | 2 | In-distribution accuracy (`8×8`)     | NES matches iterative solvers                    |
# | 3 | Zero-shot on `4×4`, `16×16`, `32×32` | Shape-independent generalization                 |
# | 4 | MWME vs. MRE vs. ε variants          | Flexible selection framework                     |
# | 5 | CCE vs. CE duals                     | Both solution concepts are learnable             |
# | 6 | 2×2 canonical games                  | Qualitative correctness & polytope visualization |


FLAGS = flags.FLAGS

# Training
flags.DEFINE_integer("train_steps", 10_000, "Training steps for convergence.")
flags.DEFINE_integer("log_every", 500, "Log/eval frequency.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("eval_batch_size", 64, "Eval batch size.")
flags.DEFINE_integer("seed", 42, "Global seed.")

# Architecture
flags.DEFINE_list("payoff_channel_list", [128, 64, 128], "Payoff channels.")
flags.DEFINE_list("dual_channel_list", [32, 32], "Dual channels.")
flags.DEFINE_integer("dual_channels", 32, "Payoff-to-dual channels.")

# Objectives
flags.DEFINE_float("welfare_coeff", 1.0, "μ from the paper")
flags.DEFINE_float("entropy_coeff", 10.0, "ρ from the paper")
flags.DEFINE_float("epsilon_max", 10.0, "ε⁺ from the paper")
flags.DEFINE_integer("norm", 2, "m norm for L_m normalisation.")

# Output
flags.DEFINE_string(
  "output_path", "/tmp/nes_study_results.json", "Results JSON."
)


def make_network_config() -> dict[str, Any]:
  return {
    "dual_channels": FLAGS.dual_channels,
    "payoff_channel_list": [int(c) for c in FLAGS.payoff_channel_list],
    "dual_channel_list": [int(c) for c in FLAGS.dual_channel_list],
  }


def make_solver(
  game,
  mode: networks.Mode,
  objective: samplers.Objective,
  num_strategies: chex.Shape | None = None,
  max_actions: int | None = None,
  train_steps: int | None = None,
) -> nes.NESolver:
  """Factory to build a solver with consistent hyperparameters."""
  game_kwargs = {}
  if num_strategies is not None:
    game_kwargs["num_strategies"] = num_strategies
  if max_actions is not None:
    game_kwargs["max_actions"] = max_actions

  return nes.NESolver(
    game=game,
    mode=mode,
    network_config=make_network_config(),
    objective=objective,
    welfare_coeff=FLAGS.welfare_coeff,
    entropy_coeff=FLAGS.entropy_coeff,
    epsilon_max=FLAGS.epsilon_max,
    norm=FLAGS.norm,
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    learning_rate=utils.lr_schedule(1e-4),
    weight_decay=1e-7,
    network_train_steps=train_steps or FLAGS.train_steps,
    log_every=FLAGS.log_every,
    seed=FLAGS.seed,
    game_kwargs=game_kwargs if game_kwargs else None,
    allow_checkpointing=False,
  )


def exact_baseline(
  payoffs: chex.Array,
  mode: networks.Mode,
  mu: float = 1.0,
  rho: float = 10.0,
  eps_max: float = 10.0,
) -> dict[str, Any]:
  """Solve the exact ε-MWMRE problem via CVXPY for small games."""
  N, *A = payoffs.shape
  joint_size = utils.compute_joint_action_size(A)
  sigma_hat = np.ones(A) / joint_size
  eps_hat = np.zeros(N)
  mask = np.ones(A, dtype=bool)

  result = eu.mwmre_solver(
    payoffs=payoffs,
    welfare=None,
    strat_pred=sigma_hat,
    epsilon_target=eps_hat,
    joint_mask=mask,
    welfare_coeff=mu,
    entropy_coeff=rho,
    epsilon_max=eps_max,
    mode=mode.name,
    verbose=False,
  )
  return result


def experiment_1_convergence() -> dict[str, Any]:
  """Train on 8x8 L2-invariant games and track loss + gap over time."""
  logging.info("=== Experiment 1: Convergence on 8x8 ===")
  solver = make_solver(
    games.Game.L2_INVARIANT,
    networks.Mode.CCE,
    samplers.Objective.EPS_MWMRE,
    num_strategies=(8, 8),
    train_steps=FLAGS.train_steps,
  )

  logs = solver.solve()

  # Extract metrics
  losses = [float(log["eval_loss"]) for log in logs if "eval_loss" in log]
  gaps = [float(log.get("CCE_gap", log.get("CE_gap", 0.0))) for log in logs]

  return {
    "description": "Training convergence on 8x8 L2-invariant CCE",
    "final_loss": losses[-1] if losses else None,
    "final_gap": gaps[-1] if gaps else None,
    "loss_curve": losses,
    "gap_curve": gaps,
  }


def experiment_2_indistribution() -> dict[str, Any]:
  """Compare NES against exact CVXPY baseline on held-out 8x8 games."""
  logging.info("=== Experiment 2: In-Distribution Accuracy (8x8) ===")
  solver = make_solver(
    games.Game.L2_INVARIANT,
    networks.Mode.CCE,
    samplers.Objective.EPS_MWMRE,
    num_strategies=(8, 8),
    train_steps=FLAGS.train_steps,
  )
  solver.solve()

  # Evaluate on fresh held-out batch
  eval_sampler = samplers.RandomGameSampler(
    game=games.Game.L2_INVARIANT,
    num_strategies=(8, 8),
    game_settings={},
    obj=samplers.Objective.EPS_MWMRE,
    m=FLAGS.norm,
    z_m=FLAGS.epsilon_max,
    seed=FLAGS.seed + 999,
  )
  log = solver.evaluate(0, FLAGS.eval_batch_size, sampler=eval_sampler)

  # Also compute exact baseline on a single game for sanity
  key = jax.random.key(FLAGS.seed)
  payoff_tensor_fn = games.generate_payoffs(games.Game.L2_INVARIANT, {}, (8, 8))
  payoffs, _ = payoff_tensor_fn(key)
  exact = exact_baseline(
    np.array(payoffs),
    networks.Mode.CCE,
    FLAGS.welfare_coeff,
    FLAGS.entropy_coeff,
    FLAGS.epsilon_max,
  )

  return {
    "description": "In-distribution 8x8 accuracy",
    "nes_eval": {k: float(v) for k, v in log.items()},
    "exact_baseline_welfare": float(exact.get("welfare", 0.0)),
    "exact_baseline_status": exact.get("status"),
  }


def experiment_3_zeroshot() -> dict[str, Any]:
  """Train on 8x8, test on 2x2, 4x4, 16x16, 32x32 without retraining."""
  logging.info("=== Experiment 3: Zero-Shot Generalization ===")
  solver = make_solver(
    games.Game.L2_INVARIANT,
    networks.Mode.CCE,
    samplers.Objective.EPS_MWMRE,
    num_strategies=(8, 8),
    max_actions=32,  # pad to largest test size from the start
    train_steps=FLAGS.train_steps,
  )
  solver.solve()

  results = {}
  for size in (2, 4, 8, 16, 32):
    test_sampler = samplers.RandomGameSampler(
      game=games.Game.L2_INVARIANT,
      num_strategies=(size, size),
      game_settings={},
      obj=samplers.Objective.EPS_MWMRE,
      m=FLAGS.norm,
      z_m=FLAGS.epsilon_max,
      seed=FLAGS.seed + size,
    )
    log = solver.evaluate(size, FLAGS.eval_batch_size, sampler=test_sampler)
    results[f"{size}x{size}"] = {k: float(v) for k, v in log.items()}

  return {
    "description": "Zero-shot generalization across board sizes",
    "results": results,
  }


def experiment_4_objectives() -> dict[str, Any]:
  """Train identical architectures with different objective configs."""
  logging.info("=== Experiment 4: Objective Ablations ===")
  configs = [
    ("MWME", samplers.Objective.MWME, 1.0, 10.0),
    ("MRE", samplers.Objective.MRE, 0.0, 10.0),
    ("eps_MWME", samplers.Objective.EPS_MWME, 1.0, 10.0),
    ("eps_MRE", samplers.Objective.EPS_MRE, 0.0, 10.0),
    ("eps_MWMRE", samplers.Objective.EPS_MWMRE, 1.0, 10.0),
  ]

  results = {}
  for name, obj, mu, rho in configs:
    logging.info(f"Running {name}")
    solver = make_solver(
      games.Game.L2_INVARIANT,
      networks.Mode.CCE,
      obj,
      num_strategies=(8, 8),
      train_steps=FLAGS.train_steps,
    )
    # Override coeffs to match objective semantics
    solver._mu = mu
    solver._rho = rho
    _ = solver.solve()

    # Final eval
    eval_sampler = samplers.RandomGameSampler(
      game=games.Game.L2_INVARIANT,
      num_strategies=(8, 8),
      game_settings={},
      obj=obj,
      m=FLAGS.norm,
      z_m=FLAGS.epsilon_max,
      seed=FLAGS.seed + 1,
    )
    log = solver.evaluate(0, FLAGS.eval_batch_size, sampler=eval_sampler)
    results[name] = {k: float(v) for k, v in log.items()}

  return {
    "description": "Ablations over (ε-)MWME and (ε-)MRE variants",
    "results": results,
  }


def experiment_5_modes() -> dict[str, Any]:
  """Compare CCE and CE on identical 8x8 games."""
  logging.info("=== Experiment 5: CCE vs CE ===")
  results = {}
  for mode in (networks.Mode.CCE, networks.Mode.CE):
    name = mode.name
    solver = make_solver(
      games.Game.L2_INVARIANT,
      mode,
      samplers.Objective.EPS_MWMRE,
      num_strategies=(8, 8),
      train_steps=FLAGS.train_steps,
    )
    _ = solver.solve()

    eval_sampler = samplers.RandomGameSampler(
      game=games.Game.L2_INVARIANT,
      num_strategies=(8, 8),
      game_settings={},
      obj=samplers.Objective.EPS_MWMRE,
      m=FLAGS.norm,
      z_m=FLAGS.epsilon_max,
      seed=FLAGS.seed + 2,
    )
    log = solver.evaluate(0, FLAGS.eval_batch_size, sampler=eval_sampler)
    results[name] = {k: float(v) for k, v in log.items()}

  return {
    "description": "CCE vs CE on 8x8 L2-invariant games",
    "results": results,
  }


def experiment_6_canonical() -> dict[str, Any]:
  """Qualitative correctness on classic 2x2 games."""
  logging.info("=== Experiment 6: Canonical 2x2 Games ===")
  canonical = {
    "RPS": "matrix_rps",
    "MP": "matrix_mp",
    "PD": "matrix_pd",
    "BoS": "matrix_bos",
    "Coordination": "matrix_coordination",
  }

  results = {}
  for name, game_name in canonical.items():
    game = pyspiel.load_game(game_name)
    solver = make_solver(
      game,
      networks.Mode.CCE,
      samplers.Objective.EPS_MWMRE,
      train_steps=2_000,  # small games converge fast
    )
    solver.solve()

    # Single-game eval
    log = solver.evaluate(0, 1)
    results[name] = {k: float(v) for k, v in log.items()}

    # Exact baseline for reference
    payoffs = jnp.array(game_payoffs_array(game), dtype=jnp.float32)
    exact = exact_baseline(
      np.array(payoffs),
      networks.Mode.CCE,
      FLAGS.welfare_coeff,
      FLAGS.entropy_coeff,
      FLAGS.epsilon_max,
    )
    results[name]["exact_welfare"] = float(exact.get("welfare", 0.0))
    results[name]["exact_status"] = exact.get("status")

  return {
    "description": "Canonical 2x2 games",
    "results": results,
  }


def main(_):
  start = time.time()
  all_results = {
    "config": {
      "train_steps": FLAGS.train_steps,
      "batch_size": FLAGS.batch_size,
      "welfare_coeff": FLAGS.welfare_coeff,
      "entropy_coeff": FLAGS.entropy_coeff,
      "epsilon_max": FLAGS.epsilon_max,
      "network_config": make_network_config(),
    },
    "exp1_convergence": experiment_1_convergence(),
    "exp2_indistribution": experiment_2_indistribution(),
    "exp3_zeroshot": experiment_3_zeroshot(),
    "exp4_objectives": experiment_4_objectives(),
    "exp5_modes": experiment_5_modes(),
    "exp6_canonical": experiment_6_canonical(),
    "elapsed_seconds": time.time() - start,
  }

  with open(FLAGS.output_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

  logging.info(f"Study complete. Results written to {FLAGS.output_path}")


if __name__ == "__main__":
  app.run(main)
