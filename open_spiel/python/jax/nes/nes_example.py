from absl import app
from absl import flags
from absl import logging


from open_spiel.python.jax.nes import nes
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import games


FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string(
  "checkpoint_dir", "/tmp/dqn_test", "Directory to save/load the agent models."
)
flags.DEFINE_integer(
  "save_every",
  int(1e4),
  "Episode frequency at which the the solver's weights are saved.",
)
flags.DEFINE_integer(
  "num_train_episodes", int(1e6), "Number of training episodes."
)
flags.DEFINE_integer(
  "eval_every", 1000, "Episode frequency at which the NESolver is evaluated."
)

# DQN model hyper-parameters
flags.DEFINE_list(
  "payoff_channel_list",
  [32, 32, 32, 32],
  "Number of hidden units in the payoff-to-payoff MLP",
)
flags.DEFINE_list(
  "dual_channel_list",
  [32, 32],
  "Number of hidden units in the dual-to-dual MLP",
)
flags.DEFINE_integer(
  "dual_channels", 32, "Number of hidden units in the payoff-to-dual layer"
)
flags.DEFINE_enum(
  "game",
  "L2_INVARIANT",
  [m.name for m in games.Game],
  "What type of game should be used?",
)
flags.DEFINE_enum(
  "mode",
  "CCE",
  [m.name for m in networks.Mode],
  "What equilibrium, CE or CCE do we look for?",
)
flags.DEFINE_list(
  "num_strategies",
  default=None,
  required=True,
  help="A tuple defining number of the players' actions (A1, ..., AN)",
)

flags.DEFINE_integer("iterations", 10, "Number of solver iterations.")
flags.DEFINE_float("max_grad_norm", None, "Max allowed gradient norm.")
flags.DEFINE_float("decay", 1e-7, "Weight decay of the optimiser.")
flags.DEFINE_float(
  "learning_rate", 1e-3, "Learning rate of the solver's updates."
)

flags.DEFINE_float(
  "mu", 1.0, "Maximum Welfare coefficient of the loss function."
)
flags.DEFINE_float(
  "rho", 1.0, "Minimum Relative Entropy coefficient of the loss function."
)
flags.DEFINE_integer("norm", 2, "Norm of the payoff tensor.")
flags.DEFINE_integer("seed", 42, "A random seed.")

flags.DEFINE_integer(
  "batch_size", 32, "Number of transitions to sample at each learning step."
)
flags.DEFINE_bool("use_checkpoints", False, "Save/load neural network weights.")


def main(unused) -> None:
  network_config = dict(
    dual_channels=FLAGS.dual_channels,
    payoff_channel_list=FLAGS.payoff_channel_list,
    dual_channel_list=FLAGS.dual_channel_list,
  )
  solver = nes.NESolver(
    games.Game[FLAGS.game],
    networks.Mode[FLAGS.mode],
    network_config,
    rho=FLAGS.rho,
    mu=FLAGS.mu,
    norm=FLAGS.norm,
    batch_size=FLAGS.batch_size,
    learning_rate=FLAGS.learning_rate,
    weight_decay=FLAGS.decay,
    network_train_steps=FLAGS.iterations,
    gradient_clipping=FLAGS.max_grad_norm,
    seed=FLAGS.seed,
    game_kwargs={"num_strategies": tuple(int(_) for _ in FLAGS.num_strategies)},
    allow_checkpointing=FLAGS.use_checkpoints,
  )
  solver.solve()


if __name__ == "__main__":
  app.run(main)
