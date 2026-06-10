from absl import app
from absl import flags
from absl import logging


from open_spiel.python.jax.nes import nes
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import games
from open_spiel.python.jax.nes import utils

import pyspiel

logger = logging
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
  "iterations", int(6e4), "Number of training episodes."
)
flags.DEFINE_integer(
  "log_every", 5000, "Episode frequency at which the NESolver is evaluated."
)

# DQN model hyper-parameters
flags.DEFINE_list(
  "payoff_channel_list",
  [32, 32, 32, 32, 32],
  "Number of hidden units in the payoff-to-payoff MLP",
)
flags.DEFINE_list(
  "dual_channel_list",
  [32, 32],
  "Number of hidden units in the dual-to-dual MLP",
)
flags.DEFINE_integer(
  "dual_channels", 64, "Number of hidden units in the payoff-to-dual layer"
)

flags.DEFINE_multi_enum(
    "random_games",
    [],  # default list
    [m.name for m in games.Game],
    "Random game types to sample from. Can specify multiple times.",
)

flags.DEFINE_multi_enum(
    "openspiel_games",
    [],  # default list
    list(pyspiel.registered_names()),
    "OpenSpiel games to sample from. Can specify multiple times.",
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

flags.DEFINE_integer(
  "max_actions", None, "Number of maximum actions for all games."
)

flags.DEFINE_float("max_grad_norm", 1e-3, "Max allowed gradient norm.")
flags.DEFINE_float("decay", 1e-7, "Weight decay of the optimiser.")
flags.DEFINE_float(
  "learning_rate", 4e-4, "Learning rate of the solver's updates."
)

flags.DEFINE_float(
  "welfare_coeff", 2.0, "Maximum Welfare coefficient of the loss function."
)
flags.DEFINE_float(
  "entropy_coeff", 10.0, "Minimum Relative Entropy coefficient of the loss function."
)
flags.DEFINE_float(
  "epsilon_max", None, "Epsilon plus coefficient of the loss function."
)

flags.DEFINE_integer("norm", 2, "Norm of the payoff tensor.")
flags.DEFINE_integer("seed", 42, "A random seed.")

flags.DEFINE_integer(
  "batch_size", 64, "Number of transitions to sample at each learning step."
)
flags.DEFINE_bool("use_checkpoints", False, "Save/load neural network weights.")


def main(_) -> None:

  network_config = dict(
    dual_channels=FLAGS.dual_channels,
    payoff_channel_list=FLAGS.payoff_channel_list,
    dual_channel_list=FLAGS.dual_channel_list,
  )

  random_games = [games.Game[game] for game in FLAGS.random_games]
  openspiel_names = FLAGS.openspiel_games  
  assert len(openspiel_names) + len(random_games) > 0, "Any games should be specified"

  solver = nes.NESolver(
    random_games + openspiel_names,
    networks.Mode[FLAGS.mode],
    network_config,
    entropy_coeff=FLAGS.entropy_coeff,
    welfare_coeff=FLAGS.welfare_coeff,
    epsilon_max=FLAGS.epsilon_max,
    norm=FLAGS.norm,  
    batch_size=FLAGS.batch_size,
    learning_rate=utils.lr_schedule(FLAGS.learning_rate),
    weight_decay=FLAGS.decay,
    network_train_steps=FLAGS.iterations,
    gradient_clipping=FLAGS.max_grad_norm,
    log_every=FLAGS.log_every,
    seed=FLAGS.seed,
    game_kwargs={
      "num_strategies": tuple(int(_) for _ in FLAGS.num_strategies),
      "max_actions": FLAGS.max_actions
    },
    allow_checkpointing=FLAGS.use_checkpoints,
  )
  solver.solve()

if __name__ == "__main__":
  app.run(main)
