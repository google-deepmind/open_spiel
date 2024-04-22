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
"""Runs mean field proximal policy optimaztion agents."""

# pylint: disable=consider-using-from-import

import logging
import os
import time

from absl import flags
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from open_spiel.python import policy as policy_std
from open_spiel.python import rl_environment
from open_spiel.python.mfg import utils
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import Agent as mfg_ppo_agent
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import calculate_advantage
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import calculate_explotability
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import learn
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import Policy as mfg_ppo_policy
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import rollout
from open_spiel.python.mfg.games import factory
from open_spiel.python.utils import app


FLAGS = flags.FLAGS


flags.DEFINE_integer("seed", default=0, help="Set a random seed.")
flags.DEFINE_string(
    "exp_name", default="mf-ppo", help="Set the name of this experiment"
)
flags.DEFINE_string(
    "game_setting",
    default="crowd_modelling_2d_four_rooms",
    help=(
        "Set the game to benchmark options:(crowd_modelling_2d_four_rooms)     "
        "                and (crowd_modelling_2d_maze)"
    ),
)
flags.DEFINE_float("lr", default=1e-3, help="Learning rate of the optimizer")
flags.DEFINE_integer(
    "num_episodes",
    default=5,
    help=(
        "set the number of episodes                      of to collect per"
        " rollout"
    ),
)
flags.DEFINE_integer(
    "update_episodes",
    default=20,
    help="set the number of episodes                      of the inner loop",
)
flags.DEFINE_integer(
    "update_iterations",
    default=100,
    help=(
        "Set the number of global                      update steps of the"
        " outer loop"
    ),
)
flags.DEFINE_string(
    "optimizer", default="Adam", help="Set the optimizer (Adam) or (SGD)"
)
flags.DEFINE_boolean(
    "cuda", default=False, help="Use Gpu to run the experiment"
)

# MFPPO parameters
flags.DEFINE_float("gamma", default=0.9, help="set discount factor gamma")
flags.DEFINE_integer(
    "num_minibatches", default=5, help="the number of mini-batches"
)
flags.DEFINE_integer(
    "update_epochs", default=5, help="the K epochs to update the policy"
)
flags.DEFINE_float(
    "clip_coef", default=0.2, help="the surrogate clipping coefficient"
)
flags.DEFINE_float("ent_coef", default=0.01, help="coefficient of the entropy")
flags.DEFINE_float(
    "max_grad_norm",
    default=5,
    help="the maximum norm for the gradient clipping",
)
flags.DEFINE_float(
    "alpha",
    default=0.5,
    help=(
        "Set alpha to controll the iteration                    and epsiode"
        " policy updates"
    ),
)
flags.DEFINE_float(
    "eps_eps", default=0.2, help="eps to update the episode learned policy"
)
flags.DEFINE_float(
    "itr_eps", default=0.05, help="eps to update the episode learned policy"
)


def set_seed(seed):
  """Set the random seed for reproducibility."""
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")


def main(unused_argv):
  """Main function to run the experiment."""

  # Set the random seed for reproducibility
  set_seed(FLAGS.seed)

  # Set the device (in our experiments CPU vs GPU does not improve time at all)
  # we recommend CPU
  device = torch.device(
      "cuda" if torch.cuda.is_available() and FLAGS.cuda else "cpu"
  )

  # Set the name of the experiment's folder
  fname = "./mfppo_experiments/"

  # Log the experiments
  run_name = (
      f"{FLAGS.exp_name}_{FLAGS.game_setting}_{FLAGS.optimizer}_num_update_epochs_"
      "     "
      f" {FLAGS.update_epochs}_num_episodes_per_rollout_{FLAGS.num_episodes}_number_of_mini_batches_"
      "     "
      f" {FLAGS.num_minibatches}_{time.asctime(time.localtime(time.time()))}"
  )
  log_name = os.path.join(fname, run_name)
  tb_writer = SummaryWriter(log_name)
  logging.basicConfig(
      filename=log_name + "_log.txt",
      filemode="a",
      level=logging.DEBUG,
      force=True,
  )

  # Console handler
  console = logging.StreamHandler()
  console.setLevel(logging.ERROR)
  logging.getLogger("").addHandler(console)

  logger = logging.getLogger()
  logger.debug("Initialization")

  tb_writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s"
      % "\n".join([f"|{key}|{value}" for key, value in vars(FLAGS).items()]),
  )
  # Create the game instance
  game = factory.create_game_with_setting(
      "mfg_crowd_modelling_2d", FLAGS.game_setting
  )

  # Set the initial policy to uniform and generate the distribution
  uniform_policy = policy_std.UniformRandomPolicy(game)
  mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
  env = rl_environment.Environment(
      game, mfg_distribution=mfg_dist, mfg_population=0
  )

  # Set the environment seed for reproduciblility
  env.seed(FLAGS.seed)

  # Creat the agent and population policies
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent = mfg_ppo_agent(info_state_size, num_actions).to(device)
  ppo_policy = mfg_ppo_policy(game, agent, None, device)
  pop_agent = mfg_ppo_agent(info_state_size, num_actions).to(device)

  if FLAGS.optimizer == "Adam":
    optimizer_actor = optim.Adam(
        agent.actor.parameters(), lr=FLAGS.lr, eps=1e-5
    )
    optimizer_critic = optim.Adam(
        agent.critic.parameters(), lr=FLAGS.lr, eps=1e-5
    )
  else:
    optimizer_actor = optim.SGD(
        agent.actor.parameters(), lr=FLAGS.lr, momentum=0.9
    )
    optimizer_critic = optim.SGD(
        agent.critic.parameters(), lr=FLAGS.lr, momentum=0.9
    )

  # Used to log data for debugging
  steps = FLAGS.num_episodes * env.max_game_length
  episode_entropy = []
  total_entropy = []
  nash_con_vect = []
  eps_reward = []
  total_reward = []

  for k in range(FLAGS.update_iterations):
    for _ in range(FLAGS.update_episodes):
      # collect rollout data
      history = rollout(
          env, pop_agent, agent, FLAGS.num_episodes, steps, device
      )
      # store rewards and entropy for debugging
      episode_entropy.append(history["entropies"].mean().item())
      eps_reward.append(history["rewards"].sum().item() / FLAGS.num_episodes)
      # Calculate the advantage function
      adv, returns = calculate_advantage(
          FLAGS.gamma,
          True,
          history["rewards"],
          history["values"],
          history["dones"],
          device,
      )
      history["advantages"] = adv
      history["returns"] = returns
      # Update the learned policy and report loss for debugging
      v_loss = learn(
          history,
          optimizer_actor,
          optimizer_critic,
          agent,
          num_minibatches=FLAGS.num_minibatches,
          update_epochs=FLAGS.update_epochs,
          itr_eps=FLAGS.itr_eps,
          eps_eps=FLAGS.eps_eps,
          alpha=FLAGS.alpha,
          ent_coef=FLAGS.ent_coef,
          max_grad_norm=FLAGS.max_grad_norm,
      )

    # Collect and print the metrics
    total_reward.append(np.mean(eps_reward))
    total_entropy.append(np.mean(episode_entropy))

    print("Value_loss", v_loss.item())
    print("iteration num:", k + 1)
    print("Mean reward", total_reward[-1])

    # Update the iteration policy with the new policy
    pop_agent.load_state_dict(agent.state_dict())

    # Update the distribution
    distrib = distribution.DistributionPolicy(game, ppo_policy)

    # calculate the exploitability
    m = calculate_explotability(game, distrib, ppo_policy)
    nashc = m["nash_conv_ppo"]
    nash_con_vect.append(nashc)

    # log the results to tensor board
    tb_writer.add_scalar("initial_state_value", m["ppo_br/initial"], k + 1)
    tb_writer.add_scalar("rewards", total_reward[-1], k + 1)
    tb_writer.add_scalar("entorpy", total_entropy[-1], k + 1)
    tb_writer.add_scalar("nash_conv_ppo", nashc, k + 1)
    logger.debug(
        "ppo_br: %s, and nash_conv: %s, reward: %s, entropy: %s",
        m["ppo_br/initial"],
        nashc,
        total_reward[-1],
        total_entropy[-1],
    )
    print(
        "ppo_br: %s, and nash_conv: %s, reward: %s, entropy: %s"
        % (m["ppo_br/initial"], nashc, total_reward[-1], total_entropy[-1])
    )

    # Update the environment distribution
    env.update_mfg_distribution(distrib)

  # if lower than upper_nash we save the weights and distribution
  upper_nash = 300
  if nash_con_vect[-1] < upper_nash:
    # Save the distribution and weights for further analysis
    filename = os.path.join(fname, f"distribution_{run_name}.pkl")
    utils.save_parametric_distribution(distrib, filename)
    torch.save(
        agent.actor.state_dict(),
        fname
        + f"alpha_{FLAGS.alpha},                itr_eps_{FLAGS.itr_eps},"
        f" eps_eps_{FLAGS.eps_eps}_agent_actor_weights.pth",
    )
    torch.save(
        agent.critic.state_dict(),
        fname
        + f"alpha_{FLAGS.alpha},                itr_eps_{FLAGS.itr_eps},"
        f" eps_eps_{FLAGS.eps_eps}_agent_critic_weights.pth",
    )


if __name__ == "__main__":
  app.run(main)
