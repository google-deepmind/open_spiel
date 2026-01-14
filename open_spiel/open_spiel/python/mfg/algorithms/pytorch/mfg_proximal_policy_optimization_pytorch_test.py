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
"""Tests for Mean field proximal policy optimaztion."""

# pylint: disable=consider-using-from-import
# pylint: disable=g-importing-member

from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import Agent as mfg_ppo_agent
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import calculate_advantage
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import calculate_explotability
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import learn
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import Policy as mfg_ppo_policy
from open_spiel.python.mfg.algorithms.pytorch.mfg_proximal_policy_optimization import rollout
import torch
import torch.optim as optim

from open_spiel.python import policy as policy_std
from open_spiel.python import rl_environment
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.games import factory


class PolicyTest(parameterized.TestCase):
  """Test the policy."""

  @parameterized.named_parameters(
      ("python", "mfg_crowd_modelling_2d", "crowd_modelling_2d_four_rooms")
  )
  def test_train(self, name, setting):
    """Checks that the training works."""
    device = torch.device("cpu")
    args = {
        "num_episodes": 5,
        "gamma": 0.9,
    }
    game = factory.create_game_with_setting(name, setting)
    uniform_policy = policy_std.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(
        game, mfg_distribution=mfg_dist, mfg_population=0
    )

    # Set the environment seed for reproduciblility
    env.seed(0)

    # Creat the agent and population policies
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = mfg_ppo_agent(info_state_size, num_actions).to(device)
    ppo_policy = mfg_ppo_policy(game, agent, None, device)
    pop_agent = mfg_ppo_agent(info_state_size, num_actions).to(device)

    optimizer_actor = optim.Adam(agent.actor.parameters(), lr=1e-3, eps=1e-5)
    optimizer_critic = optim.Adam(agent.critic.parameters(), lr=1e-3, eps=1e-5)

    # calculate the exploitability
    m = calculate_explotability(game, mfg_dist, ppo_policy)
    init_nashc = m["nash_conv_ppo"]

    steps = args["num_episodes"] * env.max_game_length

    for _ in range(3):
      # collect rollout data
      history = rollout(
          env, pop_agent, agent, args["num_episodes"], steps, device
      )
      # Calculate the advantage function
      adv, returns = calculate_advantage(
          args["gamma"],
          True,
          history["rewards"],
          history["values"],
          history["dones"],
          device,
      )
      history["advantages"] = adv
      history["returns"] = returns
      # Update the learned policy and report loss for debugging
      learn(history, optimizer_actor, optimizer_critic, agent)

    # Update the iteration policy with the new policy
    pop_agent.load_state_dict(agent.state_dict())

    # Update the distribution
    distrib = distribution.DistributionPolicy(game, ppo_policy)

    # calculate the exploitability
    m = calculate_explotability(game, distrib, ppo_policy)
    nashc = m["nash_conv_ppo"]

    # update the environment distribution
    env.update_mfg_distribution(distrib)

    # Test convergence
    self.assertLessEqual(nashc, 2 * init_nashc)


if __name__ == "__main__":
  absltest.main()
