# Copyright 2019 DeepMind Technologies Limited
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

from absl.testing import absltest
import jax
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python.examples.rrps_poprl import impala
from open_spiel.python.examples.rrps_poprl import rl_environment
import pyspiel


# A simple two-action game encoded as an EFG game. Going left gets -1, going
# right gets a +1.
SIMPLE_EFG_DATA = """
  EFG 2 R "Simple single-agent problem" { "Player 1" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { -1.0 }
    t "R" 2 "Outcome R" { 1.0 }
"""


class FixedSequenceAgent(rl_agent.AbstractAgent):
  """An example agent class."""

  def __init__(
      self, player_id, num_actions, sequence, name="fixed_sequence_agent"
  ):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions
    self._sequence = sequence
    self._seq_idx = 0

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    probs = np.zeros(self._num_actions)
    action = self._sequence[self._seq_idx]
    self._seq_idx += 1
    if self._seq_idx >= len(self._sequence):
      self._seq_idx = 0
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


class IMPALATest(absltest.TestCase):

  def test_simple_game(self):
    game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)
    env = rl_environment.Environment(game=game)
    max_abs_reward = max(
        abs(env.game.min_utility()), abs(env.game.max_utility())
    )
    agent = impala.IMPALA(
        0,
        state_representation_size=game.information_state_tensor_shape()[0],
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        unroll_len=20,
        net_factory=impala.BasicRNN,
        rng_key=jax.random.PRNGKey(42),
        max_abs_reward=max_abs_reward,
        learning_rate=5e-3,
        hidden_layers_sizes=[16],
        batch_size=5,
    )

    total_reward = 0
    for _ in range(1000):
      time_step = env.reset()
      while not time_step.last():
        agent_output = agent.step(time_step)
        time_step = env.step([agent_output.action])
        total_reward += time_step.rewards[0]
      agent.step(time_step)
    print(total_reward)
    self.assertGreaterEqual(total_reward, 500)

  @absltest.skip("Takes too long to run, but does approach 1.")
  def test_catch(self):
    env = rl_environment.Environment("catch")
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    num_players = env.num_players
    max_abs_reward = max(
        abs(env.game.min_utility()), abs(env.game.max_utility())
    )
    agent = impala.IMPALA(
        0,
        state_representation_size=state_size,
        num_actions=num_actions,
        num_players=num_players,
        unroll_len=20,
        net_factory=impala.BasicRNN,
        rng_key=jax.random.PRNGKey(42),
        max_abs_reward=max_abs_reward,
        learning_rate=5e-3,
        hidden_layers_sizes=[16],
        batch_size=2,
    )

    window_sum = 0
    window_width = 50
    window_tick = 0
    for ep in range(10000):
      episode_reward = 0
      time_step = env.reset()
      while not time_step.last():
        agent_output = agent.step(time_step)
        time_step = env.step([agent_output.action])
        episode_reward += time_step.rewards[0]
      # print(f"Total reward: {total_reward}")
      # avg_rew = total_reward / (ep + 1)
      agent.step(time_step)
      window_sum += episode_reward
      window_tick += 1
      if window_tick >= window_width:
        avg_window_reward = window_sum / window_width
        window_tick = 0
        window_sum = 0
        print(f"Ep {ep}, avg window rew: {avg_window_reward}")

  @absltest.skip("Takes too long to run, but does approach 1000.")
  def test_run_rps(self):
    env = rl_environment.Environment(
        f"repeated_game(stage_game=matrix_rps(),num_repetitions={pyspiel.ROSHAMBO_NUM_THROWS},recall=1)"
    )
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    num_players = env.num_players
    max_abs_reward = max(
        abs(env.game.min_utility()), abs(env.game.max_utility())
    )
    agents = [
        impala.IMPALA(  # pylint: disable=g-complex-comprehension
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            num_players=num_players,
            unroll_len=20,
            net_factory=impala.BasicRNN,
            rng_key=jax.random.PRNGKey(seed),
            max_abs_reward=max_abs_reward,
            entropy=0.001,
            learning_rate=0.001,
            hidden_layers_sizes=[64, 32],
            prediction_weight=0,
            discount_factor=0.9,
            batch_size=16,
        )
        for (player_id, seed) in [(0, 238576517), (1, 738328671)]
    ]
    agents[0] = FixedSequenceAgent(0, num_actions, [0, 1, 2, 2, 1, 0])
    for ep in range(1000):
      time_step = env.reset()
      total_rewards = np.zeros(2, dtype=np.float32)
      while not time_step.last():
        agent_outputs = [agents[0].step(time_step), agents[1].step(time_step)]
        actions = [agent_outputs[0].action, agent_outputs[1].action]
        time_step = env.step(actions)
        total_rewards += np.array(time_step.rewards)
      for agent in agents:
        agent.step(time_step)
      print(f"{ep} {total_rewards}")


if __name__ == "__main__":
  absltest.main()
