# Copyright 2023 DeepMind Technologies Limited
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

"""TD Learning with N-Tuple Networks for 2048."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "2048", "Name of the game.")
flags.DEFINE_integer("num_train_episodes", 15000,
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 1000,
                     "Episode frequency at which the agent is evaluated.")
flags.DEFINE_float("alpha", 0.02, "Learning rate")


class NTupleNetwork:
  """An N-tuple Network class.

  N-Tuple Networks are an effective way of reducing the storage requirement for
  evaluating and learning state values. This is accomplished by defining a
  collection of N-Tuples that represent various segments in a game's
  ObservationTensor.

  The value of a given state is defined as the sum of values of each N-Tuple,
  which are stored in a look up table. The policy of the agent is to chose an
  action that maximises the value of the after-state. After each episode, all
  the states that were reached in that episode is used for updating the state
  values using Temporal Difference Learning.

  References:
  [1] Szubert, Marcin and Wojciech Jaśkowski. "Temporal difference learning of
  n-tuple networks for the game 2048." Computational Intelligence and Games
  (CIG), 2014 IEEE Conference on. IEEE, 2014.
  """

  def __init__(self, n_tuple_size, max_tuple_index, n_tuples):
    for tuples in n_tuples:
      if len(tuples) != n_tuple_size:
        raise ValueError("n_tuple_size does not match size of tuples")
    n_tuple_network_size = len(n_tuples)
    look_up_table_shape = (n_tuple_network_size,) + (
        max_tuple_index,
    ) * n_tuple_size

    self.n_tuples = n_tuples
    self.look_up_table = np.zeros(look_up_table_shape)

  def learn(self, states):
    target = 0
    while states:
      state = states.pop()
      error = target - self.value(state)
      target = state.rewards()[0] + self.update(state, FLAGS.alpha * error)

  def update(self, state, adjust):
    v = 0
    for idx, n_tuple in enumerate(self.n_tuples):
      v += self.update_tuple(idx, n_tuple, state, adjust)
    return v

  def update_tuple(self, idx, n_tuple, state, adjust):
    observation_tensor = state.observation_tensor(0)
    index = (idx,) + tuple(
        [
            0
            if observation_tensor[tile] == 0
            else int(np.log2(observation_tensor[tile]))
            for tile in n_tuple
        ]
    )
    self.look_up_table[index] += adjust
    return self.look_up_table[index]

  def evaluator(self, state, action):
    working_state = state.clone()
    working_state.apply_action(action)
    return working_state.rewards()[0] + self.value(working_state)

  def value(self, state):
    """Returns the value of this state."""

    observation_tensor = state.observation_tensor(0)
    v = 0
    for idx, n_tuple in enumerate(self.n_tuples):
      lookup_tuple_index = [
          0
          if observation_tensor[tile] == 0
          else int(np.log2(observation_tensor[tile]))
          for tile in n_tuple
      ]
      lookup_index = (idx,) + tuple(lookup_tuple_index)
      v += self.look_up_table[lookup_index]
    return v


def main(_):
  n_tuple_network = NTupleNetwork(
      6,
      15,
      [
          [0, 1, 2, 3, 4, 5],
          [4, 5, 6, 7, 8, 9],
          [0, 1, 2, 4, 5, 6],
          [4, 5, 6, 8, 9, 10],
      ],
  )
  game = pyspiel.load_game(FLAGS.game)
  sum_rewards = 0
  largest_tile = 0
  max_score = 0
  for ep in range(FLAGS.num_train_episodes):
    state = game.new_initial_state()
    states_in_episode = []
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
      else:
        legal_actions = state.legal_actions(state.current_player())
        # pylint: disable=cell-var-from-loop
        best_action = max(
            legal_actions,
            key=lambda action: n_tuple_network.evaluator(state, action),
        )
        state.apply_action(best_action)
        states_in_episode.append(state.clone())

    sum_rewards += state.returns()[0]
    largest_tile_from_episode = max(state.observation_tensor(0))
    if largest_tile_from_episode > largest_tile:
      largest_tile = largest_tile_from_episode
    if state.returns()[0] > max_score:
      max_score = state.returns()[0]

    n_tuple_network.learn(states_in_episode)

    if (ep + 1) % FLAGS.eval_every == 0:
      logging.info(
          "[%s] Average Score: %s, Max Score: %s, Largest Tile Reached: %s",
          ep + 1,
          int(sum_rewards / FLAGS.eval_every),
          int(max_score),
          int(largest_tile),
      )
      sum_rewards = 0
      largest_tile = 0
      max_score = 0


if __name__ == "__main__":
  app.run(main)
