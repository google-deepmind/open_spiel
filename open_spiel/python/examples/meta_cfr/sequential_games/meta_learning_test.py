# Copyright 2022 DeepMind Technologies Limited
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

"""Tests for meta CFR Algorithm."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import mock
import numpy as np
import optax

from open_spiel.python.examples.meta_cfr.sequential_games import meta_learning
from open_spiel.python.examples.meta_cfr.sequential_games import models
from open_spiel.python.examples.meta_cfr.sequential_games import openspiel_api

FLAGS = flags.FLAGS


def meta_cfr_agent(game_name='kuhn_poker'):
  return meta_learning.MetaCFRRegretAgent(
      training_epochs=1,
      meta_learner_training_epochs=1,
      game_name=game_name,
      game_config={'players': 2},
      perturbation=False,
      seed=0,
      model_type='MLP',
      best_response=True)


class MetaLearningTest(parameterized.TestCase):

  def setup_optimizer(self, num_actions, num_infostates):
    if FLAGS.use_infostate_representation:
      dummy_input = np.zeros(
          shape=[FLAGS.batch_size, 1, num_actions + num_infostates])
    else:
      dummy_input = np.zeros(shape=[FLAGS.batch_size, 1, num_actions])

    def mlp_forward(dummy_input):
      mlp = hk.nets.MLP([10, num_actions])
      return mlp(dummy_input)
    forward = hk.transform(mlp_forward)

    rng_seq = jax.random.PRNGKey(10)
    params = forward.init(rng_seq, dummy_input)
    lr_scheduler_fn = optax.polynomial_schedule(
        init_value=0.2, end_value=0.0001, power=1., transition_steps=100)
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(), optax.scale_by_schedule(lr_scheduler_fn),
        optax.scale(-0.2))
    net_apply = forward.apply
    opt_state = opt_init(params)
    return params, net_apply, opt_state, opt_update

  @parameterized.named_parameters(('kuhn_poker_game', 'kuhn_poker'),
                                  ('leduc_poker_game', 'leduc_poker'))
  def test_worldstate_initialization(self, game_name):
    self._world_state = openspiel_api.WorldState(
        game_name, {'players': 2}, perturbation=False, random_seed=0)
    self._all_actions = self._world_state.get_distinct_actions()
    self.assertNotEmpty(self._all_actions,
                        'Number of distinct actions should be greater that 0.')

  @parameterized.named_parameters(('kuhn_poker_game', 'kuhn_poker'),
                                  ('leduc_poker_game', 'leduc_poker'))
  def test_meta_cfr_agent_initialization(self, game_name):
    with mock.patch.object(meta_learning.MetaCFRRegretAgent,
                           'get_num_infostates') as mock_get_num_infostates:
      mock_get_num_infostates.return_value = (mock.MagicMock(),
                                              mock.MagicMock())
      meta_learning.MetaCFRRegretAgent(
          training_epochs=1,
          meta_learner_training_epochs=1,
          game_name=game_name,
          game_config={'players': 2},
          perturbation=False,
          seed=0,
          model_type='MLP',
          best_response=True)
    mock_get_num_infostates.assert_called_once_with()

  @parameterized.named_parameters(('kuhn_poker_game', 'kuhn_poker'),
                                  ('leduc_poker_game', 'leduc_poker'))
  def test_meta_learning_training(self, game_name):
    agent = meta_learning.MetaCFRRegretAgent(
        training_epochs=1,
        meta_learner_training_epochs=1,
        game_name=game_name,
        game_config={'players': 2},
        perturbation=False,
        seed=0,
        model_type=models.ModelType.MLP.value,
        best_response=True)
    num_infostates, _ = agent.get_num_infostates()
    num_actions = len(agent._all_actions)
    params, net_apply, opt_state, opt_update = self.setup_optimizer(
        num_actions, num_infostates)
    agent.training_optimizer()
    agent.optimizer.net_apply = net_apply
    agent.optimizer.opt_state = opt_state
    agent.optimizer.net_params = params
    agent.optimizer.opt_update = opt_update

    world_state = openspiel_api.WorldState(
        game_name, {'players': 2}, perturbation=False, random_seed=0)
    best_response_val_player_2 = agent.next_policy(world_state)
    self.assertGreater(best_response_val_player_2[-1], 0)

if __name__ == '__main__':
  absltest.main()
