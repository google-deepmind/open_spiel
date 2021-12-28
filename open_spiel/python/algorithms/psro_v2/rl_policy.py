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

# Lint as: python3
"""DQN as a policy.

Treating RL Oracles as policies allows us to streamline their use with tabular
policies and other policies in OpenSpiel, and freely mix populations using
different types of oracles.
"""

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import policy_gradient


def rl_policy_factory(rl_class):
  """Transforms an RL Agent into an OpenSpiel policy.

  Args:
    rl_class: An OpenSpiel class inheriting from 'rl_agent.AbstractAgent' such
      as policy_gradient.PolicyGradient or dqn.DQN.

  Returns:
    An RLPolicy class that wraps around an instance of rl_class to transform it
    into a policy.
  """

  class RLPolicy(policy.Policy):
    """A 'policy.Policy' wrapper around an 'rl_agent.AbstractAgent' instance."""

    def __init__(self, env, player_id, **kwargs):
      """Constructs an RL Policy.

      Args:
        env: An OpenSpiel RL Environment instance.
        player_id: The ID of the DQN policy's player.
        **kwargs: Various kwargs used to initialize rl_class.
      """
      game = env.game

      super(RLPolicy, self).__init__(game, player_id)
      self._policy = rl_class(**{"player_id": player_id, **kwargs})

      self._frozen = False
      self._rl_class = rl_class
      self._env = env
      self._obs = {
          "info_state": [None] * self.game.num_players(),
          "legal_actions": [None] * self.game.num_players()
      }

    def get_time_step(self):
      time_step = self._env.get_time_step()
      return time_step

    def action_probabilities(self, state, player_id=None):
      cur_player = state.current_player()
      legal_actions = state.legal_actions(cur_player)

      step_type = rl_environment.StepType.LAST if state.is_terminal(
      ) else rl_environment.StepType.MID

      self._obs["current_player"] = cur_player
      self._obs["info_state"][cur_player] = (
          state.information_state_tensor(cur_player))
      self._obs["legal_actions"][cur_player] = legal_actions

      # pylint: disable=protected-access
      rewards = state.rewards()
      if rewards:
        time_step = rl_environment.TimeStep(
            observations=self._obs, rewards=rewards,
            discounts=self._env._discounts, step_type=step_type)
      else:
        rewards = [0] * self._num_players
        time_step = rl_environment.TimeStep(
            observations=self._obs, rewards=rewards,
            discounts=self._env._discounts,
            step_type=rl_environment.StepType.FIRST)
      # pylint: enable=protected-access

      p = self._policy.step(time_step, is_evaluation=True).probs
      prob_dict = {action: p[action] for action in legal_actions}
      return prob_dict

    def step(self, time_step, is_evaluation=False):
      # The _frozen attribute freezes the weights of the current policy. This
      # effect is achieved by considering that we always are evaluating when the
      # current policy's weights are frozen. For more details, see the freeze()
      # method.
      is_evaluation = (is_evaluation) or (self._frozen)
      return self._policy.step(time_step, is_evaluation)

    def freeze(self):
      """This method freezes the policy's weights.

      The weight freezing effect is implemented by preventing any training to
      take place through calls to the step function. The weights are therefore
      not effectively frozen, and unconventional calls may trigger weights
      training.

      The weight-freezing effect is especially needed in PSRO, where all
      policies that aren't being trained by the oracle must be static. Freezing
      trained policies permitted us not to change how 'step' was called when
      introducing self-play (By not changing 'is_evaluation' depending on the
      current player).
      """
      self._frozen = True

    def unfreeze(self):
      self._frozen = False

    def is_frozen(self):
      return self._frozen

    def get_weights(self):
      return self._policy.get_weights()

    def copy_with_noise(self, sigma=0.0):
      copied_object = RLPolicy.__new__(RLPolicy)
      super(RLPolicy, copied_object).__init__(self.game, self.player_ids)
      setattr(copied_object, "_rl_class", self._rl_class)
      setattr(copied_object, "_obs", self._obs)
      setattr(copied_object, "_policy",
              self._policy.copy_with_noise(sigma=sigma))
      setattr(copied_object, "_env", self._env)
      copied_object.unfreeze()

      return copied_object

  return RLPolicy


# Generating policy classes for Policy Gradient and DQN
# pylint: disable=invalid-name
PGPolicy = rl_policy_factory(policy_gradient.PolicyGradient)
DQNPolicy = rl_policy_factory(dqn.DQN)
# pylint: enable=invalid-name
