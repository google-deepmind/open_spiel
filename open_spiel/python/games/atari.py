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
"""OpenSpiel support for the Atari Learning Environment (ALE).

Originally introduced in (Bellemare et al., 2013):
https://arxiv.org/abs/1207.4708.

Uses environment wrappers from OpenAI Gym (https://gym.openai.com/) and Stable
Baselines 3 (https://jmlr.org/papers/v22/20-1364.html) to convert observations
into a suitable format for training.
"""

# pylint: disable=g-importing-member
from math import prod
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import pyspiel


_NUM_PLAYERS = 1
_GAME_TYPE = pyspiel.GameType(
    short_name='atari',
    long_name='atari',
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        'gym_id': 'ALE/Breakout-v5',
        'seed': 1,
        'idx': 0,
        'capture_video': False,
        'run_name': 'default',
        'use_episodic_life_env': True
    })
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=4,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=2000)


# NOTE: We include this wrapper by hand because the default wrapper
# threw errors (see modified lines).
class NoopResetEnv(gym.Wrapper):
  """Sample initial states by taking random number of no-ops on reset.

  No-op is assumed to be action 0. :param env: the environment to wrap :param
  noop_max: the maximum value of no-ops to run
  """

  def __init__(self, env: gym.Env, noop_max: int = 30):
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

  def reset(self, **kwargs) -> np.ndarray:
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      #### MODIFIED LINES: note method is named integers now ###
      noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
      ### END MODIFIED LINES ###
    assert noops > 0
    obs = np.zeros(0)
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs


class AtariGame(pyspiel.Game):
  """An OpenSpiel wrapper for the OpenAI Gym Atari games."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
    self.gym_id = params.get('gym_id', 'BreakoutNoFrameskip-v4')
    self.seed = params.get('seed', 1)
    self.idx = params.get('idx', 0)
    self.capture_video = params.get('capture_video', False)
    self.run_name = params.get('run_name', 'default')
    self.use_episodic_life_env = params.get('use_episodic_life_env', True)

    env = gym.make(self.gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if self.capture_video and self.idx == 0:
      env = gym.wrappers.RecordVideo(env, f'videos/{self.run_name}')

    # Apply the standard set of wrappers from CleanRL's PPO implementation.
    # These wrappers have been tested on Breakout; different games may
    # benefit from different wrappers (e.g., Space Invaders might benefit
    # from frameskip=3 instead of 4; see https://arxiv.org/abs/1312.5602).
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if self.use_episodic_life_env:
      env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
      env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(self.seed)
    env.action_space.seed(self.seed)
    env.observation_space.seed(self.seed)
    self.observation_shape = env.reset().shape
    self.env = env

  def observation_tensor_shape(self):
    return self.observation_shape

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return AtariState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if params is None:
      params = dict()

    params['observation_shape'] = self.observation_shape
    return AtariObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class AtariState(pyspiel.State):
  """A python version of the Atari Game state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._is_terminal = False
    self.tracked_rewards = 0
    self.env = game.env
    self.observation = self.env.reset()
    self.last_reward = None
    self.last_info = dict()

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else 0

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    return list(range(self.env.action_space.n))

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    observation, reward, done, info = self.env.step(action)
    self.last_info = info
    self.last_reward = reward
    self.tracked_rewards += reward
    if done:
      self._is_terminal = True
    self.observation = observation  # Store this for later

  def _action_to_string(self, player, action):
    return self.env.get_action_meanings()[action]

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def rewards(self):
    return [self.last_reward]

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self.tracked_rewards]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return 'DEBUG'


class AtariObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  # pylint: disable=unused-argument
  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    # Determine which observation pieces we want to include.
    pieces = []
    pieces.append(('observation', prod(params['observation_shape']),
                   params['observation_shape']))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros((total_size), np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if 'observation' in self.dict:
      self.dict['observation'][:] = state.observation

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    return ' '.join(str(p) for p in pieces)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, AtariGame)
