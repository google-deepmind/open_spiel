import gym
import numpy as np
import pyspiel

from stable_baselines3.common.atari_wrappers import (
  ClipRewardEnv,
  EpisodicLifeEnv,
  FireResetEnv,
  MaxAndSkipEnv,
  NoopResetEnv
)

### NOTE: We include this wrapper by hand because the default wrapper threw errors (see modified lines).
class NoopResetEnv(gym.Wrapper):
  """
  Sample initial states by taking random number of no-ops on reset.
  No-op is assumed to be action 0.
  :param env: the environment to wrap
  :param noop_max: the maximum value of no-ops to run
  """

  def __init__(self, env: gym.Env, noop_max: int = 30):
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == "NOOP"

  def reset(self, **kwargs) -> np.ndarray:
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      #### MODIFIED LINES ###
      noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
      ### END MODIFIED LIENS ###
    assert noops > 0
    obs = np.zeros(0)
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs

_NUM_PLAYERS = 1
_GAME_TYPE = pyspiel.GameType(
  short_name="atari",
  long_name="atari",
  dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
  chance_mode=pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC,
  information=pyspiel.GameType.Information.PERFECT_INFORMATION,
  utility=pyspiel.GameType.Utility.ZERO_SUM,
  reward_model=pyspiel.GameType.RewardModel.REWARDS,
  max_num_players=_NUM_PLAYERS,
  min_num_players=_NUM_PLAYERS,
  provides_information_state_string=False,
  provides_information_state_tensor=True,
  provides_observation_string=False,
  provides_observation_tensor=False,
  parameter_specification={"gym_id": 'ALE/Breakout-v5', "seed": 1, "idx": 0, "capture_video": False, 'run_name': 'default', 'use_episodic_life_env': True})
_GAME_INFO = pyspiel.GameInfo(
  num_distinct_actions=4,
  max_chance_outcomes=0,
  num_players=_NUM_PLAYERS,
  min_utility=-1.0,
  max_utility=1.0,
  utility_sum=0.0,
  max_game_length=2000)

class AtariGame(pyspiel.Game):

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
      env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
    
    # Wrappers are a bit specialized right nwo to Breakout - different games may want different wrappers.
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if self.use_episodic_life_env:
      env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
      env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(self.seed)
    env.action_space.seed(self.seed)
    env.observation_space.seed(self.seed)
    self.env = env

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return AtariState(self,)

  def information_state_tensor_size(self):
    return AtariState(self).information_state_tensor(0).shape

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
    self.observation = observation # Store this for later

  def information_state_tensor(self, player_id):
    return self.observation

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
    return "DEBUG"

# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, AtariGame)
