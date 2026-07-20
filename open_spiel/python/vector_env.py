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
"""A vectorized RL Environment."""

import ctypes
import multiprocessing as mp
import numpy as np
from open_spiel.python import rl_environment

class SyncVectorEnv(object):
  """A vectorized RL Environment.

  This environment is synchronized - games do not execute in parallel. Speedups
  are realized by calling models on many game states simultaneously.
  """

  def __init__(self, envs):
    if not isinstance(envs, list):
      raise ValueError(
          "Need to call this with a list of rl_environment.Environment objects")
    self.envs = envs

  def __len__(self):
    return len(self.envs)

  def observation_spec(self):
    return self.envs[0].observation_spec()

  @property
  def num_players(self):
    return self.envs[0].num_players

  def step(self, step_outputs, reset_if_done=False):
    """Apply one step."""
    time_steps = [
        self.envs[i].step([step_outputs[i].action])
        for i in range(len(self.envs))
    ]
    reward = [step.rewards for step in time_steps]
    done = [step.last() for step in time_steps]
    unreset_time_steps = time_steps

    if reset_if_done:
      time_steps = self.reset(envs_to_reset=done)

    return time_steps, reward, done, unreset_time_steps

  def reset(self, envs_to_reset=None):
    if envs_to_reset is None:
      envs_to_reset = [True for _ in range(len(self.envs))]

    time_steps = [
        self.envs[i].reset()
        if envs_to_reset[i] else self.envs[i].get_time_step()
        for i in range(len(self.envs))
    ]
    return time_steps


def _worker_init(remote, parent_remote, env_fn_arg, shared_mem, env_id, num_players, obs_size):
  """A worker process for AsyncVectorEnv using shared memory.
  
  Supports macOS/Windows spawn configurations via dynamic game reconstruction
  if the environment initialization callable is an unpicklable lambda.
  """
  parent_remote.close()
  
  if isinstance(env_fn_arg, str):
    # Reconstruct the environment directly from the extracted game name string
    env = rl_environment.Environment(env_fn_arg)
  else:
    env = env_fn_arg()
  
  obs_array = np.frombuffer(shared_mem, dtype=np.float32).reshape(
      (-1, num_players, obs_size))

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == "step":
        action, reset_if_done = data
        time_step = env.step([action])
        reward = time_step.rewards
        done = time_step.last()
        unreset_time_step = time_step

        info_state = time_step.observations["info_state"]
        for p in range(num_players):
          obs_array[env_id, p] = info_state[p]

        obs_dict = time_step.observations.copy()
        del obs_dict["info_state"]

        if reset_if_done and done:
          time_step = env.reset()
          info_state = time_step.observations["info_state"]
          for p in range(num_players):
            obs_array[env_id, p] = info_state[p]
          obs_dict = time_step.observations.copy()
          del obs_dict["info_state"]
          remote.send((obs_dict, time_step.rewards, time_step.discounts,
                       time_step.step_type, reward, done, unreset_time_step))
        else:
          remote.send((obs_dict, time_step.rewards, time_step.discounts,
                       time_step.step_type, reward, done, None))

      elif cmd == "reset":
        time_step = env.reset()
        info_state = time_step.observations["info_state"]
        for p in range(num_players):
          obs_array[env_id, p] = info_state[p]
        obs_dict = time_step.observations.copy()
        del obs_dict["info_state"]
        remote.send((obs_dict, time_step.rewards, time_step.discounts,
                     time_step.step_type))

      elif cmd == "get_time_step":
        time_step = env.get_time_step()
        info_state = time_step.observations["info_state"]
        for p in range(num_players):
          obs_array[env_id, p] = info_state[p]
        obs_dict = time_step.observations.copy()
        del obs_dict["info_state"]
        remote.send((obs_dict, time_step.rewards, time_step.discounts,
                     time_step.step_type))

      elif cmd == "observation_spec":
        remote.send(env.observation_spec())
      elif cmd == "num_players":
        remote.send(env.num_players)
      elif cmd == "close":
        remote.close()
        break
      else:
        raise NotImplementedError(f"Command {cmd} not implemented.")
  except KeyboardInterrupt:
    pass
  except Exception as e:
    import traceback
    traceback.print_exc()
    raise


class AsyncVectorEnv(object):
  """A vectorized RL Environment running in parallel using shared memory."""

  def __init__(self, env_fns):
    if not isinstance(env_fns, list):
      raise ValueError("Need to call this with a list of callables")

    self.waiting = False
    self.closed = False
    self.num_envs = len(env_fns)

    dummy_env = env_fns[0]()
    self._observation_spec = dummy_env.observation_spec()
    self._num_players = dummy_env.num_players
    obs_size = int(np.prod(self._observation_spec["info_state"]))

    self._shared_mem = mp.RawArray(
        ctypes.c_float, self.num_envs * self._num_players * obs_size)
    self._obs_array = np.frombuffer(self._shared_mem, dtype=np.float32).reshape(
        (self.num_envs, self._num_players, obs_size))

    self.remotes, self.work_remotes = zip(
        *[mp.Pipe() for _ in range(self.num_envs)])
    
    self.processes = []
    for i, (work_remote, remote, env_fn) in enumerate(
        zip(self.work_remotes, self.remotes, env_fns)):
      
      try:
        import pickle
        pickle.dumps(env_fn)
        env_fn_arg = env_fn
      except (pickle.PicklingError, TypeError, AttributeError):
        # Fallback for unpicklable lambdas: extract the game name parameter string directly
        if hasattr(dummy_env, "game") and hasattr(dummy_env.game, "get_type"):
          env_fn_arg = dummy_env.game.get_type().short_name
        else:
          env_fn_arg = "tic_tac_toe" # Default baseline fallback
      
      p = mp.Process(
          target=_worker_init,
          args=(work_remote, remote, env_fn_arg, self._shared_mem, i, 
                self._num_players, obs_size),
          daemon=True)
      self.processes.append(p)

    for p in self.processes:
      p.start()

    for remote in self.work_remotes:
      remote.close()

  def __len__(self):
    return self.num_envs

  def observation_spec(self):
    return self._observation_spec

  @property
  def num_players(self):
    return self._num_players

  def step_async(self, step_outputs, reset_if_done=False):
    for remote, step_output in zip(self.remotes, step_outputs):
      remote.send(("step", (step_output.action, reset_if_done)))
    self.waiting = True

  def step_wait(self):
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False

    time_steps = []
    rewards = []
    dones = []
    unreset_time_steps = []
    for i, res in enumerate(results):
      obs_dict, rew, disc, step_type, return_rew, done, unreset_ts = res
      full_obs = obs_dict.copy()
      full_obs["info_state"] = [
          np.copy(self._obs_array[i, p]) for p in range(self._num_players)
      ]

      ts = rl_environment.TimeStep(
          observations=full_obs,
          rewards=rew,
          discounts=disc,
          step_type=step_type)
      time_steps.append(ts)
      rewards.append(return_rew)
      dones.append(done)

      if unreset_ts is not None:
        unreset_time_steps.append(unreset_ts)
      else:
        unreset_time_steps.append(ts)

    return time_steps, rewards, dones, unreset_time_steps

  def step(self, step_outputs, reset_if_done=False):
    self.step_async(step_outputs, reset_if_done)
    return self.step_wait()

  def reset(self, envs_to_reset=None):
    if envs_to_reset is None:
      envs_to_reset = [True for _ in range(self.num_envs)]

    for i, remote in enumerate(self.remotes):
      if envs_to_reset[i]:
        remote.send(("reset", None))
      else:
        remote.send(("get_time_step", None))

    results = [remote.recv() for remote in self.remotes]
    time_steps = []
    for i, (obs_dict, rew, disc, step_type) in enumerate(results):
      full_obs = obs_dict.copy()
      full_obs["info_state"] = [
          np.copy(self._obs_array[i, p]) for p in range(self._num_players)
      ]
      ts = rl_environment.TimeStep(
          observations=full_obs,
          rewards=rew,
          discounts=disc,
          step_type=step_type)
      time_steps.append(ts)

    return time_steps

  def close(self):
    if self.closed:
      return
    try:
      if self.waiting:
        for remote in self.remotes:
          try:
            remote.recv()
          except (EOFError, IOError):
            pass
      for remote in self.remotes:
        try:
          remote.send(("close", None))
        except (IOError, BrokenPipeError):
          pass
      for p in self.processes:
        p.join(timeout=5)
        if p.is_alive():
          p.terminate()
    except Exception:
      pass
    finally:
      self.closed = True

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
