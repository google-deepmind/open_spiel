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

import multiprocessing as mp


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
    """Apply one step.

    Args:
      step_outputs: the step outputs
      reset_if_done: if True, automatically reset the environment
          when the epsiode ends

    Returns:
      time_steps: the time steps,
      reward: the reward
      done: done flag
      unreset_time_steps: unreset time steps
    """
    time_steps = [
        self.envs[i].step([step_outputs[i].action])
        for i in range(len(self.envs))
    ]
    reward = [step.rewards for step in time_steps]
    done = [step.last() for step in time_steps]
    unreset_time_steps = time_steps  # Copy these because you may want to look
                                     # at the unreset versions to extract
                                     # information from them

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


import ctypes
import numpy as np
from open_spiel.python import rl_environment


def _worker(remote, parent_remote, env_fn, shared_mem, env_id, num_players, obs_size):
  """A worker process for AsyncVectorEnv using shared memory."""
  parent_remote.close()
  env = env_fn()
  obs_array = np.frombuffer(shared_mem, dtype=np.float32).reshape((-1, num_players, obs_size))
  
  try:
    while True:
      cmd, data = remote.recv()
      if cmd == "step":
        action, reset_if_done = data
        time_step = env.step([action])
        reward = time_step.rewards
        done = time_step.last()
        unreset_time_step = time_step
        if reset_if_done and done:
          time_step = env.reset()
          
        for p in range(num_players):
            obs_array[env_id, p] = time_step.observations["info_state"][p]
            
        obs_dict = time_step.observations.copy()
        del obs_dict["info_state"]

        if reset_if_done and done:
            remote.send((obs_dict, time_step.rewards, time_step.discounts, time_step.step_type, reward, done, unreset_time_step))
        else:
            remote.send((obs_dict, time_step.rewards, time_step.discounts, time_step.step_type, reward, done, None))
            
      elif cmd == "reset":
        time_step = env.reset()
        for p in range(num_players):
            obs_array[env_id, p] = time_step.observations["info_state"][p]
        obs_dict = time_step.observations.copy()
        del obs_dict["info_state"]
        remote.send((obs_dict, time_step.rewards, time_step.discounts, time_step.step_type))
        
      elif cmd == "get_time_step":
        time_step = env.get_time_step()
        for p in range(num_players):
            obs_array[env_id, p] = time_step.observations["info_state"][p]
        obs_dict = time_step.observations.copy()
        del obs_dict["info_state"]
        remote.send((obs_dict, time_step.rewards, time_step.discounts, time_step.step_type))
        
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
  finally:
    pass


class AsyncVectorEnv(object):
  """A vectorized RL Environment that runs in multiple processes.

  This environment is asynchronous - games execute in parallel using Python's
  multiprocessing module. Speedups are realized by evaluating games on multiple
  CPU cores simultaneously. Uses shared memory to avoid IPC pickling overhead.
  """

  def __init__(self, env_fns):
    """Initializes the AsyncVectorEnv.

    Args:
      env_fns: A list of callables that return an rl_environment.Environment.
        Using callables prevents pickling issues with C++ game objects across
        process boundaries.
    """
    if not isinstance(env_fns, list):
      raise ValueError("Need to call this with a list of callables returning "
                       "rl_environment.Environment objects")

    self.waiting = False
    self.closed = False
    self.num_envs = len(env_fns)

    dummy_env = env_fns[0]()
    self._observation_spec = dummy_env.observation_spec()
    self._num_players = dummy_env.num_players
    obs_size = int(np.prod(self._observation_spec["info_state"]))
    
    self._shared_mem = mp.RawArray(ctypes.c_float, self.num_envs * self._num_players * obs_size)
    self._obs_array = np.frombuffer(self._shared_mem, dtype=np.float32).reshape((self.num_envs, self._num_players, obs_size))

    self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
    self.processes = [
        mp.Process(
            target=_worker, args=(work_remote, remote, env_fn, self._shared_mem, i, self._num_players, obs_size), daemon=True)
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns))
    ]

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
    """Sends step commands to all worker processes."""
    for remote, step_output in zip(self.remotes, step_outputs):
      remote.send(("step", (step_output.action, reset_if_done)))
    self.waiting = True

  def step_wait(self):
    """Waits for and retrieves step results from all worker processes."""
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False
    
    time_steps = []
    rewards = []
    dones = []
    unreset_time_steps = []
    for i, res in enumerate(results):
        obs_dict, rew, disc, step_type, return_rew, done, unreset_ts = res
        full_obs = obs_dict.copy()
        full_obs["info_state"] = [np.copy(self._obs_array[i, p]) for p in range(self._num_players)]
        
        ts = rl_environment.TimeStep(
            observations=full_obs,
            rewards=rew,
            discounts=disc,
            step_type=step_type
        )
        time_steps.append(ts)
        rewards.append(return_rew)
        dones.append(done)
        
        if unreset_ts is not None:
            unreset_time_steps.append(unreset_ts)
        else:
            unreset_time_steps.append(ts)
            
    return time_steps, rewards, dones, unreset_time_steps

  def step(self, step_outputs, reset_if_done=False):
    """Apply one step synchronously by calling step_async and step_wait."""
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
        full_obs["info_state"] = [np.copy(self._obs_array[i, p]) for p in range(self._num_players)]
        ts = rl_environment.TimeStep(
            observations=full_obs,
            rewards=rew,
            discounts=disc,
            step_type=step_type
        )
        time_steps.append(ts)
        
    return time_steps

  def close(self):
    """Cleanly shuts down all worker processes."""
    if self.closed:
      return
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(("close", None))
    for p in self.processes:
      p.join()
    self.closed = True

  def __del__(self):
    self.close()
