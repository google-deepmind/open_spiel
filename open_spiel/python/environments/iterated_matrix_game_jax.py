from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pyspiel
from pyspiel import PlayerId
import numpy as np
import open_spiel.python.rl_environment
from open_spiel.python import rl_environment

from open_spiel.python.rl_environment import Environment, TimeStep, StepType


def make_env_fns(env: Environment, batch_size: int, max_iters: int, payoffs: jnp.array):
        num_actions = jnp.prod(jnp.array([n for n in env.action_spec()['num_actions']]))
        cases = jnp.arange(num_actions) + 1
        cases = jnp.reshape(cases, env.action_spec()['num_actions'])
        indices = jnp.eye(num_actions + 1)
        initial_obs = {
            'info_state': jnp.stack([indices[jnp.zeros(batch_size, dtype=jnp.int32)]] * env.num_players, axis=0),
            'legal_actions': np.array([[np.arange(env.action_spec()['num_actions'][p])] * batch_size for p in range(env.num_players)]),
            'current_player': -2,
            't': 0
        }
        def step(state: TimeStep, action: jnp.array) -> TimeStep:
            t = state.observations['t']
            rewards = payoffs[tuple(action.T)]
            info_state = [
                indices[cases[tuple(action.T)]],
                indices[cases[tuple(action[..., ::-1].T)]]
            ]
            info_state = jnp.stack(info_state, axis=0)
            discounts = jnp.ones_like(rewards)
            return TimeStep(
                observations={
                    'info_state': info_state,
                    'legal_actions': state.observations['legal_actions'],
                    'current_player': -2,
                    't': t + 1
                },
                rewards=rewards.T,
                discounts=discounts,
                step_type=jax.lax.select(t < max_iters - 1, 1, 2)
            )

        def reset() -> TimeStep:
            return TimeStep(
                observations=initial_obs,
                rewards=jnp.zeros(env.num_players),
                discounts=jnp.ones(env.num_players),
                step_type=0
            )
        #return step, reset
        return jax.jit(step), jax.jit(reset)

class IteratedMatrixGame:

    def __init__(self, payoff_matrix: jnp.ndarray, iterations: int, batch_size=1, include_remaining_iterations=True):
        self._payoff_matrix = payoff_matrix
        self._num_players = payoff_matrix.ndim - 1
        self._step, self._reset = make_env_fns(env=self, max_iters=iterations, batch_size=batch_size, payoffs=payoff_matrix)
        self._state = self._reset()

    @property
    def num_players(self):
        return self._num_players

    def observation_spec(self):
        return dict(
            info_state=tuple([np.sum(self._payoff_matrix.shape[:-1]) + 1] for _ in range(self._num_players)),
            legal_actions=tuple([self._payoff_matrix.shape[p] for p in range(self._num_players)]),
            current_player=()
        )

    def action_spec(self):
        return dict(
            num_actions=tuple([self._payoff_matrix.shape[p] for p in range(self._num_players)]),
            min=tuple([0 for p in range(self._num_players)]),
            max=tuple([self._payoff_matrix.shape[p]-1 for p in range(self._num_players)]),
            dtype=int,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, action: np.ndarray):
        self._state = self._step(self._state, action)
        return self._state

    @partial(jax.jit, static_argnums=(0,))
    def reset(self):
        self._state = self._reset()
        return self._state

def IteratedPrisonersDilemmaEnv(iterations: int, batch_size=1, include_remaining_iterations=True):
    return IteratedMatrixGame(
        payoff_matrix=jnp.array([[[-1,-1], [-3,0]], [[0,-3], [-2,-2]]]),
        iterations=iterations,
        batch_size=batch_size,
        include_remaining_iterations=include_remaining_iterations
    )

if __name__ == '__main__':
    env = IteratedPrisonersDilemmaEnv(batch_size=4, iterations=5)
    state = env.reset()
    for _ in range(5):
        state = env.step(np.zeros((4, 2), dtype=np.int32))
        print(state)
