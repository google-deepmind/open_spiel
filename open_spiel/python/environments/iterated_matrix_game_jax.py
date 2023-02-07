from functools import partial
from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.rl_environment import TimeStep, StepType


class IteratedMatrixGame(NamedTuple):
    reset: Callable[[], TimeStep]
    step: Callable[[TimeStep, jnp.ndarray], TimeStep]
    num_players: int
    observation_spec: Callable[[], dict]
    action_spec: Callable[[], dict]


def make_env_fns(payoff_matrix: jnp.ndarray, iterations: int, batch_size=1):
    num_players = payoff_matrix.ndim - 1
    actions = [payoff_matrix.shape[p] for p in range(num_players)]
    num_actions = np.prod(actions).item()
    cases = jnp.arange(num_actions) + 1
    cases = jnp.reshape(cases, actions)
    indices = jnp.eye(num_actions + 1)
    initial_obs = {
        'info_state': [indices[jnp.zeros(batch_size, dtype=jnp.int32)]] * num_players,
        'legal_actions': np.array([[np.arange(actions[p])] * batch_size for p in range(num_players)]),
        'current_player': -2,
        'batch_size': batch_size,
        't': 0
    }
    payoffs = jnp.array(payoff_matrix, dtype=jnp.float32)

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
                't': t + 1,
                'batch_size': batch_size
            },
            rewards=rewards.T,
            discounts=discounts,
            step_type=jax.lax.select(t < iterations - 1, StepType.MID, StepType.LAST)
        )

    def reset() -> TimeStep:
        return TimeStep(
            observations=initial_obs,
            rewards=jnp.zeros(num_players),
            discounts=jnp.ones(num_players),
            step_type=0
        )

    # return step, reset
    return jax.jit(step), reset



def IteratedPrisonersDilemma(iterations: int, batch_size=1) -> IteratedMatrixGame:
    step, reset = make_env_fns(
        payoff_matrix=jnp.array([[[-1, -1], [-3, 0]], [[0, -3], [-2, -2]]]),
        iterations=iterations,
        batch_size=batch_size
    )
    return IteratedMatrixGame(
        step=step,
        reset=reset,
        num_players=2,
        action_spec=lambda: dict(
            num_actions=[2,2],
            min=[0,0],
            max=[1,1],
            dtype=int,
        ),
        observation_spec=lambda: dict(
            info_state=[5,5],
            legal_actions=[2,2],
            current_player=()
        )
    )


if __name__ == '__main__':
    env = IteratedPrisonersDilemma(iterations=10, batch_size=4)
    step = env.reset()
    step = env.step(state=step, action=jnp.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
    step = env.step(state=step, action=jnp.array([[0, 1], [1, 0], [1, 1], [0, 0]]))
    step = env.step(state=step, action=jnp.array([[0, 1], [1, 0], [1, 1], [0, 0]]))
    step = env.step(state=step, action=jnp.array([[0, 1], [1, 0], [1, 1], [0, 0]]))
    step = env.step(state=step, action=jnp.array([[0, 1], [1, 0], [1, 1], [0, 0]]))

