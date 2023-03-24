import numpy as np
from pyspiel import PlayerId

from open_spiel.python.rl_environment import Environment, TimeStep, StepType


class IteratedMatrixGame(Environment):

    def __init__(self, payoff_matrix: np.ndarray, iterations: int, batch_size=1, include_remaining_iterations=True):
        self._payoff_matrix = np.array(payoff_matrix, dtype=np.float32)
        self._iterations = iterations
        self._num_players = payoff_matrix.ndim - 1
        self._batch_size = batch_size
        self._include_remaining_iterations = include_remaining_iterations
        self._t = 0
        self._actions = np.arange(np.prod(self.action_spec()['num_actions'])).reshape(*[payoff_matrix.shape[p] for p in range(self._num_players)])

    def one_hot(self, x, n):
        return np.eye(n)[x]

    @property
    def num_players(self):
        return self._num_players

    def observation_spec(self):
        return dict(
            info_state=tuple([np.prod(self._payoff_matrix.shape[:-1]) + 1 + (1 if self._include_remaining_iterations else 0)] for _ in range(self._num_players)),
            legal_actions=tuple([self._payoff_matrix.shape[p] for p in range(self._num_players)]),
            current_player=()
        )

    def action_spec(self):
        return dict(
            num_actions=tuple([self._payoff_matrix.shape[p] for p in range(self._num_players)]),
            min=tuple([0 for p in range(self._num_players)]),
            max=tuple([self._payoff_matrix.shape[p]-1 for p in range(self._num_players)]),
            dtype=int
        )

    def step(self, actions: np.ndarray):
        if actions.ndim == 1:
            actions = actions[None, :]
        payoffs = self._payoff_matrix[tuple(actions.T)]
        s1 = self.one_hot(self._actions[tuple(actions.T)] + 1, n=np.max(self._actions) + 2)
        s2 = self.one_hot(self._actions[tuple(actions[..., ::-1].T)] + 1, n=np.max(self._actions) + 2)
        rewards = [np.squeeze(p) for p in np.split(payoffs, indices_or_sections=self._num_players, axis=1)]
        discounts = [np.ones_like(r) for r in rewards]
        if self._t == self._iterations - 1:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        self._t += 1
        remaining_iters = float((self._iterations - self._t)) / self._iterations

        info_state = [s1, s2]
        if self._include_remaining_iterations:
            info_state = np.concatenate([info_state, np.full((self._batch_size, 1), fill_value=remaining_iters)],
                                        axis=-1)
        return TimeStep(
            observations=dict(
                info_state=info_state,
                legal_actions=np.array([[np.arange(self.action_spec()['num_actions'][p])] * self._batch_size for p in range(self.num_players)]),
                batch_size=actions.shape[0],
                current_player=PlayerId.SIMULTANEOUS
            ),
            rewards=rewards,
            discounts=discounts,
            step_type=step_type
        )

    def reset(self):
        self._t = 0
        info_state = np.zeros((self.num_players, self._batch_size, *self.observation_spec()["info_state"][0]))
        info_state[..., 0] = 1.0
        if self._include_remaining_iterations:
            info_state[..., -1] = 1.0
        rewards = np.squeeze(np.zeros((self.num_players, self._batch_size)))
        discounts = np.squeeze(np.ones((self.num_players, self._batch_size)))
        return TimeStep(
            observations=dict(
                info_state=[np.squeeze(s).astype(np.float32) for s in info_state],
                legal_actions=np.array([[np.arange(self.action_spec()['num_actions'][p])] * self._batch_size for p in range(self.num_players)]),
                batch_size=self._batch_size,
                current_player=PlayerId.SIMULTANEOUS
            ),
            rewards=[np.squeeze(a).astype(np.float32) for a in rewards],
            discounts=[np.squeeze(a).astype(np.float32) for a in discounts],
            step_type=StepType.FIRST
        )

def IteratedPrisonersDilemma(iterations: int, batch_size=1):
    return IteratedMatrixGame(
        payoff_matrix=np.array([[[-1,-1], [-3,0]], [[0,-3], [-2,-2]]]),
        iterations=iterations,
        batch_size=batch_size,
        include_remaining_iterations=False
    )

def IteratedMatchingPennies(iterations: int, batch_size=1):
    return IteratedMatrixGame(
        payoff_matrix=np.array([[[1,-1], [-1,1]], [[-1, 1], [1, -1]]]),
        iterations=iterations,
        batch_size=batch_size,
        include_remaining_iterations=False
    )

if __name__ == '__main__':
    env= IteratedPrisonersDilemma(iterations=10, batch_size=4)
    ts = env.reset()
    while not ts.last():
        ts = env.step(np.random.randint(0, 2, size=(4, 2)))
        print(ts)