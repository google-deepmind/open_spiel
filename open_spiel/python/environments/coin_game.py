"""
Coin Game environment.
"""
import numpy as np

from pyspiel import PlayerId

from open_spiel.python.rl_environment import Environment, TimeStep, StepType


class CoinGameVec(Environment):
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]

    def __init__(self, max_steps, batch_size, grid_size=2):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size

        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]
        self.NUM_STATES = np.prod(self.ob_space_shape)
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]
        self.prng = np.random.RandomState()
        self.step_count = None

    def seed(self, seed=None):
        self.prng = np.random.RandomState(seed)

    def observation_spec(self):
        return dict(
            info_state=tuple([4*self.grid_size*self.grid_size] for _ in range(self._num_players)),
            legal_actions=tuple([self.NUM_ACTIONS for _ in range(self._num_players)]),
            current_player=()
        )

    def action_spec(self):
        return dict(
            num_actions=tuple([self._payoff_matrix.shape[p] for p in range(self._num_players)]),
            min=tuple([0 for p in range(self._num_players)]),
            max=tuple([self._payoff_matrix.shape[p] - 1 for p in range(self._num_players)]),
            dtype=int,
        )

    def reset(self):
        self.step_count = 0
        self.red_coin = self.prng.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos  = self.prng.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = self.prng.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = self.prng.randint(self.grid_size, size=2)
            self._generate_coin(i)
        state = self._generate_state()
        state = np.reshape(state, (self.batch_size, -1))
        observations = [state, state]
        return TimeStep(
            observations=dict(
                info_state=[s.astype(np.float32) for s in observations],
                legal_actions=[np.arange(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)],
                batch_size=self.batch_size,
                current_player=PlayerId.SIMULTANEOUS
            ),
            rewards=[np.zeros(self.batch_size) for _ in range(self.NUM_AGENTS)],
            discounts=[np.ones(self.batch_size) for _ in range(self.NUM_AGENTS)],
            step_type=StepType.FIRST
        )

    def _generate_coin(self, i):
        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agents
        success = 0
        while success < 2:
            success = 0
            self.coin_pos[i] = self.prng.randint(self.grid_size, size=(2))
            success = 1 - self._same_pos(self.red_pos[i],
                                          self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i],
                                          self.coin_pos[i])

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_state(self):
        state = np.zeros([self.batch_size] + self.ob_space_shape)
        for i in range(self.batch_size):
            state[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1
            state[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1
            if self.red_coin[i]:
                state[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
            else:
                state[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        return state

    def step(self, actions):
        ac0, ac1 = actions[:, 0], actions[:, 1]

        self.step_count += 1

        for j in range(self.batch_size):
            a0, a1 = ac0[j], ac1[j]
            assert a0 in {0, 1, 2, 3} and a1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = \
                (self.red_pos[j] + self.MOVES[a0]) % self.grid_size
            self.blue_pos[j] = \
                (self.blue_pos[j] + self.MOVES[a1]) % self.grid_size

        # Compute rewards
        reward_red, reward_blue = np.zeros(self.batch_size), np.zeros(self.batch_size)
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -2
                    reward_blue[i] += 1
            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1

            if generate:
                self._generate_coin(i)

        reward = [reward_red, reward_blue]
        state = self._generate_state().reshape((self.batch_size, -1))
        observations = [state, state]
        done = (self.step_count == self.max_steps)

        return TimeStep(
            observations=dict(
                info_state=observations,
                legal_actions=[np.arange(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)],
                batch_size=self.batch_size,
                current_player=PlayerId.SIMULTANEOUS
            ),
            rewards=reward,
            discounts=[np.ones(self.batch_size) * (1-done) for _ in range(self.NUM_AGENTS)],
            step_type=StepType.MID if not done else StepType.LAST
        )

if __name__ == '__main__':
    env = CoinGameVec(max_steps=10, batch_size=4, grid_size=5)
    obs = env.reset()
    while not obs.last():
        obs = env.step(np.random.randint(4, size=(4,2)))