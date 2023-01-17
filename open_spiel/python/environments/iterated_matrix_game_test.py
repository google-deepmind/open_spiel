import numpy as np
from absl.testing import absltest
from open_spiel.python.rl_environment import StepType

from open_spiel.python.environments.iterated_matrix_game import IteratedMatrixGame

class IteratedMatrixGameTest(absltest.TestCase):

    def test_obs_spec(self):
        # Tests different number of actions for 3 players.
        # Player 0 has 2 actions, player 1 has 4 actions, player 2 has 3 actions.
        three_player_game = np.array([
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        ])

        env = IteratedMatrixGame(three_player_game, iterations=5, batch_size=4, include_remaining_iterations=True)
        obs_specs = env.observation_spec()
        self.assertLen(obs_specs['info_state'], 3) # 3 players
        num_actions = [2, 4, 3]
        for i in range(3):
            self.assertEqual(obs_specs['info_state'][i][0], np.sum(num_actions) + 1)
            self.assertEqual(obs_specs['legal_actions'][i], num_actions[i])


    def test_action_spec(self):
        three_player_game = np.array([
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        ])

        env = IteratedMatrixGame(three_player_game, iterations=5, batch_size=4, include_remaining_iterations=True)
        action_specs = env.action_spec()
        num_actions = [2, 4, 3]
        for i, n_a in enumerate(action_specs['num_actions']):
            self.assertEqual(n_a, num_actions[i])

    def test_reset(self):
        payoff = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        env = IteratedMatrixGame(payoff, iterations=5, batch_size=4, include_remaining_iterations=True)
        timestep = env.reset()
        self.assertEqual(timestep.step_type, StepType.FIRST)
        self.assertLen(timestep.observations['info_state'], env.num_players)
        self.assertEqual(timestep.observations['info_state'][0].shape, (4, 2+2+1)) # batch_size, 2 actions + 2 actions + 1
        for i in range(env.num_players):
            self.assertTrue(np.all(timestep.observations['info_state'][i][..., :-1] == 0))
            self.assertTrue(np.all(timestep.observations['info_state'][i][..., -1] == 1))

    def test_step(self):
        payoff = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        actions = [[0, 0], [0, 1], [1,0], [1, 1]]
        env = IteratedMatrixGame(payoff, iterations=len(actions), batch_size=1, include_remaining_iterations=True)
        timestep = env.reset()
        for a, b in actions:
            timestep = env.step(np.array([a, b]))
            self.assertTrue(np.all(np.equal(timestep.rewards, payoff[a, b])))
        self.assertEqual(timestep.step_type, StepType.LAST)