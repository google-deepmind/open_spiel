import numpy as np
from games.optimal_stopping.optimal_stopping_game_config import OptimalStoppingGameConfig
import pyspiel

class OptimalStoppingGameState(pyspiel.State):
    """Current state of the game."""

    def __init__(self, game, config: OptimalStoppingGameConfig):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.config = config
        self.current_iteration = 1
        self.is_chance = False
        self.game_over = False
        self.rewards = np.zeros(config.num_players)
        self.returns = np.zeros(config.num_players)
        self.intrusion = 0
        self.l = config.L
        self.latest_obs = Chance.OBS_0

    def observation_tensor(self, player):
        return [1,1]

    def information_state_tensor(self, player):
        return [1,1]

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every simultaneous-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self._is_chance:
            return pyspiel.PlayerId.CHANCE
        else:
            return pyspiel.PlayerId.SIMULTANEOUS

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        return [Action.CONTINUE, Action.STOP]

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self._is_chance
        if self.intrusion == 0:
            return [(Chance.TERMINAL, self._termination_probability),
                    (Chance.OBS_0, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_1, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_2, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_3, 2 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_4, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_5, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_6, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_7, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_8, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_9, 1 * (1 - self._termination_probability) / 20)
                    ]
        else:
            return [(Chance.TERMINAL, self._termination_probability),
                    (Chance.OBS_0, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_1, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_2, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_3, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_4, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_5, 1 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_6, 2 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_7, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_8, 4 * (1 - self._termination_probability) / 20),
                    (Chance.OBS_9, 4 * (1 - self._termination_probability) / 20)
                    ]

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        # This is not called at simultaneous-move states.
        assert self._is_chance and not self._game_over
        self._current_iteration += 1
        self._is_chance = False
        self._game_over = (action == Chance.TERMINAL)
        if self._current_iteration > self.get_game().max_game_length():
            self._game_over = True

    def _apply_actions(self, actions):
        """Applies the specified actions (per player) to the state."""
        assert not self._is_chance and not self._game_over
        self._is_chance = True
        self._current_iteration += 1
        if self._current_iteration > self.get_game().max_game_length():
            self._game_over = True

        r = OptimalStoppingUtil.reward_function(s=self.intrusion, a1=actions[0], a2=actions[1], R_SLA=self.R_SLA,
                                                R_ST=self.R_ST, R_COST=self.R_COST, L=self.L, R_INT=self.R_INT, l=self.l)
        self._rewards[0] = r
        self._rewards[1] = -r
        self._returns += self._rewards

        s_prime = OptimalStoppingUtil.next_state(s=self.intrusion, a1=actions[0], a2=actions[1], l=self.l)

        # Game ended
        if s_prime == 2:
            self._game_over = True
        else:
            self.intrusion = s_prime

        # Decrement stops left
        if actions[0] == 1:
            self.l -= 1

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return Chance(action).name
        else:
            return Action(action).name

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def rewards(self):
        """Reward at the previous step."""
        return self._rewards

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self._returns

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return (f"p0:{self.action_history_string(0)} "
                f"p1:{self.action_history_string(1)}")

    def action_history_string(self, player):
        return "".join(
            self._action_to_string(pa.player, pa.action)[0]
            for pa in self.full_history()
            if pa.player == player)