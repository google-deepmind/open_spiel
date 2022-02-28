from typing import List,Tuple
import numpy as np
import random
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_action import OptimalStoppingGameAction
from open_spiel.python.games.optimal_stopping_game_util import OptimalStoppingGameUtil
from open_spiel.python.games.optimal_stopping_game_observation_type import OptimalStoppingGameObservationType
import pyspiel


class OptimalStoppingGameState(pyspiel.State):

    def __init__(self, game, config: OptimalStoppingGameConfig):
        """
        Initializes the game state

        :param game: the optimal stopping game
        :param config: the game config
        """
        super().__init__(game)
        self.config = config
        self.current_iteration = 1
        self.is_chance = False
        self.game_over = False
        self._rewards = np.zeros(config.num_players)
        self._returns = np.zeros(config.num_players)
        self.intrusion = 0
        self.l = config.L
        self.latest_obs = 0
        self.latest_actions = []
        self.b1 = config.initial_belief

    def current_player(self):
        """
        Method to conform to PySpiel's API

        :return: the player that will move next
        """
        if self.game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self.is_chance:
            return pyspiel.PlayerId.CHANCE
        else:
            return pyspiel.PlayerId.SIMULTANEOUS

    def _legal_actions(self, player):
        """
        Method to conform to PySpiel's API

        :param player: the player to get the legal actions of
        :return: a list of legal actions, sorted in ascending order.
        """
        # Since actions of attacker and defender are the same, return same list always.
        return [OptimalStoppingGameAction.CONTINUE, OptimalStoppingGameAction.STOP]

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        """
        Method to follow pyspiel's API

        :return: the possible chance outcomes and their probabilities
        """
        assert self.is_chance
        if self.game_over:
            s = 2
        else:
            s = self.intrusion
        return self.config.obs_prob_chance_dists[s]

    def _apply_action(self, obs: int) -> None:
        """
        Applies a chance node observation to the state (Method to conform to PySpiel's API)

        :param obs: the observation
        :return: None
        """
        assert self.is_chance and not self.game_over
        self.current_iteration += 1
        self.is_chance = False
        self.game_over = (self.game_over or (OptimalStoppingGameUtil.get_observation_type(obs=obs, config=self.config)
                           == OptimalStoppingGameObservationType.TERMINAL))
        if self.current_iteration > self.get_game().max_game_length():
            self.game_over = True
        self.latest_obs = obs
        if self.config.use_beliefs and not self.game_over:
            # TODO Fix this
            pi_2 = [
                [0.5,0.5],
                [0.5,0.5],
                [0.5,0.5]
            ]
            self.b1 = OptimalStoppingGameUtil.next_belief(o=obs, a1=self.latest_actions[0], pi_2=pi_2, b=self.b1,
                                                          config=self.config, l=self.l)

        # Decrement stops left. This has to be done after belief update.
        if self.latest_actions[0] == 1:
            self.l -= 1

    def _apply_actions(self, actions : List[int]) -> None:
        """
        Apply defender and attacker actions at a simultaneous-move node in the game (Method to conform to PySpiel's API)

        :param actions: the list of actions to apply
        :return: None
        """
        assert not self.is_chance and not self.game_over
        self.latest_actions = actions

        # Compute reward
        r = OptimalStoppingGameUtil.reward_function(state=self.intrusion, defender_action=actions[0],
                                                    attacker_action=actions[1], l=self.l, config=self.config)
        self._rewards[0] = r
        self._rewards[1] = -r
        self._returns += self._rewards

        # Compute next state
        s_prime = OptimalStoppingGameUtil.next_state(state=self.intrusion, defender_action=actions[0],
                                                     attacker_action=actions[1], l=self.l)
        if s_prime == 2:
            self.game_over = True
        else:
            self.intrusion = s_prime

        self.current_iteration += 1
        # Check if game has ended
        if self.current_iteration > self.get_game().max_game_length():
            self.game_over = True

        # Sample random detection
        if random.random() <= self.config.p:
            self.game_over = True

        # If game did not end, next node will be a chance node
        if not self.game_over:
            self.is_chance = True

    def _action_to_string(self, player: pyspiel.PlayerId, action: int) -> str:
        """
        Method to conform to PySpiel's API

        :param player: the player that took the action
        :param action: the action
        :return: a string representation of an action
        """
        if player == pyspiel.PlayerId.CHANCE:
            return OptimalStoppingGameUtil.get_observation_type(obs=action, config=self.config).name
        else:
            return OptimalStoppingGameAction(action).name

    def is_terminal(self) -> bool:
        """
        Method to conform to PySpiel's API

        :return: True if the game has ended
        """
        return self.game_over

    def rewards(self) -> np.ndarray:
        """
        Method to conform to PySpiel's API

        :return: rewards at the previous step
        """
        return self._rewards

    def returns(self) -> np.ndarray:
        """
        Method to conform to PySpiel's API

        :return: Total reward for each player over the course of the game so far.
        """
        return self._returns

    def __str__(self):
        """
        Method to conform to PySpiel's API

        :return: String for debug purposes. No particular semantics are required.
        """
        return (f"p0_history:{self.action_history_string(0)}, "
                f"p1_history:{self.action_history_string(1)}, l:{self.l}, belief:{self.b1}, intrusion: {self.intrusion}")

    def action_history_string(self, player):
        """
        Method to conform to PySpiel's API

        :param player: the player to get the history of
        :return: String representation of the history of actions of a given player
        """
        return "".join(
            self._action_to_string(pa.player, pa.action)[0]
            for pa in self.full_history()
            if pa.player == player)

