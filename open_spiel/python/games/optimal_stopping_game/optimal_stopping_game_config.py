from typing import List
import numpy as np
import pyspiel
from open_spiel.python.games.optimal_stopping_game.optimal_stopping_game_action import OptimalStoppingGameAction


class OptimalStoppingGameConfig:

    def __init__(self, p: float = 0.001, T_max: int = 5, L: int = 3, R_ST: int = 100, R_SLA: int = 10,
                 R_COST: int = -50, R_INT: int = -100, obs: List = None,
                 obs_dist: List = None, obs_dist_intrusion: List = None):
        """
        DTO class representing the configuration of the optimal stopping game

        :param p: the probability that the attacker is detected at any time-step
        :param T_max: the maximum length of the game (could be infinite)
        :param L: the number of stop actions of the defender
        :param R_ST: constant for defining the reward function
        :param R_SLA: constant for defining the reward function
        :param R_COST: constant for defining the reward function
        :param R_INT: constant for defining the reward function
        :param obs: the list of observations
        :param obs_dist_intrusion: the observation distribution
        """
        assert obs is not None
        assert obs_dist is not None
        assert obs_dist_intrusion is not None
        assert len(obs) == len(obs_dist_intrusion)
        assert len(obs) == len(obs_dist)
        assert sum(obs_dist_intrusion) == 1
        assert sum(obs_dist) == 1
        self.p = p
        self.T_max= T_max
        self.L = L
        self.R_ST = R_ST
        self.R_SLA = R_SLA
        self.R_COST = R_COST
        self.R_INT = R_INT
        self.obs = np.array(obs)
        self.obs_dist = np.array(obs_dist)
        self.obs_dist_intrusion = np.array(obs_dist_intrusion)
        self.num_players = 2
        self.observation_tensor_size = 1
        self.observation_tensor_shape = self.obs[0].shape
        self.information_state_tensor_size = 2
        self.information_state_tensor_shape = (2,)
        self.actions = [OptimalStoppingGameAction.STOP, OptimalStoppingGameAction.CONTINUE]
        self.game_type = pyspiel.GameType(
            short_name="python_optimal_stopping_game",
            long_name="Python Optimal Stopping Game",
            dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.REWARDS,
            max_num_players=self.num_players,
            min_num_players=self.num_players,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
            provides_observation_string=True,
            provides_observation_tensor=True,
            provides_factored_observation_string=True,
            parameter_specification=self.params_dict())
        self.game_info = pyspiel.GameInfo(
            num_distinct_actions=len(self.actions),
            max_chance_outcomes=self.obs,
            num_players=self.num_players,
            min_utility=self.R_INT,
            max_utility=self.R_ST,
            utility_sum=0.0,
            max_game_length=self.T_max)

    def params_dict(self) -> dict:
        """
        :return: a dict representation of the object
        """
        return {"p": self.p, "T_max": self.T_max, "L": self.L, "R_ST": self.R_ST,
                "R_SLA": self.R_SLA, "R_COST": self.R_COST, "R_INT": self.R_INT, "obs": self.obs,
                "obs_dist": self.obs_dist, "obs_dist_intrusion": self.obs_dist_intrusion}

    def __str__(self) -> str:
        """
        :return: a string representation of the object
        """
        return f"p:{self.p}, T_max:{self.T_max}, L: {self.L}, R_ST:{self.R_ST}, R_SLA:{self.R_SLA}, " \
               f"R_COST:{self.R_COST}, R_INT:{self.R_INT}, observations:{self.obs}, " \
               f"obs_dist:{self.obs_dist}, obs_dist_intrusion:{self.obs_dist_intrusion}, " \
               f"actions:{self.actions}"

    @staticmethod
    def from_params_dict(params_dict: dict) -> "OptimalStoppingGameConfig":
        """
        Creates a config object from a user-supplied dict with parameters

        :param params_dict: the dict with parameters
        :return: a config object corresponding to the parameters in the dict
        """
        return OptimalStoppingGameConfig(
            p=params_dict["p"], T_max=params_dict["T_max"], L=params_dict["L"], R_ST=params_dict["R_ST"],
            R_SLA=params_dict["R_SLA"], R_COST=params_dict["R_COST"], R_INT=params_dict["R_INT"],
            obs=params_dict["obs"], obs_dist_intrusion=params_dict["obs_dist_intrusion"],
            obs_dist=params_dict["obs_dist"]
        )

    @staticmethod
    def default_params() -> dict:
        """
        :return: default parameters
        """
        return {"p": 0.001, "T_max": 5, "L": 3, "R_ST": 100, "R_SLA": 10, "R_COST": -50, "R_INT": -100,
                "obs" : [0,1,2,3,4,5,6,7,8,9],
                "obs_dist": [4/20,4/20,4/20,2/20,1/20,1/20,1/20,1/20,1/20,1/20],
                "obs_dist_intrusion": [1/20,1/20,1/20,1/20,1/20,1/20,2/20,4/20,4/20,4/20]
                }