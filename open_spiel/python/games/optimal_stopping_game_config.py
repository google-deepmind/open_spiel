from typing import List
import numpy as np
import pyspiel
from open_spiel.python.games.optimal_stopping_game_action import OptimalStoppingGameAction


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
        assert round(sum(obs_dist_intrusion),2) == 1
        assert round(sum(obs_dist),2) == 1
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
        self.params = self.params_dict()
        self.game_type = self.create_game_type()
        self.game_info = self.create_game_info()

    def create_game_type(self) -> pyspiel.GameType:
        """
        :return: GameType object
        """
        return pyspiel.GameType(
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
            parameter_specification=self.params)

    def create_game_info(self) -> pyspiel.GameInfo:
        """
        :return: GameInfo object
        """
        return pyspiel.GameInfo(
            num_distinct_actions=len(self.actions),
            max_chance_outcomes=len(self.obs),
            num_players=self.num_players,
            min_utility=self.R_INT,
            max_utility=self.R_ST,
            utility_sum=0.0,
            max_game_length=self.T_max)

    def params_dict(self) -> dict:
        """
        :return: a dict representation of the object
        """
        d = {}
        d["p"] = self.p
        d["T_max"] = self.T_max
        d["L"] = self.L
        d["R_ST"] = self.R_ST
        d["R_SLA"] = self.R_SLA
        d["R_COST"] = self.R_COST
        d["R_INT"] = self.R_INT
        d["obs"] = ",".join(list(map(lambda x: str(x), self.obs.tolist())))
        d["obs_dist"] = ",".join(list(map(lambda x: str(x), self.obs_dist.tolist())))
        d["obs_dist_intrusion"] = ",".join(list(map(lambda x: str(x), self.obs_dist_intrusion.tolist())))
        return d

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
            obs=list(map(lambda x: int(x), params_dict["obs"].split(","))),
            obs_dist_intrusion=list(map(lambda x: float(x), params_dict["obs_dist_intrusion"].split(","))),
            obs_dist=list(map(lambda x: float(x), params_dict["obs_dist"].split(",")))
        )

    @staticmethod
    def default_params() -> dict:
        """
        :return: default parameters
        """
        d = {}
        d["p"] = 0.001
        d["T_max"] = 5
        d["L"] = 3
        d["R_ST"] = 100
        d["R_SLA"] = 10
        d["R_COST"] = -50
        d["R_INT"] = -100
        d["obs"] = ",".join(list(map(lambda x: str(x),[0,1,2,3,4,5,6,7,8,9])))
        d["obs_dist"] = ",".join(list(map(lambda x: str(x),[4/20,4/20,4/20,2/20,1/20,1/20,1/20,1/20,1/20,1/20])))
        d["obs_dist_intrusion"] = ",".join(list(map(lambda x: str(x),[1/20,1/20,1/20,1/20,1/20,1/20,2/20,4/20,4/20,4/20])))
        return d


    def __getstate__(self) -> dict:
        """
        Serialize the object

        :return: dict state
        """

        # start with a copy so we don't accidentally modify the object state
        # or cause other conflicts
        state = self.__dict__.copy()

        # remove unpicklable entries
        del state["game_type"]
        del state["game_info"]
        return state

    def __setstate__(self, state) -> None:
        """
        Deserialize the object

        :param state: the state
        :return: None
        """
        # restore the state which was picklable
        self.__dict__.update(state)

        # restore unpicklable entries
        self.game_type = self.create_game_type()
        self.game_info = self.create_game_info()