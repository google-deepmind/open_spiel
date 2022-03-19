from typing import List
import numpy as np
import pyspiel
from open_spiel.python.games.optimal_stopping_game_action import OptimalStoppingGameAction
from open_spiel.python.games.optimal_stopping_game_state_type import OptimalStoppingGameStateType


class OptimalStoppingGameConfigSequential:


    def __init__(self, p: float = 0.001, T_max: int = 5, L: int = 3, R_ST: int = 100, R_SLA: int = 10,
                 R_COST: int = -50, R_INT: int = -100, obs: str = "",
                 obs_dist: str = "", obs_dist_intrusion: str = "", initial_belief: str = "", use_beliefs: bool = False):
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
        :param initial_belief: the initial belief
        :param use_beliefs: boolean flag whether to use beliefs or not. If this is false, use observations instead.
        """
        assert obs is not None and obs != ""
        assert obs_dist is not None and obs_dist != ""
        assert obs_dist_intrusion is not None and obs_dist_intrusion != ""
        assert initial_belief is not None and initial_belief != ""
        self.obs_str = obs
        self.obs_dist_str = obs_dist
        self.obs_dist_intrusion_str = obs_dist_intrusion
        self.initial_belief_str = initial_belief
        obs = list(map(lambda x: int(x), obs.split(" ")))
        obs_dist = list(map(lambda x: float(x), obs_dist.split(" ")))
        obs_dist_intrusion = list(map(lambda x: float(x), obs_dist_intrusion.split(" ")))
        initial_belief = list(map(lambda x: float(x), initial_belief.split(" ")))
        assert len(obs) == len(obs_dist_intrusion)
        assert len(obs) == len(obs_dist)
        assert round(sum(obs_dist_intrusion),2) == 1
        assert round(sum(obs_dist),2) == 1
        assert round(sum(initial_belief),2) == 1
        self.p = p
        self.T_max= T_max
        self.L = L
        self.R_ST = R_ST
        self.R_SLA = R_SLA
        self.R_COST = R_COST
        self.R_INT = R_INT
        self.use_beliefs = use_beliefs
        self.obs = obs
        self.obs_dist = obs_dist
        self.obs_dist_intrusion = obs_dist_intrusion
        self.obs_dist_terminal = np.zeros(len(self.obs))
        self.obs_dist_terminal[-1] = 1
        self.initial_belief = initial_belief
        self.num_players = 2
        self.observation_tensor_size = 3
        self.observation_tensor_shape = (3,)
        self.information_state_tensor_size = 3
        self.information_state_tensor_shape = (3,)
        self.params = self.params_dict()
        self.A1 = list(map(lambda x: x.value, self.get_actions()))
        self.A2 = list(map(lambda x: x.value, self.get_actions()))
        self.S = list(map(lambda x: x.value, self.get_states()))
        self.O = self.obs.copy()
        self.Z = self.observation_tensor()
        self.T = []
        self.R = []

        for l in range(0, self.L):
            self.T.append(self.transition_tensor(l=l+1).tolist())
            self.R.append(self.reward_tensor(l=l+1).tolist())
        self.T = np.array(self.T)
        self.R = np.array(self.R)
        obs_prob_chance_dists = []
        for s in self.S:
            obs_prob_chance_dists.append(self.get_observation_chance_dist(state=s))
        self.obs_prob_chance_dists = obs_prob_chance_dists

    def get_observation_chance_dist(self, state: int):
        """
        Computes a vector with observation probabilities for a chance node

        :param config: the game configuration
        :param state: the state of the game
        :return: a vector with tuples: (obs, prob)
        """
        if state == 0:
            return  [(x, self.obs_dist[i]) for i,x in enumerate(self.obs[0:-1])]
        elif state == 1:
            return [(x, self.obs_dist_intrusion[i]) for i,x in enumerate(self.obs[0:-1])]
        elif state == 2:
            return [(self.obs[-1], self.obs_dist_terminal[-1])]
        else:
            raise ValueError(f"Invalid state:{state}")


    def get_actions(self) -> List[int]:
        """
        Get the actions in the game. The actions are the same for both players

        :return: a list with the actions
        """
        return [OptimalStoppingGameAction.CONTINUE, OptimalStoppingGameAction.STOP]

    def get_states(self) -> List[int]:
        """
        Get the states of the game

        :return: a list with the states
        """
        return [OptimalStoppingGameStateType.NO_INTRUSION,OptimalStoppingGameStateType.INTRUSION,
                OptimalStoppingGameStateType.TERMINAL]

    def create_game_type(self) -> pyspiel.GameType:
        """
        :return: GameType object
        """
        return pyspiel.GameType(
            short_name="python_optimal_stopping_game_sequential",
            long_name="Python Optimal Stopping Game Sequential",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
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
            num_distinct_actions=len(self.get_actions()),
            max_chance_outcomes=len(self.obs) + 1,
            num_players=self.num_players,
            min_utility=self.R_INT*10,
            max_utility=self.R_ST*10,
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
        d["obs"] = self.obs_str
        d["obs_dist"] = self.obs_dist_str
        d["obs_dist_intrusion"] = self.obs_dist_intrusion_str
        d["initial_belief"] = self.initial_belief_str
        d["use_beliefs"] = self.use_beliefs
        return d

    def __str__(self) -> str:
        """
        :return: a string representation of the object
        """
        return f"p:{self.p}, T_max:{self.T_max}, L: {self.L}, R_ST:{self.R_ST}, R_SLA:{self.R_SLA}, " \
               f"R_COST:{self.R_COST}, R_INT:{self.R_INT}, observations:{self.obs}, " \
               f"obs_dist:{self.obs_dist}, obs_dist_intrusion:{self.obs_dist_intrusion}, " \
               f"actions:{self.get_actions()}, initial_belief:{self.initial_belief}, use_beliefs:{self.use_beliefs}"


    @staticmethod
    def from_params_dict(params_dict: dict) -> "OptimalStoppingGameConfigSequential":
        """
        Creates a config object from a user-supplied dict with parameters

        :param params_dict: the dict with parameters
        :return: a config object corresponding to the parameters in the dict
        """
        return OptimalStoppingGameConfigSequential(
            p=params_dict["p"], T_max=params_dict["T_max"], L=params_dict["L"], R_ST=params_dict["R_ST"],
            R_SLA=params_dict["R_SLA"], R_COST=params_dict["R_COST"], R_INT=params_dict["R_INT"],
            obs=params_dict["obs"],
            obs_dist_intrusion=params_dict["obs_dist_intrusion"],
            obs_dist=params_dict["obs_dist"], initial_belief=params_dict["initial_belief"],
            use_beliefs=params_dict["use_beliefs"]
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
        d["obs"] = " ".join(list(map(lambda x: str(x),[0,1,2,3,4,5,6,7,8,9,10])))
        d["obs_dist"] = " ".join(list(map(lambda x: str(x),[4/20,4/20,4/20,2/20,1/20,1/20,1/20,1/20,1/20,1/20,0])))
        d["obs_dist_intrusion"] = " ".join(list(map(lambda x: str(x),[1/20,1/20,1/20,1/20,1/20,1/20,2/20,4/20,4/20,4/20,0])))
        d["initial_belief"] = " ".join(list(map(lambda x: str(x),[1,0,0])))
        d["use_beliefs"] = False
        return d


    def observation_tensor(self) -> np.ndarray:
        """
        :return:  a |A1|x|A2|x|S|x|O| tensor
        """
        Z = []
        for _ in self.A1:
            a1_obs_probs = []
            for _ in self.A2:
                a1_a2_obs_probs = []
                a1_a2_obs_probs.append(self.obs_dist)
                a1_a2_obs_probs.append(self.obs_dist_intrusion)
                a1_a2_obs_probs.append(self.obs_dist_terminal)
                a1_obs_probs.append(a1_a2_obs_probs)
            Z.append(a1_obs_probs)
        return np.array(Z)

    def reward_tensor(self, l: int) -> np.ndarray:
        """
        :param l: the number of stops remaining
        :return: a |A1|x|A2|x|S| tensor
        """
        R = np.array(
            [
                # Defender continues
                [
                    # Attacker stops
                    [self.R_SLA, self.R_SLA, 0],
                    # Attacker continues
                    [self.R_SLA, self.R_SLA + self.R_INT, 0]
                ],
                # Defender stops
                [
                    # Attacker stops
                    [self.R_COST/self.L + self.R_ST/l, self.R_COST/self.L, 0],
                    # Attacker continues
                    [self.R_COST/self.L, self.R_COST/self.L + self.R_ST/l, 0]
                ]
            ]
        )
        return R

    def transition_tensor(self, l: int) -> np.ndarray:
        """
        :param l: the number of stops remaining
        :return: a |A1|x|A2||S|^2 tensor
        """
        if l == 1:
            return np.array(
                [
                    # Defender continues
                    [
                        # Attacker continues
                        [
                            [1-self.p, 0, self.p], # No intrusion
                            [0, 1-self.p, self.p], # Intrusion
                            [0, 0, 1] # Terminal
                        ],
                        # Attacker stops
                        [
                            [0, 1-self.p, self.p], # No intrusion
                            [0, 0, 1], # Intrusion
                            [0, 0, 1]  # Terminal
                        ]
                    ],

                    # Defender stops
                    [
                        # Attacker continues
                        [
                            [0, 0, 1], # No intrusion
                            [0, 0, 1], # Intrusion
                            [0, 0, 1] # Terminal
                        ],
                        # Attacker stops
                        [
                            [0, 0, 1], # No Intrusion
                            [0, 0, 1], # Intrusion
                            [0, 0, 1] # Terminal
                        ]
                    ]
                ]
            )
        else:
            return np.array(
                [
                    # Defender continues
                    [
                        # Attacker continues
                        [
                            [1-self.p, 0, self.p], # No intrusion
                            [0, 1-self.p, self.p], # Intrusion
                            [0, 0, 1] # Terminal
                        ],
                        # Attacker stops
                        [
                            [0, 1-self.p, self.p], # No intrusion
                            [0, 0, 1], # Intrusion
                            [0, 0, 1]  # Terminal
                        ]
                    ],

                    # Defender stops
                    [
                        # Attacker continues
                        [
                            [1-self.p, 0, self.p], # No intrusion
                            [0, 1-self.p, self.p], # Intrusion
                            [0, 0, 1] # Terminal
                        ],
                        # Attacker stops
                        [
                            [0, 1-self.p, self.p], # No Intrusion
                            [0, 0, 1], # Intrusion
                            [0, 0, 1] # Terminal
                        ]
                    ]
                ]
            )