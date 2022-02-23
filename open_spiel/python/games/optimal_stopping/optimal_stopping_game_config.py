import pyspiel

class OptimalStoppingGameConfig:

    def __init__(self, p: float = 0.001, T_max: int = 5, L: int = 3, R_ST: int = 100, R_SLA: int = 10,
                 R_COST: int = -50, R_INT: int = -100):
        self.p = p
        self.T_max= T_max
        self.L = L
        self.R_ST = R_ST
        self.R_SLA = R_SLA
        self.R_COST = R_COST
        self.R_INT = R_INT
        self.num_players = 2
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



    def params_dict(self):
        return {"p": self.p, "max_game_length": self.T_max, "L": self.L, "R_ST": self.R_ST,
                "R_SLA": self.R_SLA, "R_COST": self.R_COST, "R_INT": self.R_INT}


    def __str__(self):
        return f"p:{self.p}, T_max:{self.T_max}, L: {self.L}, R_ST:{self.R_ST}, R_SLA:{self.R_SLA}, " \
               f"R_COST:{self.R_COST}, R_INT:{self.R_INT}"