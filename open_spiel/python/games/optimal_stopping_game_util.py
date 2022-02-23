from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
from open_spiel.python.games.optimal_stopping_game_observation_type import OptimalStoppingGameObservationType


class OptimalStoppingGameUtil:

    @staticmethod
    def next_state(state: int, defender_action: int, attacker_action: int, l: int) -> int:
        """
        Computes the next state of the game given the current state and a defender action and an attacker action
        :param state:
        :param defender_action: the action of the defender
        :param attacker_action: the action of the attacker
        :param l: the number of stops remaining
        :return:
        """

        # Terminal state already
        if state == 2:
            return 2

        # Attacker aborts
        if state == 1 and attacker_action == 1:
            return 2

        # Defender final stop
        if defender_action == 1 and l == 1:
            return 2

        # Intrusion starts
        if state == 0 and attacker_action == 1:
            return 1

        # Stay in the current state
        return state

    @staticmethod
    def reward_function(state: int, defender_action: int, attacker_action: int, l: int,
                        config: OptimalStoppingGameConfig):
        """
        Computes the defender reward (negative of attacker reward)

        :param state: the state of the game
        :param defender_action: the defender action
        :param attacker_action: the attacker action
        :param l: the number of stops remaining
        :param config: the game config
        :return: the reward
        """
        # Terminal state
        if state == 2:
            return 0

        # No intrusion state
        if state == 0:
            # Continue and Wait
            if defender_action == 0 and attacker_action == 0:
                return config.R_SLA
            # Continue and Attack
            if defender_action == 0 and attacker_action == 1:
                return config.R_SLA + config.R_ST / l
            # Stop and Wait
            if defender_action == 1 and attacker_action == 0:
                return config.R_COST / config.L
            # Stop and Attack
            if defender_action == 1 and attacker_action == 1:
                return config.R_COST / config.L + config.R_ST / config.L

        # Intrusion state
        if state == 1:
            # Continue and Continue
            if defender_action == 0 and attacker_action == 0:
                return config.R_SLA + config.R_INT
            # Continue and Stop
            if defender_action == 0 and attacker_action == 1:
                return config.R_SLA
            # Stop and Continue
            if defender_action == 1 and attacker_action == 0:
                return config.R_COST / config.L + config.R_ST / l
            # Stop and Stop
            if defender_action == 1 and attacker_action == 1:
                return config.R_COST / config.L

        raise ValueError("Invalid input, s:{}, a1:{}, a2:{}".format(state, defender_action, attacker_action))


    @staticmethod
    def get_observation_chance_dist(config: OptimalStoppingGameConfig, state: int):
        """
        Computes a vector with observation probabilities for a chance node

        :param config: the game configuration
        :param state: the state of the game
        :return: a vector with tuples: (obs, prob)
        """
        if state == 0:
            return [(-1, config.p)] + [(x, (1-config.p)*config.obs_dist[i]) for i,x in enumerate(config.obs)]
        elif state == 1:
            return [(-1, config.p)] + [(x, (1-config.p)*config.obs_dist_intrusion[i]) for i,x in enumerate(config.obs)]
        else:
            raise ValueError(f"Cannot sample chance distribution if game is in the terminal state:{state}")



    @staticmethod
    def get_observation_type(obs: int) -> OptimalStoppingGameObservationType:
        """
        Returns the type of the observation

        :param obs: the observation to get the type of
        :return: observation type
        """
        if obs == -1:
            return OptimalStoppingGameObservationType.TERMINAL
        else:
            return OptimalStoppingGameObservationType.NON_TERMINAL
