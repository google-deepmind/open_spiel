class OptimalStoppingUtil:

    @staticmethod
    def num_players() -> int:
        return 2

    @staticmethod
    def default_params() -> dict:
        return {"termination_probability": 0.001, "max_game_length": 5, "L": 3, "R_ST": 20.0,
                "R_SLA": 5.0, "R_COST": -5.0, "R_INT": -10.0}

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
    def reward_function(state: int, defender_action: int, attacker_action: int, R_SLA: int, R_ST: int,
                        R_COST: int, L: int, R_INT: int, l: int):
        """
        Computes the defender reward (negative of attacker reward)

        :param state: the state of the game
        :param defender_action: the defender action
        :param attacker_action: the attacker action
        :param R_SLA: the R_SLA scaling constant
        :param R_ST: the R_ST scaling constant
        :param R_COST: the R_COST scaling constant
        :param L: the maximum number of stops
        :param R_INT: the R_INT scaling constant
        :param l: the number of stops remaining
        :return: the reward
        """
        # Terminal state
        if state == 2:
            return 0

        # No intrusion state
        if state == 0:
            # Continue and Wait
            if defender_action == 0 and attacker_action == 0:
                return R_SLA
            # Continue and Attack
            if defender_action == 0 and attacker_action == 1:
                return R_SLA + R_ST / l
            # Stop and Wait
            if defender_action == 1 and attacker_action == 0:
                return R_COST / L
            # Stop and Attack
            if defender_action == 1 and attacker_action == 1:
                return R_COST / L + R_ST / L

        # Intrusion state
        if state == 1:
            # Continue and Continue
            if defender_action == 0 and attacker_action == 0:
                return R_SLA + R_INT
            # Continue and Stop
            if defender_action == 0 and attacker_action == 1:
                return R_SLA
            # Stop and Continue
            if defender_action == 1 and attacker_action == 0:
                return R_COST / L + R_ST / l
            # Stop and Stop
            if defender_action == 1 and attacker_action == 1:
                return R_COST / L

        raise ValueError("Invalid input, s:{}, a1:{}, a2:{}".format(state, defender_action, attacker_action))
