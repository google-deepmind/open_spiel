import numpy as np


class OptimalStoppingGameObserver:

    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {"observation": np.array([0,0,0])}

    def set_from(self, state, player):
        if state.config.use_beliefs:
            rounded_belief_1 = round(state.b1[1],2)
        else:
            rounded_belief_1 = state.latest_obs

        if player == 1:
            intrusion_state = state.intrusion
        else:
            intrusion_state = 0
            if state.config.use_beliefs:
                intrusion_state = rounded_belief_1

        l = state.l
        self.dict = {"observation": np.array([l, rounded_belief_1, intrusion_state])}

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        if self.iig_obs_type.public_info:
            h = state.history()
            return ",".join(list(map(lambda x: str(x), h)))
        else:
            return None
