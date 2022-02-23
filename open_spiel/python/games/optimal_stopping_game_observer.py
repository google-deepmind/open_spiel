import numpy as np


class OptimalStoppingGameObserver:

    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {"observation": np.array([0])}

    def set_from(self, state, player):
        if state.config.use_beliefs:
            rounded_belief = round(state.b1[1],2)
            self.dict = {"observation": np.array([rounded_belief])}
        else:
            self.dict = {"observation": np.array([state.latest_obs])}

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        if self.iig_obs_type.public_info:
            return (f"us:{state.action_history_string(player)} "
                    f"op:{state.action_history_string(1 - player)}")
        else:
            return None