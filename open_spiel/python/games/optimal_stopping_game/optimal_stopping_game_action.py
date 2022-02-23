import enum


class OptimalStoppingGameAction(enum.IntEnum):
    """
    Enum class representing the different actions in the optimal stopping game (of both the attacker and the defender)
    """
    CONTINUE = 0
    STOP = 1