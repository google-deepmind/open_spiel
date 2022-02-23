import enum


class OptimalStoppingGameStateType(enum.IntEnum):
    NO_INTRUSION = 0
    INTRUSION = 1
    TERMINAL = 2