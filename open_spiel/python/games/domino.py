import copy
import itertools

import numpy as np

import pyspiel

_NUM_PLAYERS = 2

# The first player to play is the one holding the highest rank tile.
# The rank of tiles is the following:
#   1. Highest double.
#   2. If none of the players hold a double, then highest weight.
#   3. If the highest weighted tile of both players has the same weight
#      then the highest single edge of the highest weighted tile.

# full deck sorted by rank:
_DECK = frozenset([(6., 6.), (5., 5.), (4., 4.), (3., 3.), (2., 2.), (1., 1.), (0., 0.),
                   (5., 6.),
                   (4., 6.),
                   (3., 6.), (4., 5.),
                   (2., 6.), (3., 5.),
                   (1., 6.), (2., 5.), (3., 4.),
                   (0., 6.), (1., 5.), (2., 4.),
                   (0., 5.), (1., 4.), (2., 3.),
                   (0., 4.), (1., 3.),
                   (0., 3.), (1., 2.),
                   (0., 2.),
                   (0., 1.)])

_HAND_SIZE = 7

_GAME_TYPE = pyspiel.GameType(
    short_name="python_domino",
    long_name="Python domino",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=8,
    max_chance_outcomes=len(_DECK),
    min_utility=-69,
    max_utility=69,
    # first player hand: (6,6) (6,5) (5,5) (6,4) (4,5) (6,3) (4,4) , second player hand is empty. can be reduced.
    num_players=_NUM_PLAYERS,
    max_game_length=28,  # deal: 14 chance nodes + play: 14 player nodes
    utility_sum=0.0)


class Action:
    """ represent player possible action """

    def __init__(self, tile_to_put, pip_to_play_on, player, edges):
        self.tile_to_put = tile_to_put
        self.open_pip = pip_to_play_on
        self.player = player
        self.edges = edges
        self.new_edges = self.edges_after_action()

    def edges_after_action(self):
        new_edges = []
        if len(self.edges) == 0:  # first tile on board
            new_edges.append(self.tile_to_put[0])
            new_edges.append(self.tile_to_put[1])
        else:
            edge_to_stay = self.edges[0] if self.edges[0] != self.open_pip else self.edges[1]
            new_edge = self.tile_to_put[0] if self.tile_to_put[0] != self.open_pip else self.tile_to_put[1]
            new_edges.append(edge_to_stay)
            new_edges.append(new_edge)

        new_edges.sort()
        return new_edges

    def __str__(self):
        return f'p{self.player} tile:{self.tile_to_put} pip:{self.open_pip} new_edges:{self.new_edges}'

    def __repr__(self):
        return self.__str__()


class DominoGame(pyspiel.Game):
    """A Python version of Domino Block."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return DominoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return DominoObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


class DominoState(pyspiel.State):
    """A python version of the Domino state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.gameHistory = []
        self.open_edges = []
        self.player_legal_actions = []
        self.hands = [[], []]
        self.deck = copy.deepcopy(list(_DECK))
        self._game_over = False
        self._next_player = pyspiel.PlayerId.CHANCE

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every sequential-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif len(self.player_legal_actions) == 0:
            return pyspiel.PlayerId.CHANCE
        else:
            return self._next_player

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0
        assert player == self._next_player
        return list(range(0, len(self.player_legal_actions)))

    def get_legal_actions(self, player):
        """Returns a list of legal actions."""
        assert player >= 0

        actions = []
        hand = self.hands[player]
        # first move, no open edges
        if len(self.open_edges) == 0:
            for tile in hand:
                actions.append(Action(tile, None, player, []))
            return actions

        for tile in hand:
            if tile[0] in self.open_edges:
                actions.append(Action(tile, tile[0], player, self.open_edges))
            if tile[0] != tile[1] and tile[1] in self.open_edges:
                actions.append(Action(tile, tile[1], player, self.open_edges))

        return actions

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self.is_chance_node()
        p = 1.0 / len(self.deck)
        return [(i, p) for i in range(len(self.deck))]

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        if self.is_chance_node():
            hand_to_add_tile = self.hands[0] if len(self.hands[0]) != _HAND_SIZE else self.hands[1]
            hand_to_add_tile.append(self.deck.pop(action))

            if not len(self.hands[0]) == len(self.hands[1]) == _HAND_SIZE:
                return  # another tile to deal
            # check which hand is playing first, and assigned it to player 0
            hand0_starting_value = max(map(lambda t: list(_DECK).index(t), self.hands[0]))
            hand1_starting_value = max(map(lambda t: list(_DECK).index(t), self.hands[1]))
            staring_hand = 0 if hand0_starting_value > hand1_starting_value else 1
            if staring_hand == 1:
                self.hands[0], self.hands[1] = self.hands[1], self.hands[0]

            self.hands[0].sort()
            self.hands[1].sort()

            self._next_player = 0
            # calc all possible move for the first player to play
            self.player_legal_actions = self.get_legal_actions(self._next_player)
        else:
            action = self.player_legal_actions[action]
            self.gameHistory.append(action)
            my_idx = action.player
            my_hand = self.hands[my_idx]
            my_hand.remove(action.tile_to_put)
            self.open_edges = action.new_edges

            if not my_hand:
                self._game_over = True  # player played his last tile
                return

            opp_idx = 1 - my_idx
            opp_legal_actions = self.get_legal_actions(opp_idx)

            if opp_legal_actions:
                self._next_player = opp_idx
                self.player_legal_actions = opp_legal_actions
                return

            my_legal_actions = self.get_legal_actions(my_idx)
            if my_legal_actions:
                self._next_player = my_idx
                self.player_legal_actions = my_legal_actions
                return

            self._game_over = True  # both players are blocked

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal {self.deck[action]}"
        return str(self.player_legal_actions[action])

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def returns(self):
        """Total reward for each player over the course of the game so far."""

        if not self.is_terminal():
            return [0, 0]

        sum_of_pips0 = sum(t[0] + t[1] for t in self.hands[0])
        sum_of_pips1 = sum(t[0] + t[1] for t in self.hands[1])

        if sum_of_pips1 == sum_of_pips0:
            return [0, 0]

        if sum_of_pips1 > sum_of_pips0:
            return [sum_of_pips1, -sum_of_pips1]
        return [-sum_of_pips0, sum_of_pips0]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        hand0 = [str(c) for c in self.hands[0]]
        hand1 = [str(c) for c in self.hands[1]]
        history = [str(a) for a in self.gameHistory]
        s = f'hand0:{hand0} hand1:{hand1} history:{history}'
        return s


class DominoObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")

        # Determine which observation pieces we want to include.
        pieces = [("player", 2, (2,))]

        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("hand", 21, (7, 3)))

        if iig_obs_type.public_info:
            if iig_obs_type.perfect_recall:
                pieces.append(("history", 84, (14, 6)))
            else:
                pieces.append(("last_move", 6, (6,)))
                pieces.append(("hand_sizes", 2, (2,)))

        # Build the single flat tensor.
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)

        # Build the named & reshaped views of the bits of the flat tensor.
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""

        self.tensor.fill(0)

        if "player" in self.dict:
            self.dict["player"][player] = 1
            self.dict["player"][1 - player] = 0

        if "hand_sizes" in self.dict:
            my_hand_size = len(state.hands[player])
            opp_hand_size = len(state.hands[1 - player])
            self.dict["hand_sizes"][0] = my_hand_size
            self.dict["hand_sizes"][1] = opp_hand_size

        if "edges" in self.dict:
            if state.open_edges:
                self.dict["edges"][0] = state.open_edges[0]
                self.dict["edges"][1] = state.open_edges[1]
            else:
                self.dict["edges"][0] = 0.
                self.dict["edges"][1] = 0.

        if "hand" in self.dict:
            for i, tile in enumerate(state.hands[player]):
                self.dict["hand"][i][0] = tile[0]
                self.dict["hand"][i][1] = tile[1]
                self.dict["hand"][i][2] = 1.


        if "history" in self.dict:
            for i, action in enumerate(state.gameHistory):
                self.dict["history"][i][0] = action.tile_to_put[0]
                self.dict["history"][i][1] = action.tile_to_put[1]
                self.dict["history"][i][2] = action.new_edges[0]
                self.dict["history"][i][3] = action.new_edges[1]
                self.dict["history"][i][4] = 1. if action.player == state.current_player() else 0.
                self.dict["history"][i][5] = 1.

        if "last_move" in self.dict:
            if state.gameHistory:
                action = state.gameHistory[-1]
                self.dict["last_move"][0] = action.tile_to_put[0]
                self.dict["last_move"][1] = action.tile_to_put[1]
                self.dict["last_move"][2] = action.new_edges[0]
                self.dict["last_move"][3] = action.new_edges[1]
                self.dict["last_move"][4] = 1. if action.player == state.current_player() else 0.
                self.dict["last_move"][5] = 1.

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f'p{player}')
        if "hand" in self.dict:
            pieces.append(f"hand:{state.hands[player]}")
        if "history" in self.dict:
            pieces.append(f"history:{str(state.gameHistory)}")
        if "last_move" in self.dict and state.gameHistory:
            pieces.append(f"last_move:{str(state.gameHistory[-1])}")
        return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, DominoGame)
