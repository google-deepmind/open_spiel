# Python Games

This directory contains games implemented in Python. The majority of OpenSpiel
games are in C++, which is significantly faster, but Python may still be
suitable for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to have poor performance if
the algorithm relies on processing and updating states as it goes, e.g. MCTS.

Suggested games to use as a basis for your own implementations:

*   [kuhn_poker](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/games/kuhn_poker.py)
    for imperfect information games with chance
*   [tic_tac_toe](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/games/tic_tac_toe.py)
    for perfect information games without chance
*   [iterated_prisoners_dilemma](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/games/iterated_prisoners_dilemma.py)
    for games with simultaneous moves

### Implementation Notes

The Python game implementation sticks quite closely to the C++ one. The main
differences are as follows:

*   Observations should be supported entirely through the Observation API, entry
    point the `make_py_observer` method on the game class. See
    [kuhn_poker](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/games/kuhn_poker.py)
    for a complete example.

*   Parameter handling is significantly simplified. Default parameters are
    provided in the GameType; the parameters supplied to the constructor will
    have had default parameters applied. The C++ GameParameter type is not used.
    See
    [iterated_prisoners_dilemma](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/games/iterated_prisoners_dilemma.py)
    for a very simple example.

*   `_legal_actions` only needs to handle the case where the game is in progress
    and it is that player's turn. Cases which require special handling in C++
    games, such as terminal states, chance nodes, not the player's turn, are
    instead handled in the Python/C++ layer.

*   `_action_to_string` always receives the correct player as an argument,
    unlike the C++ version where it may be omitted, and hence this case must be
    handled by the game implementor

*   `_apply_action` and `_apply_actions` correspond to `DoApplyAction` and
    `DoApplyActions`; the C++ history will be updated after the relevant one of
    these functions is called.
