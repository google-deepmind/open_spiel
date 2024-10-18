## First examples

One can run an example of a game running (in the `build/` folder):

```bash
./examples/example --game=tic_tac_toe
```

Similar examples using the Python API (run from one above `build`):

```bash
# Similar to the C++ example:
python3 open_spiel/python/examples/example.py --game_string=breakthrough

# Play a game against a random or MCTS bot:
python3 open_spiel/python/examples/mcts.py --game=tic_tac_toe --player1=human --player2=random
python3 open_spiel/python/examples/mcts.py --game=tic_tac_toe --player1=human --player2=mcts
```

## Concepts

The following documentation describes the high-level concepts. Refer to the code
comments for specific API descriptions.

Note that, in English, the word "game" is used for both the description of the
rules (e.g. the game of chess) and for a specific instance of a playthrough
(e.g. "we played a game of chess yesterday"). We will be using "playthrough" or
"trajectory" to refer to the second concept.

The methods names are in `CamelCase` in C++ and `snake_case` in Python without
any other difference (e.g. `state.ApplyAction` in C++ will be
`state.apply_action` in Python).

### The tree representation

There are mainly 2 concepts to know about (defined in
[open_spiel/spiel.h](https://github.com/deepmind/open_spiel/blob/master/open_spiel/spiel.h)):

*   A `Game` object contains the high level description for a game (e.g. whether
    it is simultaneous or sequential, the number of players, the maximum and
    minimum scores).
*   A `State`, which describes a specific point (e.g. a specific board position
    in chess, a specific set of player cards, public cards and past bets in
    Poker) within a trajectory.

All possible trajectories in a game are represented as a tree. In this tree, a
node is a `State` and is associated to a specific history of moves for all
players. Transitions are actions taken by players (in case of a simultaneous
node, the transition is composed of the actions for all players).

Note that in most games, we deal with chance (i.e. any source of randomness)
using a an explicit player (the "chance" player, which has id
`kChancePlayerId`). For example, in Poker, the root state would just be the
players without any cards, and the first transitions will be chance nodes to
deal the cards to the players (in practice once card is dealt per transition).

See `spiel.h` for the full API description. For example,
`game.NewInitialState()` will return the root `State`. Then,
`state.LegalActions()` can be used to get the possible legal actions and
`state.ApplyAction(action)` can be used to update `state` in place to play the
given `action` (use `state.Child(action)` to create a new state and apply the
action to it).

## Loading a game

The games are all implemented in C++ in [open_spiel/games](https://github.com/deepmind/open_spiel/blob/master/open_spiel/games).
Available games names can be listed using `RegisteredNames()`.

A game can be created from its name and its arguments (which usually have
defaults). There are 2 ways to create a game:

*   Using the game name and a structured `GameParameters` object (which, in
    Python, is a dictionary from argument name to compatible types (int, bool,
    str or a further dict). e.g. `{"players": 3}` with `LoadGame`.
*   Using a string representation such as `kuhn_poker(players=3)`, giving
    `LoadGame(kuhn_poker(players=3))`. See `open_spiel/game_parameters.cc` for
    the exact syntax.

#### Creating sequential games from simultaneous games

It is possible to apply generic game transformations (see
[open_spiel/game_transforms/](https://github.com/deepmind/open_spiel/blob/master/open_spiel/game_transforms/)) such as loading an `n`-players
simultaneous games into an equivalent turn-based game where simultaneous moves
are encoded as `n` turns.

One can use `LoadGameAsTurnBased(game)`, or use the string representation, such
as
`turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=4,points_order=descending))`.

## Playing a trajectory

Here are for example the Python code to play one trajectory:

```python
import random
import pyspiel
import numpy as np

game = pyspiel.load_game("kuhn_poker")
state = game.new_initial_state()
while not state.is_terminal():
  legal_actions = state.legal_actions()
  if state.is_chance_node():
    # Sample a chance event outcome.
    outcomes_with_probs = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(action)
  else:
    # The algorithm can pick an action based on an observation (fully observable
    # games) or an information state (information available for that player)
    # We arbitrarily select the first available action as an example.
    action = legal_actions[0]
    state.apply_action(action)
```

See [open_spiel/python/examples/example.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/example.py) for a more
thorough example that covers more use of the core API.

See [open_spiel/python/examples/playthrough.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/playthrough.py) (and
[open_spiel/python/algorithms/generate_playthrough.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/generate_playthrough.py)) for an
richer example generating a playthrough and printing all available information.

In C++, see [open_spiel/examples/example.cc](https://github.com/deepmind/open_spiel/blob/master/open_spiel/examples/example.cc) which generates
random trajectories.
