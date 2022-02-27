## The code structure

Generally speaking, the directories directly under `open_spiel` are C++ (except
for `integration_tests` and `python`). A similar structure is available in
`open_spiel/python`, containing the Python equivalent code.

Some top level directories are special:

*   `open_spiel/integration_tests`: Generic (python) tests for all the games.
*   `open_spiel/tests`: The C++ common test utilities.
*   `open_spiel/scripts`: The scripts useful for development (building, running
    tests, etc).

For example, we have for C++:

*   `open_spiel/`: Contains the game abstract C++ API.
*   `open_spiel/games`: Contains the games C++ implementations.
*   `open_spiel/algorithms`: The C++ algorithms implemented in OpenSpiel.
*   `open_spiel/examples`: The C++ examples.
*   `open_spiel/tests`: The C++ common test utilities.

For Python you have:

*   `open_spiel/python/examples`: The Python examples.
*   `open_spiel/python/algorithms/`: The Python algorithms.

## C++ and Python implementations.

Some objects (e.g. `Policy`, `CFRSolver`, `BestResponse`) are available both in
C++ and Python. The goal is to be able to use C++ objects in place of Python
objects for most of the cases. In particular, for the objects that are well
supported, expect to have in the test for the Python object, a test checking
that both the C++ and the Python implementation behave the same.

## Adding a game

We describe here only the simplest and fastest way to add a new game. It is
ideal to first be aware of the general API (see `spiel.h`).

1.  Choose a game to copy from in `games/` (or `python/games/`). Suggested
    games: Tic-Tac-Toe and Breakthrough for perfect information without chance
    events, Backgammon or Pig for perfect information games with chance events,
    Goofspiel and Oshi-Zumo for simultaneous move games, and Leduc poker and
    Liar’s dice for imperfect information games. For the rest of these steps, we
    assume Tic-Tac-Toe.
2.  Copy the header and source: `tic_tac_toe.h`, `tic_tac_toe.cc`, and
    `tic_tac_toe_test.cc` to `new_game.h`, `new_game.cc`, and `new_game_test.cc`
    (or `tic_tac_toe.py` and `tic_tac_toe_test.py`).
3.  Configure CMake:
    *   If you are working with C++: add the new game’s source files to
        `games/CMakeLists.txt`.
    *   If you are working with C++: add the new game’s test target to
        `games/CMakeLists.txt`.
    *   If you are working with Python: add the test to `python/CMakeLists.txt`
        and import it in `python/games/__init__.py`
4.  Update boilerplate C++/Python code:
    *   In `new_game.h`, rename the header guard at the the top and bottom of
        the file.
    *   In the new files, rename the inner-most namespace from `tic_tac_toe` to
        `new_game`.
    *   In the new files, rename `TicTacToeGame` and `TicTacToeState` to
        `NewGameGame` and `NewGameState`.
    *   At the top of `new_game.cc`, change the short name to `new_game` and
        include the new game’s header.
5.  Update Python integration tests:
    *   Add the short name to the list of expected games in
        `python/tests/pyspiel_test.py`.
6.  You should now have a duplicate game of Tic-Tac-Toe under a different name.
    It should build and the test should run, and can be verified by rebuilding
    and running the example `examples/example --game=new_game`.
7.  Now, change the implementations of the functions in `NewGameGame` and
    `NewGameState` to reflect your new game’s logic. Most API functions should
    be clear from the game you copied from. If not, each API function that is
    overridden will be fully documented in superclasses in `spiel.h`.
8.  Once done, rebuild and rerun the tests to ensure everything passes
    (including your new game’s test!).
9.  Add a playthrough file to catch regressions:
    *   Run `./open_spiel/scripts/generate_new_playthrough.sh new_game` to
        generate a random game, to be used by integration tests to prevent any
        regression. `open_spiel/integration_tests/playthrough_test.py` will
        automatically load the playthroughs and compare them to newly generated
        playthroughs.
    *   If you have made a change that affects playthroughs, run
        `./scripts/regenerate_playthroughs.sh` to update them.

## Conditional dependencies

The goal is to make it possible to optionally include external dependencies and
build against them. The setup was designed to met the following needs:

-   **Single source of truth**: We want a single action to be sufficient to
    manage the conditional install and build. Thus, we use bash environment
    variables, that are read both by the install script (`install.sh`) to know
    whether we should clone the dependency, and by CMake to know whether we
    should include the files in the target. Tests can also access the bash
    environment variable.
-   **Light and safe defaults**: By default, we exclude the dependencies to
    diminish install time and compilation time. If the bash variable is unset,
    we download the dependency and we do not build against it.
-   **Respect the user-defined values**: The `global_variables.sh` script, which
    is included in all the scripts that needs to access the constant values, do
    not override the constants but set them if and only if they are undefined.
    This respects the user-defined values, e.g. on their `.bashrc` or on the
    command line.

When you add a new conditional dependency, you need to touch:

-   the root CMakeLists.txt to add the option, with an OFF default
-   add the option to `scripts/global_variables.sh`
-   change `install.sh` to make sure the dependency is installed
-   use constructs like `if (${OPEN_SPIEL_BUILD_WITH_HANABI})` in CMake to
    optionally add the targets to build.

## Debugging tools

For complex games it may be tricky to get all the details right. Reading through
the playthrough (or visually inspecting random games via the example) is the
first step in verifying the game mechanics. You can visualize small game trees
using [open_spiel/python/examples/treeviz_example.py](https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/examples/treeviz_example.py) or for
large games there is an interactive viewer for OpenSpiel games called
[SpielViz](https://github.com/michalsustr/spielviz).

## Adding Game-Specific Functionality

OpenSpiel focuses on maintaining a general API to an underlying suite of games,
but sometimes it is convenient to work on specific games. In this section, we
describe how to get (or set) game-specific information from/to the generic state
objects, and how to expose these functions to python.

Suppose, for example, we want to look at (or set) the private cards in a game of
Leduc poker. We will use an example based on this
[this commit](https://github.com/deepmind/open_spiel/commit/4cd1e5889e447d285eb3f16901ccab5c14e62187).

1.  First, locate the game you want to access. The game implementations are in
    the `games/` subdirectory and have two main files: e.g. `leduc_poker.h`
    (header) and `leduc_poker.cc` (implementation).
2.  For simple accessor methods that just return the information and feel free
    have the full implementation to the game's header file (e.g.
    `LeducState::GetPrivateCards`). You can also declare the function in the
    header and provide the implementation in source file (e.g.
    `LeducPoker::SetPrivateCards`).
3.  That's it for the core game logic. To expose these methods to Python, add
    them to the Python module (via pybind11). Some games already have
    game-specific functionality, so if a files named `games_leduc_poker.h` and
    `games_leduc_poker.cc` exist within `python/pybind11`, add to them (skip to
    Step 5).
4.  If the games-specific files do not exist for your game of interest, then:
    *   Add the files. Copy one of the other ones, adapt the names, and remove
        most of the bindings code.
    *   Add the new files to the `PYTHON_BINDINGS` list in
        `python/CMakeFiles.txt`.
    *   Modify `pyspiel.cc`: include the header at the top, and call the init
        function at the bottom.
5.  Add the custom methods to the game-specific python bindings
    (`games_leduc_poker.cc`, i.e. `LeducPoker::GetPrivateCards` and
    `LeducPoker::SetPrivateCards`). For simple types, this should be relatively
    straight-forward; you can see how by looking at the other game-specific
    functions. For complex types, you may have to bind additional code (see e.g.
    `games_backgammon.cc`). If it is unclear, do not hesitate to ask, but also
    please check the
    [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/).
6.  Add a simple test to `python/games_sim_test.py` to check that it worked. For
    inspiration, see e.g. `test_leduc_get_and_set_private_cards`.

## Language APIs

There are currently four other language APIs that expose functionality from the
C++ core.

-   [Python](https://github.com/deepmind/open_spiel/tree/master/open_spiel/python).
-   [Julia](https://github.com/deepmind/open_spiel/tree/master/open_spiel/julia)
-   [Go](https://github.com/deepmind/open_spiel/tree/master/open_spiel/go)
    (experimental)
-   [Rust](https://github.com/deepmind/open_spiel/tree/master/open_spiel/rust)
    (experimental)
