# Playing games in a browser

OpenSpiel includes a local web server and a browser client for playing any
registered game: choose a game from the catalogue, play it against people or
bots, step through its history, and see the state as any seat, as a
spectator, or with everything revealed.

> **Note:** The OpenSpiel GUI is intended for personal research use on your own
> machine. It is not designed or reviewed as a production or secured system: do
> not expose it to untrusted networks or users. The server has no
> authentication, and the oracle viewer sees every player's private
> information.

The server is the Python package `open_spiel.python.gui`. The client is in
`open_spiel/gui/`, written in TypeScript; it is built once with Node.js and
then served by the server as static files.

## Requirements

-   OpenSpiel installed for Python (see [Installation](install.md)).
-   Node.js 22.22.2 or later, with `npm`, to build the client.
-   A checkout of the OpenSpiel repository, which holds the client's source.

## From a source checkout

From the root of the repository, with OpenSpiel installed from it (for
example with `python3 -m pip install -e .`, or the `PYTHONPATH` setup in
[Installation](install.md)):

```sh
(cd open_spiel/gui && npm ci && npm run build)
python3 -m open_spiel.python.gui
```

Open the address the server prints, `http://127.0.0.1:8080/` by default. The
page at `/` lists every game; each game is played at `/game/<name>`.

## With OpenSpiel installed from pip

A wheel ships the server but not the client. Build the client from a
checkout of the same version, and point the server at it:

```sh
(cd open_spiel/gui && npm ci && npm run build)
python3 -m open_spiel.python.gui --client-dir "$PWD/open_spiel/gui/dist"
```

Without a built client the server still answers its JSON API, and its page
at `/` says how to build the client.

## Boards and bots

Tic-tac-toe, connect four, chess, checkers, Kuhn poker, Leduc poker, gin
rummy, hearts and euchre have their own boards, drawn by a per-game adapter
in `open_spiel/python/gui/adapters/` and a renderer in
`open_spiel/gui/src/games/`. Every other game is drawn from its text. The
catalogue badges each game Board UI, Text only or Unavailable, and can be
filtered by badge.

A game page opens in analysis, which shows the full state: View ›
Perspective chooses the seat the board is drawn from, a board's Show all
cards turns the other hands face up, and moves are made for every seat a
person plays. An address that names a seat (`viewer=1`) or a spectator shows
only what that viewer may see.

`open_spiel/gui/src/INTERFACES.md` says how to add a board, a card deck or a
card back, and how to swap in a different chess board or chess pieces.

A seat is played by a person or a bot: `random`, `mcts`, or one a plugin
registers. For a game implemented in C++, `mcts` is pyspiel's `MCTSBot`; for
a game implemented in Python, it is OpenSpiel's Python MCTS, which stops at
the move budget and at a cap on tree nodes. At startup the server checks
whether pyspiel's `MCTSBot` stops at its simulation count when it is also
given a wall-clock limit, and reports the answer in `/api/health`. If it
does, `mcts` runs up to its simulation count with a wall-clock limit inside
the move budget; if not, it sets no wall-clock limit and caps the
simulations at 250.

Bots are built in the background, so a seat whose bot loads a model shows
it preparing while the rest of the page stays usable. A plugin is a module
with a function `register_into(controllers)`; the server loads plugins only
when `--controllers` names them, and stops with an error naming the plugin
if one fails to load (see `open_spiel/python/gui/plugins.py`).

## Options

```
python3 -m open_spiel.python.gui --help
```

| Flag                | Meaning                                                    |
| ------------------- | ---------------------------------------------------------- |
| `--host`            | The address to listen on; `127.0.0.1` by default.          |
| `--port`            | The port to listen on; `8080` by default, `0` for any.     |
| `--allow-host NAME` | A further host name requests may use. Repeatable.          |
| `--client-dir DIR`  | The directory of the built client; `open_spiel/gui/dist`.  |
| `--controllers MODULE[,MODULE...]` | Controller plugins to load before serving; `auto` loads every installed `open_spiel.gui.controllers` entry point. Repeatable; none by default. |
| `--move-budget-seconds SECONDS` | How long a bot may take to choose a move; `10` by default. |
| `--build-budget-seconds SECONDS` | How long a bot may take to load and warm up; `120` by default. |

The server has no authentication, and the oracle viewer is shown every
player's private information. Keep the default loopback address unless the
network is trusted; to play from another machine, forward the port over SSH
rather than listening on a public address.

## Tests

The server's tests run with the other Python tests (`ctest -R python/gui`
from the build directory) and take well under a minute. They sweep a few
representative games; `python3 -m open_spiel.python.gui.sweep_all` sweeps
every registered game and takes a few minutes. The client has its own checks
and tests: from `open_spiel/gui/`, `npm run check` and `npm test`.
