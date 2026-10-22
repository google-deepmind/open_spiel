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
