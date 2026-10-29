# OpenSpiel GUI server

> **Note:** The OpenSpiel GUI is intended for personal research use on your
> own machine. It is not designed or reviewed as a production or secured
> system: do not expose it to untrusted networks or users. The server has no
> authentication, and the oracle viewer sees every player's private
> information.

A local web server that lets people play any OpenSpiel game in a browser. The
browser client lives in `open_spiel/gui/`.

Run it with `python3 -m open_spiel.python.gui`; `--help` lists its flags. It
listens on `127.0.0.1` by default. `CONTRACT.md` specifies its JSON API.

A game is drawn by its adapter in `adapters/`, one module per game, which
presents each position to each viewer; `open_spiel/gui/src/INTERFACES.md`
says how to add one, with its client renderer. A game without an adapter is
drawn from its text.

A seat is played by a person or a controller: `random`, `mcts`, a remote bot
(`remote`, below), or one a plugin registers (`controllers.py`). MCTS = pyspiel
C++ MCTSBot (memory cap 1000 MB) for C++ games; OpenSpiel Python MCTS for
Python-implemented games, which stops at the move budget and at its node budget.
At startup the server checks whether pyspiel's C++ search honours its simulation
count when given a wall-clock time, logs the answer and reports it in
`/api/health` (`capabilities.cpp_mcts_limits`). If it does, MCTS runs up to
10000 simulations with a wall-clock backstop inside the move budget; if not, it
sets no clock and caps the simulations at 250 (`mcts_controller.py`;
`CONTRACT.md`, Controllers, with the measurements behind the cap).

The tests are the `*_test.py` files beside the modules. They sweep a few
representative games; `python3 -m open_spiel.python.gui.sweep_all` sweeps
every registered game against `sweep_exceptions.json`, which takes minutes.
Run it after changing the server or adding a game.

The format, lint and type-check configuration is in `pyproject.toml`, which
also pins the tools that run it.

## Remote bots and agents

- [Play gin rummy against a trained bot](REMOTE_BOTS.md#play-gin-rummy-against-a-trained-bot):
  one command starts a bot server with a checkpoint and the GUI, and opens a
  game against it; the checkpoint is a variable.
- [Running a remote bot](REMOTE_BOTS.md): a seat played by a bot server,
  on this machine or another, sent the seat's view or the full state; the
  steps, the reference bot server that wraps any bot (a trained checkpoint
  included), and the protocol.
- [Driving sessions from an agent (MCP)](MCP.md): an MCP server through
  which an agent lists games, opens and closes sessions, reads frames and
  plays.
