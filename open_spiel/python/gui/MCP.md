# Driving sessions from an agent (MCP)

`python -m open_spiel.python.gui.mcp` is a Model Context Protocol server that
lets an agent drive the GUI server's sessions. It speaks MCP over stdin and
stdout to the agent that starts it, opens no port of its own, and calls the
GUI server's API; it holds no keys or login, and drives a GUI server on this
machine unless started with `--allow-remote-server`. Register it with an MCP
client as a command, for example:

```json
{
  "mcpServers": {
    "openspiel": {
      "command": "python",
      "args": ["-m", "open_spiel.python.gui.mcp", "--server", "http://127.0.0.1:8080"]
    }
  }
}
```

| Tool | What it does |
| ---- | ------------ |
| `health` | the server's version, controllers and capabilities |
| `list_games` | games, their players, and whether the GUI draws a board or text |
| `describe_game` | a game's parameters, and each controller with whether it can play the game and its settings' schema |
| `open_session` | opens a session (seats, parameters, seed, chance policy) and returns its browser URL |
| `close_session` | closes a session |
| `get_frame` | a position for a viewer, live or at an earlier `ply`, as a summary or the frame's JSON |
| `get_legal_actions` | the legal actions or chance outcomes, as id and label |
| `act` | plays an action by id, or by its exact label, live or from an earlier `ply` |
| `choose_chance` | chooses a chance outcome, under the `expose` policy |
| `advance` | lets the bots play until a person must act |
| `undo`, `reset` | takes back an action; restarts from a history |
| `set_seat` | gives a seat a controller, a remote bot included |

Every session tool takes `viewer`: `oracle` (analysis, the default),
`spectator` or a seat index, and returns what that viewer may see. A session
an agent opens shows in a browser at the URL `open_session` returns, and a
session opened in a browser can be driven by the id in its address.
