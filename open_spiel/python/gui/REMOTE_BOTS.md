# Running a remote bot

A **Remote bot** seat is played by a bot server: a separate process, on this
machine or another, that the GUI server sends each position to over HTTP and
that answers with a move. Any bot can be served this way, a trained
checkpoint included, without the GUI knowing anything of it. The steps,
each with its check, run from the repository root with OpenSpiel installed
(`pip install -e .`):

1. **Start the GUI server** in one terminal:

   ```sh
   python -m open_spiel.python.gui --port 8080
   ```

   Check: it prints `Serving OpenSpiel games at http://127.0.0.1:8080/`.

2. **Start a bot server** in a second terminal. The reference server,
   `remote_bot_server`, serves `random` (either state mode), `mcts`
   (OpenSpiel's Python MCTS on the full state), or your own bot:

   ```sh
   python -m open_spiel.python.gui.remote_bot_server --port 9001 --bot random
   ```

   Check: it prints `Serving the random bot at http://127.0.0.1:9001`, and
   `http://127.0.0.1:9001/health`, opened in a browser, shows
   `{"ok": true, ...}`.

3. **Plug it into a seat.** In the browser, open a game
   (`http://127.0.0.1:8080/game/kuhn_poker`), then Setup › Players ›
   Player 1 › Settings: Controller **Remote bot**, URL
   `http://127.0.0.1:9001`, State `seat_view` (or `full_state` for a bot that
   needs the whole state), Autoplay on, and Save. An agent does the same
   with the MCP server's `set_seat` tool ([MCP.md](MCP.md)). Any other
   client uses the JSON API (CONTRACT.md), where every session route names
   its viewer in the `viewer` query parameter (`oracle` here) and every
   body is `application/json`. In a third terminal:

   ```sh
   GUI=http://127.0.0.1:8080/api
   SESSION=$(curl -s -X POST "$GUI/sessions?viewer=oracle" \
     -H 'Content-Type: application/json' \
     -d '{"game": "kuhn_poker", "parameters": {}, "chance_policy": "auto", "seed": null,
          "seats": [{"controller": "human", "config": {}, "autoplay": false},
                    {"controller": "human", "config": {}, "autoplay": false}]}' \
     | python -c 'import json, sys; print(json.load(sys.stdin)["session"]["id"])')
   curl -s -X PUT "$GUI/sessions/$SESSION/seats/1?viewer=oracle" \
     -H 'Content-Type: application/json' \
     -d '{"controller": "remote", "config": {"url": "http://127.0.0.1:9001", "state": "seat_view"}, "autoplay": true}' \
     | python -c 'import json, sys; f = json.load(sys.stdin); print(f["seats"][1]["status"] if "seats" in f else f["message"])'
   curl -s -X POST "$GUI/sessions/$SESSION/actions?viewer=oracle" \
     -H 'Content-Type: application/json' -d '{"action": 0, "ply": null}' \
     | python -c 'import json, sys; print([h["action"] for h in json.load(sys.stdin)["history"]])'
   ```

   Check: the seat reads "Remote bot" (briefly "Preparing" while the GUI
   server calls the bot's `/health`; the `PUT` prints `ready`, `preparing`
   for a bot slow to answer, or why the seat was refused). After your move
   the bot server logs `POST /spiel/observe` (or `/spiel/step`), and the
   bot's move appears in Actions › History: the last `curl` prints four
   actions, the two cards dealt, your pass and the bot's move.

4. **Serve your own bot.** Create `my_bots/my_bot.py` (any importable
   module) with a function that returns the bot:

   ```python
   # my_bots/my_bot.py
   import pyspiel


   class MyBot:
     state_modes = ("full_state",)  # or ("seat_view",), or both
     games = None  # or ("hearts", "gin_rummy"): the games it plays

     def __init__(self):
       self.policy = load_my_checkpoint("/path/to/checkpoint")  # yours

     def step(self, mode, request):
       state = pyspiel.load_game(request["game"]).new_initial_state()
       for action in request["history"]:
         state.apply_action(action)
       action = self.policy.best_action(state, request["seat"])
       return {"action": action, "notice": None}


   def make_bot():
     return MyBot()
   ```

   and serve it:

   ```sh
   python -m open_spiel.python.gui.remote_bot_server --port 9001 --bot my_bots.my_bot:make_bot
   ```

   Check: `http://127.0.0.1:9001/info`, opened in a browser, lists its
   `state_modes` and `games`; a seat set to a mode or a game it does not list is
   refused as unavailable. A `step` that raises `ValueError` is answered as a
   refusal (HTTP 422) with its message; anything else, as HTTP 500.

5. **A slow bot** (a large checkpoint) needs a longer move budget: start the
   GUI server with `--move-budget-seconds 60`. The seat's `timeout_seconds`
   waits less than the budget; a bot that does not answer in time fails its
   seat, which pauses until an undo, a reset or a new controller.

6. **A bot on another machine** (here `gpu.lan`) answers only requests that
   carry a token. On that machine, with a token of your choosing in
   `BOT_TOKEN`:

   ```sh
   BOT_TOKEN=<token> python -m open_spiel.python.gui.remote_bot_server \
     --host 0.0.0.0 --port 9001 --bot my_bots.my_bot:make_bot --token-env BOT_TOKEN
   ```

   The bot server refuses to listen beyond this machine without
   `--token-env`, and answers HTTP 401 to a request without the token. On
   the GUI's machine, with the same token in `BOT_TOKEN`:

   ```sh
   BOT_TOKEN=<token> python -m open_spiel.python.gui --port 8080 \
     --remote-bot-host gpu.lan --remote-bot-token-env gpu.lan:9001=BOT_TOKEN
   ```

   Then give the seat the URL `http://gpu.lan:9001`. Check: the seat
   reaches ready; the same seat on a GUI server started without
   `--remote-bot-token-env` is refused as unavailable, saying the bot server
   answered HTTP 401.

   Without `--remote-bot-host` the GUI server posts only to this machine,
   so that whoever can reach the page cannot point a seat at an arbitrary
   host (`*` admits any host). A token goes only to the host and port it is
   given for, never to any other URL a seat names, and it is never part of
   a seat's settings, which every viewer sees.

A seat set to `full_state` is sent the whole state, opponents' private
information included, in any game; its settings, which every viewer sees,
say so, and the bot is trusted to use only what its seat may see where that
matters. That trust extends to its notice (below): a notice is free text
the bot writes, shown to the seat's own viewer, so a full-state bot's
notice can carry hidden information, such as an opponent's cards, to
whoever views that seat. The server does not filter notices; its state
policy (`controller_registry.full_state_notice`) is the place to drop or
limit the notices of full-state seats.

## The remote bot protocol

Every request is sent to the seat's `url` plus a path, with the protocol
version in the `Spiel-Protocol: 1` header, so that its JSON body is exactly
the bot's input. The seat's settings are `url`, `state` (`seat_view` or
`full_state`), `config` (a JSON object passed on as `config`), `extras`,
`tensors` and `timeout_seconds`.

| Request | Body |
| ------- | ---- |
| `GET /health` | none; any 2xx answer means ready. Called when the seat is given the bot. |
| `GET /info` | none; optional. A 2xx answer `{"games": [...] \| null, "state_modes": [...]}` limits what the seat may ask for. |
| `POST /spiel/step` (`full_state`) | `{"game": str(game), "state": state.serialize(), "history": [action ids], "seat", "legal_actions", "seed", "game_encoding": {"num_distinct_actions"}, "config"}` |
| `POST /spiel/observe` (`seat_view`) | `{"game": str(game), "observation": {"information_state_string", "seat", "legal_actions", "seed"}, "config", "game_encoding": {"num_distinct_actions"}}` |

A seat-view request carries those keys and no others unless the seat asks
for more, so a strict seat-view bot, one that refuses any key it does not
take (a state or a history would hand it hidden information), accepts it.
`information_state_string` is the seat's `information_state_string(seat)`,
or null where the game provides none. With `extras` on, `observation` also
carries `observation_string` (null where the game provides none),
`legal_action_labels` (each legal action's `action_to_string`), and
`history`: the history as the seat's own frame shows it, `{"seat",
"action", "label", "public_label"}` per ply, the seat's own actions in full
and every other ply (chance included) only as its public label, in a game
that hides information. With `tensors` on, it also carries
`observation_tensor` and `information_state_tensor`, each null where the
game provides none. A strict bot refuses either addition. `seed` changes
from ply to ply and is the same at the same position, so a bot that draws
from it moves reproducibly.

The reply is `{"action": <id>}`, or `{"actions": [<id>, ...]}`, of which the
first is played and the bot is asked again at the next position. It may add
`"notice"`: plain text of at most 200 characters about the move, such as a
fallback the bot took. The seat's own viewer and omniscient viewers see it
after the controller's name in Setup › Players (`SeatView.notice`); it has no
effect on play. Other fields are ignored.

A reply with HTTP 400, 409 or 422, or a 2xx reply with `"error"` set, is a
refusal (`{"error": ..., "reason": ...}`), and the seat fails with
`RemoteBotRefusedError`. Any other status, no connection, or no answer within
the timeout fails it with `RemoteBotUnreachableError`; a reply that is not
the protocol's, or whose action is not legal, with `RemoteBotReplyError`.
Nothing is retried. As for every controller, what went wrong reaches only
omniscient viewers (`failure.message`).
