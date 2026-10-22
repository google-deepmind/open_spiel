# OpenSpiel GUI client

> **Note:** The OpenSpiel GUI is intended for personal research use on your
> own machine. It is not designed or reviewed as a production or secured
> system: do not expose it to untrusted networks or users. The server has no
> authentication, and the oracle viewer sees every player's private
> information.

The browser client for the OpenSpiel GUI server in `open_spiel/python/gui/`.
It is written in TypeScript with React and follows [STYLE.md](STYLE.md).

## Playing

From the repository root, with OpenSpiel installed (`pip install -e .`):

```sh
(cd open_spiel/gui && npm ci && npm run build)
python -m open_spiel.python.gui
```

and open the address it prints, `http://127.0.0.1:8080/` by default. The
page at `/` lists every game the server knows; each game is played at
`/game/<name>`. The address keeps the session and the viewer, so a reload
shows the same game to the same viewer.

A game page docks its panels: Gameplay, on the left, with a status line
over the board; beside it, Actions | State over Setup | View. Session is
closed at first and opens from Panel options as a tab beside Actions; Players
is a section of Setup. Every panel but Gameplay is a group of sections, and
each game remembers the section each shows. Moves are made in Actions ›
Legal, and the positions are stepped through from Actions › History, where
the oracle, or any viewer of a game that hides nothing, may also play on from
an earlier position, a new line that takes back the moves after it. A game
of perfect information opens for the oracle, any other as the first seat; any
viewer but a spectator changes who plays a seat, and in a game that hides
information, when another person's seat is to move, the page offers to hand
the device over behind a screen. Panel options reopens a closed panel and
keeps named layouts in the browser.

## Development

Requires Node.js 22.22.2 or later, the oldest release jsdom, the test
environment, supports. From this directory:

```sh
npm ci                     # install the pinned dependencies
npm run check              # format, lint, style gates and type-check
npm test                   # run the tests
npm run build              # build the client into dist/
npm run format             # rewrite files in the project format
npm run generate:contract  # regenerate the wire-format types
npm run lint:style         # the style gate alone
npm run contrast           # the contrast check alone
```

`npm run smoke:session -- http://127.0.0.1:8080 tic_tac_toe` plays one game
through a running server: it creates a session, plays to the end, goes back to
ply 2, undoes, resets to a three-ply history, and shows the position to seat 0,
checking that seat 0 is shown nothing the contract withholds from it.

`npm run smoke:setup -- http://127.0.0.1:8080 kuhn_poker players=3` sets up
games as the Setup panel does: it checks that the defaults of every loadable
game's Game options read back from their fields, starts a new game with the
typed value (here three seats, as setting Players does), and replays a typed
history.

The server serves the client from `dist/`, so a build is all it needs.
Third-party notices are in `licenses/`.

## End-to-end tests

`e2e/` drives the built client in headless Chromium with Playwright, against
the real server. `npm run e2e` builds the client, starts
`python -m open_spiel.python.gui` from the repository root on a free port,
runs every test and stops the server. The tests:

- play tic-tac-toe as seat 0 to the end against Random, then against MCTS,
  and make moves of 9x9 go from the legal list;
- view Kuhn poker as seat 0, which is sent and shown its own card and never
  its opponent's, then as the oracle, which is shown both;
- use each panel: the catalogue, Setup (Kuhn poker set to three players
  starts a three-seat game), Players, the legal list's
  windowing, State, the history's replay control and playing on from an
  earlier position, a copy icon, the theme and a saved layout across a
  reload;
- fail any test in which the page logs an error or a request fails.

They find every control by its accessible role and name; `e2e/app.ts` lists
the names they rely on. Install the browser once with
`npx playwright install --with-deps chromium`, or run in the Playwright
container image of the pinned version, `mcr.microsoft.com/playwright:v1.64.1`,
with Python and OpenSpiel installed in it. `npm run e2e:list` lists the tests
without running them.

| Variable         | Meaning                                                                |
| ---------------- | ---------------------------------------------------------------------- |
| `E2E_PYTHON`     | The Python interpreter to start the server with; `python3` by default. |
| `E2E_BASE_URL`   | A running server to test instead of starting one.                      |
| `E2E_SERVER_LOG` | Where the server's output goes; `e2e-server.log` by default.           |

## The wire format

The server defines the wire format, and publishes it as the JSON Schema
`open_spiel/python/gui/contract.schema.json`. `src/contract/contract.ts` holds
the TypeScript types generated from that schema. Never edit it by hand: after a
schema change, run `npm run generate:contract`. A test fails while the two
disagree.

The schema's `Error` definition becomes the type `Error`, which would hide the
global `Error`; import it under another name:
`import type {Error as ErrorBody} from './contract/contract.ts';`.

## Sessions

`src/session/` is the client's only path to the server. `SessionClient` has
one method per route, and turns a refusal into a `ContractError` carrying the
contract's error `code`. `SessionStore` holds one session, shown to one viewer
at a time: it runs commands one at a time, each from the position the previous
one left and as the viewer it was issued under, and drops the commands queued
behind a failed one. It never shows a frame built for another viewer.
`useSession` gives a component the store's frame, whether that frame is live or
a replay, the busy flag, the latest error and the commands.

## Setup

`src/setup/` is the Setup panel, a panel group (`src/dock/panel_group.tsx`)
whose inner tabs are Game options, Position and Players. Game options is left
out for a game with no parameters to offer, and Position for a spectator.

Game options offers one control per `ParameterSpec` in the catalogue, as a
card row with the parameter's name for people ("Board Size") on the left: a
checkbox for a boolean, a list for a choice, a number box for a number, a
text area for text that spans lines, and a text field otherwise. A valid
change starts a new session, in which each seat keeps its player: a checkbox
or a list at once, a typed number or text once it is committed by leaving
the field or pressing Enter. An invalid value is named and not sent, and a
change the server refuses is explained under its control. The seat-count
parameter (`players`, `num_players` or `numPlayers`) is offered as Players
when the catalogue lets the game have from `min_seats` to `max_seats` seats:
a number box bounded to that range, holding the session's number of seats
and defaulting to the catalogue's `num_players`. Committing another number
starts a new game with that many seats, the first ones keeping their
players and any added ones human, and Players then lists them. A game with
a fixed number of seats does not offer it, so a game whose only parameter
is a fixed seat count has no Game options tab. Go offers only its board size
once a game is open, and opens on the 9×9 board.

Position's Set state replays typed action ids, or an episode object's
`history`, `actions` or `sequence`, from the start of the game; its Reset,
once confirmed, starts the game again.

Who the game is shown to is a perspective, not a permission: the server lets
every viewer but a spectator create, reset and change seats, and the panel
offers its controls to every viewer but a spectator, who sees the values
read-only.

## View

`src/view/` is the View panel, a panel group of how the game is shown. Its
Perspective section chooses whose view the page shows: a seat, the oracle or
a spectator. The page's theme is an app setting, in the header's Settings.

## State and replay

`src/state/` shows one frame's position and steps through the live game.
`StatePanel` shows the frame as a group of sections: Text (Observation for a
seat that may not see the full state), Struct, Returns and Rewards, in that
order. A section the frame lacks has no tab; a missing text says why in a
line. The text and the struct sit in a card with a copy icon; the returns
and rewards are compact, with none. `ReplayControl`, the footer of Actions ›
History, moves to the start, the previous and next move and the end, and
along a slider, and in a game with chance can skip the positions where
chance acts. `STATE_GROUP` is the dock's State panel, reading the page's
frame; `StatePanel` draws the same group for a frame it is given.
`ReplayControl` takes
a frame, the session's busy flag, the game's chance mode and the session's
`goTo`. While a request is in flight, the slider keeps only its latest move and
sends it once the session is idle, so a drag sends one request per round trip.

## Colours and themes

Every colour is a token in `src/styles/tokens.css`, defined for the light and
the dark theme; `src/styles/dock.css` maps the dock's variables onto them.
`ThemeProvider` resolves the user's choice (system, light or dark) and sets
`<html data-theme>`, and an inline script in `index.html` does the same before
the first paint.

Two gates, both part of `npm run check`, keep the styling readable:

- `npm run lint:style` refuses opacity, filters, a colour outside the tokens,
  text under 12 px, and a disabled control styled by more than its cursor and
  edge.
- `npm run contrast` measures every text colour against every surface (at
  least 4.5:1) and the focus ring and control edges (at least 3:1), in both
  themes.

## Actions

`src/actions/` holds the three sections of the Actions panel. Each takes the
`Session` that `useSession` returns.

- `History` lists every action of the live game after a row for the initial
  position, a turn column and a label each. A viewer whose frame is
  `omniscient` (the oracle, or any viewer of a game that hides nothing) shows a
  position from its row; for any other viewer the rows are read-only, and a
  run of actions withheld from them is one row. The list follows new
  actions while it is scrolled to the bottom.
- `LegalActions` lists the legal actions at the position shown in id order,
  each with its id, and at a chance node the client picks, the outcomes with
  their probabilities. A list of more than 200 renders only the rows in
  view. At an earlier position an `omniscient` viewer plays on from there.
  When the viewer cannot play, a sentence says why, and a list withheld from
  the viewer is never shown as empty. A played row's focus moves to the row
  that takes its place. `hidden` makes it draw an empty box and do no work.
- `HistoryText` writes the history as a JSON list, of labels or ids, that is
  also a Python list literal, in a card with a copy icon.

## Players

`src/players/` is the Players section of Setup: who plays each seat, chosen
from the session's controllers, with each controller's settings and whether
it autoplays. A controller that cannot play the game is listed as not
available here, with the server's reason. A seat whose controller waits to be
advanced has a Step button for its turn, as does an autoplaying seat that
autoplay left to move (after an undo, or past the plies one request plays),
and a failed seat a Retry. Opening a
seat's settings starts a draft from the settings it plays with; the draft is
compared by value with them, a setting it was given without counting as its
default, and saved or reset from the bars above the controller and after the
last setting, one of which sticks to the scrolling seat list. A draft is
dropped once the server's settings for its seat change, by a save or by
another client; a spectator sees every seat's settings read-only. A refused
save is the session's error, shown where the dock shows session errors; the
panel keeps the draft so that it can be corrected. While Players is open,
Setup's dock group stays tall enough to show every seat. `src/form/` builds
the settings form from a controller's `config_schema`: a row per setting, its
description behind an "i" toggle and its `x-presets` as chips, and falls back
to a JSON box, where empty means no settings, for a schema a flat form cannot
show.

## Gameplay board

`src/board/` holds `GameplayBoard`, the board every game is played on. It takes
the `Session` that `useSession` returns and draws the game with text alone:

- The game's text for the viewer, verbatim, with no copy button (the text is
  copied from State): the full state for an `omniscient` viewer, otherwise the
  seat's information state, or its observation where the game gives no
  information state. The server has
  already withheld what the viewer may not see, so the board only chooses
  among the texts that arrived. When none did, a sentence says why.
- Nothing else: the moves are made in Actions › Legal, and the Gameplay
  panel's status line gives the turn, who is to move, the viewer and, once
  the game is over, the returns.
