# GUI wire contract

The game server speaks JSON over HTTP. Each response body is a **frame**, the
**catalogue**, the **health** report or an **error**. This document is the
specification of those bodies, of the request bodies, and of the routes.

The definitions are Python dataclasses: [`contract.py`](contract.py),
[`frame.py`](frame.py), [`catalogue.py`](catalogue.py),
[`health.py`](health.py), [`errors.py`](errors.py) and
[`request.py`](request.py). Their constructors enforce the invariants below,
so the server cannot build a body that breaks them.
[`contract.schema.json`](contract.schema.json) is the JSON Schema generated
from them by [`schema.py`](schema.py); clients generate their types from it.

Every object carries every field. `null` is a value: a nullable field is
present and null, never absent. Objects admit no extra fields.

## Versioning

Every response carries `contract_version`, currently `2`. It increases on any
change a client could break on: a field added, removed, renamed or retyped,
an enum value added or removed, or a route changed.

## Routes

Every session route takes the viewer in the required `viewer` query
parameter: `oracle`, `spectator`, or a seat index. Each route answers `200`
with the response shown, or an error. A path below that exists, requested
with a method it does not accept, is refused with `method_not_allowed`; a
path that does not exist, with `not_found`.

| Method | Path | Request | Response | Errors |
|---|---|---|---|---|
| `GET` | `/api/health` | | `Health` | |
| `GET` | `/api/games` | | `Catalogue` | |
| `POST` | `/api/sessions` | `CreateSession` | `Frame`, live | `bad_request`, `not_found`, `game_unavailable`, `viewer_unavailable`, `controller_unavailable` |
| `GET` | `/api/sessions/{id}` | | `Frame`, live | `bad_request`, `not_found`, `viewer_unavailable` |
| `GET` | `/api/sessions/{id}/plies/{ply}` | | `Frame`, at `ply` | `bad_request`, `not_found`, `viewer_unavailable` |
| `POST` | `/api/sessions/{id}/actions` | `ApplyAction` | `Frame`, live | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `illegal_action`, `controller_failed` |
| `POST` | `/api/sessions/{id}/chance` | `ApplyChanceOutcome` | `Frame`, live | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `illegal_action`, `controller_failed` |
| `POST` | `/api/sessions/{id}/advance` | `Advance` | `Frame`, live | `bad_request`, `not_found`, `viewer_unavailable`, `controller_failed` |
| `POST` | `/api/sessions/{id}/undo` | | `Frame`, live | `bad_request`, `not_found`, `viewer_unavailable`, `nothing_to_undo` |
| `POST` | `/api/sessions/{id}/reset` | `Reset` | `Frame`, live | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `illegal_action` |
| `PUT` | `/api/sessions/{id}/seats/{seat}` | `SeatConfig` | `Frame`, live | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `controller_unavailable` |

- **Create** builds the game and returns its initial position, after any
  autoplay seats and `auto` chance nodes have acted.
- **Actions** plays an action for the seat to move. A seat viewer may act
  only for itself, and only when its seat is human; the oracle may act for
  any human seat; a spectator never acts.
- **Chance** chooses an outcome under the `expose` policy. In a game that
  hides information only the oracle may choose; otherwise any viewer may,
  a spectator or the seat the outcome favours included. Choosing chance is
  not acting for a seat: `expose` trusts the client with chance, for
  analysis and teaching, and a session that must keep chance fair uses
  `auto`.
- **Playing from an earlier position.** Actions and Chance take the `ply`
  of the frame the choice was made from; null, or `num_plies`, is the live
  position. An earlier ply plays from that position: the plies after it are
  taken back and the choice is applied there, all at once, and an illegal
  choice changes nothing. Taking back plies changes the game for every
  seat, so in a game that hides information only the oracle plays from an
  earlier position; any other viewer is refused with `not_your_turn`. A
  `ply` past the live position is refused with `not_found`. As after an
  undo, a paused session resumes, and `auto` chance outcomes that follow
  are drawn afresh.
- **Advance** lets controllers play their seats, until a human seat or an
  exposed chance node must act, the game ends, a controller fails, or
  `max_plies` seat actions have been applied. `max_plies` is at most 1000,
  and null means 1000: the request holds its session until play stops, so
  play is bounded. Under `auto`, chance outcomes are sampled as they come
  and do not count.
- **Undo** removes the most recent action a seat chose, with every ply after
  it: chance outcomes and controllers' actions. A seat takes back only its
  own action, and only while no other human seat has acted after it; the
  oracle takes back the most recent action of any seat. Under `auto`, the
  chance outcomes that follow are drawn afresh rather than repeated, so a
  seat cannot see an outcome, undo, and choose again knowing it.
- **Reset** restarts the game from its initial state and replays `history`,
  chance outcomes included, under either chance policy. Whoever the client
  shows the game to sets it up, so any viewer but a spectator resets; a
  spectator, who only watches, is refused with `not_your_turn`. An entry
  that is not legal where it stands is refused with `illegal_action` naming
  its index in `ply`, and the session is left unchanged.
- **Seats** hands a seat to another controller. On a paused seat this clears
  the failure and resumes play. As with reset, any viewer but a spectator
  changes a seat; a spectator is refused with `not_your_turn`.
- **Health** identifies the running code, so a client or sweep can refuse a
  server it does not expect.

A replay (`/plies/{ply}`) is rebuilt from the history and never changes the
session. The server refuses to serve a game whose positions it cannot
rebuild; its catalogue entry says so.

## Viewers

A frame is built for exactly one viewer.

| `Viewer.kind` | `Viewer.seat` | Sees |
|---|---|---|
| `oracle` | `null` | the full state |
| `seat` | the seat index | what that player may observe |
| `spectator` | `null` | public state only |

`Frame.omniscient` is true exactly for the oracle and for every viewer of a
game that hides nothing; `session.hides_information` says which games hide
something, and the frame refuses an `omniscient` flag that disagrees. A game
hides nothing when it declares perfect information and keeps no private
state; every other game is treated as hiding information.

### What a viewer that is not `omniscient` receives

| Field | Seat `s` | Spectator |
|---|---|---|
| `session.seed` | `null` | `null` |
| `state_text`, `struct` | `null` | `null` |
| `observation_text`, `information_state_text` | `s`'s own, where the game provides them | `null` |
| `legal_actions` | the actions when `s` is to move, otherwise `null` | `null` |
| `chance_outcomes` | `null` | `null` |
| `history[i]` | in full when `s` acted, otherwise `action`, `label` and `rewards` are `null` | `action`, `label` and `rewards` `null` |
| `rewards` | as the history entry that reached the position | as for a seat |
| `seats[i].failure.message` | `null` | `null` |

An `omniscient` viewer receives everything, except that `session.seed` goes to
the oracle alone: the seed determines every chance outcome to come. A history
entry's `seat` is always given, because which seat moved is public. `returns`
are given to every viewer.

## Frame

A frame is one position of a session. It is **live** when `ply == num_plies`
and a **replay** of an earlier position otherwise.

| Field | Type | Meaning |
|---|---|---|
| `contract_version` | `2` | |
| `session` | `SessionInfo` | the session's fixed properties |
| `viewer` | `Viewer` | who the frame is for |
| `omniscient` | boolean | whether the viewer sees the full state |
| `status` | `playing` \| `paused` \| `terminal` | the state of the live game |
| `seats` | `SeatView[]` | every seat, in seat order |
| `ply` | integer | actions applied to reach this position, chance included |
| `num_plies` | integer | actions in the live game |
| `to_move` | `ToMove` | who acts at this position |
| `legal_actions` | `Action[]` \| null | the acting seat's choices |
| `chance_outcomes` | `ChanceOutcome[]` \| null | the choices at an exposed chance node |
| `history` | `HistoryEntry[]` | every action of the live game |
| `rewards` | number[] \| null | each seat's reward for the action that reached this position |
| `returns` | number[] \| null | each seat's return at a terminal position |
| `state_text` | string \| null | the full state as text |
| `observation_text` | string \| null | the viewer's observation as text |
| `information_state_text` | string \| null | the viewer's information state as text |
| `struct` | any JSON \| null | the full state as the game's structured JSON |

The texts and `struct` are the game's own. The texts are for display only;
`struct` follows the game's struct definition where it has one.

| Object | Fields |
|---|---|
| `SessionInfo` | `id`, `game` (short name), `parameters`, `num_seats`, `chance_policy`, `turn_based`, `hides_information`, `seed`, `controllers` |
| `Viewer` | `kind`, `seat` |
| `SeatView` | `seat`, `controller` (a `Controller.name`), `config`, `autoplay`, `failure` |
| `Controller` | `name` (stable identifier), `label`, `config_schema` (a JSON Schema object whose `properties` give each setting's `type`, `default` and bounds), `unavailable_reason` (why it cannot play the game, or null) |
| `ControllerFailure` | `error` (exception name), `message` |
| `ToMove` | `kind` (`seat` \| `chance` \| `terminal`), `seat` |
| `Action` | `id` (the OpenSpiel action id), `label` |
| `ChanceOutcome` | `id`, `label`, `probability` |
| `HistoryEntry` | `seat` (null for chance), `action`, `label`, `rewards` |

### Invariants

- `Viewer.seat` is set exactly when `kind` is `seat`; likewise `ToMove.seat`.
- `omniscient` is true exactly when the viewer is the oracle or
  `session.hides_information` is false. `state_text` is non-null exactly
  when `omniscient`; `struct` is null unless `omniscient`.
- `session.seed` is null unless the viewer is the oracle.
- `observation_text` and `information_state_text` are null unless the viewer
  holds a seat.
- An `omniscient` viewer gets non-null `legal_actions` and `chance_outcomes`,
  and every history entry in full.
- A viewer that is not `omniscient` gets no chance outcomes and no failure
  messages, gets legal actions only as the seat to move, and, holding a seat,
  sees its own actions in full.
- A history entry withholds its `action`, `label` and `rewards` together.
- `0 <= ply <= num_plies == len(history)`.
- `rewards` equals `history[ply - 1].rewards`, and is null at `ply == 0`.
- `legal_actions` is non-empty only at a position where a seat acts, and on
  a replay only for an `omniscient` viewer, who may play from there.
- `chance_outcomes` is non-empty only at a chance node under `expose`.
- `returns` is non-null exactly at a terminal position. `rewards`, `returns`
  and every non-null `HistoryEntry.rewards` have one entry per seat.
- On a live frame, `status` is `terminal` exactly when `to_move.kind` is.
- `status` is `paused` exactly when some seat has a `failure`.
- `seats` lists seats `0 .. num_seats - 1` in order, each played by a
  controller in `session.controllers`, which includes `human`. A human seat
  never autoplays.

## Chance

| `chance_policy` | Behaviour |
|---|---|
| `auto` | The server samples every outcome from a generator seeded with `session.seed`. Frames never rest at a chance node. |
| `expose` | The server stops at each chance node, and the outcome is chosen through the chance route. |

## Simultaneous games

Simultaneous-move games are served through OpenSpiel's turn-based wrapper:
seats act one after another, and a seat's choice is withheld from the others
like any private action. `session.turn_based` is true for such sessions.

## Controllers

`session.controllers` lists every controller, starting with `human`. One
that cannot play the game says why in `unavailable_reason`, which is null
for every controller that can; no seat is played by one with a reason, and
a request for one is refused with `controller_unavailable`. A seat's
`config` satisfies its controller's `config_schema`. An `autoplay` seat moves
as soon as it is to move; any other non-human seat moves on an advance. One
request plays at most as many seat actions as an advance may, so autoplay
can stop with an `autoplay` seat to move, which then moves on an advance; so
can an undo, which plays nothing.

A seat's `controller`, `config` and `autoplay` go to every viewer, so a
controller's settings hold no secret. A controller draws its randomness from
a seed derived from `session.seed`, which only the oracle sees, never from
its `config`.

A controller fails when it raises, returns something other than an action
id, or returns an illegal action. Its seat is then **paused**: `failure` is
set, `status` becomes `paused`, the request that met the failure answers
`controller_failed`, and nothing plays in the seat's place. Actions already
applied stand. Play resumes after an undo, a reset, or a change of that
seat's controller.

## Catalogue

`Catalogue.games` lists every registered game once, ordered by `name`.

| Field | Meaning |
|---|---|
| `name` | short name, used to create a session |
| `long_name` | display name |
| `dynamics` | `sequential` \| `simultaneous` \| `mean_field` |
| `chance_mode` | `deterministic` \| `explicit_stochastic` \| `sampled_stochastic` |
| `information` | `one_shot` \| `perfect_information` \| `imperfect_information`; `imperfect_information` for a game that declares perfect information but is served as hiding it (`session.hides_information`) |
| `min_seats`, `max_seats` | supported seat counts |
| `num_players` | seat count with the default parameters, between `min_seats` and `max_seats`; null exactly when not `loadable` |
| `parameters` | `ParameterSpec[]`, ordered by name |
| `loadable` | whether the server can serve the game |
| `unavailable_reason` | why not; null exactly when `loadable` |
| `controllers` | controller names that can play the default game, including `human` exactly when `loadable` |

A `ParameterSpec` describes one game parameter so a client can offer it:

| Field | Meaning |
|---|---|
| `name` | the key in `CreateSession.parameters` |
| `type` | `bool` \| `int` \| `float` \| `string` |
| `kind` | `free` (any value of its type), `range` or `choice` |
| `default` | the value used when none is given, of type `type` |
| `minimum`, `maximum` | inclusive bounds around `default`, set exactly when `kind` is `range` |
| `choices` | the admitted values, including `default`, set exactly when `kind` is `choice` |
| `description` | what the parameter controls, or null |

## Health

| Field | Meaning |
|---|---|
| `version` | the installed OpenSpiel package's version |
| `revision` | the source revision the server runs from, or null when unknown |
| `capabilities.controllers` | every registered `Controller`, `human` first |
| `capabilities.chance_policies` | the chance policies the server accepts |
| `capabilities.python_game_search` | whether search controllers can play games written in Python |

## Errors

An error has a `code`, a `message` safe to show the requester, and the
`seat` and history index `ply` it concerns, each null when not applicable.
A message never holds state the requester may not see: it names only public
facts and what the request itself named, such as its action id. A
`controller_failed` message names only the failing seat; what the controller
raised reaches only an `omniscient` viewer, as that seat's `failure.message`.

A message never repeats what OpenSpiel or a game raised, whose text may name
files or other details of the server. When a game refuses its parameters, or
cannot start with them, the message says so and the server logs the detail.
A request the server fails on, rather than refuses, is answered `500` with a
plain-text body that points to the server's log.

A parameter that names a file or directory on the server (listed in
`SERVER_PATH_PARAMETERS` in [`loader.py`](loader.py)) admits only its
default. The catalogue offers it as a `choice` of that one value, and any
other value is a `bad_request`.

| `code` | HTTP | Meaning |
|---|---|---|
| `bad_request` | 400 | the request is malformed |
| `not_found` | 404 | no such session, game, seat or ply |
| `method_not_allowed` | 405 | the path exists but does not accept the method |
| `game_unavailable` | 409 | the game cannot be served |
| `viewer_unavailable` | 409 | the game cannot build a frame for this viewer |
| `not_your_turn` | 409 | the viewer may not choose the next action |
| `illegal_action` | 409 | the action is not legal at the position |
| `controller_unavailable` | 409 | the controller cannot play the game, or its `config` is invalid |
| `controller_failed` | 409 | a seat's controller failed and the session is paused |
| `nothing_to_undo` | 409 | the live game has no action to undo |

## Sweep exceptions

The sweep (`sweep.py`) plays a game through a session and checks every
frame. `python -m open_spiel.python.gui.sweep_all` sweeps every registered
game, or the games it names; it takes minutes, so it is a manual command, not
a test. `games_smoke_test.py` sweeps a few representative games as a test.
The games the sweep is expected to fail on are listed in
[`sweep_exceptions.json`](sweep_exceptions.json), an object keyed by game
short name and ordered by key:

```json
{
  "<game>": {"kind": "unavailable", "reason": "<why>"},
  "<other_game>": {"kind": "backend", "reason": "<why>", "outcome": "unfinished"}
}
```

| `kind` | Meaning |
|---|---|
| `unavailable` | the catalogue marks the game not loadable |
| `backend` | the server opens a session, but the sweep fails on it |
| `browser` | the sweep passes, but a client playing through this API fails on it |

A `backend` entry, and only one, names the `outcome` the sweep ends with:

| `outcome` | The sweep ends |
|---|---|
| `unfinished` | with the game not over at the ply cap |
| `invariant` | when a frame breaks a check |
| `paused` | when a seat's controller fails |
| `error` | when the session raises |

The list may name games that some builds of OpenSpiel register and others
do not. A listed game the running build does not register is not judged; the
summary names it ("listed but not registered in this build"). Every game the
build registers is judged, listed or not. A run of every game also checks
that the registered `unavailable` games are exactly those the catalogue marks
not loadable. The `reason` is written for this list and is not compared with
the catalogue's `unavailable_reason`.

A game is a problem when it is not listed and fails the sweep, and when it is
listed and passes or ends another way than listed. A sweep that runs out of
wall-clock time before reaching any of these ends `too_slow`: that says
nothing about the game, which is reported as not judged. `sweep_all` prints a
line per game and a summary, and exits with status 1 if any game is a problem
or was not judged.

## Changing the contract

Edit the definitions, regenerate the schema with
`python -m open_spiel.python.gui.schema`, and update this document.
`schema_test.py` fails while the checked-in schema is stale.
