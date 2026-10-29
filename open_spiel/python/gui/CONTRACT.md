# GUI wire contract

The game server speaks JSON over HTTP. Each response body is a **frame**, the
**catalogue**, the **health** report, a **closed session** or an **error**.
This document is the specification of those bodies, of the request bodies,
and of the routes.

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

Every response carries `contract_version`, currently `8`. It increases on any
change a client could break on: a field added, removed, renamed or retyped,
an enum value added or removed, or a route changed. The server and the
client change version together: the client refuses a response of any other
version.

| Version | Change                                                                                                                                       |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `8`     | `SeatView.notice`, a controller's note about its latest move, sent only to that seat's viewer and to omniscient viewers; `DELETE /api/sessions/{id}`, answered with `SessionClosed` |
| `7`     | `CatalogueEntry.renderer`: the renderer the game's adapter names, so a client can show which games it draws as a board before opening one |
| `6`     | `Health.capabilities.cpp_mcts_limits` (`simulations_or_clock` \| `simulations_only`): how the served pyspiel's C++ MCTS bounds a search, which sets MCTS's limits |
| `5`     | `Health.capabilities.python_game_search` removed: MCTS picks its search by how the game is implemented and plays games implemented in Python as in C++ |
| `4`     | `SeatView.status` (`ready` \| `preparing` \| `unavailable`) and `SeatView.message`: a seat's controller is built off the request                    |
| `3`     | `Frame.observation_struct`, a seat's observation as the game's structured JSON; `Frame.presentation`, the game adapter's view for the viewer |
| `2`     | The current form of every other body                                                                                                         |

## Routes

Every session route takes the viewer in the required `viewer` query
parameter: `oracle`, `spectator`, or a seat index. Each route answers `200`
with the response shown, or an error. A path below that exists, requested
with a method it does not accept, is refused with `method_not_allowed`; a
path that does not exist, with `not_found`.

| Method | Path                              | Request              | Response          | Errors                                                                                                   |
| ------ | --------------------------------- | -------------------- | ----------------- | -------------------------------------------------------------------------------------------------------- |
| `GET`  | `/api/health`                     |                      | `Health`          |                                                                                                          |
| `GET`  | `/api/games`                      |                      | `Catalogue`       |                                                                                                          |
| `POST` | `/api/sessions`                   | `CreateSession`      | `Frame`, live     | `bad_request`, `not_found`, `game_unavailable`, `viewer_unavailable`, `controller_unavailable`           |
| `GET`  | `/api/sessions/{id}`              |                      | `Frame`, live     | `bad_request`, `not_found`, `viewer_unavailable`                                                         |
| `GET`  | `/api/sessions/{id}/plies/{ply}`  |                      | `Frame`, at `ply` | `bad_request`, `not_found`, `viewer_unavailable`                                                         |
| `POST` | `/api/sessions/{id}/actions`      | `ApplyAction`        | `Frame`, live     | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `illegal_action`, `controller_failed` |
| `POST` | `/api/sessions/{id}/chance`       | `ApplyChanceOutcome` | `Frame`, live     | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `illegal_action`, `controller_failed` |
| `POST` | `/api/sessions/{id}/advance`      | `Advance`            | `Frame`, live     | `bad_request`, `not_found`, `viewer_unavailable`, `controller_failed`                                    |
| `POST` | `/api/sessions/{id}/undo`         |                      | `Frame`, live     | `bad_request`, `not_found`, `viewer_unavailable`, `nothing_to_undo`                                      |
| `POST` | `/api/sessions/{id}/reset`        | `Reset`              | `Frame`, live     | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `illegal_action`                      |
| `PUT`  | `/api/sessions/{id}/seats/{seat}` | `SeatConfig`         | `Frame`, live     | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`, `controller_unavailable`              |
| `DELETE` | `/api/sessions/{id}`            |                      | `SessionClosed`   | `bad_request`, `not_found`, `viewer_unavailable`, `not_your_turn`                                        |

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
- **Close** forgets the session; its id is `not_found` from then on. As
  with reset, any viewer but a spectator closes it; a spectator is refused
  with `not_your_turn`. The answer is `SessionClosed`: `contract_version`
  and the `id` closed. Like a change, a close from another origin is
  refused.
- **Health** identifies the running code, so a client or sweep can refuse a
  server it does not expect.

A replay (`/plies/{ply}`) is rebuilt from the history and never changes the
session. The server refuses to serve a game whose positions it cannot
rebuild; its catalogue entry says so.

## Viewers

A frame is built for exactly one viewer.

| `Viewer.kind` | `Viewer.seat`  | Sees                         |
| ------------- | -------------- | ---------------------------- |
| `oracle`      | `null`         | the full state               |
| `seat`        | the seat index | what that player may observe |
| `spectator`   | `null`         | public state only            |

`Frame.omniscient` is true exactly for the oracle and for every viewer of a
game that hides nothing; `session.hides_information` says which games hide
something, and the frame refuses an `omniscient` flag that disagrees. A game
hides nothing when it declares perfect information and keeps no private
state; every other game is treated as hiding information.

### What a viewer that is not `omniscient` receives

| Field                                        | Seat `s`                                                                     | Spectator                              |
| -------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------- |
| `session.seed`                               | `null`                                                                       | `null`                                 |
| `state_text`, `struct`                       | `null`                                                                       | `null`                                 |
| `observation_struct`                         | `s`'s own, where the game provides it                                        | `null`                                 |
| `presentation.view`                          | what `s` may see: its own private information and what is public             | what is public                         |
| `presentation.actions`                       | hints for `s`'s legal actions when `s` is to move, otherwise `null`          | `null`                                 |
| `observation_text`, `information_state_text` | `s`'s own, where the game provides them                                      | `null`                                 |
| `legal_actions`                              | the actions when `s` is to move, otherwise `null`                            | `null`                                 |
| `chance_outcomes`                            | `null`                                                                       | `null`                                 |
| `history[i]`                                 | in full when `s` acted, otherwise `action`, `label` and `rewards` are `null` | `action`, `label` and `rewards` `null` |
| `rewards`                                    | as the history entry that reached the position                               | as for a seat                          |
| `seats[i].failure.message`                   | `null`                                                                       | `null`                                 |
| `seats[i].notice`                            | seat `s`'s own (`i == s`), otherwise `null`                                  | `null`                                 |

An `omniscient` viewer receives everything, except that `session.seed` goes to
the oracle alone: the seed determines every chance outcome to come. A history
entry's `seat` is always given, because which seat moved is public. A
withheld entry's `public_label` is what the rules let every viewer read of
the action, as the game's adapter says (`Adapter.public_action_label`): a
card discarded face up, a draw from a face-up pile. It is null when the
action is wholly hidden, and for a game whose adapter makes nothing public.
`returns` are given to every viewer.

## Frame

A frame is one position of a session. It is **live** when `ply == num_plies`
and a **replay** of an earlier position otherwise.

| Field                    | Type                                | Meaning                                                      |
| ------------------------ | ----------------------------------- | ------------------------------------------------------------ |
| `contract_version`       | `8`                                 |                                                              |
| `session`                | `SessionInfo`                       | the session's fixed properties                               |
| `viewer`                 | `Viewer`                            | who the frame is for                                         |
| `omniscient`             | boolean                             | whether the viewer sees the full state                       |
| `status`                 | `playing` \| `paused` \| `terminal` | the state of the live game                                   |
| `seats`                  | `SeatView[]`                        | every seat, in seat order                                    |
| `ply`                    | integer                             | actions applied to reach this position, chance included      |
| `num_plies`              | integer                             | actions in the live game                                     |
| `to_move`                | `ToMove`                            | who acts at this position                                    |
| `legal_actions`          | `Action[]` \| null                  | the acting seat's choices                                    |
| `chance_outcomes`        | `ChanceOutcome[]` \| null           | the choices at an exposed chance node                        |
| `history`                | `HistoryEntry[]`                    | every action of the live game                                |
| `rewards`                | number[] \| null                    | each seat's reward for the action that reached this position |
| `returns`                | number[] \| null                    | each seat's return at a terminal position                    |
| `state_text`             | string \| null                      | the full state as text                                       |
| `observation_text`       | string \| null                      | the viewer's observation as text                             |
| `information_state_text` | string \| null                      | the viewer's information state as text                       |
| `struct`                 | any JSON \| null                    | the full state as the game's structured JSON                 |
| `observation_struct`     | any JSON \| null                    | the viewer's observation as the game's structured JSON       |
| `presentation`           | `Presentation` \| null              | the position as the game's adapter presents it to the viewer |

The texts and the structs are the game's own. The texts are for display
only; `struct` and `observation_struct` follow the game's struct definitions
(OpenSpiel's `StateStruct` and `ObservationStruct`) where it has them. A
client draws a game from the structs, and reads the texts only for a game
that has none.

`observation_struct` is what the game's `ToObservationStruct` gives the
viewer's seat, so it holds what that seat may observe and nothing more. The
oracle reads `struct` instead and is given no observation struct. A spectator
holds no seat, and OpenSpiel defines observations per player only, so a
spectator is given none either; in a game that hides nothing a spectator is
omniscient and reads `struct`. A seat of such a game receives both.

`presentation` is built by the game's adapter (`presenters.py`, one module
per game in `adapters/`), on the server, for the one viewer the frame is
for; it is null for a game with no adapter. `renderer` names the client
renderer that draws it, `view` is the adapter's own JSON and holds only what
the viewer may see, and `actions` gives a hint, the adapter's own JSON, for
some or all of the frame's `legal_actions` (a card, a pair of cells), so
that the renderer can map a press to an action. A game's rules may make
private information public, as a knock does in gin rummy; the adapter then
shows it to every viewer.

| Object              | Fields                                                                                                                                                                                                                                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SessionInfo`       | `id`, `game` (short name), `parameters`, `num_seats`, `chance_policy`, `turn_based`, `hides_information`, `seed`, `controllers`                                                                                                                                                                          |
| `Viewer`            | `kind`, `seat`                                                                                                                                                                                                                                                                                           |
| `SeatView`          | `seat`, `controller` (a `Controller.name`), `config`, `autoplay`, `failure`, `status` (`ready` \| `preparing` \| `unavailable`), `message` (why the seat is not ready, or null), `notice` (the controller's note about its latest move, at most 200 characters, or null)                             |
| `Controller`        | `name` (stable identifier), `label`, `config_schema` (a JSON Schema object whose `properties` give each setting's `type`, `default` and constraints, and whose `required` lists the settings a seat must be given, which have no `default`), `unavailable_reason` (why it cannot play the game, or null) |
| `ControllerFailure` | `error` (exception name), `message`                                                                                                                                                                                                                                                                      |
| `ToMove`            | `kind` (`seat` \| `chance` \| `terminal`), `seat`                                                                                                                                                                                                                                                        |
| `Action`            | `id` (the OpenSpiel action id), `label`                                                                                                                                                                                                                                                                  |
| `ChanceOutcome`     | `id`, `label`, `probability`                                                                                                                                                                                                                                                                             |
| `HistoryEntry`      | `seat` (null for chance), `action`, `label`, `rewards`, `public_label`                                                                                                                                                                                                                                   |
| `Presentation`      | `renderer`, `view` (any JSON), `actions` (`ActionHint[]` or null)                                                                                                                                                                                                                                        |
| `ActionHint`        | `id` (a legal action's id), `hint` (any JSON)                                                                                                                                                                                                                                                            |

### Invariants

- `Viewer.seat` is set exactly when `kind` is `seat`; likewise `ToMove.seat`.
- `omniscient` is true exactly when the viewer is the oracle or
  `session.hides_information` is false. `state_text` is non-null exactly
  when `omniscient`; `struct` is null unless `omniscient`.
- `session.seed` is null unless the viewer is the oracle.
- `observation_text`, `information_state_text` and `observation_struct` are
  null unless the viewer holds a seat.
- `presentation.actions` is null exactly when `legal_actions` is null, and
  each hint names a legal action of the frame, once.
- An `omniscient` viewer gets non-null `legal_actions` and `chance_outcomes`,
  and every history entry in full.
- A viewer that is not `omniscient` gets no chance outcomes and no failure
  messages, gets legal actions only as the seat to move, and, holding a seat,
  sees its own actions in full. It gets no seat's `notice` but its own.
- A human seat has no `notice`.
- A history entry withholds its `action`, `label` and `rewards` together;
  only a withheld entry has a non-null `public_label`.
- `0 <= ply <= num_plies == len(history)`.
- `rewards` equals `history[ply - 1].rewards`, and is null at `ply == 0`.
- `legal_actions` is non-empty only at a position where a seat acts, and on
  a replay only for an `omniscient` viewer, who may play from there.
- `chance_outcomes` is non-empty only at a chance node under `expose`.
- `returns` is non-null exactly at a terminal position. `rewards`, `returns`
  and every non-null `HistoryEntry.rewards` have one entry per seat.
- On a live frame, `status` is `terminal` exactly when `to_move.kind` is.
- `status` is `paused` exactly when it is not `terminal` and some seat has a
  `failure`.
- A seat's `message` is null when its `status` is `ready`. A `preparing`
  seat has no `failure`; an `unavailable` seat has a `failure` and a
  `message`. A human seat is always `ready`.
- `seats` lists seats `0 .. num_seats - 1` in order, each played by a
  controller in `session.controllers`, which includes `human`. A human seat
  never autoplays.

## Chance

| `chance_policy` | Behaviour                                                                                                         |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `auto`          | The server samples every outcome from a generator seeded with `session.seed`. Frames never rest at a chance node. |
| `expose`        | The server stops at each chance node, and the outcome is chosen through the chance route.                         |

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

A `config_schema` is `{}` or an object schema whose `properties` each give a
`type` (`boolean`, `integer`, `number` or `string`), a `description` and,
unless the setting is listed in the schema's `required`, a `default`. A
number may give `minimum` and `maximum`; a string `minLength`, `maxLength`
(in characters) and a `pattern` that it must contain a match of; a string or
a number an `enum` of the only values it takes. A setting listed in
`required` must be in a seat's `config`: a seat given without it is refused
with `controller_unavailable`, naming the setting. Every other setting a
seat is given without plays at its `default`.

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

A controller may give a **notice** with a move: a line of plain text, at most
200 characters, such as a remote bot reporting a fallback it took. The seat
shows it in `notice` until the seat moves again, or an undo, a reset or a
change of controller clears it. It reaches only that seat's viewer and
omniscient viewers, and it has no effect on play. A notice that is not
non-blank text of at most 200 characters fails the seat (`InvalidNotice`).
A notice is text the controller writes, so a controller that reads the full
state (a `remote` seat sent `full_state`, say) can put hidden information in
it, which then reaches that seat's viewer. Every notice is passed on, as
that seat's `config` says what it reads; the server's state policy
(`controller_registry.full_state_notice`) is the place to drop or limit such
a seat's notices.

The server registers four controllers: `human`, `random` (a uniformly random
legal action), `mcts`, and `remote`, a bot server the seat's `url` names, sent
what the seat's `state` setting says: `seat_view`, what the seat may see (its
information state string, legal actions and a seed, with more on request), or
`full_state`, the whole state ([REMOTE_BOTS.md](REMOTE_BOTS.md) gives the
protocol). A `remote` seat may be sent the full state in a game that hides
information; its `config`, which every viewer receives, says which it is sent.
The server posts to bot servers on its own machine only, unless started with
`--remote-bot-host`. MCTS = pyspiel C++ MCTSBot for C++ games; OpenSpiel Python
MCTS for Python-implemented games. Its limits follow a capability check the
server makes once at startup, reported as `Health.capabilities.cpp_mcts_limits`.

A game implemented in C++ is searched by OpenSpiel's C++ search
(`pyspiel.MCTSBot`), its tree capped at 1000 MB. That search cannot be
stopped once started, and its only bounds are a simulation count and a
wall-clock time. Some pyspiel builds stop it at whichever comes first;
others, OpenSpiel's releases so far among them, let a wall-clock time
replace the count. At startup the server searches tic-tac-toe with 5
simulations and 0.2 seconds, and counts the simulations from the root's
visit count: 5, in well under a millisecond, means the count is honoured;
thousands, over the full 0.2 seconds, means the clock replaces it.

| `cpp_mcts_limits`      | `max_simulations`              | Wall-clock time                         | Moves reproducible                |
| ---------------------- | ------------------------------ | --------------------------------------- | --------------------------------- |
| `simulations_or_clock` | at most 10000, 1000 by default | the move budget less 1 s, as a backstop | when the simulations finish first |
| `simulations_only`     | at most 250, its default       | none                                    | always                            |

With `simulations_only`, `max_simulations` is capped at 250: a move at the
cap fits the 10 second move budget on the slowest catalogued C++ game MCTS
plays (`hive`, one random rollout per simulation) with room to spare, also
on a host half as fast as a typical server. Every other catalogued C++ game
MCTS plays is faster.

With `simulations_or_clock`, a slow game's move at a high count ends at the
backstop instead (half the budget when the budget is under 2 s).

A game implemented in Python, or a C++ game that wraps one, is searched by
OpenSpiel's Python MCTS (`open_spiel.python.algorithms.mcts`) with its random
rollouts played in C++: the C++ search releases Python's interpreter lock
while it runs, and cannot call such a game safely. The Python search is
cancellable at the move budget: it stops between simulations once the
budget has expired, rather than searching on with no one waiting. It
takes the same `max_simulations` setting, with the same bounds, and it stops
once its tree holds 1,000,000 nodes (about 180 MB), playing the best move
found so far. It has no wall-clock limit of its own: with
`simulations_or_clock`, a move whose simulations outlast the move budget
fails its seat with `Timeout`.

Apart from the backstop, both searches count work, not time, so a move is
reproducible from the seed on any machine.

### Building a controller

A seat's controller is built when the seat is given it, by create or seats,
and a build can take a while: a bot that plays a trained model loads and
warms it up. The request checks the controller and its settings at once,
refusing one that cannot play with `controller_unavailable`, then builds it
on a thread of its own and waits for it a quarter of a second at most. A
build that finishes by then is used at once: a request seats a fast
controller as if it built it itself, and a build that fails by then refuses
the request with `controller_unavailable`, leaving the seat as it was.
Otherwise the request answers with the seat **preparing**:

| `SeatView.status` | Meaning                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ready`           | The controller is built and plays the seat; a human seat is always ready. `message` is null.                                                     |
| `preparing`       | The controller is being built. The seat does not move, and an advance stops before it; `message` may say what the server is doing.               |
| `unavailable`     | The build failed or overran the build budget. `failure` names the error, `message` says why to every viewer, and the session is paused. |

A preparing seat becomes `ready` when its build finishes, and autoplay goes
on from there, with no request: a client that shows a preparing seat fetches
the live frame again until it is ready. A build has a budget, 120 seconds
unless the server is started with `--build-budget-seconds`; one that raises,
or has not finished within it, leaves its seat `unavailable`. Its `failure`
is `UnavailableError` for a controller that finds it cannot play after all
(its `message` is the reason, as a refusal would give it), `Timeout` for one
over budget, and the exception's name for any other, whose `message` names
only the exception: what it said may describe the seed, and reaches only an
omniscient viewer, in `failure.message`. An undo or a reset does not resume
an unavailable seat, which has no controller to resume; giving it a
controller does.

Giving a preparing seat another controller supersedes its build: what the
build returns is discarded, and the seat waits only on its latest
controller.

## Catalogue

`Catalogue.games` lists every registered game once, ordered by `name`,
except the games the server is started with `--exclude-games` to leave out,
whose sessions it refuses as `not_found`.
The server loads each game once per process and keeps what it finds, so a
game registered after the first catalogue is not listed until a restart.
`renderer` is not kept: it is read from the adapter registry on every
request, so an adapter registered or removed while serving shows in the
next catalogue.

| Field                    | Meaning                                                                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                   | short name, used to create a session                                                                                                                                                          |
| `long_name`              | display name                                                                                                                                                                                  |
| `dynamics`               | `sequential` \| `simultaneous` \| `mean_field`                                                                                                                                                |
| `chance_mode`            | `deterministic` \| `explicit_stochastic` \| `sampled_stochastic`                                                                                                                              |
| `information`            | `one_shot` \| `perfect_information` \| `imperfect_information`; `imperfect_information` for a game that declares perfect information but is served as hiding it (`session.hides_information`) |
| `min_seats`, `max_seats` | supported seat counts                                                                                                                                                                         |
| `num_players`            | seat count with the default parameters, between `min_seats` and `max_seats`; null exactly when not `loadable`                                                                                 |
| `parameters`             | `ParameterSpec[]`, ordered by name                                                                                                                                                            |
| `loadable`               | whether the server can serve the game                                                                                                                                                         |
| `unavailable_reason`     | why not; null exactly when `loadable`                                                                                                                                                         |
| `controllers`            | controller names that can play the default game, including `human` exactly when `loadable`                                                                                                    |
| `renderer`               | the `Presentation.renderer` its frames name, from the game's registered adapter; null for a game with no adapter and for one not `loadable`, which a client shows as its text                |

A `ParameterSpec` describes one game parameter so a client can offer it:

| Field                | Meaning                                                                       |
| -------------------- | ----------------------------------------------------------------------------- |
| `name`               | the key in `CreateSession.parameters`                                         |
| `type`               | `bool` \| `int` \| `float` \| `string`                                        |
| `kind`               | `free` (any value of its type), `range` or `choice`                           |
| `default`            | the value used when none is given, of type `type`                             |
| `minimum`, `maximum` | inclusive bounds around `default`, set exactly when `kind` is `range`         |
| `choices`            | the admitted values, including `default`, set exactly when `kind` is `choice` |
| `description`        | what the parameter controls, or null                                          |

## Health

| Field                          | Meaning                                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------------------- |
| `version`                      | the installed OpenSpiel package's version                                                    |
| `revision`                     | the source revision the server runs from, or null when unknown                               |
| `capabilities.controllers`     | every registered `Controller`, `human` first                                                 |
| `capabilities.chance_policies` | the chance policies the server accepts                                                       |
| `capabilities.cpp_mcts_limits` | how the C++ MCTS bounds a search (Controllers): `simulations_or_clock` or `simulations_only` |

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

| `code`                   | HTTP | Meaning                                                         |
| ------------------------ | ---- | --------------------------------------------------------------- |
| `bad_request`            | 400  | the request is malformed                                        |
| `not_found`              | 404  | no such session, game, seat or ply                              |
| `method_not_allowed`     | 405  | the path exists but does not accept the method                  |
| `game_unavailable`       | 409  | the game cannot be served                                       |
| `viewer_unavailable`     | 409  | the game cannot build a frame for this viewer                   |
| `not_your_turn`          | 409  | the viewer may not choose the next action                       |
| `illegal_action`         | 409  | the action is not legal at the position                         |
| `controller_unavailable` | 409  | the controller cannot play the game, or its `config` is invalid |
| `controller_failed`      | 409  | a seat's controller failed and the session is paused            |
| `nothing_to_undo`        | 409  | the live game has no action to undo                             |

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
  "<game>": { "kind": "unavailable", "reason": "<why>" },
  "<other_game>": {
    "kind": "backend",
    "reason": "<why>",
    "outcome": "unfinished"
  }
}
```

| `kind`        | Meaning                                                             |
| ------------- | ------------------------------------------------------------------- |
| `unavailable` | the catalogue marks the game not loadable                           |
| `backend`     | the server opens a session, but the sweep fails on it               |
| `browser`     | the sweep passes, but a client playing through this API fails on it |

A `backend` entry, and only one, names the `outcome` the sweep ends with:

| `outcome`    | The sweep ends                        |
| ------------ | ------------------------------------- |
| `unfinished` | with the game not over at the ply cap |
| `invariant`  | when a frame breaks a check           |
| `paused`     | when a seat's controller fails        |
| `error`      | when the session raises               |

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
`python -m open_spiel.python.gui.schema`, regenerate the client's types with
`npm run generate:contract` in `open_spiel/gui`, and update this document. A
change a client could break on also raises `CONTRACT_VERSION` in
`contract.py` and in the client's `src/session/decode.ts`, together.
`schema_test.py` fails while the checked-in schema is stale, and the client's
contract test while its types are.
