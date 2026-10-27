// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Generated from open_spiel/python/gui/contract.schema.json by
// scripts/generate_contract.ts. Do not edit; run npm run generate:contract.

/**
 * Every response body the server sends.
 */
export type ResponseBody = Frame | Catalogue | Health | Error;

/**
 * A legal action at a position.
 */
export interface Action {
  /**
   * The OpenSpiel action id, which a client sends to play it.
   */
  id: number;
  /**
   * The game's text for the action.
   */
  label: string;
}

/**
 * What a game's adapter says about one legal action, for its renderer.
 */
export interface ActionHint {
  /**
   * The legal action's OpenSpiel id.
   */
  id: number;
  /**
   * The adapter's description of the action, such as the card it plays or the cells it joins.
   */
  hint: unknown;
}

/**
 * Lets controllers play their seats.
 *
 * Play stops when a human seat or an exposed chance node must act, the game ends, a controller fails, or `max_plies` seat actions have been applied. Under the `auto` policy, chance outcomes are sampled as they come and do not count.
 *
 * One request holds its session until play stops, so play is bounded.
 */
export interface Advance {
  /**
   * The most seat actions to apply, from 1 to 1000, or null for 1000.
   */
  max_plies: number | null;
}

/**
 * Plays an action for the seat to move.
 */
export interface ApplyAction {
  /**
   * The OpenSpiel action id, one of the frame's `legal_actions`.
   */
  action: number;
  /**
   * The position to play at, the `ply` of the frame the action was chosen from: null, or `num_plies`, for the live position. An earlier ply plays from that position and takes back every later ply, as only an omniscient viewer may.
   */
  ply: number | null;
}

/**
 * Chooses the outcome at a chance node under the `expose` policy.
 */
export interface ApplyChanceOutcome {
  /**
   * The OpenSpiel action id, one of the frame's `chance_outcomes`.
   */
  outcome: number;
  /**
   * The position to choose at, as for `ApplyAction`.
   */
  ply: number | null;
}

/**
 * What the server can do, independent of any one game.
 */
export interface Capabilities {
  /**
   * Every registered controller, starting with the human one.
   */
  controllers: Controller[];
  /**
   * The chance policies the server accepts.
   */
  chance_policies: ChancePolicy[];
  /**
   * How the served pyspiel's C++ MCTS bounds a search, and so which limits MCTS plays with.
   */
  cpp_mcts_limits: MctsLimits;
}

/**
 * Every game the server knows.
 */
export interface Catalogue {
  /**
   * The contract version the catalogue was built against.
   */
  contract_version: 7;
  /**
   * One entry per game, ordered by name.
   */
  games: CatalogueEntry[];
}

/**
 * A game the server knows, and whether it can serve it.
 */
export interface CatalogueEntry {
  /**
   * The game's short name, which a client sends to create a session.
   */
  name: string;
  /**
   * The game's display name.
   */
  long_name: string;
  /**
   * How players move.
   */
  dynamics: Dynamics;
  /**
   * How the game uses chance.
   */
  chance_mode: ChanceMode;
  /**
   * What the players observe.
   */
  information: Information;
  /**
   * The fewest seats the game supports.
   */
  min_seats: number;
  /**
   * The most seats the game supports.
   */
  max_seats: number;
  /**
   * The number of seats the game has with its default parameters, between `min_seats` and `max_seats`; null when not loadable.
   */
  num_players: number | null;
  /**
   * The game's parameters, ordered by name.
   */
  parameters: ParameterSpec[];
  /**
   * Whether the server can serve the game.
   */
  loadable: boolean;
  /**
   * Why the server cannot serve the game; null when loadable.
   */
  unavailable_reason: string | null;
  /**
   * The `Controller.name`s that can play the game with its default parameters; empty when not loadable.
   */
  controllers: string[];
  /**
   * The renderer its frames name in `Presentation.renderer`, from the game's registered adapter: the board a client draws it with, if the client has that renderer. Null for a game with no adapter, and for one that is not loadable; such a game is shown as its text.
   */
  renderer: string | null;
}

/**
 * How the game uses chance, as the game declares it.
 *
 * - `deterministic`: The game has no chance nodes.
 * - `explicit_stochastic`: Chance nodes list their outcomes and probabilities.
 * - `sampled_stochastic`: Chance outcomes are sampled inside the state.
 */
export type ChanceMode =
  'deterministic' | 'explicit_stochastic' | 'sampled_stochastic';

/**
 * A chance outcome at a chance node.
 */
export interface ChanceOutcome {
  /**
   * The OpenSpiel action id, which a client sends to choose it.
   */
  id: number;
  /**
   * The game's text for the outcome.
   */
  label: string;
  /**
   * The outcome's probability, in (0, 1].
   */
  probability: number;
}

/**
 * How chance nodes are resolved.
 *
 * - `auto`: The server samples each outcome from a seeded generator.
 * - `expose`: The client chooses each outcome; in a game that hides information, only from the oracle view.
 */
export type ChancePolicy = 'auto' | 'expose';

/**
 * A kind of agent that can play a seat.
 */
export interface Controller {
  /**
   * The stable identifier used in requests and in `SeatView.controller`.
   */
  name: string;
  /**
   * The human-readable name.
   */
  label: string;
  /**
   * A JSON Schema object that a seat's `config` for this controller must satisfy; `{}` when it takes no settings. Each of its `properties` gives a setting's `type`, `default` and constraints; a setting listed in its `required` has no `default`, and a seat must be given it.
   */
  config_schema: Record<string, unknown>;
  /**
   * Why the controller cannot play the session's game; null when it can. A seat is never played by a controller with a reason.
   */
  unavailable_reason: string | null;
}

/**
 * Why a seat's controller stopped playing.
 */
export interface ControllerFailure {
  /**
   * The name of the exception the controller raised.
   */
  error: string;
  /**
   * The exception's message; null for every viewer that is not omniscient, the failing seat included, since the message may describe any seat's private state.
   */
  message: string | null;
}

/**
 * Starts a session.
 */
export interface CreateSession {
  /**
   * The game's short name.
   */
  game: string;
  /**
   * Game parameters; omitted ones take their defaults.
   */
  parameters: Record<string, boolean | number | string>;
  /**
   * Who plays each seat, in seat order, one per seat of the game.
   */
  seats: SeatConfig[];
  /**
   * How chance nodes are resolved.
   */
  chance_policy: ChancePolicy;
  /**
   * The chance generator's seed, a non-negative integer, or null for the server to choose one.
   */
  seed: number | null;
}

/**
 * How players move, as the game declares it.
 *
 * - `sequential`: One player acts at a time.
 * - `simultaneous`: Players act together; served through the turn-based wrapper.
 * - `mean_field`: A mean-field game.
 */
export type Dynamics = 'sequential' | 'simultaneous' | 'mean_field';

/**
 * A refused request.
 */
export interface Error {
  /**
   * The contract version the error was built against.
   */
  contract_version: 7;
  /**
   * Why the request was refused.
   */
  code: ErrorCode;
  /**
   * A human-readable explanation, safe to show the requester: it never holds state the requester may not see.
   */
  message: string;
  /**
   * The seat the refusal concerns, or null.
   */
  seat: number | null;
  /**
   * The index of the history entry the refusal concerns, or null.
   */
  ply: number | null;
}

/**
 * Why a request was refused.
 *
 * - `bad_request`: The request is malformed.
 * - `not_found`: The session, game, seat or ply does not exist.
 * - `method_not_allowed`: The path exists but does not accept the method.
 * - `game_unavailable`: The game cannot be served; its catalogue entry says why.
 * - `viewer_unavailable`: The game cannot build a frame for the viewer.
 * - `not_your_turn`: The viewer may not choose the next action.
 * - `illegal_action`: The action is not legal at the position.
 * - `controller_unavailable`: The controller cannot play the game, or its settings do not satisfy its `config_schema`.
 * - `controller_failed`: A seat's controller failed and the session is paused.
 * - `nothing_to_undo`: The live game has no action to undo.
 */
export type ErrorCode =
  | 'bad_request'
  | 'not_found'
  | 'method_not_allowed'
  | 'game_unavailable'
  | 'viewer_unavailable'
  | 'not_your_turn'
  | 'illegal_action'
  | 'controller_unavailable'
  | 'controller_failed'
  | 'nothing_to_undo';

/**
 * A position of a session, built for one viewer.
 *
 * A frame is live when `ply == num_plies`, and a replay of an earlier position otherwise. A seat acts at the live position; an omniscient viewer may also play from a replay, which takes back the plies after it.
 */
export interface Frame {
  /**
   * The contract version the frame was built against.
   */
  contract_version: 7;
  /**
   * The session's fixed properties.
   */
  session: SessionInfo;
  /**
   * The audience the frame is built for.
   */
  viewer: Viewer;
  /**
   * Whether the viewer sees the full state: exactly when the viewer is the oracle or the game hides nothing (`session.hides_information` is false).
   */
  omniscient: boolean;
  /**
   * The state of the session's live game.
   */
  status: SessionStatus;
  /**
   * Every seat, in seat order.
   */
  seats: SeatView[];
  /**
   * The number of actions, chance outcomes included, applied to reach this position.
   */
  ply: number;
  /**
   * The number of actions in the live game.
   */
  num_plies: number;
  /**
   * Who acts at this position.
   */
  to_move: ToMove;
  /**
   * The actions the acting seat may choose; empty where no seat acts and, for a viewer that is not omniscient, on a replay; null when withheld from the viewer.
   */
  legal_actions: Action[] | null;
  /**
   * The outcomes the client may choose at a chance node under the `expose` policy; empty otherwise, null when withheld from the viewer.
   */
  chance_outcomes: ChanceOutcome[] | null;
  /**
   * Every action of the live game, in order, as the viewer may see it.
   */
  history: HistoryEntry[];
  /**
   * Each seat's reward for the action that reached this position, in seat order; null at the initial position or when that action is withheld from the viewer.
   */
  rewards: number[] | null;
  /**
   * Each seat's return, in seat order, at a terminal position; otherwise null.
   */
  returns: number[] | null;
  /**
   * The full state as text; null unless `omniscient`.
   */
  state_text: string | null;
  /**
   * The viewer's observation as text; null unless the viewer holds a seat and the game provides observations.
   */
  observation_text: string | null;
  /**
   * The viewer's information state as text; null unless the viewer holds a seat and the game provides information states.
   */
  information_state_text: string | null;
  /**
   * The full state as the game's structured JSON; null unless `omniscient` and the game provides it.
   */
  struct: unknown;
  /**
   * The viewer's observation as the game's structured JSON; null unless the viewer holds a seat and the game provides observation structs.
   */
  observation_struct: unknown;
  /**
   * The position as the game's adapter presents it to the viewer; null when no adapter is registered for the game.
   */
  presentation: Presentation | null;
}

/**
 * The server's identity and capabilities.
 */
export interface Health {
  /**
   * The contract version the server speaks.
   */
  contract_version: 7;
  /**
   * The installed OpenSpiel package's version.
   */
  version: string;
  /**
   * The source revision the server runs from, or null when unknown.
   */
  revision: string | null;
  /**
   * What the server can do.
   */
  capabilities: Capabilities;
}

/**
 * One action of the live game, as the viewer may see it.
 */
export interface HistoryEntry {
  /**
   * The seat that acted, or null for a chance outcome.
   */
  seat: number | null;
  /**
   * The OpenSpiel action id, or null when withheld from the viewer.
   */
  action: number | null;
  /**
   * The game's text for the action, or null when withheld.
   */
  label: string | null;
  /**
   * Each seat's reward for the action, in seat order, or null when withheld.
   */
  rewards: number[] | null;
  /**
   * What every viewer may read of a withheld action, as the game's adapter says, such as a card discarded face up; null when the action is shown in full or wholly hidden.
   */
  public_label: string | null;
}

/**
 * What the players observe, as the game declares it.
 *
 * - `one_shot`: A one-shot game.
 * - `perfect_information`: Every player observes the full state.
 * - `imperfect_information`: Some state is hidden from some players.
 */
export type Information =
  'one_shot' | 'perfect_information' | 'imperfect_information';

/**
 * How the served pyspiel's C++ MCTS (`pyspiel.MCTSBot`) bounds a search.
 *
 * The server finds out once, at startup, and picks MCTS's limits by it.
 *
 * - `simulations_or_clock`: A search stops at its simulation count or its wall-clock time, whichever comes first. MCTS runs up to 10000 simulations, 1000 by default, with a wall-clock backstop a little inside the move budget.
 * - `simulations_only`: A wall-clock time replaces the simulation count rather than bounding it. MCTS sets no wall-clock time, and its simulations are capped at 250, which fits the move budget on the slowest catalogued game.
 */
export type MctsLimits = 'simulations_or_clock' | 'simulations_only';

/**
 * Who acts at a position.
 *
 * - `seat`: A seat chooses the next action.
 * - `chance`: Chance chooses the next action.
 * - `terminal`: The game is over and nobody acts.
 */
export type MoverKind = 'seat' | 'chance' | 'terminal';

/**
 * Which values a game parameter admits.
 *
 * - `free`: Any value of its type.
 * - `range`: A number between `minimum` and `maximum`, inclusive.
 * - `choice`: One of `choices`.
 */
export type ParameterKind = 'free' | 'range' | 'choice';

/**
 * A game parameter, as a client needs it to offer the parameter.
 */
export interface ParameterSpec {
  /**
   * The parameter's name, a key of a session's `parameters`.
   */
  name: string;
  /**
   * The type of the parameter's value.
   */
  type: ParameterType;
  /**
   * Which values the parameter admits.
   */
  kind: ParameterKind;
  /**
   * The value the game uses when none is given.
   */
  default: boolean | number | string;
  /**
   * The least admitted value when `kind` is `range`, otherwise null.
   */
  minimum: number | null;
  /**
   * The greatest admitted value when `kind` is `range`, otherwise null.
   */
  maximum: number | null;
  /**
   * The admitted values when `kind` is `choice`, otherwise null.
   */
  choices: Array<boolean | number | string> | null;
  /**
   * What the parameter controls, or null when the game does not say.
   */
  description: string | null;
}

/**
 * The type of a game parameter's value.
 *
 * - `bool`: A boolean.
 * - `int`: An integer.
 * - `float`: A real number.
 * - `string`: A string.
 */
export type ParameterType = 'bool' | 'int' | 'float' | 'string';

/**
 * A position as the game's adapter presents it to one viewer.
 */
export interface Presentation {
  /**
   * The name of the client renderer that draws the view.
   */
  renderer: string;
  /**
   * The adapter's view of the position, holding only what the viewer may see.
   */
  view: unknown;
  /**
   * The adapter's hints for the legal actions, each naming an action of the frame's `legal_actions` at most once; null exactly when the frame's `legal_actions` is null.
   */
  actions: ActionHint[] | null;
}

/**
 * Restarts the live game, keeping the session's seats.
 */
export interface Reset {
  /**
   * Actions to replay from the initial state, chance outcomes included, as in a frame's `history`; null or empty for a fresh game.
   */
  history: number[] | null;
}

/**
 * Who plays a seat.
 */
export interface SeatConfig {
  /**
   * The `Controller.name` to play the seat.
   */
  controller: string;
  /**
   * The controller's settings, satisfying its `config_schema`.
   */
  config: Record<string, boolean | number | string>;
  /**
   * Whether the server plays the seat's moves as soon as it is to move. Must be false for a human seat.
   */
  autoplay: boolean;
}

/**
 * Whether a seat's controller is ready to play.
 *
 * - `ready`: The controller is built and plays the seat; a human seat is always ready.
 * - `preparing`: The server is building the controller, which can take a while (a trained model loads and warms up). The seat does not move until it is ready; then autoplay goes on.
 * - `unavailable`: The controller could not be built, or was not built within the build budget. The seat has a `failure`, so the session is paused until the seat is given a controller again.
 */
export type SeatStatus = 'ready' | 'preparing' | 'unavailable';

/**
 * One seat of a session.
 */
export interface SeatView {
  /**
   * The seat index, which is the OpenSpiel player id.
   */
  seat: number;
  /**
   * The `Controller.name` playing the seat.
   */
  controller: string;
  /**
   * The controller's settings, satisfying its `config_schema`. Every viewer receives them, so they hold no secret.
   */
  config: Record<string, boolean | number | string>;
  /**
   * Whether the server plays the seat's moves as soon as it is to move, rather than on an advance request. Never true for a human seat.
   */
  autoplay: boolean;
  /**
   * Why the controller failed, or null while it plays.
   */
  failure: ControllerFailure | null;
  /**
   * Whether the controller is ready, being built, or could not be built.
   */
  status: SeatStatus;
  /**
   * Why the seat is not ready, for every viewer: what the server is doing while it is `preparing`, or why it is `unavailable`; null when it is `ready`, and optional while it is `preparing`.
   */
  message: string | null;
}

/**
 * The fixed properties of a session.
 */
export interface SessionInfo {
  /**
   * The session's identifier.
   */
  id: string;
  /**
   * The game's short name.
   */
  game: string;
  /**
   * The game's parameters, defaults included.
   */
  parameters: Record<string, boolean | number | string>;
  /**
   * The number of seats.
   */
  num_seats: number;
  /**
   * How chance nodes are resolved.
   */
  chance_policy: ChancePolicy;
  /**
   * Whether a simultaneous game is served through the turn-based wrapper, which has its seats act one after another.
   */
  turn_based: boolean;
  /**
   * Whether some seat may not see the full state. When false, every viewer is omniscient.
   */
  hides_information: boolean;
  /**
   * The chance generator's seed; null for every viewer but the oracle, since it determines every chance outcome to come.
   */
  seed: number | null;
  /**
   * The controllers that can play a seat of this game.
   */
  controllers: Controller[];
}

/**
 * The state of a session's live game.
 *
 * - `playing`: The game is in progress.
 * - `paused`: A seat's controller failed; play resumes only after an undo, a reset or a change of that seat's controller. A seat whose controller could not be built (`SeatStatus.UNAVAILABLE`) resumes only after a change of its controller.
 * - `terminal`: The game is over.
 */
export type SessionStatus = 'playing' | 'paused' | 'terminal';

/**
 * Who acts at a position.
 */
export interface ToMove {
  /**
   * Whether a seat or chance acts, or the game is over.
   */
  kind: MoverKind;
  /**
   * The acting seat when `kind` is `seat`, otherwise null.
   */
  seat: number | null;
}

/**
 * The audience a frame is built for.
 */
export interface Viewer {
  /**
   * Whether the viewer holds a seat, watches, or sees everything.
   */
  kind: ViewerKind;
  /**
   * The viewer's seat when `kind` is `seat`, otherwise null.
   */
  seat: number | null;
}

/**
 * Who a frame is built for.
 *
 * - `seat`: A player, who sees what that player may observe.
 * - `spectator`: A watcher in no seat, who sees only public state.
 * - `oracle`: An omniscient observer, who sees the full state.
 */
export type ViewerKind = 'seat' | 'spectator' | 'oracle';
