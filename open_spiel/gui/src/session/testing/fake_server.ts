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

// An in-memory server for tests. It answers the session routes for one
// two-seat game, "count": seats alternate, each playing 0 or 1, and the game
// ends after LENGTH plies with seat 0 winning. The game hides each seat's
// actions from the other when the server is built with `hidesInformation`,
// and the frames then withhold what CONTRACT.md withholds from a viewer that
// is not omniscient.

import type {
  Catalogue,
  Error as ErrorBody,
  ErrorCode,
  Frame,
  Health,
  HistoryEntry,
  SeatView,
  Viewer,
} from '../../contract/contract.ts';
import type {Fetch} from '../client.ts';

/** The plies a game of "count" lasts. */
export const LENGTH = 6;
const SEATS = 2;
const LABELS = ['zero', 'one'];
// The seed of a session created without one.
const DEFAULT_SEED = 7;
const HTTP_STATUS: Record<ErrorCode, number> = {
  bad_request: 400,
  not_found: 404,
  method_not_allowed: 405,
  game_unavailable: 409,
  viewer_unavailable: 409,
  not_your_turn: 409,
  illegal_action: 409,
  controller_unavailable: 409,
  controller_failed: 409,
  nothing_to_undo: 409,
};

/** A request the fake received. */
export interface Received {
  readonly method: string;
  readonly path: string;
  readonly viewer: string | null;
  readonly body: unknown;
}

/** A transport that never answers, and rejects once its request is aborted. */
export function unanswered(
  _url: string,
  {signal}: RequestInit,
): Promise<Response> {
  return new Promise((_resolve, reject) => {
    if (!signal) throw new Error('The request has no signal to abort it.');
    signal.addEventListener('abort', () => {
      reject(new Error('The request was aborted.', {cause: signal.reason}));
    });
  });
}

/** How the fake plays its game. */
export interface FakeServerOptions {
  /** Whether each seat's actions are hidden from the other. */
  readonly hidesInformation?: boolean;
}

interface Session {
  readonly history: number[];
  readonly seed: number;
}

/** The fake server. `fetch` is its transport. */
export class FakeServer {
  /** Every request, in order. */
  readonly received: Received[] = [];
  private readonly hidesInformation: boolean;
  private readonly sessions = new Map<string, Session>();
  private held: Array<() => void> | null = null;
  private refusal: [ErrorCode, string] | null = null;

  constructor(options: FakeServerOptions = {}) {
    this.hidesInformation = options.hidesInformation ?? false;
  }

  readonly fetch: Fetch = async (url, init) => {
    const parsed = new URL(url, 'http://fake');
    const body: unknown =
      typeof init.body === 'string' ? JSON.parse(init.body) : null;
    const method = init.method ?? 'GET';
    const viewer = parsed.searchParams.get('viewer');
    this.received.push({method, path: parsed.pathname, viewer, body});
    if (this.held !== null) {
      const held = this.held;
      await new Promise<void>(resolve => held.push(resolve));
    }
    const refusal = this.refusal;
    this.refusal = null;
    if (refusal !== null) return refuse(...refusal);
    return this.answer(method, parsed.pathname, body, viewer);
  };

  /** Holds every response until `release`. */
  hold(): void {
    this.held = [];
  }

  /** Sends the held responses, in the order their requests arrived. */
  release(): void {
    const held = this.held ?? [];
    this.held = null;
    for (const resolve of held) resolve();
  }

  /** Refuses the next request with `code`. */
  refuseNext(code: ErrorCode, message = 'Refused.'): void {
    this.refusal = [code, message];
  }

  /** The live history of session `id`. */
  history(id: string): readonly number[] {
    return this.sessions.get(id)?.history ?? [];
  }

  private answer(
    method: string,
    path: string,
    body: unknown,
    viewerName: string | null,
  ): Response {
    if (path === '/api/health') return json(HEALTH);
    if (path === '/api/games') return json(this.catalogue());
    const viewer = parseViewer(viewerName);
    const field = (name: string): unknown =>
      (body as Record<string, unknown>)[name];
    if (method === 'POST' && path === '/api/sessions') {
      const id = `session-${String(this.sessions.size + 1)}`;
      const seed = field('seed');
      const session: Session = {
        history: [],
        seed: typeof seed === 'number' ? seed : DEFAULT_SEED,
      };
      this.sessions.set(id, session);
      return this.frame(id, session, 0, viewer);
    }
    const match = /^\/api\/sessions\/([^/]+)(\/.*)?$/.exec(path);
    const id = match?.[1] ?? '';
    const session = this.sessions.get(id);
    if (session === undefined) return refuse('not_found', 'No such session.');
    const {history} = session;
    const route = `${method} ${match?.[2] ?? ''}`;
    const replay = /^GET \/plies\/(\d+)$/.exec(route);
    if (replay !== null) {
      const ply = Number(replay[1]);
      if (ply > history.length) return refuse('not_found', 'No such ply.');
      return this.frame(id, session, ply, viewer);
    }
    switch (route) {
      case 'GET ':
        break;
      case 'POST /actions': {
        const action = field('action');
        const named = field('ply');
        const ply = typeof named === 'number' ? named : history.length;
        if (ply > history.length) return refuse('not_found', 'No such ply.');
        const omniscient = viewer.kind === 'oracle' || !this.hidesInformation;
        if (ply < history.length && !omniscient) {
          return refuse('not_your_turn', 'Only the oracle plays from there.');
        }
        if (ply >= LENGTH || (action !== 0 && action !== 1)) {
          return refuse('illegal_action', 'Not legal here.');
        }
        const mover = ply % SEATS;
        if (!(viewer.kind === 'oracle' || viewer.seat === mover)) {
          return refuse('not_your_turn', `Seat ${String(mover)} is to move.`);
        }
        history.splice(ply, history.length - ply, action);
        break;
      }
      case 'POST /undo':
        if (history.pop() === undefined) {
          return refuse('nothing_to_undo', 'Nothing to undo.');
        }
        break;
      case 'POST /reset':
        if (viewer.kind === 'spectator') {
          return refuse('not_your_turn', 'A spectator never resets the game.');
        }
        history.splice(0, history.length, ...(field('history') as number[]));
        break;
      default:
        return refuse('method_not_allowed', `The fake has no ${route}.`);
    }
    return this.frame(id, session, history.length, viewer);
  }

  private frame(
    id: string,
    {history, seed}: Session,
    ply: number,
    viewer: Viewer,
  ): Response {
    const omniscient = viewer.kind === 'oracle' || !this.hidesInformation;
    // Whether `viewer` sees the action seat `seat` chose.
    const sees = (seat: number): boolean => omniscient || viewer.seat === seat;
    const over = ply === LENGTH;
    const mover = ply % SEATS;
    // An omniscient viewer may play from an earlier position.
    const actions =
      (ply === history.length || omniscient) && !over
        ? LABELS.map((label, action) => ({id: action, label}))
        : [];
    const entries: HistoryEntry[] = history.map((action, index) => {
      const seat = index % SEATS;
      return sees(seat)
        ? {
            seat,
            action,
            label: LABELS[action] ?? null,
            rewards: [0, 0],
            public_label: null,
          }
        : {seat, action: null, label: null, rewards: null, public_label: null};
    });
    const seats: SeatView[] = Array.from({length: SEATS}, (_, seat) => ({
      seat,
      controller: 'human',
      config: {},
      autoplay: false,
      failure: null,
      status: 'ready',
      message: null,
    }));
    const body: Frame = {
      contract_version: 7,
      session: {
        id,
        game: 'count',
        parameters: {},
        num_seats: SEATS,
        chance_policy: 'auto',
        turn_based: false,
        hides_information: this.hidesInformation,
        seed: viewer.kind === 'oracle' ? seed : null,
        controllers: HEALTH.capabilities.controllers,
      },
      viewer,
      omniscient,
      status: history.length === LENGTH ? 'terminal' : 'playing',
      seats,
      ply,
      num_plies: history.length,
      to_move: over
        ? {kind: 'terminal', seat: null}
        : {kind: 'seat', seat: mover},
      legal_actions:
        omniscient || (!over && viewer.seat === mover) ? actions : null,
      chance_outcomes: omniscient ? [] : null,
      history: entries,
      rewards: ply === 0 ? null : (entries[ply - 1]?.rewards ?? null),
      returns: over ? [1, -1] : null,
      state_text: omniscient ? history.slice(0, ply).join(' ') : null,
      observation_text:
        viewer.kind === 'seat'
          ? entries
              .slice(0, ply)
              .map(entry => entry.label ?? '?')
              .join(' ')
          : null,
      information_state_text: null,
      struct: null,
      observation_struct: null,
      presentation: null,
    };
    return json(body);
  }

  private catalogue(): Catalogue {
    const game = CATALOGUE.games[0];
    if (game === undefined) throw new Error('The catalogue lists no game.');
    const information = this.hidesInformation
      ? 'imperfect_information'
      : 'perfect_information';
    return {...CATALOGUE, games: [{...game, information}]};
  }
}

function parseViewer(name: string | null): Viewer {
  if (name === 'oracle' || name === 'spectator') {
    return {kind: name, seat: null};
  }
  return {kind: 'seat', seat: Number(name)};
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
}

function refuse(code: ErrorCode, message: string): Response {
  const body: ErrorBody = {
    contract_version: 7,
    code,
    message,
    seat: null,
    ply: null,
  };
  return json(body, HTTP_STATUS[code]);
}

const HEALTH: Health = {
  contract_version: 7,
  version: '0.0.0',
  revision: null,
  capabilities: {
    controllers: [
      {
        name: 'human',
        label: 'Human',
        config_schema: {},
        unavailable_reason: null,
      },
    ],
    chance_policies: ['auto', 'expose'],
    cpp_mcts_limits: 'simulations_only',
  },
};

const CATALOGUE: Catalogue = {
  contract_version: 7,
  games: [
    {
      name: 'count',
      long_name: 'Count',
      dynamics: 'sequential',
      chance_mode: 'deterministic',
      information: 'perfect_information',
      min_seats: SEATS,
      max_seats: SEATS,
      num_players: SEATS,
      parameters: [],
      loadable: true,
      unavailable_reason: null,
      controllers: ['human'],
      renderer: null,
    },
  ],
};
