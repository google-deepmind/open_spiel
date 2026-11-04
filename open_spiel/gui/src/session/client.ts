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

import type {
  Catalogue,
  CreateSession,
  Frame,
  Health,
  SeatConfig,
  Viewer,
} from '../contract/contract.ts';
import {type Bodies, decode} from './decode.ts';
import {TransportError} from './errors.ts';

/** The subset of `fetch` the client uses. */
export type Fetch = (url: string, init: RequestInit) => Promise<Response>;

/** The milliseconds a request may take when `ClientOptions` names none. */
export const DEFAULT_TIMEOUT = 120_000;

/** How a client reaches the server. */
export interface ClientOptions {
  /** The transport; the global `fetch` when omitted. */
  readonly fetch?: Fetch;
  /**
   * The server's origin, such as `http://localhost:8080`; the page's own when
   * omitted. In a browser, leave it out: the server refuses a POST or PUT
   * from another origin. Another value is for callers outside a browser.
   */
  readonly origin?: string;
  /**
   * The milliseconds a request may take, answer included; `DEFAULT_TIMEOUT`
   * when omitted. The server does not bound a controller's search, so a
   * caller whose controllers think longer raises it.
   */
  readonly timeout?: number;
}

/** The value of the `viewer` query parameter that names `viewer`. */
export function viewerParameter(viewer: Viewer): string {
  if (viewer.kind !== 'seat') return viewer.kind;
  if (viewer.seat === null) throw new Error('A seat viewer names its seat.');
  return String(viewer.seat);
}

/** Whether `left` and `right` name the same viewer. */
export function sameViewer(left: Viewer, right: Viewer): boolean {
  return viewerParameter(left) === viewerParameter(right);
}

/**
 * The server's API, one method per route. Every session route takes the
 * viewer explicitly, because the server builds each frame for one viewer.
 *
 * Each method resolves to the response body, or rejects with a
 * `ContractError` when the server refuses the request and a `TransportError`
 * when there is no answer in the contract, including none within the
 * timeout.
 */
export class SessionClient {
  private readonly fetch: Fetch;
  private readonly origin: string;
  private readonly timeout: number;

  constructor(options: ClientOptions = {}) {
    // Called through a closure: the global fetch refuses any other `this`.
    this.fetch = options.fetch ?? ((url, init) => globalThis.fetch(url, init));
    this.origin = options.origin ?? '';
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    if (!(Number.isInteger(this.timeout) && this.timeout > 0)) {
      throw new Error('A timeout is a positive number of milliseconds.');
    }
  }

  health(): Promise<Health> {
    return this.request('GET', '/api/health', 'health');
  }

  catalogue(): Promise<Catalogue> {
    return this.request('GET', '/api/games', 'catalogue');
  }

  create(viewer: Viewer, body: CreateSession): Promise<Frame> {
    return this.request(
      'POST',
      `/api/sessions?${query(viewer)}`,
      'frame',
      body,
    );
  }

  /** The live position. */
  frame(id: string, viewer: Viewer): Promise<Frame> {
    return this.sessionRequest('GET', id, '', viewer);
  }

  /** The position after `ply` actions; the session is unchanged. */
  replay(id: string, viewer: Viewer, ply: number): Promise<Frame> {
    return this.sessionRequest('GET', id, `/plies/${String(ply)}`, viewer);
  }

  /**
   * Plays `action` at the position after `ply` plies: null for the live
   * position; an earlier ply takes back every later ply first.
   */
  act(
    id: string,
    viewer: Viewer,
    action: number,
    ply: number | null = null,
  ): Promise<Frame> {
    return this.sessionRequest('POST', id, '/actions', viewer, {action, ply});
  }

  /** Chooses a chance outcome at the position after `ply`, as `act` does. */
  chooseChance(
    id: string,
    viewer: Viewer,
    outcome: number,
    ply: number | null = null,
  ): Promise<Frame> {
    return this.sessionRequest('POST', id, '/chance', viewer, {outcome, ply});
  }

  /**
   * Lets controllers play, at most `maxPlies` seat actions; null leaves the
   * bound to the server.
   */
  advance(id: string, viewer: Viewer, maxPlies: number | null): Promise<Frame> {
    return this.sessionRequest('POST', id, '/advance', viewer, {
      max_plies: maxPlies,
    });
  }

  undo(id: string, viewer: Viewer): Promise<Frame> {
    // The route reads no fields, but the server takes every POST as JSON.
    return this.sessionRequest('POST', id, '/undo', viewer, {});
  }

  /** Restarts the game and replays `history`, chance outcomes included. */
  reset(
    id: string,
    viewer: Viewer,
    history: readonly number[],
  ): Promise<Frame> {
    return this.sessionRequest('POST', id, '/reset', viewer, {history});
  }

  setSeat(
    id: string,
    viewer: Viewer,
    seat: number,
    config: SeatConfig,
  ): Promise<Frame> {
    return this.sessionRequest(
      'PUT',
      id,
      `/seats/${String(seat)}`,
      viewer,
      config,
    );
  }

  private sessionRequest(
    method: string,
    id: string,
    route: string,
    viewer: Viewer,
    body?: object,
  ): Promise<Frame> {
    const path = `/api/sessions/${encodeURIComponent(id)}${route}`;
    return this.request(method, `${path}?${query(viewer)}`, 'frame', body);
  }

  private async request<K extends keyof Bodies>(
    method: string,
    path: string,
    kind: K,
    body?: object,
  ): Promise<Bodies[K]> {
    // The signal also aborts reading the answer, which `decode` reports.
    const signal = AbortSignal.timeout(this.timeout);
    const init: RequestInit =
      body === undefined
        ? {method, signal}
        : {
            method,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
            signal,
          };
    let response: Response;
    try {
      response = await this.fetch(this.origin + path, init);
    } catch (e) {
      const message = signal.aborted
        ? `The server did not answer within ${String(this.timeout)} ms.`
        : 'The server could not be reached.';
      throw new TransportError(message, null, {cause: e});
    }
    return decode(response, kind);
  }
}

function query(viewer: Viewer): string {
  return new URLSearchParams({viewer: viewerParameter(viewer)}).toString();
}
