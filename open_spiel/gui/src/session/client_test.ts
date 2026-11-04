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

import {afterEach, describe, expect, it, vi} from 'vitest';

import type {Viewer} from '../contract/contract.ts';
import {SessionClient, sameViewer, viewerParameter} from './client.ts';
import {ContractError, TransportError} from './errors.ts';
import {FakeServer, unanswered} from './testing/fake_server.ts';

const ORACLE: Viewer = {kind: 'oracle', seat: null};
const SEAT_1: Viewer = {kind: 'seat', seat: 1};
const SEAT_CONFIG = {controller: 'random', config: {}, autoplay: true};
const ANY_SIGNAL: unknown = expect.any(AbortSignal);

describe('viewerParameter', () => {
  it('names each kind of viewer', () => {
    expect(viewerParameter(ORACLE)).toBe('oracle');
    expect(viewerParameter({kind: 'spectator', seat: null})).toBe('spectator');
    expect(viewerParameter(SEAT_1)).toBe('1');
  });

  it('refuses a seat viewer without a seat', () => {
    expect(() => viewerParameter({kind: 'seat', seat: null})).toThrow(
      'names its seat',
    );
  });
});

describe('sameViewer', () => {
  it('compares viewers by what they name', () => {
    expect(sameViewer(SEAT_1, {kind: 'seat', seat: 1})).toBe(true);
    expect(sameViewer(SEAT_1, {kind: 'seat', seat: 0})).toBe(false);
    expect(sameViewer(ORACLE, {kind: 'spectator', seat: null})).toBe(false);
  });
});

// Each route: a call, and the method, path and body it must send.
interface Route {
  readonly name: string;
  readonly call: (client: SessionClient) => Promise<unknown>;
  readonly method: string;
  readonly path: string;
  readonly body?: object;
}

const ROUTES: Route[] = [
  {
    name: 'health',
    call: client => client.health(),
    method: 'GET',
    path: '/api/health',
  },
  {
    name: 'catalogue',
    call: client => client.catalogue(),
    method: 'GET',
    path: '/api/games',
  },
  {
    name: 'frame',
    call: client => client.frame('a b', SEAT_1),
    method: 'GET',
    path: '/api/sessions/a%20b?viewer=1',
  },
  {
    name: 'replay',
    call: client => client.replay('s', ORACLE, 3),
    method: 'GET',
    path: '/api/sessions/s/plies/3?viewer=oracle',
  },
  {
    name: 'act',
    call: client => client.act('s', SEAT_1, 4),
    method: 'POST',
    path: '/api/sessions/s/actions?viewer=1',
    body: {action: 4, ply: null},
  },
  {
    name: 'chooseChance',
    call: client => client.chooseChance('s', ORACLE, 2),
    method: 'POST',
    path: '/api/sessions/s/chance?viewer=oracle',
    body: {outcome: 2, ply: null},
  },
  {
    name: 'act at an earlier ply',
    call: client => client.act('s', ORACLE, 4, 2),
    method: 'POST',
    path: '/api/sessions/s/actions?viewer=oracle',
    body: {action: 4, ply: 2},
  },
  {
    name: 'advance',
    call: client => client.advance('s', ORACLE, null),
    method: 'POST',
    path: '/api/sessions/s/advance?viewer=oracle',
    body: {max_plies: null},
  },
  {
    name: 'undo',
    call: client => client.undo('s', SEAT_1),
    method: 'POST',
    path: '/api/sessions/s/undo?viewer=1',
    body: {},
  },
  {
    name: 'reset',
    call: client => client.reset('s', ORACLE, [1, 2]),
    method: 'POST',
    path: '/api/sessions/s/reset?viewer=oracle',
    body: {history: [1, 2]},
  },
  {
    name: 'setSeat',
    call: client => client.setSeat('s', ORACLE, 1, SEAT_CONFIG),
    method: 'PUT',
    path: '/api/sessions/s/seats/1?viewer=oracle',
    body: SEAT_CONFIG,
  },
];

describe('SessionClient routes', () => {
  it.each(ROUTES)(
    '$name sends its method, path and body',
    async ({call, method, path, body}) => {
      const requests: Array<[string, RequestInit]> = [];
      const client = new SessionClient({
        fetch: (url, init) => {
          requests.push([url, init]);
          // Any answer will do: only the request is checked.
          return Promise.resolve(new Response(null, {status: 204}));
        },
        origin: 'http://host:1',
      });
      await expect(call(client)).rejects.toThrow(TransportError);
      expect(requests).toEqual([
        [
          `http://host:1${path}`,
          body === undefined
            ? {method, signal: ANY_SIGNAL}
            : {
                method,
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
                signal: ANY_SIGNAL,
              },
        ],
      ]);
    },
  );
});

describe('SessionClient', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('creates a session with the request body and the viewer', async () => {
    const server = new FakeServer();
    const client = new SessionClient({fetch: server.fetch});
    const body = {
      game: 'count',
      parameters: {},
      seats: [],
      chance_policy: 'auto' as const,
      seed: 3,
    };
    const frame = await client.create(SEAT_1, body);
    expect(frame.viewer).toEqual(SEAT_1);
    expect(server.received).toEqual([
      {method: 'POST', path: '/api/sessions', viewer: '1', body},
    ]);
  });

  it('rejects a refusal with a ContractError', async () => {
    const server = new FakeServer();
    const client = new SessionClient({fetch: server.fetch});
    const error = await client.undo('missing', ORACLE).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ContractError);
    expect(error).toMatchObject({code: 'not_found', status: 404});
  });

  it('rejects an unreachable server with a TransportError', async () => {
    const cause = new TypeError('fetch failed');
    const client = new SessionClient({fetch: () => Promise.reject(cause)});
    const error = await client.health().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(TransportError);
    expect(error).toMatchObject({status: null, cause});
  });

  it('rejects a server that does not answer in time with a TransportError', async () => {
    const client = new SessionClient({fetch: unanswered, timeout: 10});
    const error = await client.health().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(TransportError);
    expect(error).toMatchObject({
      message: 'The server did not answer within 10 ms.',
      status: null,
    });
  });

  it('refuses a timeout that is not a positive number of milliseconds', () => {
    for (const timeout of [0, -1, 0.5, Number.NaN]) {
      expect(() => new SessionClient({timeout})).toThrow('positive number');
    }
  });

  it('calls the global fetch on the global object, from the page origin', async () => {
    const server = new FakeServer();
    const global = vi.fn(function (
      this: unknown,
      url: string,
      init: RequestInit,
    ) {
      expect(this).toBe(globalThis);
      return server.fetch(url, init);
    });
    vi.stubGlobal('fetch', global);
    await new SessionClient().health();
    expect(global).toHaveBeenCalledWith('/api/health', {
      method: 'GET',
      signal: ANY_SIGNAL,
    });
  });
});
