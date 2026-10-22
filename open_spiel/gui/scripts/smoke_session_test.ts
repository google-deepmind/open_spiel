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

import {describe, expect, it} from 'vitest';

import {SessionClient} from '../src/session/client.ts';
import {FakeServer} from '../src/session/testing/fake_server.ts';
import {smokeSession} from './smoke_session.ts';

describe('smokeSession', () => {
  it('plays, replays, undoes and resets a game', async () => {
    const server = new FakeServer();
    const log: string[] = [];
    await smokeSession(
      new SessionClient({fetch: server.fetch}),
      'count',
      line => {
        log.push(line);
      },
    );
    expect(log).toEqual([
      'server 0.0.0, revision unknown',
      'created session-1',
      'played to terminal in 6 plies, returns [1,-1]',
      'went back to ply 2; the live history is unchanged',
      'undid to 5 plies',
      'reset to a three-ply history',
      'showed ply 3 to seat 0, omniscient true',
    ]);
    expect(server.history('session-1')).toEqual([0, 0, 0]);
  });

  it('checks what seat 0 is shown of a game that hides information', async () => {
    const server = new FakeServer({hidesInformation: true});
    const log: string[] = [];
    await smokeSession(
      new SessionClient({fetch: server.fetch}),
      'count',
      line => {
        log.push(line);
      },
    );
    expect(log.at(-1)).toBe('showed ply 3 to seat 0, omniscient false');
  });

  it('fails on a game the catalogue does not list', async () => {
    const client = new SessionClient({fetch: new FakeServer().fetch});
    await expect(
      smokeSession(client, 'chess', () => undefined),
    ).rejects.toThrow('smoke failed: the catalogue has no game chess');
  });
});
