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

import type {Catalogue} from '../src/contract/contract.ts';
import {type Fetch, SessionClient} from '../src/session/client.ts';
import {FakeServer} from '../src/session/testing/fake_server.ts';
import {SUICIDE} from '../src/setup/testing/fixtures.ts';
import {withLength} from '../src/setup/testing/with_length.ts';
import {smokeSetup} from './smoke_setup.ts';

function client(server: FakeServer): SessionClient {
  return new SessionClient({fetch: withLength(server)});
}

// The transport of `server`, with "count" given a boolean "suicide" as well.
function withSuicide(server: FakeServer): Fetch {
  const fetch = withLength(server);
  return async (url, init) => {
    const response = await fetch(url, init);
    if (new URL(url, 'http://fake').pathname !== '/api/games') {
      return response;
    }
    const catalogue = (await response.json()) as Catalogue;
    const games = catalogue.games.map(game => ({
      ...game,
      parameters: [...game.parameters, SUICIDE],
    }));
    return new Response(JSON.stringify({...catalogue, games}), {
      headers: response.headers,
    });
  };
}

describe('smokeSetup', () => {
  it('reads back the defaults, starts a new game and replays a history', async () => {
    const server = new FakeServer();
    const log: string[] = [];
    await smokeSetup(client(server), 'count', 'length=4', line => {
      log.push(line);
    });
    expect(log).toEqual([
      'Defaults read back from their drafts (games: 1, parameters: 1)',
      'started session-2 with length=4 and 2 seats',
      'replayed the typed history [0, 0, 0]',
    ]);
    expect(server.received.at(-1)).toMatchObject({
      path: '/api/sessions/session-2/reset',
      body: {history: [0, 0, 0]},
    });
  });

  it('fails on a value the parameter does not admit', async () => {
    await expect(
      smokeSetup(
        client(new FakeServer()),
        'count',
        'length=4.5',
        () => undefined,
      ),
    ).rejects.toThrow('smoke failed: length=4.5: Enter a whole number.');
  });

  it('fails on a checkbox typed as neither true nor false', async () => {
    const suicide = new SessionClient({fetch: withSuicide(new FakeServer())});
    await expect(
      smokeSetup(suicide, 'count', 'suicide=yes', () => undefined),
    ).rejects.toThrow('smoke failed: suicide=yes: Enter true or false.');
  });

  it('fails on a parameter the game does not have', async () => {
    await expect(
      smokeSetup(client(new FakeServer()), 'count', 'size=4', () => undefined),
    ).rejects.toThrow('smoke failed: count offers no parameter size');
  });
});
