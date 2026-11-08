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

// The fake server's game, "count", given one integer parameter, "length".
// The catalogue lists it, and every frame reports the value its session was
// created with, though the fake plays six plies whatever it is.

import type {CatalogueEntry, ParameterSpec} from '../../contract/contract.ts';
import type {Fetch} from '../../session/client.ts';
import type {FakeServer} from '../../session/testing/fake_server.ts';
import type {ParameterValues} from '../parameters.ts';

/** The "length" parameter: a whole number, 6 by default, with no bounds. */
export const LENGTH: ParameterSpec = {
  name: 'length',
  type: 'int',
  kind: 'free',
  default: 6,
  minimum: null,
  maximum: null,
  choices: null,
  description: null,
};

// The parts of a response body this rewrites.
interface Body {
  games?: CatalogueEntry[];
  session?: {id: string; parameters: ParameterValues};
}

/** The transport of `server`, with "count" given its "length" parameter. */
export function withLength(server: FakeServer): Fetch {
  const parameters = new Map<string, ParameterValues>();
  return async (url, init) => {
    const response = await server.fetch(url, init);
    const body = (await response.json()) as Body;
    const {games, session} = body;
    if (games !== undefined) {
      body.games = games.map(game => ({...game, parameters: [LENGTH]}));
    }
    if (session !== undefined) {
      const created = new URL(url, 'http://fake').pathname === '/api/sessions';
      if (created && typeof init.body === 'string') {
        const request = JSON.parse(init.body) as {parameters: ParameterValues};
        parameters.set(session.id, {
          length: LENGTH.default,
          ...request.parameters,
        });
      }
      const values = parameters.get(session.id);
      if (values === undefined) throw new Error(`No session ${session.id}.`);
      session.parameters = values;
    }
    return new Response(JSON.stringify(body), {
      status: response.status,
      headers: response.headers,
    });
  };
}
