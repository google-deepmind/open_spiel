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

// Sets up games through a running server as the Setup panel does, with its
// parsing and its requests. For every loadable game, the drafts of the
// defaults of the parameters Game options offers read back as them. For
// `game`, a value typed for one of those parameters starts a new game with
// that value and a seat per player, as setting Players does, and a typed
// history replays. Each check failing throws.
//
//   npm run smoke:setup -- http://127.0.0.1:8080 kuhn_poker players=3

/// <reference types="node" />

import process from 'node:process';

import type {
  CatalogueEntry,
  ParameterSpec,
  Viewer,
} from '../src/contract/contract.ts';
import {defaultSession} from '../src/game/game_defaults.ts';
import {SessionClient} from '../src/session/client.ts';
import {
  type Draft,
  type ParameterValue,
  isCheckbox,
  newGame,
  parseDraft,
  parseDrafts,
  sessionDrafts,
  sessionParameters,
  setupParameters,
} from '../src/setup/parameters.ts';
import {
  formatHistory,
  historyActions,
  parseHistory,
} from '../src/setup/position_text.ts';
import {isEntryPoint} from './entry_point.ts';

const ORACLE: Viewer = {kind: 'oracle', seat: null};
// Plies played before the history is typed back.
const PLIES = 3;

/**
 * Runs the smoke test of `game`, typing `assignment` (`name=value`) for one
 * of its parameters and reporting each step through `log`.
 */
export async function smokeSetup(
  client: SessionClient,
  game: string,
  assignment: string,
  log: (line: string) => void,
): Promise<void> {
  const {games} = await client.catalogue();
  const loadable = games.filter(entry => entry.loadable);
  const count = loadable.reduce(
    (total, entry) => total + checkDefaults(entry),
    0,
  );
  log(
    `Defaults read back from their drafts (games: ${String(loadable.length)}, parameters: ${String(count)})`,
  );

  const entry = loadable.find(listed => listed.name === game);
  check(entry !== undefined, `the catalogue has no loadable game ${game}`);
  const first = await client.create(ORACLE, {
    ...defaultSession(entry),
    seed: 1,
  });
  const [name, value] = typed(entry, assignment);
  const body = newGame(entry, first, {
    ...sessionParameters(entry, first),
    [name]: value,
  });
  let frame = await client.create(ORACLE, body);
  check(
    frame.session.id !== first.session.id &&
      frame.session.parameters[name] === value &&
      frame.session.num_seats === body.seats.length,
    `the new game does not have ${name}=${String(value)} and ${String(body.seats.length)} seats`,
  );
  log(
    `started ${frame.session.id} with ${name}=${String(value)} and ${String(frame.session.num_seats)} seats`,
  );

  while (frame.num_plies < PLIES && frame.status === 'playing') {
    const action = frame.legal_actions?.[0];
    check(action !== undefined, `ply ${String(frame.ply)} offers no action`);
    frame = await client.act(frame.session.id, ORACLE, action.id);
  }
  const played = historyActions(frame);
  const parsed = parseHistory(formatHistory(played));
  check(parsed.ok, `the typed history ${formatHistory(played)} does not read`);
  frame = await client.reset(frame.session.id, ORACLE, parsed.history);
  // Automatic chance may move after the replay, so the replay is a prefix.
  const replayed = historyActions(frame).slice(0, played.length);
  check(
    replayed.length === played.length &&
      replayed.every((action, ply) => action === played[ply]),
    `the reset did not replay ${formatHistory(played)}`,
  );
  log(`replayed the typed history ${formatHistory(played)}`);
}

// Checks that the drafts of the defaults of the parameters Game options
// offers for `entry` read back as them, and returns how many it checked.
function checkDefaults(entry: CatalogueEntry): number {
  const specs = setupParameters(entry);
  const defaults = Object.fromEntries(
    specs.map(spec => [spec.name, spec.default]),
  );
  const {values, problems} = parseDrafts(specs, sessionDrafts(specs, defaults));
  check(
    values !== null && specs.every(spec => values[spec.name] === spec.default),
    `${entry.name}: the drafts of the defaults read back as ${JSON.stringify(values ?? Object.fromEntries(problems))}`,
  );
  return specs.length;
}

// The parameter of Game options `assignment` names and the value its typed
// text reads as.
function typed(
  entry: CatalogueEntry,
  assignment: string,
): [string, ParameterValue] {
  const [name = '', text = ''] = assignment.split('=', 2);
  const spec = setupParameters(entry).find(listed => listed.name === name);
  check(spec !== undefined, `${entry.name} offers no parameter ${name}`);
  const parsed = parseDraft(spec, typedDraft(spec, assignment, text));
  check(parsed.ok, `${assignment}: ${parsed.ok ? '' : parsed.problem}`);
  return [name, parsed.value];
}

// The draft that typing `text` leaves in `spec`'s control; a checkbox's text
// is true or false.
function typedDraft(
  spec: ParameterSpec,
  assignment: string,
  text: string,
): Draft {
  if (!isCheckbox(spec)) return text;
  check(
    text === 'true' || text === 'false',
    `${assignment}: Enter true or false.`,
  );
  return text === 'true';
}

function check(condition: boolean, failure: string): asserts condition {
  if (!condition) throw new Error(`smoke failed: ${failure}`);
}

if (isEntryPoint(import.meta.filename)) {
  const [origin, game, assignment] = process.argv.slice(2);
  if (origin === undefined || game === undefined || assignment === undefined) {
    throw new Error(
      'usage: smoke_setup.ts <server-origin> <game> <name=value>',
    );
  }
  await smokeSetup(new SessionClient({origin}), game, assignment, line => {
    console.log(line);
  });
  console.log('smoke passed');
}
