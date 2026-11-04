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

// Plays one game through a running server with the session store, as the
// client does: create a session, play the first legal action until the game
// ends, go back to ply 2, undo, reset to a three-ply history, and show the
// position to seat 0, checking that seat 0 is shown nothing CONTRACT.md
// withholds from it. Each check failing throws.
//
//   npm run smoke:session -- http://127.0.0.1:8080 tic_tac_toe

/// <reference types="node" />

import {realpathSync} from 'node:fs';
import process from 'node:process';

import type {Frame, Viewer} from '../src/contract/contract.ts';
import {defaultSession} from '../src/game/game_defaults.ts';
import {SessionClient} from '../src/session/client.ts';
import {SessionStore} from '../src/session/session_store.ts';

// More plies than any game this is pointed at should need.
const MAX_PLIES = 10_000;
const SEAT_0: Viewer = {kind: 'seat', seat: 0};

/** Runs the smoke test of `game`, reporting each step through `log`. */
export async function smokeSession(
  client: SessionClient,
  game: string,
  log: (line: string) => void,
): Promise<void> {
  const health = await client.health();
  log(`server ${health.version}, revision ${health.revision ?? 'unknown'}`);
  const store = new SessionStore(client, {kind: 'oracle', seat: null});
  let frame = await create(client, store, game);
  log(`created ${frame.session.id}`);

  while (frame.status === 'playing') {
    check(
      frame.num_plies < MAX_PLIES,
      `${game} did not end in ${String(MAX_PLIES)} plies`,
    );
    const action = frame.legal_actions?.[0];
    check(
      action !== undefined,
      `ply ${String(frame.ply)} offers no legal action`,
    );
    frame = await run(store, () => {
      store.act(action.id);
    });
  }
  check(frame.status === 'terminal', `the game stopped ${frame.status}`);
  const played = actions(frame);
  log(
    `played to terminal in ${String(played.length)} plies, returns ${JSON.stringify(frame.returns)}`,
  );
  check(played.length > 3, 'the game ended before ply 4');

  frame = await run(store, () => {
    store.goTo(2);
  });
  check(frame.ply === 2, `going to ply 2 reached ply ${String(frame.ply)}`);
  check(
    sameList(actions(frame), played),
    'the replay changed the live history',
  );
  log('went back to ply 2; the live history is unchanged');

  frame = await run(store, store.undo);
  const kept = actions(frame);
  check(frame.ply === kept.length, 'undo did not return to the live position');
  check(kept.length < played.length, 'undo kept every ply');
  check(
    sameList(kept, played.slice(0, kept.length)),
    'undo rewrote the history',
  );
  log(`undid to ${String(kept.length)} plies`);

  frame = await run(store, () => {
    store.reset(played.slice(0, 3));
  });
  check(
    sameList(actions(frame), played.slice(0, 3)),
    'reset did not replay its history',
  );
  check(
    frame.ply === 3 && frame.status === 'playing',
    'reset is not live at ply 3',
  );
  log('reset to a three-ply history');

  frame = await run(store, () => {
    store.setViewer(SEAT_0);
    check(
      store.getSnapshot().frame === null,
      "the oracle's frame outlived the change of viewer",
    );
  });
  check(
    frame.viewer.seat === 0 && frame.ply === 3,
    'seat 0 is not shown ply 3',
  );
  if (frame.session.hides_information) checkWithheld(frame);
  log(`showed ply 3 to seat 0, omniscient ${String(frame.omniscient)}`);
}

// Starts a session of `game` with every seat human.
async function create(
  client: SessionClient,
  store: SessionStore,
  game: string,
): Promise<Frame> {
  const entry = (await client.catalogue()).games.find(
    listed => listed.name === game,
  );
  check(entry !== undefined, `the catalogue has no game ${game}`);
  check(
    entry.loadable,
    `${game} is not loadable: ${String(entry.unavailable_reason)}`,
  );
  return run(store, () => {
    store.create({...defaultSession(entry), seed: 1});
  });
}

// Issues a command, waits for the store to go idle, and returns its frame.
async function run(store: SessionStore, command: () => void): Promise<Frame> {
  const idle = new Promise<void>(resolve => {
    const unsubscribe = store.subscribe(() => {
      if (store.getSnapshot().busy) return;
      unsubscribe();
      resolve();
    });
  });
  command();
  await idle;
  const {frame, error} = store.getSnapshot();
  if (error !== null) throw error;
  check(frame !== null, 'no frame is shown');
  return frame;
}

// What CONTRACT.md withholds from seat 0 in a game that hides information.
function checkWithheld(frame: Frame): void {
  check(!frame.omniscient, 'seat 0 is omniscient');
  check(frame.session.seed === null, 'seat 0 is shown the seed');
  check(
    frame.state_text === null && frame.struct === null,
    'seat 0 is shown the full state',
  );
  check(frame.chance_outcomes === null, 'seat 0 is shown chance outcomes');
  const shown = frame.history.findIndex(
    entry => entry.seat !== 0 && entry.action !== null,
  );
  check(shown === -1, `seat 0 is shown the action at ply ${String(shown)}`);
}

// The oracle sees every action in full.
function actions(frame: Frame): number[] {
  return frame.history.map(({action}, ply) => {
    check(action !== null, `the oracle's history withholds ply ${String(ply)}`);
    return action;
  });
}

function sameList(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function check(condition: boolean, failure: string): asserts condition {
  if (!condition) throw new Error(`smoke failed: ${failure}`);
}

const entryPoint = process.argv[1];
if (
  entryPoint !== undefined &&
  realpathSync(entryPoint) === import.meta.filename
) {
  const [origin, game] = process.argv.slice(2);
  if (origin === undefined || game === undefined) {
    throw new Error('usage: smoke_session.ts <server-origin> <game>');
  }
  await smokeSession(new SessionClient({origin}), game, line => {
    console.log(line);
  });
  console.log('smoke passed');
}
