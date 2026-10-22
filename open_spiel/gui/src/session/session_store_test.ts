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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {CreateSession, Frame, Viewer} from '../contract/contract.ts';
import {SessionClient} from './client.ts';
import {ContractError, TransportError} from './errors.ts';
import {SessionStore} from './session_store.ts';
import {FakeServer, LENGTH, unanswered} from './testing/fake_server.ts';

const ORACLE: Viewer = {kind: 'oracle', seat: null};
const SEAT_0: Viewer = {kind: 'seat', seat: 0};
const SEAT_1: Viewer = {kind: 'seat', seat: 1};
const COUNT: CreateSession = {
  game: 'count',
  parameters: {},
  seats: [],
  chance_policy: 'auto',
  seed: null,
};

// Lets every queued command finish.
async function settle(store: SessionStore): Promise<void> {
  await vi.waitFor(
    () => {
      expect(store.getSnapshot().busy).toBe(false);
    },
    {interval: 1},
  );
}

function shown(store: SessionStore): Frame {
  const {frame} = store.getSnapshot();
  if (frame === null) throw new Error('no frame shown');
  return frame;
}

function actions(frame: Frame): Array<number | null> {
  return frame.history.map(entry => entry.action);
}

let server: FakeServer;
let store: SessionStore;

// Serves the game, hiding each seat's actions when `hidesInformation`, to a
// store whose first viewer is `viewer`.
function serve(hidesInformation: boolean, viewer = ORACLE): void {
  server = new FakeServer({hidesInformation});
  store = new SessionStore(new SessionClient({fetch: server.fetch}), viewer);
}

// The viewer of every frame the store shows from now on, or null while it
// shows none.
function watchViewers(): Array<Viewer | null> {
  const seen: Array<Viewer | null> = [];
  store.subscribe(() => {
    seen.push(store.getSnapshot().frame?.viewer ?? null);
  });
  return seen;
}

beforeEach(() => {
  serve(false);
});

afterEach(() => {
  vi.restoreAllMocks();
});

// Creates a session and plays `moves` in it.
async function play(...moves: number[]): Promise<void> {
  store.create(COUNT);
  for (const move of moves) store.act(move);
  await settle(store);
}

describe('SessionStore playing from an earlier position', () => {
  it('plays from an earlier position, taking back the later plies', async () => {
    await play(1, 0, 1);
    store.goTo(1);
    store.act(1);
    await settle(store);
    expect(store.getSnapshot().error).toBeNull();
    expect(server.history('session-1')).toEqual([1, 1]);
    expect(shown(store)).toMatchObject({ply: 2, num_plies: 2});
  });

  it('plays at the live position without naming a ply', async () => {
    await play(1);
    expect(server.received.at(-1)).toMatchObject({
      path: '/api/sessions/session-1/actions',
      body: {action: 1, ply: null},
    });
  });
});

describe('SessionStore commands', () => {
  it('starts idle, with no frame', () => {
    expect(store.getSnapshot()).toEqual({
      frame: null,
      busy: false,
      error: null,
    });
  });

  it('creates a session and plays in it', async () => {
    await play(1, 0);
    expect(shown(store).ply).toBe(2);
    expect(actions(shown(store))).toEqual([1, 0]);
    expect(server.history('session-1')).toEqual([1, 0]);
  });

  it('plays a game to its end', async () => {
    await play(...Array.from({length: LENGTH}, () => 0));
    expect(shown(store).status).toBe('terminal');
    expect(shown(store).returns).toEqual([1, -1]);
  });

  it('sends the viewer with every request', async () => {
    await play(1);
    store.step(-1);
    await settle(store);
    expect(server.received.map(request => request.viewer)).toEqual([
      'oracle',
      'oracle',
      'oracle',
    ]);
  });

  it('is busy until the queue drains, and notifies each change', async () => {
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    server.hold();
    store.create(COUNT);
    expect(store.getSnapshot().busy).toBe(true);
    server.release();
    await settle(store);
    expect(store.getSnapshot().frame).not.toBeNull();
    const calls = listener.mock.calls.length;
    unsubscribe();
    store.refresh();
    await settle(store);
    expect(listener).toHaveBeenCalledTimes(calls);
  });

  it('runs commands in order, each on the frame the last one left', async () => {
    await play(1, 1, 0, 1);
    server.hold();
    store.step(-1);
    store.step(-1);
    store.step(-1);
    server.release();
    await settle(store);
    expect(shown(store).ply).toBe(1);
    expect(server.received.slice(-3).map(request => request.path)).toEqual([
      '/api/sessions/session-1/plies/3',
      '/api/sessions/session-1/plies/2',
      '/api/sessions/session-1/plies/1',
    ]);
  });

  it('answers a change of the game with the live position', async () => {
    await play(1, 0, 1);
    store.goTo(1);
    store.undo();
    await settle(store);
    expect(shown(store).ply).toBe(2);
    expect(shown(store).num_plies).toBe(2);
  });

  it('resets to a history, which takes back plies by prefix', async () => {
    await play(1, 0, 1, 1);
    store.reset([1, 0]);
    await settle(store);
    expect(actions(shown(store))).toEqual([1, 0]);
    store.reset();
    await settle(store);
    expect(shown(store).num_plies).toBe(0);
  });
});

describe('SessionStore replay', () => {
  it('keeps the live history while stepping through a replay', async () => {
    await play(1, 0, 1);
    store.step(-2);
    await settle(store);
    expect(shown(store).ply).toBe(1);
    expect(actions(shown(store))).toEqual([1, 0, 1]);
    expect(shown(store).state_text).toBe('1');
  });

  it('steps back to the live position by its own route', async () => {
    await play(1, 0);
    store.goTo(0);
    store.step(5);
    await settle(store);
    expect(shown(store).ply).toBe(2);
    expect(server.received.at(-1)?.path).toBe('/api/sessions/session-1');
  });

  it('sends nothing for a step that stays put', async () => {
    await play(1);
    const sent = server.received.length;
    store.step(3);
    store.goTo(1);
    await settle(store);
    expect(server.received).toHaveLength(sent);
  });

  it('goes to the live position for a ply past it', async () => {
    await play(1, 0);
    store.goTo(0);
    store.goTo(9);
    await settle(store);
    expect(shown(store).ply).toBe(2);
    expect(server.received.at(-1)?.path).toBe('/api/sessions/session-1');
  });

  it('refreshes a replay at its ply', async () => {
    await play(1, 0);
    store.goTo(1);
    store.refresh();
    await settle(store);
    expect(server.received.at(-1)?.path).toBe(
      '/api/sessions/session-1/plies/1',
    );
  });
});

describe('SessionStore viewers', () => {
  it('fetches the shown position again for a new viewer', async () => {
    await play(1, 0);
    store.goTo(1);
    await settle(store);
    store.setViewer(SEAT_0);
    await settle(store);
    expect(shown(store).viewer).toEqual(SEAT_0);
    expect(shown(store).ply).toBe(1);
    expect(server.received.at(-1)).toMatchObject({
      path: '/api/sessions/session-1/plies/1',
      viewer: '0',
    });
  });

  it('sends nothing when the viewer is unchanged or no session is open', async () => {
    store.setViewer({kind: 'oracle', seat: null});
    store.setViewer(SEAT_0);
    await settle(store);
    expect(server.received).toEqual([]);
  });

  it('opens an existing session', async () => {
    await play(1);
    const other = new SessionStore(
      new SessionClient({fetch: server.fetch}),
      SEAT_0,
    );
    other.open('session-1');
    await settle(other);
    expect(shown(other).session.id).toBe('session-1');
    expect(shown(other).viewer).toEqual(SEAT_0);
  });

  it('shows a session being created to the viewer that follows', async () => {
    server.hold();
    store.create(COUNT);
    store.setViewer(SEAT_0);
    server.release();
    await settle(store);
    expect(shown(store).viewer).toEqual(SEAT_0);
    expect(server.received.map(({method, viewer}) => [method, viewer])).toEqual(
      [
        ['POST', 'oracle'],
        ['GET', '0'],
      ],
    );
  });
});

describe('SessionStore hidden information', () => {
  beforeEach(() => {
    serve(true);
  });

  it('withholds from a seat what the oracle sees', async () => {
    await play(1, 0);
    expect(shown(store)).toMatchObject({state_text: '1 0', omniscient: true});
    store.setViewer(SEAT_1);
    await settle(store);
    expect(shown(store)).toMatchObject({
      viewer: SEAT_1,
      omniscient: false,
      state_text: null,
      chance_outcomes: null,
    });
    expect(shown(store).session.seed).toBeNull();
    expect(actions(shown(store))).toEqual([null, 0]);
  });

  it('clears the frame as soon as the viewer changes', async () => {
    await play(1);
    server.hold();
    store.setViewer(SEAT_1);
    expect(store.getSnapshot().frame).toBeNull();
    server.release();
    await settle(store);
    expect(shown(store).viewer).toEqual(SEAT_1);
  });

  it('never shows an answer in flight for the old viewer', async () => {
    await play(1);
    server.hold();
    store.refresh();
    const sent = server.received.length + 1;
    await vi.waitFor(() => {
      expect(server.received).toHaveLength(sent);
    });
    store.setViewer(SEAT_1);
    const seen = watchViewers();
    server.release();
    await settle(store);
    expect(seen).not.toContainEqual(ORACLE);
    expect(seen.at(-1)).toEqual(SEAT_1);
  });

  it('shows the new viewer the position when a command before fails', async () => {
    await play(1);
    server.hold();
    store.act(7);
    store.setViewer(SEAT_1);
    server.release();
    await settle(store);
    expect(store.getSnapshot().error).toMatchObject({code: 'illegal_action'});
    expect(shown(store)).toMatchObject({viewer: SEAT_1, ply: 1});
  });

  it('shows no frame when the new viewer cannot be served', async () => {
    await play(1);
    server.refuseNext('viewer_unavailable');
    store.setViewer(SEAT_1);
    await settle(store);
    expect(store.getSnapshot()).toMatchObject({
      frame: null,
      error: {code: 'viewer_unavailable'},
    });
    store.refresh();
    await settle(store);
    expect(shown(store).viewer).toEqual(SEAT_1);
  });
});

describe('SessionStore hand-over', () => {
  beforeEach(() => {
    serve(true);
  });

  it('sends an action as the seat that took it, after a hand-over', async () => {
    serve(true, SEAT_0);
    store.create(COUNT);
    await settle(store);
    server.hold();
    store.act(1);
    store.setViewer(SEAT_1);
    const seen = watchViewers();
    server.release();
    await settle(store);
    expect(store.getSnapshot().error).toBeNull();
    expect(server.history('session-1')).toEqual([1]);
    expect(server.received.at(-2)).toMatchObject({
      path: '/api/sessions/session-1/actions',
      viewer: '0',
    });
    expect(seen).not.toContainEqual(SEAT_0);
    expect(shown(store)).toMatchObject({viewer: SEAT_1, ply: 1});
    expect(actions(shown(store))).toEqual([null]);
  });

  it("sends the oracle's reset as the oracle, after a hand-over", async () => {
    await play(1, 0, 1);
    server.hold();
    store.reset([1]);
    store.setViewer(SEAT_0);
    server.release();
    await settle(store);
    expect(store.getSnapshot().error).toBeNull();
    expect(server.history('session-1')).toEqual([1]);
    expect(server.received.at(-2)).toMatchObject({
      path: '/api/sessions/session-1/reset',
      viewer: 'oracle',
    });
    expect(shown(store)).toMatchObject({viewer: SEAT_0, state_text: null});
  });
});

describe('SessionStore failures', () => {
  it('shows a refusal, keeps the frame, and drops the queue behind it', async () => {
    await play(1);
    const before = shown(store);
    server.hold();
    store.act(7);
    store.act(0);
    server.release();
    await settle(store);
    const {frame, error} = store.getSnapshot();
    expect(error).toBeInstanceOf(ContractError);
    expect(error).toMatchObject({code: 'illegal_action'});
    expect(frame).toBe(before);
    expect(server.history('session-1')).toEqual([1]);
  });

  it('clears the error when the next command is issued', async () => {
    await play();
    server.refuseNext('not_your_turn');
    store.act(0);
    await settle(store);
    expect(store.getSnapshot().error).toMatchObject({code: 'not_your_turn'});
    server.hold();
    store.act(0);
    expect(store.getSnapshot().error).toBeNull();
    server.release();
    await settle(store);
    expect(shown(store).ply).toBe(1);
  });
});

describe('SessionStore defects', () => {
  it('refuses a ply or step that is not a whole number', () => {
    expect(() => {
      store.goTo(-1);
    }).toThrow(RangeError);
    expect(() => {
      store.step(0.5);
    }).toThrow(RangeError);
  });

  it('shows a failed transport', async () => {
    const down = new SessionClient({
      fetch: () => Promise.reject(new TypeError('fetch failed')),
    });
    store = new SessionStore(down, ORACLE);
    store.create(COUNT);
    await settle(store);
    expect(store.getSnapshot().error).toBeInstanceOf(TransportError);
  });

  it('shows an answer cut off before its end', async () => {
    const cut = new SessionClient({
      fetch: async (url, init) => {
        const answer = await server.fetch(url, init);
        answer.text = () => Promise.reject(new TypeError('terminated'));
        return answer;
      },
    });
    store = new SessionStore(cut, ORACLE);
    store.create(COUNT);
    await settle(store);
    expect(store.getSnapshot().error).toBeInstanceOf(TransportError);
  });

  it('ends a request the server never answers, and runs the next command', async () => {
    let answers = false;
    const stalling = new SessionClient({
      fetch: (url, init) =>
        answers ? server.fetch(url, init) : unanswered(url, init),
      timeout: 10,
    });
    store = new SessionStore(stalling, ORACLE);
    store.create(COUNT);
    await settle(store);
    expect(store.getSnapshot().error).toBeInstanceOf(TransportError);
    answers = true;
    store.create(COUNT);
    await settle(store);
    expect(store.getSnapshot()).toMatchObject({error: null});
    expect(shown(store).history).toEqual([]);
  });

  it('reports a command with no session as a defect, and drops the queue behind it', async () => {
    const reported: Array<() => void> = [];
    vi.spyOn(globalThis, 'queueMicrotask').mockImplementation(callback => {
      reported.push(callback);
    });
    store.act(0);
    store.create(COUNT);
    await settle(store);
    expect(store.getSnapshot()).toMatchObject({frame: null, error: null});
    expect(server.received).toEqual([]);
    expect(reported).toHaveLength(1);
    expect(reported[0]).toThrow('No session is open.');
  });
});
