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

import {act, type JSX, useEffect} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';

import type {CreateSession, Viewer} from '../contract/contract.ts';
import {SessionClient} from './client.ts';
import {FakeServer} from './testing/fake_server.ts';
import {type Session, useSession} from './use_session.ts';

const ORACLE: Viewer = {kind: 'oracle', seat: null};
const SEAT_1: Viewer = {kind: 'seat', seat: 1};
const COUNT: CreateSession = {
  game: 'count',
  parameters: {},
  seats: [],
  chance_policy: 'auto',
  seed: null,
};

let server: FakeServer;
let client: SessionClient;
let root: Root;
let session: Session | undefined;
// The hook's value at every render, with the viewer it was rendered for.
let renders: Array<[Viewer, Session]>;

// Publishes the hook's value after each render.
function Probe({viewer}: {viewer: Viewer}): JSX.Element {
  const value = useSession(client, viewer);
  renders.push([viewer, value]);
  useEffect(() => {
    session = value;
  });
  return <></>;
}

function current(): Session {
  if (session === undefined) throw new Error('not rendered');
  return session;
}

// Runs `change`, then lets the requests it issued finish inside act.
async function perform(change: () => void): Promise<void> {
  await act(async () => {
    change();
    await new Promise(resolve => setTimeout(resolve, 0));
  });
}

beforeAll(() => {
  Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: true});
});

beforeEach(() => {
  server = new FakeServer();
  client = new SessionClient({fetch: server.fetch});
  root = createRoot(document.createElement('div'));
  renders = [];
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  session = undefined;
  vi.restoreAllMocks();
});

describe('useSession', () => {
  it('renders each frame, and whether it is live', async () => {
    await perform(() => {
      root.render(<Probe viewer={ORACLE} />);
    });
    expect(current()).toMatchObject({frame: null, live: false, busy: false});
    await perform(() => {
      current().commands.create(COUNT);
      current().commands.act(1);
    });
    expect(current()).toMatchObject({live: true, busy: false, error: null});
    expect(current().frame?.ply).toBe(1);
    await perform(() => {
      current().commands.step(-1);
    });
    expect(current().live).toBe(false);
    expect(current().frame?.ply).toBe(0);
  });

  it('renders a refusal as the error', async () => {
    await perform(() => {
      root.render(<Probe viewer={ORACLE} />);
    });
    await perform(() => {
      current().commands.open('missing');
    });
    expect(current()).toMatchObject({frame: null, error: {code: 'not_found'}});
  });

  it('fetches the frame again when the viewer changes', async () => {
    server = new FakeServer({hidesInformation: true});
    client = new SessionClient({fetch: server.fetch});
    await perform(() => {
      root.render(<Probe viewer={ORACLE} />);
    });
    await perform(() => {
      current().commands.create(COUNT);
    });
    await perform(() => {
      root.render(<Probe viewer={SEAT_1} />);
    });
    expect(current().frame).toMatchObject({viewer: SEAT_1, state_text: null});
    expect(server.received.at(-1)?.viewer).toBe('1');
    const shownToSeat = renders
      .filter(([viewer]) => viewer === SEAT_1)
      .map(([, value]) => value.frame?.viewer ?? null);
    expect(shownToSeat).not.toContainEqual(ORACLE);
    expect(shownToSeat[0]).toBeNull();
  });

  it('refuses a change of client', async () => {
    await perform(() => {
      root.render(<Probe viewer={ORACLE} />);
    });
    client = new SessionClient({fetch: server.fetch});
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    await expect(
      perform(() => {
        root.render(<Probe viewer={ORACLE} />);
      }),
    ).rejects.toThrow('keeps the client of its first render');
  });

  it('keeps one store across renders with an equal viewer', async () => {
    await perform(() => {
      root.render(<Probe viewer={ORACLE} />);
    });
    const {commands} = current();
    await perform(() => {
      root.render(<Probe viewer={{kind: 'oracle', seat: null}} />);
    });
    expect(current().commands).toBe(commands);
    expect(server.received).toEqual([]);
  });
});
