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
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import type {Frame, Viewer} from '../contract/contract.ts';
import {SessionClient} from '../session/client.ts';
import {FakeServer, LENGTH} from '../session/testing/fake_server.ts';
import {type Session, useSession} from '../session/use_session.ts';
import {frame, session} from '../testing/frames.ts';
import {only} from '../testing/render.tsx';
import {GameplayBoard} from './board.tsx';

const ORACLE: Viewer = {kind: 'oracle', seat: null};
const SEAT_0: Viewer = {kind: 'seat', seat: 0};
const SPECTATOR: Viewer = {kind: 'spectator', seat: null};
const TIC_TAC_TOE = '...\n.x.\n..o\n';

let container: HTMLElement;
let root: Root;

function show(shown: Session): void {
  act(() => {
    root.render(<GameplayBoard session={shown} />);
  });
}

// The text of every status line and note, in document order.
function notes(): Array<string | null> {
  return Array.from(
    container.querySelectorAll('[role="status"]:not(.visually-hidden), p'),
    note => note.textContent,
  );
}

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
});

describe('GameplayBoard text', () => {
  it('says only that there is no position before a frame arrives', () => {
    act(() => {
      root.render(<GameplayBoard session={session(null)} />);
    });
    expect(notes()).toEqual(['No position to show.']);
  });

  it('shows the game text verbatim, in a named region that takes the focus', () => {
    show(session(frame({state_text: TIC_TAC_TOE})));
    const text = only(container, 'pre');
    expect(text.textContent).toBe(TIC_TAC_TOE);
    expect(text.getAttribute('role')).toBe('region');
    expect(text.tabIndex).toBe(0);
    expect(text.getAttribute('aria-label')).toBe('State');
    expect(container.querySelector('h3')).toBeNull();
  });

  it('draws the board alone: no copy button and no moves', () => {
    act(() => {
      root.render(<GameplayBoard session={session(frame())} />);
    });
    expect(container.querySelectorAll('button')).toHaveLength(0);
  });

  it('shows a seat its own observation', () => {
    const shown: Partial<Frame> = {
      viewer: SEAT_0,
      omniscient: false,
      state_text: null,
      observation_text: 'Card: K',
    };
    show(session(frame(shown)));
    expect(only(container, 'pre').getAttribute('aria-label')).toBe(
      "Seat 0's observation",
    );
    expect(only(container, 'pre').textContent).toBe('Card: K');
  });

  it('says why a spectator of a game that hides information sees no text, and cannot play', () => {
    const shown: Partial<Frame> = {
      viewer: SPECTATOR,
      omniscient: false,
      state_text: null,
      legal_actions: null,
    };
    show(session(frame(shown)));
    expect(container.querySelector('pre')).toBeNull();
    expect(notes()).toEqual([
      'The game hides information, so a spectator is shown no text.',
    ]);
  });
});

describe('GameplayBoard moves', () => {
  it('draws no legal list and no chance outcomes: they are in Actions', () => {
    show(
      session(
        frame({
          to_move: {kind: 'chance', seat: null},
          legal_actions: [],
          chance_outcomes: [{id: 4, label: 'Heads', probability: 1}],
        }),
      ),
    );
    expect(container.querySelector('ul')).toBeNull();
    expect(container.textContent).not.toContain('Heads');
    show(session(frame()));
    expect(container.querySelector('ul')).toBeNull();
    expect(container.textContent).not.toContain('action 1');
  });
});

describe('GameplayBoard on a session', () => {
  const settle = (): Promise<void> =>
    act(() => new Promise<void>(resolve => setTimeout(resolve, 0)));

  async function start(
    server: FakeServer,
    viewer: Viewer,
  ): Promise<() => Session> {
    const client = new SessionClient({fetch: server.fetch});
    let live: Session | undefined;
    function Board(): JSX.Element {
      const shown = useSession(client, viewer);
      useEffect(() => {
        live = shown;
      });
      return <GameplayBoard session={shown} />;
    }
    act(() => {
      root.render(<Board />);
    });
    act(() => {
      if (live === undefined) throw new Error('The board did not render.');
      live.commands.create({
        game: 'count',
        parameters: {},
        seats: [],
        chance_policy: 'auto',
        seed: null,
      });
    });
    await settle();
    return () => {
      if (live === undefined) throw new Error('The board did not render.');
      return live;
    };
  }

  it('shows each position played, to the end', async () => {
    const server = new FakeServer();
    const live = await start(server, ORACLE);
    for (let ply = 0; ply < LENGTH; ply++) {
      act(() => {
        live().commands.act(ply % 2 === 0 ? 1 : 0);
      });
      await settle();
    }
    expect(server.history('session-1')).toEqual([1, 0, 1, 0, 1, 0]);
    expect(only(container, 'pre').textContent).toBe('1 0 1 0 1 0');
  });

  it('shows a seat of a game that hides information only its own text', async () => {
    const server = new FakeServer({hidesInformation: true});
    const live = await start(server, SEAT_0);
    act(() => {
      live().commands.act(1);
    });
    await settle();
    expect(only(container, 'pre').getAttribute('aria-label')).toBe(
      "Seat 0's observation",
    );
    expect(only(container, 'pre').textContent).toBe('one');
    const seat1 = new SessionClient({fetch: server.fetch});
    await seat1.act('session-1', {kind: 'seat', seat: 1}, 0);
    act(() => {
      live().commands.refresh();
    });
    await settle();
    expect(server.history('session-1')).toEqual([1, 0]);
    expect(only(container, 'pre').getAttribute('aria-label')).toBe(
      "Seat 0's observation",
    );
    expect(only(container, 'pre').textContent).toBe('one ?');
  });
});
