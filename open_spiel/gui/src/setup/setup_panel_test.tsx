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

import {act, type JSX, useState} from 'react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {CatalogueEntry, Viewer} from '../contract/contract.ts';
import {GameContext} from '../game/game_context.ts';
import {SessionClient} from '../session/client.ts';
import {FakeServer} from '../session/testing/fake_server.ts';
import {useSession} from '../session/use_session.ts';
import {stubResizeObserver} from '../testing/dom.ts';
import type {Rendered} from '../testing/render.tsx';
import {only, render} from '../testing/render.tsx';
import {SetupPanel} from './setup_panel.tsx';
import {enter} from './testing/input.ts';
import {withLength} from './testing/with_length.ts';

const ORACLE: Viewer = {kind: 'oracle', seat: null};
const SEAT_0: Viewer = {kind: 'seat', seat: 0};
const SPECTATOR: Viewer = {kind: 'spectator', seat: null};
const HUMAN = {controller: 'human', config: {}, autoplay: false};

let rendered: Rendered | undefined;

beforeEach(() => {
  stubResizeObserver();
});

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  vi.unstubAllGlobals();
});

function Panel({
  client,
  entry,
}: {
  client: SessionClient;
  entry: CatalogueEntry;
}): JSX.Element {
  const [viewer, setViewer] = useState<Viewer>(ORACLE);
  const session = useSession(client, viewer);
  return (
    <>
      <button
        type="button"
        onClick={() => {
          setViewer(SEAT_0);
        }}
      >
        View as seat 0
      </button>
      <button
        type="button"
        onClick={() => {
          setViewer(SPECTATOR);
        }}
      >
        View as a spectator
      </button>
      <button
        type="button"
        onClick={() => {
          session.commands.create({
            game: 'count',
            parameters: {},
            seats: [HUMAN, HUMAN],
            chance_policy: 'auto',
            seed: 1,
          });
        }}
      >
        Open
      </button>
      <GameContext
        value={{
          entry,
          session,
          viewer,
          setViewer,
          handOver: () => undefined,
        }}
      >
        <SetupPanel />
      </GameContext>
    </>
  );
}

// Clicks the button that reads `text`, and lets the requests it issues
// finish.
async function click(container: HTMLElement, text: string): Promise<void> {
  const button = Array.from(container.querySelectorAll('button')).find(
    candidate => candidate.textContent === text,
  );
  if (button === undefined) throw new Error(`No button reads ${text}.`);
  await act(async () => {
    button.click();
    await new Promise(resolve => setTimeout(resolve, 0));
  });
}

async function open(): Promise<{server: FakeServer; container: HTMLElement}> {
  const server = new FakeServer();
  const client = new SessionClient({fetch: withLength(server)});
  const [entry] = (await client.catalogue()).games;
  if (entry === undefined) throw new Error('The catalogue lists no game.');
  rendered = render(<Panel client={client} entry={entry} />);
  expect(rendered.container.textContent).toContain('No game is open.');
  await click(rendered.container, 'Open');
  return {server, container: rendered.container};
}

// The inner tabs' labels.
function tabs(container: HTMLElement): Array<string | null> {
  return Array.from(
    container.querySelectorAll('[role="tab"]'),
    tab => tab.textContent,
  );
}

// The open inner tab's panel.
function shown(container: HTMLElement): HTMLElement {
  return only(container, '[role="tabpanel"]:not([hidden])');
}

describe('SetupPanel', () => {
  it('offers Game options, Position and Players as inner tabs', async () => {
    const {container} = await open();
    expect(tabs(container)).toEqual(['Game options', 'Position', 'Players']);
    expect(only(container, '[role="tablist"]').getAttribute('aria-label')).toBe(
      'Setup sections',
    );
    expect(only(shown(container), 'h3').textContent).toBe('Game Options');
  });

  it('starts a new session once a changed option is committed', async () => {
    const {server, container} = await open();
    await act(async () => {
      const field = only(container, 'input[type="number"]');
      enter(field, '4');
      field.dispatchEvent(new FocusEvent('focusout', {bubbles: true}));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(server.received.at(-1)).toMatchObject({
      method: 'POST',
      path: '/api/sessions',
      body: {
        game: 'count',
        parameters: {length: 4},
        seats: [HUMAN, HUMAN],
        chance_policy: 'auto',
        seed: null,
      },
    });
    expect(
      (only(container, 'input[type="number"]') as HTMLInputElement).value,
    ).toBe('4');
  });

  it('keeps the open tab while the position loads for a new viewer', async () => {
    const {server, container} = await open();
    await click(container, 'Players');
    server.hold();
    await click(container, 'View as seat 0');
    expect(only(container, '.setup__notice').textContent).toBe(
      'Loading the position.',
    );
    await act(async () => {
      server.release();
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    expect(server.received.at(-1)).toMatchObject({
      path: '/api/sessions/session-1',
      viewer: '0',
    });
    expect(only(shown(container), 'h2').textContent).toBe('Players');
  });

  it('says when the position for a new viewer cannot be loaded', async () => {
    const {server, container} = await open();
    server.refuseNext('viewer_unavailable', 'No seat 0.');
    await click(container, 'View as seat 0');
    expect(only(container, '.setup__notice').textContent).toBe(
      'The position could not be loaded. No seat 0.',
    );
  });
});

describe('SetupPanel, its position', () => {
  it('sets the position, then resets the game', async () => {
    const {server, container} = await open();
    await click(container, 'Position');
    await click(container, 'Set state');
    enter(only(container, 'textarea'), '1 0 1');
    await click(container, 'Apply');
    expect(server.history('session-1')).toEqual([1, 0, 1]);

    await click(container, 'Reset');
    await click(only(document.body, '[role="dialog"]'), 'Reset');
    expect(server.history('session-1')).toEqual([]);
  });

  it('leaves Position out for a spectator, who sees the options read only', async () => {
    const {container} = await open();
    await click(container, 'View as a spectator');
    expect(tabs(container)).toEqual(['Game options', 'Players']);
    expect(shown(container).querySelector('input')).toBeNull();
  });
});
