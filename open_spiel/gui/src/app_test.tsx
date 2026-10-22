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

import {act} from 'react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {App} from './app.tsx';
import {SessionClient} from './session/client.ts';
import {FakeServer} from './session/testing/fake_server.ts';
import {fakeColorScheme} from './testing/color_scheme.ts';
import {enter, labelled} from './testing/input.ts';
import {button, render, type Rendered} from './testing/render.tsx';
import {ThemeProvider} from './theme/theme_provider.tsx';

/** A ResizeObserver that never reports: the dock keeps its first layout. */
class SilentResizeObserver {
  observe(): void {
    // Nothing is laid out under jsdom, so nothing ever resizes.
  }
  unobserve(): void {
    // As observe.
  }
  disconnect(): void {
    // As observe.
  }
}

let rendered: Rendered | undefined;
let server: FakeServer;

/** Renders the app at `path` against the fake server. */
function open(path: string): HTMLElement {
  const client = new SessionClient({fetch: server.fetch});
  const url = new URL(path, window.location.origin);
  rendered = render(
    <ThemeProvider>
      <App url={url} client={client} />
    </ThemeProvider>,
  );
  return rendered.container;
}

/** Waits, letting the fake server answer, until `check` passes. */
async function until(check: () => void): Promise<void> {
  await vi.waitFor(async () => {
    await act(async () => {
      await Promise.resolve();
    });
    check();
  });
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

/** The legal action labelled `label`, in Actions › Legal, which it opens. */
function legal(container: HTMLElement, label: string): HTMLElement {
  const tab = [...container.querySelectorAll<HTMLElement>('[role="tab"]')].find(
    candidate => candidate.textContent.startsWith('Legal'),
  );
  if (tab === undefined) throw new Error('No Legal tab.');
  if (tab.getAttribute('aria-selected') !== 'true') click(tab);
  const row = [
    ...container.querySelectorAll<HTMLElement>('button.legal__row'),
  ].find(candidate => candidate.textContent.endsWith(label));
  if (row === undefined) throw new Error(`No legal action ${label}.`);
  return row;
}

/** The Perspective list, in the View panel, which it brings forward. */
function perspective(container: HTMLElement): HTMLSelectElement {
  const tab = [...document.querySelectorAll('.dv-tab')].find(
    candidate => candidate.textContent === 'View',
  );
  if (tab === undefined) throw new Error('No View tab.');
  act(() => {
    tab.dispatchEvent(new MouseEvent('pointerdown', {bubbles: true}));
  });
  return labelled(container, 'Perspective') as HTMLSelectElement;
}

beforeEach(() => {
  window.localStorage.clear();
  fakeColorScheme(false);
  vi.stubGlobal('ResizeObserver', SilentResizeObserver);
  server = new FakeServer();
});

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  vi.unstubAllGlobals();
  window.history.replaceState(null, '', '/');
  window.localStorage.clear();
});

describe('App', () => {
  it('lists the catalogue at the root, each game linking to its page', async () => {
    const container = open('/');
    await until(() => {
      expect(container.querySelector('a.catalogue__title')).not.toBeNull();
    });
    const link = container.querySelector('a.catalogue__title');
    expect(link?.textContent).toBe('Count');
    expect(link?.getAttribute('href')).toBe('/game/count');
    expect(container.textContent).toContain('1 of 1 games');
  });

  it('says why the catalogue could not be loaded', async () => {
    server.refuseNext('bad_request', 'No games today.');
    const container = open('/');
    await until(() => {
      expect(container.textContent).toContain(
        'The games could not be loaded. No games today.',
      );
    });
  });

  it.each([
    ['/game/nope', 'There is no game "nope".'],
    ['/nowhere', 'Nothing is at /nowhere.'],
  ])('says what is missing at %s', async (path, sentence) => {
    const container = open(path);
    await until(() => {
      expect(container.textContent).toContain(sentence);
    });
  });
});

describe('App, its header', () => {
  it('holds the title, then the dock’s controls, then the settings', async () => {
    const container = open('/game/count');
    await until(() => {
      expect(
        container.querySelector('.app__page-actions .layouts'),
      ).not.toBeNull();
    });
    const header = container.querySelector('header');
    expect(header?.querySelector('h1')?.textContent).toBe('Count');
    const controls = [
      ...(header?.querySelectorAll('.app__actions button[aria-haspopup]') ??
        []),
    ].map(control => control.getAttribute('aria-label') ?? control.textContent);
    expect(controls).toEqual(['Layouts ▾', 'Panel options', 'Settings']);
    expect(container.querySelector('.dock-toolbar, .dock__actions')).toBeNull();
  });

  it('names OpenSpiel, then a bullet hidden from readers, then the title', () => {
    const brand = open('/game/count').querySelector('header .app__brand');
    const parts = [...(brand?.children ?? [])].map(part => [
      part.tagName,
      part.textContent,
      part.getAttribute('aria-hidden'),
    ]);
    expect(parts).toEqual([
      ['A', 'OpenSpiel', null],
      ['SPAN', '•', 'true'],
      ['H1', 'Games', null],
    ]);
  });
});

describe('App, on a game page', () => {
  it('starts a game, plays it, and keeps its session in the address', async () => {
    const container = open('/game/count');
    await until(() => {
      expect(legal(container, 'zero')).toBeDefined();
    });
    const create = server.received.find(({method}) => method === 'POST');
    expect(create?.body).toMatchObject({
      game: 'count',
      seats: [{controller: 'human'}, {controller: 'human'}],
    });
    click(legal(container, 'one'));
    await until(() => {
      expect(container.querySelector('.history')?.textContent).toContain(
        '#1one',
      );
    });
    const url = new URL(window.location.href);
    expect(url.pathname).toBe('/game/count');
    expect(url.searchParams.get('viewer')).toBe('oracle');
    const id = url.searchParams.get('session');
    expect(id).not.toBeNull();
    expect(server.history(id ?? '')).toEqual([1]);
  });
});

describe('App, choosing the viewer', () => {
  it('opens for the default viewer when the address names a seat the game lacks', async () => {
    const container = open('/game/count?viewer=7');
    await until(() => {
      expect(legal(container, 'zero')).toBeDefined();
    });
    expect(perspective(container).value).toBe('oracle');
    expect(server.received.every(({viewer}) => viewer !== '7')).toBe(true);
  });

  it('shows the game to the viewer chosen', async () => {
    const container = open('/game/count');
    await until(() => {
      expect(legal(container, 'zero')).toBeDefined();
    });
    enter(perspective(container), 'spectator');
    await until(() => {
      expect(server.received.at(-1)?.viewer).toBe('spectator');
      expect(container.querySelector('.gameplay__status')?.textContent).toBe(
        'Turn 0 · Player 0 · Spectating',
      );
    });
    expect(new URL(window.location.href).searchParams.get('viewer')).toBe(
      'spectator',
    );
  });

  it('hands the device over behind a screen in a game that hides information', async () => {
    server = new FakeServer({hidesInformation: true});
    const container = open('/game/count');
    await until(() => {
      expect(legal(container, 'zero')).toBeDefined();
    });
    expect(perspective(container).value).toBe('0');
    click(legal(container, 'zero'));
    await until(() => {
      expect(button('Hand over to Player 1', container)).toBeDefined();
    });
    click(button('Hand over to Player 1', container));
    const screen = document.querySelector('[role="dialog"][aria-modal]');
    expect(screen?.textContent).toContain('Pass the device to Player 1');
    expect(container.querySelector('.game-page')?.hasAttribute('inert')).toBe(
      true,
    );
    click(button('Show Player 1’s view', container));
    expect(document.querySelector('[aria-modal]')).toBeNull();
    await until(() => {
      expect(legal(container, 'one')).toBeDefined();
    });
    expect(perspective(container).value).toBe('1');
  });
});
