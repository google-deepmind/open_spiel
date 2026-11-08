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

import type {IDockviewPanelProps} from 'dockview-core';
import {act, type JSX} from 'react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {button, only, render, type Rendered} from '../testing/render.tsx';
import {Dock} from './dock.tsx';
import {useShowsSessionError} from './dock_context.ts';
import {HeaderActionsContext} from './header_actions.ts';
import {PanelGroup} from './panel_group.tsx';
import type {PanelGroupDefinition} from './panel_groups.ts';
import {PANELS, type PanelId} from './testing/panels.ts';
import {SETTLE_MS} from './resize.ts';

/** A panel that shows the session error where the dock says, and can close. */
function TestPanel({api}: IDockviewPanelProps): JSX.Element {
  const showsError = useShowsSessionError(api.id);
  return (
    <div data-panel={api.id}>
      {showsError && <p role="alert">The session failed.</p>}
      <button
        type="button"
        onClick={() => {
          api.close();
        }}
      >
        Close {api.title}
      </button>
    </div>
  );
}

/** The Setup panel's sections, a group whose sections can pop out. */
const SETUP_GROUP: PanelGroupDefinition = {
  id: 'setup',
  title: 'Setup',
  sections: [
    {id: 'options', title: 'Options', Content: () => <p>The options.</p>},
    {id: 'position', title: 'Position', Content: () => <p>The position.</p>},
  ],
};

function SetupPanel(props: IDockviewPanelProps): JSX.Element {
  return (
    <>
      <TestPanel {...props} />
      <PanelGroup group={SETUP_GROUP} />
    </>
  );
}

const COMPONENTS = {
  ...(Object.fromEntries(PANELS.list.map(({id}) => [id, TestPanel])) as Record<
    PanelId,
    typeof TestPanel
  >),
  setup: SetupPanel,
};

/** The one button under `parent` named `name` by its aria-label. */
function named(name: string, parent: ParentNode = document): HTMLElement {
  return only(parent, `button[aria-label="${name}"]`);
}

/** What each ResizeObserver watches, and the callback it calls. */
const watchers = new Map<Element, ResizeObserverCallback>();
let rendered: Rendered | undefined;

class FakeResizeObserver {
  readonly #callback: ResizeObserverCallback;
  constructor(callback: ResizeObserverCallback) {
    this.#callback = callback;
  }
  observe(element: Element): void {
    watchers.set(element, this.#callback);
  }
  unobserve = vi.fn();
  disconnect = vi.fn();
}

beforeEach(() => {
  vi.useFakeTimers({toFake: ['setTimeout', 'clearTimeout']});
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  window.localStorage.clear();
  rendered = render(
    <Dock
      panels={PANELS}
      components={COMPONENTS}
      groups={[SETUP_GROUP]}
      game="test"
    />,
  );
});

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  watchers.clear();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

/** Each dock group's tab titles, `|` between tabs, in document order. */
function groups(): string[] {
  return [...document.querySelectorAll('.dv-groupview')].map(group =>
    [...group.querySelectorAll('.dv-tab')]
      .map(tab => tab.textContent)
      .join('|'),
  );
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

/** Resizes the dock to `width` by `height`, then lets `elapsed` ms pass. */
function resize(width: number, height: number, elapsed: number): void {
  const area = only(document, '.dock-area');
  Object.defineProperties(area, {
    clientWidth: {configurable: true, value: width},
    clientHeight: {configurable: true, value: height},
  });
  const watcher = watchers.get(area);
  if (watcher === undefined) {
    throw new Error('Nothing watches the dock area for resizes.');
  }
  act(() => {
    watcher([], new FakeResizeObserver(watcher));
  });
  act(() => {
    vi.advanceTimersByTime(elapsed);
  });
}

/** Brings the dock tab titled `title` forward, as pressing it does. */
function pressTab(title: string): void {
  const [tab, ...others] = [...document.querySelectorAll('.dv-tab')].filter(
    candidate => candidate.textContent === title,
  );
  if (tab === undefined || others.length > 0) {
    throw new Error(`Expected one "${title}" tab.`);
  }
  act(() => {
    tab.dispatchEvent(new MouseEvent('pointerdown', {bubbles: true}));
  });
}

function panelsShowingTheError(): Array<string | undefined> {
  return [...document.querySelectorAll('[role="alert"]')].map(
    alert => alert.closest<HTMLElement>('[data-panel]')?.dataset['panel'],
  );
}

describe('Dock', () => {
  it('opens the panels open at first, drawn with the dock theme', () => {
    expect(groups()).toEqual([
      'Gameplay',
      'Actions|State',
      'Setup|View',
      'Players',
    ]);
    expect(document.querySelector('.dockview-theme-open-spiel')).not.toBeNull();
    click(named('Panel options'));
    const drawer = only(document, '[role="dialog"]');
    expect(drawer.textContent).toContain('Closed panels');
    click(button('Session', drawer));
    expect(groups()).toEqual([
      'Gameplay',
      'Actions|State',
      'Setup|View',
      'Players|Session',
    ]);
    expect(drawer.textContent).toContain('All panels are open');
  });

  it('reopens a closed panel from the drawer, in its group', () => {
    click(button('Close Actions'));
    expect(groups()).toEqual(['Gameplay', 'State', 'Setup|View', 'Players']);
    click(named('Panel options'));
    click(button('Actions'));
    expect(groups()).toEqual([
      'Gameplay',
      'State|Actions',
      'Setup|View',
      'Players',
    ]);
  });

  it('puts its controls in the page header when there is one', () => {
    rendered?.unmount();
    const header = document.createElement('div');
    document.body.append(header);
    rendered = render(
      <HeaderActionsContext value={header}>
        <Dock panels={PANELS} components={COMPONENTS} game="test" />
      </HeaderActionsContext>,
    );
    expect(document.querySelector('.dock__actions')).toBeNull();
    expect(named('Panel options', header)).not.toBeNull();
    expect(only(header, '[aria-haspopup="menu"]').textContent).toBe(
      'Layouts ▾',
    );
    header.remove();
  });
});

describe('Dock, its panel groups', () => {
  it('pops a section out beside its group, and docks it back', () => {
    click(named('Pop out Options'));
    expect(groups()).toEqual([
      'Gameplay',
      'Actions|State',
      'Setup|View|Options',
      'Players',
    ]);
    expect(only(document, '.panel-group__from').textContent).toBe('From Setup');
    pressTab('Setup');
    const tabs = [...document.querySelectorAll('[role="tab"]')].map(
      tab => tab.textContent,
    );
    expect(tabs).toEqual(['Position']);
    pressTab('Options');
    click(named('Dock Options into Setup'));
    expect(groups()).toEqual([
      'Gameplay',
      'Actions|State',
      'Setup|View',
      'Players',
    ]);
    const selected = only(document, '[role="tab"][aria-selected="true"]');
    expect(selected.textContent).toBe('Options');
  });

  it('remembers the open section for the game', () => {
    click(button('Position'));
    expect(
      JSON.parse(localStorage.getItem('open_spiel.panel_groups') ?? ''),
    ).toEqual({version: 1, games: {test: {setup: 'position'}}});
  });

  it('keeps a popped out section through a rebuild', () => {
    click(button('Position'));
    click(named('Pop out Position'));
    resize(800, 600, SETTLE_MS);
    expect(groups()).toEqual([
      'Gameplay|Actions|State|Setup|View|Players|Position',
    ]);
  });
});

describe('Dock, its session error and its focus', () => {
  it('shows the session error once, in Gameplay', () => {
    expect(panelsShowingTheError()).toEqual(['gameplay']);
  });

  it('shows the session error once, in the first panel showing, while Gameplay is closed', () => {
    click(button('Close Gameplay'));
    expect(panelsShowingTheError()).toEqual(['actions']);
  });

  it('moves the session error to the tab showing in the tabbed tier', () => {
    resize(800, 600, SETTLE_MS);
    expect(panelsShowingTheError()).toEqual(['gameplay']);
    pressTab('Actions');
    expect(panelsShowingTheError()).toEqual(['actions']);
    pressTab('Gameplay');
    expect(panelsShowingTheError()).toEqual(['gameplay']);
  });

  it('moves focus in the drawer to the panel listed next as one reopens', () => {
    click(button('Close Actions'));
    click(button('Close Setup'));
    click(named('Panel options'));
    expect(document.activeElement).toBe(button('Actions'));
    click(button('Actions'));
    expect(document.activeElement).toBe(button('Setup'));
    click(button('Setup'));
    expect(document.activeElement).toBe(button('Session'));
    click(button('Session'));
    expect(document.activeElement).toBe(named('Close panel options'));
  });
});

describe('Dock, resized', () => {
  it('rebuilds once a resize settles in another tier, closed panels closed', () => {
    click(button('Close Players'));
    resize(800, 600, SETTLE_MS - 1);
    expect(groups()).toHaveLength(3);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(groups()).toEqual(['Gameplay|Actions|State|Setup|View']);
  });

  it('keeps its panels through a resize within its tier', () => {
    const gameplay = only(document, '[data-panel="gameplay"]');
    resize(1700, 900, SETTLE_MS);
    expect(only(document, '[data-panel="gameplay"]')).toBe(gameplay);
  });

  it('remounts its panels when it rebuilds', () => {
    const gameplay = only(document, '[data-panel="gameplay"]');
    resize(1200, 900, SETTLE_MS);
    expect(groups()).toEqual([
      'Gameplay',
      'Actions|State',
      'Setup|View',
      'Players',
    ]);
    expect(only(document, '[data-panel="gameplay"]')).not.toBe(gameplay);
  });
});

describe('useShowsSessionError', () => {
  it('refuses a component outside a Dock', () => {
    function Outside(): JSX.Element {
      return <p>{String(useShowsSessionError('gameplay'))}</p>;
    }
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    expect(() => render(<Outside />)).toThrow(
      'useShowsSessionError is called outside a Dock.',
    );
  });
});
