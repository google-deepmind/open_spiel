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

import {Orientation, type SerializedDockview} from 'dockview-core';
import {act, type JSX} from 'react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {enter} from '../testing/input.ts';
import {only, render, type Rendered} from '../testing/render.tsx';
import {
  type DockLayout,
  LAYOUTS_STORAGE_KEY,
  readLayouts,
  writeLayouts,
} from './layout_store.ts';
import {SavedLayoutsBar, scrollToShow} from './saved_layouts.tsx';
import {
  type DockEvent,
  type LayoutActions,
  useSavedLayouts,
} from './use_saved_layouts.ts';

const GAME = 'tic_tac_toe';
const PANELS = new Set(['gameplay', 'state']);
function storage(): Storage {
  return window.localStorage;
}

/** A dock whose one group shows `views`. */
function dockOf(...views: string[]): SerializedDockview {
  return {
    grid: {
      root: {
        type: 'branch',
        data: [{type: 'leaf', data: {id: '1', views, activeView: views[0]}}],
      },
      height: 600,
      width: 800,
      orientation: Orientation.HORIZONTAL,
    },
    panels: Object.fromEntries(
      views.map(id => [id, {id, contentComponent: id}]),
    ),
  } as SerializedDockview;
}

const BUILT: DockLayout = {tier: 'two', dock: dockOf('gameplay')};
const CHANGED: DockLayout = {tier: 'two', dock: dockOf('gameplay', 'state')};

/** A dock the saved layouts drive, which records what they do to it. */
class FakeDock implements LayoutActions {
  now: DockLayout = BUILT;
  readonly listeners = new Set<(event: DockEvent) => void>();
  readonly restore = vi.fn((layout: DockLayout) => {
    this.now = {tier: layout.tier, dock: layout.dock};
    this.emit('change');
  });
  readonly reset = vi.fn(() => {
    this.now = BUILT;
    this.emit('build');
  });
  capture = (): DockLayout => this.now;
  subscribe = (listener: (event: DockEvent) => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
  /** The user rearranges the dock. */
  change(layout: DockLayout): void {
    act(() => {
      this.now = layout;
      this.emit('change');
    });
  }
  private emit(event: DockEvent): void {
    for (const listener of this.listeners) listener(event);
  }
}

let rendered: Rendered | undefined;
let dock: FakeDock;

function Bar(): JSX.Element {
  const layouts = useSavedLayouts({game: GAME, panels: PANELS, actions: dock});
  return <SavedLayoutsBar layouts={layouts} />;
}

function mount(): HTMLElement {
  rendered?.unmount();
  rendered = render(<Bar />);
  return rendered.container;
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

function menuButton(): HTMLElement {
  return only(document, '[aria-haspopup="menu"]');
}

/** The menu item that reads `text`, the menu opened first if it is shut. */
function item(text: string): HTMLButtonElement {
  if (document.querySelector('[role="menu"]') === null) click(menuButton());
  const found = [
    ...document.querySelectorAll<HTMLButtonElement>('[role="menuitem"]'),
  ].filter(candidate => candidate.textContent === text);
  const [element] = found;
  if (found.length !== 1 || element === undefined) {
    const items = [...document.querySelectorAll('[role="menuitem"]')];
    throw new Error(
      `No item "${text}" in ${items.map(i => i.textContent).join(' | ')}.`,
    );
  }
  return element;
}

function using(): string {
  if (document.querySelector('[role="menu"]') === null) click(menuButton());
  return only(document, '.layouts__using').textContent;
}

function closeMenu(): void {
  act(() => {
    document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
  });
}

/** Names a layout through the menu's field and submits it. */
function name(text: string): void {
  const field = only(document, '.layouts__form input') as HTMLInputElement;
  enter(field, text);
  click(only(document, '.layouts__form button[type="submit"]'));
}

/** The bar's buttons and labels, in order, as they read. */
function strip(): string[] {
  return [...only(document, '.layouts__strip').children].map(child =>
    child.getAttribute('aria-pressed') === 'true'
      ? `[${child.textContent}]`
      : child.textContent,
  );
}

function saved(): unknown {
  return JSON.parse(localStorage.getItem(LAYOUTS_STORAGE_KEY) ?? 'null');
}

beforeEach(() => {
  window.localStorage.clear();
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe = vi.fn();
      disconnect = vi.fn();
    },
  );
  dock = new FakeDock();
});

afterEach(() => {
  closeMenu();
  rendered?.unmount();
  rendered = undefined;
  vi.unstubAllGlobals();
});

describe('SavedLayoutsBar, saving', () => {
  it('says the built-in layout is in use while nothing is saved', () => {
    mount();
    expect(strip()).toEqual([]);
    expect(menuButton().textContent).toBe('Layouts ▾');
    expect(using()).toBe(
      "Using: this game's built-in layout (no default is set)",
    );
  });

  it('saves the dock for this game and shows it, pressed', () => {
    mount();
    click(item('Save for this game…'));
    name(' wide ');
    expect(strip()).toEqual(['This game', '[wide]']);
    expect(saved()).toMatchObject({
      version: 2,
      layouts: [{name: 'wide', game: GAME, ...BUILT}],
      defaults: {all: null, games: {}},
    });
    expect(using()).toBe('Using: ‘wide’');
  });

  it('says a layout is modified, and saves it again', () => {
    mount();
    click(item('Save for this game…'));
    name('wide');
    dock.change(CHANGED);
    expect(using()).toBe('Using: ‘wide’ (modified)');
    click(item('Save ‘wide’ (modified)'));
    expect(saved()).toMatchObject({layouts: [CHANGED]});
    expect(using()).toBe('Using: ‘wide’');
  });

  it('refuses a name this game uses already', () => {
    mount();
    click(item('Save for this game…'));
    name('wide');
    click(item('Save for this game…'));
    enter(only(document, '.layouts__form input') as HTMLInputElement, 'Wide');
    expect(only(document, '.layouts__problem').textContent).toBe(
      'A layout named "Wide" is saved already.',
    );
    expect(
      (only(document, '.layouts__form button') as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('calls the dock an unsaved arrangement once it is changed', () => {
    mount();
    click(item('Save for this game…'));
    name('wide');
    click(item('Delete ‘wide’…'));
    click(item('Really delete ‘wide’?'));
    expect(strip()).toEqual([]);
    expect(saved()).toMatchObject({layouts: []});
    writeLayouts(storage, {
      layouts: [{id: 'a', name: 'a', game: GAME, ...BUILT}],
      defaults: {all: null, games: {}},
    });
    mount();
    dock.change(CHANGED);
    expect(strip()).toEqual(['Unsaved arrangement', 'This game', 'a']);
  });
});

describe('SavedLayoutsBar, defaults', () => {
  it('makes the dock the default for this game, and opens with it', () => {
    mount();
    click(item('Make current default for this game…'));
    name('mine');
    expect(strip()).toEqual(['This game', '[mine(default)]']);
    expect(using()).toBe('Using: ‘mine’, default for this game');
    closeMenu();
    dock = new FakeDock();
    mount();
    expect(dock.restore).toHaveBeenCalledOnce();
    expect(strip()).toEqual(['This game', '[mine(default)]']);
  });

  it('keeps a default for all games under its own label', () => {
    mount();
    click(item('Make default for all games…'));
    name('everywhere');
    expect(strip()).toEqual(['All games', '[everywhere(default)]']);
    expect(using()).toBe('Using: ‘everywhere’, default for all games');
    click(item('Stop using ‘everywhere’ as default for all games'));
    expect(saved()).toMatchObject({defaults: {all: null}});
  });

  it('asks before resetting to the built-in layout, then says so', () => {
    mount();
    click(item('Make current default for this game…'));
    name('mine');
    click(item('Reset to built-in layout…'));
    expect(dock.reset).not.toHaveBeenCalled();
    click(item('Really reset to the built-in layout?'));
    expect(dock.reset).toHaveBeenCalledOnce();
    expect(strip()).toEqual([
      'Built-in layout (you reset)',
      'This game',
      'mine(default)',
    ]);
    expect(using()).toBe(
      "Using: this game's built-in layout, because you reset to it. Default still set: ‘mine’ (this game)",
    );
    click(item('Reset to ‘mine’ (this game)…'));
    click(item('Really reset to ‘mine’?'));
    expect(dock.restore).toHaveBeenCalledTimes(1);
  });
});

describe('SavedLayoutsBar, editing', () => {
  it('renames a layout and moves it among this game’s', () => {
    mount();
    for (const layout of ['one', 'two']) {
      click(item('Save for this game…'));
      name(layout);
    }
    click(item('Rename ‘two’…'));
    name('second');
    expect(strip()).toEqual(['This game', 'one', '[second]']);
    expect(item('Move ‘second’ right').disabled).toBe(true);
    click(item('Move ‘second’ left'));
    expect(strip()).toEqual(['This game', '[second]', 'one']);
  });

  it('shows the built-in layout when a saved one cannot be restored', () => {
    writeLayouts(storage, {
      layouts: [{id: 'a', name: 'wide', game: GAME, ...CHANGED}],
      defaults: {all: null, games: {}},
    });
    dock.restore.mockImplementation(() => {
      throw new Error('bad grid');
    });
    mount();
    click(only(document, '.layouts__button'));
    expect(dock.reset).toHaveBeenCalledOnce();
    expect(menuButton().textContent).toBe('Layouts (failed) ▾');
    click(menuButton());
    expect(only(document, '.layouts__error').textContent).toContain('bad grid');
    click(only(document, '.layouts__error button'));
    expect(menuButton().textContent).toBe('Layouts ▾');
  });
});

describe('SavedLayoutsBar, sharing storage with other pages', () => {
  it('keeps a layout another tab saved after this one read them', () => {
    mount();
    writeLayouts(storage, {
      layouts: [{id: 'other', name: 'other tab', game: GAME, ...BUILT}],
      defaults: {all: null, games: {}},
    });
    click(item('Save for this game…'));
    name('wide');
    const reading = readLayouts(storage, PANELS);
    expect(
      reading.ok && reading.store.layouts.map(layout => layout.name),
    ).toEqual(['other tab', 'wide']);
    expect(strip()).toEqual(['This game', 'other tab', '[wide]']);
  });

  it('never overwrites layouts a newer page wrote after this one read them', () => {
    mount();
    const newer = '{"version": 9, "layouts": []}';
    localStorage.setItem(LAYOUTS_STORAGE_KEY, newer);
    click(item('Save for this game…'));
    name('wide');
    expect(localStorage.getItem(LAYOUTS_STORAGE_KEY)).toBe(newer);
    expect(only(document, '.layouts__error').textContent).toContain(
      'can no longer be read',
    );
    expect(only(document, '[role="alert"]').textContent).toContain('version 9');
  });

  it('leaves unreadable layouts alone until they are cleared', () => {
    localStorage.setItem(LAYOUTS_STORAGE_KEY, '{"version": 9}');
    mount();
    click(item('Clear saved layouts'));
    expect(localStorage.getItem(LAYOUTS_STORAGE_KEY)).toBeNull();
    expect(using()).toBe(
      "Using: this game's built-in layout (no default is set)",
    );
  });
});

describe('scrollToShow', () => {
  it.each([
    [0, 100, 20, 30, 0],
    [50, 100, 20, 30, 20],
    [0, 100, 90, 30, 20],
  ])(
    'from %i in a %i px strip shows [%i, +%i) at %i',
    (scrollLeft, viewport, left, width, expected) => {
      expect(scrollToShow(scrollLeft, viewport, left, width)).toBe(expected);
    },
  );
});
