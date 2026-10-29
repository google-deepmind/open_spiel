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
import {beforeEach, describe, expect, it} from 'vitest';

import {
  EMPTY_STORE,
  LAYOUTS_STORAGE_KEY,
  type LayoutStore,
  MAX_NAME_LENGTH,
  type SavedLayout,
  clearLayouts,
  layoutPanels,
  layoutSignature,
  nameProblem,
  readLayouts,
  writeLayouts,
} from './layout_store.ts';

const PANELS = new Set(['gameplay', 'actions']);
function storage(): Storage {
  return window.localStorage;
}
function refusing(): Storage {
  throw new DOMException('Storage is disabled.', 'SecurityError');
}

/** A dock of the given panels, each drawn by its own component. */
function dockOf(...ids: string[]): SerializedDockview {
  return {
    grid: {
      root: {type: 'branch', data: []},
      height: 600,
      width: 800,
      orientation: Orientation.HORIZONTAL,
    },
    panels: Object.fromEntries(
      ids.map(id => [id, {id, contentComponent: id, title: id}]),
    ),
  };
}

const WIDE: SavedLayout = {
  id: 'wide-id',
  name: 'wide',
  game: 'tic_tac_toe',
  tier: 'three',
  dock: dockOf('gameplay'),
};

const SHARED: SavedLayout = {...WIDE, id: 'shared-id', game: null};

const STORE: LayoutStore = {
  layouts: [WIDE, SHARED],
  defaults: {all: 'shared-id', games: {tic_tac_toe: 'wide-id'}},
};

function store(value: unknown): void {
  window.localStorage.setItem(LAYOUTS_STORAGE_KEY, JSON.stringify(value));
}

function stored(layouts: unknown[], defaults: unknown = EMPTY_STORE.defaults) {
  return {version: 2, layouts, defaults};
}

beforeEach(() => {
  window.localStorage.clear();
});

describe('readLayouts', () => {
  it('reads the empty store when nothing is saved', () => {
    expect(readLayouts(storage, PANELS)).toEqual({
      ok: true,
      store: EMPTY_STORE,
    });
  });

  it('reads back what writeLayouts stored', () => {
    writeLayouts(storage, STORE);
    expect(readLayouts(storage, PANELS)).toEqual({ok: true, store: STORE});
    expect(layoutPanels(WIDE)).toEqual(['gameplay']);
  });

  it('reads layouts of version 1 as layouts for all games', () => {
    const {name, tier, dock} = WIDE;
    store({version: 1, layouts: [{name, tier, dock}]});
    expect(readLayouts(storage, PANELS)).toEqual({
      ok: true,
      store: {
        layouts: [{id: 'version-1:wide', name, game: null, tier, dock}],
        defaults: EMPTY_STORE.defaults,
      },
    });
  });
});

describe('readLayouts, refusing storage', () => {
  it.each<[string, unknown, string]>([
    ['a newer version', {version: 3, layouts: []}, 'version 3'],
    ['no version', {layouts: []}, 'no version'],
    ['no list', {version: 2, layouts: {}}, 'not a list'],
    ['an entry with no name', stored([{...WIDE, name: ''}]), 'no valid name'],
    ['an entry with no id', stored([{...WIDE, id: ''}]), 'has no id'],
    ['an entry with no game', stored([{...WIDE, game: 1}]), 'names no game'],
    [
      'an entry with an unknown tier',
      stored([{...WIDE, tier: 'four'}]),
      'no valid tier',
    ],
    ['an entry with no dock', stored([{...WIDE, dock: 1}]), 'holds no dock'],
    [
      'an unknown panel',
      stored([{...WIDE, dock: dockOf('gameplay', 'debug')}]),
      'unknown panel "debug"',
    ],
    [
      'a panel drawn by another component',
      stored([
        {
          ...WIDE,
          dock: {
            ...dockOf(),
            panels: {gameplay: {id: 'gameplay', contentComponent: 'actions'}},
          },
        },
      ]),
      'describes panel "gameplay" as another',
    ],
    [
      'windows of its own',
      stored([{...WIDE, dock: {...dockOf('gameplay'), popoutGroups: [{}]}}]),
      'opens windows',
    ],
    ['two layouts of one id', stored([WIDE, WIDE]), 'share an id'],
    [
      'two layouts of one name for one game',
      stored([WIDE, {...WIDE, id: 'other', name: 'WIDE'}]),
      'share the name',
    ],
    ['no defaults', {version: 2, layouts: []}, 'defaults are not readable'],
    [
      'a default for all games that is one game’s',
      stored([WIDE], {all: 'wide-id', games: {}}),
      'not a layout for all games',
    ],
    [
      'a default of another game',
      stored([WIDE], {all: null, games: {chess: 'wide-id'}}),
      'default for chess',
    ],
  ])('refuses %s, whole', (_, value, problem) => {
    store(value);
    const reading = readLayouts(storage, PANELS);
    expect(reading.ok).toBe(false);
    expect(reading.ok ? '' : reading.problem).toContain(problem);
  });
});

describe('readLayouts, other storage', () => {
  it('admits one name for a game and for all games', () => {
    writeLayouts(storage, {...STORE, defaults: EMPTY_STORE.defaults});
    expect(readLayouts(storage, PANELS).ok).toBe(true);
  });

  it('refuses text that is not JSON', () => {
    window.localStorage.setItem(LAYOUTS_STORAGE_KEY, '{');
    expect(readLayouts(storage, PANELS)).toEqual({
      ok: false,
      problem: 'The saved layouts are not valid JSON.',
    });
  });

  it('says so when the browser keeps nothing', () => {
    const reading = readLayouts(refusing, PANELS);
    expect(reading.ok ? '' : reading.problem).toContain('keeps no layouts');
  });
});

describe('writeLayouts and clearLayouts', () => {
  it('throws when storage does', () => {
    expect(() => {
      writeLayouts(refusing, STORE);
    }).toThrow('Storage is disabled.');
  });

  it('clears every layout', () => {
    writeLayouts(storage, STORE);
    clearLayouts(storage);
    expect(window.localStorage.getItem(LAYOUTS_STORAGE_KEY)).toBeNull();
  });
});

describe('nameProblem', () => {
  it('admits a name of up to the limit', () => {
    expect(nameProblem('a'.repeat(MAX_NAME_LENGTH))).toBeNull();
  });

  it.each(['', '   ', 'a'.repeat(MAX_NAME_LENGTH + 1)])('refuses %j', name => {
    expect(nameProblem(name)).not.toBeNull();
  });

  it('refuses a name its siblings use, in any case, but not its own', () => {
    expect(nameProblem(' Wide ', [WIDE])).toBe(
      'A layout named "Wide" is saved already.',
    );
    expect(nameProblem('Wide', [WIDE], WIDE.id)).toBeNull();
  });
});

describe('layoutSignature', () => {
  it('ignores sizes and keeps which panels are where', () => {
    const leaf = (views: string[], size: number): unknown => ({
      type: 'leaf',
      size,
      data: {id: views.join(), views, activeView: views[0]},
    });
    const docked = (...leaves: unknown[]): SavedLayout => ({
      ...WIDE,
      dock: {
        ...dockOf('gameplay', 'actions'),
        grid: {...dockOf().grid, root: {type: 'branch', data: leaves}},
      } as SerializedDockview,
    });
    const wide = docked(leaf(['gameplay'], 600), leaf(['actions'], 400));
    const narrow = docked(leaf(['gameplay'], 500), leaf(['actions'], 300));
    const swapped = docked(leaf(['actions'], 600), leaf(['gameplay'], 400));
    expect(layoutSignature(wide)).toBe(layoutSignature(narrow));
    expect(layoutSignature(wide)).not.toBe(layoutSignature(swapped));
  });
});
