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

import {
  Orientation,
  createDockview,
  type DockviewApi,
  type SerializedDockview,
} from 'dockview-core';
import {afterEach, beforeAll, describe, expect, it, vi} from 'vitest';

import {buildDock, openPanel} from './build.ts';
import {definePanels} from './panels.ts';
import {PANELS} from './testing/panels.ts';

const ALL = new Set(PANELS.list.map(({id}) => id));

function without(...closed: string[]): Set<string> {
  return new Set([...ALL].filter(id => !closed.includes(id)));
}
const HEIGHT = 900;
const WIDE = {width: 1800, height: HEIGHT};
/** Two columns, but too short for three groups of 200 px stacked. */
const SHORT = {width: 1200, height: 500};

/** A board and a notes panel beside it whose minimums need 1100 px. */
const WIDE_MINIMUMS = definePanels([
  {
    id: 'board',
    title: 'Board',
    primary: true,
    minimumWidth: 600,
    minimumHeight: 200,
  },
  {
    id: 'notes',
    title: 'Notes',
    position: {referencePanel: 'board', direction: 'right'},
    minimumWidth: 500,
    minimumHeight: 200,
  },
]);
const docks: DockviewApi[] = [];

beforeAll(() => {
  // Dockview watches its element with a ResizeObserver, which jsdom lacks;
  // these tests lay the dock out by hand instead.
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = vi.fn();
    },
  );
});

afterEach(() => {
  for (const dock of docks.splice(0)) {
    dock.dispose();
  }
  document.body.replaceChildren();
});

function emptyDock(): DockviewApi {
  const element = document.createElement('div');
  document.body.append(element);
  const dock = createDockview(element, {
    createComponent: () => ({
      element: document.createElement('div'),
      init: vi.fn(),
    }),
  });
  docks.push(dock);
  return dock;
}

/**
 * The dock's groups as rows and columns of tab groups, `|` between tabs. A
 * row or column of one is written as its member, which it looks like.
 */
function shape(dock: DockviewApi): string {
  const {grid} = dock.toJSON();
  return describeNode(grid.root, grid.orientation);
}

function describeNode(
  node: SerializedDockview['grid']['root'],
  orientation: Orientation,
): string {
  if (!Array.isArray(node.data)) {
    return node.data.views.join('|');
  }
  const across =
    orientation === Orientation.HORIZONTAL
      ? Orientation.VERTICAL
      : Orientation.HORIZONTAL;
  const [only, ...others] = node.data;
  if (only !== undefined && others.length === 0) {
    return describeNode(only, across);
  }
  const children = node.data.map(child => describeNode(child, across));
  const kind = orientation === Orientation.HORIZONTAL ? 'row' : 'column';
  return `${kind}(${children.join(', ')})`;
}

function groupHeight(dock: DockviewApi, id: string): number {
  const panel = dock.getPanel(id);
  expect(panel).toBeDefined();
  return panel?.group.api.height ?? 0;
}

describe('buildDock', () => {
  it('builds three columns at 1800 px', () => {
    const dock = emptyDock();
    expect(buildDock(dock, PANELS, ALL, WIDE)).toEqual({
      tier: 'three',
      widthScale: 1,
      heightScale: 1,
    });
    expect(shape(dock)).toBe(
      'row(gameplay, actions|state, column(setup|view, players|session))',
    );
  });

  it('stacks three groups beside Gameplay at 1200 px', () => {
    const dock = emptyDock();
    expect(buildDock(dock, PANELS, ALL, {width: 1200, height: HEIGHT})).toEqual(
      {
        tier: 'two',
        widthScale: 1,
        heightScale: 1,
      },
    );
    expect(shape(dock)).toBe(
      'row(gameplay, column(actions|state, setup|view, players|session))',
    );
  });

  it('makes one group of tabs, Gameplay showing, at 800 px', () => {
    const dock = emptyDock();
    expect(buildDock(dock, PANELS, ALL, {width: 800, height: HEIGHT})).toEqual({
      tier: 'tabs',
      widthScale: 1,
      heightScale: 1,
    });
    expect(shape(dock)).toBe(
      'gameplay|actions|state|setup|view|players|session',
    );
    expect(dock.activePanel?.id).toBe('gameplay');
  });

  it('shows the first panel of each group', () => {
    const dock = emptyDock();
    buildDock(dock, PANELS, ALL, WIDE);
    const showing = dock.groups.map(group => group.activePanel?.id);
    expect(showing).toEqual(['gameplay', 'actions', 'setup', 'players']);
  });

  it('shares a column evenly among its groups', () => {
    const dock = emptyDock();
    buildDock(dock, PANELS, ALL, {width: 1200, height: HEIGHT});
    const heights = ['actions', 'setup', 'players'].map(id =>
      groupHeight(dock, id),
    );
    expect(heights.reduce((sum, height) => sum + height, 0)).toBe(HEIGHT);
    expect(Math.max(...heights) - Math.min(...heights)).toBeLessThanOrEqual(1);
  });

  it('fits every group inside a dock too short for its minimum heights', () => {
    const dock = emptyDock();
    const fit = buildDock(dock, PANELS, ALL, SHORT);
    expect(fit.heightScale).toBeCloseTo(500 / 600);
    const heights = ['actions', 'setup', 'players'].map(id =>
      groupHeight(dock, id),
    );
    expect(heights.reduce((sum, height) => sum + height, 0)).toBe(500);
    expect(groupHeight(dock, 'gameplay')).toBe(500);
  });

  it('gives each column its width', () => {
    const dock = emptyDock();
    buildDock(dock, PANELS, ALL, WIDE);
    const widths = ['gameplay', 'actions', 'setup'].map(
      id => dock.getPanel(id)?.group.api.width,
    );
    expect(widths).toEqual([1100, 360, 340]);
  });

  it('replaces whatever the dock held', () => {
    const dock = emptyDock();
    buildDock(dock, PANELS, ALL, WIDE);
    buildDock(dock, PANELS, new Set(['gameplay', 'players']), {
      width: 1200,
      height: HEIGHT,
    });
    expect(shape(dock)).toBe('row(gameplay, players)');
  });
});

describe('openPanel', () => {
  it('returns a closed panel to its group', () => {
    const dock = emptyDock();
    const fit = buildDock(dock, PANELS, ALL, WIDE);
    dock.getPanel('actions')?.api.close();
    expect(shape(dock)).toBe(
      'row(gameplay, state, column(setup|view, players|session))',
    );
    openPanel(dock, PANELS, 'actions', fit, WIDE);
    expect(shape(dock)).toBe(
      'row(gameplay, state|actions, column(setup|view, players|session))',
    );
    expect(dock.activePanel?.id).toBe('actions');
  });

  it('returns a closed group to its place', () => {
    const dock = emptyDock();
    const fit = buildDock(dock, PANELS, ALL, WIDE);
    dock.getPanel('players')?.api.close();
    dock.getPanel('session')?.api.close();
    openPanel(dock, PANELS, 'players', fit, WIDE);
    expect(shape(dock)).toBe(
      'row(gameplay, actions|state, column(setup|view, players))',
    );
  });

  it('returns Gameplay to the left edge', () => {
    const dock = emptyDock();
    const size = {width: 1200, height: HEIGHT};
    const fit = buildDock(dock, PANELS, ALL, size);
    dock.getPanel('gameplay')?.api.close();
    openPanel(dock, PANELS, 'gameplay', fit, size);
    expect(shape(dock)).toBe(
      'row(gameplay, column(actions|state, setup|view, players|session))',
    );
  });

  it('returns a closed panel to the one group of the tabbed tier', () => {
    const dock = emptyDock();
    const size = {width: 800, height: HEIGHT};
    const fit = buildDock(dock, PANELS, ALL, size);
    dock.getPanel('gameplay')?.api.close();
    openPanel(dock, PANELS, 'gameplay', fit, size);
    expect(shape(dock)).toBe(
      'gameplay|actions|state|setup|view|players|session',
    );
  });

  it('brings an open panel forward', () => {
    const dock = emptyDock();
    const fit = buildDock(dock, PANELS, ALL, WIDE);
    expect(openPanel(dock, PANELS, 'state', fit, WIDE)).toBe(fit);
    expect(dock.getPanel('state')?.group.activePanel?.id).toBe('state');
    expect(dock.panels).toHaveLength(ALL.size);
  });
});

describe('openPanel, in a dock scaled to fit', () => {
  it('reopens a panel with the minimum width the dock was scaled to', () => {
    const dock = emptyDock();
    const size = {width: 1000, height: HEIGHT};
    const fit = buildDock(
      dock,
      WIDE_MINIMUMS,
      new Set(['board', 'notes']),
      size,
    );
    expect(fit.widthScale).toBeCloseTo(1000 / 1100);
    dock.getPanel('notes')?.api.close();
    expect(openPanel(dock, WIDE_MINIMUMS, 'notes', fit, size)).toBe(fit);
    expect(shape(dock)).toBe('row(board, notes)');
    expect(dock.getPanel('notes')?.minimumWidth).toBe(454);
  });

  it('rebuilds a dock whose minimums no longer fit once a panel reopens', () => {
    const dock = emptyDock();
    const size = {width: 1000, height: HEIGHT};
    const fit = buildDock(dock, WIDE_MINIMUMS, new Set(['board']), size);
    expect(fit.widthScale).toBe(1);
    const refit = openPanel(dock, WIDE_MINIMUMS, 'notes', fit, size);
    expect(refit.widthScale).toBeCloseTo(1000 / 1100);
    expect(shape(dock)).toBe('row(board, notes)');
    const widths = ['board', 'notes'].map(
      id => dock.getPanel(id)?.group.api.width,
    );
    expect(widths).toEqual([546, 454]);
    expect(dock.activePanel?.id).toBe('notes');
  });

  it('rebuilds a dock whose column no longer fits its height', () => {
    const dock = emptyDock();
    const fit = buildDock(dock, PANELS, without('players', 'session'), SHORT);
    expect(fit.heightScale).toBe(1);
    const refit = openPanel(dock, PANELS, 'players', fit, SHORT);
    expect(refit.heightScale).toBeCloseTo(500 / 600);
    const heights = ['actions', 'setup', 'players'].map(id =>
      groupHeight(dock, id),
    );
    expect(heights.reduce((sum, height) => sum + height, 0)).toBe(500);
    expect(dock.activePanel?.id).toBe('players');
  });
});
