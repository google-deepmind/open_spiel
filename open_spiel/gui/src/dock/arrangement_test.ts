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

import {describe, expect, it} from 'vitest';

import {
  SIDE_COLUMN_WIDTH,
  THREE_COLUMNS_FROM,
  TWO_COLUMNS_FROM,
  arrange,
  fitScale,
  panelOptions,
  placementOf,
  tierFor,
  type Size,
} from './arrangement.ts';
import {definePanels, type Panels} from './panels.ts';
import {PANELS, type PanelId} from './testing/panels.ts';

const ALL = new Set(PANELS.list.map(({id}) => id));

/** A dock `width` pixels wide and tall enough for any column. */
function wide(width: number): Size {
  return {width, height: 900};
}

function without(...closed: string[]): Set<string> {
  return new Set([...ALL].filter(id => !closed.includes(id)));
}

describe('tierFor', () => {
  it.each([
    [0, 'three'],
    [THREE_COLUMNS_FROM + 400, 'three'],
    [THREE_COLUMNS_FROM, 'three'],
    [THREE_COLUMNS_FROM - 1, 'two'],
    [TWO_COLUMNS_FROM, 'two'],
    [TWO_COLUMNS_FROM - 1, 'tabs'],
    [390, 'tabs'],
  ])('lays out a dock %i px wide in the %s tier', (width, tier) => {
    expect(tierFor(width)).toBe(tier);
  });
});

describe('placementOf', () => {
  it('uses the declared position while the reference panel is open', () => {
    expect(placementOf(PANELS, 'players', 'three', ALL)).toEqual({
      referencePanel: 'setup',
      direction: 'below',
    });
  });

  it.each([
    ['actions', [], {referencePanel: 'state', direction: 'within'}],
    ['state', ['actions'], {referencePanel: 'setup', direction: 'within'}],
    ['players', ['setup'], {referencePanel: 'session', direction: 'within'}],
    [
      'players',
      ['setup', 'session'],
      {referencePanel: 'view', direction: 'below'},
    ],
    [
      'view',
      ['setup', 'actions'],
      {referencePanel: 'players', direction: 'within'},
    ],
    [
      'setup',
      ['actions', 'state'],
      {referencePanel: 'view', direction: 'within'},
    ],
  ])('places %s with %j closed', (id, closed, placement) => {
    expect(placementOf(PANELS, id, 'three', without(id, ...closed))).toEqual(
      placement,
    );
  });

  it('folds a third column under the second in the two-column tier', () => {
    const open = without('setup', 'view', 'players', 'session');
    expect(placementOf(PANELS, 'players', 'two', open)).toEqual({
      referencePanel: 'actions',
      direction: 'below',
    });
  });

  it('follows its references to the panel that took their place', () => {
    const open = new Set(['session', 'view']);
    expect(placementOf(PANELS, 'state', 'three', open)).toEqual({
      referencePanel: 'view',
      direction: 'within',
    });
  });

  it('returns the primary panel to the left edge', () => {
    expect(placementOf(PANELS, 'gameplay', 'two', without('gameplay'))).toEqual(
      {direction: 'left'},
    );
  });

  it('returns the primary panel to the first tab in the tabbed tier', () => {
    expect(
      placementOf(PANELS, 'gameplay', 'tabs', without('gameplay')),
    ).toEqual({referencePanel: 'actions', direction: 'within', index: 0});
  });

  it('makes every other panel a tab of the primary in the tabbed tier', () => {
    expect(placementOf(PANELS, 'players', 'tabs', ALL)).toEqual({
      referencePanel: 'gameplay',
      direction: 'within',
    });
  });

  it('gives the first panel of an empty dock no position', () => {
    expect(placementOf(PANELS, 'session', 'three', new Set())).toBeUndefined();
  });
});

describe('fitScale', () => {
  it('leaves minimums that fit alone', () => {
    expect(fitScale([300, 400], 700)).toBe(1);
  });

  it('shrinks minimums that do not fit to the length', () => {
    expect(fitScale([600, 400], 800)).toBe(0.8);
  });

  it('leaves an unmeasured dock alone', () => {
    expect(fitScale([600, 400], 0)).toBe(1);
  });
});

/** The options `arrange` gives a panel of PANELS, minimum sizes included. */
function sized(
  id: PanelId,
  title: string,
  position?: {referencePanel: string; direction: string},
): object {
  const {minimumWidth, minimumHeight} = PANELS.get(id);
  return {
    id,
    component: id,
    title,
    inactive: position?.direction === 'within',
    ...(position && {position}),
    minimumWidth,
    minimumHeight,
  };
}

describe('panelOptions', () => {
  const players = PANELS.get('players');
  const position = {referencePanel: 'setup', direction: 'below'} as const;

  it('scales the minimum sizes by the fit', () => {
    const fit = {tier: 'two', widthScale: 0.5, heightScale: 0.75} as const;
    expect(panelOptions(players, position, fit)).toEqual({
      id: 'players',
      component: 'players',
      title: 'Players',
      position,
      minimumWidth: 130,
      minimumHeight: 150,
    });
  });

  it('sets no minimum size in the tabbed tier', () => {
    const fit = {tier: 'tabs', widthScale: 1, heightScale: 1} as const;
    const options = panelOptions(players, undefined, fit);
    expect(options).toEqual({
      id: 'players',
      component: 'players',
      title: 'Players',
    });
  });
});

describe('arrange, in a wide dock', () => {
  const arrangement = arrange(PANELS, ALL, wide(1800));

  it('gives the panels three columns', () => {
    expect(arrangement.tier).toBe('three');
    expect(arrangement.panels).toEqual([
      sized('gameplay', 'Gameplay'),
      sized('actions', 'Actions', {
        referencePanel: 'gameplay',
        direction: 'right',
      }),
      sized('state', 'State', {referencePanel: 'actions', direction: 'within'}),
      sized('setup', 'Setup', {referencePanel: 'actions', direction: 'right'}),
      sized('view', 'View', {referencePanel: 'setup', direction: 'within'}),
      sized('players', 'Players', {
        referencePanel: 'setup',
        direction: 'below',
      }),
      sized('session', 'Session', {
        referencePanel: 'players',
        direction: 'within',
      }),
    ]);
  });

  it('fits without scaling', () => {
    expect(arrangement.widthScale).toBe(1);
    expect(arrangement.heightScale).toBe(1);
  });

  it('gives Gameplay the width the side columns leave', () => {
    expect(arrangement.columns).toEqual([
      {panel: 'gameplay', width: 1100},
      {panel: 'actions', width: 360},
      {panel: 'setup', width: SIDE_COLUMN_WIDTH},
    ]);
    expect(arrangement.stacks).toEqual([['setup', 'players']]);
  });

  it('builds only the open panels, each where its references allow', () => {
    const fewer = arrange(PANELS, without('actions', 'setup'), wide(1800));
    expect(fewer.panels.map(({id, position}) => [id, position])).toEqual([
      ['gameplay', undefined],
      ['state', {referencePanel: 'gameplay', direction: 'right'}],
      ['view', {referencePanel: 'state', direction: 'right'}],
      ['players', {referencePanel: 'view', direction: 'below'}],
      ['session', {referencePanel: 'players', direction: 'within'}],
    ]);
    expect(fewer.columns).toEqual([
      {panel: 'gameplay', width: 1120},
      {panel: 'state', width: SIDE_COLUMN_WIDTH},
      {panel: 'view', width: SIDE_COLUMN_WIDTH},
    ]);
    expect(fewer.stacks).toEqual([['view', 'players']]);
  });
});

describe('arrange, in narrower docks', () => {
  it('stacks three groups beside Gameplay at mid width', () => {
    const arrangement = arrange(PANELS, ALL, wide(1200));
    expect(arrangement.tier).toBe('two');
    expect(arrangement.panels.map(({id, position}) => [id, position])).toEqual([
      ['gameplay', undefined],
      ['actions', {referencePanel: 'gameplay', direction: 'right'}],
      ['state', {referencePanel: 'actions', direction: 'within'}],
      ['setup', {referencePanel: 'actions', direction: 'below'}],
      ['view', {referencePanel: 'setup', direction: 'within'}],
      ['players', {referencePanel: 'setup', direction: 'below'}],
      ['session', {referencePanel: 'players', direction: 'within'}],
    ]);
    expect(arrangement.columns).toEqual([
      {panel: 'gameplay', width: 840},
      {panel: 'actions', width: 360},
    ]);
    expect(arrangement.stacks).toEqual([['actions', 'setup', 'players']]);
  });

  it('makes every panel a tab of Gameplay when narrow', () => {
    const arrangement = arrange(PANELS, ALL, wide(800));
    expect(arrangement.tier).toBe('tabs');
    const tab = {referencePanel: 'gameplay', direction: 'within'};
    expect(arrangement.panels.map(({position}) => position)).toEqual([
      undefined,
      ...Array.from({length: 6}, (): unknown => tab),
    ]);
    expect(arrangement.panels.map(({inactive}) => inactive)).toEqual([
      false,
      ...Array.from({length: 6}, (): boolean => true),
    ]);
    for (const options of arrangement.panels) {
      expect(options).not.toHaveProperty('minimumWidth');
      expect(options).not.toHaveProperty('minimumHeight');
    }
    expect(arrangement.columns).toEqual([{panel: 'gameplay', width: 800}]);
    expect(arrangement.stacks).toEqual([]);
  });
});

/** A primary board of `minimumWidth` and panels positioned by it. */
function board(
  minimumWidth: number,
  ...others: Array<{
    id: string;
    direction: 'right' | 'below';
    minimumWidth: number;
  }>
): Panels {
  return definePanels([
    {
      id: 'board',
      title: 'Board',
      primary: true,
      minimumWidth,
      minimumHeight: 200,
    },
    ...others.map(({id, direction, minimumWidth: width}) => ({
      id,
      title: id,
      position: {referencePanel: 'board', direction},
      minimumWidth: width,
      minimumHeight: 200,
    })),
  ]);
}

describe('arrange, in an unmeasured dock', () => {
  it('places every panel but sizes no group', () => {
    const arrangement = arrange(PANELS, ALL, {width: 0, height: 0});
    expect(arrangement.tier).toBe('three');
    expect(arrangement.panels).toHaveLength(ALL.size);
    expect(arrangement.columns).toEqual([]);
    expect(arrangement.stacks).toEqual([]);
  });

  it('sizes columns but no stack before the dock has a height', () => {
    const arrangement = arrange(PANELS, ALL, {width: 1200, height: 0});
    expect(arrangement.heightScale).toBe(1);
    expect(arrangement.columns).toHaveLength(2);
    expect(arrangement.stacks).toEqual([]);
  });
});

describe('arrange, with minimums', () => {
  it('scales minimum widths that do not fit the dock', () => {
    const broad = board(600, {
      id: 'notes',
      direction: 'right',
      minimumWidth: 500,
    });
    const {panels, columns, widthScale} = arrange(
      broad,
      new Set(['board', 'notes']),
      wide(1000),
    );
    expect(widthScale).toBeCloseTo(1000 / 1100);
    expect(panels.map(({minimumWidth}) => minimumWidth)).toEqual([545, 454]);
    expect(columns).toEqual([
      {panel: 'board', width: 546},
      {panel: 'notes', width: 454},
    ]);
  });

  it('narrows side columns that would leave the first below its minimum', () => {
    const asking = definePanels([
      {
        id: 'board',
        title: 'Board',
        primary: true,
        minimumWidth: 600,
        minimumHeight: 200,
      },
      ...['notes', 'log'].map(id => ({
        id,
        title: id,
        position: {referencePanel: 'board', direction: 'right'} as const,
        minimumWidth: 100,
        minimumHeight: 200,
        initialWidth: 500,
      })),
    ]);
    const all = new Set(asking.list.map(({id}) => id));
    expect(arrange(asking, all, wide(1000)).columns).toEqual([
      {panel: 'board', width: 600},
      {panel: 'log', width: 200},
      {panel: 'notes', width: 200},
    ]);
  });

  it('refuses a column beside a column of stacked groups', () => {
    const stacked = board(
      300,
      {id: 'below', direction: 'below', minimumWidth: 300},
      {id: 'beside', direction: 'right', minimumWidth: 300},
    );
    const all = new Set(stacked.list.map(({id}) => id));
    expect(() => arrange(stacked, all, wide(1800))).toThrow(
      'Panel "beside" opens beside a panel whose column has stacked groups.',
    );
  });
});

describe('arrange, in a short dock', () => {
  it('scales minimum heights so every column fits the height', () => {
    const arrangement = arrange(PANELS, ALL, {width: 1200, height: 500});
    expect(arrangement.heightScale).toBeCloseTo(500 / 600);
    expect(
      arrangement.panels.map(({id, minimumHeight}) => [id, minimumHeight]),
    ).toEqual([
      ['gameplay', 200],
      ['actions', 166],
      ['state', 166],
      ['setup', 166],
      ['view', 166],
      ['players', 166],
      ['session', 166],
    ]);
  });

  it('fits the column that needs the most height', () => {
    const tall = board(
      300,
      {id: 'beside', direction: 'right', minimumWidth: 300},
      {id: 'below', direction: 'below', minimumWidth: 300},
    );
    const all = new Set(tall.list.map(({id}) => id));
    const {heightScale} = arrange(tall, all, {width: 1200, height: 300});
    expect(heightScale).toBe(0.75);
  });
});
