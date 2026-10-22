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

import {arrange} from '../dock/arrangement.ts';
import {GAME_PANELS} from './game_panels.ts';

const INITIAL = GAME_PANELS.initial;

describe('GAME_PANELS', () => {
  it('starts with Gameplay, the primary panel', () => {
    expect(GAME_PANELS.list[0]).toBe(GAME_PANELS.primary);
    expect(GAME_PANELS.primary.id).toBe('gameplay');
  });

  it('lists the panels in tab order', () => {
    expect(GAME_PANELS.list.map(({title}) => title)).toEqual([
      'Gameplay',
      'Actions',
      'State',
      'Session',
      'Setup',
      'View',
    ]);
  });

  it('opens every panel but Session at first', () => {
    expect([...INITIAL]).toEqual([
      'gameplay',
      'actions',
      'state',
      'setup',
      'view',
    ]);
  });

  it('opens Session as a tab of Actions', () => {
    const all = new Set(GAME_PANELS.list.map(({id}) => id));
    const arrangement = arrange(GAME_PANELS, all, {width: 1800, height: 900});
    const session = arrangement.panels.find(({id}) => id === 'session');
    expect(session?.position).toEqual({
      referencePanel: 'actions',
      direction: 'within',
    });
  });

  it.each([1800, 1200])(
    'puts one column of two groups beside Gameplay at %i px',
    width => {
      const arrangement = arrange(GAME_PANELS, INITIAL, {width, height: 900});
      expect(arrangement.columns.map(({panel}) => panel)).toEqual([
        'gameplay',
        'actions',
      ]);
      expect(arrangement.stacks).toEqual([['actions', 'setup']]);
    },
  );

  it('makes every panel a tab of one group when narrow', () => {
    const arrangement = arrange(GAME_PANELS, INITIAL, {
      width: 800,
      height: 900,
    });
    expect(arrangement.columns).toHaveLength(1);
    expect(arrangement.stacks).toEqual([]);
  });
});
