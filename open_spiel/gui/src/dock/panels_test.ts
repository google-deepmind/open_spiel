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

import {definePanels, type PanelDefinition} from './panels.ts';

const gameplay: PanelDefinition = {
  id: 'gameplay',
  title: 'Gameplay',
  primary: true,
  minimumWidth: 300,
  minimumHeight: 200,
};
const unplaced: PanelDefinition = {
  id: 'unplaced',
  title: 'Unplaced',
  minimumWidth: 200,
  minimumHeight: 200,
};
const side: PanelDefinition = {
  id: 'side',
  title: 'Side',
  position: {referencePanel: 'gameplay', direction: 'right'},
  minimumWidth: 200,
  minimumHeight: 200,
};

describe('definePanels', () => {
  it('keeps the definitions in order and finds each by id', () => {
    const panels = definePanels([gameplay, side]);
    expect(panels.list).toEqual([gameplay, side]);
    expect(panels.primary).toBe(gameplay);
    expect(panels.get('side')).toBe(side);
    expect(() => panels.get('missing')).toThrow('There is no panel "missing".');
  });

  it.each<[string, PanelDefinition[], string]>([
    [
      'a duplicate id',
      [gameplay, side, side],
      'Panel "side" is defined twice.',
    ],
    [
      'a blank title',
      [gameplay, {...side, title: ' '}],
      'Panel "side" has a blank title.',
    ],
    [
      'a reference to an unknown panel',
      [
        gameplay,
        {...side, position: {referencePanel: 'nowhere', direction: 'right'}},
      ],
      'Panel "side" is positioned by "nowhere", which is not defined before it.',
    ],
    [
      'a reference to a panel defined later',
      [side, gameplay],
      'Panel "side" is positioned by "gameplay", which is not defined before it.',
    ],
    // The first panel can be positioned by none, so it must be primary.
    ['no primary panel', [unplaced], 'Panel "unplaced" has no position.'],
    [
      'two primary panels',
      [gameplay, {...unplaced, primary: true}],
      'Exactly one panel must be primary, not 2.',
    ],
    [
      'a positioned primary panel',
      [
        {
          ...gameplay,
          position: {referencePanel: 'gameplay', direction: 'right'},
        },
      ],
      'The primary panel "gameplay" has a position.',
    ],
    [
      'a panel without a position',
      [gameplay, unplaced],
      'Panel "unplaced" has no position.',
    ],
    [
      'a minimum width of zero',
      [gameplay, {...side, minimumWidth: 0}],
      'Panel "side" has minimumWidth 0.',
    ],
    [
      'a minimum height that is not a number',
      [gameplay, {...side, minimumHeight: Number.NaN}],
      'Panel "side" has minimumHeight NaN.',
    ],
    [
      'an initial width below the minimum',
      [gameplay, {...side, initialWidth: 100}],
      'Panel "side" has an initialWidth below its minimumWidth.',
    ],
    ['no panels', [], 'Exactly one panel must be primary, not 0.'],
  ])('refuses %s', (_, definitions, message) => {
    expect(() => definePanels(definitions)).toThrow(message);
  });
});
