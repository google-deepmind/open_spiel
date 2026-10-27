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
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import {only, type Rendered, render} from '../../testing/render.tsx';
import {TableCardSettingsFields} from './table_card_settings_fields.tsx';

let rendered: Rendered | null = null;

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  rendered?.unmount();
  rendered = null;
  localStorage.clear();
});

function selectLabelled(
  container: HTMLElement,
  name: string,
): HTMLSelectElement {
  const label = [...container.querySelectorAll('label')].find(
    l => l.querySelector('span')?.textContent === name,
  );
  const select = label?.querySelector('select');
  if (!select) throw new Error(`no select labelled ${name}`);
  return select;
}

function options(select: HTMLSelectElement): string[][] {
  return [...select.options].map(o => [o.value, o.textContent]);
}

describe('TableCardSettingsFields card picker', () => {
  it('names each deck by what it is for, and shows its "best for" line', () => {
    rendered = render(<TableCardSettingsFields sections={['cards']} />);
    const deck = selectLabelled(rendered.container, 'Card deck');
    expect(options(deck)).toEqual([
      ['simple', 'Centred rank and suit'],
      ['simple-corner', 'Large corner index'],
      ['basic', 'Corner index + large pip'],
    ]);
    expect(deck.value).toBe('simple');
    expect(
      only(rendered.container, '[data-testid="deck-description"]').textContent,
    ).toMatch(/^Best for cards seen one at a time/);

    act(() => {
      deck.value = 'simple-corner';
      deck.dispatchEvent(new Event('change', {bubbles: true}));
    });
    expect(
      only(rendered.container, '[data-testid="deck-description"]').textContent,
    ).toMatch(/^Best for fanned hands/);
  });

  it('names each card back, after the deck default', () => {
    rendered = render(<TableCardSettingsFields sections={['cards']} />);
    expect(options(selectLabelled(rendered.container, 'Card back'))).toEqual([
      ['default', 'Deck default'],
      ['blue', 'Blue lattice'],
      ['red', 'Red lattice'],
      ['diamond-red', 'Red diamonds'],
      ['diamond-black', 'Black diamonds'],
      ['stripe-navy', 'Navy stripes'],
    ]);
  });
});
