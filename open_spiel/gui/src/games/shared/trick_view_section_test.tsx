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

import {afterEach, describe, expect, it} from 'vitest';

import {DEFAULT_CARD_STYLE, getDeck} from '../../cards/library/index.ts';
import {DEFAULT_TABLE_CARDS} from '../../cards/table_cards/index.ts';
import {InGame} from '../../testing/game.tsx';
import {type Rendered, only, render} from '../../testing/render.tsx';
import {EuchreGameplayConfigExtras} from '../euchre/euchre_gameplay_config_extras.tsx';
import {HeartsGameplayConfigExtras} from '../hearts/hearts_gameplay_config_extras.tsx';
import {CARD_STYLES, DEFAULT_CARD_STYLE_ID} from './trick_view_section.tsx';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

/**
 * What a View section shows: each field's label, and a select's options
 * and value.
 */
function outline(section: HTMLElement): unknown[] {
  return [...section.querySelectorAll('label')].map(element => {
    const name = element.querySelector('span')?.textContent ?? '';
    const select = element.querySelector('select');
    const input = element.querySelector('input');
    return select !== null
      ? [name, [...select.options].map(o => o.value), select.value]
      : [name, input?.type, input?.checked];
  });
}

function sectionOf(game: 'euchre' | 'hearts'): HTMLElement {
  rendered = render(
    <InGame viewer={{kind: 'oracle', seat: null}}>
      {game === 'euchre' ? (
        <EuchreGameplayConfigExtras />
      ) : (
        <HeartsGameplayConfigExtras />
      )}
    </InGame>,
  );
  return only(rendered.container, 'section');
}

describe('the euchre and hearts View sections', () => {
  it('show the same fields, choices, order and defaults', () => {
    const euchre = sectionOf('euchre');
    expect(euchre.getAttribute('aria-label')).toBe('Euchre');
    // A card like the Perspective tab's, with no heading repeating the tab.
    expect(euchre.classList.contains('view-card')).toBe(true);
    expect(euchre.querySelectorAll('h1, h2, h3, h4')).toHaveLength(0);
    const euchreOutline = outline(euchre);
    rendered?.unmount();
    const hearts = sectionOf('hearts');
    expect(hearts.getAttribute('aria-label')).toBe('Hearts');
    expect(hearts.className).toBe(euchre.className);
    expect(outline(hearts)).toEqual(euchreOutline);
    expect(euchreOutline.map(field => (field as string[])[0])).toEqual([
      'Card style',
      'Trick hold',
      'Show all cards',
    ]);
  });

  it('have no fields beyond Card style, Trick hold and Show all cards', () => {
    const text = sectionOf('hearts').textContent;
    for (const gone of [
      'Hand shape',
      'Trick layout',
      'Card deck',
      'Card back',
      'Suit symbols',
      'This game only',
      'Table',
      'Custom colours',
    ]) {
      expect(text).not.toContain(gone);
    }
  });
});

describe('the default card face', () => {
  it('is corner index + large pip everywhere', () => {
    expect(DEFAULT_CARD_STYLE_ID).toBe('basic');
    expect(CARD_STYLES[0]).toBe('basic');
    expect(DEFAULT_CARD_STYLE.deck).toBe('basic');
    expect(DEFAULT_TABLE_CARDS.cards.deck).toBe('basic');
    expect(getDeck('basic').label).toBe('Corner index + large pip');
    for (const game of ['euchre', 'hearts'] as const) {
      const select = only(sectionOf(game), '[data-testid="card-style-select"]');
      expect((select as HTMLSelectElement).value).toBe('basic');
      rendered?.unmount();
      rendered = null;
    }
  });
});
