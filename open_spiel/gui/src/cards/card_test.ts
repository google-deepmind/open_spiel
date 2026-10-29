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

import type {Presentation} from '../contract/contract.ts';
import {cardCode, parseCard, parseShown} from './card.ts';
import {actionForCard, cardActions} from './card_hints.ts';
import {registerSkin, skinFor, STANDARD_SKIN} from './skins.ts';

function presentation(hints: Array<[number, unknown]>): Presentation {
  return {
    renderer: 'cards',
    view: {},
    actions: hints.map(([id, hint]) => ({id, hint})),
  };
}

describe('parseCard', () => {
  it('reads rank then suit, in either case, with 10 as T', () => {
    expect(parseCard('As')).toEqual({rank: 'A', suit: 's'});
    expect(parseCard('10H')).toEqual({rank: 'T', suit: 'h'});
    expect(cardCode(parseCard('td'))).toBe('Td');
    expect(parseShown(null)).toBeNull();
  });

  it('refuses a code that names no card', () => {
    expect(() => parseCard('1s')).toThrow('"1s" is not a card.');
    expect(() => parseCard('Ax')).toThrow('not a card');
  });
});

describe('skins', () => {
  it('names and colours a card', () => {
    const ace = parseCard('Ah');
    expect(STANDARD_SKIN.cardName(ace)).toBe('Ace of hearts');
    expect(STANDARD_SKIN.suitGlyph('h')).toBe('♥');
    expect(STANDARD_SKIN.suitTone('h')).toBe('red');
    expect(STANDARD_SKIN.rankLabel('T')).toBe('10');
  });

  it('looks skins up by name, falling back to the standard one', () => {
    expect(skinFor('four_colour').suitTone('d')).toBe('blue');
    expect(skinFor('no such skin')).toBe(STANDARD_SKIN);
    expect(skinFor(null)).toBe(STANDARD_SKIN);
    expect(() => {
      registerSkin(STANDARD_SKIN);
    }).toThrow('already registered');
  });
});

describe('cardActions', () => {
  it('maps each hinted card to its action, skipping other hints', () => {
    const actions = cardActions(
      presentation([
        [3, {card: 'as'}],
        [5, {button: 'Knock'}],
        [7, null],
      ]),
    );
    expect([...actions]).toEqual([['As', 3]]);
    expect(actionForCard(actions, 'AS')).toBe(3);
    expect(actionForCard(actions, null)).toBeUndefined();
  });

  it('refuses a card two actions play, and a hint that is not a card', () => {
    expect(() =>
      cardActions(
        presentation([
          [1, {card: 'As'}],
          [2, {card: 'As'}],
        ]),
      ),
    ).toThrow('Two actions play the card As.');
    expect(() => cardActions(presentation([[1, {card: 'Zz'}]]))).toThrow(
      'not a card',
    );
  });
});
