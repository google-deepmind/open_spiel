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

// The built-in decks: their picker names and "best for" lines, and the card
// backs they share, which are drawn in code.
import {renderToStaticMarkup} from 'react-dom/server';
import {describe, expect, it} from 'vitest';

import {getDeck, listDecks} from '../index.ts';
import {CARD_BACK_LABELS, CARD_BACKS, DEFAULT_CARD_BACK} from './backs.tsx';

const BUILT_IN = ['simple', 'simple-corner', 'basic'];

function backMarkup(deckId: string, variant?: string): string {
  return renderToStaticMarkup(<>{getDeck(deckId).renderBack({variant})}</>);
}

describe('built-in decks', () => {
  it('keep their ids and name each style by what it is for', () => {
    expect(listDecks().map(d => d.id)).toEqual(BUILT_IN);
    expect(Object.fromEntries(listDecks().map(d => [d.id, d.label]))).toEqual({
      simple: 'Centred rank and suit',
      'simple-corner': 'Large corner index',
      basic: 'Corner index + large pip',
    });
  });

  it('give each style one "best for" line', () => {
    for (const deck of listDecks()) {
      expect(deck.description, deck.id).toMatch(/^Best for [^\n]+\.$/);
    }
    expect(getDeck('simple-corner').description).toContain('fanned hands');
    expect(getDeck('basic').description).toContain('table piles and tricks');
    expect(getDeck('simple').description).toContain('not for fans');
  });

  it('claim the tighter fan only for the deck whose index box is smaller', () => {
    // A fan's step comes from the index box alone (hand_measure.ts), so a
    // "best for" line may not credit the big corner index with tight fans.
    const {index: basic} = getDeck('basic');
    const {index: corner} = getDeck('simple-corner');
    expect(basic.width).toBeLessThan(corner.width);
    expect(basic.height).toBeLessThan(corner.height);
    expect(getDeck('simple-corner').description).not.toMatch(/tight/);
    expect(getDeck('simple-corner').description).not.toMatch(/nothing in/);
  });
});

describe('card backs', () => {
  it('are the same five on every built-in deck, each with a name', () => {
    expect(CARD_BACKS).toEqual([
      'blue',
      'red',
      'diamond-red',
      'diamond-black',
      'stripe-navy',
    ]);
    for (const id of BUILT_IN) {
      const deck = getDeck(id);
      expect(deck.backs, id).toEqual(CARD_BACKS);
      expect(deck.backLabels, id).toEqual(CARD_BACK_LABELS);
    }
    expect(Object.values(CARD_BACK_LABELS)).toEqual([
      'Blue lattice',
      'Red lattice',
      'Red diamonds',
      'Black diamonds',
      'Navy stripes',
    ]);
  });

  it('default to the blue lattice', () => {
    expect(DEFAULT_CARD_BACK).toBe('blue');
    for (const id of BUILT_IN) {
      const markup = backMarkup(id);
      expect(markup, id).toContain('data-back="blue"');
      expect(markup, id).toMatch(/<pattern id="card-back-blue-[\w-]+"/);
      expect(markup, id).toMatch(/fill="url\(#card-back-blue-[\w-]+\)"/);
    }
  });

  it('draw every back as a patterned field, never a flat fill', () => {
    for (const back of CARD_BACKS) {
      const markup = backMarkup('basic', back);
      expect(markup, back).toContain(`data-back="${back}"`);
      const id = new RegExp(
        `<(?:pattern|linearGradient) id="(card-back-${back}-[\\w-]+)"`,
      ).exec(markup)?.[1];
      expect(id, back).toBeDefined();
      expect(markup, back).toContain(`fill="url(#${String(id)})"`);
    }
  });

  it('are drawn in code: no image, HTML or external reference', () => {
    for (const back of CARD_BACKS) {
      const markup = backMarkup('simple', back);
      expect(markup, back).not.toMatch(/<image\b|<foreignObject\b/i);
      expect(markup, back).not.toMatch(/href=|data:/);
      expect(markup, back).not.toMatch(/url\((?!#card-back-)/);
    }
  });

  it('give each card its own pattern id', () => {
    const markup = renderToStaticMarkup(
      <>
        {getDeck('basic').renderBack({})}
        {getDeck('basic').renderBack({})}
      </>,
    );
    const ids = [...markup.matchAll(/<pattern id="([^"]+)"/g)].map(m => m[1]);
    expect(ids).toHaveLength(2);
    expect(new Set(ids).size).toBe(2);
  });

  it('refuse an unknown back, naming the deck and its backs', () => {
    expect(() => backMarkup('simple', 'crosshatch')).toThrow(
      "simple deck has no back variant 'crosshatch'; have: blue, red, diamond-red, diamond-black, stripe-navy",
    );
  });
});

/** The numbers `re`'s groups capture in `markup`; throws when it does not match. */
function captured(re: RegExp, markup: string, what: string): number[] {
  const m = re.exec(markup);
  if (m === null) throw new Error(`${what}: no match for ${String(re)}`);
  return m.slice(1).map(Number);
}

describe('card back geometry', () => {
  it('paints the stripes as one repeating gradient, with no tile seams', () => {
    const markup = backMarkup('basic', 'stripe-navy');
    expect(markup).not.toContain('<pattern');
    expect(markup).not.toContain('patternTransform');
    expect(markup).toMatch(/<linearGradient [^>]*spreadMethod="repeat"/);
  });

  it('fills the diamond field with whole tiles from its corner, unframed', () => {
    for (const back of ['diamond-red', 'diamond-black']) {
      const markup = backMarkup('basic', back);
      const [px = NaN, py = NaN, tw = NaN, th = NaN] = captured(
        /<pattern id="[^"]+" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"/,
        markup,
        back,
      );
      const [fw = NaN, fh = NaN] = captured(
        new RegExp(
          `<rect x="${String(px)}" y="${String(py)}" width="([\\d.]+)" height="([\\d.]+)" fill="url`,
        ),
        markup,
        back,
      );
      expect(Number.isInteger(fw / tw), back).toBe(true);
      expect(Number.isInteger(fh / th), back).toBe(true);
      // Centred in the 250x350 card, and no stroked frame line.
      expect(px * 2 + fw, back).toBe(250);
      expect(py * 2 + fh, back).toBe(350);
      expect(markup, back).not.toContain('stroke');
    }
  });
});
