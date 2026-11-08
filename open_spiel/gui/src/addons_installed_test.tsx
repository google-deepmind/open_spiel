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

// The check of the user's own add-ons, without a browser
// (`npm run test:addons`): loads every module in addons/, as the entry point
// does, and draws each card deck an add-on registers, its faces and every
// back it offers. With no add-ons it checks only that there are none to load.

import type {ReactNode} from 'react';
import {describe, expect, it} from 'vitest';

import {loadAddOns} from './addons.ts';
import './cards/library/decks/index.ts';
import {listDecks} from './cards/library/registry.ts';
import type {Card, Deck} from './cards/library/types.ts';
import {render} from './testing/render.tsx';

const MODULES = import.meta.glob('../addons/*.{ts,tsx}', {eager: true});
const BEFORE = new Set(listDecks().map(deck => deck.id));
loadAddOns(MODULES);
const ADDED: Deck[] = listDecks().filter(deck => !BEFORE.has(deck.id));

const CARDS: readonly Card[] = [
  {rank: 'A', suit: 'S'},
  {rank: '10', suit: 'H'},
  {rank: 'Q', suit: 'D'},
  {rank: '7', suit: 'C'},
];

/** Whether `node` draws an element. */
function draws(node: ReactNode): boolean {
  const rendered = render(<>{node}</>);
  const drawn = rendered.container.firstElementChild !== null;
  rendered.unmount();
  return drawn;
}

describe('the add-ons in addons/', () => {
  // loadAddOns ran above, as the file loaded: an add-on that fails to
  // register fails this file, naming the module.
  it(`load: ${Object.keys(MODULES).sort().join(', ') || 'none'}`, () => {
    expect(BEFORE.size).toBeGreaterThan(0);
  });

  for (const deck of ADDED) {
    const backs = deck.backs ?? [];
    it(`deck ${deck.id} ("${deck.label}") draws its faces and the backs: default${backs.map(back => `, ${back}`).join('')}`, () => {
      for (const card of CARDS) {
        expect(draws(deck.renderFace(card, {scheme: 'two'}))).toBe(true);
      }
      expect(draws(deck.renderBack({}))).toBe(true);
      for (const back of backs) {
        expect(draws(deck.renderBack({variant: back}))).toBe(true);
      }
    });
  }
});
