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

import type {JSX} from 'react';
import {describe, expect, it, vi} from 'vitest';

import {ADD_ON_API, type AddOnApi, loadAddOns} from './addons.ts';
import './cards/library/decks/index.ts';
import {
  getDeck,
  listDecks,
  unregisterDeckForTest,
} from './cards/library/registry.ts';
import {render} from './testing/render.tsx';

function fakeApi(): AddOnApi {
  return {
    replaceRenderer: vi.fn(),
    chessRenderer: vi.fn(),
    registerPieceSet: vi.fn(),
    registerDeck: vi.fn(),
    getDeck: vi.fn(),
  };
}

describe('loadAddOns', () => {
  it('calls each registerInto with the api, in path order', () => {
    const api = fakeApi();
    const calls: string[] = [];
    loadAddOns(
      {
        '../addons/b.ts': {registerInto: () => calls.push('b')},
        '../addons/a.tsx': {
          registerInto: (given: AddOnApi) => {
            expect(given).toBe(api);
            calls.push('a');
          },
        },
      },
      api,
    );
    expect(calls).toEqual(['a', 'b']);
  });

  it('loads nothing when there are no add-ons', () => {
    const api = fakeApi();
    loadAddOns({}, api);
    expect(api.replaceRenderer).not.toHaveBeenCalled();
  });

  it('refuses a module without registerInto, naming it', () => {
    expect(() => {
      loadAddOns({'../addons/x.ts': {}}, fakeApi());
    }).toThrow('Add-on ../addons/x.ts exports no registerInto function.');
  });

  it('names the add-on whose registerInto throws', () => {
    const failing = {
      registerInto: () => {
        throw new Error('taken');
      },
    };
    expect(() => {
      loadAddOns({'../addons/y.ts': failing}, fakeApi());
    }).toThrow('Add-on ../addons/y.ts failed to register.');
  });

  it('gives add-ons the client extension points', () => {
    expect(Object.keys(ADD_ON_API).sort()).toEqual([
      'chessRenderer',
      'getDeck',
      'registerDeck',
      'registerPieceSet',
      'replaceRenderer',
    ]);
  });
});

describe('the deck recipe', () => {
  it('registers a deck with one more back, as the deck recipe does', () => {
    // src/INTERFACES.md, "Adding a card deck or back".
    function GreenBack(): JSX.Element {
      return <svg data-back="green" />;
    }
    loadAddOns({
      '../addons/green_back.tsx': {
        registerInto: (api: AddOnApi) => {
          const basic = api.getDeck('basic');
          api.registerDeck({
            ...basic,
            id: 'basic_green',
            label: 'Corner index + large pip, green back',
            provenance: 'Drawn in code for this test',
            backs: [...(basic.backs ?? []), 'green'],
            renderBack: opts =>
              opts.variant === 'green' ? <GreenBack /> : basic.renderBack(opts),
          });
        },
      },
    });
    try {
      expect(listDecks().map(deck => deck.label)).toContain(
        'Corner index + large pip, green back',
      );
      const deck = getDeck('basic_green');
      expect(deck.backs).toEqual([...(getDeck('basic').backs ?? []), 'green']);
      expect(deck.backs).toContain('blue');
      expect(deck.backs).toContain('red');
      const green = render(<>{deck.renderBack({variant: 'green'})}</>);
      expect(green.container.querySelector('[data-back="green"]')).not.toBe(
        null,
      );
      green.unmount();
      for (const variant of [undefined, 'blue', 'red']) {
        const plain = render(<>{deck.renderBack({variant})}</>);
        expect(plain.container.querySelector('svg')).not.toBe(null);
        expect(plain.container.querySelector('[data-back="green"]')).toBe(null);
        plain.unmount();
      }
    } finally {
      unregisterDeckForTest('basic_green');
    }
  });
});
