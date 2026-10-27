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
import {afterEach, describe, expect, it} from 'vitest';

import type {CatalogueEntry} from '../contract/contract.ts';
import {type Rendered, button, render} from '../testing/render.tsx';
import {CataloguePage} from './catalogue_page.tsx';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

function entry(
  name: string,
  fields: Partial<CatalogueEntry> = {},
): CatalogueEntry {
  return {
    name,
    long_name: name,
    dynamics: 'sequential',
    chance_mode: 'deterministic',
    information: 'perfect_information',
    min_seats: 2,
    max_seats: 2,
    num_players: 2,
    parameters: [],
    loadable: true,
    unavailable_reason: null,
    controllers: ['human'],
    renderer: null,
    ...fields,
  };
}

// Chess and euchre have renderers in the client's registry; go has none.
const GAMES = [
  entry('chess', {renderer: 'chess'}),
  entry('efg_game', {
    num_players: null,
    loadable: false,
    unavailable_reason: 'The game needs parameters to load.',
    controllers: [],
  }),
  entry('euchre', {renderer: 'euchre'}),
  entry('go'),
];

function rows(container: HTMLElement): Array<[string, string | null]> {
  return [...container.querySelectorAll('.catalogue__game')].map(row => [
    row.querySelector('.catalogue__name')?.textContent ?? '',
    row.querySelector('.catalogue__badge')?.textContent ?? null,
  ]);
}

describe('CataloguePage', () => {
  it('badges every game Board UI, Text only or Unavailable', () => {
    rendered = render(
      <CataloguePage catalogue={{contract_version: 7, games: GAMES}} />,
    );
    expect(rows(rendered.container)).toEqual([
      ['chess', 'Board UI'],
      ['efg_game', 'Unavailable'],
      ['euchre', 'Board UI'],
      ['go', 'Text only'],
    ]);
  });

  it('filters to the board games or the text games, keeping the count', () => {
    rendered = render(
      <CataloguePage catalogue={{contract_version: 7, games: GAMES}} />,
    );
    const {container} = rendered;
    const status = (): string | null =>
      container.querySelector('.catalogue__count')?.textContent ?? null;
    expect(status()).toBe('4 of 4 games');
    // Every game is under exactly one view, so the counts add up.
    const counts = [...container.querySelectorAll('button')]
      .map(b => /\((\d+)\)$/.exec(b.textContent)?.[1])
      .filter((n): n is string => n !== undefined)
      .map(Number);
    expect(counts).toEqual([2, 1, 1]);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(GAMES.length);
    act(() => {
      button('Board UI (2)', container).click();
    });
    expect(rows(container).map(([name]) => name)).toEqual(['chess', 'euchre']);
    expect(status()).toBe('2 of 4 games');
    act(() => {
      button('Text only (1)', container).click();
    });
    expect(rows(container).map(([name]) => name)).toEqual(['go']);
    act(() => {
      button('Unavailable (1)', container).click();
    });
    expect(rows(container).map(([name]) => name)).toEqual(['efg_game']);
    expect(status()).toBe('1 of 4 games');
    act(() => {
      button('All', container).click();
    });
    expect(rows(container)).toHaveLength(4);
  });
});
