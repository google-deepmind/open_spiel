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

import type {CatalogueEntry} from '../contract/contract.ts';
import {filterGames, gameFacts} from './games.ts';

function entry(
  name: string,
  longName: string,
  fields: Partial<CatalogueEntry> = {},
): CatalogueEntry {
  return {
    name,
    long_name: longName,
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
    ...fields,
  };
}

const GAMES = [
  entry('chess', 'Chess'),
  entry('kuhn_poker', 'Kuhn Poker'),
  entry('tic_tac_toe', 'Tic Tac Toe'),
];

describe('filterGames', () => {
  it('keeps every game for an empty query', () => {
    expect(filterGames(GAMES, '  ')).toEqual(GAMES);
  });

  it('matches the short or the display name, ignoring case', () => {
    expect(filterGames(GAMES, 'POKER').map(({name}) => name)).toEqual([
      'kuhn_poker',
    ]);
    expect(filterGames(GAMES, 'tac t').map(({name}) => name)).toEqual([
      'tic_tac_toe',
    ]);
  });

  it('finds nothing for a query no name holds', () => {
    expect(filterGames(GAMES, 'go')).toEqual([]);
  });
});

describe('gameFacts', () => {
  it('names a fixed number of players', () => {
    expect(gameFacts(entry('chess', 'Chess'))).toBe(
      '2 players, perfect information',
    );
    expect(
      gameFacts(entry('catch', 'Catch', {min_seats: 1, max_seats: 1})),
    ).toBe('1 player, perfect information');
  });

  it('names a range of players and the information', () => {
    expect(
      gameFacts(
        entry('hearts', 'Hearts', {
          min_seats: 3,
          max_seats: 6,
          information: 'imperfect_information',
        }),
      ),
    ).toBe('3 to 6 players, imperfect information');
  });
});
