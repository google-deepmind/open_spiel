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
import {RENDERERS} from '../renderers/registry.ts';
import {defineRenderers} from '../renderers/renderer.ts';
import {filterGames, gameFacts, gameView} from './games.ts';

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
    renderer: null,
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

describe('gameView', () => {
  const drawn = defineRenderers([{name: 'chess', render: () => <></>}]);

  it('is a board when the client has the renderer the server names', () => {
    expect(gameView(entry('chess', 'Chess', {renderer: 'chess'}), drawn)).toBe(
      'board',
    );
  });

  it('is text when the server names no renderer, or one the client lacks', () => {
    expect(gameView(entry('go', 'Go'), drawn)).toBe('text');
    expect(
      gameView(entry('chess960', 'Chess960', {renderer: 'chess960'}), drawn),
    ).toBe('text');
  });

  it('is unavailable for a game that cannot be played here', () => {
    const unloadable = entry('efg_game', 'EFG', {
      num_players: null,
      loadable: false,
      unavailable_reason: 'No.',
      controllers: [],
    });
    expect(gameView(unloadable, drawn)).toBe('unavailable');
  });

  it("reads the client's own registry by default", () => {
    for (const name of RENDERERS.names) {
      expect(gameView(entry(name, name, {renderer: name}))).toBe('board');
    }
  });
});

describe('filterGames by view', () => {
  const drawn = defineRenderers([{name: 'chess', render: () => <></>}]);
  const games = [
    entry('chess', 'Chess', {renderer: 'chess'}),
    entry('go', 'Go'),
    entry('efg_game', 'EFG', {
      num_players: null,
      loadable: false,
      unavailable_reason: 'No.',
      controllers: [],
    }),
  ];

  it('keeps the board, text or unavailable games, or every game', () => {
    const names = (view: 'all' | 'board' | 'text' | 'unavailable'): string[] =>
      filterGames(games, '', view, drawn).map(({name}) => name);
    expect(names('board')).toEqual(['chess']);
    expect(names('text')).toEqual(['go']);
    expect(names('unavailable')).toEqual(['efg_game']);
    expect(names('all')).toEqual(['chess', 'go', 'efg_game']);
  });

  it('narrows by name and view together', () => {
    expect(filterGames(games, 'go', 'board', drawn)).toEqual([]);
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
