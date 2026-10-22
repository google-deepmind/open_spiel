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

import {gamePath, routeOf, sessionPath, viewerOf} from './route.ts';

function at(path: string): URL {
  return new URL(path, 'http://localhost');
}

describe('routeOf', () => {
  it('shows the catalogue at the root', () => {
    expect(routeOf(at('/'))).toEqual({page: 'catalogue'});
  });

  it('shows a game, with the session and viewer its address names', () => {
    expect(routeOf(at('/game/kuhn_poker?session=s1&viewer=1'))).toEqual({
      page: 'game',
      game: 'kuhn_poker',
      session: 's1',
      viewer: {kind: 'seat', seat: 1},
    });
  });

  it('starts a new session for the default viewer when none is named', () => {
    expect(routeOf(at('/game/go/'))).toEqual({
      page: 'game',
      game: 'go',
      session: null,
      viewer: null,
    });
  });

  it('decodes the game name', () => {
    const route = routeOf(at(gamePath('a b/c')));
    expect(route).toMatchObject({page: 'game', game: 'a b/c'});
  });

  it.each([
    '/games',
    '/game',
    '/game/go/extra',
    '/index.html',
    '/game/%',
    '/game/%E0%A4%A',
  ])('finds nothing at %s', path => {
    expect(routeOf(at(path))).toEqual({page: 'missing', path});
  });
});

describe('viewerOf', () => {
  it.each([
    ['oracle', {kind: 'oracle', seat: null}],
    ['spectator', {kind: 'spectator', seat: null}],
    ['0', {kind: 'seat', seat: 0}],
    ['12', {kind: 'seat', seat: 12}],
  ])('reads %s', (parameter, viewer) => {
    expect(viewerOf(parameter)).toEqual(viewer);
  });

  it.each([null, '', 'seat', '-1', '01', '1.5', '99999'])(
    'reads no viewer from %s',
    parameter => {
      expect(viewerOf(parameter)).toBeNull();
    },
  );
});

describe('sessionPath', () => {
  it('names the session and the viewer, and reads back', () => {
    const path = sessionPath('tic_tac_toe', 'abc', {kind: 'seat', seat: 1});
    expect(path).toBe('/game/tic_tac_toe?session=abc&viewer=1');
    expect(routeOf(at(path))).toEqual({
      page: 'game',
      game: 'tic_tac_toe',
      session: 'abc',
      viewer: {kind: 'seat', seat: 1},
    });
  });
});
