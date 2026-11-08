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
import {
  defaultSession,
  defaultViewer,
  initialViewer,
  modeLabel,
} from './game_defaults.ts';

function entry(fields: Partial<CatalogueEntry> = {}): CatalogueEntry {
  return {
    name: 'game',
    long_name: 'Game',
    dynamics: 'sequential',
    chance_mode: 'deterministic',
    information: 'perfect_information',
    min_seats: 1,
    max_seats: 10,
    num_players: 3,
    parameters: [],
    loadable: true,
    unavailable_reason: null,
    controllers: ['human'],
    renderer: null,
    ...fields,
  };
}

describe('defaultSession', () => {
  it('seats a person in each of the catalogue’s seats', () => {
    expect(defaultSession(entry())).toEqual({
      game: 'game',
      parameters: {},
      seats: Array.from({length: 3}, () => ({
        controller: 'human',
        config: {},
        autoplay: false,
      })),
      chance_policy: 'auto',
      seed: null,
    });
  });

  it('opens Go on the 9×9 board', () => {
    expect(
      defaultSession(entry({name: 'go', num_players: 2})).parameters,
    ).toEqual({board_size: 9});
  });

  it('opens gin rummy on Oklahoma rules', () => {
    expect(
      defaultSession(entry({name: 'gin_rummy', num_players: 2})).parameters,
    ).toEqual({oklahoma: true});
  });

  it('refuses a game the server cannot load', () => {
    const unloadable = entry({
      num_players: null,
      loadable: false,
      unavailable_reason: 'No.',
      controllers: [],
      renderer: null,
    });
    expect(() => defaultSession(unloadable)).toThrow('not loadable');
  });
});

describe('defaultViewer', () => {
  it('opens every game in analysis, as the oracle', () => {
    expect(defaultViewer()).toEqual({kind: 'oracle', seat: null});
  });
});

describe('initialViewer', () => {
  const oracle = {kind: 'oracle', seat: null} as const;

  it('opens for the viewer the address names', () => {
    expect(initialViewer(entry(), {kind: 'seat', seat: 2})).toEqual({
      kind: 'seat',
      seat: 2,
    });
    expect(initialViewer(entry(), {kind: 'spectator', seat: null})).toEqual({
      kind: 'spectator',
      seat: null,
    });
  });

  it('opens for the default viewer when the address names no one', () => {
    expect(initialViewer(entry(), null)).toEqual(oracle);
  });

  it.each([3, 4, 9999])(
    'opens the spectator, never the oracle, for seat %i the game lacks',
    seat => {
      expect(initialViewer(entry(), {kind: 'seat', seat})).toEqual({
        kind: 'spectator',
        seat: null,
      });
    },
  );
});

describe('modeLabel', () => {
  it('names analysis, a seat in a game and a spectator', () => {
    expect(modeLabel({kind: 'oracle', seat: null})).toBe('Analysis');
    expect(modeLabel({kind: 'seat', seat: 4})).toBe('Game, seat 4');
    expect(modeLabel({kind: 'spectator', seat: null})).toBe('Game, spectating');
  });
});
