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

import {setupRefusal} from './access.ts';
import {boardFrame} from './testing/fixtures.ts';

const ORACLE = {kind: 'oracle', seat: null} as const;
const SEAT = {kind: 'seat', seat: 0} as const;
const SPECTATOR = {kind: 'spectator', seat: null} as const;

describe('setupRefusal', () => {
  it('lets the oracle and every seat set up the game', () => {
    expect(setupRefusal(boardFrame({viewer: ORACLE}))).toBeNull();
    expect(setupRefusal(boardFrame({viewer: SEAT}))).toBeNull();
    expect(
      setupRefusal(boardFrame({viewer: ORACLE, hidesInformation: true})),
    ).toBeNull();
  });

  it('lets a seat of a game that hides information set it up', () => {
    expect(
      setupRefusal(boardFrame({viewer: SEAT, hidesInformation: true})),
    ).toBeNull();
  });

  it('refuses a spectator, even of a game that hides nothing', () => {
    expect(setupRefusal(boardFrame({viewer: SPECTATOR}))).toBe(
      'Spectators watch the game; they cannot change it.',
    );
  });
});
