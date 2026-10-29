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

import {
  PERSPECTIVE_LOCKED_SEATED,
  PERSPECTIVE_LOCKED_SPECTATING,
  bottomSeatOf,
  perspectiveLock,
  sessionModeOf,
} from './bottom_seat.ts';

const ANALYSIS = {mode: 'analysis'};
const GAME = {mode: 'game'};
const ORACLE = {seat: null, viewSeat: null, isParticipant: false};

describe('sessionModeOf', () => {
  it('is analysis for the oracle and a game for anyone else', () => {
    expect(sessionModeOf({kind: 'oracle', seat: null})).toBe('analysis');
    expect(sessionModeOf({kind: 'seat', seat: 1})).toBe('game');
    expect(sessionModeOf({kind: 'spectator', seat: null})).toBe('game');
  });
});

describe('perspectiveLock', () => {
  it('is switchable in analysis', () => {
    expect(perspectiveLock(ANALYSIS, ORACLE)).toEqual({
      switchable: true,
      reason: null,
    });
  });

  it('is locked in a game, with its reasons', () => {
    expect(perspectiveLock(GAME, {seat: 2, isParticipant: true}).reason).toBe(
      PERSPECTIVE_LOCKED_SEATED,
    );
    expect(perspectiveLock(GAME, {seat: null, isParticipant: false})).toEqual({
      switchable: false,
      reason: PERSPECTIVE_LOCKED_SPECTATING,
    });
  });
});

describe('bottomSeatOf', () => {
  it('draws analysis from the seat chosen, else the view seat, else 0', () => {
    expect(bottomSeatOf(ANALYSIS, ORACLE, 3).seat).toBe(3);
    expect(bottomSeatOf(ANALYSIS, {...ORACLE, viewSeat: 2}, null).seat).toBe(2);
    expect(bottomSeatOf(ANALYSIS, ORACLE, null).seat).toBe(0);
  });

  it("draws a game from the viewer's own seat whatever was chosen", () => {
    expect(bottomSeatOf(GAME, {seat: 1, isParticipant: true}, 3)).toMatchObject(
      {seat: 1, switchable: false},
    );
    expect(bottomSeatOf(GAME, {seat: null, isParticipant: false}, 3).seat).toBe(
      0,
    );
  });
});
