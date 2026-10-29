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

import type {HistoryEntry} from '../contract/contract.ts';
import {replayStops} from './replay_stops.ts';

const CHANCE: HistoryEntry = {
  seat: null,
  action: 0,
  label: 'deal',
  rewards: [0, 0],
  public_label: null,
};

function seat(index: number): HistoryEntry {
  return {
    seat: index,
    action: null,
    label: null,
    rewards: null,
    public_label: null,
  };
}

// Two chance outcomes, two seat actions, a chance outcome, a seat action.
const HISTORY = [CHANCE, CHANCE, seat(0), seat(1), CHANCE, seat(0)];

describe('replayStops', () => {
  it('stops at every position when chance is not skipped', () => {
    expect(replayStops(HISTORY, 2, false)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(replayStops([], 0, false)).toEqual([0]);
  });

  it('skips chance to the initial, seat and live positions', () => {
    expect(replayStops(HISTORY, 0, true)).toEqual([0, 2, 3, 5, 6]);
  });

  it('keeps the position shown among the stops', () => {
    expect(replayStops(HISTORY, 1, true)).toEqual([0, 1, 2, 3, 5, 6]);
    expect(replayStops(HISTORY, 4, true)).toEqual([0, 2, 3, 4, 5, 6]);
  });

  it('reads redacted entries by their seat', () => {
    const redacted = HISTORY.map(entry => ({
      ...entry,
      action: null,
      label: null,
      rewards: null,
    }));
    expect(replayStops(redacted, 0, true)).toEqual([0, 2, 3, 5, 6]);
  });

  it('refuses a shown position outside the history', () => {
    expect(() => replayStops(HISTORY, 7, true)).toThrow(
      'Ply 7 is not between 0 and 6.',
    );
    expect(() => replayStops(HISTORY, -1, false)).toThrow(RangeError);
    expect(() => replayStops(HISTORY, 1.5, false)).toThrow(RangeError);
  });
});
