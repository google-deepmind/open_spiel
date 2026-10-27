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

// The Deadwood line during a laydown: what the composition leaves.

import {describe, expect, it} from 'vitest';

import {ginLaydownDeadwood} from './gin_gameplay_cards.ts';
import {ginComposedDeadwood} from './gin_layoffs.ts';

// Seed 2837's knock: 3s3c3h, 7s7d7h, TdJdQdKd and 4h. Discarding 4h is gin.
const HAND = ['3s', '3c', '3h', '7s', '7d', '7h', 'Td', 'Jd', 'Qd', 'Kd', '4h'];
const THREES = ['3s', '3c', '3h'];
const SEVENS = ['7s', '7d', '7h'];
const RUN = ['Td', 'Jd', 'Qd', 'Kd'];
const NONE: ReadonlySet<string> = new Set();

function staged(...melds: string[][]): {
  melded: Set<string>;
  layoffs: Set<string>;
} {
  return {melded: new Set(melds.flat()), layoffs: new Set()};
}

describe('ginLaydownDeadwood in the knocker discard sub-stage', () => {
  it('is gin (0) when the free card is the staged discard', () => {
    const all = staged(THREES, SEVENS, RUN);
    expect(ginLaydownDeadwood('discard', HAND, all, '4h')).toBe(0);
  });

  it('counts what discarding a melded card leaves (the 3s: 10, as the engine scored)', () => {
    // Marking 3s as the discard evicts the set of threes from the melds.
    const left = staged(SEVENS, RUN);
    expect(ginLaydownDeadwood('discard', HAND, left, '3s')).toBe(10);
  });

  it('changes as the staged discard changes', () => {
    const left = staged(SEVENS, RUN);
    expect(ginLaydownDeadwood('discard', HAND, left, '3s')).toBe(10);
    const all = staged(THREES, SEVENS, RUN);
    expect(ginLaydownDeadwood('discard', HAND, all, '4h')).toBe(0);
  });

  it('with no discard picked, leaves out the best discard (never the whole hand, 74)', () => {
    const all = staged(THREES, SEVENS, RUN);
    expect(ginLaydownDeadwood('discard', HAND, all, null)).toBe(0);
    // With only the sevens and the run staged, 3s 3c 3h 4h are free: the
    // best discard is 4h and 9 is left.
    const left = staged(SEVENS, RUN);
    expect(ginLaydownDeadwood('discard', HAND, left, null)).toBe(9);
  });

  it('is 0 with no discard picked and every card melded', () => {
    const run = ['9d', ...RUN];
    const hand = [...THREES, ...SEVENS, ...run];
    expect(
      ginLaydownDeadwood('discard', hand, staged(THREES, SEVENS, run), null),
    ).toBe(0);
  });
});

describe('ginLaydownDeadwood in the meld and lay-off sub-stages', () => {
  it('never counts the staged discard in the melds sub-stage', () => {
    const all = staged(THREES, SEVENS, RUN);
    expect(ginLaydownDeadwood('melds', HAND, all, '4h')).toBe(0);
    expect(ginLaydownDeadwood('melds', HAND, all, null)).toBe(4);
  });

  it("counts the defender's hand less its staged meld and lay-offs", () => {
    // A defender: 28 is left after the queens.
    const hand = ['Qs', 'Qc', 'Qd', 'Qh', '2s', '2c', '4c', '8d', '5h', '7h'];
    const queens = staged(['Qs', 'Qc', 'Qd', 'Qh']);
    expect(ginLaydownDeadwood('layoffs', hand, queens, null)).toBe(28);
    const laidOff = {melded: queens.melded, layoffs: new Set(['8d'])};
    expect(ginLaydownDeadwood('layoffs', hand, laidOff, null)).toBe(20);
  });

  it('is null outside a laydown', () => {
    expect(ginLaydownDeadwood(null, HAND, staged(), null)).toBeNull();
  });
});

describe('ginComposedDeadwood', () => {
  it('without a discard counts every free card', () => {
    expect(ginComposedDeadwood(HAND, NONE, NONE)).toBe(74);
  });

  it('leaves out the staged discard', () => {
    const melded = new Set([...SEVENS, ...RUN]);
    expect(
      ginComposedDeadwood(HAND, melded, NONE, {staged: '3s', pending: true}),
    ).toBe(10);
  });
});
