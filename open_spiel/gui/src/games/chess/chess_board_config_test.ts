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
  drawnAt,
  readPlacement,
  squareAt,
  squareIndex,
  squareName,
} from './chess_board_config.ts';

describe('squares', () => {
  it('names and indexes squares from a1', () => {
    expect(squareIndex('a1')).toEqual({file: 0, rank: 0});
    expect(squareIndex('h8')).toEqual({file: 7, rank: 7});
    expect(squareName(4, 3)).toBe('e4');
    expect(() => squareIndex('i9')).toThrow('Not a square');
  });

  it('draws the board from either side', () => {
    expect(drawnAt('a8', 'white')).toEqual({column: 0, row: 0});
    expect(drawnAt('a1', 'white')).toEqual({column: 0, row: 7});
    expect(drawnAt('h1', 'black')).toEqual({column: 0, row: 0});
    expect(drawnAt('a8', 'black')).toEqual({column: 7, row: 7});
  });

  it('finds the square drawn at a place, and none off the board', () => {
    expect(squareAt(4, 4, 'white')).toBe('e4');
    expect(squareAt(4, 4, 'black')).toBe('d5');
    expect(squareAt(8, 0, 'white')).toBeNull();
    expect(squareAt(0, -1, 'black')).toBeNull();
  });
});

describe('readPlacement', () => {
  it('reads the pieces of a FEN by square', () => {
    const placement = readPlacement(
      'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
    );
    expect(placement.size).toBe(32);
    expect(placement.get('e4')).toBe('P');
    expect(placement.get('e2')).toBeUndefined();
    expect(placement.get('e8')).toBe('k');
  });

  it('refuses a malformed placement', () => {
    expect(() => readPlacement('8/8/8 w - - 0 1')).toThrow('Not a FEN');
    expect(() => readPlacement('9/8/8/8/8/8/8/8 w - - 0 1')).toThrow(
      'Not a FEN',
    );
  });
});
