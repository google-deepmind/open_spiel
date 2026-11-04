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

import {afterEach, describe, expect, it} from 'vitest';

import type {PieceType} from './checkers_board_parts.ts';
import {boardDimensions, coordinateLabels} from './checkers_gameplay_panel.tsx';
import {
  cleanup,
  hasClass,
  mount,
  sessionContext,
} from './checkers_test_session.tsx';

afterEach(() => {
  cleanup();
});

/** A 10x10 opening as the server sends it: rows x columns, row 0 = rank 1. */
function tenByTen(): {
  board: PieceType[][];
  rows: number;
  columns: number;
  currentPlayer: number;
} {
  const board = Array.from({length: 10}, (_, row) =>
    Array.from({length: 10}, (_, col): PieceType => {
      if ((row + col) % 2 === 1) return '';
      if (row < 3) return 'b';
      if (row > 6) return 'w';
      return '';
    }),
  );
  return {board, rows: 10, columns: 10, currentPlayer: 0};
}

describe('Checkers board size', () => {
  it('takes the size from the struct, not a hard-coded 8', () => {
    expect(boardDimensions(tenByTen())).toEqual({rows: 10, columns: 10});
    expect(
      boardDimensions({board: tenByTen().board, currentPlayer: 0}),
    ).toEqual({rows: 10, columns: 10});
    expect(boardDimensions(undefined)).toEqual({rows: 8, columns: 8});
  });

  it('labels every column and rank, flipped for the second seat', () => {
    expect(coordinateLabels(10, 10, false)).toEqual({
      files: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
      ranks: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    });
    const flipped = coordinateLabels(8, 8, true);
    expect(flipped.files[0]).toBe('h');
    expect(flipped.ranks[0]).toBe(1);
  });

  it('draws a 10x10 grid with each piece on its own square', () => {
    const struct = tenByTen();
    const {getByTestId} = mount({ctx: sessionContext({struct})});
    const board = getByTestId('checkers-board');
    expect(board.querySelectorAll('.checkers-square')).toHaveLength(100);
    expect(board.style.gridTemplateColumns).toBe('repeat(10, 1fr)');
    for (let row = 0; row < 10; row++) {
      for (let col = 0; col < 10; col++) {
        const square = board.querySelector(
          `[data-row="${String(row)}"][data-col="${String(col)}"]`,
        );
        const piece = square?.querySelector('.checkers-piece') ?? null;
        const want = struct.board[row]?.[col];
        if (want === 'b') {
          expect(hasClass(piece, 'checkers-piece--black')).toBe(true);
        } else if (want === 'w') {
          expect(hasClass(piece, 'checkers-piece--white')).toBe(true);
        } else {
          expect(piece).toBeNull();
        }
      }
    }
  });
});
