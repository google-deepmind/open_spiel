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

/// <reference types="node" />

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';

import {
  isPlayableSquare,
  lastTurn,
  moveNumber,
  parseMoveSquares,
  resultText,
} from './checkers_model.ts';
import {
  cleanup,
  hasClass,
  mount,
  sessionContext,
  square,
} from './checkers_test_session.tsx';
import type {SessionContextValue} from './checkers_session.ts';

/**
 * The board against the English draughts conventions (rule books, ACF/WCDF
 * diagrams, every classic checkers UI): men on the dark squares with a dark
 * single corner at each player's left, the last turn readable at rest, a move
 * that slides, a king as two stacked men, and the game's own words for whose
 * move it is and how it ended.
 */

afterEach(() => {
  cleanup();
});

type Piece = '' | 'b' | 'B' | 'w' | 'W';

/** Puts `piece` on the square at `row`, `col` of `board`. */
function put(board: Piece[][], row: number, col: number, piece: Piece): void {
  const cells = board[row];
  if (cells === undefined) throw new Error(`No row ${String(row)}.`);
  cells[col] = piece;
}

/** The 8x8 opening exactly as the engine sets it up (rows counted from the
 * top, men on (row + col) odd), converted to the struct's bottom-up rows. */
function opening(): Piece[][] {
  const rows = 8;
  const board: Piece[][] = Array.from({length: rows}, () =>
    Array.from({length: 8}, (): Piece => ''),
  );
  for (let engineRow = 0; engineRow < rows; engineRow++) {
    for (let col = 0; col < 8; col++) {
      if ((engineRow + col) % 2 !== 1) continue;
      const row = rows - 1 - engineRow;
      if (engineRow < 3) put(board, row, col, 'w');
      if (engineRow >= rows - 3) put(board, row, col, 'b');
    }
  }
  return board;
}

interface StateOverrides {
  board?: Piece[][];
  history?: number[];
  historyActionStrings?: string[];
  historyPlayers?: number[];
  currentPlayer?: number;
  isTerminal?: boolean;
  returns?: number[] | null;
}

function context(o: StateOverrides = {}): SessionContextValue {
  return sessionContext({
    struct: {
      board: o.board ?? opening(),
      rows: 8,
      columns: 8,
      currentPlayer: o.currentPlayer ?? 0,
    },
    history: o.history ?? [],
    historyActionStrings: o.historyActionStrings ?? [],
    historyPlayers: o.historyPlayers ?? [],
    currentPlayer: o.isTerminal ? -4 : (o.currentPlayer ?? 0),
    isTerminal: o.isTerminal ?? false,
    returns: o.returns ?? null,
  });
}

describe('men stand on the dark squares', () => {
  it('a1 and h8 are dark, h1 and a8 light, on 8x8 and 10x10', () => {
    expect(isPlayableSquare(0, 0, 8)).toBe(true); // a1, single corner
    expect(isPlayableSquare(0, 7, 8)).toBe(false); // h1
    expect(isPlayableSquare(7, 7, 8)).toBe(true); // h8
    expect(isPlayableSquare(7, 0, 8)).toBe(false); // a8
    expect(isPlayableSquare(0, 0, 10)).toBe(true);
  });

  it('every man of the opening sits on a dark square, and a1 is dark', () => {
    const {getByTestId} = mount({ctx: context()});
    const board = getByTestId('checkers-board');
    const pieces = board.querySelectorAll('.checkers-piece');
    expect(pieces).toHaveLength(24);
    pieces.forEach(piece => {
      expect(hasClass(piece.parentElement, 'checkers-square--dark')).toBe(true);
    });
    expect(hasClass(square(board, 0, 0), 'checkers-square--dark')).toBe(true);
    expect(hasClass(square(board, 0, 7), 'checkers-square--light')).toBe(true);
  });
});

describe('the last turn is on the board at rest', () => {
  it('parses squares and folds a multi-jump into one turn', () => {
    expect(parseMoveSquares('a10b9')).toEqual([
      {row: 9, col: 0},
      {row: 8, col: 1},
    ]);
    const turn = lastTurn(['c3d4', 'b6c5', 'd4b6', 'b6d8'], [0, 1, 0, 0]);
    expect(turn?.seat).toBe(0);
    expect(turn?.path).toEqual([
      {row: 3, col: 3}, // d4
      {row: 5, col: 1}, // b6
      {row: 7, col: 3}, // d8
    ]);
    expect(turn?.captured).toEqual([
      {row: 4, col: 2}, // c5
      {row: 6, col: 2}, // c7
    ]);
    expect(turn?.lastLeg).toEqual({
      from: {row: 5, col: 1},
      to: {row: 7, col: 3},
    });
    expect(lastTurn([], [])).toBeNull();
    expect(() => lastTurn(['c3d4'], [])).toThrow();
  });

  it('marks every square of the turn and where each taken man stood', () => {
    const board = opening();
    put(board, 2, 2, ''); // c3 left
    put(board, 3, 3, 'b'); // d4 arrived
    const {getByTestId} = mount({
      ctx: context({
        board,
        history: [1],
        historyActionStrings: ['c3d4'],
        historyPlayers: [0],
        currentPlayer: 1,
      }),
    });
    const root = getByTestId('checkers-board');
    expect(hasClass(square(root, 2, 2), 'checkers-square--last-move')).toBe(
      true,
    );
    expect(hasClass(square(root, 3, 3), 'checkers-square--last-move')).toBe(
      true,
    );
    expect(root.querySelectorAll('.checkers-square--last-move')).toHaveLength(
      2,
    );
    expect(root.querySelectorAll('.checkers-square--captured')).toHaveLength(0);
  });

  it('a capture leaves a ring on the taken square', () => {
    const board = opening();
    put(board, 4, 2, ''); // c5, the taken man, gone
    const {getByTestId} = mount({
      ctx: context({
        board,
        history: [1, 2, 3],
        historyActionStrings: ['c3d4', 'b6c5', 'd4b6'],
        historyPlayers: [0, 1, 0],
        currentPlayer: 1,
      }),
    });
    const root = getByTestId('checkers-board');
    expect(hasClass(square(root, 4, 2), 'checkers-square--captured')).toBe(
      true,
    );
  });
});

describe('a move slides; the position it opened on does not', () => {
  it('slides the piece of a move made while open, from its start square', () => {
    const board = opening();
    const holder = {ctx: context({board})};
    const {getByTestId, again} = mount(holder);
    expect(
      getByTestId('checkers-board').querySelector('.checkers-piece--sliding'),
    ).toBeNull();

    const after = opening();
    put(after, 2, 2, '');
    put(after, 3, 3, 'b');
    holder.ctx = context({
      board: after,
      history: [1],
      historyActionStrings: ['c3d4'],
      historyPlayers: [0],
      currentPlayer: 1,
    });
    again();
    const root = getByTestId('checkers-board');
    const sliding = root.querySelectorAll('.checkers-piece--sliding');
    expect(sliding).toHaveLength(1);
    const piece = sliding[0] as HTMLElement;
    expect(piece.parentElement).toBe(square(root, 3, 3));
    // From c3 to d4: it starts one square left and one square down.
    expect(piece.style.getPropertyValue('--slide-dx')).toBe('-1');
    expect(piece.style.getPropertyValue('--slide-dy')).toBe('1');
  });
});

describe('a new game slides its first move', () => {
  it('slides the first move of a new game after a longer one', () => {
    const long = {
      history: [1, 2, 3],
      historyActionStrings: ['c3d4', 'b6c5', 'd4b6'],
      historyPlayers: [0, 1, 0],
      currentPlayer: 1,
    };
    const holder = {ctx: context(long)};
    const {getByTestId, again} = mount(holder);
    holder.ctx = context(); // a new game: the history shrinks to nothing
    again();
    expect(
      getByTestId('checkers-board').querySelector('.checkers-piece--sliding'),
    ).toBeNull();
    const after = opening();
    put(after, 2, 2, '');
    put(after, 3, 3, 'b');
    holder.ctx = context({
      board: after,
      history: [1],
      historyActionStrings: ['c3d4'],
      historyPlayers: [0],
      currentPlayer: 1,
    });
    again();
    const sliding = getByTestId('checkers-board').querySelectorAll(
      '.checkers-piece--sliding',
    );
    expect(sliding).toHaveLength(1);
    // An unrelated re-render keeps the same slide, not a restarted one.
    again();
    expect(
      getByTestId('checkers-board').querySelectorAll(
        '.checkers-piece--sliding',
      )[0],
    ).toBe(sliding[0]);
  });
});

describe('the position a board opens on', () => {
  it('does not replay the last move when the board opens on it', () => {
    const board = opening();
    put(board, 2, 2, '');
    put(board, 3, 3, 'b');
    const {getByTestId} = mount({
      ctx: context({
        board,
        history: [1],
        historyActionStrings: ['c3d4'],
        historyPlayers: [0],
        currentPlayer: 1,
      }),
    });
    expect(
      getByTestId('checkers-board').querySelector('.checkers-piece--sliding'),
    ).toBeNull();
  });
});

describe('the slide CSS', () => {
  it('the slide reads the Animation speed and stops for reduced motion', () => {
    const css = readFileSync(
      resolve(import.meta.dirname, 'checkers.css'),
      'utf-8',
    );
    expect(css).toMatch(
      /\.checkers-piece--sliding\s*\{[^}]*animation:\s*checkers-slide\s+calc\([^)]*var\(--anim-scale, 1\)\)/,
    );
    expect(css).toMatch(
      /prefers-reduced-motion: reduce\)\s*\{\s*\.checkers-piece--sliding\s*\{\s*animation:\s*none/,
    );
  });
});

describe('a king is two men, one on the other', () => {
  it('draws a plain lower man before the crowned one, in the same square', () => {
    const board = opening();
    put(board, 7, 1, 'B'); // a black king on b8
    const {getByTestId} = mount({ctx: context({board})});
    const sq = square(getByTestId('checkers-board'), 7, 1);
    const base = sq.querySelector('.checkers-king-base');
    const king = sq.querySelector('.checkers-piece--king');
    expect(base).not.toBeNull();
    expect(king).not.toBeNull();
    expect(hasClass(base, 'checkers-piece--black')).toBe(true);
    // A real element painted first, so it stays under the upper man even
    // when hover or a slide makes that one a stacking context.
    if (base === null || king === null) throw new Error('No king drawn.');
    expect(
      base.compareDocumentPosition(king) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    // A man has no lower man.
    expect(
      square(getByTestId('checkers-board'), 0, 0).querySelector(
        '.checkers-king-base',
      ),
    ).toBeNull();
  });
});

describe('a history the model cannot read', () => {
  it('is said in a notice and the board still draws', () => {
    const {getByTestId, container} = mount({
      ctx: context({
        history: [1],
        historyActionStrings: ['c3d4'],
        historyPlayers: [],
      }),
    });
    expect(
      getByTestId('checkers-board').querySelectorAll('.checkers-piece'),
    ).toHaveLength(24);
    expect(container.textContent).toContain('Last move not shown');
  });
});

describe('whose move and the result, in the game’s words', () => {
  it('names the side to move and counts moves as checkers does', () => {
    expect(moveNumber([])).toBe(1);
    expect(moveNumber([0, 1, 0, 0])).toBe(2);
    expect(moveNumber([0, 1, 1, 1])).toBe(2);
    const {getByTestId} = mount({
      ctx: context({
        history: [1],
        historyActionStrings: ['c3d4'],
        historyPlayers: [0],
        currentPlayer: 1,
      }),
    });
    expect(getByTestId('checkers-status').textContent).toContain(
      'White to move',
    );
  });

  it('says who won or that it was drawn, not raw returns', () => {
    expect(resultText([1, -1])).toBe('Black wins');
    expect(resultText([-1, 1])).toBe('White wins');
    expect(resultText([0, 0])).toBe('Draw');
    const {getByTestId, container} = mount({
      ctx: context({isTerminal: true, returns: [-1, 1]}),
    });
    expect(getByTestId('checkers-status').textContent).toContain('White wins');
    expect(container.textContent).not.toContain('Final returns');
    expect(container.textContent).not.toContain('-1.00');
  });
});
