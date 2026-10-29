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

// What the chess board is told to draw and allow, the shape the board's
// owner (use_chess_board.ts) builds and the board (chess_board_surface.tsx)
// reads. Squares are named "a1" to "h8".

import type {PieceColor} from './chess_pieces.tsx';

/** The board's settings at one position. */
export interface BoardConfig {
  /** The position, a FEN; only its placement field is drawn. */
  readonly fen: string;
  /** The side drawn at the bottom. */
  readonly orientation: PieceColor;
  readonly turnColor: PieceColor;
  /** The side whose king is in check, marked under the king. */
  readonly check: PieceColor | undefined;
  /** The last move's two squares, marked. */
  readonly lastMove: readonly [string, string] | undefined;
  /** No piece can be picked up. */
  readonly viewOnly: boolean;
  readonly highlight: {
    readonly lastMove: boolean;
    readonly check: boolean;
  };
  /** How long a piece slides to its new square. */
  readonly animation: {readonly enabled: boolean; readonly duration: number};
  readonly movable: {
    /** The side that may move, or none. */
    readonly color: PieceColor | undefined;
    /** Each square's legal destinations. */
    readonly dests: ReadonlyMap<string, readonly string[]> | undefined;
    /** Whether a picked-up piece's destinations are marked. */
    readonly showDests: boolean;
    readonly events: {
      /** A piece was moved from `orig` to `dest`, a legal move. */
      readonly after: (orig: string, dest: string) => void;
    };
  };
  readonly draggable: {readonly enabled: boolean; readonly showGhost: boolean};
  readonly selectable: {readonly enabled: boolean};
  readonly coordinates: boolean;
}

const FILES = 'abcdefgh';

/** A square's file and rank, each 0 to 7, from a1. */
export function squareIndex(square: string): {file: number; rank: number} {
  const file = FILES.indexOf(square[0] ?? '');
  const rank = Number(square[1]) - 1;
  if (file < 0 || !Number.isInteger(rank) || rank < 0 || rank > 7) {
    throw new Error(`Not a square: "${square}".`);
  }
  return {file, rank};
}

/** The square at `file` and `rank`, each 0 to 7, from a1. */
export function squareName(file: number, rank: number): string {
  return `${FILES[file] ?? '?'}${String(rank + 1)}`;
}

/**
 * The square's column and row as drawn, each 0 to 7 from the top left, for
 * the board seen from `orientation`.
 */
export function drawnAt(
  square: string,
  orientation: PieceColor,
): {column: number; row: number} {
  const {file, rank} = squareIndex(square);
  return orientation === 'white'
    ? {column: file, row: 7 - rank}
    : {column: 7 - file, row: rank};
}

/** The square drawn at `column` and `row` from the top left, or null. */
export function squareAt(
  column: number,
  row: number,
  orientation: PieceColor,
): string | null {
  if (column < 0 || column > 7 || row < 0 || row > 7) return null;
  return orientation === 'white'
    ? squareName(column, 7 - row)
    : squareName(7 - column, row);
}

/** A piece on a square: its FEN letter. */
export type Placement = ReadonlyMap<string, string>;

/** The pieces of a FEN's placement field, by square. */
export function readPlacement(fen: string): Placement {
  const placement = new Map<string, string>();
  const rows = (fen.trim().split(/\s+/)[0] ?? '').split('/');
  if (rows.length !== 8) throw new Error(`Not a FEN placement: "${fen}".`);
  rows.forEach((row, index) => {
    const rank = 7 - index;
    let file = 0;
    for (const letter of row) {
      if (/\d/.test(letter)) {
        file += Number(letter);
      } else {
        placement.set(squareName(file, rank), letter);
        file += 1;
      }
    }
    if (file !== 8) throw new Error(`Not a FEN placement: "${fen}".`);
  });
  return placement;
}
