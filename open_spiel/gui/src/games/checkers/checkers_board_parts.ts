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

// The checkers board's parts: the struct it is drawn from and the pure facts
// about squares and pieces the Gameplay panel reads.

/** The standard board, used only until the server's struct arrives. */
const DEFAULT_BOARD_SIZE = 8;

/** A cell of the struct's board: empty, or a man or king of either side. */
export type PieceType = '' | 'b' | 'B' | 'w' | 'W';

/** The adapter's struct: the board from the bottom, its size, the mover. */
export interface CheckersStruct {
  board: PieceType[][];
  rows?: number;
  columns?: number;
  currentPlayer: number;
}

function isPiece(value: unknown): value is PieceType {
  return (
    value === '' ||
    value === 'b' ||
    value === 'B' ||
    value === 'w' ||
    value === 'W'
  );
}

/**
 * The struct the adapter sent, checked; undefined when it sent none.
 *
 * @throws Error when the struct is not a checkers board.
 */
export function readStruct(value: unknown): CheckersStruct | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'object') {
    throw new Error('The checkers struct is not an object.');
  }
  const board: unknown = Reflect.get(value, 'board');
  const rows: unknown = Reflect.get(value, 'rows');
  const columns: unknown = Reflect.get(value, 'columns');
  const currentPlayer: unknown = Reflect.get(value, 'currentPlayer');
  if (
    !Array.isArray(board) ||
    !board.every(
      (row: unknown) => Array.isArray(row) && row.every(cell => isPiece(cell)),
    )
  ) {
    throw new Error('The checkers struct has no board of pieces.');
  }
  if (typeof currentPlayer !== 'number') {
    throw new Error('The checkers struct has no current player.');
  }
  return {
    board,
    ...(typeof rows === 'number' ? {rows} : {}),
    ...(typeof columns === 'number' ? {columns} : {}),
    currentPlayer,
  };
}

/** The board's size from the server's struct (the game's rows/columns
 * parameters): a 10x10 game was drawn as 8x8 with its pieces misplaced. */
export function boardDimensions(struct: CheckersStruct | undefined): {
  rows: number;
  columns: number;
} {
  const rows = struct?.rows ?? struct?.board.length ?? DEFAULT_BOARD_SIZE;
  const columns =
    struct?.columns ?? struct?.board[0]?.length ?? DEFAULT_BOARD_SIZE;
  return {rows, columns};
}

/** The struct's board, or an empty one of the size given. */
export function buildBoardFromStruct(
  struct: CheckersStruct | undefined,
  rows: number,
  columns: number,
): PieceType[][] {
  if (struct) {
    return struct.board;
  }
  // Return empty board if no struct
  return Array.from({length: rows}, () =>
    Array.from({length: columns}, (): PieceType => ''),
  );
}

/** Column letters and row numbers in display order (flipped for seat 1). */
export function coordinateLabels(
  rows: number,
  columns: number,
  flipped: boolean,
): {files: string[]; ranks: number[]} {
  const files = Array.from({length: columns}, (_, i) =>
    String.fromCharCode(97 + i),
  );
  const ranks = Array.from({length: rows}, (_, i) => rows - i);
  return flipped
    ? {files: files.reverse(), ranks: ranks.reverse()}
    : {files, ranks};
}

/** The seat whose piece `piece` is; null for an empty square. */
export function getPiecePlayer(piece: PieceType): number | null {
  if (piece === 'b' || piece === 'B') return 0; // Black/player 0
  if (piece === 'w' || piece === 'W') return 1; // White/player 1
  return null;
}

/** Whether `piece` is a king. */
export function isKing(piece: PieceType): boolean {
  return piece === 'B' || piece === 'W';
}

/** Whether a seat's controller is a person. */
export function isHumanController(
  controller?: string | null,
  slot?: number,
): boolean {
  if (!controller) {
    return slot === 0;
  }
  return controller === 'human';
}

/** A square's key in the board's sets and maps: "row,col". */
export function squareKey(row: number, col: number): string {
  return `${String(row)},${String(col)}`;
}
