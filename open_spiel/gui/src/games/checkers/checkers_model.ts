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

// Checkers facts the board draws, derived from the server's struct and the
// session history. Pure, so each rule is pinned by a unit test.
//
// Coordinates are the struct's: row 0 is rank 1 (the bottom, Black's side),
// column 0 is file a.

/** A square of the board: its row from the bottom and column from the left. */
export interface SquareCoord {
  row: number;
  col: number;
}

/** Player 0 moves first and is Black, as in English draughts (the engine's
 * 'o' glyph; the adapter maps it to 'b'). */
export const SIDE_NAMES = ['Black', 'White'] as const;

/** The side seat `seat` plays. */
export function sideName(seat: number): string {
  const name = SIDE_NAMES.at(seat);
  if (seat < 0 || name === undefined) {
    throw new Error(`checkers has no seat ${String(seat)}`);
  }
  return name;
}

/**
 * The playable (dark) squares: the ones the engine puts men on. The engine
 * counts rows from the top and plays on (row + col) odd there, so on the
 * struct's bottom-up rows a square is dark when (rows - 1 - row + col) is
 * odd. On 8x8 that makes a1 dark, the single corner on each player's left
 * and the double corner on their right, as every rule book draws it.
 */
export function isPlayableSquare(
  row: number,
  col: number,
  rows: number,
): boolean {
  return (rows - 1 - row + col) % 2 === 1;
}

/** "c3d4" or "a3c5e7" -> the squares the piece stands on, in order. A
 * column is one letter, a row any number of digits (a 10x10 "a10b9"). */
export function parseMoveSquares(label: string): SquareCoord[] {
  const squares: SquareCoord[] = [];
  for (const [, column = 'a', row = '0'] of label
    .toLowerCase()
    .matchAll(/([a-z])(\d+)/g)) {
    squares.push({
      row: Number(row) - 1,
      col: column.charCodeAt(0) - 97,
    });
  }
  return squares;
}

/** The last turn: one move, or every jump of a multiple capture. */
export interface LastTurn {
  /** The seat that made it. */
  seat: number;
  /** Every square the moving piece stood on: start, each landing, end. */
  path: SquareCoord[];
  /** Squares of the pieces it took, in the order it took them. */
  captured: SquareCoord[];
  /** The final leg, which the board slides. */
  lastLeg: {from: SquareCoord; to: SquareCoord} | null;
}

function at<T>(items: readonly T[], index: number): T {
  const item = items[index];
  if (item === undefined) {
    throw new Error(`checkers history has no entry ${String(index)}`);
  }
  return item;
}

/**
 * The last turn as a whole: a multi-jump is one action per jump with the same
 * player to move, so the turn is the trailing run of actions by the player who
 * made the last one. Null before the first move or without labels.
 */
export function lastTurn(
  labels: readonly string[] | null | undefined,
  players: readonly number[] | null | undefined,
): LastTurn | null {
  if (!labels || !players || labels.length === 0) return null;
  if (labels.length !== players.length) {
    throw new Error(
      `checkers history has ${String(labels.length)} labels for ${String(players.length)} players`,
    );
  }
  const seat = at(players, players.length - 1);
  let start = players.length - 1;
  while (start > 0 && players[start - 1] === seat) start -= 1;
  const path: SquareCoord[] = [];
  const captured: SquareCoord[] = [];
  let lastLeg: LastTurn['lastLeg'] = null;
  for (let i = start; i < labels.length; i++) {
    const label = at(labels, i);
    const squares = parseMoveSquares(label);
    if (squares.length < 2) {
      throw new Error(`unreadable checkers move label ${label}`);
    }
    squares.forEach((sq, j) => {
      if (path.length === 0 || j > 0) path.push(sq);
      if (j === 0) return;
      const prev = at(squares, j - 1);
      if (Math.abs(sq.row - prev.row) === 2) {
        captured.push({
          row: (sq.row + prev.row) / 2,
          col: (sq.col + prev.col) / 2,
        });
      }
    });
    lastLeg = {
      from: at(squares, squares.length - 2),
      to: at(squares, squares.length - 1),
    };
  }
  return {seat, path, captured, lastLeg};
}

/** The move number as checkers counts it: one move is a turn by each side,
 * and a multi-jump is one turn however many legs it has. */
export function moveNumber(
  players: readonly number[] | null | undefined,
): number {
  let turns = 0;
  (players ?? []).forEach((p, i, all) => {
    if (i === 0 || all[i - 1] !== p) turns += 1;
  });
  return Math.floor(turns / 2) + 1;
}

/** The end of the game in its own words, from the engine's returns. */
export function resultText(returns: readonly number[]): string {
  const [first, second] = returns;
  if (returns.length !== 2 || first === undefined || second === undefined) {
    throw new Error(`checkers returns ${JSON.stringify(returns)}`);
  }
  if (first > second) return `${sideName(0)} wins`;
  if (second > first) return `${sideName(1)} wins`;
  return 'Draw';
}
