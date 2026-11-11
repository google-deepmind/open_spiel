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

// The piece-set registry: how a board's piece codes are drawn. A piece set
// maps a code to a Unicode glyph and a tone, light or dark, so art can
// replace the glyphs later by registering a set, without touching any
// renderer.

/** How one piece is drawn. */
export interface Piece {
  readonly glyph: string;
  readonly tone: 'light' | 'dark';
  /** The accessible name: "white king". */
  readonly name: string;
}

/** A set of pieces, keyed by the codes an adapter sends. */
export interface PieceSet {
  readonly name: string;
  /** The piece for `code`; throws on a code the set does not have. */
  piece(code: string): Piece;
}

// U+FE0E asks for the text form of a glyph that also has an emoji form.
const TEXT = '\uFE0E';

/** Builds a piece set from a table of codes. */
export function tablePieceSet(
  name: string,
  table: Readonly<Record<string, Piece>>,
): PieceSet {
  return {
    name,
    piece: code => {
      const piece = table[code];
      if (piece === undefined) {
        throw new Error(`Piece set "${name}" has no piece "${code}".`);
      }
      return piece;
    },
  };
}

const CHESS_GLYPHS: Record<string, [string, string]> = {
  k: ['♚', 'king'],
  q: ['♛', 'queen'],
  r: ['♜', 'rook'],
  b: ['♝', 'bishop'],
  n: ['♞', 'knight'],
  p: [`♟${TEXT}`, 'pawn'],
};

function chessTable(): Record<string, Piece> {
  const table: Record<string, Piece> = {};
  for (const [letter, [glyph, name]] of Object.entries(CHESS_GLYPHS)) {
    table[letter.toUpperCase()] = {glyph, tone: 'light', name: `white ${name}`};
    table[letter] = {glyph, tone: 'dark', name: `black ${name}`};
  }
  return table;
}

/** Chess, by FEN letters: upper case white, lower case black. */
export const CHESS_PIECES = tablePieceSet('chess', chessTable());

/**
 * Draughts: "o" and "O" light man and king, "x" and "X" dark.
 *
 * "Man" and "king" are the standard draughts terms for an uncrowned and a
 * crowned piece, and light and dark name the colours of the two sets of
 * pieces. We use them because they are the game's own vocabulary; "piece"
 * would not tell a man from a king.
 */
export const CHECKERS_PIECES = tablePieceSet('checkers', {
  o: {glyph: '●', tone: 'light', name: 'light man'},
  O: {glyph: '♛', tone: 'light', name: 'light king'},
  x: {glyph: '●', tone: 'dark', name: 'dark man'},
  X: {glyph: '♛', tone: 'dark', name: 'dark king'},
});

/** Pen marks: "x" and "o". */
export const MARK_PIECES = tablePieceSet('marks', {
  x: {glyph: '✕', tone: 'dark', name: 'x'},
  o: {glyph: '◯', tone: 'dark', name: 'o'},
});

/** Stones: "x" dark and "o" light, as connection games draw them. */
export const STONE_PIECES = tablePieceSet('stones', {
  x: {glyph: '●', tone: 'dark', name: 'dark stone'},
  o: {glyph: '●', tone: 'light', name: 'light stone'},
});

const pieceSets = new Map<string, PieceSet>(
  [CHESS_PIECES, CHECKERS_PIECES, MARK_PIECES, STONE_PIECES].map(set => [
    set.name,
    set,
  ]),
);

/** Registers `set` under its name; throws if the name is taken. */
export function registerPieceSet(set: PieceSet): void {
  if (pieceSets.has(set.name)) {
    throw new Error(`Piece set "${set.name}" is already registered.`);
  }
  pieceSets.set(set.name, set);
}

/** The piece set named `name`; throws when there is none. */
export function pieceSetFor(name: string): PieceSet {
  const set = pieceSets.get(name);
  if (set === undefined) throw new Error(`There is no piece set "${name}".`);
  return set;
}
