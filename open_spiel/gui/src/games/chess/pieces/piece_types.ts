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

// The names a chess piece set draws by, and the shape of a set: one module,
// a label and a component that draws one piece.

import type {JSX} from 'react';

/** A kind of chess piece, as the board names it. */
export type PieceRole =
  'pawn' | 'knight' | 'bishop' | 'rook' | 'queen' | 'king';

/** A side's colour. */
export type PieceColor = 'white' | 'black';

/** Every kind of piece, king first. */
export const PIECE_ROLES: readonly PieceRole[] = [
  'king',
  'queen',
  'rook',
  'bishop',
  'knight',
  'pawn',
];

/** Both colours, white first. */
export const PIECE_COLORS: readonly PieceColor[] = ['white', 'black'];

/** What a set's component is given to draw one piece. */
export interface PieceArtProps {
  readonly role: PieceRole;
  readonly color: PieceColor;
}

/**
 * A piece set: its name in the Board settings and the component that draws
 * one piece. The component fills the box it is put in, square, and carries
 * the class `chess-piece-art` on its root element; it draws nothing a screen
 * reader should read, since the board names each piece itself.
 */
export interface PieceSetModule {
  readonly label: string;
  readonly Piece: (props: PieceArtProps) => JSX.Element;
  /**
   * The piece as an image URL, for a board that draws its pieces as images
   * (a CSS background, an `<img>`); absent when the set offers none.
   */
  readonly imageUrl?: (props: PieceArtProps) => string;
}
