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

// The chess pieces the board, the players' strips and the promotion choice
// draw, in the piece set the viewer picks in the Board settings
// (pieces/piece_sets.ts); the Staunton set until they pick another.

import type {JSX} from 'react';

import {usePieceSet} from './chess_config_section.tsx';
import {getPieceSet} from './pieces/piece_sets.ts';
import type {PieceColor, PieceRole} from './pieces/piece_types.ts';

export type {PieceColor, PieceRole} from './pieces/piece_types.ts';

/** The art of one chess piece, in the viewer's piece set, filling its box. */
export function ChessPieceArt({
  role,
  color,
}: {
  readonly role: PieceRole;
  readonly color: PieceColor;
}): JSX.Element {
  const [setId] = usePieceSet();
  const {Piece} = getPieceSet(setId);
  return <Piece role={role} color={color} />;
}

const ROLE_OF_LETTER: Readonly<Record<string, PieceRole>> = {
  p: 'pawn',
  n: 'knight',
  b: 'bishop',
  r: 'rook',
  q: 'queen',
  k: 'king',
};

/** The role of a FEN piece letter; throws on any other letter. */
export function roleOfLetter(letter: string): PieceRole {
  const role = ROLE_OF_LETTER[letter.toLowerCase()];
  if (role === undefined) throw new Error(`No chess piece "${letter}".`);
  return role;
}
