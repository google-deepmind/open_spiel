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

import type {JSX} from 'react';

import {
  ChessPieceArt,
  type PieceColor,
  type PieceRole,
} from './chess_pieces.tsx';

/** The piece a pawn becomes, as a FEN letter. */
export type PromotionChoice = 'q' | 'n' | 'r' | 'b';

/** The properties of the promotion choice. */
export interface PromotionDialogProps {
  /** The file (a-h) where promotion is happening */
  file: string;
  /** The color of the promoting pawn */
  color: PieceColor;
  /** Board size in pixels */
  boardSize: number;
  /** Called when a piece is selected */
  onSelect: (piece: PromotionChoice) => void;
  /** Called when promotion is cancelled (click outside) */
  onCancel: () => void;
}

const PIECES: Array<{role: PieceRole; choice: PromotionChoice}> = [
  {role: 'queen', choice: 'q'},
  {role: 'knight', choice: 'n'},
  {role: 'rook', choice: 'r'},
  {role: 'bishop', choice: 'b'},
];

const FILE_TO_INDEX: Record<string, number> = {
  a: 0,
  b: 1,
  c: 2,
  d: 3,
  e: 4,
  f: 5,
  g: 6,
  h: 7,
};

const ROLE_NAME: Record<PieceRole, string> = {
  pawn: 'Pawn',
  knight: 'Knight',
  bishop: 'Bishop',
  rook: 'Rook',
  queen: 'Queen',
  king: 'King',
};

/**
 * Promotion dialog overlay - shows piece options when a pawn promotes.
 *
 * For white (promoting on rank 8): options go downward from rank 8
 * For black (promoting on rank 1): options go upward from rank 1
 */
export function PromotionDialog({
  file,
  color,
  boardSize,
  onSelect,
  onCancel,
}: PromotionDialogProps): JSX.Element {
  const squareSize = boardSize / 8;
  const fileIndex = FILE_TO_INDEX[file] ?? 0;

  // Calculate x position (from left)
  const x = fileIndex * squareSize;

  // For white (promoting on rank 8), start at top (y=0) and go down
  // For black (promoting on rank 1), start at bottom and go up
  const isWhite = color === 'white';
  const startY = isWhite ? 0 : boardSize - squareSize;
  const direction = isWhite ? 1 : -1;

  return (
    <>
      {/* Dark overlay that cancels on click */}
      <div className="promotion-overlay" onClick={onCancel} />

      {/* Piece options */}
      {PIECES.map((piece, index) => {
        const y = startY + index * squareSize * direction;

        return (
          <button
            key={piece.choice}
            type="button"
            className="promotion-choice"
            aria-label={`Promote to ${ROLE_NAME[piece.role]}`}
            onClick={e => {
              e.stopPropagation();
              onSelect(piece.choice);
            }}
            style={{
              left: x,
              top: y,
              width: squareSize,
              height: squareSize,
            }}
          >
            <div className="promotion-circle">
              <div className={`promotion-piece piece ${color} ${piece.role}`}>
                <ChessPieceArt role={piece.role} color={color} />
              </div>
            </div>
          </button>
        );
      })}
    </>
  );
}
