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

// One square of the checkers board, with its man or king: the square's
// colour and marks (selected, a landing, the last turn, a taken man), and
// the piece, sliding in when it has just moved.

import type {CSSProperties, JSX} from 'react';

import {
  type PieceType,
  getPiecePlayer,
  isKing,
} from './checkers_board_parts.ts';

/** What one square draws. */
export interface CheckersSquareProps {
  row: number;
  col: number;
  piece: PieceType;
  isDark: boolean;
  isSelected: boolean;
  isDestination: boolean;
  isLastMove: boolean;
  wasCaptured: boolean;
  /** Whether the piece here slides in from the last leg's start. */
  slides: boolean;
  slideStyle: CSSProperties | undefined;
  /** Whether the piece here may be picked up by the side to move. */
  isMovable: boolean;
  /** Whether the side to move must capture: its movable men are ringed at rest. */
  mustCapture: boolean;
  currentPlayerSlot: number;
  historyLength: number;
  onSquareClick: (row: number, col: number) => void;
}

function squareClass(p: CheckersSquareProps): string {
  return `checkers-square${
    p.isDark ? ' checkers-square--dark' : ' checkers-square--light'
  }${p.isSelected ? ' checkers-square--selected' : ''}${
    p.isDestination ? ' checkers-square--destination' : ''
  }${p.isLastMove ? ' checkers-square--last-move' : ''}${
    p.wasCaptured ? ' checkers-square--captured' : ''
  }${p.slides ? ' checkers-square--slide-target' : ''}`;
}

/** A square of the board and the piece on it. */
export function CheckersSquare(p: CheckersSquareProps): JSX.Element {
  const {row, col, piece, slides, slideStyle, historyLength} = p;
  const piecePlayer = getPiecePlayer(piece);
  const pieceIsKing = isKing(piece);
  return (
    <div
      className={squareClass(p)}
      onClick={() => {
        p.onSquareClick(row, col);
      }}
      data-row={row}
      data-col={col}
    >
      {pieceIsKing && (
        // The lower man of a king: a real element
        // before the crowned one, so it stays
        // beneath it when the upper man is hovered
        // or sliding (each makes it a stacking
        // context, which a negative z-index cannot
        // get under).
        <div
          key={slides ? `base-slide-${String(historyLength)}` : 'base-rest'}
          aria-hidden="true"
          style={slideStyle}
          className={`checkers-king-base${
            piecePlayer === 0
              ? ' checkers-piece--black'
              : ' checkers-piece--white'
          }${slides ? ' checkers-piece--sliding' : ''}`}
        />
      )}
      {piece && (
        <div
          key={slides ? `slide-${String(historyLength)}` : 'rest'}
          style={slideStyle}
          className={`checkers-piece${
            piecePlayer === 0
              ? ' checkers-piece--black'
              : ' checkers-piece--white'
          }${pieceIsKing ? ' checkers-piece--king' : ''}${
            slides ? ' checkers-piece--sliding' : ''
          }${
            p.isMovable && piecePlayer === p.currentPlayerSlot
              ? ` checkers-piece--movable${
                  p.mustCapture ? ' checkers-piece--must-capture' : ''
                }`
              : ''
          }`}
        >
          {pieceIsKing && <span className="checkers-crown">♔</span>}
        </div>
      )}
    </div>
  );
}
