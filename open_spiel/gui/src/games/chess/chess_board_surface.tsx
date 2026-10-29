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

// The chess board itself: the squares in the theme's colours, the
// coordinates, the last move, the king in check, the selected piece and its
// moves, and the pieces, which slide to a new square at the viewer's
// animation speed and can be pressed or dragged (chess_board_input.ts).

import {type CSSProperties, type JSX, useRef, useState} from 'react';

import {
  type BoardConfig,
  drawnAt,
  readPlacement,
} from './chess_board_config.ts';
import {type BoardInput, useBoardInput} from './chess_board_input.ts';
import {
  type Tracked,
  moved,
  placementKey,
  track,
  trackFresh,
} from './chess_board_pieces.ts';
import {ChessPieceArt, roleOfLetter} from './chess_pieces.tsx';

const FILE_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

/** Where a square is drawn, as a translation in square widths. */
function at(square: string, config: BoardConfig): CSSProperties {
  const {column, row} = drawnAt(square, config.orientation);
  return {
    transform: `translate(${String(column * 100)}%, ${String(row * 100)}%)`,
  };
}

function kingSquare(fen: string, side: 'white' | 'black'): string | null {
  const king = side === 'white' ? 'K' : 'k';
  for (const [square, letter] of readPlacement(fen)) {
    if (letter === king) return square;
  }
  return null;
}

/** The marks under the pieces: last move, check, selection, destinations. */
function Marks({
  config,
  input,
}: {
  readonly config: BoardConfig;
  readonly input: BoardInput;
}): JSX.Element {
  const placement = readPlacement(config.fen);
  const marks = new Map<string, string[]>();
  const mark = (square: string | null | undefined, name: string) => {
    if (square === null || square === undefined) return;
    marks.set(square, [...(marks.get(square) ?? []), name]);
  };
  if (config.highlight.lastMove && config.lastMove !== undefined) {
    mark(config.lastMove[0], 'last-move');
    mark(config.lastMove[1], 'last-move');
  }
  if (config.highlight.check && config.check !== undefined) {
    mark(kingSquare(config.fen, config.check), 'check');
  }
  mark(input.selected, 'selected');
  for (const dest of input.dests) {
    mark(dest, placement.has(dest) ? 'move-dest oc' : 'move-dest');
  }
  return (
    <>
      {[...marks.entries()].map(([square, names]) => (
        <div
          key={square}
          className={`square ${names.join(' ')}`}
          data-square={square}
          style={at(square, config)}
        />
      ))}
    </>
  );
}

/** The coordinates, ranks down the right edge and files along the bottom. */
function Coordinates(): JSX.Element {
  return (
    <>
      <div className="coords ranks">
        {['1', '2', '3', '4', '5', '6', '7', '8'].map(rank => (
          <div key={rank} className="coord">
            {rank}
          </div>
        ))}
      </div>
      <div className="coords files">
        {FILE_LETTERS.map(file => (
          <div key={file} className="coord">
            {file}
          </div>
        ))}
      </div>
    </>
  );
}

/** The ids that follow each piece from one position to the next. */
function useTracked(config: BoardConfig, input: BoardInput): Tracked {
  const base = readPlacement(config.fen);
  const placement =
    input.pending === null
      ? base
      : moved(base, input.pending.orig, input.pending.dest);
  const key = placementKey(placement);
  const [tracked, setTracked] = useState(() => trackFresh(placement));
  if (tracked.key !== key) {
    const next = track(tracked, placement);
    setTracked(next);
    return next;
  }
  return tracked;
}

/** The properties of the board. */
export interface ChessBoardSurfaceProps {
  readonly config: BoardConfig;
  /** The board's width and height, in pixels. */
  readonly size: number;
  /** Bumped when a move attempt settles; see chess_board_input.ts. */
  readonly resync: number;
}

/** The board: squares, marks, coordinates and pieces, played by pointer. */
export function ChessBoardSurface({
  config,
  size,
  resync,
}: ChessBoardSurfaceProps): JSX.Element {
  const boardRef = useRef<HTMLDivElement | null>(null);
  const input = useBoardInput(config, boardRef, size, resync);
  const tracked = useTracked(config, input);
  const {drag, pending} = input;
  const slide = config.animation.enabled ? config.animation.duration : 0;
  const square = size / 8;
  return (
    <div
      className={`cg-wrap orientation-${config.orientation}${config.viewOnly ? ' view-only' : ''}`}
      data-testid="chess-board"
      data-fen={config.fen}
    >
      <div className="cg-container">
        <div
          ref={boardRef}
          className="cg-board"
          onPointerDown={input.onPointerDown}
          onPointerMove={input.onPointerMove}
          onPointerUp={input.onPointerUp}
          onPointerCancel={input.onPointerCancel}
        >
          <Marks config={config} input={input} />
          {tracked.pieces.map(piece => {
            const dragged = drag?.active === true && drag.orig === piece.square;
            const dropped =
              pending !== null &&
              !pending.slides &&
              pending.dest === piece.square;
            const role = roleOfLetter(piece.letter);
            const color =
              piece.letter === piece.letter.toUpperCase() ? 'white' : 'black';
            const style: CSSProperties = dragged
              ? {
                  transform: `translate(${String(drag.x - square / 2)}px, ${String(drag.y - square / 2)}px)`,
                  transition: 'none',
                }
              : {
                  ...at(piece.square, config),
                  transition:
                    dropped || slide === 0
                      ? 'none'
                      : `transform ${String(slide)}ms ease`,
                };
            return (
              <div
                key={piece.id}
                className={`piece ${role} ${color}${dragged ? ' dragging' : ''}`}
                data-square={piece.square}
                style={style}
              >
                <ChessPieceArt role={role} color={color} />
              </div>
            );
          })}
          {drag?.active === true && config.draggable.showGhost && (
            <GhostPiece config={config} square={drag.orig} />
          )}
        </div>
        {config.coordinates && <Coordinates />}
      </div>
    </div>
  );
}

/** The faint copy of a dragged piece left on its square. */
function GhostPiece({
  config,
  square,
}: {
  readonly config: BoardConfig;
  readonly square: string;
}): JSX.Element | null {
  const letter = readPlacement(config.fen).get(square);
  if (letter === undefined) return null;
  const role = roleOfLetter(letter);
  const color = letter === letter.toUpperCase() ? 'white' : 'black';
  return (
    <div className={`piece ghost ${role} ${color}`} style={at(square, config)}>
      <ChessPieceArt role={role} color={color} />
    </div>
  );
}
