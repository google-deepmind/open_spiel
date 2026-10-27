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

// The checkers Gameplay panel's content: the board for the session the
// renderer provides (checkers_session.ts), its status row and the notices
// over it. Its parts: checkers_board_parts.ts (the struct and squares),
// checkers_gameplay_state.ts (the last turn, whose turn, the moves),
// checkers_square.tsx (one square) and checkers_status_row.tsx.

import type {CSSProperties, JSX, ReactNode} from 'react';
import {useMemo} from 'react';

import {useAnalysisViewSeat} from '../../game/bottom_seat.ts';

import {
  type PieceType,
  boardDimensions,
  buildBoardFromStruct,
  coordinateLabels,
  readStruct,
  squareKey,
} from './checkers_board_parts.ts';
import {
  useBoardTheme,
  useHighlightLegalMoves,
  useShowCoordinates,
} from './checkers_config_section.tsx';
import {
  type CheckersMoves,
  type CheckersTurn,
  type LastTurnMarks,
  checkersTurn,
  useCheckersMoves,
  useLastTurn,
} from './checkers_gameplay_state.ts';
import {isPlayableSquare} from './checkers_model.ts';
import {useCheckersSession} from './checkers_session.ts';
import {CheckersSquare} from './checkers_square.tsx';
import {CheckersStatusRow} from './checkers_status_row.tsx';

import './checkers.css';

export {boardDimensions, coordinateLabels} from './checkers_board_parts.ts';

/** A notice over the board: an error the board cannot draw past. */
function GameplayNotice({
  children,
}: {
  readonly tone: 'error';
  readonly children: ReactNode;
}): JSX.Element {
  return (
    <div className="checkers-notice" role="alert">
      {children}
    </div>
  );
}

/** Everything one square of the board needs to draw itself. */
interface BoardModel {
  board: PieceType[][];
  boardRows: number;
  boardColumns: number;
  shouldFlip: boolean;
  highlightLegalMoves: boolean;
  marks: LastTurnMarks;
  moves: CheckersMoves;
  turn: CheckersTurn;
}

function slideStyleOf(
  m: BoardModel,
  row: number,
  col: number,
): {slides: boolean; slideStyle: CSSProperties | undefined} {
  const {slideLeg} = m.marks;
  const slides =
    slideLeg !== null && slideLeg.to.row === row && slideLeg.to.col === col;
  // The board is drawn flipped for seat 1, so the
  // offset back to the start square flips with it.
  const flipSign = m.shouldFlip ? -1 : 1;
  if (!slides) return {slides, slideStyle: undefined};
  const vars: Record<string, string> = {
    '--slide-dx': String((slideLeg.from.col - col) * flipSign),
    '--slide-dy': String((row - slideLeg.from.row) * flipSign),
  };
  return {slides, slideStyle: vars};
}

function renderSquare(m: BoardModel, row: number, col: number): JSX.Element {
  const piece = m.board[row]?.[col] ?? '';
  const key = squareKey(row, col);
  const {selectedSquare, destinationMap, movableSources} = m.moves;
  return (
    <CheckersSquare
      key={`${String(row)}-${String(col)}`}
      row={row}
      col={col}
      piece={piece}
      isDark={isPlayableSquare(row, col, m.boardRows)}
      isLastMove={m.marks.lastPath.has(key)}
      wasCaptured={m.marks.lastCaptured.has(key) && !piece}
      {...slideStyleOf(m, row, col)}
      isSelected={selectedSquare?.row === row && selectedSquare.col === col}
      isDestination={m.highlightLegalMoves && destinationMap.has(key)}
      isMovable={movableSources.has(key) && m.turn.canInteract}
      mustCapture={m.moves.mustCapture}
      currentPlayerSlot={m.turn.currentPlayerSlot}
      historyLength={m.marks.historyLength}
      onSquareClick={m.moves.onSquareClick}
    />
  );
}

function BoardGrid({
  m,
  boardTheme,
}: {
  readonly m: BoardModel;
  readonly boardTheme: string;
}): JSX.Element {
  const {boardRows, boardColumns, shouldFlip} = m;
  return (
    <div
      className={`checkers-board checkers-board--${boardTheme}`}
      data-testid="checkers-board"
      style={{
        gridTemplateColumns: `repeat(${String(boardColumns)}, 1fr)`,
        gridTemplateRows: `repeat(${String(boardRows)}, 1fr)`,
        aspectRatio: `${String(boardColumns)} / ${String(boardRows)}`,
      }}
    >
      {/* Render rows - flip order based on perspective */}
      {Array.from({length: boardRows}, (_, i) =>
        shouldFlip ? i : boardRows - 1 - i,
      ).map(row =>
        Array.from({length: boardColumns}, (_, i) =>
          shouldFlip ? boardColumns - 1 - i : i,
        ).map(col => renderSquare(m, row, col)),
      )}
    </div>
  );
}

function FileCoords({
  files,
  edge,
}: {
  readonly files: readonly string[];
  readonly edge: 'top' | 'bottom';
}): JSX.Element {
  return (
    <div className={`checkers-coords checkers-coords--${edge}`}>
      <span className="checkers-coords-corner" />
      {files.map(col => (
        <span key={col}>{col}</span>
      ))}
      <span className="checkers-coords-corner" />
    </div>
  );
}

function RankCoords({
  ranks,
  edge,
}: {
  readonly ranks: readonly number[];
  readonly edge: 'left' | 'right';
}): JSX.Element {
  return (
    <div className={`checkers-coords checkers-coords--${edge}`}>
      {ranks.map(row => (
        <span key={row}>{row}</span>
      ))}
    </div>
  );
}

function BoardWithCoords({m}: {readonly m: BoardModel}): JSX.Element {
  const [boardTheme] = useBoardTheme();
  const [showCoordinates] = useShowCoordinates();
  const {files, ranks} = coordinateLabels(
    m.boardRows,
    m.boardColumns,
    m.shouldFlip,
  );
  return (
    <div
      className={`checkers-board-wrapper checkers-board-wrapper--${boardTheme}`}
    >
      <div className="checkers-board-with-coords">
        {/* Top coordinates (column letters) */}
        {showCoordinates && <FileCoords files={files} edge="top" />}
        {/* Middle row: left coords + board + right coords */}
        <div className="checkers-board-row">
          {showCoordinates && <RankCoords ranks={ranks} edge="left" />}
          <BoardGrid m={m} boardTheme={boardTheme} />
          {showCoordinates && <RankCoords ranks={ranks} edge="right" />}
        </div>
        {/* Bottom coordinates (column letters) */}
        {showCoordinates && <FileCoords files={files} edge="bottom" />}
      </div>
    </div>
  );
}

function CheckersContent(): JSX.Element {
  const s = useCheckersSession();
  const {session, error, submitAction, viewer} = s;
  const [highlightLegalMoves] = useHighlightLegalMoves();
  // Seat selection for perspective (View › Perspective)
  const [analysisViewSeat] = useAnalysisViewSeat();
  // Determine effective seat (analysis mode uses selection, game mode uses actual seat)
  const isAnalysisMode = session.mode === 'analysis';
  const effectiveSeat = isAnalysisMode ? analysisViewSeat : viewer.seat;
  // Flip board when viewing from player 1's (white's) perspective
  const shouldFlip = effectiveSeat === 1;
  const {state} = session;
  const struct = useMemo(() => readStruct(state.struct), [state.struct]);
  const {rows: boardRows, columns: boardColumns} = boardDimensions(struct);
  const board = useMemo(
    () => buildBoardFromStruct(struct, boardRows, boardColumns),
    [struct, boardRows, boardColumns],
  );
  const marks = useLastTurn(state);
  const turn = checkersTurn(s);
  const moves = useCheckersMoves(state, board, turn, submitAction);
  const m: BoardModel = {
    board,
    boardRows,
    boardColumns,
    shouldFlip,
    highlightLegalMoves,
    marks,
    moves,
    turn,
  };
  return (
    <>
      {error && <GameplayNotice tone="error">{error}</GameplayNotice>}
      {marks.turnError && (
        <GameplayNotice tone="error">{marks.turnError}</GameplayNotice>
      )}
      <CheckersStatusRow state={state} turn={turn} viewer={viewer} />
      <div className="checkers-move-notice" role="status">
        {moves.notice}
      </div>
      <BoardWithCoords m={m} />
    </>
  );
}

/** The checkers Gameplay panel's content: the board for the session the
 * renderer provides (checkers_session.ts). */
export function CheckersGameplay(): JSX.Element {
  return (
    <div className="ogs-panel checkers-gameplay">
      <CheckersContent />
    </div>
  );
}
