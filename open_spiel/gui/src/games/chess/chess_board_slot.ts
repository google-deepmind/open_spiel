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

// The chess board slot: the props any chess board is drawn with, so that
// another board (a user-installed one, through an add-on) can draw chess in
// place of the built-in board. `chessRenderer` (chess_renderer.tsx) turns a
// board of this shape into the "chess" renderer; src/INTERFACES.md,
// "Replacing the chess board", is the guide.

import type {JSX} from 'react';

import type {Frame, Presentation} from '../../contract/contract.ts';
import {
  type ChessViewer,
  type MoveHint,
  type MoveSquares,
  type SessionPayload,
  type Side,
  chessSession,
  chessViewer,
} from './chess_session.ts';

export type {MoveHint, MoveSquares, Side} from './chess_session.ts';

/** The position a chess board draws, read from the server's frame. */
export interface ChessFrame {
  /** The position, as FEN. */
  readonly fen: string;
  /** The side to move, from the FEN. */
  readonly turn: Side;
  /**
   * The legal moves, each with the action that plays it: a promotion is one
   * move per piece, `promotion` its FEN letter ("q", "r", "b", "n"), else
   * null. Empty when the viewer is not given the legal actions (another
   * seat's turn, a spectator).
   */
  readonly moves: readonly MoveHint[];
  /** The last move's squares, or null before the first. */
  readonly lastMove: MoveSquares | null;
  /** The side in check, or null. */
  readonly check: Side | null;
  /** Whether the game is over (`session.state.outcome` says how). */
  readonly isTerminal: boolean;
  /** Plies played to the position shown (a replay shows an earlier one). */
  readonly ply: number;
  /** Everything the built-in board reads: the seats, the moves in SAN. */
  readonly session: SessionPayload;
  /** The frame as the server sent it. */
  readonly source: Frame;
}

/** Who looks at the board, and whether they may move now. */
export interface ChessBoardViewer extends ChessViewer {
  /** The viewer's colour: seat 0 plays black, seat 1 white; else null. */
  readonly side: Side | null;
  /**
   * The side to draw at the bottom: a seat's own side; for the oracle and a
   * spectator, the side View › Board › View from picks (white until one is
   * picked).
   */
  readonly orientation: Side;
  /** The oracle, who sees every seat and may move for either side. */
  readonly isOracle: boolean;
  /**
   * Whether `act` plays now: false on another seat's turn, for a spectator,
   * in a replay, in a finished game and while a move is in flight.
   */
  readonly canMove: boolean;
}

/**
 * Plays an action, one of `frame.moves`' `action`s, at the position shown;
 * resolves once it is handed to the session (not once the server takes it),
 * and rejects, sending nothing, when `viewer.canMove` is false. While the
 * move is in flight `viewer.canMove` is false. A move the server refuses
 * shows the session's error in the Gameplay panel and leaves `frame` as it
 * was; `viewer` is a new object each time a move goes in flight and each
 * time it settles, so a board that sets its pieces from `frame.fen`
 * whenever `frame` or `viewer` changes puts a refused move back.
 */
export type ChessAct = (action: number) => Promise<void>;

/** What a chess board is given: the position, the viewer, how to move. */
export interface ChessBoardProps {
  readonly frame: ChessFrame;
  readonly viewer: ChessBoardViewer;
  readonly act: ChessAct;
}

/** A component that draws a chess board and takes its moves. */
export type ChessBoardComponent = (props: ChessBoardProps) => JSX.Element;

/** The side to move in the FEN `fen`. */
function fenTurn(fen: string): Side {
  const field = fen.split(' ')[1];
  if (field === 'w') return 'white';
  if (field === 'b') return 'black';
  throw new Error(`FEN "${fen}" names no side to move.`);
}

/** The position of `frame`, as a chess board reads it. */
export function readChessFrame(
  frame: Frame,
  presentation: Presentation,
): ChessFrame {
  const session = chessSession(frame, presentation);
  const {state} = session;
  return {
    fen: state.text,
    turn: fenTurn(state.text),
    moves: state.moveHints,
    lastMove: state.lastMove,
    check: state.check,
    isTerminal: state.isTerminal,
    ply: state.turnNumber,
    session,
    source: frame,
  };
}

/**
 * The viewer of `frame`; `canMove` as the renderer was told, `viewFrom` the
 * side View › Board › View from picks, drawn at the bottom for a viewer
 * without a seat.
 */
export function readChessViewer(
  frame: Frame,
  canMove: boolean,
  viewFrom: Side = 'white',
): ChessBoardViewer {
  const viewer = chessViewer(frame);
  const side: Side | null =
    viewer.seat === null ? null : viewer.seat === 0 ? 'black' : 'white';
  return {
    ...viewer,
    side,
    orientation: side ?? viewFrom,
    isOracle: frame.viewer.kind === 'oracle',
    canMove,
  };
}

/**
 * The action that moves from `from` to `to` (a pawn promoting to
 * `promotion`, a FEN letter), or null when no legal move does.
 */
export function actionFor(
  frame: ChessFrame,
  from: string,
  to: string,
  promotion: string | null = null,
): number | null {
  const move = frame.moves.find(
    hint =>
      hint.from === from && hint.to === to && hint.promotion === promotion,
  );
  return move === undefined ? null : move.action;
}
