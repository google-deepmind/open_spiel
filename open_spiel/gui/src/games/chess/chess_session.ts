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

// STRUCT-GLUE(chess ObservationStruct): the board reads the chess adapter's
// view (chess_adapter.py), which stands in for the struct chess does not
// have yet.
//
// The chess board's session: the frame and the chess adapter's view, joined
// into the one record the board reads (the position, the seats and the
// moves), under the names the board's components use. The board never reads
// the frame directly.

import {createContext, useContext} from 'react';

import {readableLabel} from '../../actions/history_rows.ts';
import type {Frame, Presentation} from '../../contract/contract.ts';
import {sessionModeOf} from '../../game/bottom_seat.ts';
import {readHints} from '../../renderers/hints.ts';
import {
  boolean,
  field,
  nullable,
  optionalField,
  record,
  string,
} from '../../renderers/struct_reader.ts';

/** A side of the board. */
export type Side = 'white' | 'black';

/** A move between two squares, as the board draws it: "e2" to "e4". */
export interface MoveSquares {
  readonly from: string;
  readonly to: string;
}

/** A legal action, its label, and the squares it moves between. */
export interface LegalAction {
  readonly action: number;
  readonly label: string;
}

/** A legal move's squares and promotion piece, by action. */
export interface MoveHint extends MoveSquares {
  readonly action: number;
  /** The promotion piece, a FEN letter ("q"), or null. */
  readonly promotion: string | null;
}

/** How a finished game ended, as the position says. */
export interface ChessOutcome {
  readonly checkmate: boolean;
  readonly stalemate: boolean;
  readonly threefold: boolean;
}

/** The position as the board reads it. */
export interface ChessState {
  /** The position's FEN. */
  readonly text: string;
  readonly currentPlayer: number;
  readonly turnNumber: number;
  readonly isTerminal: boolean;
  readonly returns: readonly number[] | null;
  readonly legalActions: readonly LegalAction[];
  /** Each move of the game, in the engine's notation (SAN). */
  readonly historyActionStrings: readonly string[];
  /** The side in check, or null. */
  readonly check: Side | null;
  readonly lastMove: MoveSquares | null;
  readonly outcome: ChessOutcome;
  /** The squares of each legal move the viewer may play. */
  readonly moveHints: readonly MoveHint[];
}

/** One seat at the table. */
export interface ChessPlayer {
  readonly slot: number;
  readonly controller: string;
  readonly autoPlay: boolean;
  readonly displayName: string | null;
}

/** A controller a seat may be given, by name and label. */
export interface ControllerOption {
  readonly id: string;
  readonly label: string;
}

/**
 * The session the board shows. `mode` is 'analysis' for the oracle, who may
 * move for either seat and picks the side the board is seen from, and
 * 'game' for everyone else.
 */
export interface SessionPayload {
  readonly mode: 'game' | 'analysis';
  readonly status: 'playing' | 'paused' | 'terminal';
  readonly state: ChessState;
  readonly players: readonly ChessPlayer[];
  readonly availableControllers: readonly ControllerOption[];
}

/** The viewer: its seat, and whether it plays and is to move. */
export interface ChessViewer {
  readonly seat: number | null;
  readonly isParticipant: boolean;
  readonly isActiveSeat: boolean;
}

/** What the board is given: the session, and how to move in it. */
export interface SessionContextValue {
  readonly status: 'ready';
  readonly session: SessionPayload;
  readonly error: string | null;
  readonly viewer: ChessViewer;
  /** Whether a move is in flight, or the viewer cannot move here. */
  readonly isBusy: boolean;
  /** Plays one action. */
  readonly submitAction: (action: number) => Promise<void>;
}

function readSide(value: unknown, path: string): Side {
  const side = string(value, path);
  if (side !== 'white' && side !== 'black') {
    throw new Error(`${path} is "${side}", not white or black.`);
  }
  return side;
}

function readSquares(value: unknown, path: string): MoveSquares {
  return {
    from: field(value, path, 'from', string),
    to: field(value, path, 'to', string),
  };
}

function readOutcome(value: unknown, path: string): ChessOutcome {
  return {
    checkmate: field(value, path, 'checkmate', boolean),
    stalemate: field(value, path, 'stalemate', boolean),
    threefold: field(value, path, 'threefold', boolean),
  };
}

/** The squares and promotion of each hinted move of `presentation`. */
export function readMoveHints(presentation: Presentation): MoveHint[] {
  return readHints(presentation, (hint, path) =>
    hint === null
      ? null
      : {
          ...readSquares(hint, path),
          promotion: optionalField<string | null>(
            hint,
            path,
            'promotion',
            string,
            null,
          ),
        },
  ).map(({action, hint}) => ({action, ...hint}));
}

function readState(frame: Frame, presentation: Presentation): ChessState {
  const view = record(presentation.view, 'view');
  return {
    text: field(view, 'view', 'fen', string),
    currentPlayer:
      frame.to_move.kind === 'seat' ? (frame.to_move.seat ?? -1) : -1,
    turnNumber: frame.ply,
    isTerminal: frame.status === 'terminal',
    returns: frame.returns,
    legalActions: (frame.legal_actions ?? []).map(({id, label}) => ({
      action: id,
      label,
    })),
    // The moves up to the position shown: a replay shows an earlier one.
    historyActionStrings: frame.history
      .slice(0, frame.ply)
      .map(entry => readableLabel(entry) ?? ''),
    check: field(view, 'view', 'chessCheck', (value, path) =>
      nullable(value, path, readSide),
    ),
    lastMove: field(view, 'view', 'chessLastMove', (value, path) =>
      nullable(value, path, readSquares),
    ),
    outcome: field(view, 'view', 'chessOutcome', readOutcome),
    moveHints: readMoveHints(presentation),
  };
}

/** The session the board reads, built from `frame` and its presentation. */
export function chessSession(
  frame: Frame,
  presentation: Presentation,
): SessionPayload {
  return {
    mode: sessionModeOf(frame.viewer),
    status: frame.status,
    state: readState(frame, presentation),
    players: frame.seats.map(seat => ({
      slot: seat.seat,
      controller: seat.controller,
      autoPlay: seat.autoplay,
      displayName: null,
    })),
    availableControllers: frame.session.controllers.map(({name, label}) => ({
      id: name,
      label,
    })),
  };
}

/** The viewer of `frame`, as the board names it. */
export function chessViewer(frame: Frame): ChessViewer {
  const {seat} = frame.viewer;
  return {
    seat,
    isParticipant: seat !== null,
    isActiveSeat:
      seat !== null &&
      frame.to_move.kind === 'seat' &&
      frame.to_move.seat === seat,
  };
}

/** The board's session; provided by the chess renderer. */
export const ChessSessionContext = createContext<SessionContextValue | null>(
  null,
);

/** The session of the board a component is in. */
export function useChessSession(): SessionContextValue {
  const value = useContext(ChessSessionContext);
  if (value === null) {
    throw new Error('useChessSession is called outside the chess renderer.');
  }
  return value;
}
