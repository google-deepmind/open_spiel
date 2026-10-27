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

// The checkers board's session: the frame and the checkers adapter's view
// and move hints, joined into the one record the board reads (the state,
// the seats and the moves). The board never reads the frame directly, so
// everything it is given is here, and the view's fields arrive as the
// adapter wrote them.

import {createContext, useContext} from 'react';

import {readableLabel} from '../../actions/history_rows.ts';
import type {Frame, Presentation} from '../../contract/contract.ts';
import {sessionModeOf} from '../../game/bottom_seat.ts';
import type {SquareCoord} from './checkers_model.ts';

/** OpenSpiel's player id for a finished game. */
const TERMINAL_PLAYER = -4;
/** OpenSpiel's player id for chance. */
const CHANCE_PLAYER = -1;

/** A legal action, with the game's label and the adapter's squares for it. */
export interface LegalAction {
  readonly action: number;
  readonly label: string;
  readonly fromSquare?: SquareCoord;
  readonly toSquare?: SquareCoord;
  readonly actionType?: string;
  readonly displayLabel?: string;
}

/** One seat at the table: its number, controller and whether it autoplays. */
export interface CheckersPlayer {
  readonly slot: number;
  readonly controller: string;
  readonly autoPlay: boolean;
}

/**
 * The position as the board reads it: the adapter's view (the struct and
 * the squares that may move, see checkers_adapter.py) with the frame's
 * moves, history and returns.
 */
export interface CheckersState {
  readonly struct?: unknown;
  readonly movableSources?: readonly SquareCoord[];
  /** The seat to move; -1 for chance and -4 once the game is over. */
  readonly currentPlayer: number;
  readonly isTerminal: boolean;
  readonly legalActions: readonly LegalAction[];
  readonly returns: readonly number[] | null;
  /** Each history entry's action id, or -1 where it is withheld. */
  readonly history: readonly number[];
  /** Each history entry's seat: -1 for chance. */
  readonly historyPlayers: readonly number[];
  /** Each history entry's label, as the viewer may read it. */
  readonly historyActionStrings: readonly string[];
}

/**
 * The session the board shows. `mode` is 'analysis' for the oracle, who may
 * move for either seat, and 'game' for everyone else.
 */
export interface SessionPayload {
  readonly sessionId: string;
  readonly mode: 'game' | 'analysis';
  readonly state: CheckersState;
  readonly players: readonly CheckersPlayer[];
}

/** The viewer: its seat, and whether it plays and is to move. */
export interface CheckersViewer {
  readonly seat: number | null;
  readonly isParticipant: boolean;
  readonly isActiveSeat: boolean;
}

/** What the board is given: the session, and how to move in it. */
export interface SessionContextValue {
  readonly session: SessionPayload;
  readonly error: string | null;
  readonly viewer: CheckersViewer;
  /** Whether a move is in flight, or the viewer cannot move here. */
  readonly isBusy: boolean;
  /** Plays one action. */
  readonly submitAction: (action: number) => Promise<void>;
}

function viewRecord(presentation: Presentation): Record<string, unknown> {
  const {view} = presentation;
  if (view === null || typeof view !== 'object' || Array.isArray(view)) {
    throw new Error('The checkers view is not an object.');
  }
  return Object.fromEntries(Object.entries(view));
}

function isSquare(value: unknown): value is SquareCoord {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof Reflect.get(value, 'row') === 'number' &&
    typeof Reflect.get(value, 'col') === 'number'
  );
}

function squares(value: unknown): SquareCoord[] {
  if (!Array.isArray(value)) {
    throw new Error('The checkers view has no list of movable squares.');
  }
  return value.map((item: unknown) => {
    if (!isSquare(item)) {
      throw new Error('A checkers movable square is not a {row, col}.');
    }
    return {row: item.row, col: item.col};
  });
}

function hintOf(hint: unknown): Omit<LegalAction, 'action' | 'label'> {
  if (hint === null || typeof hint !== 'object') return {};
  const from: unknown = Reflect.get(hint, 'fromSquare');
  const to: unknown = Reflect.get(hint, 'toSquare');
  const actionType: unknown = Reflect.get(hint, 'actionType');
  const displayLabel: unknown = Reflect.get(hint, 'displayLabel');
  return {
    ...(isSquare(from) ? {fromSquare: from} : {}),
    ...(isSquare(to) ? {toSquare: to} : {}),
    ...(typeof actionType === 'string' ? {actionType} : {}),
    ...(typeof displayLabel === 'string' ? {displayLabel} : {}),
  };
}

function legalActions(frame: Frame, presentation: Presentation): LegalAction[] {
  const hints = new Map(
    (presentation.actions ?? []).map(({id, hint}) => [id, hint]),
  );
  return (frame.legal_actions ?? []).map(({id, label}) => ({
    action: id,
    label,
    ...hintOf(hints.get(id)),
  }));
}

function currentPlayer(frame: Frame): number {
  const {to_move: toMove} = frame;
  if (toMove.kind === 'terminal') return TERMINAL_PLAYER;
  if (toMove.kind === 'chance') return CHANCE_PLAYER;
  if (toMove.seat === null) throw new Error('A seat to move has no seat.');
  return toMove.seat;
}

/** The session the board reads, built from `frame` and its presentation. */
export function checkersSession(
  frame: Frame,
  presentation: Presentation,
): SessionPayload {
  const view = viewRecord(presentation);
  const state: CheckersState = {
    struct: view['struct'],
    movableSources: squares(view['movableSources']),
    currentPlayer: currentPlayer(frame),
    isTerminal: frame.status === 'terminal',
    legalActions: legalActions(frame, presentation),
    returns: frame.returns,
    history: frame.history.map(entry => entry.action ?? -1),
    historyPlayers: frame.history.map(entry => entry.seat ?? -1),
    historyActionStrings: frame.history.map(
      entry => readableLabel(entry) ?? '',
    ),
  };
  return {
    sessionId: frame.session.id,
    mode: sessionModeOf(frame.viewer),
    state,
    players: frame.seats.map(seat => ({
      slot: seat.seat,
      controller: seat.controller,
      autoPlay: seat.autoplay,
    })),
  };
}

/** The viewer of `frame`, as the board names it. */
export function checkersViewer(frame: Frame): CheckersViewer {
  const {seat} = frame.viewer;
  const {to_move: toMove} = frame;
  return {
    seat,
    isParticipant: frame.viewer.kind === 'seat',
    isActiveSeat:
      seat !== null && toMove.kind === 'seat' && toMove.seat === seat,
  };
}

/** The board's session; provided by the checkers renderer. */
export const CheckersSessionContext = createContext<SessionContextValue | null>(
  null,
);

/** The session of the board a component is in. */
export function useCheckersSession(): SessionContextValue {
  const value = useContext(CheckersSessionContext);
  if (value === null) {
    throw new Error(
      'useCheckersSession is called outside the checkers renderer.',
    );
  }
  return value;
}
