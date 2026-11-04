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

// The euchre board's session: the frame and the euchre adapter's view,
// joined into the one record the board reads (the state, the seats, the
// moves and the viewer). The board never reads the frame directly, so
// everything it is given is here, and the view's fields arrive as the
// adapter wrote them (euchre_adapter.py).

import {createContext, useContext} from 'react';

import {readableLabel} from '../../actions/history_rows.ts';
import type {Frame, Presentation} from '../../contract/contract.ts';
import {sessionModeOf} from '../../game/bottom_seat.ts';

/**
 * A legal action at the position shown: the game's label for it and the
 * adapter's description (its card, its kind and the suit it calls).
 */
export interface LegalAction {
  readonly action: number;
  readonly label: string;
  readonly cardCode?: string | null;
  readonly actionType?: string;
  readonly calledSuit?: string | null;
  readonly displayLabel?: string;
}

/** One seat at the table: its number, controller and whether it autoplays. */
export interface EuchrePlayer {
  readonly slot: number;
  readonly controller: string;
  readonly autoPlay: boolean;
}

/**
 * The position as the board reads it: the adapter's struct and the seat to
 * act, with the frame's moves, history and returns.
 */
export interface EuchreState {
  readonly struct?: unknown;
  readonly currentPosition?: unknown;
  readonly currentPlayer: number;
  readonly isTerminal: boolean;
  readonly legalActions: readonly LegalAction[];
  readonly returns: readonly number[] | null;
  /** Every action of the game, as the viewer may read it. */
  readonly history: readonly string[];
  /** Each history entry's seat: -1 for chance. */
  readonly historyPlayers: readonly number[];
}

/**
 * The session the board shows. `mode` is 'analysis' for the oracle, who sees
 * every hand and may move for any seat, and 'game' for everyone else.
 */
export interface SessionPayload {
  readonly sessionId: string;
  readonly mode: 'game' | 'analysis';
  readonly state: EuchreState;
  readonly players: readonly EuchrePlayer[];
  readonly gameParams: Readonly<Record<string, boolean | number | string>>;
  /** An earlier position of the game is on show, not the live one. */
  readonly historyView: boolean;
}

/** The viewer: its seat, and whether it plays and is to act. */
export interface ViewerContext {
  readonly seat: number | null;
  readonly viewSeat: number | null;
  readonly isParticipant: boolean;
  readonly isActiveSeat: boolean;
}

/** Whether a replay is on show, and at which step of how many. */
export interface ReplayState {
  readonly isActive: boolean;
  readonly step: number;
  readonly totalSteps: number;
}

/** What the board is given: the session, and how to move in it. */
export interface SessionContextValue {
  readonly session: SessionPayload;
  readonly error: string | null;
  readonly viewer: ViewerContext;
  readonly replay: ReplayState;
  /** Whether a move is in flight, or the viewer cannot move here. */
  readonly isBusy: boolean;
  /** Plays one action. */
  readonly submitAction: (action: number) => void;
}

function viewRecord(presentation: Presentation): Record<string, unknown> {
  const {view} = presentation;
  if (view === null || typeof view !== 'object' || Array.isArray(view)) {
    throw new Error('The euchre view is not an object.');
  }
  return Object.fromEntries(Object.entries(view));
}

function hintsById(presentation: Presentation): Map<number, object> {
  const hints = new Map<number, object>();
  for (const {id, hint} of presentation.actions ?? []) {
    if (hint === null || typeof hint !== 'object') {
      throw new Error(
        `The euchre hint for action ${String(id)} is not a record.`,
      );
    }
    hints.set(id, hint);
  }
  return hints;
}

function currentPlayer(frame: Frame): number {
  const {to_move: toMove} = frame;
  if (toMove.kind === 'chance') return -1;
  return toMove.seat ?? -1;
}

/** The session the board reads, built from `frame` and its presentation. */
export function euchreSession(
  frame: Frame,
  presentation: Presentation,
): SessionPayload {
  const view = viewRecord(presentation);
  const hints = hintsById(presentation);
  const state: EuchreState = {
    struct: view['struct'],
    currentPosition: view['currentPosition'],
    currentPlayer: currentPlayer(frame),
    isTerminal: frame.status === 'terminal',
    legalActions: (frame.legal_actions ?? []).map(({id, label}) => ({
      ...hints.get(id),
      action: id,
      label,
    })),
    returns: frame.returns,
    history: frame.history.map(entry => readableLabel(entry) ?? ''),
    historyPlayers: frame.history.map(entry => entry.seat ?? -1),
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
    gameParams: frame.session.parameters,
    historyView: frame.ply < frame.num_plies,
  };
}

/** The viewer of `frame`, as the board names it. */
export function euchreViewer(frame: Frame): ViewerContext {
  const {seat} = frame.viewer;
  const isParticipant = seat !== null;
  return {
    seat,
    viewSeat: seat,
    isParticipant,
    isActiveSeat: isParticipant && currentPlayer(frame) === seat,
  };
}

/** The replay the board is on: none, since the shell scrubs the frames. */
export function euchreReplay(frame: Frame): ReplayState {
  return {isActive: false, step: frame.ply, totalSteps: frame.num_plies};
}

/**
 * Whether `controller` is a person at the board. A seat with no controller
 * is a person only at seat 0, as a fresh table seats the viewer there.
 */
export function isHumanController(
  controller?: string | null,
  slot?: number,
): boolean {
  if (controller === undefined || controller === null || controller === '') {
    return slot === 0;
  }
  return controller === 'human' || controller === 'manual';
}

/** The board's session; provided by the euchre renderer. */
export const EuchreSessionContext = createContext<SessionContextValue | null>(
  null,
);

/** The session of the board a component is in. */
export function useEuchreSession(): SessionContextValue {
  const value = useContext(EuchreSessionContext);
  if (value === null) {
    throw new Error('useEuchreSession is called outside the euchre renderer.');
  }
  return value;
}
