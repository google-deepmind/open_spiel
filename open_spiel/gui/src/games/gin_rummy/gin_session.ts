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

// The gin board's session: the frame and the gin_rummy adapter's view, joined
// into the one record the board reads (the state, the seats and the moves).
// The board never reads the frame directly, so everything it is given is
// here, and the view's fields arrive as the adapter wrote them.

import {createContext, useContext} from 'react';

import {readableLabel} from '../../actions/history_rows.ts';
import type {Frame, Presentation} from '../../contract/contract.ts';
import {sessionModeOf} from '../../game/bottom_seat.ts';

/** A legal action at the position shown, with the game's label for it. */
export interface LegalAction {
  readonly action: number;
  readonly label: string;
}

/** One seat at the table: its number, controller and whether it autoplays. */
export interface GinPlayer {
  readonly slot: number;
  readonly controller: string;
  readonly autoPlay: boolean;
}

/**
 * The position as the board reads it: the adapter's view (the struct and its
 * analysis, see gin_rummy_adapter.py) with the frame's moves, history and
 * returns. A field is absent when the adapter did not send it.
 */
export interface GinState {
  readonly struct?: unknown;
  readonly observationStruct?: unknown;
  readonly ginPhase?: unknown;
  readonly ginKnocked?: unknown;
  readonly ginRevealHands?: unknown;
  readonly ginVisibleHands?: unknown;
  readonly ginLayedMelds?: unknown;
  readonly ginAnalysisMeldsBySeat?: unknown;
  readonly ginAnalysisBestIndicesBySeat?: unknown;
  readonly ginLaydownSeat?: unknown;
  readonly ginLaydownStage?: unknown;
  readonly ginLaydownPassAction?: unknown;
  readonly ginLaydownPlans?: unknown;
  readonly knownCardsBySeat?: unknown;
  readonly currentPlayerLabel: string | null;
  readonly text: string;
  readonly lastAction: string | null;
  readonly turnNumber: number;
  readonly isTerminal: boolean;
  readonly legalActions: readonly LegalAction[];
  readonly returns: readonly number[] | null;
  /** Each history entry's seat: -1 for chance. */
  readonly historyPlayers: readonly number[];
  /** Each history entry's label, as the viewer may read it. */
  readonly historyActionStrings: readonly string[];
}

/**
 * The session the board shows. `mode` is 'analysis' for the oracle, who sees
 * every hand and may move for either seat, and 'game' for everyone else.
 */
export interface SessionPayload {
  readonly sessionId: string;
  readonly mode: 'game' | 'analysis';
  readonly state: GinState;
  readonly players: readonly GinPlayer[];
}

/** The viewer: its seat (null for a spectator or the oracle). */
export interface GinViewer {
  readonly seat: number | null;
  readonly viewSeat: number | null;
  readonly controller: string | null;
}

/** What the board is given: the session, and how to move in it. */
export interface SessionContextValue {
  readonly session: SessionPayload;
  readonly error: string | null;
  readonly viewer: GinViewer;
  /** Whether a move is in flight, or the viewer cannot move here. */
  readonly isBusy: boolean;
  /** Plays one action. */
  readonly submitAction: (action: number) => Promise<void>;
  /**
   * Plays `actions` in order, all or nothing from the board's side: the
   * session after the last, or a rejection naming the first that failed.
   */
  readonly submitAtomicSequence: (
    actions: readonly number[],
  ) => Promise<SessionPayload>;
}

const VIEW_FIELDS = [
  'struct',
  'observationStruct',
  'ginPhase',
  'ginKnocked',
  'ginRevealHands',
  'ginVisibleHands',
  'ginLayedMelds',
  'ginAnalysisMeldsBySeat',
  'ginAnalysisBestIndicesBySeat',
  'ginLaydownSeat',
  'ginLaydownStage',
  'ginLaydownPassAction',
  'ginLaydownPlans',
  'knownCardsBySeat',
] as const;

function viewRecord(presentation: Presentation): Record<string, unknown> {
  const {view} = presentation;
  if (view === null || typeof view !== 'object' || Array.isArray(view)) {
    throw new Error('The gin_rummy view is not an object.');
  }
  return Object.fromEntries(Object.entries(view));
}

function currentPlayerLabel(frame: Frame): string | null {
  const {to_move: toMove} = frame;
  if (toMove.kind === 'chance') return 'Chance';
  if (toMove.seat === null) return null;
  return `Player_${String(toMove.seat)}`;
}

function lastAction(frame: Frame): string | null {
  const last = frame.history.at(-1);
  return last === undefined ? null : readableLabel(last);
}

/** The session the board reads, built from `frame` and its presentation. */
export function ginSession(
  frame: Frame,
  presentation: Presentation,
): SessionPayload {
  const view = viewRecord(presentation);
  const fields: Record<string, unknown> = {};
  for (const name of VIEW_FIELDS) {
    if (name in view) fields[name] = view[name];
  }
  const state: GinState = {
    ...fields,
    currentPlayerLabel: currentPlayerLabel(frame),
    text: frame.observation_text ?? frame.state_text ?? '',
    lastAction: lastAction(frame),
    turnNumber: frame.ply,
    isTerminal: frame.status === 'terminal',
    legalActions: (frame.legal_actions ?? []).map(({id, label}) => ({
      action: id,
      label,
    })),
    returns: frame.returns,
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
export function ginViewer(frame: Frame): GinViewer {
  const {seat} = frame.viewer;
  const controller =
    seat === null
      ? null
      : (frame.seats.find(entry => entry.seat === seat)?.controller ?? null);
  return {seat, viewSeat: seat, controller};
}

/** The board's session; provided by the gin_rummy renderer. */
export const GinSessionContext = createContext<SessionContextValue | null>(
  null,
);

/** The session of the board a component is in. */
export function useGinSession(): SessionContextValue {
  const value = useContext(GinSessionContext);
  if (value === null) {
    throw new Error('useGinSession is called outside the gin_rummy renderer.');
  }
  return value;
}
