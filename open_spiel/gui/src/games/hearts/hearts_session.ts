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

// The hearts board's session: the frame and the hearts adapter's view, joined
// into the one record the board reads (the state, the seats, the viewer and
// the moves). The board never reads the frame directly, so everything it is
// given is here, and the view's fields arrive as the adapter wrote them.

import {createContext, useContext} from 'react';

import type {Frame, Presentation} from '../../contract/contract.ts';
import {sessionModeOf} from '../../game/bottom_seat.ts';

/** A legal action, with the game's label and the card it moves. */
export interface LegalAction {
  readonly action: number;
  readonly label: string;
  readonly cardCode?: string;
}

/** One seat at the table: its number, controller and whether it autoplays. */
export interface HeartsPlayer {
  readonly slot: number;
  readonly controller: string;
  readonly autoPlay: boolean;
  readonly label?: string;
}

/**
 * The position as the board reads it: the adapter's struct (see
 * hearts_adapter.py) with the frame's moves, history and rewards.
 */
export interface HeartsSessionState {
  readonly struct?: unknown;
  readonly currentPlayer: number;
  readonly turnNumber: number;
  readonly isTerminal: boolean;
  readonly legalActions: readonly LegalAction[];
  /** The final returns at the end, else the last action's rewards. */
  readonly rewards: readonly number[] | null;
  /** The actions applied to reach this position, one entry each. */
  readonly history: readonly number[];
}

/**
 * The session the board shows. `mode` is 'analysis' for the oracle, who sees
 * every hand and may move for any seat, and 'game' for everyone else.
 */
export interface SessionPayload {
  readonly sessionId: string;
  readonly mode: 'game' | 'analysis';
  readonly state: HeartsSessionState;
  readonly players: readonly HeartsPlayer[];
  readonly historyView?: boolean;
}

/** The viewer: its seat, and whether it plays and is on move. */
export interface ViewerContext {
  readonly seat: number | null;
  readonly viewSeat: number | null;
  readonly isParticipant: boolean;
  readonly isActiveSeat: boolean;
}

/** Whether an earlier position than the live one is shown. */
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
  readonly submitAction: (action: number) => Promise<void>;
  /** Plays `actions` in order, all or nothing from the board's side. */
  readonly submitAtomicSequence: (
    actions: readonly number[],
  ) => Promise<SessionPayload>;
}

function viewRecord(presentation: Presentation): Record<string, unknown> {
  const {view} = presentation;
  if (view === null || typeof view !== 'object' || Array.isArray(view)) {
    throw new Error('The hearts view is not an object.');
  }
  return Object.fromEntries(Object.entries(view));
}

function cardCodes(presentation: Presentation): Map<number, string> {
  const codes = new Map<number, string>();
  for (const {id, hint} of presentation.actions ?? []) {
    const code: unknown =
      hint !== null && typeof hint === 'object'
        ? Reflect.get(hint, 'cardCode')
        : undefined;
    if (typeof code === 'string') codes.set(id, code);
  }
  return codes;
}

function currentPlayer(frame: Frame): number {
  const {to_move: toMove} = frame;
  if (toMove.kind === 'seat' && toMove.seat !== null) return toMove.seat;
  return toMove.kind === 'chance' ? -1 : -4;
}

/** The session the board reads, built from `frame` and its presentation. */
export function heartsSession(
  frame: Frame,
  presentation: Presentation,
): SessionPayload {
  const view = viewRecord(presentation);
  const codes = cardCodes(presentation);
  const state: HeartsSessionState = {
    ...('struct' in view ? {struct: view['struct']} : {}),
    currentPlayer: currentPlayer(frame),
    turnNumber: frame.ply,
    isTerminal: frame.status === 'terminal',
    legalActions: (frame.legal_actions ?? []).map(({id, label}) => {
      const code = codes.get(id);
      return code === undefined
        ? {action: id, label}
        : {action: id, label, cardCode: code};
    }),
    rewards: frame.returns ?? frame.rewards,
    history: frame.history.slice(0, frame.ply).map(entry => entry.action ?? -1),
  };
  return {
    sessionId: frame.session.id,
    mode: sessionModeOf(frame.viewer),
    state,
    players: frame.seats.map(seat => ({
      slot: seat.seat,
      controller: seat.controller,
      autoPlay: seat.autoplay,
      // The seat's label is "player <n>" ("player 0"): the
      // table names seats by where they sit, never by who plays them.
      label: `player ${String(seat.seat)}`,
    })),
  };
}

/** The viewer of `frame`, as the board names it. */
export function heartsViewer(frame: Frame): ViewerContext {
  const {seat} = frame.viewer;
  const {to_move: toMove} = frame;
  const isParticipant = frame.viewer.kind === 'seat' && seat !== null;
  return {
    seat,
    viewSeat: seat,
    isParticipant,
    isActiveSeat:
      isParticipant && toMove.kind === 'seat' && toMove.seat === seat,
  };
}

/** Whether `frame` is a replay of an earlier position. */
export function heartsReplay(frame: Frame): ReplayState {
  return {
    isActive: frame.ply !== frame.num_plies,
    step: frame.ply,
    totalSteps: frame.num_plies,
  };
}

/** Whether a seat's controller is a person at the table. */
export function isHumanController(
  controller?: string | null,
  slot?: number,
): boolean {
  if (!controller) return slot === 0;
  return controller === 'human';
}

/** The board's session; provided by the hearts renderer. */
export const HeartsSessionContext = createContext<SessionContextValue | null>(
  null,
);

/** The session of the board a component is in. */
export function useHeartsSession(): SessionContextValue {
  const value = useContext(HeartsSessionContext);
  if (value === null) {
    throw new Error('useHeartsSession is called outside the hearts renderer.');
  }
  return value;
}
