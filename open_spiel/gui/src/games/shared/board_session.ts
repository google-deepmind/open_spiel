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

// The session a board panel reads (connect four, tic-tac-toe): the
// frame and its adapter's view, joined into the record the panels were
// written against (the state, the seats, the viewer and the move). The view's
// fields arrive under the names the adapter wrote them
// (connect_four_adapter.py, tic_tac_toe_adapter.py); the legal actions and
// seats come from the frame, as the server gave them to this viewer.

import {createContext, useContext} from 'react';

import type {Frame, Presentation} from '../../contract/contract.ts';
import {sessionModeOf} from '../../game/bottom_seat.ts';

/** A legal action at the position shown, with the game's label for it. */
export interface LegalAction {
  readonly action: number;
  readonly label: string;
}

/** The position as a board panel reads it. */
export interface OpenSpielState {
  /** Every action applied to reach the position, in order. */
  readonly history: readonly number[];
  /** The game's text for the history. */
  readonly historyStr: string;
  /** The seat to move; negative for chance and the terminal position. */
  readonly currentPlayer: number;
  /** Who is to move, in words ("player 0"), or "terminal". */
  readonly currentPlayerLabel: string;
  readonly isTerminal: boolean;
  readonly legalActions: readonly LegalAction[];
  /** Each seat's return at a terminal position; otherwise null. */
  readonly returns: readonly number[] | null;
  /** The number of actions applied. */
  readonly turnNumber: number;
}

/** One seat: its number, controller and whether the server autoplays it. */
export interface SessionPlayer {
  readonly slot: number;
  readonly controller: string;
  readonly autoPlay: boolean;
}

/**
 * The session a board shows. `mode` is 'analysis' for the oracle, who may
 * move for either seat, and 'game' for a seat or a spectator.
 */
export interface SessionPayload {
  readonly sessionId: string;
  readonly mode: 'game' | 'analysis';
  /** The game's parameters, defaults included, as the adapter sent them. */
  readonly gameParams: Record<string, unknown>;
  readonly state: OpenSpielState;
  readonly players: readonly SessionPlayer[];
}

/** The viewer: whether it holds a seat, and whether that seat is to move. */
export interface ViewerContext {
  readonly seat: number | null;
  readonly isParticipant: boolean;
  readonly isActiveSeat: boolean;
  readonly viewSeat: number | null;
}

/** What a board panel is given: the session, and how to move in it. */
export interface SessionContextValue {
  readonly session: SessionPayload;
  readonly error: string | null;
  readonly viewer: ViewerContext;
  /** Whether a move is in flight, or the viewer cannot move here. */
  readonly isBusy: boolean;
  /** Plays one action. */
  readonly submitAction: (action: number) => Promise<void>;
}

function record(value: unknown, what: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${what} is not an object.`);
  }
  return Object.fromEntries(Object.entries(value));
}

function integer(value: unknown, what: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new Error(`${what} is not an integer: ${JSON.stringify(value)}.`);
  }
  return value;
}

function text(value: unknown, what: string): string {
  if (typeof value !== 'string') throw new Error(`${what} is not a string.`);
  return value;
}

function numbers(value: unknown, what: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${what} is not a list.`);
  return value.map((item: unknown) => {
    if (typeof item !== 'number') {
      throw new Error(`${what} holds ${String(item)}.`);
    }
    return item;
  });
}

/** The session a board reads, built from `frame` and its presentation. */
export function boardSession(
  frame: Frame,
  presentation: Presentation,
): SessionPayload {
  const view = record(presentation.view, `The ${presentation.renderer} view`);
  const isTerminal = view['isTerminal'];
  if (typeof isTerminal !== 'boolean') {
    throw new Error('The view says no isTerminal.');
  }
  const returns = view['returns'];
  const state: OpenSpielState = {
    history: numbers(view['history'], 'history').map(action =>
      integer(action, 'An action'),
    ),
    historyStr: text(view['historyStr'], 'historyStr'),
    currentPlayer: integer(view['currentPlayer'], 'currentPlayer'),
    currentPlayerLabel: text(view['currentPlayerLabel'], 'currentPlayerLabel'),
    isTerminal,
    legalActions: (frame.legal_actions ?? []).map(({id, label}) => ({
      action: id,
      label,
    })),
    returns: returns === null ? null : numbers(returns, 'returns'),
    turnNumber: integer(view['turnNumber'], 'turnNumber'),
  };
  return {
    sessionId: frame.session.id,
    mode: sessionModeOf(frame.viewer),
    gameParams: record(view['gameParams'], 'gameParams'),
    state,
    players: frame.seats.map(seat => ({
      slot: seat.seat,
      controller: seat.controller,
      autoPlay: seat.autoplay,
    })),
  };
}

/** The viewer of `frame`, given the seat to move at `state`. */
export function boardViewer(
  frame: Frame,
  state: OpenSpielState,
): ViewerContext {
  const {seat} = frame.viewer;
  const isParticipant = seat !== null;
  return {
    seat,
    isParticipant,
    isActiveSeat: isParticipant && state.currentPlayer === seat,
    viewSeat: seat,
  };
}

/** The board's session; provided by the board's renderer. */
export const BoardSessionContext = createContext<SessionContextValue | null>(
  null,
);

/** The session of the board a component is in. */
export function useBoardSession(): SessionContextValue {
  const value = useContext(BoardSessionContext);
  if (value === null) {
    throw new Error('useBoardSession is called outside a board renderer.');
  }
  return value;
}
