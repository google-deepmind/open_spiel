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

// The poker board's session: the frame and the poker adapter's view, joined
// into the one record the board reads (the state, the seats and the moves).
// The board never reads the frame directly, so everything it is given is
// here, and the view's fields arrive as the adapter wrote them
// (adapters/poker_table.py).

import {createContext, useContext} from 'react';

import {readableLabel} from '../../actions/history_rows.ts';
import type {Frame, Presentation} from '../../contract/contract.ts';
import {sessionModeOf} from '../../game/bottom_seat.ts';

/** A legal action at the position shown, with the board's label for it. */
export interface LegalAction {
  readonly action: number;
  readonly label: string;
}

/** One seat at the table: its number, its name and its controller. */
export interface PokerPlayer {
  readonly slot: number;
  readonly label: string;
  readonly controller: string;
}

/**
 * What the board needs of the game beyond the struct: its title, its betting
 * ("limit" has one bet size, so the board offers no sizing), the hole cards a
 * seat holds, and the public cards turned before each round.
 */
export interface PokerGame {
  readonly title: string;
  readonly betting: 'limit' | 'nolimit';
  readonly holeCards: number;
  readonly boardSizes: readonly number[];
}

/**
 * The position as the board reads it: the adapter's struct and history
 * labels, with the frame's moves and returns.
 */
export interface PokerState {
  readonly struct?: unknown;
  readonly currentPlayer: number;
  readonly isTerminal: boolean;
  readonly legalActions: readonly LegalAction[];
  readonly returns: number[] | null;
  /** Each history entry's seat: -1 for chance. */
  readonly historyPlayers: readonly number[];
  /** Each history entry's label in the board's words. */
  readonly historyActionStrings: readonly string[];
}

/**
 * The session the board shows. `mode` is 'analysis' for the oracle, who sees
 * every hand and may move for any seat, and 'game' for everyone else.
 */
export interface SessionPayload {
  readonly sessionId: string;
  readonly mode: 'game' | 'analysis';
  readonly game: PokerGame;
  readonly state: PokerState;
  readonly players: readonly PokerPlayer[];
}

/** The viewer: its seat, and whether it may move now. */
export interface PokerViewer {
  readonly participantId: string | null;
  readonly canAct: boolean;
}

/** How a new hand is dealt: at the same table, or with `seats` seats. */
export interface NewHand {
  readonly seats: number;
}

/** What the board is given: the session, and how to move in it. */
export interface SessionContextValue {
  readonly status: 'ready';
  readonly session: SessionPayload;
  readonly error: string | null;
  readonly viewer: PokerViewer;
  /** Whether a move is in flight, or the viewer cannot move here. */
  readonly isBusy: boolean;
  /** Plays one action. */
  readonly submitAction: (action: number) => Promise<void>;
  /** Deals a new hand. */
  readonly createSession: (hand: NewHand) => void;
}

function viewRecord(presentation: Presentation): Record<string, unknown> {
  const {view} = presentation;
  if (view === null || typeof view !== 'object' || Array.isArray(view)) {
    throw new Error('The poker view is not an object.');
  }
  return Object.fromEntries(Object.entries(view));
}

function pokerGame(value: unknown): PokerGame {
  if (value === null || typeof value !== 'object') {
    throw new Error('The poker view has no pokerGame.');
  }
  const game = value as Record<string, unknown>;
  const {title, betting, holeCards, boardSizes} = game;
  if (
    typeof title !== 'string' ||
    (betting !== 'limit' && betting !== 'nolimit') ||
    typeof holeCards !== 'number' ||
    !Array.isArray(boardSizes) ||
    !boardSizes.every(size => typeof size === 'number')
  ) {
    throw new Error('The poker view has a malformed pokerGame.');
  }
  return {title, betting, holeCards, boardSizes};
}

function historyLabels(frame: Frame, view: Record<string, unknown>): string[] {
  const labels = view['historyActionStrings'];
  if (!Array.isArray(labels)) {
    return frame.history.map(entry => readableLabel(entry) ?? '');
  }
  if (labels.length !== frame.history.length) {
    throw new Error(
      `The poker view labels ${String(labels.length)} history entries, ` +
        `not the frame's ${String(frame.history.length)}.`,
    );
  }
  return labels.map(label => String(label));
}

// The board's label of each legal action: the adapter's hint, which names
// the action in the board's words, else the game's own label.
function legalActions(frame: Frame, presentation: Presentation): LegalAction[] {
  const hints = new Map<number, unknown>(
    (presentation.actions ?? []).map(({id, hint}) => [id, hint]),
  );
  return (frame.legal_actions ?? []).map(({id, label}) => {
    const hint = hints.get(id);
    const named =
      hint !== null && typeof hint === 'object' && 'label' in hint
        ? hint.label
        : undefined;
    return {action: id, label: typeof named === 'string' ? named : label};
  });
}

function currentPlayer(frame: Frame): number {
  if (frame.status === 'terminal') return -4;
  const {to_move: toMove} = frame;
  if (toMove.kind === 'chance') return -1;
  return toMove.seat ?? -1;
}

/** The session the board reads, built from `frame` and its presentation. */
export function pokerSession(
  frame: Frame,
  presentation: Presentation,
): SessionPayload {
  const view = viewRecord(presentation);
  const state: PokerState = {
    struct: view['struct'],
    currentPlayer: currentPlayer(frame),
    isTerminal: frame.status === 'terminal',
    legalActions: legalActions(frame, presentation),
    returns: frame.returns === null ? null : [...frame.returns],
    historyPlayers: frame.history.map(entry => entry.seat ?? -1),
    historyActionStrings: historyLabels(frame, view),
  };
  return {
    sessionId: frame.session.id,
    mode: sessionModeOf(frame.viewer),
    game: pokerGame(view['pokerGame']),
    state,
    players: frame.seats.map(seat => ({
      slot: seat.seat,
      label: `player ${String(seat.seat)}`,
      controller: seat.controller,
    })),
  };
}

/** The board's session; provided by the poker renderer. */
export const PokerSessionContext = createContext<SessionContextValue | null>(
  null,
);

/** The session of the board a component is in. */
export function usePokerSession(): SessionContextValue {
  const value = useContext(PokerSessionContext);
  if (value === null) {
    throw new Error('usePokerSession is called outside the poker renderer.');
  }
  return value;
}
