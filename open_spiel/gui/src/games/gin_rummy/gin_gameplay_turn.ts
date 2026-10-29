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

// Whose turn the gin board shows and what the viewer may do: the knocker,
// each seat's controller, whether the viewer can act now (and why not), the
// last turn's cue, and the status line.

import {useMemo} from 'react';

import {
  ginHistorySteps,
  lastTurnCue,
  type GinBoardCue,
} from './gin_move_cues.ts';
import type {GinPlayer, SessionContextValue} from './gin_session.ts';
import {
  GIN_BLOCK_REASON,
  GIN_STATUS,
  GIN_STATUS_JOIN,
  ginToMove,
  ginWaitingFor,
} from './gin_turn_line.ts';
import type {GinView} from './gin_gameplay_view.ts';
import type {GinRummyDerivedState} from './use_gin_rummy_derived_state.ts';

/** Whether a seat's controller is a person. */
export function isHumanController(
  controller: string | null | undefined,
): boolean {
  return controller === 'human';
}

/** The seats that knocked, from the struct's knocked flags. */
export function knockerSeatsOf(raw: unknown): Set<number> {
  const seats = new Set<number>();
  if (Array.isArray(raw)) {
    raw.forEach((value: unknown, index) => {
      if (value === true) seats.add(index);
    });
  } else if (raw === true) {
    seats.add(0);
  }
  return seats;
}

/** Whose turn it is and what the viewer may do. */
export interface GinTurn {
  knockerSeats: Set<number>;
  viewerSeatForRole: number | null;
  viewerIsKnocker: boolean;
  viewerIsHuman: boolean;
  currentPlayerSlot: number;
  currentAutoPlay: boolean;
  waitingOnBot: boolean;
  viewerIsActiveSeat: boolean;
  canInteract: boolean;
  interactionBlockReason: string | null;
}

function playerAt(
  players: readonly GinPlayer[],
  slot: number | null,
): GinPlayer | undefined {
  return slot === null
    ? undefined
    : players.find(player => player.slot === slot);
}

/** The board's turn: its roles, controllers and the viewer's options. */
export function ginTurnOf(
  s: SessionContextValue,
  d: GinRummyDerivedState,
  view: GinView,
): GinTurn {
  const {players} = s.session;
  const {viewerSeatIndex, effectiveViewSeat, isAnalysisMode} = view;
  const knockerSeats = knockerSeatsOf(d.structRecord?.['knocked']);
  const viewerSeatForRole = isAnalysisMode
    ? effectiveViewSeat
    : viewerSeatIndex;
  const viewerController =
    playerAt(players, viewerSeatIndex)?.controller ?? s.viewer.controller;
  const viewerIsHuman = isHumanController(viewerController);
  const currentPlayerSlot = d.currentPlayerId ?? -1;
  const current =
    currentPlayerSlot >= 0 ? playerAt(players, currentPlayerSlot) : undefined;
  const currentHuman =
    currentPlayerSlot >= 0 && isHumanController(current?.controller);
  const viewerIsActiveSeat =
    viewerSeatIndex !== null &&
    currentPlayerSlot === viewerSeatIndex &&
    viewerIsHuman;
  const canInteract =
    viewerIsActiveSeat &&
    !s.isBusy &&
    !d.isTerminal &&
    viewerSeatIndex === effectiveViewSeat;
  return {
    knockerSeats,
    viewerSeatForRole,
    viewerIsKnocker:
      viewerSeatForRole !== null && knockerSeats.has(viewerSeatForRole),
    viewerIsHuman,
    currentPlayerSlot,
    currentAutoPlay: current?.autoPlay ?? false,
    waitingOnBot: !d.isTerminal && currentPlayerSlot >= 0 && !currentHuman,
    viewerIsActiveSeat,
    canInteract,
    interactionBlockReason: blockReason(s, d, view, {
      canInteract,
      viewerIsHuman,
      currentPlayerSlot,
    }),
  };
}

function blockReason(
  s: SessionContextValue,
  d: GinRummyDerivedState,
  view: GinView,
  t: {canInteract: boolean; viewerIsHuman: boolean; currentPlayerSlot: number},
): string | null {
  const {viewerSeatIndex, effectiveViewSeat, isAnalysisMode} = view;
  if (isAnalysisMode || t.canInteract) return null;
  if (d.isTerminal) return GIN_BLOCK_REASON.finished;
  if (s.isBusy) return GIN_BLOCK_REASON.busy;
  if (viewerSeatIndex === null || viewerSeatIndex < 0) {
    return GIN_BLOCK_REASON.noSeat;
  }
  if (s.session.mode !== 'game') return GIN_BLOCK_REASON.notGame;
  if (!t.viewerIsHuman) return GIN_BLOCK_REASON.botSeat;
  if (viewerSeatIndex !== effectiveViewSeat) return GIN_BLOCK_REASON.otherView;
  if (t.currentPlayerSlot !== viewerSeatIndex) {
    return ginWaitingFor(t.currentPlayerSlot);
  }
  return null;
}

/**
 * The last turn, for the board's resting-state cues. The viewer's own moves
 * are not cued while they play them; a seat a bot plays for them
 * (autoplay) is.
 */
export function useMoveCue(
  s: SessionContextValue,
  d: GinRummyDerivedState,
  view: GinView,
  turn: GinTurn,
): GinBoardCue | null {
  const {viewerSeatIndex} = view;
  const autoPlay =
    playerAt(s.session.players, viewerSeatIndex)?.autoPlay ?? false;
  const excludeSeat = turn.viewerIsHuman && !autoPlay ? viewerSeatIndex : null;
  const {historyPlayers, historyActionStrings} = s.session.state;
  const bottomSeat = d.playerHandSlot ?? view.effectiveViewSeat;
  const prevUpcard = d.prevUpcardNormalised;
  return useMemo((): GinBoardCue | null => {
    const steps = ginHistorySteps(historyPlayers, historyActionStrings);
    if (steps === null) return null;
    const cue = lastTurnCue(steps, {excludeSeat, prevUpcard});
    if (cue === null) return null;
    return {...cue, zone: cue.seat === bottomSeat ? 'player' : 'opponent'};
  }, [
    historyPlayers,
    historyActionStrings,
    excludeSeat,
    prevUpcard,
    bottomSeat,
  ]);
}

/**
 * The status line: the derived status with its first phrase replaced by
 * whose turn the board shows ("Your turn", "Opponent turn", a spectator's
 * "Player 1 to move").
 */
export function ginStatusText(
  d: GinRummyDerivedState,
  view: GinView,
  turn: GinTurn,
): string {
  const {statusText} = d;
  const segments =
    statusText === ''
      ? []
      : statusText.includes(GIN_STATUS_JOIN)
        ? statusText.split(GIN_STATUS_JOIN)
        : [statusText];
  const slot = turn.currentPlayerSlot;
  // A spectator has no turn: say whose it is, by seat.
  const turnText = d.isTerminal
    ? null
    : slot < 0
      ? GIN_STATUS.waiting
      : view.viewerSeatIndex === null && !view.isAnalysisMode
        ? ginToMove(`Player ${String(slot)}`)
        : slot === view.effectiveViewSeat
          ? GIN_STATUS.yourTurn
          : GIN_STATUS.opponentTurn;
  if (turnText !== null) {
    if (segments.length > 0) segments[0] = turnText;
    else segments.push(turnText);
  }
  return segments.length > 0
    ? segments.join(GIN_STATUS_JOIN)
    : GIN_STATUS.ready;
}
