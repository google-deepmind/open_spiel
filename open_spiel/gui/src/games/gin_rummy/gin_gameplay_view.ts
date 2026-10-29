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

// Whose view the gin board shows: the viewer's seat, the seat at the foot
// of the table, whether hands are revealed, and in analysis which seat the
// board is seen from (View › Perspective).

import {useEffect, useMemo} from 'react';

import {useAnalysisViewSeat} from '../../game/bottom_seat.ts';
import {
  SHOW_KNOWLEDGE_KEY,
  SHOW_OPPONENT_KEY,
} from './gin_rummy_gameplay_config_extras.tsx';
import type {GinPlayer, SessionContextValue} from './gin_session.ts';
import {useViewSetting} from '../shared/view_settings.ts';
import type {GinRummyDerivedState} from './use_gin_rummy_derived_state.ts';

/** The seats the analysis view may show, kept for the View panel. */
export const SEAT_OPTIONS_KEY = 'gin.seatOptions';

/** Whose view the board shows. */
export interface GinView {
  viewerSeatIndex: number | null;
  opponentSeatIndex: number;
  isAnalysisMode: boolean;
  manualOpponentReveal: boolean;
  preferObservationStruct: boolean;
  effectiveViewSeat: number;
  analysisViewSeat: number | null;
  setAnalysisViewSeat: (seat: number | null) => void;
  showKnowledgeHints: boolean;
}

function visibleFlag(raw: unknown, seat: number): boolean {
  if (typeof raw !== 'object' || raw === null) return false;
  return Boolean(Reflect.get(raw, String(seat)));
}

/** The board's view of the session, from the viewer and the settings. */
export function useGinView(s: SessionContextValue): GinView {
  const {session, viewer} = s;
  // The seat chosen in View › Perspective, shared with every board.
  const [analysisViewSeat, setAnalysisViewSeat] = useAnalysisViewSeat();
  const [showOpponentHand, setShowOpponentHand] = useViewSetting<boolean>(
    SHOW_OPPONENT_KEY,
    false,
  );
  const [showKnowledgeHints] = useViewSetting<boolean>(
    SHOW_KNOWLEDGE_KEY,
    true,
  );
  const viewerSeatIndex = viewer.seat;
  const opponentSeatIndex =
    viewerSeatIndex === 0 ? 1 : viewerSeatIndex === null ? 1 : 0;
  const isAnalysisMode = session.mode === 'analysis';
  const autoRevealHands = session.state.ginRevealHands === true;
  const stateVisibleOpponent = visibleFlag(
    session.state.ginVisibleHands,
    opponentSeatIndex,
  );
  const manualOpponentReveal = isAnalysisMode && showOpponentHand;
  const baseViewSeat = viewer.viewSeat ?? viewer.seat ?? 0;
  useEffect(() => {
    if (!isAnalysisMode && analysisViewSeat !== null) setAnalysisViewSeat(null);
  }, [isAnalysisMode, analysisViewSeat, setAnalysisViewSeat]);
  useEffect(() => {
    if (!isAnalysisMode && showOpponentHand) setShowOpponentHand(false);
  }, [isAnalysisMode, setShowOpponentHand, showOpponentHand]);
  return {
    viewerSeatIndex,
    opponentSeatIndex,
    isAnalysisMode,
    manualOpponentReveal,
    preferObservationStruct:
      session.mode === 'game' &&
      !manualOpponentReveal &&
      !autoRevealHands &&
      !stateVisibleOpponent,
    effectiveViewSeat:
      isAnalysisMode && analysisViewSeat !== null
        ? analysisViewSeat
        : baseViewSeat,
    analysisViewSeat,
    setAnalysisViewSeat,
    showKnowledgeHints,
  };
}

/** The seats the analysis view may show: the players and the hands dealt. */
function seatOptionsOf(
  players: readonly GinPlayer[],
  playerHandSlot: number | null,
  opponentHandSlot: number | null,
  hands: unknown,
): number[] {
  const seats = new Set<number>(players.map(player => player.slot));
  if (playerHandSlot !== null) seats.add(playerHandSlot);
  if (opponentHandSlot !== null) seats.add(opponentHandSlot);
  const handCount = Array.isArray(hands) ? hands.length : 0;
  for (let i = 0; i < handCount; i += 1) seats.add(i);
  if (seats.size === 0) {
    seats.add(0);
    seats.add(1);
  }
  return [...seats].sort((a, b) => a - b);
}

/**
 * Keeps the seat options the View panel offers up to date, and the analysis
 * view seat among them.
 */
export function useSeatOptions(
  players: readonly GinPlayer[],
  d: GinRummyDerivedState,
  view: GinView,
): void {
  const [, setSeatOptions] = useViewSetting<readonly number[]>(
    SEAT_OPTIONS_KEY,
    [],
  );
  const {playerHandSlot, opponentHandSlot} = d;
  const hands = d.structRecord?.['hands'];
  const options = useMemo(
    () => seatOptionsOf(players, playerHandSlot, opponentHandSlot, hands),
    [players, playerHandSlot, opponentHandSlot, hands],
  );
  useEffect(() => {
    setSeatOptions(prev =>
      prev.length === options.length &&
      prev.every((value, i) => value === options[i])
        ? prev
        : options,
    );
  }, [options, setSeatOptions]);
  const {isAnalysisMode, analysisViewSeat, setAnalysisViewSeat} = view;
  useEffect(() => {
    if (!isAnalysisMode) return;
    if (
      analysisViewSeat !== null &&
      options.length > 0 &&
      !options.includes(analysisViewSeat)
    ) {
      setAnalysisViewSeat(options[0] ?? null);
    }
  }, [analysisViewSeat, options, isAnalysisMode, setAnalysisViewSeat]);
}
