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

// The page's mode and the seat a table draws at the bottom, as the games
// app's analysis mode has them (useBottomSeat and GameplayConfigProvider's
// analysisViewSeat there).
//
//   - Analysis: the page is shown the oracle's frame, the full state, and
//     the viewer sits at any seat (View › Perspective), which only chooses
//     the seat drawn at the bottom; it fetches nothing. A board's own
//     "Show all cards" turns the other hands face up.
//   - Game: the page is shown one seat's frame (or a spectator's), which the
//     server has stripped of what that seat may not see, and the
//     perspective is locked to it.

import {useEffect} from 'react';

import type {Viewer} from '../contract/contract.ts';
import {useViewSetting} from '../games/shared/view_settings.ts';
import {useGame} from './game_context.ts';

/** How a page shows its session: analysis, or a game at one seat. */
export type SessionMode = 'analysis' | 'game';

/** The mode of a page shown to `viewer`: analysis exactly for the oracle. */
export function sessionModeOf(viewer: Viewer): SessionMode {
  return viewer.kind === 'oracle' ? 'analysis' : 'game';
}

/**
 * Whether the page is in analysis: what a board's View section reads to
 * decide whether its reveal toggle is enabled.
 */
export function useIsAnalysisMode(): boolean {
  return sessionModeOf(useGame().viewer) === 'analysis';
}

/** Whose seat a table draws at the bottom, and whether it may change. */
export interface BottomSeat {
  readonly seat: number;
  readonly switchable: boolean;
  /** Why the Perspective cannot be changed (null when it can). */
  readonly reason: string | null;
}

/** The reason a seat's Perspective is locked. */
export const PERSPECTIVE_LOCKED_SEATED = 'Locked to your seat during a game';
/** The reason a spectator's Perspective is locked; it names no hidden hands. */
export const PERSPECTIVE_LOCKED_SPECTATING =
  'Spectating: the perspective is fixed during a game';

/** The shared analysis seat's key in the view settings. */
export const ANALYSIS_VIEW_SEAT_KEY = 'analysisViewSeat';

type SessionLike = {readonly mode?: string} | null | undefined;

/** The viewer as a table names it. */
export interface SeatViewer {
  readonly seat: number | null;
  /** The seat the page opened at in analysis (the address's `viewSeat`). */
  readonly viewSeat?: number | null;
  readonly isParticipant: boolean;
}

function seatNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0 ? v : null;
}

/** Whether the viewer may choose the bottom seat, and if not, why. */
export function perspectiveLock(
  session: SessionLike,
  viewer: SeatViewer,
): {readonly switchable: boolean; readonly reason: string | null} {
  if (session?.mode === 'analysis') return {switchable: true, reason: null};
  const seated = viewer.isParticipant && seatNumber(viewer.seat) !== null;
  return {
    switchable: false,
    reason: seated ? PERSPECTIVE_LOCKED_SEATED : PERSPECTIVE_LOCKED_SPECTATING,
  };
}

/**
 * The bottom seat (pure; useBottomSeat is the hook). Analysis: the seat
 * `chosen` in View › Perspective, else the viewer's view seat, else its own,
 * else `fallback`. A game: the viewer's own seat; a spectator gets
 * `fallback` for the layout only, and the server sends it backs.
 */
export function bottomSeatOf(
  session: SessionLike,
  viewer: SeatViewer,
  chosen: number | null,
  fallback = 0,
): BottomSeat {
  const lock = perspectiveLock(session, viewer);
  const own = seatNumber(viewer.seat);
  if (!lock.switchable) return {seat: own ?? fallback, ...lock};
  const seat =
    seatNumber(chosen) ?? seatNumber(viewer.viewSeat) ?? own ?? fallback;
  return {seat, ...lock};
}

/**
 * The seat chosen in View › Perspective in analysis, shared by the
 * Perspective select and every board, and how to choose it; null until one
 * is chosen.
 */
export function useAnalysisViewSeat(): [
  number | null,
  (seat: number | null) => void,
] {
  return useViewSetting<number | null>(ANALYSIS_VIEW_SEAT_KEY, null);
}

/**
 * The bottom seat for a board (see bottomSeatOf). A seat chosen while
 * switchable is cleared once it is not (analysis to a game), so a seat
 * picked in analysis never carries into a game.
 */
export function useBottomSeat(
  session: SessionLike,
  viewer: SeatViewer,
  fallback = 0,
): BottomSeat {
  const [analysisViewSeat, setAnalysisViewSeat] = useAnalysisViewSeat();
  const bottom = bottomSeatOf(session, viewer, analysisViewSeat, fallback);
  useEffect(() => {
    if (!bottom.switchable && analysisViewSeat !== null) {
      setAnalysisViewSeat(null);
    }
  }, [bottom.switchable, analysisViewSeat, setAnalysisViewSeat]);
  return bottom;
}
