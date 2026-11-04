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

import {useEffect, useState} from 'react';
import type {GinTurnCue} from '../gin_move_cues.ts';

/**
 * How long the board stays still after a staged leg LANDS, ms at Normal
 * speed, before the next leg starts; the viewer's Animation speed scales it
 * (Slow 500 ms). The leg itself flies for the card library's flight (240 ms
 * at Normal, 600 ms at Slow), so a staged frame lasts flight + rest and each
 * leg lands, and is seen at rest, before the next one leaves.
 */
export const GIN_REST_MS = 200;

/**
 * The frame the board draws before a turn's final position:
 * - 'prior': the previous seat's discard has landed on the pile, and this
 *   turn has not drawn yet (the viewer's own discard, when the bot's answer
 *   arrives in the same update as it);
 * - 'drawn': this turn's draw is in the hand, its discard not yet made.
 */
export type GinStagePhase = 'prior' | 'drawn';

/** The staged frames' timing. */
export interface GinStageTiming {
  /**
   * One leg's flight, ms (the card library's motionDuration; 0 under Instant
   * or reduced motion, which stages nothing).
   */
  legMs: number;
  /** The stillness after a leg lands, ms (GIN_REST_MS at the viewer's speed). */
  restMs: number;
  /**
   * At the update that brings the turn: the previous seat's discard is new
   * to the board too (the pile has not drawn it yet), so it gets its own leg
   * before this turn's draw.
   */
  priorLeg: boolean;
}

/**
 * Which staged frame the board should draw, or null for the final one. It
 * stages only when a turn's draw and discard reach the board in the same
 * update (a bot answers with both at once): then, leg by leg, the previous
 * seat's discard lands ('prior', when it is new too), the drawn card goes
 * pile -> hand ('drawn'), and last the discard goes hand -> pile. Each frame
 * lasts one flight plus the rest, so the hold starts when a leg lands, not
 * when it leaves. A draw the board has already drawn on its own is never
 * staged again, so a discard arriving later animates alone.
 *
 * Coalesces: a newer turn arriving while one is staged replaces it (its own
 * frames, or none), so nothing queues. Never on the first render (a page load
 * shows where things are, not how they got there), never when `enabledByBoard`
 * is false (a view that cannot draw the frames) or the flight is 0 (Instant,
 * reduced motion). It only chooses which frame to draw; the board sends no
 * move from a staged frame (gin_rummy_board_surface.tsx): an input while it
 * stages skips the turn to its final frame instead.
 */
export function useGinTurnStaging(
  cue: GinTurnCue | null,
  enabledByBoard: boolean,
  timing: GinStageTiming,
): GinStagePhase | null {
  const {legMs, restMs, priorLeg} = timing;
  if (
    !Number.isFinite(legMs) ||
    legMs < 0 ||
    !Number.isFinite(restMs) ||
    restMs < 0
  ) {
    throw new Error(
      `useGinTurnStaging: legMs and restMs must be finite ms >= 0, got ${String(legMs)} and ${String(restMs)}`,
    );
  }
  const enabled = enabledByBoard && legMs > 0;
  const drawKey = cue?.drawIndex ?? null;
  // The draw last seen; undefined before the first render.
  const [seen, setSeen] = useState<number | null | undefined>(undefined);
  const [staged, setStaged] = useState<{
    key: number;
    phase: GinStagePhase;
  } | null>(null);
  let stagedNow = staged;
  if (drawKey !== seen) {
    // A new turn's draw: adjust during render, so the first staged frame
    // is the first one drawn for it (never the final frame, then a middle).
    const both = cue !== null && cue.discardIndex !== null;
    stagedNow =
      seen !== undefined && enabled && both && drawKey !== null
        ? {key: drawKey, phase: priorLeg ? 'prior' : 'drawn'}
        : null;
    setSeen(drawKey);
    setStaged(stagedNow);
  }
  const frameMs = legMs + restMs;
  useEffect(() => {
    if (staged === null) return;
    const t = setTimeout(() => {
      setStaged(s =>
        s?.key !== staged.key
          ? s
          : s.phase === 'prior'
            ? {key: s.key, phase: 'drawn'}
            : null,
      );
    }, frameMs);
    return () => {
      clearTimeout(t);
    };
  }, [staged, frameMs]);
  return enabled && stagedNow !== null && stagedNow.key === drawKey
    ? stagedNow.phase
    : null;
}
