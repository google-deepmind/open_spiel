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

// Plays that reach the table together are shown one at a time. A trick game
// served by bots gets several seats' plays in one update (the bots answer the
// viewer at once); drawn together they all fly in the same frame and nobody
// can see who led (hearts at Slow: 2c and 5c from two bots in the same
// frame). This hook says how many of the plays to draw: the ones already on
// the table, then one more each time the last one has landed and rested, at
// the viewer's Animation speed. It applies to any game whose plays fly.
import {useEffect, useState} from 'react';

/** How long the table stays still after a staged play lands, ms at Normal
 *  speed, before the next play leaves (Slow 500 ms). */
export const PLAY_REST_MS = 200;

/** How long a staged play flies and rests. */
export interface StagedPlaysTiming {
  /** One play's flight, ms (the card library's motionDuration; 0 under
   *  Instant or reduced motion, which stages nothing). */
  legMs: number;
  /** The stillness after a play lands, ms (PLAY_REST_MS at the viewer's speed). */
  restMs: number;
}

/** What to draw of the plays now. */
export interface StagedPlays {
  /** How many of the plays to draw now. */
  shown: number;
  /** True from the update that started a staging until its last play has
   *  LANDED (one flight after it left). A trick's hold waits for this, so
   *  it never starts, or ends, under a card still in the air. */
  settling: boolean;
}

/**
 * Which of `keys` (the plays on the table, in play order, one key per play)
 * to draw now. All of them on the first render (a page load shows where
 * things are), when the flight is 0, or when at most one play is new. When
 * more than one is new, the plays already drawn stay, the first new one flies
 * at once, and each later one leaves when the one before has landed and
 * rested (legMs + restMs). A list that does not start with what is drawn (a
 * new trick) stages from its first play. Coalesces: a newer list replaces the
 * staging in progress, keeping what is already drawn. It never blocks input;
 * it only chooses what to draw.
 */
export function useStagedPlays(
  keys: readonly string[],
  timing: StagedPlaysTiming,
): StagedPlays {
  const {legMs, restMs} = timing;
  if (
    !Number.isFinite(legMs) ||
    legMs < 0 ||
    !Number.isFinite(restMs) ||
    restMs < 0
  ) {
    throw new Error(
      `useStagedPlays: legMs and restMs must be finite ms >= 0, got ${String(legMs)} and ${String(restMs)}`,
    );
  }
  const sig = keys.join('|');
  // What was last drawn: the list, how many of it, and whether a staging
  // of it is still settling.
  const [drawn, setDrawn] = useState<{
    sig: string;
    keys: readonly string[];
    shown: number;
    settling: boolean;
  } | null>(null);
  let now = drawn;
  if (drawn?.sig !== sig) {
    let shown = keys.length;
    let settling = false;
    if (drawn !== null && legMs > 0) {
      const before = drawn.keys.slice(0, drawn.shown);
      const prefix = before.every((k, i) => keys[i] === k);
      // Plays already drawn that come back in another order: never pull
      // a card off the table to fly it again; draw the list as it is.
      const reordered =
        !prefix && before.length > 0 && before.every(k => keys.includes(k));
      const start = prefix ? before.length : 0;
      if (!reordered && keys.length - start > 1) {
        shown = start + 1;
        settling = true;
      }
    }
    now = {sig, keys, shown, settling};
    setDrawn(now);
  }
  const shown = now?.shown ?? keys.length;
  const settling = now?.settling ?? false;
  const total = keys.length;
  useEffect(() => {
    if (legMs <= 0 || (!settling && shown >= total)) return;
    const t = setTimeout(
      () => {
        setDrawn(d => {
          if (d?.sig !== sig) return d;
          // The next play leaves; or, the last one having landed, the staging is done.
          return d.shown < d.keys.length
            ? {...d, shown: d.shown + 1}
            : {...d, settling: false};
        });
      },
      shown < total ? legMs + restMs : legMs,
    );
    return () => {
      clearTimeout(t);
    };
  }, [sig, shown, total, settling, legMs, restMs]);
  // Instant mid-staging (the viewer changed the setting): draw everything, now.
  return legMs > 0 ? {shown, settling} : {shown: total, settling: false};
}
