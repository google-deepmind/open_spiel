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

// useTrickCollect: the hold -> collect -> clear sequence of a finished trick
// (trickCollect.ts), for any trick-taking table, whatever draws its cards.
//
// The game passes a TrickFeed per render (from the served payload). When a
// live delta finishes a trick, the hook returns that trick with phase 'hold'
// IN THE SAME RENDER (no frame shows the cleared table first), then 'collect'
// after holdMs, then null after collectMs. While it is non-null the game
// draws the finished trick in the trick area instead of the live one, and
// keeps the live trick's cards in their players' hands; the live state is
// accepted underneath and shows when the trick clears.
//
// Nothing cuts a hold or a collect short. Input is never held back (the
// game's handlers stay live and a move goes to the server at once), and a
// trick that finishes while another is on show (the viewer moved during the
// hold and the bots answered, or a bot reply) is QUEUED: it gets its own full
// hold and collect when the one on show has cleared. The game keeps the
// queued tricks' cards in their players' hands meanwhile (`pending`).
// Under prefers-reduced-motion a trick clears at the end of its hold, with
// no collect. holdMs 0 (the 'off' setting) shows nothing.
import {useEffect, useState} from 'react';
import {prefersReducedMotion} from '../motion/service.ts';
import {
  detectCompletedTrick,
  sameTrickFeed,
  TRICK_COLLECT_MS,
  type CompletedTrick,
  type TrickFeed,
} from '../trick_collect.ts';

/** A finished trick held on show, or flying to its winner. */
export type TrickCollectPhase = 'hold' | 'collect';

/** The finished trick on show, if any, and its phase. */
export interface TrickCollectState<C = string, S = string> {
  /** The finished trick on show, or null: draw the live trick. */
  trick: CompletedTrick<C, S> | null;
  phase: TrickCollectPhase | null;
  /** Distinct per trick on show (session and index): a key for the collect. */
  key: string | null;
  /** The collect's duration, ms (for the animation the game runs). */
  collectMs: number;
  /** Tricks finished since, waiting their turn, oldest first. Their cards
   *  are still in the players' hands as far as the table shows. */
  pending: ReadonlyArray<CompletedTrick<C, S>>;
}

/** How long a finished trick is held and collected. */
export interface TrickCollectOptions {
  /** The hold, ms (TRICK_HOLD_MS of the viewer's setting). 0: off. */
  holdMs: number;
  /** Default TRICK_COLLECT_MS. */
  collectMs?: number | undefined;
  /** Default: the user's prefers-reduced-motion. */
  reducedMotion?: boolean | undefined;
  /** True while the game is still drawing plays onto the table (plays that
   *  arrived together, shown one at a time).
   *  The hold of the trick on show does not run meanwhile, so it starts
   *  once the last card has landed, and never collects a card still in the
   *  air or one not drawn yet. Default false. */
  settling?: boolean | undefined;
}

interface Shown<C, S> {
  trick: CompletedTrick<C, S>;
  phase: TrickCollectPhase;
  key: string;
}

interface Show<C, S> {
  /** On the table now; null: nothing. */
  shown: Shown<C, S> | null;
  /** Waiting, oldest first. */
  queue: ReadonlyArray<Shown<C, S>>;
}

const NOTHING: Show<never, never> = {shown: null, queue: []};

function checkTimes(holdMs: number, collectMs: number): void {
  if (!Number.isFinite(holdMs) || holdMs < 0) {
    throw new Error(
      `useTrickCollect: holdMs must be >= 0, got ${String(holdMs)}`,
    );
  }
  if (!Number.isFinite(collectMs) || collectMs < 0) {
    throw new Error(
      `useTrickCollect: collectMs must be >= 0, got ${String(collectMs)}`,
    );
  }
}

// Whether what was on show belongs to a game no longer on screen: scrubbing,
// another session or a shorter history.
function offScreen<C, S>(
  seen: TrickFeed<C, S> | null,
  next: TrickFeed<C, S> | null,
): boolean {
  if (next === null || seen === null || next.scrubbing) return true;
  return next.session !== seen.session || next.history < seen.history;
}

/** Holds a finished trick on show, then collects it to its winner. */
export function useTrickCollect<C, S>(
  feed: TrickFeed<C, S> | null | undefined,
  o: TrickCollectOptions,
): TrickCollectState<C, S> {
  const {holdMs} = o;
  const collectMs = o.collectMs ?? TRICK_COLLECT_MS;
  checkTimes(holdMs, collectMs);
  const [seen, setSeen] = useState(feed ?? null);
  const [show, setShow] = useState<Show<C, S>>(NOTHING);

  // Adjust state to a new payload during render, so the render that first
  // shows the new state already holds the finished trick.
  let current = show;
  if (!sameTrickFeed(seen, feed)) {
    const next = feed ?? null;
    const done = detectCompletedTrick(seen, next);
    setSeen(next);
    let after = show;
    if (offScreen(seen, next)) after = NOTHING;
    if (done && holdMs > 0 && next) {
      const item: Shown<C, S> = {
        trick: done,
        phase: 'hold',
        key: `${next.session}:${String(done.index)}`,
      };
      after = after.shown
        ? {shown: after.shown, queue: [...after.queue, item]}
        : {shown: item, queue: []};
    }
    if (after !== show) setShow(after);
    current = after;
  }
  if (holdMs <= 0) {
    // No hold: nothing is on show, and nothing waits to be.
    if (show !== NOTHING) setShow(NOTHING);
    current = NOTHING;
  }

  const reduced = o.reducedMotion;
  const settling = o.settling ?? false;
  // The timer belongs to the trick and phase on show: a trick queued behind
  // it must not restart (and so lengthen) the hold under way.
  const onKey = show.shown?.key ?? null;
  const onPhase = show.shown?.phase ?? null;
  useEffect(() => {
    if (onKey === null || holdMs <= 0) return;
    // The hold waits for the table to finish drawing its plays.
    if (onPhase === 'hold' && settling) return;
    const id = setTimeout(
      () => {
        setShow(s => {
          const on = s.shown;
          if (on?.key !== onKey || on.phase !== onPhase) return s;
          const still = reduced ?? prefersReducedMotion();
          if (on.phase === 'hold' && !still && collectMs > 0) {
            return {shown: {...on, phase: 'collect'}, queue: s.queue};
          }
          const [head, ...rest] = s.queue;
          return head ? {shown: head, queue: rest} : NOTHING;
        });
      },
      onPhase === 'hold' ? holdMs : collectMs,
    );
    return () => {
      clearTimeout(id);
    };
  }, [onKey, onPhase, holdMs, collectMs, reduced, settling]);

  return {
    trick: current.shown?.trick ?? null,
    phase: current.shown?.phase ?? null,
    key: current.shown?.key ?? null,
    collectMs,
    pending: current.queue.map(q => q.trick),
  };
}
