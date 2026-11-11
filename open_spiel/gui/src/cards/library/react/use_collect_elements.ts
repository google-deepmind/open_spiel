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

// The trick collect for a table that draws its own card elements (not a
// shared <Table>): when `collect` gets a new key, the elements `cards()`
// returns slide together toward the centre of `toward()` and fade
// (motion/collect.ts collectElements). Changing or clearing `collect` stops
// it; the game removes the cards from its state when the hook's collect
// phase ends. Reduced motion: nothing moves (the hook skips the phase too).
import {useLayoutEffect, useRef} from 'react';
import {collectElements} from '../motion/collect.ts';
import {prefersReducedMotion} from '../motion/service.ts';

/** The cards of a finished trick, and the seat they fly to. */
export interface ElementCollect {
  key: string;
  durationMs: number;
  /** The finished trick's card elements. */
  cards(): HTMLElement[];
  /** The winner's seat. Null: the seat is not on screen (throws). */
  toward(): Element | null;
}

/** Flies a finished trick's cards to its winner once per `collect.key`. */
export function useCollectElements(collect: ElementCollect | null): void {
  const latest = useRef(collect);
  useLayoutEffect(() => {
    latest.current = collect;
  });
  const key = collect?.key ?? null;
  useLayoutEffect(() => {
    const c = latest.current;
    if (!c || prefersReducedMotion()) return;
    const cards = c.cards();
    if (cards.length === 0) return;
    const seat = c.toward();
    if (!seat) {
      throw new Error(
        `trick collect '${c.key}': the winner's seat is not on screen`,
      );
    }
    const r = seat.getBoundingClientRect();
    const run = collectElements(
      cards,
      {x: r.left + r.width / 2, y: r.top + r.height / 2},
      {durationMs: c.durationMs},
    );
    return () => {
      run.dispose();
    };
  }, [key]);
}
