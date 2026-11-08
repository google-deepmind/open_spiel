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

// A card arriving where it now sits, from where it last was: the play flight
// for a table that draws its own card elements (not a shared <Table>, whose
// FLIP does this by card identity). The element is already in its new place;
// it is drawn back at the old box and slides home, by the Web Animations API.
// The game says where each arriving card came from (its own card element in
// the hand, a seat for a hidden hand, the up-card), measured on the commit
// before it arrived. Twin of collect.ts collectElements, the other leg.
import type {CollectRun} from './collect.ts';

/** A box in viewport px. */
export interface ArriveBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

const NONE: CollectRun = {
  done: Promise.resolve(),
  dispose: () => {
    // Nothing moved, so nothing to remove.
  },
};

/** Slide `el` from `from` (viewport px, the box it left) to where it is laid
 *  out now, over `durationMs` with the CSS `easing`. Without the Web
 *  Animations API (jsdom), a layout box or a duration, nothing moves. */
export function arriveFrom(
  el: HTMLElement,
  from: ArriveBox,
  o: {durationMs: number; easing: string},
): CollectRun {
  for (const k of ['left', 'top', 'width', 'height'] as const) {
    if (!Number.isFinite(from[k])) {
      throw new Error(
        `arriveFrom: from.${k} must be finite, got ${String(from[k])}`,
      );
    }
  }
  if (!Number.isFinite(o.durationMs) || o.durationMs < 0) {
    throw new Error(
      `arriveFrom: durationMs must be finite ms >= 0, got ${String(o.durationMs)}`,
    );
  }
  if (o.durationMs === 0 || typeof el.animate !== 'function') return NONE;
  const to = el.getBoundingClientRect();
  if (to.width <= 0 || from.width <= 0) return NONE;
  // Viewport px -> the element's own px (an ancestor may scale it).
  const k = el.offsetWidth > 0 ? to.width / el.offsetWidth : 1;
  const dx = (from.left + from.width / 2 - (to.left + to.width / 2)) / k;
  const dy = (from.top + from.height / 2 - (to.top + to.height / 2)) / k;
  if (dx === 0 && dy === 0) return NONE;
  // Added on top of the card's own transform (its place in the trick).
  const anim = el.animate(
    [
      {transform: `translate(${String(dx)}px, ${String(dy)}px)`},
      {transform: 'translate(0px, 0px)'},
    ],
    {duration: o.durationMs, easing: o.easing, composite: 'add'},
  );
  return {
    done: anim.finished.then(
      () => undefined,
      () => undefined,
    ),
    dispose: () => {
      anim.cancel();
    },
  };
}
