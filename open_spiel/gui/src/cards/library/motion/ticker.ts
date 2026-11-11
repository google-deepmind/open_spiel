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

// Tickers: the frame source for motion. Owned by the motion module.
import {
  createContext,
  createElement,
  useContext,
  type JSX,
  type ReactNode,
} from 'react';

/** A frame source: request a callback for the next frame, cancel it, read the clock. */
export interface Ticker {
  request(f: (tMs: number) => void): number;
  cancel(h: number): void;
  now(): number;
}

/** requestAnimationFrame. Outside a browser (SSR) it falls back to timers. */
export const rafTicker: Ticker = {
  /** The next animation frame. */
  request: f =>
    typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame(f)
      : (setTimeout(() => {
          f(Date.now());
        }, 16) as unknown as number), // motion-exempt: one frame of the frame loop, not a duration
  /** Cancels a requested frame. */
  cancel: h => {
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(h);
    else clearTimeout(h);
  },
  /** The clock, ms. */
  now: () =>
    typeof performance !== 'undefined' ? performance.now() : Date.now(),
};

/** One manual frame, ms: advance() steps the clock in frames of at most this. */
export const MANUAL_FRAME_MS = 16;

/** Deterministic ticker for tests: advance(ms) runs the pending frames.
 *  The clock steps in frames of at most MANUAL_FRAME_MS (a final partial
 *  frame included), and each frame runs the callbacks requested before it,
 *  so advance(1000) behaves like a second of requestAnimationFrame. A callback
 *  requested inside a frame runs in the next one. */
export function manualTicker(): Ticker & {advance(ms: number): void} {
  let t = 0;
  let next = 1;
  let pending = new Map<number, (tMs: number) => void>();
  let advancing = false;
  const frame = () => {
    const run = pending;
    pending = new Map();
    for (const f of run.values()) f(t);
  };
  return {
    request: f => {
      const h = next++;
      pending.set(h, f);
      return h;
    },
    cancel: h => {
      pending.delete(h);
    },
    now: () => t,
    advance: ms => {
      if (!(ms >= 0) || !Number.isFinite(ms)) {
        throw new Error(
          `manualTicker.advance needs a finite ms >= 0, got ${String(ms)}`,
        );
      }
      if (advancing) {
        throw new Error('manualTicker.advance called from inside a frame');
      }
      advancing = true;
      try {
        let left = ms;
        while (left > 0) {
          const step = Math.min(MANUAL_FRAME_MS, left);
          t += step;
          left -= step;
          frame();
        }
      } finally {
        advancing = false;
      }
    },
  };
}

const TickerContext = createContext<Ticker>(rafTicker);

/** Provider override (tests and headless recording). */
export function TickerProvider(p: {
  ticker: Ticker;
  children: ReactNode;
}): JSX.Element {
  return createElement(TickerContext.Provider, {value: p.ticker}, p.children);
}

/** The ticker in effect: the provider's, else requestAnimationFrame. */
export function useTicker(): Ticker {
  return useContext(TickerContext);
}
