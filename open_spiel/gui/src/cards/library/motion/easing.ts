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

// CSS easing strings as functions (internal to motion/): FLIP runs on the
// scene's ticker, not CSS transitions, so CardStyle.motion.easing is
// evaluated here. Unknown strings throw, naming the value.

/** An easing: progress in [0, 1] to eased progress. */
export type Easing = (p: number) => number;

const NAMED: Record<string, readonly [number, number, number, number]> = {
  ease: [0.25, 0.1, 0.25, 1],
  'ease-in': [0.42, 0, 1, 1],
  'ease-out': [0, 0, 0.58, 1],
  'ease-in-out': [0.42, 0, 0.58, 1],
};

/** The unit cubic Bézier through (0,0), (x1,y1), (x2,y2), (1,1) as y(x). */
export function cubicBezier(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): Easing {
  for (const [n, v] of [
    ['x1', x1],
    ['x2', x2],
  ] as const) {
    if (!(v >= 0 && v <= 1)) {
      throw new Error(`cubic-bezier ${n} must be in [0,1], got ${String(v)}`);
    }
  }
  const cx = 3 * x1;
  const bx = 3 * (x2 - x1) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * y1;
  const by = 3 * (y2 - y1) - cy;
  const ay = 1 - cy - by;
  const sx = (t: number) => ((ax * t + bx) * t + cx) * t;
  const sy = (t: number) => ((ay * t + by) * t + cy) * t;
  const dx = (t: number) => (3 * ax * t + 2 * bx) * t + cx;
  return p => {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    // Newton first, bisection if it stalls.
    let t = p;
    for (let i = 0; i < 8; i++) {
      const e = sx(t) - p;
      if (Math.abs(e) < 1e-6) return sy(t);
      const d = dx(t);
      if (Math.abs(d) < 1e-6) break;
      t -= e / d;
    }
    let lo = 0;
    let hi = 1;
    t = p;
    for (let i = 0; i < 40; i++) {
      const v = sx(t);
      if (Math.abs(v - p) < 1e-6) break;
      if (v < p) lo = t;
      else hi = t;
      t = (lo + hi) / 2;
    }
    return sy(t);
  };
}

/** 'linear', 'ease', 'ease-in', 'ease-out', 'ease-in-out' or 'cubic-bezier(a,b,c,d)'. */
export function parseEasing(s: string): Easing {
  const v = s.trim();
  if (v === 'linear') return p => Math.min(1, Math.max(0, p));
  const named = NAMED[v];
  if (named) return cubicBezier(...named);
  const m = /^cubic-bezier\(\s*([^,]+),([^,]+),([^,]+),([^)]+)\)$/.exec(v);
  if (m) {
    const [a, b, c, d] = m.slice(1).map(x => Number(x.trim()));
    if (
      a !== undefined &&
      b !== undefined &&
      c !== undefined &&
      d !== undefined &&
      [a, b, c, d].every(x => Number.isFinite(x))
    ) {
      return cubicBezier(a, b, c, d);
    }
  }
  throw new Error(
    `CardStyle.motion.easing ${JSON.stringify(s)} is not linear, ease, ease-in, ease-out, ease-in-out or cubic-bezier(a,b,c,d)`,
  );
}
