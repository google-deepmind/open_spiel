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

// A fan with turned cards (a tapped card in a row in play): each card takes
// its turned footprint along the axis, and the step past it clears its
// turned index box.
import {itemAt} from './checked.ts';
import {cornerIndex, type IndexCorner} from './index_scale.ts';
import {
  fanAxis,
  PAD_FRACTION,
  PAD_MIN_PX,
  type AxisGeometry,
  type PileGeometryInput,
  type PileMode,
} from './pile_geometry.ts';
import {DEFAULT_CARD_STYLE} from './style.ts';
import type {IndexGeometry} from './types.ts';

/** The step past a face-up card turned `turn` degrees in a fan: its turned
 *  index box's far edge along the axis (from the covered side for
 *  fan-left/fan-up) plus padding, px, and the corner that gives it. */
function turnedClearance(
  deck: PileGeometryInput['deck'],
  scale: number,
  turn: number,
  mode: PileMode,
  fw: number,
  fh: number,
): {px: number; corner: IndexCorner} {
  const axis = fanAxis(mode);
  const reversed = mode === 'fan-left' || mode === 'fan-up';
  const size = axis === 'down' ? fh : fw;
  const pad = Math.max(PAD_MIN_PX, PAD_FRACTION * size);
  const corners: IndexCorner[] = deck.rotatedTwin
    ? ['top-left', 'bottom-right']
    : ['top-left'];
  let best: {px: number; corner: IndexCorner} | null = null;
  for (const c of corners) {
    const idx = rotateTurn(cornerIndex(deck, scale, c), turn);
    const near = axis === 'down' ? idx.top : idx.left;
    const len = axis === 'down' ? idx.height : idx.width;
    const far = reversed ? 1 - near : near + len;
    const px = Math.ceil(far * size + pad);
    if (!best || px < best.px) best = {px, corner: c};
  }
  if (best === null) throw new Error('turnedClearance: no corner');
  return best;
}

/** IndexGeometry on a card turned `deg` clockwise, as fractions of the turned footprint. */
function rotateTurn(index: IndexGeometry, deg: number): IndexGeometry {
  const {top, left, width, height} = index;
  switch (deg) {
    case 90:
      return {
        left: 1 - (top + height),
        top: left,
        width: height,
        height: width,
      };
    case 180:
      return {left: 1 - (left + width), top: 1 - (top + height), width, height};
    case 270:
      return {left: top, top: 1 - (left + width), width: height, height: width};
    default:
      return index;
  }
}

/** The constants turnedAxis is given by computeAxis. */
export interface AxisConstants {
  axis: 'down' | 'right';
  axisSize: number;
  crossSize: number;
  upFactor: number;
  downFraction: number;
  minUp: number;
  minDown: number;
  index: IndexGeometry;
}

type AxisBase = Pick<
  AxisGeometry,
  'corner' | 'cardHeight' | 'minFaceUpOffset' | 'minFaceDownOffset'
>;

// Each gap's floor (face-down gaps squeeze to minDown) and each card's index
// corner.
interface Floors {
  turns: number[];
  corners: IndexCorner[];
  upMin: number[];
  isDown: boolean[];
}

function floors(
  input: PileGeometryInput,
  base: AxisBase,
  k: AxisConstants,
): Floors {
  const {deck, cardWidth, faceUp, mode} = input;
  const h = base.cardHeight;
  const scale = input.indexScale ?? DEFAULT_CARD_STYLE.indexScale;
  const n = faceUp.length;
  const turns = faceUp.map((_, i) => input.turns?.[i] ?? 0);
  const corners: IndexCorner[] = turns.map(() => base.corner);
  const clearance = (i: number): {px: number; corner: IndexCorner} => {
    const t = itemAt(turns, i);
    const quarter = t === 90 || t === 270;
    return turnedClearance(
      deck,
      scale,
      t,
      mode,
      quarter ? h : cardWidth,
      quarter ? cardWidth : h,
    );
  };
  const upMin: number[] = [];
  const isDown: boolean[] = [];
  for (let i = 0; i < n - 1; i++) {
    if (faceUp[i] === false) {
      isDown.push(true);
      upMin.push(0);
    } else if (itemAt(turns, i)) {
      const c = clearance(i);
      corners[i] = c.corner;
      isDown.push(false);
      upMin.push(c.px);
    } else {
      isDown.push(false);
      upMin.push(k.minUp);
    }
  }
  if (n > 0 && itemAt(turns, n - 1)) corners[n - 1] = clearance(n - 1).corner;
  return {turns, corners, upMin, isDown};
}

// The gaps after squeezing to `budget`: face-down gaps first, then face-up
// ones toward their own floors in proportion; nothing goes below a floor.
function squeeze(
  f: Floors,
  upPref: readonly number[],
  prefDown: number,
  k: AxisConstants,
  budget: number | undefined,
): {gaps: number[]; fits: boolean; compressed: boolean} {
  const {upMin, isDown} = f;
  const downGaps = isDown.filter(Boolean).length;
  let down = prefDown;
  const gaps = upPref.map((p, i) => (isDown[i] ? down : p));
  if (budget === undefined) return {gaps, fits: true, compressed: false};
  const sumUp = (g: readonly number[]): number =>
    g.reduce((a, v, i) => a + (isDown[i] ? 0 : v), 0);
  if (sumUp(gaps) + down * downGaps <= budget) {
    return {gaps, fits: true, compressed: false};
  }
  if (downGaps > 0) {
    const fit = Math.floor((budget - sumUp(gaps)) / downGaps);
    down = Math.max(k.minDown, Math.min(prefDown, fit));
  }
  if (sumUp(gaps) + down * downGaps > budget) {
    const minSum = sumUp(upMin);
    const span = sumUp(upPref) - minSum;
    const share =
      span > 0
        ? Math.max(0, Math.min(1, (budget - down * downGaps - minSum) / span))
        : 0;
    gaps.forEach((_, i) => {
      if (isDown[i]) return;
      const lo = itemAt(upMin, i);
      gaps[i] = Math.max(lo, Math.floor(lo + share * (itemAt(upPref, i) - lo)));
    });
  }
  gaps.forEach((_, i) => {
    if (isDown[i]) gaps[i] = down;
  });
  return {
    gaps,
    fits: sumUp(gaps) + down * downGaps <= budget,
    compressed: true,
  };
}

/** computeAxis for a fan with turned cards: per-card footprints and index
 *  floors. Face-down gaps squeeze first, then face-up ones toward their own
 *  floors in proportion; nothing goes below a floor. */
export function turnedAxis(
  input: PileGeometryInput,
  base: AxisBase,
  k: AxisConstants,
): AxisGeometry {
  const {cardWidth, faceUp, maxExtent} = input;
  const h = base.cardHeight;
  const n = faceUp.length;
  const f = floors(input, base, k);
  const quarter = (i: number): boolean => {
    const t = itemAt(f.turns, i);
    return t === 90 || t === 270;
  };
  // A turned card's footprint swaps its width and height.
  const along = (i: number): number =>
    (k.axis === 'right') !== quarter(i) ? cardWidth : h;
  const across = (i: number): number =>
    (k.axis === 'right') !== quarter(i) ? h : cardWidth;
  const upPref = f.upMin.map(m => Math.max(m, Math.round(m * k.upFactor)));
  const prefDown = Math.max(k.minDown, Math.round(k.downFraction * k.axisSize));
  const lastAlong = n > 0 ? along(n - 1) : k.axisSize;
  const budget = maxExtent === undefined ? undefined : maxExtent - lastAlong;
  const {gaps, fits, compressed} = squeeze(f, upPref, prefDown, k, budget);
  const offsets: number[] = n > 0 ? [0] : [];
  for (let i = 0; i < n - 1; i++) {
    offsets.push(itemAt(offsets, i) + itemAt(gaps, i));
  }
  const extent = n > 0 ? itemAt(offsets, n - 1) + lastAlong : k.axisSize;
  const crossExtent = Math.max(k.crossSize, ...faceUp.map((_, i) => across(i)));
  return {
    ...base,
    offsets,
    extent,
    crossExtent,
    fits,
    compressed,
    turned: {turns: f.turns, corners: f.corners},
  };
}
