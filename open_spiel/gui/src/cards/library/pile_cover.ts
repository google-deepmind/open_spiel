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

// The legibility check of laid-out cards: which face-up cards have their
// index box covered by a card painted after them, upright or rotated.
import {itemAt} from './checked.ts';
import {mirrorIndex, quarterTurnIndex} from './index_scale.ts';
import type {PileGeometry} from './pile_geometry.ts';
import type {IndexGeometry} from './types.ts';

/** A rectangle, px. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A card placed in a pile: its rectangle, turn and index box. */
export interface PlacedCard {
  /** Card rectangle in a shared coordinate space, before rotation. */
  rect: Rect;
  /** Degrees clockwise about the rectangle's centre (arc fans, rotated seats). */
  rotate?: number | undefined;
  faceUp: boolean;
  /** The card's deck index box; cards from different decks may share a check. */
  index: IndexGeometry;
}

/** Overlaps smaller than this (px) are browser rounding, not cover. */
const COVER_EPS = 0.5;

function intersects(a: Rect, b: Rect): boolean {
  // Touching edges do not cover: strict inequalities, with a sub-pixel epsilon
  // so browser rounding of fractional layouts is not reported as an overlap.
  const eps = COVER_EPS;
  return (
    a.x < b.x + b.width - eps &&
    b.x < a.x + a.width - eps &&
    a.y < b.y + b.height - eps &&
    b.y < a.y + a.height - eps
  );
}

/** A point, px. */
export interface Point {
  x: number;
  y: number;
}

/** The corners of `rect` rotated `deg` clockwise about `pivot` (default: its centre). */
export function quadOf(rect: Rect, deg = 0, pivot?: Point): Point[] {
  const corners = [
    {x: rect.x, y: rect.y},
    {x: rect.x + rect.width, y: rect.y},
    {x: rect.x + rect.width, y: rect.y + rect.height},
    {x: rect.x, y: rect.y + rect.height},
  ];
  if (!deg) return corners;
  const c = pivot ?? {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
  const cos = Math.cos((deg * Math.PI) / 180);
  const sin = Math.sin((deg * Math.PI) / 180);
  return corners.map(p => ({
    x: c.x + (p.x - c.x) * cos - (p.y - c.y) * sin,
    y: c.y + (p.x - c.x) * sin + (p.y - c.y) * cos,
  }));
}

function quadArea(q: readonly Point[]): number {
  let twice = 0;
  for (let i = 0; i < 4; i++) {
    const p = itemAt(q, i);
    const n = itemAt(q, (i + 1) % 4);
    twice += p.x * n.y - n.x * p.y;
  }
  return Math.abs(twice) / 2;
}

/** Separating-axis test for two convex quads; overlaps within COVER_EPS do not count. */
function quadsIntersect(a: readonly Point[], b: readonly Point[]): boolean {
  // A zero-area quad (an unlaid-out or collapsed element) has no edge to
  // test and covers nothing: without this, no separating axis is found and
  // two points would read as overlapping.
  if (quadArea(a) < 1e-9 || quadArea(b) < 1e-9) return false;
  for (const poly of [a, b]) {
    for (let i = 0; i < 4; i++) {
      const p = itemAt(poly, i);
      const q = itemAt(poly, (i + 1) % 4);
      const len = Math.hypot(q.x - p.x, q.y - p.y);
      if (len === 0) continue;
      const nx = -(q.y - p.y) / len;
      const ny = (q.x - p.x) / len;
      let aMin = Infinity;
      let aMax = -Infinity;
      let bMin = Infinity;
      let bMax = -Infinity;
      for (const v of a) {
        const d = v.x * nx + v.y * ny;
        aMin = Math.min(aMin, d);
        aMax = Math.max(aMax, d);
      }
      for (const v of b) {
        const d = v.x * nx + v.y * ny;
        bMin = Math.min(bMin, d);
        bMax = Math.max(bMax, d);
      }
      if (aMax - COVER_EPS <= bMin || bMax - COVER_EPS <= aMin) return false;
    }
  }
  return true;
}

/** The index box of a card rectangle, in the same coordinate space (unrotated). */
export function indexRect(rect: Rect, index: IndexGeometry): Rect {
  return {
    x: rect.x + index.left * rect.width,
    y: rect.y + index.top * rect.height,
    width: index.width * rect.width,
    height: index.height * rect.height,
  };
}

/** The index box of a placed card as a quad, rotated with its card. */
export function indexQuad(card: PlacedCard): Point[] {
  const {rect} = card;
  return quadOf(indexRect(rect, card.index), card.rotate, {
    x: rect.x + rect.width / 2,
    y: rect.y + rect.height / 2,
  });
}

/**
 * The legibility check. Given cards in paint order (later cards drawn on top),
 * return the positions of face-up cards whose index box is overlapped by any
 * card painted after them. An empty result means every rank and suit is visible.
 * Rotated cards are compared as rotated quads.
 */
export function findCoveredIndexes(cards: readonly PlacedCard[]): number[] {
  const covered: number[] = [];
  cards.forEach((card, i) => {
    if (!card.faceUp) return;
    const straight = !card.rotate;
    const box = indexRect(card.rect, card.index);
    const quad = straight ? null : indexQuad(card);
    for (const other of cards.slice(i + 1)) {
      const hit =
        straight && !other.rotate
          ? intersects(box, other.rect)
          : quadsIntersect(
              quad ?? quadOf(box),
              quadOf(other.rect, other.rotate),
            );
      if (hit) {
        covered.push(i);
        break;
      }
    }
  });
  return covered;
}

/** Place a computed pile's cards as rectangles, for the legibility check.
 *  `index` is the deck's top-left box at the style's scale; upright cards are
 *  checked on the geometry's corner, a rotated base on its turned box. Holes
 *  are not cards and are skipped. */
export function placePile(
  geometry: PileGeometry,
  faceUp: ReadonlyArray<boolean | null>,
  index: IndexGeometry,
): PlacedCard[] {
  const upright = geometry.corner === 'top-left' ? index : mirrorIndex(index);
  const placed: PlacedCard[] = [];
  geometry.slots.forEach((slot, i) => {
    const f = itemAt(faceUp, i);
    if (f === null) return;
    const {rotated, turn, corner, ...rect} = slot;
    if (turn) {
      placed.push({
        faceUp: f,
        index: corner === 'bottom-right' ? mirrorIndex(index) : index,
        rect,
        rotate: turn,
      });
      return;
    }
    placed.push({
      faceUp: f,
      index: rotated ? quarterTurnIndex(index) : upright,
      rect,
    });
  });
  return placed;
}
