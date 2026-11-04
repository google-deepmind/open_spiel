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

// The geometry layoutScene builds on: turning points and boxes between a
// slot's frame and the table, bounds, and the caret box of an insert.
import type {IndexGeometry} from '../types.ts';
import {rotateIndex, turnsInPlace} from '../table_spec.ts';
import {quadOf} from '../pile_geometry.ts';
import {
  isToken,
  type Box,
  type Point,
  type Rect,
  type TableItem,
} from './identity.ts';
import {defined} from './defined.ts';
import type {Placement, SlotBox, SlotLayout} from './scene_types.ts';

/** `p` turned `deg` degrees clockwise about `pivot`. */
export function rotateAbout(p: Point, pivot: Point, deg: number): Point {
  if (!deg) return p;
  const a = (deg * Math.PI) / 180;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  return {
    x: pivot.x + (p.x - pivot.x) * cos - (p.y - pivot.y) * sin,
    y: pivot.y + (p.x - pivot.x) * sin + (p.y - pivot.y) * cos,
  };
}

/** A local rect of a slot, as a table-px Box (rotation about its own centre). */
export function toTableBox(b: SlotBox, r: Rect, localRotate: number): Box {
  const c = rotateAbout(
    {x: b.left + r.x + r.width / 2, y: b.top + r.y + r.height / 2},
    {x: b.left + b.origin.x, y: b.top + b.origin.y},
    b.rotate,
  );
  return {
    x: c.x - r.width / 2,
    y: c.y - r.height / 2,
    width: r.width,
    height: r.height,
    rotate: norm(b.rotate + localRotate),
  };
}

/** A point in a slot's local frame, in table px. */
export function toTablePoint(b: SlotBox, p: Point): Point {
  return rotateAbout(
    {x: b.left + p.x, y: b.top + p.y},
    {x: b.left + b.origin.x, y: b.top + b.origin.y},
    b.rotate,
  );
}

/** Degrees in [0, 360). */
export function norm(deg: number): number {
  const d = ((deg % 360) + 360) % 360;
  return Math.abs(d - 360) < 1e-9 ? 0 : d;
}

/** The corners of a Box, clockwise from its (unrotated) top-left. */
export function boxQuad(b: Box): Point[] {
  return quadOf(b, b.rotate);
}

/** Axis-aligned bounds of a Box. */
export function boxBounds(b: Box): Rect {
  const q = boxQuad(b);
  const xs = q.map(p => p.x);
  const ys = q.map(p => p.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return {x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y};
}

/** Whether two rects overlap by more than half a pixel. */
export function rectsMeet(a: Rect, b: Rect): boolean {
  return (
    a.x < b.x + b.width - 0.5 &&
    b.x < a.x + a.width - 0.5 &&
    a.y < b.y + b.height - 0.5 &&
    b.y < a.y + a.height - 0.5
  );
}

/** The per-card turns a fan lays out (a vertical fan's quarter-turned
 *  position 0 is its rotated base, not a turn in the fan). */
export function fanTurns(
  cards: ReadonlyArray<{turn?: number}>,
  rotatedBase = false,
): number[] | undefined {
  const turns = cards.map((c, i) =>
    rotatedBase && i === 0 ? 0 : (c.turn ?? 0),
  );

  return turns.some(t => t) ? turns : undefined;
}

/** The lowest point a slot box reaches, turned. */
export function slotBottomOf(b: SlotBox): number {
  return Math.max(
    ...quadOf(
      {x: b.left, y: b.top, width: b.width, height: b.height},
      b.rotate,
      {
        x: b.left + b.origin.x,
        y: b.top + b.origin.y,
      },
    ).map(q => q.y),
  );
}

/** An item's drawn size in a free slot: a token's own, else a card's. */
export function itemSize(
  it: TableItem,
  cw: number,
  ch: number,
  unit: number,
): {width: number; height: number} {
  if (isToken(it)) {
    return {width: it.size.width * unit, height: it.size.height * unit};
  }
  // A quarter turn in a free slot rotates about the centre; the unrotated
  // size stays.
  return {width: cw, height: ch};
}

/** The axis-aligned table-px bounds of an index box on a card. */
export function indexBounds(
  b: SlotBox,
  r: Rect,
  localRotate: number,
  idx: IndexGeometry,
): Rect {
  const local: Rect = {
    x: r.x + idx.left * r.width,
    y: r.y + idx.top * r.height,
    width: idx.width * r.width,
    height: idx.height * r.height,
  };
  const cardCentre = {x: r.x + r.width / 2, y: r.y + r.height / 2};
  const corners = quadOf(local).map(p =>
    toTablePoint(b, rotateAbout(p, cardCentre, localRotate)),
  );
  const xs = corners.map(p => p.x);
  const ys = corners.map(p => p.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return {x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y};
}

/** A top corner of the card clear of its visible index (local px). */
export function badgeCorner(r: Rect, idx: IndexGeometry | null): Point {
  const indexOnLeft = idx === null || idx.left + idx.width / 2 <= 0.5;
  return indexOnLeft ? {x: r.x + r.width, y: r.y} : {x: r.x, y: r.y};
}

/** Card-local badge anchor and index box for a placement (renderOverlay). */
export function localCardGeometry(p: Placement): {
  badgeAnchor: Point;
  indexBox: Rect | null;
} {
  // Undo the box: the anchor in the card's own unrotated frame.
  const c = {x: p.box.x + p.box.width / 2, y: p.box.y + p.box.height / 2};
  const a = rotateAbout(p.badgeAnchor, c, -p.box.rotate);
  const badgeAnchor = {x: a.x - p.box.x, y: a.y - p.box.y};
  let indexBox: Rect | null = null;
  if (p.indexBox) {
    // The table-px index box is axis-aligned bounds; for upright cards it is exact.
    const q = quadOf(p.indexBox).map(pt => rotateAbout(pt, c, -p.box.rotate));
    const xs = q.map(pt => pt.x - p.box.x);
    const ys = q.map(pt => pt.y - p.box.y);
    const x = Math.min(...xs);
    const y = Math.min(...ys);
    indexBox = {x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y};
  }
  return {badgeAnchor, indexBox};
}

/** RegionSpec.attach.also: the unturned box round what the slots that
 *  hold an item (or positional holes) draw, or the first slot's own box
 *  when none does. */
export function unionBox(region: string, ls: readonly SlotLayout[]): SlotBox {
  for (const l of ls) {
    if (!turnsInPlace(l.spec)) {
      throw new Error(
        `table region '${region}' attaches to several slots, but '${l.spec.id}' is turned ${String(l.box.rotate)}°`,
      );
    }
  }
  const held = ls.filter(l => l.positions > 0);
  // A union is in the table's frame: a half-turned slot covers the same box.
  if (held.length === 0) {
    const first = defined(ls[0], `slot for region '${region}'`);
    return {
      ...first.box,
      rotate: 0,
      origin: {x: first.box.width / 2, y: first.box.height / 2},
    };
  }
  const left = Math.min(...held.map(l => l.box.left));
  const top = Math.min(...held.map(l => l.box.top));
  const width = Math.max(...held.map(l => l.box.left + l.box.width)) - left;
  const height = Math.max(...held.map(l => l.box.top + l.box.height)) - top;
  return {
    left,
    top,
    width,
    height,
    rotate: 0,
    origin: {x: width / 2, y: height / 2},
  };
}

// A thin bar along the leading (or trailing) edge of a card's local rect.
function edgeBar(
  e: Rect,
  horizontal: boolean,
  far: boolean,
  thick: number,
): Rect {
  return horizontal
    ? {
        x: (far ? e.x + e.width : e.x) - thick / 2,
        y: e.y,
        width: thick,
        height: e.height,
      }
    : {
        x: e.x,
        y: (far ? e.y + e.height : e.y) - thick / 2,
        width: e.width,
        height: thick,
      };
}

/** The caret box for inserting at `index` (0..items) in a slot: a thin bar at the
 *  leading edge of the card now at `index`, or past the last card. */
export function insertBox(l: SlotLayout, index: number): Box {
  const n = l.items.length;
  if (!(Number.isInteger(index) && index >= 0 && index <= n)) {
    throw new Error(
      `insert index ${String(index)} out of range 0..${String(n)} for slot '${l.spec.id}'`,
    );
  }
  const drawn = l.entries.filter(e => !e.rotatedBase);
  const horizontal =
    l.kind === 'hand' ||
    l.spec.pile === 'fan-right' ||
    l.spec.pile === 'fan-left' ||
    l.kind === 'free';
  const thick = Math.max(4, 0.08 * l.cw);
  if (drawn.length === 0 || l.spec.pile === 'squared') {
    return toTableBox(
      l.box,
      {x: 0, y: 0, width: l.box.width, height: l.box.height},
      0,
    );
  }
  // An insert index counts the slot's items, not its positional holes (the
  // same as the position in a slot without holes).
  const rank = new Map(l.items.map((it, k) => [it.id, k] as const));
  const at = drawn.find(e => rank.get(e.placement.id) === index);
  const reversed = l.spec.pile === 'fan-left' || l.spec.pile === 'fan-up';
  if (at) {
    return toTableBox(
      l.box,
      edgeBar(at.local, horizontal, reversed, thick),
      at.localRotate,
    );
  }
  const after = index >= n;
  const last = defined(
    after ? drawn[drawn.length - 1] : drawn[0],
    `drawn card in slot '${l.spec.id}'`,
  );
  const lead = after !== reversed;
  return toTableBox(
    l.box,
    edgeBar(last.local, horizontal, lead, thick),
    last.localRotate,
  );
}

/** The index box of a deck turned by `turn` degrees (legibility of a turned card). */
export function turnedIndex(index: IndexGeometry, turn: number): IndexGeometry {
  return rotateIndex(index, turn);
}
