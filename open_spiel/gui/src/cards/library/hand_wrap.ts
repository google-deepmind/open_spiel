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

// A hand wrapped onto several rows (hand_geometry.ts says when): balanced,
// row-major, each row a one-row hand of its cards, centred and stacked.
import {itemAt} from './checked.ts';
import {
  arcCards,
  arcRow,
  handRowSizes,
  lineRow,
  type HandCardPlacement,
  type HandGeometry,
  type HandGeometryInput,
} from './hand_geometry.ts';
import {mirrorIndex} from './index_scale.ts';
import {
  indexQuad,
  PAD_FRACTION,
  PAD_MIN_PX,
  quadOf,
  type PlacedCard,
  type Rect,
} from './pile_geometry.ts';

/** Axis-aligned bounds of card placements (rotation included). */
function boundsOf(
  cards: readonly HandCardPlacement[],
  cardWidth: number,
  cardHeight: number,
): Rect {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const c of cards) {
    for (const p of quadOf(
      {x: c.x, y: c.y, width: cardWidth, height: cardHeight},
      c.rotate,
    )) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
  }
  return {x: minX, y: minY, width: maxX - minX, height: maxY - minY};
}

// One row in its own frame: its cards as drawn (lifted), the bounds of its
// cards at rest, and the lowest point of any index box at rest.
interface LaidRow {
  drawn: HandCardPlacement[];
  band: Rect;
  left: number;
  width: number;
  indexBottom: number;
  fits: boolean;
  compressed: boolean;
}

interface RowCards {
  drawn: HandCardPlacement[];
  rest: HandCardPlacement[];
  band: Rect;
  fits: boolean;
  compressed: boolean;
}

function rowCards(
  sub: HandGeometryInput,
  selected: readonly boolean[],
  r: number,
  cardHeight: number,
  lift: number,
): RowCards {
  const {cardWidth, mode} = sub;
  if (mode === 'hand-flat' || mode === 'hand-overlap') {
    const line = lineRow(sub, cardHeight);
    const at = (lifted: boolean): HandCardPlacement[] =>
      line.xs.map((x, k): HandCardPlacement => {
        const corner = line.corners[k];
        return {
          x,
          y: itemAt(line.ys, k) - (lifted && selected[k] ? lift : 0),
          rotate: itemAt(line.turns, k),
          ...(corner && corner !== 'top-left' ? {corner} : {}),
          row: r,
        };
      });
    // The row's band is the upright card's height, as in one row.
    const band = {x: 0, y: 0, width: line.width, height: cardHeight};
    const {fits, compressed} = line;
    return {drawn: at(true), rest: at(false), band, fits, compressed};
  }
  const arc = arcRow(sub, cardHeight);
  const lay = (sel: readonly boolean[]): HandCardPlacement[] =>
    arcCards(arc.gaps, sel, cardWidth, cardHeight, arc.radius, lift).map(c => ({
      ...c,
      row: r,
    }));
  const rest = lay(sub.faceUp.map(() => false));
  return {
    drawn: lay(selected),
    rest,
    band: boundsOf(rest, cardWidth, cardHeight),
    fits: arc.fits,
    compressed: arc.compressed,
  };
}

function layRow(
  sub: HandGeometryInput,
  selected: readonly boolean[],
  r: number,
  cardHeight: number,
  lift: number,
): LaidRow {
  const {cardWidth, deck} = sub;
  const {drawn, rest, band, fits, compressed} = rowCards(
    sub,
    selected,
    r,
    cardHeight,
    lift,
  );
  // Width: the band, and any lifted card that leaves it sideways (an arc
  // card lifts along its own axis).
  const all = boundsOf(drawn, cardWidth, cardHeight);
  const left = Math.min(band.x, all.x);
  const right = Math.max(band.x + band.width, all.x + all.width);
  let indexBottom = -Infinity;
  rest.forEach((c, k) => {
    if (!sub.faceUp[k]) return;
    const placed: PlacedCard = {
      faceUp: true,
      index: c.corner === 'bottom-right' ? mirrorIndex(deck.index) : deck.index,
      rotate: c.rotate,
      rect: {x: c.x, y: c.y, width: cardWidth, height: cardHeight},
    };
    for (const q of indexQuad(placed)) indexBottom = Math.max(indexBottom, q.y);
  });
  return {
    drawn,
    band,
    left,
    width: right - left,
    indexBottom,
    fits,
    compressed,
  };
}

/** A hand wrapped onto `rows` rows: balanced, row-major, each row a one-row
 *  hand of its cards, centred, stacked. */
export function wrappedRows(
  input: HandGeometryInput,
  rows: number,
  cardHeight: number,
): HandGeometry {
  const {faceUp} = input;
  const selected = input.selected ?? faceUp.map(() => false);
  const lift = input.lift ?? 0;
  const rowGap = input.rowGap ?? 0;
  const lifting = lift > 0 && selected.some(Boolean);
  const pad = Math.max(PAD_MIN_PX, PAD_FRACTION * cardHeight);
  let first = 0;
  const laid = handRowSizes(faceUp.length, rows).map((size, r) => {
    const idx = Array.from({length: size}, (_, k) => first + k);
    first += size;
    const rowSelected = idx.map(i => itemAt(selected, i));
    const sub: HandGeometryInput = {
      ...input,
      faceUp: idx.map(i => itemAt(faceUp, i)),
      selected: rowSelected,
      turns: idx.map(i => input.turns?.[i] ?? 0),
    };
    return layRow(sub, rowSelected, r, cardHeight, lift);
  });

  const width = Math.max(...laid.map(l => l.width));
  const cards: HandCardPlacement[] = [];
  let y = lifting ? lift : 0;
  laid.forEach((l, r) => {
    const dx = (width - l.width) / 2 - l.left;
    const dy = y - l.band.y;
    for (const c of l.drawn) cards.push({...c, x: c.x + dx, y: c.y + dy});
    if (r < laid.length - 1) {
      // The next row starts rowGap below this one, and lower still if a
      // lifted card in it would otherwise reach this row's index.
      const clearIndex =
        lifting && Number.isFinite(l.indexBottom)
          ? l.indexBottom - l.band.y + pad + lift
          : 0;
      y += Math.max(l.band.height + rowGap, clearIndex);
    } else {
      y += l.band.height;
    }
  });
  return {
    cards,
    width,
    height: y,
    cardHeight: cardHeight,
    fits: laid.every(l => l.fits),
    compressed: laid.some(l => l.compressed),
    rows: laid.length,
  };
}
