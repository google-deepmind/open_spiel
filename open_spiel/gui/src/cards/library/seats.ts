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

// Seat geometry for tables of hands: generic helpers any layout builds on;
// no game is named here. Units are card widths
// (table.ts). A seat is a hand region sized from the active deck's
// HandMeasure (handMeasure.ts): its preferred room, or what the deck needs,
// whichever is larger, so a hand never compresses past its legibility floor.
import {itemAt} from './checked.ts';
import type {SlotSpec, TableSpec} from './table_spec.ts';
import type {HandMode} from './hand_geometry.ts';
import type {HandMeasure} from './hand_measure.ts';

/** Margin between a seat and the table's edge or the next region. */
export const SEAT_GAP = 0.15;

/** One seat's hand: its mode, size and faces. */
export interface HandSeat {
  mode: HandMode;
  /** Cards in the largest hand this seat holds. */
  cards: number;
  faceUp: boolean;
  selectable?: boolean | undefined;
  /** Preferred room along the fan axis, card widths. */
  room: number;
  rotate?: number | undefined;
  /** Wrap the hand onto up to this many rows (default 1): the seat is
   *  sized for its longest row, its depth for the rows stacked, and its
   *  slot wraps ('auto', maxRows) when the hand does not fit one row. */
  rows?: number | undefined;
}

/** A seat's extent (preferred room, or what the deck needs) and its depth
 *  drawn in that extent (an arc given room spreads, and deepens). */
export function seatSize(
  m: HandMeasure,
  s: HandSeat,
): {extent: number; depth: number} {
  const opts = {
    faceUp: s.faceUp,
    selectable: s.selectable,
    ...(s.rows !== undefined ? {rows: s.rows} : {}),
  };
  const extent = Math.max(s.room, m.extent(s.mode, s.cards, opts));
  return {extent, depth: m.depth(s.mode, s.cards, {...opts, room: extent})};
}

/** The wrap fields of a seat's hand slot: none for one row. */
export function seatWrap(s: HandSeat): Pick<SlotSpec, 'wrap' | 'maxRows'> {
  if (s.rows === undefined || s.rows === 1) return {};
  if (!(Number.isInteger(s.rows) && s.rows > 1)) {
    throw new Error(
      `hand seat rows must be a whole number >= 1, got ${String(s.rows)}`,
    );
  }
  return {wrap: 'auto', maxRows: s.rows};
}

/** A seat's region on screen, card widths: across and down, after any turn. */
export interface SeatBox {
  width: number;
  height: number;
}

/**
 * The screen box a four-seat table's hand takes: a top or bottom seat runs
 * across ({extent, depth}); a side seat turned 90° or 270° runs down
 * ({depth, extent}), an upright one across.
 */
export function seatBox(
  m: HandMeasure,
  s: HandSeat,
  place: 'across' | 'side',
  upright: boolean,
): SeatBox {
  const {extent, depth} = seatSize(m, s);
  return place === 'side' && !upright
    ? {width: depth, height: extent}
    : {width: extent, height: depth};
}

/** The larger of two seat boxes in each direction. */
export function largerBox(a: SeatBox, b: SeatBox): SeatBox {
  return {
    width: Math.max(a.width, b.width),
    height: Math.max(a.height, b.height),
  };
}

/** A table of four seats around a centre. */
export interface FourSeatOptions {
  minSize: {width: number; height: number};
  /** The top seat; `rotate` applies (180 faces a hand of backs to its
   *  seat) unless `upright`. */
  north: HandSeat;
  /** Both side seats: the left turned 90°, the right 270°, unless
   *  `upright`. */
  side: HandSeat;
  /** The bottom seat, drawn upright. */
  south: HandSeat;
  /** Four single-card centre positions, `dx` / `dy` card widths from the
   *  table's middle, with ids `${prefix}n`, `${prefix}e`, `${prefix}s`,
   *  `${prefix}w`. */
  centre: {dx: number; dy: number; prefix: string};
  /** More slots centred on the table's middle (their x and y are ignored). */
  extra?: readonly SlotSpec[] | undefined;
  /** Draw north's and the side hands upright (rotate 0), as the bottom
   *  seat's player reads them: a face-up hand. Turned hands face their own
   *  player, which only suits backs (a turned face reads upside down or
   *  sideways from the bottom seat). */
  upright?: boolean | undefined;
  /** The least screen box (card widths, seatBox) north's and each side
   *  seat's region keeps. Two tables that reserve each other's boxes put
   *  every seat, trick position and the table's size in the same place:
   *  one with hidden hands (turned backs) and one with the same hands face
   *  up (upright), so showing the cards moves nothing. */
  reserve?: {north?: SeatBox; side?: SeatBox} | undefined;
}

// A seat's hand slot, centre-anchored at (x, y).
function seatSlot(
  id: string,
  label: string,
  seat: HandSeat,
  extent: number,
  x: number,
  y: number,
  rotate?: number,
): SlotSpec {
  return {
    id,
    x,
    y,
    pile: seat.mode,
    rotate,
    anchor: 'center',
    maxExtent: extent,
    label,
    ...seatWrap(seat),
  };
}

// The four centre positions round (cx, cy): single squared cards, side by
// side. Nothing here overlaps, so nothing depends on the deck.
function centreSlots(
  prefix: string,
  dx: number,
  dy: number,
  cx: number,
  cy: number,
): SlotSpec[] {
  const at = {n: [0, -dy], e: [dx, 0], s: [0, dy], w: [-dx, 0]} as const;
  return (['n', 'e', 's', 'w'] as const).map((k): SlotSpec => ({
    id: `${prefix}${k}`,
    x: cx + at[k][0],
    y: cy + at[k][1],
    pile: 'squared',
    anchor: 'center',
  }));
}

/**
 * Four seats — 'north', 'west', 'east', 'south' — around a play area of four
 * centre positions. The side hands sit between the top and bottom seats (the
 * table is tall enough for that), so the top and bottom hands may run the
 * full width; the centre positions sit between the side hands. Seats and
 * centre positions are centre-anchored so a hand keeps its seat as it grows
 * or shrinks, and every hand has a maxExtent sized for its largest hand.
 */
export function fourSeatTable(m: HandMeasure, t: FourSeatOptions): TableSpec {
  const given: unknown = t.centre;
  const prefixGiven: unknown =
    given !== null && typeof given === 'object' ? t.centre.prefix : undefined;
  if (typeof prefixGiven !== 'string' || prefixGiven === '') {
    throw new Error(
      `fourSeatTable: centre.prefix is required (a non-empty slot id prefix), got ${JSON.stringify(prefixGiven)}`,
    );
  }
  const GAP = SEAT_GAP;
  const up = !!t.upright;
  // Screen boxes: north runs across; a side seat runs down when turned,
  // across when upright. Each at least its reserve.
  const n = largerBox(
    seatBox(m, t.north, 'across', up),
    t.reserve?.north ?? {width: 0, height: 0},
  );
  const w = largerBox(
    seatBox(m, t.side, 'side', up),
    t.reserve?.side ?? {width: 0, height: 0},
  );
  const s = seatSize(m, t.south);
  const {dx, dy, prefix} = t.centre;
  const centreW = 2 * dx + 1;
  const centreH = 2 * dy + m.cardHeight;
  const width = Math.max(
    t.minSize.width,
    n.width + 2 * GAP,
    s.extent + 2 * GAP,
    centreW + 2 * (w.width + 2 * GAP),
  );
  const height = Math.max(
    t.minSize.height,
    n.height + s.depth + 4 * GAP + Math.max(w.height, centreH),
  );
  const mid = (n.height - s.depth + height) / 2;
  const hand = seatSlot;
  const centre = centreSlots(prefix, dx, dy, width / 2, mid);
  return {
    width,
    height,
    slots: [
      hand(
        'north',
        'North',
        t.north,
        n.width,
        width / 2,
        GAP + n.height / 2,
        up ? 0 : t.north.rotate,
      ),
      hand(
        'west',
        'West',
        t.side,
        up ? w.width : w.height,
        GAP + w.width / 2,
        mid,
        up ? 0 : 90,
      ),
      hand(
        'east',
        'East',
        t.side,
        up ? w.width : w.height,
        width - GAP - w.width / 2,
        mid,
        up ? 0 : 270,
      ),
      hand(
        'south',
        'South',
        t.south,
        s.extent,
        width / 2,
        height - GAP - s.depth / 2,
      ),
      ...centre,
      ...(t.extra ?? []).map(c => ({...c, x: width / 2, y: mid})),
    ],
  };
}

/** A table of 3 to 8 seats in a ring around a centre. */
export interface RingSeatOptions {
  minSize: {width: number; height: number};
  /** Seat slot ids, 3 to 8: the bottom seat first, then in play order
   *  clockwise (up the left, across the top, down the right). */
  ids: readonly string[];
  /** The bottom seat, drawn upright across the bottom. */
  south: HandSeat;
  /** Every other seat: a HandSeat drawn upright (an open hand), or
   *  'stack', a squared pile of backs with a count (a hidden hand), drawn
   *  at `stackScale` (default 1). */
  others: HandSeat | 'stack';
  stackScale?: number | undefined;
  /** Other seats per row across the top (default: all in one row); more
   *  seats wrap onto further rows, still in play order. */
  perRow?: number | undefined;
  /** The band under each other seat (its name tag) and over the bottom
   *  seat: its height, the least column width it needs, and its gap. */
  label: {height: number; minWidth: number; gap: number};
  /** One single-card play position per seat on a ring round the middle,
   *  id `${prefix}${seat id}`: the bottom seat's at the bottom, the others
   *  clockwise from the left, so each card lies toward its seat. `gap`:
   *  least space between two of them, card widths. */
  centre: {prefix: string; gap: number};
  /** Seat the first and the last other seat level with the ring, at the
   *  table's left and right edges, and the rest across the top: play order
   *  still runs up the left, across the top, down the right. Needs at
   *  least three other seats. A table whose width is its row of name
   *  columns gets two columns narrower. */
  sides?: boolean | undefined;
  /** Keep the label band over the bottom seat (default true). False: the
   *  game draws the bottom seat's label beside its hand, and the table is
   *  that band lower. */
  labelOverSouth?: boolean | undefined;
  /** A box kept clear beside the ring on the left ('left'), in a band
   *  over it ('above'), or in the bottom-left corner beside the bottom
   *  seat ('corner', with as much room kept right of that seat): a stock,
   *  a turn-up. */
  aside?:
    | {width: number; height: number; place: 'left' | 'above' | 'corner'}
    | undefined;
  /** A band kept clear under the ring (a status line), card widths high. */
  under?: number | undefined;
}

/** What a ring table keeps clear for the game's own regions. */
export interface RingSeatTable {
  spec: TableSpec;
  /** Width of each other seat's column (its name tag's room). */
  column: number;
  /** The ring of play positions, its bounding box. */
  ring: {x: number; y: number; width: number; height: number};
  /** The box kept for `aside`, top-left. */
  aside: {x: number; y: number} | null;
  /** The band under the ring, top-left and width. */
  under: {x: number; y: number; width: number};
}

/** Half-axes of the smallest ring of `n` single cards (1 x h card widths)
 *  at angles 90° + k·360°/n (y down) in which no two are closer than `gap`. */
export function ringRadii(
  n: number,
  h: number,
  gap: number,
): {rx: number; ry: number} {
  if (!(Number.isInteger(n) && n >= 3 && n <= 8)) {
    throw new Error(
      `ringRadii: n must be a whole number 3..8, got ${String(n)}`,
    );
  }
  const at = (rx: number, ry: number) =>
    Array.from({length: n}, (_, k) => {
      const a = ((90 + (k * 360) / n) * Math.PI) / 180;
      return {x: rx * Math.cos(a), y: ry * Math.sin(a)};
    });
  const ASPECT = 0.85;
  for (let rx = 0.5; rx < 8; rx += 0.01) {
    const ry = rx * ASPECT;
    const p = at(rx, ry);
    let clear = true;
    for (let i = 0; i < n && clear; i++) {
      for (let j = i + 1; j < n && clear; j++) {
        const a = itemAt(p, i);
        const b = itemAt(p, j);
        if (Math.abs(a.x - b.x) < 1 + gap && Math.abs(a.y - b.y) < h + gap) {
          clear = false;
        }
      }
    }
    if (clear) return {rx, ry};
  }
  throw new Error(
    `ringRadii: no ring of ${String(n)} cards ${String(h)} high fits`,
  );
}

function checkRing(t: RingSeatOptions): number {
  const n = t.ids.length;
  if (!(n >= 3 && n <= 8)) {
    throw new Error(`ringSeatTable: 3 to 8 seats, got ${String(n)}`);
  }
  if (new Set(t.ids).size !== n) {
    throw new Error(
      `ringSeatTable: seat ids must be unique, got ${t.ids.join(', ')}`,
    );
  }
  const centre: unknown = t.centre;
  const prefix: unknown =
    centre !== null && typeof centre === 'object' ? t.centre.prefix : undefined;
  if (typeof prefix !== 'string' || prefix === '') {
    throw new Error('ringSeatTable: centre.prefix is required');
  }
  if (t.sides && n - 1 < 3) {
    throw new Error(
      `ringSeatTable: sides needs at least three other seats, got ${String(n - 1)}`,
    );
  }
  const stackScale = t.stackScale ?? 1;
  if (!(stackScale > 0 && stackScale <= 1)) {
    throw new Error(
      `ringSeatTable: stackScale must be in (0, 1], got ${String(stackScale)}`,
    );
  }
  return n;
}

// The sizes and places a ring table is drawn from, in card widths.
interface Ring {
  n: number;
  sideIds: string[];
  top: string[];
  perRow: number;
  stackScale: number;
  box: {w: number; h: number};
  column: number;
  rowH: number;
  south: {extent: number; depth: number};
  rx: number;
  ry: number;
  ringW: number;
  ringH: number;
  width: number;
  height: number;
  mid: {x: number; y: number};
  asideTop: number;
  underY: number;
  southTop: number;
}

function ringWidth(
  t: RingSeatOptions,
  south: {extent: number},
  column: number,
  perRow: number,
  ringW: number,
): number {
  const GAP = SEAT_GAP;
  const aside = t.aside;
  const middleW =
    ringW +
    (aside?.place === 'left' ? 2 * (aside.width + GAP) : 0) +
    (t.sides ? 2 * (column + GAP) : 0);
  return Math.max(
    t.minSize.width,
    perRow * column + (perRow + 1) * GAP,
    south.extent + 2 * GAP,
    middleW + 2 * GAP,
    aside?.place === 'above' ? aside.width + 2 * GAP : 0,
    aside?.place === 'corner' ? south.extent + 2 * (aside.width + 2 * GAP) : 0,
  );
}

function ringLayout(m: HandMeasure, t: RingSeatOptions): Ring {
  const n = checkRing(t);
  const GAP = SEAT_GAP;
  const h = m.cardHeight;
  const others = t.ids.slice(1);
  const sideIds = t.sides
    ? [itemAt(others, 0), itemAt(others, others.length - 1)]
    : [];
  const top = t.sides ? others.slice(1, -1) : others;
  const perRow = Math.max(1, Math.min(top.length, t.perRow ?? top.length));
  const rows = Math.ceil(top.length / perRow);
  const stackScale = t.stackScale ?? 1;
  let box = {w: stackScale, h: h * stackScale};
  if (t.others !== 'stack') {
    const size = seatSize(m, t.others);
    box = {w: size.extent, h: size.depth};
  }
  const column = Math.max(box.w, t.label.minWidth);
  const rowH = box.h + t.label.gap + t.label.height;
  const south = seatSize(m, t.south);
  const {rx, ry} = ringRadii(n, h, t.centre.gap);
  const ringW = 2 * rx + 1;
  const ringH = 2 * ry + h;
  const aside = t.aside;
  const width = ringWidth(t, south, column, perRow, ringW);
  const rowsBottom = GAP + rows * rowH + (rows - 1) * GAP;
  const asideTop = rowsBottom + GAP;
  const ringTop =
    aside?.place === 'above' ? asideTop + aside.height + GAP : rowsBottom + GAP;
  const middleH = Math.max(
    ringH,
    aside?.place === 'left' ? aside.height : 0,
    t.sides ? rowH : 0,
  );
  const mid = {x: width / 2, y: ringTop + middleH / 2};
  const underY = ringTop + middleH + GAP;
  const southBand = Math.max(
    south.depth +
      (t.labelOverSouth === false ? 0 : t.label.height + t.label.gap),
    aside?.place === 'corner' ? aside.height : 0,
  );
  const height = Math.max(
    t.minSize.height,
    underY + (t.under ?? 0) + GAP + southBand + GAP,
  );
  const southTop = height - GAP - south.depth;
  return {
    n,
    sideIds,
    top,
    perRow,
    stackScale,
    box,
    column,
    rowH,
    south,
    rx,
    ry,
    ringW,
    ringH,
    width,
    height,
    mid,
    asideTop,
    underY,
    southTop,
  };
}

function ringSlots(t: RingSeatOptions, g: Ring): SlotSpec[] {
  const GAP = SEAT_GAP;
  const {box, column, rowH, perRow, top, width, mid} = g;
  const seat = (id: string, x: number, y: number): SlotSpec =>
    t.others === 'stack'
      ? {
          id,
          x,
          y,
          pile: 'squared',
          anchor: 'center',
          stagger: 2,
          countBadge: true,
          faceDown: true,
          scale: g.stackScale,
        }
      : {
          id,
          x,
          y,
          pile: t.others.mode,
          anchor: 'center',
          maxExtent: box.w,
          ...seatWrap(t.others),
        };
  const slots: SlotSpec[] = [];
  // Side seats: the seat and its label band centred on the ring's middle.
  const sideY = mid.y - rowH / 2 + box.h / 2;
  const [left, right] = g.sideIds;
  if (left !== undefined) slots.push(seat(left, GAP + column / 2, sideY));
  top.forEach((id, i) => {
    const r = Math.floor(i / perRow);
    const inRow = Math.min(perRow, top.length - r * perRow);
    const c = i - r * perRow;
    const x0 = width / 2 - (inRow * column + (inRow - 1) * GAP) / 2;
    slots.push(
      seat(
        id,
        x0 + c * (column + GAP) + column / 2,
        GAP + r * (rowH + GAP) + box.h / 2,
      ),
    );
  });
  if (right !== undefined) {
    slots.push(seat(right, width - GAP - column / 2, sideY));
  }
  slots.unshift({
    id: itemAt(t.ids, 0),
    x: width / 2,
    y: g.southTop + g.south.depth / 2,
    pile: t.south.mode,
    anchor: 'center',
    maxExtent: g.south.extent,
    ...seatWrap(t.south),
  });
  t.ids.forEach((id, k) => {
    const a = ((90 + (k * 360) / g.n) * Math.PI) / 180;
    slots.push({
      id: `${t.centre.prefix}${id}`,
      x: mid.x + g.rx * Math.cos(a),
      y: mid.y + g.ry * Math.sin(a),
      pile: 'squared',
      anchor: 'center',
    });
  });
  return slots;
}

/**
 * N seats round a ring of play positions: the bottom seat across
 * the bottom, the others in play order in rows across the top (left to
 * right is clockwise from the bottom seat), or with `sides` the first and
 * last of them level with the ring at the edges, each with a label band
 * under it, the ring of play positions in the middle with an optional box
 * beside or over it (or in the bottom-left corner) and a band under it.
 * Every seat and play position is centre-anchored, so a hand keeps its place as it grows or shrinks.
 */
export function ringSeatTable(
  m: HandMeasure,
  t: RingSeatOptions,
): RingSeatTable {
  const GAP = SEAT_GAP;
  const g = ringLayout(m, t);
  const {width, height, mid, ringW, ringH} = g;
  const aside = t.aside;
  const ring = {
    x: mid.x - ringW / 2,
    y: mid.y - ringH / 2,
    width: ringW,
    height: ringH,
  };
  return {
    spec: {width, height, slots: ringSlots(t, g)},
    column: g.column,
    ring,
    aside: !aside
      ? null
      : aside.place === 'left'
        ? {x: ring.x - GAP - aside.width, y: mid.y - aside.height / 2}
        : aside.place === 'corner'
          ? {x: GAP, y: height - GAP - aside.height}
          : {x: width / 2 - aside.width / 2, y: g.asideTop},
    under: {x: GAP, y: g.underY, width: width - 2 * GAP},
  };
}
