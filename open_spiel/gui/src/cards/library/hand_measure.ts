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

// Hand measures: how much room a hand needs, for the deck and style a
// table is drawn with. The hand-and-table recipes are laid out from these, the
// hand-aware twin of TableClearance: where a deck draws its index
// decides how tightly a hand may overlap, so a seat cannot be a fixed number
// of card widths. A corner index fans a 13-card hand into ~4.5 card widths;
// the centred 'simple' block needs ~9. A recipe states its preferred room and
// takes the larger of that and the measure, so a deck that needs more gets a
// bigger table (fitted smaller, scrolling below the minimum card width) and
// never a covered index.
import type {Deck, IndexGeometry} from './types.ts';
import type {CardStyle} from './style.ts';
import {
  handRowSizes,
  type HandCardShape,
  type HandMode,
} from './hand_geometry.ts';
import {handGeometryFor} from './hand.tsx';
import {tableClearance} from './table_spec.ts';
import {deckIndex, mirrorIndex} from './index_scale.ts';
import {indexClearance} from './pile_geometry.ts';

/** How a hand is measured: its faces, selection, lift and rows. */
export interface HandMeasureOptions {
  /** Face-up hand (the default) or backs. */
  faceUp?: boolean | undefined;
  /** The hand can hold lifted (selected) cards: room for the lift is kept. */
  selectable?: boolean | undefined;
  /** With selectable: the most a card is lifted, px (a game tag's lift, such
   *  as a picked card's). Default CardStyle.selection.lift. */
  lift?: number | undefined;
  /** depth() only: the room the hand is drawn in, card widths (its slot's
   *  maxExtent). An arc spreads into the room it is given, up to the
   *  style's fan spacing, and grows deeper as it does, so a seat given more
   *  than the floor needs the depth at that room. Omitted: the depth at the
   *  legibility floor. */
  room?: number | undefined;
  /** The hand wrapped onto this many rows (a whole number >= 1, default 1;
   *  the slot's maxRows with wrap 'auto'): extent() is its longest row's
   *  (handRowSizes), depth() the rows stacked as the slot draws them, in
   *  that extent (or `room`). */
  rows?: number | undefined;
  /** Between rows, × card width (default CardStyle.hand.rowGap): the
   *  slot's rowGap, if it sets one. */
  rowGap?: number | undefined;
}

/** Room for hands at one deck and style, in card widths, holding for every
 *  card width in the range it was measured over. */
export interface HandMeasure {
  /** Least extent along the fan axis of an n-card hand at legible spacing. */
  extent(mode: HandMode, n: number, opts?: HandMeasureOptions): number;
  /** Its depth across the fan axis, lift headroom included when selectable. */
  depth(mode: HandMode, n: number, opts?: HandMeasureOptions): number;
  /** Card height. */
  cardHeight: number;
  /** Most a fan-right pile advances per face-up card (TableClearance.fanRight). */
  fanRight: number;
  /** Most a fan-down pile advances per face-up card as <Pile> lays it out
   *  (the index floor times CardStyle fan.faceUp, rounding included), so a
   *  recipe can leave room for a fanned column (a hand laid out by suit). */
  fanDown: number;
  /** The index boxes a card is identified by, as drawn at this style: the
   *  top-left one, and its bottom-right twin where the deck draws one
   *  (Deck.rotatedTwin). A card is legible while any one is in view (a
   *  layout that overlaps cards on purpose, as the overlapping trick). */
  indexes: readonly IndexGeometry[];
}

/** Rounded up to 1e-3 so a spec built from it clears the px geometry it came from. */
function up(v: number): number {
  return Math.ceil(v * 1000 - 1e-6) / 1000;
}

interface Need {
  extent: number;
  depth: number;
}

// The options of one measure, checked, with the style's defaults applied.
interface Checked {
  faceUp: boolean;
  selectable: boolean;
  lift: number;
  rows: number;
  rowGap: number;
}

function checkOptions(
  style: CardStyle,
  n: number,
  opts: HandMeasureOptions,
): Checked {
  if (!(Number.isInteger(n) && n >= 0)) {
    throw new Error(
      `measureHands card count must be a whole number >= 0, got ${String(n)}`,
    );
  }
  const lift = opts.lift ?? style.selection.lift;
  if (!(Number.isFinite(lift) && lift >= 0)) {
    throw new Error(
      `measureHands lift must be a number of px >= 0, got ${String(opts.lift)}`,
    );
  }
  const rows = opts.rows ?? 1;
  if (!(Number.isInteger(rows) && rows >= 1)) {
    throw new Error(
      `measureHands rows must be a whole number >= 1, got ${String(opts.rows)}`,
    );
  }
  const rowGap = opts.rowGap ?? style.hand.rowGap;
  if (!(Number.isFinite(rowGap) && rowGap >= 0)) {
    throw new Error(
      `measureHands rowGap must be a number of card widths >= 0, got ${String(opts.rowGap)}`,
    );
  }
  return {
    faceUp: opts.faceUp ?? true,
    selectable: !!opts.selectable,
    lift,
    rows,
    rowGap,
  };
}

function checkRoom(room: number, given: number | undefined): void {
  if (!(Number.isFinite(room) && room >= 0)) {
    throw new Error(
      `measureHands room must be a number of card widths >= 0, got ${String(given)}`,
    );
  }
}

// Measures hands over every whole-px card width in range, caching each.
class Measurer {
  private readonly cache = new Map<string, Need>();

  private readonly deck: Deck;
  private readonly style: CardStyle;
  private readonly ws: readonly number[];

  constructor(deck: Deck, style: CardStyle, ws: readonly number[]) {
    this.deck = deck;
    this.style = style;
    this.ws = ws;
  }

  measure(mode: HandMode, n: number, opts: HandMeasureOptions = {}): Need {
    const o = checkOptions(this.style, n, opts);
    return o.rows > 1
      ? this.wrapped(mode, n, o, opts.room)
      : this.single(mode, n, o, opts.room);
  }

  // Wrapped: the longest row sets the extent; the depth is the whole hand
  // drawn wrapping in that extent (or the room given).
  private wrapped(
    mode: HandMode,
    n: number,
    o: Checked,
    given: number | undefined,
  ): Need {
    const {faceUp, selectable, lift, rows, rowGap} = o;
    const longest = handRowSizes(n, rows)[0] ?? 0;
    const row = this.measure(mode, longest, {faceUp, selectable, lift});
    const room = given ?? row.extent;
    checkRoom(room, given);
    const key = `${mode}:${String(n)}:${String(faceUp)}:${String(selectable)}:${String(lift)}:${String(room)}:rows${String(rows)}:${String(rowGap)}`;
    const hit = this.cache.get(key);
    if (hit) return hit;
    const wrap = {wrap: 'auto' as const, maxRows: rows, rowGap};
    const cards: HandCardShape[] = Array.from({length: n}, () => ({faceUp}));
    let depth = 0;
    for (const w of this.ws) {
      const maxExtent = room * w;
      const shape = {mode, cardWidth: w, maxExtent, lift, ...wrap};
      const g = handGeometryFor(this.deck, this.style, cards, shape);
      if (!g.fits) {
        throw new Error(
          `measureHands: ${String(n)} ${mode} cards do not fit ${String(rows)} rows in ${String(room)} card widths at ${String(w)}px`,
        );
      }
      let height = g.height;
      if (selectable && n > 0) {
        // Every card lifted is the deepest selection: each row gains the
        // lift, and the row pitch the most it grows.
        const lifted = cards.map(c => ({...c, selected: true}));
        const l = handGeometryFor(this.deck, this.style, lifted, shape);
        height = Math.max(height + lift, l.height);
      }
      depth = Math.max(depth, height / w);
    }
    const m = {extent: row.extent, depth: up(depth)};
    this.cache.set(key, m);
    return m;
  }

  private single(
    mode: HandMode,
    n: number,
    o: Checked,
    given: number | undefined,
  ): Need {
    const {faceUp, selectable, lift} = o;
    const room = given ?? 0;
    checkRoom(room, given);
    const key = `${mode}:${String(n)}:${String(faceUp)}:${String(selectable)}:${String(lift)}:${String(room)}`;
    const hit = this.cache.get(key);
    if (hit) return hit;
    const cards: HandCardShape[] = Array.from({length: n}, () => ({faceUp}));
    const lifted: HandCardShape[] = cards.map(c => ({...c, selected: true}));
    let extent = 0;
    let depth = 0;
    for (const w of this.ws) {
      // Room 0 (maxExtent 0): the hand compresses to its legibility floor;
      // given room, an arc spreads into it (deeper) up to its fan spacing.
      const maxExtent = room * w;
      const shape = {mode, cardWidth: w, maxExtent, wrap: 'never' as const};
      const g = handGeometryFor(this.deck, this.style, cards, shape);
      let width = g.width;
      let height = g.height;
      if (selectable && n > 0) {
        // Hands are symmetric, so the wider of none and all lifted is the
        // widest of any selection; a lifted card moves at most `lift` px,
        // so the unlifted depth plus the lift bounds any selection's.
        const l = handGeometryFor(this.deck, this.style, lifted, {
          ...shape,
          lift,
        });
        width = Math.max(width, l.width);
        height = Math.max(height + lift, l.height);
      }
      extent = Math.max(extent, width / w);
      depth = Math.max(depth, height / w);
    }
    const m = {extent: up(extent), depth: up(depth)};
    this.cache.set(key, m);
    return m;
  }
}

/** The room the hands of a table need at a card width. */
export function measureHands(
  deck: Deck,
  style: CardStyle,
  widths: {min: number; max: number},
): HandMeasure {
  const {min, max} = widths;
  if (!(
    Number.isFinite(min) &&
    min > 0 &&
    Number.isFinite(max) &&
    max >= min
  )) {
    throw new Error(
      `measureHands widths must satisfy 0 < min <= max, got ${String(min)}..${String(max)}`,
    );
  }
  // Every whole-px width in range, and both ends: px paddings, the px lift
  // and rounding make the need in card widths vary with the width.
  const ws: number[] = [min];
  for (let w = Math.floor(min) + 1; w < max; w++) ws.push(w);
  if (max > min) ws.push(max);
  const measurer = new Measurer(deck, style, ws);
  return {
    extent: (mode, n, opts) =>
      measurer.measure(mode, n, {...opts, room: undefined}).extent,
    depth: (mode, n, opts) => measurer.measure(mode, n, opts).depth,
    cardHeight: up(1 / deck.aspect),
    fanRight: tableClearance(deck, style, widths).fanRight,
    fanDown: fanDownStep(deck, style, widths),
    indexes: indexBoxes(deck, style),
  };
}

/** The most a fan-down pile advances per face-up card over every whole-px
 *  card width in range, card widths (HandMeasure.fanDown): <Pile>'s own
 *  offset, the index clearance times CardStyle fan.faceUp, rounded to px. */
export function fanDownStep(
  deck: Pick<Deck, 'aspect' | 'index' | 'indexAt'>,
  style: Pick<CardStyle, 'indexScale' | 'fan'>,
  widths: {min: number; max: number},
): number {
  const {min, max} = widths;
  if (!(
    Number.isFinite(min) &&
    min > 0 &&
    Number.isFinite(max) &&
    max >= min
  )) {
    throw new Error(
      `fanDownStep widths must satisfy 0 < min <= max, got ${String(min)}..${String(max)}`,
    );
  }
  const index = deckIndex(deck, style.indexScale);
  const factor = Math.max(1, style.fan.faceUp);
  let step = 0;
  for (let w = min; w <= max; w = w === min ? Math.floor(min) + 1 : w + 1) {
    const floor = indexClearance(index, w, w / deck.aspect, 'down');
    step = Math.max(step, Math.max(floor, Math.round(floor * factor)) / w);
  }
  return up(step);
}

/** The index boxes a card is identified by at a style (HandMeasure.indexes). */
export function indexBoxes(
  deck: Pick<Deck, 'index' | 'indexAt' | 'rotatedTwin'>,
  style: Pick<CardStyle, 'indexScale'>,
): IndexGeometry[] {
  const i = deckIndex(deck, style.indexScale);
  return deck.rotatedTwin ? [i, mirrorIndex(i)] : [i];
}
