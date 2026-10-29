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

// Hand geometry: where each card of a player's hand sits — flat,
// overlapped, or on an arc (rotated cards on a curve) — derived from the
// deck's IndexGeometry under the same legibility rule as pileGeometry.ts:
// a face-up card's index box is never covered by the cards after it.
//
// Flat hands do not overlap. Overlapped hands are a fan-right pile and reuse
// its offsets. Arc hands rotate every card about one pivot below the hand; the
// smallest legible angle between neighbours is solved against the rotated
// index box (findCoveredIndexes handles rotation), so the rule holds for any
// deck's index shape rather than for the corner box an analytic formula
// would assume.
//
// A selected card is lifted by `lift` px: straight up in flat and overlapped
// hands, outward along its own axis on an arc. Neither moves the edge that
// covers the neighbour's index (a card's left edge slides along itself), so a
// lift never changes the spacing and the hand does not reflow on selection.
//
// Turned cards (a card turned 90/180/270 in a hand: a tapped or sideways
// card). Flat and overlapped hands take them exactly as a fan-right row does:
// a turned card takes its turned footprint along the hand, and the step past
// it clears its turned index (pileGeometry turns), so the legibility rule
// holds for it as for an upright card. An arc hand refuses them: a turned
// card on an arc lies across its neighbours, and no single arc step keeps
// both its index and theirs clear.
//
// Wrapped hands (wrap 'auto'). A hand that cannot fit maxExtent in one row,
// even compressed to its legibility floor, wraps onto the fewest rows (at
// most maxRows) whose every row fits. Rows are balanced and row-major: the
// first cards fill the top row, and each row is laid out exactly as a
// one-row hand of its cards (flat, overlapped, or its own smaller arc), so
// every card keeps the spacing and index clearance it has in one row. Rows
// are centred and stacked `rowGap` px apart; they never overlap at rest. A
// lifted card rises into the row above it only as far as that row's lowest
// index box (plus padding) allows: the row pitch grows if the lift would
// otherwise reach an index. If even maxRows rows cannot fit, the hand keeps
// maxRows rows at legible spacing and reports fits: false (it scrolls).
import type {Deck} from './types.ts';
import {itemAt} from './checked.ts';
import {wrappedRows} from './hand_wrap.ts';
import {mirrorIndex, type IndexCorner} from './index_scale.ts';
import {
  cardHeightFor,
  computePileGeometry,
  findCoveredIndexes,
  quadOf,
  type PlacedCard,
} from './pile_geometry.ts';

/** How a hand is laid out: flat, overlapped or on an arc. */
export type HandMode = 'hand-flat' | 'hand-overlap' | 'hand-arc';
/** Every hand mode. */
export const HAND_MODES: readonly HandMode[] = [
  'hand-flat',
  'hand-overlap',
  'hand-arc',
];

/** Whether `mode` is a hand mode. */
export function isHandMode(mode: string): mode is HandMode {
  return (HAND_MODES as readonly string[]).includes(mode);
}

/** 'auto': a hand that cannot fit its maxExtent in one row wraps onto more
 *  rows (up to maxRows); 'never': it compresses, then scrolls. */
export type HandWrap = 'auto' | 'never';
/** Every hand wrap setting. */
export const HAND_WRAPS: readonly HandWrap[] = ['auto', 'never'];

/** What a hand's geometry needs of a card: its face, and whether it is
 *  selected (lifted). A TableCard maps to it with isFaceUp and its tags. */
export interface HandCardShape {
  faceUp: boolean;
  selected?: boolean | undefined;
}

/** What a hand is laid out from: the deck, cards, mode and room. */
export interface HandGeometryInput {
  /** `rotatedTwin` lets a turned card keep its bottom-right twin visible
   *  where that stays freer (as in a fan). */
  deck: Pick<Deck, 'aspect' | 'index'> &
    Partial<Pick<Deck, 'rotatedTwin' | 'id'>>;
  /** Rendered card width in px. */
  cardWidth: number;
  /** Face of each card, leftmost (bottom of the paint order) first. */
  faceUp: readonly boolean[];
  /** Lifted cards; same length as faceUp when given. */
  selected?: readonly boolean[] | undefined;
  mode: HandMode;
  /** Available width in px; overlapped and arc hands compress to fit. */
  maxExtent?: number | undefined;
  /** Face-up spacing × the legibility floor (CardStyle.fan.horizontal);
   *  values below 1 are clamped to 1. */
  spread?: number | undefined;
  /** Face-down spacing × card width, the fan axis of a hand (CardStyle.fan.faceDown). */
  faceDownSpread?: number | undefined;
  /** Selected-card lift in px (CardStyle.selection.lift). */
  lift?: number | undefined;
  /** Flat hands: px between cards. */
  gap?: number | undefined;
  /** Arc hands: distance from the pivot to each card's centre, × card height. */
  arcRadius?: number | undefined;
  /** Flat and overlapped hands: each card's turn, degrees clockwise
   *  (0/90/180/270; missing entries are 0). Arc hands throw on a turn. */
  turns?: readonly number[] | undefined;
  /** 'auto': wrap onto up to maxRows rows when one row cannot fit
   *  maxExtent at the legibility floor. Default 'never'. */
  wrap?: HandWrap | undefined;
  /** Most rows a wrapping hand takes, a whole number >= 1. Required with
   *  wrap 'auto'. */
  maxRows?: number | undefined;
  /** px between the rows of a wrapped hand (default 0). */
  rowGap?: number | undefined;
}

/** One card of a laid-out hand: its rectangle, turn and index box. */
export interface HandCardPlacement {
  /** Top-left of the card's unrotated box, px, in the hand's own box. */
  x: number;
  y: number;
  /** Degrees clockwise about the card's centre (the arc angle, or a turn). */
  rotate: number;
  /** A turned card: the index corner it keeps visible (default top-left). */
  corner?: IndexCorner | undefined;
  /** The row the card is in, 0 = top (a one-row hand: 0). Rows are
   *  row-major: row r holds the cards after every card of rows 0..r-1. */
  row: number;
}

/** A laid-out hand: its cards' placements and its size. */
export interface HandGeometry {
  cards: HandCardPlacement[];
  /** The hand's bounding box, px, including lift headroom. */
  width: number;
  height: number;
  cardHeight: number;
  /** False when even minimum spacing exceeds maxExtent: the caller must scroll. */
  fits: boolean;
  /** True when spacing was reduced below its preferred value. */
  compressed: boolean;
  /** Rows drawn: 1, or more for a wrapped hand. */
  rows: number;
}

/** An arc hand's radius, × card width. */
export const DEFAULT_ARC_RADIUS = 3;
/** The gap between cards of a flat hand, × card width. */
export const DEFAULT_FLAT_GAP_FRACTION = 0.08;
const DEFAULT_SPREAD = 1.3;
const DEFAULT_FACE_DOWN_SPREAD = 0.1;
/** Face-down neighbours on an arc: at least this many px apart at the top edge. */
const FACE_DOWN_MIN_PX = 3;
/** Clearance between an index box and the next card on an arc, px. */
const ARC_PAD_PX = 2;
const DEG = Math.PI / 180;

function assertInput(input: HandGeometryInput): void {
  const {cardWidth, maxExtent, deck, faceUp, selected, arcRadius, lift, gap} =
    input;
  if (!(cardWidth > 0)) {
    throw new Error(`hand cardWidth must be > 0, got ${String(cardWidth)}`);
  }
  if (!(deck.aspect > 0)) {
    throw new Error(`hand deck aspect must be > 0, got ${String(deck.aspect)}`);
  }
  if (maxExtent !== undefined && !(maxExtent >= 0)) {
    throw new Error(`hand maxExtent must be >= 0, got ${String(maxExtent)}`);
  }
  if (selected && selected.length !== faceUp.length) {
    throw new Error(
      `hand selected has ${String(selected.length)} entries for ${String(faceUp.length)} cards`,
    );
  }
  if (arcRadius !== undefined && !(arcRadius > 0)) {
    throw new Error(`hand arcRadius must be > 0, got ${String(arcRadius)}`);
  }
  if (lift !== undefined && !(lift >= 0)) {
    throw new Error(`hand lift must be >= 0, got ${String(lift)}`);
  }
  if (gap !== undefined && !(gap >= 0)) {
    throw new Error(`hand gap must be >= 0, got ${String(gap)}`);
  }
  const {wrap, maxRows, rowGap} = input;
  if (wrap !== undefined && !HAND_WRAPS.includes(wrap)) {
    throw new Error(
      `hand wrap must be one of ${HAND_WRAPS.join('/')}, got ${JSON.stringify(wrap)}`,
    );
  }
  if (maxRows !== undefined && !(Number.isInteger(maxRows) && maxRows >= 1)) {
    throw new Error(
      `hand maxRows must be a whole number >= 1, got ${String(maxRows)}`,
    );
  }
  if (wrap === 'auto' && maxRows === undefined) {
    throw new Error(
      `hand wrap 'auto' needs maxRows (the most rows it may take)`,
    );
  }
  if (rowGap !== undefined && !(Number.isFinite(rowGap) && rowGap >= 0)) {
    throw new Error(
      `hand rowGap must be a finite number >= 0, got ${String(rowGap)}`,
    );
  }
  if (input.turns) {
    if (input.turns.length > faceUp.length) {
      throw new Error(
        `hand turns has ${String(input.turns.length)} entries for ${String(faceUp.length)} cards`,
      );
    }
    input.turns.forEach((t, i) => {
      if (![0, 90, 180, 270].includes(t)) {
        throw new Error(
          `hand turns[${String(i)}] must be 0/90/180/270, got ${String(t)}`,
        );
      }
      if (t && input.mode === 'hand-arc') {
        throw new Error(
          `hand turns[${String(i)}] is ${String(t)}: an arc hand cannot carry a turned card (a turned card on an arc lies across its neighbours); use 'hand-overlap' or 'hand-flat', or a fan slot`,
        );
      }
    });
  }
}

/** Card placements as PlacedCards for the legibility check, in paint order. */
export function placeHand(
  g: HandGeometry,
  cardWidth: number,
  faceUp: readonly boolean[],
  index: Deck['index'],
): PlacedCard[] {
  return g.cards.map((c, i) => ({
    faceUp: itemAt(faceUp, i),
    // A turned card keeps the corner the geometry chose (its 180° twin
    // is the mirrored box).
    index: c.corner === 'bottom-right' ? mirrorIndex(index) : index,
    rotate: c.rotate,
    rect: {x: c.x, y: c.y, width: cardWidth, height: g.cardHeight},
  }));
}

/** Translate placements so the bounding box of every card starts at 0,0. */
function normalise(
  cards: HandCardPlacement[],
  cardWidth: number,
  cardHeight: number,
): {cards: HandCardPlacement[]; width: number; height: number} {
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
  return {
    cards: cards.map(c => ({...c, x: c.x - minX, y: c.y - minY})),
    width: maxX - minX,
    height: maxY - minY,
  };
}

/** Arc placements for per-gap angles (degrees), centred on 0°. */
export function arcCards(
  gaps: readonly number[],
  selected: readonly boolean[],
  cardWidth: number,
  cardHeight: number,
  radius: number,
  lift: number,
): HandCardPlacement[] {
  const total = gaps.reduce((s, g) => s + g, 0);
  let angle = -total / 2;
  return selected.map((sel, i) => {
    if (i > 0) angle += itemAt(gaps, i - 1);
    const r = radius + (sel ? lift : 0);
    const cx = r * Math.sin(angle * DEG);
    const cy = -r * Math.cos(angle * DEG);
    // The middle card of an odd hand sums to ~1e-15°: that is 0, not a turn.
    return {
      x: cx - cardWidth / 2,
      y: cy - cardHeight / 2,
      rotate: Math.abs(angle) < 1e-9 ? 0 : angle,
      row: 0,
    };
  });
}

/**
 * The smallest angle (degrees) between two arc neighbours that keeps the first
 * one's index box, grown by ARC_PAD_PX, clear of the second card. Every
 * neighbouring pair on the arc is congruent, so one pair decides it.
 */
export function minArcStep(
  index: Deck['index'],
  cardWidth: number,
  cardHeight: number,
  radius: number,
): number {
  const padded = (w: number, h: number) => ({
    top: Math.max(0, index.top - ARC_PAD_PX / h),
    left: Math.max(0, index.left - ARC_PAD_PX / w),
    width: index.width + (2 * ARC_PAD_PX) / w,
    height: index.height + (2 * ARC_PAD_PX) / h,
  });
  const box = padded(cardWidth, cardHeight);
  const clear = (step: number) => {
    const pair = arcCards(
      [step],
      [false, false],
      cardWidth,
      cardHeight,
      radius,
      0,
    );
    const placed: PlacedCard[] = pair.map((c, i) => ({
      faceUp: i === 0,
      index: box,
      rotate: c.rotate,
      rect: {x: c.x, y: c.y, width: cardWidth, height: cardHeight},
    }));
    return findCoveredIndexes(placed).length === 0;
  };
  let hi = 1;
  while (!clear(hi)) {
    hi *= 2;
    if (hi > 180) {
      throw new Error(
        `no legible arc step for index ${JSON.stringify(index)} at radius ${String(radius)}`,
      );
    }
  }
  let lo = 0;
  for (let k = 0; k < 40; k++) {
    const mid = (lo + hi) / 2;
    if (clear(mid)) hi = mid;
    else lo = mid;
  }
  return hi;
}

/** Cards per row of an n-card hand wrapped onto `rows` rows: balanced, the
 *  longer rows first (row-major: row r takes the next sizes[r] cards). */
export function handRowSizes(n: number, rows: number): number[] {
  if (!(Number.isInteger(n) && n >= 0)) {
    throw new Error(
      `handRowSizes card count must be a whole number >= 0, got ${String(n)}`,
    );
  }
  if (!(Number.isInteger(rows) && rows >= 1)) {
    throw new Error(
      `handRowSizes rows must be a whole number >= 1, got ${String(rows)}`,
    );
  }
  const r = Math.max(1, Math.min(rows, n));
  const base = Math.floor(n / r);
  const extra = n % r;
  return Array.from({length: r}, (_, k) => base + (k < extra ? 1 : 0));
}

/** One row of a flat or overlapped hand: each card's upright rect x and y
 *  (before lift), its index corner, and the row's width along the hand. */
export interface LineRow {
  xs: number[];
  ys: number[];
  corners: Array<IndexCorner | undefined>;
  turns: number[];
  width: number;
  fits: boolean;
  compressed: boolean;
}

/** One row of a flat or overlapped hand, laid out. */
export function lineRow(input: HandGeometryInput, cardHeight: number): LineRow {
  const {deck, cardWidth, faceUp, mode, maxExtent} = input;
  const n = faceUp.length;
  const spread = Math.max(1, input.spread ?? DEFAULT_SPREAD);
  const faceDownSpread = input.faceDownSpread ?? DEFAULT_FACE_DOWN_SPREAD;
  const turns = faceUp.map((_, i) => input.turns?.[i] ?? 0);
  const turned = turns.some(t => t);
  if (mode === 'hand-flat') {
    const gap = input.gap ?? Math.round(DEFAULT_FLAT_GAP_FRACTION * cardWidth);
    // A quarter-turned card is cardHeight wide on the row, centred on
    // its footprint; flat cards never overlap, so every index shows.
    const along = turns.map(t =>
      t === 90 || t === 270 ? cardHeight : cardWidth,
    );
    const xs: number[] = [];
    let start = 0;
    along.forEach((a, i) => {
      xs.push(start + (a - cardWidth) / 2);
      start += a + (i < n - 1 ? gap : 0);
    });
    return {
      xs,
      ys: faceUp.map(() => 0),
      corners: faceUp.map(() => undefined),
      turns,
      width: start,
      fits: maxExtent === undefined || start <= maxExtent,
      compressed: false,
    };
  }
  const pile = computePileGeometry({
    deck,
    cardWidth,
    faceUp,
    mode: 'fan-right',
    maxExtent,
    spread,
    faceDownSpread,
    ...(turned ? {turns} : {}),
  });
  return {
    xs: pile.slots.map(sl => sl.x),
    ys: pile.slots.map(sl => sl.y),
    corners: pile.slots.map(sl => sl.corner),
    turns,
    width: pile.extent,
    fits: pile.fits,
    compressed: pile.compressed,
  };
}

/** One arc row: the gap angles (degrees) that fit maxExtent, compressed
 *  toward the legibility floor as needed, and the pivot radius. */
export interface ArcRow {
  gaps: number[];
  radius: number;
  fits: boolean;
  compressed: boolean;
}

/** One arc row, laid out. */
export function arcRow(input: HandGeometryInput, cardHeight: number): ArcRow {
  const {deck, cardWidth, faceUp, maxExtent} = input;
  const n = faceUp.length;
  const selected = input.selected ?? faceUp.map(() => false);
  const lift = input.lift ?? 0;
  const spread = Math.max(1, input.spread ?? DEFAULT_SPREAD);
  const faceDownSpread = input.faceDownSpread ?? DEFAULT_FACE_DOWN_SPREAD;
  // Each gap is sized by the face of the card it covers, like a pile.
  const radius = (input.arcRadius ?? DEFAULT_ARC_RADIUS) * cardHeight;
  const outer = radius + cardHeight / 2;
  const minUp = minArcStep(deck.index, cardWidth, cardHeight, radius);
  const minDown = Math.min(minUp, FACE_DOWN_MIN_PX / outer / DEG);
  const prefUp = minUp * spread;
  const prefDown = Math.min(
    prefUp,
    Math.max(minDown, (faceDownSpread * cardWidth) / outer / DEG),
  );
  const gapsAt = (t: number) =>
    faceUp
      .slice(0, n - 1)
      .map(up =>
        up ? minUp + t * (prefUp - minUp) : minDown + t * (prefDown - minDown),
      );
  const widthAt = (t: number) =>
    normalise(
      arcCards(gapsAt(t), selected, cardWidth, cardHeight, radius, lift),
      cardWidth,
      cardHeight,
    ).width;

  let t = 1;
  let fits = true;
  let compressed = false;
  if (maxExtent !== undefined && widthAt(1) > maxExtent) {
    compressed = true;
    if (widthAt(0) > maxExtent) {
      t = 0;
      fits = false;
    } else {
      let lo = 0;
      let hi = 1;
      for (let k = 0; k < 30; k++) {
        const mid = (lo + hi) / 2;
        if (widthAt(mid) <= maxExtent) lo = mid;
        else hi = mid;
      }
      t = lo;
    }
  }
  return {gaps: gapsAt(t), radius, fits, compressed};
}

/** A one-row hand. */
function oneRow(input: HandGeometryInput, cardHeight: number): HandGeometry {
  const {cardWidth, faceUp, mode} = input;
  const selected = input.selected ?? faceUp.map(() => false);
  const lift = input.lift ?? 0;
  if (mode === 'hand-flat' || mode === 'hand-overlap') {
    const row = lineRow(input, cardHeight);
    const top = selected.some(Boolean) ? lift : 0;
    const cards = row.xs.map((x, i): HandCardPlacement => ({
      x,
      y: top + itemAt(row.ys, i) - (selected[i] ? lift : 0),
      rotate: itemAt(row.turns, i),
      ...(row.corners[i] && row.corners[i] !== 'top-left'
        ? {corner: row.corners[i]}
        : {}),
      row: 0,
    }));
    return {
      cards,
      width: row.width,
      height: cardHeight + top,
      cardHeight,
      fits: row.fits,
      compressed: row.compressed,
      rows: 1,
    };
  }
  const arc = arcRow(input, cardHeight);
  const {cards, width, height} = normalise(
    arcCards(arc.gaps, selected, cardWidth, cardHeight, arc.radius, lift),
    cardWidth,
    cardHeight,
  );
  return {
    cards,
    width,
    height,
    cardHeight,
    fits: arc.fits,
    compressed: arc.compressed,
    rows: 1,
  };
}

/** Where every card of a hand sits, for the hand's mode and room. */
export function computeHandGeometry(input: HandGeometryInput): HandGeometry {
  assertInput(input);
  const {deck, cardWidth, faceUp} = input;
  const cardHeight = cardHeightFor(deck, cardWidth);
  const n = faceUp.length;

  if (n === 0) {
    return {
      cards: [],
      width: cardWidth,
      height: cardHeight,
      cardHeight,
      fits: true,
      compressed: false,
      rows: 1,
    };
  }
  const one = oneRow(input, cardHeight);
  if (input.wrap !== 'auto' || one.fits || input.maxExtent === undefined) {
    return one;
  }
  // The fewest rows that fit; if none does, the most rows allowed, at
  // legible spacing (the caller scrolls).
  const most = Math.min(input.maxRows ?? 1, n);
  let g = one;
  for (let rows = 2; rows <= most; rows++) {
    g = wrappedRows(input, rows, cardHeight);
    if (g.fits) return g;
  }
  return g;
}
