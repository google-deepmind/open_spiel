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

// Pile geometry: where each card of a pile sits, derived from the deck's
// IndexGeometry so that overlaps never hide a face-up card's rank or suit.
//
// THE LEGIBILITY RULE. In a fanned pile each card is covered by the
// cards after it. A face-up card stays identifiable only while its index box
// (deck.index, fractions of the card) is uncovered, so the offset to the next
// card along the fan axis is at least the far edge of the index box plus
// padding. That is a floor, not a preference: a pile given a maximum extent
// compresses towards it and, if it still does not fit, reports `fits: false`
// so the caller overflows visibly. Nothing here ever returns an offset below
// the floor.
//
// Every preference is a CardStyle knob: the face-up and fan-right
// offsets are multiples of the floor (clamped to >= 1), the face-down offset a
// fraction of the card, the squared stagger px per layer, and the index box is
// the deck's at the style's indexScale. The padding stays fixed: it is part of
// the floor's definition, not a preference.
//
// Survey gaps. fan-left and fan-up lay each card over the LEFT/TOP
// of the one before, so the covered card shows its bottom-right corner: the
// floor is measured on the 180° twin (cornerIndex), which only decks that
// declare Deck.rotatedTwin have. A `null` face is a positional hole: it keeps
// a face-up-sized slot so the cards after it never move when it fills. A
// rotated base is a card turned a quarter under an upright pile; the pile
// sits beside its index, never over it.
//
// A card turned in a fan (a tapped card in a row in play) takes its turned
// footprint along the axis, and the step to the next card clears ITS turned
// index (the corner that stays freer: the twin too, on decks that draw one),
// so the index rule holds for it exactly as for an upright card.
import {itemAt} from './checked.ts';
import {
  cornerIndex,
  quarterTurnIndex,
  type IndexCorner,
} from './index_scale.ts';
import type {Rect} from './pile_cover.ts';
import {turnedAxis, type AxisConstants} from './pile_turned.ts';
import {DEFAULT_CARD_STYLE} from './style.ts';
import type {Deck, IndexGeometry} from './types.ts';

export {
  findCoveredIndexes,
  indexQuad,
  indexRect,
  placePile,
  quadOf,
  type PlacedCard,
  type Point,
  type Rect,
} from './pile_cover.ts';

/** How a pile is laid out: squared, or fanned in a direction. */
export type PileMode =
  'squared' | 'fan-down' | 'fan-right' | 'fan-left' | 'fan-up';

/** The corner index a mode leaves uncovered on each face-up card. */
export function indexCornerFor(mode: PileMode): IndexCorner {
  return mode === 'fan-left' || mode === 'fan-up' ? 'bottom-right' : 'top-left';
}

/** Fan axis of a mode; squared piles stack along both. */
export function fanAxis(mode: PileMode): 'down' | 'right' {
  return mode === 'fan-right' || mode === 'fan-left' ? 'right' : 'down';
}

/** What a pile is laid out from: the deck, cards, mode and room. */
export interface PileGeometryInput {
  deck: Pick<Deck, 'aspect' | 'index' | 'indexAt' | 'rotatedTwin'> & {
    id?: string | undefined;
  };
  /** Rendered card width in px. */
  cardWidth: number;
  /** Face of each card, bottom of the pile first; null is a positional hole
   *  (fan modes only), which keeps a face-up-sized slot. */
  faceUp: ReadonlyArray<boolean | null>;
  /** The bottom card lies turned a quarter anticlockwise under an upright
   *  pile (squared, fan-down, fan-up). */
  rotatedBase?: boolean | undefined;
  mode: PileMode;
  /** Available length along the fan axis, in px; the pile compresses to fit. */
  maxExtent?: number | undefined;
  /** Fan modes: each card's turn in play, degrees clockwise (0/90/180/270;
   *  missing entries are 0). A turned card fans by its turned footprint and
   *  its turned index. Squared piles turn cards in place (not here). */
  turns?: readonly number[] | undefined;
  /** Squared piles only: draw up to this many depth layers under the top card. */
  stagger?: number | undefined;
  /** Face-up offset, × the legibility floor; clamped to >= 1. CardStyle
   *  fan.faceUp (fan.horizontal for fan-right); DEFAULT_CARD_STYLE's when omitted. */
  spread?: number | undefined;
  /** Face-down offset, × the card's size along the fan axis. CardStyle fan.faceDown. */
  faceDownSpread?: number | undefined;
  /** Squared piles: px per depth layer. CardStyle squaredStagger. */
  squaredStagger?: number | undefined;
  /** CardStyle.indexScale; honoured by decks with indexAt. */
  indexScale?: number | undefined;
}

/** A card's slot in the pile's frame, px. A rotated slot is the turned
 *  card's bounding box (cardHeight wide, cardWidth tall). */
export interface PileSlot extends Rect {
  rotated: boolean;
  /** A card turned in a fan: the rect is the UPRIGHT card centred on its
   *  turned footprint, drawn rotated `turn` degrees about its centre, and
   *  `corner` is the index corner it keeps visible. */
  turn?: number | undefined;
  corner?: IndexCorner | undefined;
}

/** A laid-out pile: its cards' placements and its size. */
export interface PileGeometry {
  /** Distance of each card along the fan axis from the first, in px. For
   *  fan-left and fan-up the first card is at the far end of the frame. */
  offsets: number[];
  /** Where each card is drawn, holes included. */
  slots: PileSlot[];
  /** The frame the slots fill. */
  width: number;
  height: number;
  /** The corner index each upright face-up card keeps visible. */
  corner: IndexCorner;
  /** Size of the laid-out pile along the fan axis (height for fan-down). */
  extent: number;
  /** Size across the fan axis. */
  crossExtent: number;
  cardHeight: number;
  /** False when even minimum offsets exceed maxExtent: the caller must scroll. */
  fits: boolean;
  /** True when offsets were reduced below their preferred values. */
  compressed: boolean;
  /** The floors actually applied (px). */
  minFaceUpOffset: number;
  minFaceDownOffset: number;
}

/** Gap between the index box's far edge and the next card, as a fraction of
 *  the card's size along the axis, with a pixel floor. */
export const PAD_FRACTION = 0.03;
/** The least padding around a pile, px. */
export const PAD_MIN_PX = 2;
/** Face-down cards have no index; the offset only shows that a card is there.
 *  Compression squeezes them to this fraction (or the style's, if smaller),
 *  never below the pixel floor. */
const FACE_DOWN_MIN_FRACTION = 0.05;
const FACE_DOWN_MIN_PX = 3;

/** A card's height at `cardWidth`, from its deck's aspect. */
export function cardHeightFor(
  deck: Pick<Deck, 'aspect'>,
  cardWidth: number,
): number {
  return cardWidth / deck.aspect;
}

/**
 * The smallest offset along `axis` that keeps a face-up card's index box fully
 * uncovered by the next card: the index box's far edge, plus padding, in px.
 */
export function indexClearance(
  index: IndexGeometry,
  cardWidth: number,
  cardHeight: number,
  axis: 'down' | 'right',
): number {
  const size = axis === 'down' ? cardHeight : cardWidth;
  const far =
    axis === 'down' ? index.top + index.height : index.left + index.width;
  return Math.ceil(far * size + Math.max(PAD_MIN_PX, PAD_FRACTION * size));
}

function assertInput(input: PileGeometryInput): void {
  const {
    cardWidth,
    maxExtent,
    deck,
    spread,
    faceDownSpread,
    squaredStagger,
    indexScale,
    mode,
    faceUp,
  } = input;
  if (mode === 'squared' && faceUp.some(f => f === null)) {
    throw new Error('a squared pile has no positional holes');
  }
  if (input.turns) {
    if (mode === 'squared' && input.turns.some(t => t)) {
      throw new Error('a squared pile turns cards in place, not through turns');
    }
    input.turns.forEach((t, i) => {
      if (![0, 90, 180, 270].includes(t)) {
        throw new Error(
          `pile turns[${String(i)}] must be 0/90/180/270, got ${String(t)}`,
        );
      }
      if (t && faceUp[i] === null) {
        throw new Error(
          `pile turns[${String(i)}]: a positional hole cannot turn`,
        );
      }
    });
  }
  if (input.rotatedBase) {
    if (mode === 'fan-right' || mode === 'fan-left') {
      throw new Error(`a rotated base needs a vertical pile, got ${mode}`);
    }
    if (faceUp.length === 0 || faceUp[0] === null) {
      throw new Error('a rotated base needs a card at position 0');
    }
  }
  for (const [name, v] of [
    ['spread', spread],
    ['faceDownSpread', faceDownSpread],
    ['squaredStagger', squaredStagger],
    ['indexScale', indexScale],
  ] as const) {
    if (v !== undefined && !Number.isFinite(v)) {
      throw new Error(`pile ${name} must be a finite number, got ${String(v)}`);
    }
  }
  if (!(cardWidth > 0)) {
    throw new Error(`pile cardWidth must be > 0, got ${String(cardWidth)}`);
  }
  if (!(deck.aspect > 0)) {
    throw new Error(`pile deck aspect must be > 0, got ${String(deck.aspect)}`);
  }
  if (maxExtent !== undefined && !(maxExtent >= 0)) {
    throw new Error(`pile maxExtent must be >= 0, got ${String(maxExtent)}`);
  }
}

/** Where every card of a pile sits, for the pile's mode and room. */
export function computePileGeometry(input: PileGeometryInput): PileGeometry {
  assertInput(input);
  if (input.rotatedBase) return withRotatedBase(input);
  const g = computeAxis(input);
  return {...withoutTurns(g), ...frame(g, input.mode, input.cardWidth)};
}

function withoutTurns(g: AxisGeometry): Omit<AxisGeometry, 'turned'> {
  const geometry = {...g};
  delete geometry.turned;
  return geometry;
}

/** A pile laid out along its axis, before its slots are framed. */
export type AxisGeometry = Omit<PileGeometry, 'slots' | 'width' | 'height'> & {
  /** Present when a card turns in the fan: each card's turn and index corner. */
  turned?: {turns: number[]; corners: IndexCorner[]} | undefined;
};

/** Lay the slots out from the axis offsets. fan-left/fan-up run from the far end. */
function frame(
  g: AxisGeometry,
  mode: PileMode,
  cardWidth: number,
): Pick<PileGeometry, 'slots' | 'width' | 'height'> {
  const horizontal = fanAxis(mode) === 'right';
  const reversed = mode === 'fan-left' || mode === 'fan-up';
  const turned = g.turned;
  if (!turned) {
    const last = g.offsets.at(-1) ?? 0;
    const slots = g.offsets.map(o => {
      const along = reversed ? last - o : o;
      const x = mode === 'squared' ? o : horizontal ? along : 0;
      const y = mode === 'squared' ? o : horizontal ? 0 : along;
      return {x, y, width: cardWidth, height: g.cardHeight, rotated: false};
    });
    return {
      slots,
      width: horizontal ? g.extent : g.crossExtent,
      height: horizontal ? g.crossExtent : g.extent,
    };
  }
  // Turned cards: every card centred on the cross axis's middle line; each
  // footprint starts at its offset (from the far end for fan-left/fan-up).
  const h = g.cardHeight;
  const slots = g.offsets.map((o, i) => {
    const t = itemAt(turned.turns, i);
    const q = t === 90 || t === 270;
    const fw = q ? h : cardWidth;
    const fh = q ? cardWidth : h;
    const along = horizontal ? fw : fh;
    const start = reversed ? g.extent - o - along : o;
    const cx = horizontal ? start + fw / 2 : g.crossExtent / 2;
    const cy = horizontal ? g.crossExtent / 2 : start + fh / 2;
    const slot: PileSlot = {
      x: cx - cardWidth / 2,
      y: cy - h / 2,
      width: cardWidth,
      height: h,
      rotated: false,
    };
    if (t) {
      slot.turn = t;
      slot.corner = itemAt(turned.corners, i);
    }
    return slot;
  });
  return {
    slots,
    width: horizontal ? g.extent : g.crossExtent,
    height: horizontal ? g.crossExtent : g.extent,
  };
}

/** The base card turned a quarter anticlockwise; the upright rest is shifted
 *  across so the base's index (now bottom-left) stays clear, and the base is
 *  centred on the first upright slot along the axis. */
function withRotatedBase(input: PileGeometryInput): PileGeometry {
  const {deck, cardWidth, mode} = input;
  const rest = computeAxis({
    ...input,
    faceUp: input.faceUp.slice(1),
    turns: input.turns?.slice(1),
    rotatedBase: false,
  });
  const restFrame = frame(rest, mode, cardWidth);
  const h = rest.cardHeight;
  const index = cornerIndex(
    deck,
    input.indexScale ?? DEFAULT_CARD_STYLE.indexScale,
    'top-left',
  );
  const shift =
    input.faceUp.length > 1
      ? indexClearance(quarterTurnIndex(index), h, cardWidth, 'right')
      : 0;
  // The first upright card sits where a lone card would: at the far end for fan-up.
  const firstY = mode === 'fan-up' ? Math.max(0, rest.extent - h) : 0;
  const base: PileSlot = {
    x: 0,
    y: firstY + (h - cardWidth) / 2,
    width: h,
    height: cardWidth,
    rotated: true,
  };
  const slots = [
    base,
    ...restFrame.slots.map(sl => ({...sl, x: sl.x + shift})),
  ];
  const width = Math.max(h, shift + restFrame.width);
  const height = restFrame.height;
  return {
    ...withoutTurns(rest),
    offsets: [base.y, ...rest.offsets],
    slots,
    width,
    height,
    extent: height,
    crossExtent: width,
  };
}

function computeAxis(input: PileGeometryInput): AxisGeometry {
  const {deck, cardWidth, faceUp, mode} = input;
  const cardHeight = cardHeightFor(deck, cardWidth);
  const axis = fanAxis(mode);
  const corner = indexCornerFor(mode);
  const axisSize = axis === 'down' ? cardHeight : cardWidth;
  const crossSize = axis === 'down' ? cardWidth : cardHeight;
  // Floors first: a factor below 1 would hide the index, so it is clamped.
  const defaultSpread =
    axis === 'right'
      ? DEFAULT_CARD_STYLE.fan.horizontal
      : DEFAULT_CARD_STYLE.fan.faceUp;
  const upFactor = Math.max(1, input.spread ?? defaultSpread);
  const downFraction = Math.max(
    0,
    input.faceDownSpread ?? DEFAULT_CARD_STYLE.fan.faceDown,
  );

  // The twin's clearance from the right/bottom edge equals the top-left
  // box's from the left/top (180° symmetry), so it is measured on that box
  // (mirroring twice would drift by an ulp and move the ceil). cornerIndex
  // is still asked for the twin: it throws for a deck that has none.
  const scale = input.indexScale ?? DEFAULT_CARD_STYLE.indexScale;
  cornerIndex(deck, scale, corner);
  const index = cornerIndex(deck, scale, 'top-left');
  const minUp = indexClearance(index, cardWidth, cardHeight, axis);
  const minDownFraction = Math.min(FACE_DOWN_MIN_FRACTION, downFraction);
  const minDown = Math.min(
    minUp,
    Math.max(FACE_DOWN_MIN_PX, Math.round(minDownFraction * axisSize)),
  );
  const base = {
    corner,
    cardHeight,
    minFaceUpOffset: minUp,
    minFaceDownOffset: minDown,
  };
  const k: AxisConstants = {
    axis,
    axisSize,
    crossSize,
    upFactor,
    downFraction,
    minUp,
    minDown,
    index,
  };
  if (faceUp.length === 0) {
    return {
      ...base,
      offsets: [],
      extent: axisSize,
      crossExtent: crossSize,
      fits: true,
      compressed: false,
    };
  }
  if (mode === 'squared') return squaredAxis(input, base);
  if ((input.turns ?? []).some(t => t)) return turnedAxis(input, base, k);
  return straightAxis(input, base, k);
}

type AxisBase = Pick<
  AxisGeometry,
  'corner' | 'cardHeight' | 'minFaceUpOffset' | 'minFaceDownOffset'
>;

// A squared pile: only the top card is visible by design; depth layers are
// cosmetic.
function squaredAxis(input: PileGeometryInput, base: AxisBase): AxisGeometry {
  const {faceUp, cardWidth} = input;
  const n = faceUp.length;
  const staggerPx = Math.max(
    0,
    input.squaredStagger ?? DEFAULT_CARD_STYLE.squaredStagger,
  );
  const layers = Math.max(0, Math.min(input.stagger ?? 0, n - 1));
  const offsets = faceUp.map(
    (_, i) => Math.max(0, i - (n - 1 - layers)) * staggerPx,
  );
  const extra = layers * staggerPx;
  return {
    ...base,
    offsets,
    extent: base.cardHeight + extra,
    crossExtent: cardWidth + extra,
    fits: true,
    compressed: false,
  };
}

// A fan of upright cards. Each card except the last contributes one gap,
// sized by ITS face: the card being covered is the one whose index must stay
// visible. A hole's gap is face-up sized, so filling it moves nothing.
function straightAxis(
  input: PileGeometryInput,
  base: AxisBase,
  k: AxisConstants,
): AxisGeometry {
  const {faceUp, maxExtent} = input;
  const {minUp, minDown, axisSize} = k;
  const n = faceUp.length;
  const upGap = (i: number): boolean => faceUp[i] !== false;
  let upGaps = 0;
  let downGaps = 0;
  for (let i = 0; i < n - 1; i++) {
    if (upGap(i)) upGaps++;
    else downGaps++;
  }
  const prefUp = Math.max(minUp, Math.round(minUp * k.upFactor));
  const prefDown = Math.max(minDown, Math.round(k.downFraction * axisSize));

  let up = prefUp;
  let down = prefDown;
  let fits = true;
  let compressed = false;
  if (maxExtent !== undefined) {
    const budget = maxExtent - axisSize;
    const total = (u: number, d: number): number => u * upGaps + d * downGaps;
    if (total(up, down) > budget) {
      compressed = true;
      // Squeeze face-down gaps first (they carry no information), then face-up.
      if (downGaps > 0) {
        const fit = Math.floor((budget - up * upGaps) / downGaps);
        down = Math.max(minDown, Math.min(prefDown, fit));
      }
      if (total(up, down) > budget && upGaps > 0) {
        const fit = Math.floor((budget - down * downGaps) / upGaps);
        up = Math.max(minUp, Math.min(prefUp, fit));
      }
      fits = total(up, down) <= budget;
    }
  }

  const offsets: number[] = [0];
  for (let i = 0; i < n - 1; i++) {
    offsets.push(itemAt(offsets, i) + (upGap(i) ? up : down));
  }
  const extent = itemAt(offsets, n - 1) + axisSize;
  return {...base, offsets, extent, crossExtent: k.crossSize, fits, compressed};
}
