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

// The parts every hearts table shares (hearts_layout.ts): the region ids, the
// hand shapes and dresses, the seats, the name and message boxes, and the
// slot and region literals the tables repeat. Units are card widths.
import {
  seatWrap,
  type HandMeasure,
  type HandSeat,
  type RegionSpec,
  type SlotSpec,
} from '../../cards/library/index.ts';

/** Region ids for the centre (pass button, "No Pass", game over) and the
 *  status message under the trick. */
export const CENTRE_REGION = 'centre';
/** The status message's region id. */
export const MESSAGE_REGION = 'message';

/** Your hand's shape (Hearts view setting): 'flat', a straight fan (the
 *  default), or 'curved', an arc where the deck
 *  allows (see southSeat). */
export type HeartsHandShape = 'flat' | 'curved';
/** Every hand shape, in the order the View panel offers them. */
export const HEARTS_HAND_SHAPES: readonly HeartsHandShape[] = [
  'flat',
  'curved',
];
/** The hand shape until one is chosen. */
export const DEFAULT_HEARTS_HAND_SHAPE: HeartsHandShape = 'flat';

/** Hidden opponents are a tight fan of backs (no index to keep clear),
 *  turned to face their seats; open hands (analysis, show all cards) take
 *  the room their faces need, upright: north's one row (or two), each side
 *  hand wrapped onto rows (SeatRows.side) so it is a column no wider than
 *  the side band allows. Your hand is an arc here, flattened for the 'flat'
 *  shape (southSeat). */
export const SEATS: Record<
  'closed' | 'open',
  {north: HandSeat; side: HandSeat; south: HandSeat}
> = {
  closed: {
    north: {
      mode: 'hand-overlap',
      cards: 13,
      faceUp: false,
      room: 4.2,
      rotate: 180,
    },
    side: {mode: 'hand-overlap', cards: 13, faceUp: false, room: 4.2},
    south: {
      mode: 'hand-arc',
      cards: 13,
      faceUp: true,
      selectable: true,
      room: 6.4,
    },
  },
  open: {
    north: {
      mode: 'hand-overlap',
      cards: 13,
      faceUp: true,
      selectable: true,
      room: 5.6,
    },
    side: {
      mode: 'hand-overlap',
      cards: 13,
      faceUp: true,
      selectable: true,
      room: 0,
    },
    south: {
      mode: 'hand-arc',
      cards: 13,
      faceUp: true,
      selectable: true,
      room: 6.4,
    },
  },
};

/** Which table is drawn: 'closed' for play (backs only), 'hidden' and
 *  'open' for analysis (one geometry, backs or faces; see the header). */
export type HeartsDress = 'closed' | 'hidden' | 'open';
/** Every dress. */
export const HEARTS_DRESSES: readonly HeartsDress[] = [
  'closed',
  'hidden',
  'open',
];

/**
 * Your hand: a straight fan for the 'flat' shape. For 'curved', an arc when
 * the deck's arc fits its preferred room, otherwise a straight fan. A deck
 * whose index is centred on the card (e.g. 'simple') needs an arc half again
 * as wide as its room, and an arc that wide is also far deeper than a fan:
 * the table would be drawn at the smallest card, or scroll, in the game panel.
 */
export function southSeat(
  m: HandMeasure,
  arc: HandSeat,
  shape: HeartsHandShape,
): HandSeat {
  if (shape === 'flat') return {...arc, mode: 'hand-overlap'};
  if ((shape as string) !== 'curved') {
    throw new Error(
      `hearts layout: unknown hand shape ${JSON.stringify(shape)}`,
    );
  }
  const rows = arc.rows !== undefined ? {rows: arc.rows} : {};
  if (
    m.extent(arc.mode, arc.cards, {
      faceUp: arc.faceUp,
      selectable: arc.selectable,
      ...rows,
    }) <= arc.room
  ) {
    return arc;
  }
  return {...arc, mode: 'hand-overlap'};
}

/** Rows the hands of the four-seat table wrap onto: the side hands (which
 *  set the table's height, or upright its side bands' width), north's and
 *  yours (which set its width). */
export interface SeatRows {
  side: number;
  north: number;
  south: number;
}

/** The four-seat table's trick: a diamond, card widths from the middle. */
export const TRICK = {dx: 1.15, dy: 1.25};
/** The four-seat table's least size, card widths. */
export const MIN_SIZE = {width: 6.6, height: 9.4};
/** The least room, card widths, between west's or east's trick card and its
 *  hand. fourSeatTable alone leaves SEAT_GAP there when the trick, not a
 *  hand, sets the table's width, so a side seat's card touched its hand and
 *  barely travelled (a video playtest found it); north and south
 *  keep about 0.85. */
export const TRICK_SIDE_CLEAR = 0.4;

/** Name labels, card widths: height, gap from the hand, width range. A name
 *  is drawn at its text's width inside its box and truncates only past `max`. */
export const NAME = {height: 0.36, gap: 0.12, min: 1.1, max: 2.4};
/** Region classes (hearts.css) that set a name to one end of its box;
 *  without one it is centred. */
export const NAME_CLASS = {
  left: 'hearts-name--left',
  right: 'hearts-name--right',
} as const;
/** The status message's box, card widths: width, height, gap round it. */
export const MESSAGE = {width: 4.4, height: 0.42, gap: 0.08};

/** A name box `w` card widths wide, at most NAME.max; less than NAME.min
 *  throws (the layout left no room for the name). */
export function nameWidth(w: number, id: string): number {
  if (w < NAME.min) {
    throw new Error(
      `hearts layout: only ${w.toFixed(2)} card widths for the ${id} name`,
    );
  }
  return Math.min(NAME.max, w);
}

/** Compact table: an opponent's stack, card widths of room for its depth
 *  layers (CardStyle squaredStagger px each) beyond the card. */
export const STACK = {layers: 2, pad: 0.1};
/** The compact table is at least this wide: room for three names across
 *  the top (NAME.min each, see compactClosed). */
export const COMPACT_MIN_WIDTH = 4.4;
/** Open phone table: the trick as one row, card widths between cards. */
export const TRICK_ROW_GAP = 0.1;

/** `seat` wrapped onto `rows` rows (one row: as it is). */
export function withRows(seat: HandSeat, rows: number): HandSeat {
  return rows > 1 ? {...seat, rows} : seat;
}

/** Your hand along the bottom, centred, its top at `top`. */
export function southSlot(
  width: number,
  top: number,
  south: {extent: number; depth: number},
  seat: HandSeat,
): SlotSpec {
  return {
    id: 'south',
    x: width / 2,
    y: top + south.depth / 2,
    pile: seat.mode,
    anchor: 'center',
    maxExtent: south.extent,
    label: 'South',
    ...seatWrap(seat),
  };
}

/** Your name over the left end of your hand. */
export function southName(extent: number): RegionSpec {
  return {
    id: 'name-south',
    width: nameWidth(extent, 'south'),
    height: NAME.height,
    attach: {slot: 'south', side: 'top', gap: NAME.gap, align: 'start'},
    className: NAME_CLASS.left,
  };
}

/** The centre (pass button, "No Pass", game over) over the trick, `w` by `h`
 *  centred across `width`, its top at `y`; it covers nothing. */
export function centreRegion(
  width: number,
  y: number,
  w: number,
  h: number,
): RegionSpec {
  return {
    id: CENTRE_REGION,
    x: width / 2 - w / 2,
    y,
    width: w,
    height: h,
    occludes: false,
  };
}

/** The status message, `w` wide centred across `width`, its top at `y`. */
export function messageRegion(width: number, y: number, w: number): RegionSpec {
  return {
    id: MESSAGE_REGION,
    x: width / 2 - w / 2,
    y,
    width: w,
    height: MESSAGE.height,
  };
}

/** The trick as one row across `width` at `y`, in the order of `seats`. */
export function trickRow(
  seats: ReadonlyArray<'n' | 'e' | 's' | 'w'>,
  width: number,
  y: number,
): SlotSpec[] {
  const trickX = (i: number): number =>
    width / 2 + (i - 1.5) * (1 + TRICK_ROW_GAP);
  return seats.map((k, i): SlotSpec => ({
    id: `trick-${k}`,
    x: trickX(i),
    y,
    pile: 'squared',
    anchor: 'center',
    tags: ['hide-empty'],
  }));
}
