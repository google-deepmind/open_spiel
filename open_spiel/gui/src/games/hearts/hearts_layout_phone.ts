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

// The hearts phone tables (hearts_layout.ts): the compact table for hidden
// opponents (stacks across the top), and for open hands the columns across
// the top or the four corners. Same slots and regions as the four-seat
// table, so the game's items and region content do not change.
import {
  SEAT_GAP,
  seatSize,
  seatWrap,
  type HandMeasure,
  type HandSeat,
  type RegionSpec,
  type SlotSpec,
  type TableVariant,
} from '../../cards/library/index.ts';
import {
  COMPACT_MIN_WIDTH,
  MESSAGE,
  NAME,
  NAME_CLASS,
  SEATS,
  STACK,
  TRICK,
  TRICK_ROW_GAP,
  centreRegion,
  messageRegion,
  nameWidth,
  southName,
  southSeat,
  southSlot,
  trickRow,
  withRows,
  type HeartsHandShape,
} from './hearts_layout_parts.ts';

/** An opponent's squared stack of backs with a count badge (compactClosed). */
function stackSlot(id: string, label: string, x: number, y: number): SlotSpec {
  return {
    id,
    x,
    y,
    pile: 'squared',
    anchor: 'center',
    stagger: STACK.layers,
    countBadge: true,
    label,
  };
}

/** An open column's name, over its first row (compactOpen). */
function columnName(id: string, extent: number): RegionSpec {
  return {
    id: `name-${id}`,
    width: nameWidth(extent, id),
    height: NAME.height,
    attach: {slot: id, side: 'top', gap: NAME.gap, align: 'start'},
    className: NAME_CLASS.left,
  };
}

/**
 * The phone table for hidden opponents: north, west and east as squared
 * stacks of backs with a count badge, across the top in seat order, each
 * name under its stack; the trick below them, a tight cross; the message; your hand along
 * the bottom at the least room the deck allows. Same slots and regions as
 * the full table, so the game's items and region content do not change.
 * The stacks are upright, so their counts read. With `southRows` > 1 your
 * hand wraps onto that many rows when one row does not fit (a 13-card hand
 * on the centred-index 'simple' deck is ~10 card widths, wider than a phone
 * at the smallest card).
 */
export function compactClosed(
  m: HandMeasure,
  southRows: number,
  shape: HeartsHandShape,
): TableVariant['spec'] {
  const seat = southSeat(m, withRows(SEATS.closed.south, southRows), shape);
  const south = seatSize(m, {...seat, room: 0});
  const h = m.cardHeight;
  // A tight cross: west's and east's cards beside the middle, north's and
  // yours just clear of each other above and below it (a phone is short).
  const cross = {dx: TRICK.dx, dy: (h + TRICK_ROW_GAP) / 2};
  const centreWidth = 2 * cross.dx + 1;
  const width = Math.max(
    COMPACT_MIN_WIDTH,
    south.extent + 2 * SEAT_GAP,
    centreWidth + 2 * SEAT_GAP,
  );
  const band = NAME.gap + NAME.height;
  // Top strip: the stacks, then their names.
  const stackY = SEAT_GAP + h / 2;
  const trickTop = SEAT_GAP + h + STACK.pad + band + SEAT_GAP;
  const mid = trickTop + cross.dy + h / 2;
  const trickHalf = cross.dy + h / 2;
  const messageWidth = Math.min(MESSAGE.width, width - 2 * SEAT_GAP);
  const messageY = mid + trickHalf + MESSAGE.gap;
  const southTop = messageY + MESSAGE.height + MESSAGE.gap + band;
  const height = southTop + south.depth + SEAT_GAP;
  const stack = (id: string, label: string, x: number): SlotSpec =>
    stackSlot(id, label, x, stackY);
  const at = {
    n: [0, -cross.dy],
    e: [cross.dx, 0],
    s: [0, cross.dy],
    w: [-cross.dx, 0],
  } as const;
  const trick = (['n', 'e', 's', 'w'] as const).map((k): SlotSpec => ({
    id: `trick-${k}`,
    x: width / 2 + at[k][0],
    y: mid + at[k][1],
    pile: 'squared',
    anchor: 'center',
    tags: ['hide-empty'],
  }));
  // Three names across the top, none touching the next: west's starts at
  // its stack's left edge and east's ends at its stack's right edge (the
  // table's edges), north's is centred, with NAME.gap between the boxes.
  const across = nameWidth((width / 2 - SEAT_GAP - NAME.gap) / 1.5, 'opponent');
  const under = (
    hand: string,
    align: 'start' | 'center' | 'end',
    className?: string,
  ): RegionSpec => ({
    id: `name-${hand}`,
    width: across,
    height: NAME.height,
    attach: {slot: hand, side: 'bottom', gap: NAME.gap, align},
    ...(className === undefined ? {} : {className}),
  });
  const centre = {width: centreWidth, height: 2 * trickHalf};
  return {
    width,
    height,
    slots: [
      stack('north', 'North', width / 2),
      stack('west', 'West', SEAT_GAP + 0.5),
      stack('east', 'East', width - SEAT_GAP - 0.5),
      southSlot(width, southTop, south, seat),
      ...trick,
    ],
    regions: [
      southName(south.extent),
      under('north', 'center'),
      under('west', 'start', NAME_CLASS.left),
      under('east', 'end', NAME_CLASS.right),
      centreRegion(width, trickTop, centre.width, centre.height),
      messageRegion(width, messageY, messageWidth),
    ],
  };
}

/** Open phone table: the rows an opponent's column may take, tried in order
 *  (fewest first: the widest table, drawn at the largest card that fits),
 *  and the most rows your hand may wrap onto. */
export const OPEN_COLUMN_ROWS: readonly number[] = [3, 4, 5, 6];
const OPEN_SOUTH_MAX_ROWS = 3;
/**
 * The phone table for open hands (show all cards), drawn only where the
 * four-seat table does not fit even with its hands wrapped (a 13-card hand
 * on the centred-index 'simple' deck is ~10 card widths): every face must
 * stay legible, so no seat shrinks to a stack. Instead every hand is upright
 * and wrapped:
 *   - west, north and east as columns across the top, in seat order, each
 *     wrapped onto up to `columnRows` rows and hung from its name, so the
 *     names and the columns' first rows line up whatever each hand holds;
 *   - the trick as one row, west to east with north's and yours between;
 *   - the message; your hand along the bottom, on the fewest rows (up to
 *     OPEN_SOUTH_MAX_ROWS) that fit the table's width.
 * Same slots and regions as the full table.
 */
export function compactOpen(
  m: HandMeasure,
  columnRows: number,
): TableVariant['spec'] {
  const open = SEATS.open;
  const column: HandSeat = {
    mode: 'hand-overlap',
    cards: 13,
    faceUp: true,
    selectable: true,
    room: 0,
    rows: columnRows,
  };
  const col = seatSize(m, column);
  const trickWidth = 4 + 3 * TRICK_ROW_GAP;
  let width = Math.max(
    COMPACT_MIN_WIDTH,
    3 * col.extent + 4 * SEAT_GAP,
    trickWidth + 2 * SEAT_GAP,
  );
  // Your hand: the fewest rows that fit the width the columns set; at the
  // most rows, the table widens to it.
  const southFor = (rows: number): HandSeat =>
    withRows({...open.south, mode: 'hand-overlap', room: 0}, rows);
  const southRows =
    Array.from({length: OPEN_SOUTH_MAX_ROWS}, (_, i) => i + 1).find(
      r => seatSize(m, southFor(r)).extent + 2 * SEAT_GAP <= width,
    ) ?? OPEN_SOUTH_MAX_ROWS;
  const seat = southFor(southRows);
  const south = seatSize(m, seat);
  width = Math.max(width, south.extent + 2 * SEAT_GAP);
  const h = m.cardHeight;
  const band = NAME.gap + NAME.height;
  const colTop = SEAT_GAP + band;
  const trickY = colTop + col.depth + SEAT_GAP + h / 2;
  const messageWidth = Math.min(MESSAGE.width, width - 2 * SEAT_GAP);
  const messageY = trickY + h / 2 + MESSAGE.gap;
  const southTop = messageY + MESSAGE.height + MESSAGE.gap + band;
  const height = southTop + south.depth + SEAT_GAP;
  // Columns spread across the width: west's at the left edge, east's at
  // the right, north's centred. Start-anchored: a column's top-left stays
  // put as its hand shrinks.
  const colLeft = {
    west: SEAT_GAP,
    north: width / 2 - col.extent / 2,
    east: width - SEAT_GAP - col.extent,
  };
  const hand = (id: 'west' | 'north' | 'east', label: string): SlotSpec => ({
    id,
    x: colLeft[id],
    y: colTop,
    pile: column.mode,
    anchor: 'start',
    maxExtent: col.extent,
    label,
    ...seatWrap(column),
  });
  const trick = trickRow(['w', 'n', 's', 'e'], width, trickY);
  const name = (id: string): RegionSpec => columnName(id, col.extent);
  const centre = {width: trickWidth, height: h};
  return {
    width,
    height,
    slots: [
      hand('north', 'North'),
      hand('west', 'West'),
      hand('east', 'East'),
      southSlot(width, southTop, south, seat),
      ...trick,
    ],
    regions: [
      southName(south.extent),
      name('north'),
      name('west'),
      name('east'),
      centreRegion(width, trickY - h / 2, centre.width, centre.height),
      messageRegion(width, messageY, messageWidth),
    ],
  };
}

/**
 * The phone table for open hands where even the open columns do not fit (a
 * 13-card hand on the centred-index 'simple' deck in a short phone, or a
 * desktop panel dragged narrow): the four hands in the table's four
 * corners, the compass turned a quarter of the way clockwise, so play still
 * goes round the table clockwise: west top left, north top right, east
 * bottom right, you bottom left. Every hand is upright, wrapped onto up to
 * `rows` rows and hung from its name (at the table's outer edge), so the
 * names and first rows line up. The trick is one row across the middle in
 * play order from your seat (you, west, north, east: each card on its
 * player's side); under it one line holds your name, the message and east's
 * name, since a phone is short.
 * Same slots and regions as the full table.
 */
export function quadOpen(m: HandMeasure, rows: number): TableVariant['spec'] {
  const seat: HandSeat = {
    mode: 'hand-overlap',
    cards: 13,
    faceUp: true,
    selectable: true,
    room: 0,
    rows,
  };
  const hand = seatSize(m, seat);
  const h = m.cardHeight;
  const trickWidth = 4 + 3 * TRICK_ROW_GAP;
  // A name as wide as the longest the table shows reads whole; the
  // message takes the rest of its line, at least QUAD_MESSAGE_MIN.
  const nameW = nameWidth(Math.min(hand.extent, QUAD_NAME_WIDTH), 'corner');
  const beside = 2 * (SEAT_GAP + nameW + NAME.gap);
  const width = Math.max(
    COMPACT_MIN_WIDTH,
    2 * hand.extent + 3 * SEAT_GAP,
    trickWidth + 2 * SEAT_GAP,
    beside + QUAD_MESSAGE_MIN,
  );
  const messageWidth = Math.min(MESSAGE.width, width - beside);
  // A phone is short: the table's top and bottom margins, and the gap over
  // the trick, are the narrow ones.
  const topHands = QUAD_EDGE + NAME.height + NAME.gap;
  const trickY = topHands + hand.depth + QUAD_EDGE + h / 2;
  const line = Math.max(MESSAGE.height, NAME.height);
  const lineTop = trickY + h / 2 + MESSAGE.gap;
  const bottomHands = lineTop + line + NAME.gap;
  const height = bottomHands + hand.depth + QUAD_EDGE;
  // West's and your hands hang from their top-left, north's and east's
  // from their top-right, so each shrinks toward its own edge of the table
  // and its name stays over its end.
  const at = {
    west: {x: SEAT_GAP, y: topHands, anchor: 'start'},
    north: {x: width - SEAT_GAP, y: topHands, anchor: 'end'},
    east: {x: width - SEAT_GAP, y: bottomHands, anchor: 'end'},
    south: {x: SEAT_GAP, y: bottomHands, anchor: 'start'},
  } as const;
  const label = {
    west: 'West',
    north: 'North',
    east: 'East',
    south: 'South',
  } as const;
  const hands = (['north', 'west', 'east', 'south'] as const).map(
    (id): SlotSpec => ({
      id,
      x: at[id].x,
      y: at[id].y,
      pile: seat.mode,
      anchor: at[id].anchor,
      maxExtent: hand.extent,
      label: label[id],
      ...seatWrap(seat),
    }),
  );
  const trick = trickRow(['s', 'w', 'n', 'e'], width, trickY);
  // Names at the table's outer edges: west's and yours over the left end
  // of the hand, north's and east's over the right end, so the message
  // has the middle of its line.
  const name = (id: string, end: 'start' | 'end'): RegionSpec => ({
    id: `name-${id}`,
    width: nameW,
    height: NAME.height,
    attach: {slot: id, side: 'top', gap: NAME.gap, align: end},
    className: end === 'start' ? NAME_CLASS.left : NAME_CLASS.right,
  });
  return {
    width,
    height,
    slots: [...hands, ...trick],
    regions: [
      name('south', 'start'),
      name('north', 'end'),
      name('west', 'start'),
      name('east', 'end'),
      centreRegion(width, trickY - h / 2, trickWidth, h),
      messageRegion(width, lineTop + line - MESSAGE.height, messageWidth),
    ],
  };
}

/** Open corner table: the widest a name is drawn, card widths ("Player 0"
 *  whole at the smallest card), and the least room for the message between
 *  the two bottom names. */
const QUAD_NAME_WIDTH = 1.8;
const QUAD_MESSAGE_MIN = 2.6;
/** Open corner table: its top and bottom margins, and the gap over the
 *  trick, card widths. */
const QUAD_EDGE = 0.08;
/** Rows each hand takes on the open corner table, tried in order. */
export const QUAD_ROWS: readonly number[] = [3, 4];

/** Rows your hand takes on the compact phone table, tried in order. */
export const COMPACT_SOUTH_ROWS: readonly number[] = [1, 2];
