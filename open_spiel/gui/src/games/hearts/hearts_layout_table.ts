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

// The hearts four-seat table (hearts_layout.ts): the opponents round the
// trick, your hand along the bottom, every name a band beside its hand, the
// centre over the trick and the message under it.
import {
  fourSeatTable,
  largerBox,
  seatBox,
  seatSize,
  SEAT_GAP,
  type HandMeasure,
  type HandSeat,
  type RegionSpec,
  type SlotSpec,
  type TableSpec,
} from '../../cards/library/index.ts';
import {
  MESSAGE,
  MIN_SIZE,
  NAME,
  NAME_CLASS,
  SEATS,
  TRICK,
  TRICK_SIDE_CLEAR,
  centreRegion,
  messageRegion,
  nameWidth,
  southSeat,
  withRows,
  type HeartsDress,
  type HeartsHandShape,
  type SeatRows,
} from './hearts_layout_parts.ts';

type Box = ReturnType<typeof seatBox>;

/** The seats of one dress and their screen boxes. */
interface TableSeats {
  readonly analysis: boolean;
  readonly upright: boolean;
  readonly backs: {north: HandSeat; side: HandSeat};
  readonly reserve: {north: Box; side: Box} | undefined;
  readonly seats: {north: HandSeat; side: HandSeat; south: HandSeat};
  readonly south: {extent: number; depth: number};
  readonly north: Box;
  readonly side: Box;
}

function tableSeats(
  m: HandMeasure,
  dress: HeartsDress,
  rows: SeatRows,
  shape: HeartsHandShape,
): TableSeats {
  const analysis = dress !== 'closed';
  // The opponents as backs (turned to face their seats) and as faces
  // (upright). Play draws backs at their own size; analysis reserves, at
  // every seat, the larger of the two, so both dresses share one geometry.
  const backs = {
    north: withRows(SEATS.closed.north, analysis ? 1 : rows.north),
    side: withRows(SEATS.closed.side, analysis ? 1 : rows.side),
  };
  const faces = {
    north: withRows(SEATS.open.north, rows.north),
    side: withRows(SEATS.open.side, rows.side),
  };
  const own = dress === 'open' ? faces : backs;
  const upright = dress === 'open';
  const reserve = analysis
    ? {
        north: largerBox(
          seatBox(m, backs.north, 'across', false),
          seatBox(m, faces.north, 'across', true),
        ),
        side: largerBox(
          seatBox(m, backs.side, 'side', false),
          seatBox(m, faces.side, 'side', true),
        ),
      }
    : undefined;
  const seats = {
    ...own,
    south: southSeat(m, withRows(SEATS.closed.south, rows.south), shape),
  };
  const south = seatSize(m, seats.south);
  // Screen boxes (card widths): north's across the top, a side seat's
  // down its band.
  const north = reserve?.north ?? seatBox(m, own.north, 'across', upright);
  const side = reserve?.side ?? seatBox(m, own.side, 'side', upright);

  return {analysis, upright, backs, reserve, seats, south, north, side};
}

/** Where the four-seat table puts the trick, the message and the names. */
interface TableGeometry {
  readonly trickHalf: number;
  readonly centreWidth: number;
  readonly width: number;
  readonly messageWidth: number;
  readonly messageTop: number;
  readonly minSize: {width: number; height: number};
}

function tableGeometry(m: HandMeasure, t: TableSeats): TableGeometry {
  const {north, side, south} = t;

  // Every name is a band beside its hand, across the fan axis:
  //   - your name over the start of your hand, under the message, which is
  //     under your trick card;
  //   - north's under the start of its hand (its player's left, the screen's
  //     right), over the trick;
  //   - the side names over the side hands, under north's name band.
  // fourSeatTable centres the side hands and the trick at
  // mid = (north height - south depth + height) / 2, so each band is a
  // minimum height. The message is under your trick card, and under the
  // side hands too when it is wider than the room between them.
  const band = NAME.gap + NAME.height;
  const trickHalf = TRICK.dy + m.cardHeight / 2;
  // fourSeatTable's width: the longer of north and south, or the trick
  // between the side hands.
  const centreWidth = 2 * TRICK.dx + 1;
  // The trick between the side hands, TRICK_SIDE_CLEAR from each (more
  // than fourSeatTable's own SEAT_GAP): passed to it as the least width.
  const sideClearWidth =
    centreWidth + 2 * (side.width + SEAT_GAP + TRICK_SIDE_CLEAR);
  const width = Math.max(
    MIN_SIZE.width,
    north.width + 2 * SEAT_GAP,
    south.extent + 2 * SEAT_GAP,
    sideClearWidth,
  );
  const messageWidth = Math.min(MESSAGE.width, width - 2 * SEAT_GAP);
  const between = width - 2 * (side.width + 2 * SEAT_GAP);
  const messageTop =
    Math.max(trickHalf, messageWidth > between ? side.height / 2 : 0) +
    MESSAGE.gap;
  const underMid = messageTop + MESSAGE.height + 2 * MESSAGE.gap + band;
  const overTrick = band + NAME.gap;
  const overSide =
    SEAT_GAP + north.height + band + NAME.gap + band + side.height / 2;
  const minSize = {
    width: Math.max(MIN_SIZE.width, sideClearWidth),
    height: Math.max(
      MIN_SIZE.height,
      2 * (underMid + SEAT_GAP) + north.height + south.depth,
      2 * (overTrick + SEAT_GAP + trickHalf) + north.height + south.depth,
      2 * overSide - north.height + south.depth,
    ),
  };

  return {trickHalf, centreWidth, width, messageWidth, messageTop, minSize};
}

/** A centre-anchored slot of the four-seat table `spec`. */
function slotOf(spec: TableSpec, id: string): SlotSpec {
  const s = spec.slots.find(x => x.id === id);
  if (s?.anchor !== 'center') {
    throw new Error(
      `hearts layout: no centre-anchored '${id}' slot in the four-seat table`,
    );
  }
  return s;
}

/** A name band attached to `hand`'s `side`. */
function name(
  hand: string,
  side: 'left' | 'right' | 'top',
  width: number,
  align?: 'start',
  className?: string,
): RegionSpec {
  return {
    id: `name-${hand}`,
    width,
    height: NAME.height,
    attach: {
      slot: hand,
      side,
      gap: NAME.gap,
      ...(align === undefined ? {} : {align}),
    },
    ...(className === undefined ? {} : {className}),
  };
}

// Analysis: each side hand against its table edge, SEAT_GAP in, backs
// and faces alike, and centred on the side seats' midline: switching
// show all cards changes how far a hand reaches toward the trick, never
// its margin from the felt. The backs are turned (centre-anchored,
// their screen width constant); the faces are upright, hung from the
// edge ('start' west, 'end' east) and centred down on what they draw
// (block), so a hand that shrinks keeps its edge and its midline.

function sideHand(
  s: SlotSpec,
  t: TableSeats,
  specWidth: number,
  backsWidth: number,
): SlotSpec {
  const {analysis, upright, side} = t;

  if (s.id !== 'west' && s.id !== 'east') return s;
  const west = s.id === 'west';
  const pin = west ? ('left' as const) : ('right' as const);
  if (!analysis) return {...s, pin};
  if (!upright) {
    return {
      ...s,
      x: west
        ? SEAT_GAP + backsWidth / 2
        : specWidth - SEAT_GAP - backsWidth / 2,
      pin,
    };
  }
  return {
    ...s,
    anchor: west ? 'start' : 'end',
    x: west ? SEAT_GAP : specWidth - SEAT_GAP,
    y: s.y - side.height / 2,
    block: {id: s.id, y: s.y},
    pin,
  };
}

/** The opponents' names: fixed at the reserved boxes in analysis, attached
 *  to their hands in play. */
function opponentNames(t: TableSeats, spec: TableSpec): RegionSpec[] {
  const {north, side} = t;
  const slot = (id: string): SlotSpec => slotOf(spec, id);

  // A side name is centred over its hand, inside the side column; north's
  // and yours run along the hand from its start, at most the hand's room.
  const over = nameWidth(2 * slot('west').x - SEAT_GAP, 'side');
  const northWidth = nameWidth(north.width, 'north');
  // Analysis: the names are fixed at the reserved boxes, the same for backs
  // and faces (a name attached to a hand would move with what it holds):
  // north's centred under its box, the side names over theirs, each at its
  // table edge, as its hand is (sideHand), so each starts where its hand
  // does whether it holds a short fan of backs or a long row of faces.
  const sideName = (hand: 'west' | 'east'): RegionSpec => ({
    id: `name-${hand}`,
    x: hand === 'west' ? SEAT_GAP : spec.width - SEAT_GAP - over,
    y: slot(hand).y - side.height / 2 - NAME.gap - NAME.height,
    width: over,
    height: NAME.height,
    pin: hand === 'west' ? 'left' : 'right',
    className: hand === 'west' ? NAME_CLASS.left : NAME_CLASS.right,
  });
  return t.analysis
    ? [
        {
          id: 'name-north',
          x: spec.width / 2 - northWidth / 2,
          y: slot('north').y + north.height / 2 + NAME.gap,
          width: northWidth,
          height: NAME.height,
        },
        sideName('west'),
        sideName('east'),
      ]
    : [
        // In each hand's own frame the name is at its start, on the side
        // facing the trick: under the right end of north's (turned 180°),
        // above west's and east's. The name hugs the hand's start: its right
        // end (on screen) for north's.
        name('north', 'top', northWidth, 'start', NAME_CLASS.right),
        name('west', 'left', over),
        name('east', 'right', over),
      ];
}

/** The four-seat table with every hand on `rows` rows, in one dress. */
export function tableFor(
  m: HandMeasure,
  dress: HeartsDress,
  rows: SeatRows,
  shape: HeartsHandShape,
): TableSpec {
  const t = tableSeats(m, dress, rows, shape);
  const {width, messageWidth, messageTop, centreWidth, trickHalf, minSize} =
    tableGeometry(m, t);

  const spec = fourSeatTable(m, {
    minSize,
    ...t.seats,
    centre: {...TRICK, prefix: 'trick-'},
    upright: t.upright,
    ...(t.reserve === undefined ? {} : {reserve: t.reserve}),
  });
  if (Math.abs(spec.width - width) > 1e-9) {
    throw new Error(
      `hearts layout: fourSeatTable drew ${String(spec.width)} card widths wide, the message was placed for ${String(width)}`,
    );
  }

  const slot = (id: string): SlotSpec => slotOf(spec, id);
  const backsWidth = seatBox(m, t.backs.side, 'side', false).width;

  // The centre covers the four trick positions.
  const centre = {width: centreWidth, height: 2 * trickHalf};
  return {
    ...spec,
    // A wide felt widens the table (TableSpec.fillWidth): the side hands
    // and their names keep their margin from its edges (pin), the rest
    // stays centred.
    fillWidth: true,
    // An empty trick position draws nothing (a legal drop still marks it).
    slots: spec.slots.map(s =>
      s.id.startsWith('trick-')
        ? {...s, tags: ['hide-empty']}
        : sideHand(s, t, spec.width, backsWidth),
    ),
    regions: [
      // Your name over the left end of your hand (the same hand in every
      // dress, so it follows the hand as it shrinks).
      name(
        'south',
        'top',
        nameWidth(t.south.extent, 'south'),
        'start',
        NAME_CLASS.left,
      ),
      ...opponentNames(t, spec),
      // The centre shows only when the trick does not, so it covers nothing.
      centreRegion(
        spec.width,
        slot('trick-w').y - centre.height / 2,
        centre.width,
        centre.height,
      ),
      messageRegion(
        width,
        slot('trick-s').y - TRICK.dy + messageTop,
        messageWidth,
      ),
    ],
  };
}
