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

// The gin board as one table, and the variants the library picks from to
// fit it to the panel (gin_rummy_layout.ts holds the sizes and ids,
// gin_layout_blocks.ts the blocks).

import type {
  HandMeasure,
  TableClearance,
  TableLayout,
  TableSpec,
} from '../../cards/library/index.ts';
import {
  GAP,
  NARROW_STATS,
  SIDE,
  gridBlock,
  handBlock,
  shift,
  supplyBlock,
  type Block,
  type GridPack,
} from './gin_layout_blocks.ts';
import {
  GIN_STATS_CARD_PX,
  GIN_STATS_FONT,
  GIN_WIDE_MIN_CARD_PX,
  ginStatsHeight,
} from './gin_layout_stats.ts';
import {
  GIN_HAND_MAX_ROWS,
  GIN_LAYOUT_COST,
  GIN_REGION,
  GIN_SLOT,
  GRID,
  MAX_GHOSTS,
  MAX_HAND,
  PICK_LIFT,
  SMALL,
  ginMinCardWidth,
  type GinTableOptions,
} from './gin_rummy_layout.ts';
import {TEXT_FLOOR_PX} from '../../styles/text_floor.ts';

/**
 * The widest the wide layout's stats and action columns are offered, card
 * widths beyond the least: where the panel has width to spare (a board
 * limited by its height) a wider stats column is shorter, so the table is
 * too and its cards larger; the library draws whichever comes out largest.
 */
const SIDE_EXTRA_MAX = 2;
/** Steps the wider columns are offered in, card widths. */
const SIDE_EXTRA_STEP = 0.5;
/** Least room kept for a fanned hand, card widths (a short hand stays centred). */
const HAND_ROOM = {'hand-overlap': 6, 'hand-arc': 7} as const;

type Shape = 'wide' | 'narrow';
type Rooms = 'least' | 'all';

interface Variant {
  cost: number;
  minDrawnCardWidth?: number;
  spec: Omit<TableSpec, 'variants' | 'minWidth' | 'minDrawnCardWidth'>;
}

interface Context {
  readonly o: GinTableOptions;
  readonly m: HandMeasure;
  readonly clear: TableClearance;
  readonly cardHeight: number;
  readonly handRoomFor: number;
}

/** One shape's rows, before its stats' room is chosen. */
interface Rows {
  readonly shape: Shape;
  readonly pileScale: number;
  readonly opponent: Block;
  readonly player: Block;
  /**
   * The scales of every slot the opponent's area may draw, the fan's and
   * the grid's, whichever is drawn now: the narrow row's floor is read on
   * them, so a reveal picks no other variant.
   */
  readonly opponentScales: readonly number[];
  readonly rowsWidth: number;
  readonly leastWidth: number;
  readonly cost: number;
}

function roundUp3(v: number): number {
  return Math.ceil(v * 1000 - 1e-6) / 1000;
}

/** A shape's cost (GIN_LAYOUT_COST): its grid packing, bands and hand rows. */
function costOf(
  c: Context,
  pack: GridPack,
  bands: number,
  handRows: number,
): number {
  // Every table has a grid's room (opponentArea).
  let packing: number;
  if (c.o.grid === 'compact') {
    packing = pack === 'tight' ? GIN_LAYOUT_COST.tight : 1;
  } else {
    packing = pack === 'spread' ? 1 : GIN_LAYOUT_COST.squeeze;
  }
  return (
    packing *
    (bands > 1 ? GIN_LAYOUT_COST.bands : 1) *
    GIN_LAYOUT_COST.row ** (handRows - 1)
  );
}

/**
 * The opponent's face-down fan. It is measured whether or not it is drawn:
 * the opponent's area is the box that holds both the fan and the revealed
 * grid (opponentArea).
 */
function opponentHand(c: Context, shape: Shape): Block {
  // Narrow: full-size backs, so the smallest card sets no higher floor.
  return handBlock(
    GIN_SLOT.opponent,
    'hand-overlap',
    shape === 'wide'
      ? {n: MAX_HAND, scale: SMALL, faceUp: false, room: 4, liftRoom: 0}
      : {n: MAX_HAND, scale: 1, faceUp: false, room: 0, liftRoom: 0},
    c.m,
  );
}

function playerHand(
  c: Context,
  shape: Shape,
  pack: GridPack,
  handRows: number,
): Block | null {
  const {player} = c.o;
  if (player === 'grid') return null;
  // Pixel lift of a picked card, as card widths at the smallest table card
  // width this shape draws (the floor is the smallest card's: a spread
  // grid's cells are GRID of the table's card).
  const liftRoom =
    PICK_LIFT / (ginMinCardWidth(c.o) / (pack === 'spread' ? GRID : 1));
  return handBlock(
    GIN_SLOT.hand,
    player,
    {
      n: c.handRoomFor,
      scale: 1,
      faceUp: true,
      room: shape === 'wide' ? HAND_ROOM[player] : 0,
      liftRoom,
      rows: handRows,
    },
    c.m,
  );
}

/**
 * The opponent's area: one box as large as the face-down fan and the
 * revealed grid both need, with the one drawn centred in it. The hand's end
 * reveals the opponent's hand in this box, so the supply row and the
 * player's hand keep their place.
 */
export function opponentArea(
  shown: GinTableOptions['opponent'],
  fan: Block,
  grid: Block,
): Block {
  const width = Math.max(fan.width, grid.width);
  const height = Math.max(fan.height, grid.height);
  const drawn = shown === 'grid' ? grid : fan;
  return {
    ...shift(drawn, (width - drawn.width) / 2, (height - drawn.height) / 2),
    width,
    height,
  };
}

function rowsOf(
  c: Context,
  pack: GridPack,
  shape: Shape,
  bands: number,
  handRows: number,
): Rows {
  // The stock and upcard at the size of the player's own cards: a spread
  // grid's cells, else the table's card (a fanned hand, an overlapped grid).
  const pileScale = c.o.player === 'grid' && pack === 'spread' ? GRID : 1;
  // The supply at its least width first: the rows around it (a grid fills
  // the width the others take) set the table's width.
  const least = supplyBlock(
    shape,
    c.cardHeight,
    pileScale,
    SIDE,
    shape === 'wide' ? pileScale * c.cardHeight : NARROW_STATS,
  );
  const opponentFan = opponentHand(c, shape);
  const playerFan = playerHand(c, shape, pack, handRows);
  // The width the rows other than the grids take: a grid fills it. The
  // opponent's fan counts whether or not it is drawn, so revealing the
  // opponent's hand changes no width.
  const fill = Math.max(least.width, opponentFan.width, playerFan?.width ?? 0);
  const grid = {pack, bands, fill};
  const opponentGrid = gridBlock('opponent', c.o, grid, c.cardHeight, c.clear);
  const opponent = opponentArea(c.o.opponent, opponentFan, opponentGrid);
  const player =
    playerFan ?? gridBlock('player', c.o, grid, c.cardHeight, c.clear);
  return {
    shape,
    pileScale,
    opponent,
    player,
    opponentScales: [opponentFan, opponentGrid].flatMap(b =>
      b.slots.map(sl => sl.scale ?? 1),
    ),
    rowsWidth: Math.max(least.width, opponent.width, player.width),
    leastWidth: least.width,
    cost: costOf(c, pack, bands, handRows),
  };
}

/**
 * The shape with its stats' room: the wide column `side` wide and `stats`
 * tall, or the narrow row `stats` tall.
 */
function variant(
  c: Context,
  r: Rows,
  side: number,
  stats: number,
  minDrawnCardWidth: number | undefined,
): Variant {
  const supply = supplyBlock(r.shape, c.cardHeight, r.pileScale, side, stats);
  // The narrow shape spends no margin at the sides: the felt's is enough.
  const margin = r.shape === 'wide' ? GAP : 0;
  const width = Math.max(r.rowsWidth, supply.width) + 2 * margin;
  // Rounded up, so a row as wide as the table (a spread grid's panel on the
  // narrow shape, which has no side margin) stays in it.
  const tableWidth = roundUp3(width);
  let y = GAP;
  const placed = [r.opponent, supply, r.player].map(b => {
    const at = shift(b, (width - b.width) / 2, y);
    y += b.height + 2 * GAP;
    return at;
  });
  // The narrow stats row spans the table.
  const regions = placed
    .flatMap(b => b.regions)
    .map(region =>
      r.shape === 'narrow' && region.id === GIN_REGION.stats
        ? {...region, x: 0, width: tableWidth}
        : region,
    );
  return {
    cost: r.cost,
    ...(minDrawnCardWidth === undefined ? {} : {minDrawnCardWidth}),
    spec: {
      width: tableWidth,
      height: Math.round((y - GAP) * 1000) / 1000,
      slots: placed.flatMap(b => b.slots),
      regions,
    },
  };
}

function fontAt(px: number): number {
  return TEXT_FLOOR_PX / px;
}

/**
 * The wide column takes at least half the width the other rows leave beside
 * the piles (the actions the other half, so the piles stay centred) and is
 * as tall as its text needs at the font it is sized for.
 */
function wideVariants(
  c: Context,
  r: Rows,
  handRows: number,
  rooms: Rooms,
): Variant[] {
  const base = SIDE + (r.rowsWidth - r.leastWidth) / 2;
  const gates: readonly number[] =
    rooms === 'all' ? GIN_STATS_CARD_PX : [GIN_WIDE_MIN_CARD_PX];
  const extraMax = rooms === 'all' && handRows === 1 ? SIDE_EXTRA_MAX : 0;
  const pile = r.pileScale * c.cardHeight;
  return gates.flatMap(px => {
    const font = Math.max(GIN_STATS_FONT, fontAt(px));
    const out: Variant[] = [];
    let shortest = Infinity;
    for (let extra = 0; extra <= extraMax + 1e-9; extra += SIDE_EXTRA_STEP) {
      const side = base + extra;
      const stats = ginStatsHeight(side, font, pile);
      if (!(stats < shortest - 1e-6)) continue;
      shortest = stats;
      out.push(variant(c, r, side, stats, px));
      // At a pile's height the column is as short as it gets.
      if (stats <= pile + 1e-6) break;
    }
    return out;
  });
}

/**
 * The narrow row spans the table and is as tall as its text needs at 12 px
 * on the smallest card this shape draws (the table's floor,
 * ginMinCardWidth, over its smallest slot), or on a larger one with that
 * card as its gate. Gated rows come first (a smaller row for a larger
 * card), the ungated fallback last: the library drops a gated one under its
 * card.
 */
function narrowVariants(c: Context, r: Rows, rooms: Rooms): Variant[] {
  const smallest = Math.min(
    r.pileScale,
    ...r.opponentScales,
    ...r.player.slots.map(sl => sl.scale ?? 1),
  );
  const floorPx = ginMinCardWidth(c.o) / smallest;
  const tableWidth = roundUp3(r.rowsWidth);
  const narrow = (font: number, gate: number | undefined): Variant =>
    variant(c, r, SIDE, ginStatsHeight(tableWidth, font, NARROW_STATS), gate);
  const gates: readonly number[] = rooms === 'all' ? GIN_STATS_CARD_PX : [];
  const gated = gates
    .filter(px => px > floorPx)
    .map(px => narrow(Math.max(GIN_STATS_FONT, fontAt(px)), px));
  return [
    ...gated,
    narrow(Math.max(GIN_STATS_FONT, fontAt(floorPx)), undefined),
  ];
}

/**
 * One shape's variants: its grid packing, bands and hand rows, the supply
 * wide (stats left, actions right) or narrow (stats below). With `rooms`
 * 'least', the stats' room at its least: the wide column at its least width
 * for 60 px cards, the narrow row for 12 px on the smallest card the shape
 * draws. With 'all', also the wide column for each smaller card of
 * GIN_STATS_CARD_PX and, with the hand in one row, wider in SIDE_EXTRA_STEP
 * steps while that still shortens it (a wider one no shorter is never
 * better), and the narrow row for each larger card of GIN_STATS_CARD_PX.
 * The last-resort shapes (GIN_LAYOUT_COST.squeeze) keep 'least': every
 * layout is one more for the library to fit.
 */
function build(
  c: Context,
  pack: GridPack,
  shape: Shape,
  bands = 1,
  handRows = 1,
  rooms: Rooms = 'least',
): Variant[] {
  const r = rowsOf(c, pack, shape, bands, handRows);
  return shape === 'wide'
    ? wideVariants(c, r, handRows, rooms)
    : narrowVariants(c, r, rooms);
}

/**
 * The shapes in order of preference. The grid's style is the player's: a
 * 'normal', 'slots' or 'no-bg' grid is spread, one card to a cell, and wraps
 * its rank columns onto two bands where one band cannot fit; it overlaps
 * only where no spread shape is legible at all (GIN_LAYOUT_COST.squeeze). A
 * 'compact' grid overlaps its cells (by half a card, then down to the
 * index). Offering the overlapped shapes to every style would make the
 * largest-cards rule draw the grid compact on every panel.
 */
function shapes(c: Context, handRows: number): Variant[] {
  if (c.o.grid === 'compact') {
    return [
      ...build(c, 'half', 'wide', 1, handRows, 'all'),
      ...build(c, 'half', 'narrow', 1, handRows, 'all'),
      ...build(c, 'tight', 'narrow', 1, handRows, 'all'),
      ...build(c, 'tight', 'narrow', 2, handRows, 'all'),
    ];
  }
  return [
    ...build(c, 'spread', 'wide', 1, handRows, 'all'),
    ...build(c, 'spread', 'narrow', 1, handRows, 'all'),
    ...build(c, 'spread', 'narrow', 2, handRows, 'all'),
    // The last resort (GIN_LAYOUT_COST.squeeze): a panel too narrow for any
    // spread grid at a legible card width.
    ...build(c, 'half', 'narrow', 1, handRows),
    ...build(c, 'tight', 'narrow', 1, handRows),
    ...build(c, 'tight', 'narrow', 2, handRows),
  ];
}

function checkHandCards(o: GinTableOptions): void {
  const n = o.handCards;
  if (n === undefined) return;
  if (!(Number.isInteger(n) && n >= 0 && n <= MAX_HAND + MAX_GHOSTS)) {
    throw new Error(
      `gin handCards must be a whole number from 0 to ${String(MAX_HAND + MAX_GHOSTS)}, got ${String(n)}`,
    );
  }
}

/**
 * The gin board as one table: the opponent's hand, the supply row (stats,
 * stock, upcard, actions) and the player's hand, stacked and centred. Card
 * widths are the player's cards; the opponent's hidden hand and a spread
 * grid are drawn smaller (SlotSpec.scale). Hands take the room the active
 * deck needs for their fullest deal (`m`), grids the overlap its index
 * allows (the clearance), so no deck ever has a covered index.
 *
 * Other panels get variants, so no grid column is ever cut off and the cards
 * come out as large as the panel allows, in the grid style the player chose.
 * Every rank column is always drawn, so a grid cell never moves. The same
 * shapes follow with the fanned hand wrapped onto two rows, then three
 * (GIN_HAND_MAX_ROWS). Each variant costs what it gives up
 * (GIN_LAYOUT_COST), and the library draws the one whose cards come out
 * largest for their cost, at or above a legible card width. The fanned hand
 * is laid out for its fullest deal (SlotSpec.capacity): drawing a card never
 * moves the others.
 */
export function ginTable(o: GinTableOptions, m: HandMeasure): TableLayout {
  checkHandCards(o);
  const handRoomFor = Math.max(MAX_HAND, o.handCards ?? 0);
  return clear => {
    const c: Context = {
      o,
      m,
      clear,
      cardHeight: m.cardHeight,
      handRoomFor,
    };
    const rowCounts =
      o.player === 'grid'
        ? [1]
        : Array.from({length: GIN_HAND_MAX_ROWS}, (_, i) => i + 1);
    const [base, ...rest] = rowCounts.flatMap(rows => shapes(c, rows));
    if (base === undefined) throw new Error('gin layout: no shape');
    if (base.cost !== 1) {
      throw new Error(
        `gin layout: the first shape must cost 1, got ${String(base.cost)}`,
      );
    }
    return {
      ...base.spec,
      ...(base.minDrawnCardWidth === undefined
        ? {}
        : {minDrawnCardWidth: base.minDrawnCardWidth}),
      variants: rest.map(({spec, cost, minDrawnCardWidth}) => ({
        spec,
        ...(cost === 1 ? {} : {cost}),
        ...(minDrawnCardWidth === undefined ? {} : {minDrawnCardWidth}),
      })),
    };
  };
}
