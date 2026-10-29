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

// The blocks gin's table is built from (gin_layout_table.ts): a suit x rank
// grid, a fanned hand, and the supply row (stats, stock, upcard, actions).
// Every size is in card widths.

import {
  SYMBOL_CORNER_TAG,
  type HandMeasure,
  type RegionSpec,
  type SlotSpec,
  type TableClearance,
} from '../../cards/library/index.ts';
import {
  GIN_REGION,
  GIN_SLOT,
  GRID,
  PICK_LIFT,
  gridSlotId,
  type GinTableOptions,
  type GinZone,
} from './gin_rummy_layout.ts';

/** Margin between and around blocks, card widths. */
export const GAP = 0.15;
/** Between normal grid cells, grid-card widths. */
const GRID_GAP = 0.18;
/** The compact grid's preferred overlap: half a card each way. */
const COMPACT_STEP = 0.5;
/**
 * Least width of the stats and action columns beside the supply, card widths
 * (both take an equal share of any width the table has to spare).
 */
export const SIDE = 2.4;
/** Narrow shape: the action column's width. */
const NARROW_ACTIONS = 1.3;
/** Narrow shape: the stats row's least height. */
export const NARROW_STATS = 1;
/** Room the stock's depth layers take, card widths. */
const STOCK_DEPTH = 0.2;
/** Between the bands of a grid wrapped onto two bands, card widths. */
const BAND_GAP = GAP;

/** A part of the table: its slots and regions, and its size. */
export interface Block {
  slots: SlotSpec[];
  regions: RegionSpec[];
  width: number;
  height: number;
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/** A block moved right by dx and down by dy. */
export function shift(b: Block, dx: number, dy: number): Block {
  return {
    ...b,
    slots: b.slots.map(s => ({...s, x: round3(s.x + dx), y: round3(s.y + dy)})),
    regions: b.regions.map(r => ({
      ...r,
      x: round3((r.x ?? 0) + dx),
      y: round3((r.y ?? 0) + dy),
    })),
  };
}

/**
 * How tightly a grid's cells overlap: not at all (the normal, slots and
 * no-bg styles), by half a card (compact, and the first narrow fallback), or
 * by the least the active deck's index allows (the narrowest fallbacks).
 */
export type GridPack = 'spread' | 'half' | 'tight';

/**
 * How one grid is drawn: its packing, how many bands its rank columns wrap
 * onto, and the width the rows around it already take (card widths), which
 * an overlapped grid spreads its columns to fill.
 */
export interface GridShape {
  pack: GridPack;
  bands: number;
  fill: number;
}

type GridOptions = Pick<GinTableOptions, 'grid' | 'deck' | 'held'>;

interface GridSteps {
  scale: number;
  pad: number;
  perBand: number;
  nBands: number;
  colStep: number;
  rowStep: number;
  bandHeight: number;
}

function gridSteps(
  o: GridOptions,
  shape: GridShape,
  cardHeight: number,
  clear: TableClearance,
): GridSteps {
  const {pack, bands, fill} = shape;
  const {suitRows, rankCodes} = o.deck;
  const scale = pack === 'spread' ? GRID : 1;
  const perBand = Math.ceil(rankCodes.length / bands);
  const nBands = Math.ceil(rankCodes.length / perBand);
  const want = pack === 'tight' ? 0 : COMPACT_STEP;
  const least = pack === 'spread' ? 1 + GRID_GAP : Math.max(want, clear.right);
  // An overlapped grid spreads its columns over the width the table already
  // has (never wider than side by side): more of each card shows, and the
  // table is no wider for it.
  const pad = o.grid === 'normal' ? GRID_GAP * GRID : 0;
  const spreadTo =
    perBand > 1 ? ((fill - 2 * pad) / scale - 1) / (perBand - 1) : least;
  const colStep =
    (pack === 'spread'
      ? least
      : Math.max(least, Math.min(1 + GRID_GAP, spreadTo))) * scale;
  const rowStep =
    (pack === 'spread'
      ? cardHeight + GRID_GAP
      : Math.max(want * cardHeight, clear.down)) * scale;
  const bandHeight = (suitRows.length - 1) * rowStep + scale * cardHeight;
  return {scale, pad, perBand, nBands, colStep, rowStep, bandHeight};
}

function cellPositions(
  o: GridOptions,
  steps: GridSteps,
): Map<string, {x: number; y: number}> {
  const {pad, perBand, colStep, rowStep, bandHeight} = steps;
  const at = new Map<string, {x: number; y: number}>();
  o.deck.rankCodes.forEach((rank, i) => {
    const band = Math.floor(i / perBand);
    const c = i % perBand;
    o.deck.suitRows.forEach((suit, r) =>
      at.set(`${rank}${suit.code}`, {
        x: pad + c * colStep,
        y: pad + band * (bandHeight + BAND_GAP) + r * rowStep,
      }),
    );
  });
  return at;
}

/**
 * The cell codes in paint order: a later slot is drawn over an earlier one.
 * Empty cells go first, so a hidden one (its box kept) never lies over a
 * card.
 */
function paintOrder(zone: GinZone, o: GridOptions): string[] {
  const cells = o.deck.suitRows.flatMap(suit =>
    o.deck.rankCodes.map(rank => `${rank}${suit.code}`),
  );
  const heldCodes = o.held?.[zone];
  if (heldCodes === undefined) return cells;
  const held = new Set(heldCodes);
  return [
    ...cells.filter(c => !held.has(c)),
    ...cells.filter(c => held.has(c)),
  ];
}

function gridSlots(
  zone: GinZone,
  o: GridOptions,
  pack: GridPack,
  steps: GridSteps,
): SlotSpec[] {
  // An empty cell is a faint well only in a spread grid in the normal
  // (panelled) or slots-only style; the no-background and compact grids,
  // and the overlapped narrow variants, show only the cards (the library's
  // 'hide-empty' slot tag, per variant). Every rank column is always drawn,
  // empty or not: a cell is fixed, so a drawn or discarded card never moves
  // another.
  const bare =
    !(o.grid === 'normal' || o.grid === 'slots') || pack !== 'spread';
  const at = cellPositions(o, steps);
  return paintOrder(zone, o).map((code): SlotSpec => {
    const p = at.get(code);
    if (p === undefined) {
      throw new Error(
        `gin grid has no cell for '${code}': every rank column is drawn`,
      );
    }
    return {
      id: gridSlotId(zone, code),
      x: p.x,
      y: p.y,
      pile: 'squared',
      ...(steps.scale === 1 ? {} : {scale: steps.scale}),
      // An overlapped cell may be covered past its index, so a symbol-only
      // card there (dead, known) sits at the index corner (the library's
      // SYMBOL_CORNER_TAG), where it stays in view.
      ...(bare
        ? {
            tags:
              pack === 'spread'
                ? ['hide-empty']
                : ['hide-empty', SYMBOL_CORNER_TAG],
          }
        : {}),
    };
  });
}

/**
 * One suit x rank grid: a single-card slot per card of the deck. Overlapped
 * cells are full-size cards, each leaving its neighbour's index visible.
 */
export function gridBlock(
  zone: GinZone,
  o: GridOptions,
  shape: GridShape,
  cardHeight: number,
  clear: TableClearance,
): Block {
  if (o.deck.suitRows.length === 0 || o.deck.rankCodes.length === 0) {
    throw new Error('gin grid needs at least one suit and one rank');
  }
  if (!(Number.isInteger(shape.bands) && shape.bands >= 1)) {
    throw new Error(
      `gin grid bands must be a whole number >= 1, got ${String(shape.bands)}`,
    );
  }
  const steps = gridSteps(o, shape, cardHeight, clear);
  const {scale, pad, perBand, nBands, colStep, bandHeight} = steps;
  const width = 2 * pad + (perBand - 1) * colStep + scale;
  const height = 2 * pad + nBands * bandHeight + (nBands - 1) * BAND_GAP;
  const regions: RegionSpec[] =
    o.grid === 'normal'
      ? [
          {
            id:
              zone === 'player'
                ? GIN_REGION.playerPanel
                : GIN_REGION.opponentPanel,
            x: 0,
            y: 0,
            width,
            height,
            occludes: false,
          },
        ]
      : [];
  return {slots: gridSlots(zone, o, shape.pack, steps), regions, width, height};
}

/** The deal a hand block is laid out for. */
export interface HandBlockOptions {
  n: number;
  scale: number;
  faceUp: boolean;
  room: number;
  liftRoom: number;
  rows?: number;
}

/**
 * A hand slot centred in its block, with room for its fullest deal: in one
 * row, or wrapped onto `rows` rows (the slot wraps only when its hand does
 * not fit one row, so a short hand keeps one row, centred in the block).
 */
export function handBlock(
  id: string,
  mode: 'hand-overlap' | 'hand-arc',
  o: HandBlockOptions,
  m: HandMeasure,
): Block {
  const rows = o.rows ?? 1;
  const opts = {faceUp: o.faceUp, ...(rows > 1 ? {rows} : {})};
  // The room holds a picked card too: an arc card lifts along its own axis,
  // so it leaves the unlifted hand's width. (Its depth is liftRoom.)
  const lifted = o.liftRoom > 0 ? {selectable: true, lift: PICK_LIFT} : {};
  const room = Math.max(o.room, m.extent(mode, o.n, {...opts, ...lifted}));
  const extent = room * o.scale;
  // The depth is measured in the room the hand is given: an arc given more
  // than its floor spreads into it and deepens, and wrapped rows are as deep
  // as they are drawn there.
  const depth = m.depth(mode, o.n, {...opts, room}) * o.scale;
  const height = o.liftRoom + depth;
  return {
    slots: [
      {
        // Centred in the block: a hand with a picked card draws a box
        // liftRoom taller (at the floor width), centred on the same point.
        id,
        x: extent / 2,
        y: height / 2,
        pile: mode,
        anchor: 'center',
        maxExtent: extent,
        label: 'No cards',
        // Laid out for its fullest deal: a drawn card appears at the end and
        // no other card moves (no re-centring, re-spacing or re-wrap).
        capacity: o.n,
        ...(o.scale === 1 ? {} : {scale: o.scale}),
        ...(rows > 1 ? {wrap: 'auto' as const, maxRows: rows} : {}),
      },
    ],
    regions: [],
    width: extent,
    height,
  };
}

function supplySlots(
  stockX: number,
  discardX: number,
  pileY: number,
  scale: number,
): SlotSpec[] {
  const scaled = scale === 1 ? {} : {scale};
  return [
    {
      id: GIN_SLOT.stock,
      x: round3(stockX),
      y: round3(pileY),
      pile: 'squared',
      stagger: 3,
      countBadge: true,
      label: 'Stock',
      ...scaled,
    },
    {
      id: GIN_SLOT.discard,
      x: round3(discardX),
      y: round3(pileY),
      pile: 'squared',
      label: 'Discard',
      ...scaled,
    },
  ];
}

function statsRegion(
  shape: 'wide' | 'narrow',
  side: number,
  stats: number,
  row: number,
  width: number,
): RegionSpec {
  return shape === 'wide'
    ? {
        id: GIN_REGION.stats,
        x: 0,
        y: 0,
        width: round3(side),
        height: round3(stats),
        className: 'gin-stats-region gin-stats-region--wide',
      }
    : {
        id: GIN_REGION.stats,
        x: 0,
        y: round3(row + GAP),
        width,
        height: stats,
        className: 'gin-stats-region gin-stats-region--narrow',
      };
}

/**
 * Stock and upcard with the stats left and the actions right ('wide'), or
 * the actions right and the stats in a row below ('narrow'). The stock and
 * upcard are drawn at `scale` of the table's card width: the size of the
 * player's own cards (a spread grid's cells are GRID of it), so the pile a
 * card is drawn from and the hand it joins show it at one size. `side` is
 * the width of the wide shape's stats and action columns; `stats` the
 * stats' height: the wide column's (from the top of the block; the piles
 * and actions sit at its foot) or the narrow row's (below the piles).
 */
export function supplyBlock(
  shape: 'wide' | 'narrow',
  cardHeight: number,
  scale: number,
  side: number,
  stats: number,
): Block {
  if (!(scale > 0 && scale <= 1)) {
    throw new Error(`gin supply scale must be in (0, 1], got ${String(scale)}`);
  }
  if (!(side >= SIDE)) {
    throw new Error(
      `gin supply side must be at least ${String(SIDE)}, got ${String(side)}`,
    );
  }
  const stockX = shape === 'wide' ? side + GAP : 0;
  const discardX = stockX + scale * (1 + STOCK_DEPTH) + GAP;
  const actionsX = discardX + scale + GAP;
  const actionsW = shape === 'wide' ? side : NARROW_ACTIONS;
  const width = actionsX + actionsW;
  const row = scale * (cardHeight + STOCK_DEPTH);
  // The actions beside the piles are as tall as a pile, and so is the wide
  // stats column at least; a taller one lifts the piles to its foot.
  const sideHeight = scale * cardHeight;
  if (shape === 'wide' && !(stats >= sideHeight - 1e-9)) {
    throw new Error(
      `gin wide stats must be at least a pile tall (${String(sideHeight)}), got ${String(stats)}`,
    );
  }
  const pileY = shape === 'wide' ? stats - sideHeight : 0;
  return {
    slots: supplySlots(stockX, discardX, pileY, scale),
    regions: [
      statsRegion(shape, side, stats, row, width),
      {
        id: GIN_REGION.actions,
        x: round3(actionsX),
        y: round3(pileY),
        width: round3(actionsW),
        height: round3(sideHeight),
      },
    ],
    width: round3(width),
    height: round3(shape === 'wide' ? pileY + row : row + GAP + stats),
  };
}
