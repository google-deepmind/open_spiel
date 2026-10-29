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

// Hearts table as data: a hand-table recipe (TableSpec + CardStyle) that
// sections/table_layout.tsx draws with the shared <HandTable>. Restyling or re-laying
// out hearts means editing this file, never the game logic.
//
// Units are card widths. The layout is a function of the active deck's
// HandMeasure, so a deck whose index needs more room gets a bigger table
// (drawn smaller) instead of a covered index. A hand of backs is turned to
// face its player (west 90°, north 180°, east 270°); a face-up hand is drawn
// upright, as the bottom seat reads it (a turned face reads upside down or
// sideways). The name labels are table regions
// drawn upright beside their hands, and count in the legibility check. No
// name sits beside a hand: the table is as narrow as its hands and trick
// allow, because the game panel it fits is taller than it is wide.
//
// Three dresses of the four-seat table (HeartsDress):
//   - 'closed': play. Opponents are backs and stay backs (show all cards
//     is analysis only), so the table is sized for backs, and each name is
//     attached at the start of its hand, following it as it shrinks.
//   - 'hidden' and 'open': analysis, where show all cards can be switched
//     on and off. ONE geometry, sized for the larger of the backs and the
//     faces at every seat (fourSeatTable reserve), dressed with turned backs
//     ('hidden') or upright faces ('open'); the names are fixed at the
//     reserved boxes. Switching show all cards changes the cards and nothing
//     else: no hand, name, trick position or card size moves.
//
// On a phone (a table frame under HEARTS_COMPACT_BELOW px) the hidden table
// draws its compact variant instead: the opponents are squared stacks of
// backs with a count, across the top in seat order (west, north, east), so
// the table is as wide as your hand and the cards come out larger. Open
// hands keep every face: on a phone they draw upright, wrapped, as columns
// across the top or in the table's corners. Which variant is drawn is the
// library's choice by card width and HEARTS_LAYOUT_COST (heartsLayout).
//
// The parts every table shares are in hearts_layout_parts.ts, the four-seat
// table in hearts_layout_table.ts and the phone tables in
// hearts_layout_phone.ts.
import {
  type CardStyleOverride,
  type HandMeasure,
  type HandTableRecipe,
  type TableSpec,
  PLAYABLE_TAG,
  PLAYABLE_TAG_STYLE,
} from '../../cards/library/index.ts';
import {
  DEFAULT_HEARTS_HAND_SHAPE,
  HEARTS_DRESSES,
  type HeartsDress,
  type HeartsHandShape,
  type SeatRows,
} from './hearts_layout_parts.ts';
import {
  COMPACT_SOUTH_ROWS,
  OPEN_COLUMN_ROWS,
  QUAD_ROWS,
  compactClosed,
  compactOpen,
  quadOpen,
} from './hearts_layout_phone.ts';
import {tableFor} from './hearts_layout_table.ts';
import type {PlayerPosition} from './types.ts';

export {
  CENTRE_REGION,
  DEFAULT_HEARTS_HAND_SHAPE,
  HEARTS_DRESSES,
  HEARTS_HAND_SHAPES,
  MESSAGE_REGION,
  TRICK_SIDE_CLEAR,
  type HeartsDress,
  type HeartsHandShape,
} from './hearts_layout_parts.ts';

/** Every visual knob hearts overrides; the rest come from the app's
 *  CardStyleProvider (deck, colour scheme, fan spacing, ...). */
export const HEARTS_CARD_STYLE: CardStyleOverride = {
  selection: {lift: 14},
  // Flat: a soft shadow under each overlapped back added up to a dark halo
  // round every opponent's hand ("flat, straightforward, dignified,
  // classic").
  shadow: 'none',
  tags: {
    // A playable card under the pointer peeks up a little: the library's
    // paint-only hover lift, so nothing re-lays out and the card keeps
    // its place in the fan's stacking order (a hovered card popping to the
    // front is wrong).
    hover: {lift: 5},
    // On your turn, when only some of your cards may go, those stand up
    // out of the hand; the rest keep full contrast, never dimmed
    // (cards/library/playable.ts).
    [PLAYABLE_TAG]: PLAYABLE_TAG_STYLE,
  },
};

/** Card widths the table may be drawn at, px. Below `min` it scrolls. */
export const HEARTS_MIN_CARD_WIDTH = 40;
/** The largest card per size setting; the table fits its panel up to this. */
export const HEARTS_MAX_CARD_WIDTH = {sm: 60, md: 80, lg: 100} as const;
/** A card size setting. */
export type HeartsCardSize = keyof typeof HEARTS_MAX_CARD_WIDTH;

/** Table slot ids of each seat's hand and trick card, and its name region. */
export const SEAT_SLOT: Record<
  PlayerPosition,
  {hand: string; trick: string; name: string}
> = {
  top: {hand: 'north', trick: 'trick-n', name: 'name-north'},
  right: {hand: 'east', trick: 'trick-e', name: 'name-east'},
  bottom: {hand: 'south', trick: 'trick-s', name: 'name-south'},
  left: {hand: 'west', trick: 'trick-w', name: 'name-west'},
};

/** Table frames narrower than this, px, draw the compact hidden table
 *  (compactClosed). Phones in portrait have frames of 304 to 398 px (320 to
 *  414 px viewports); the desktop game panel at 1280 x 720 has a 434 px
 *  frame and keeps the full table. */
export const HEARTS_COMPACT_BELOW = 400;

/** Every hand on one row. */
const ONE_ROW: SeatRows = {side: 1, north: 1, south: 1};
/** The analysis four-seat tables ('hidden' and 'open'), the first preferred:
 *  each upright side hand on three rows (a column about as tall as a
 *  turned hand of backs), or four (narrower, for a deck whose index needs
 *  a wide step); north's and yours on one row, or two (a narrower table).
 *  Each hand still wraps only when it does not fit its seat in one row. */
const ANALYSIS_ROWS: readonly SeatRows[] = [
  {side: 3, north: 1, south: 1},
  {side: 4, north: 1, south: 1},
  {side: 3, north: 2, south: 2},
  {side: 4, north: 2, south: 2},
];
/** The hidden four-seat table with your hand on two rows: a desktop panel
 *  where your hand does not fit one row at a legible card width (a 13-card
 *  hand on the centred-index 'simple' deck is ~10 card widths). */
const CLOSED_WRAP: SeatRows = {side: 1, north: 1, south: 2};

/**
 * How much less hearts likes a table than the full four-seat one with every
 * hand on one row, as a ratio of card widths (TableVariant.cost): the
 * library draws the table whose cards come out largest after dividing by
 * these. A wrapped hand (per seat group wrapped) reads less like a hand;
 * the phone tables give up the table's shape (opponents as stacks, or
 * every hand upright in rows). So a desktop panel keeps the four-seat table
 * unless another draws its cards that much larger.
 */
export const HEARTS_LAYOUT_COST = {wrap: 1.3, phone: 1.25} as const;

/**
 * The hearts table for a deck's hand measures, in one dress (HeartsDress).
 * The library draws the variant whose cards come out largest for its cost
 * (HEARTS_LAYOUT_COST), at or above the smallest legible card; the
 * variants, in order of preference:
 *   - 'closed' (play), a frame at least HEARTS_COMPACT_BELOW px wide: the
 *     four-seat table, then with your hand on two rows (CLOSED_WRAP). Below
 *     that (a phone) they give way to the compact phone table
 *     (compactClosed): the opponents as stacks, your hand on one row, then
 *     two.
 *   - 'hidden' and 'open' (analysis): the four-seat table, its upright side
 *     hands on three rows or four, north's and yours on one or two
 *     (ANALYSIS_ROWS); where that does not fit (a phone, a deck with a wide
 *     hand), the open phone tables: the opponents as upright columns across
 *     the top (compactOpen), or every hand in a corner (quadOpen). Open
 *     hands show every face, so no seat shrinks to a stack; hidden ones are
 *     backs in the same places.
 */
export function heartsLayout(
  m: HandMeasure,
  dress: HeartsDress,
  shape: HeartsHandShape = DEFAULT_HEARTS_HAND_SHAPE,
): TableSpec {
  if (!HEARTS_DRESSES.includes(dress)) {
    throw new Error(`hearts layout: unknown dress ${JSON.stringify(dress)}`);
  }
  const {wrap, phone} = HEARTS_LAYOUT_COST;
  if (dress !== 'closed') {
    // 'hidden' and 'open' share every layout, in the same order, so the
    // library draws the same one at the same card width for both: the
    // phone layouts' hands are upright either way (backs or faces).
    const [first = ONE_ROW, ...rest] = ANALYSIS_ROWS;
    return {
      ...tableFor(m, dress, first, shape),
      variants: [
        ...rest.map(rows => ({
          spec: tableFor(m, dress, rows, shape),
          cost: wrap ** (rows.south > 1 ? 1 : 0),
        })),
        ...OPEN_COLUMN_ROWS.map(rows => ({
          spec: compactOpen(m, rows),
          cost: phone * wrap,
        })),
        ...QUAD_ROWS.map(rows => ({
          spec: quadOpen(m, rows),
          cost: phone * wrap,
        })),
      ],
    };
  }
  return {
    ...tableFor(m, dress, ONE_ROW, shape),
    minWidth: HEARTS_COMPACT_BELOW,
    variants: [
      {
        spec: tableFor(m, dress, CLOSED_WRAP, shape),
        minWidth: HEARTS_COMPACT_BELOW,
        cost: wrap,
      },
      ...COMPACT_SOUTH_ROWS.map(rows => ({
        spec: compactClosed(m, rows, shape),
        cost: phone * wrap ** (rows - 1),
      })),
    ],
  };
}

const RECIPE: Record<HeartsDress, {id: string; label: string}> = {
  closed: {id: 'hearts', label: 'Hearts'},
  hidden: {id: 'hearts-analysis', label: 'Hearts, analysis (hands hidden)'},
  open: {id: 'hearts-open', label: 'Hearts, open hands'},
};

function recipe(dress: HeartsDress, shape: HeartsHandShape): HandTableRecipe {
  return {
    id: `${RECIPE[dress].id}${shape === DEFAULT_HEARTS_HAND_SHAPE ? '' : `-${shape}`}`,
    label: `${RECIPE[dress].label}${shape === DEFAULT_HEARTS_HAND_SHAPE ? '' : `, ${shape} hand`}`,
    source: 'games/hearts',
    style: HEARTS_CARD_STYLE,
    layout: (m: HandMeasure) => heartsLayout(m, dress, shape),
  };
}

/** The Hearts tables of one layout: `closed` for play (hidden opponents,
 *  sized for backs), and for analysis, where show all cards switches, one
 *  geometry in two dresses: `hidden` (backs) and `open` (faces, upright). */
export type HeartsTables = Readonly<Record<HeartsDress, HandTableRecipe>>;

/** The tables per hand shape. */
export const HEARTS_TABLES: Readonly<Record<HeartsHandShape, HeartsTables>> = {
  flat: {
    closed: recipe('closed', 'flat'),
    hidden: recipe('hidden', 'flat'),
    open: recipe('open', 'flat'),
  },
  curved: {
    closed: recipe('closed', 'curved'),
    hidden: recipe('hidden', 'curved'),
    open: recipe('open', 'curved'),
  },
};

/** The tables at the default hand shape. */
export const HEARTS_TABLE = HEARTS_TABLES[DEFAULT_HEARTS_HAND_SHAPE];
