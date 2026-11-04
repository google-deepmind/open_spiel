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

// Gin rummy's table as data: ONE TableSpec (where the opponent's hand, the
// stock and upcard, the player's hand, the action bar and the stats sit, in
// card widths) plus the CardStyle gin plays in, its state tags included.
// Restyling or re-laying out the board means editing these files, never the
// game logic in the panel. This file holds the card codes, the style, the
// slot ids and the sizes; gin_layout_blocks.ts the blocks the table is built
// from; gin_layout_table.ts the table and its variants.
//
// The whole board is one shared <Table> (gin_rummy_board_surface.tsx): the
// library fits that one spec to the panel, so every card on it is drawn at
// one scale and nothing here, or in the board, measures or scales anything
// itself.

import {
  cardKey,
  type Card,
  type CardStyleOverride,
  type Rank,
  type Suit,
} from '../../cards/library/index.ts';
import {GIN_MIN_CARD_WIDTH} from './gin_layout_stats.ts';
import type {GridStyle} from './gin_view_settings.ts';

export {GIN_MIN_CARD_WIDTH} from './gin_layout_stats.ts';

const RANK_OF_CODE: Readonly<Record<string, Rank>> = {
  A: 'A',
  '2': '2',
  '3': '3',
  '4': '4',
  '5': '5',
  '6': '6',
  '7': '7',
  '8': '8',
  '9': '9',
  T: '10',
  J: 'J',
  Q: 'Q',
  K: 'K',
};

const SUIT_OF_CODE: Readonly<Record<string, Suit>> = {
  S: 'S',
  H: 'H',
  D: 'D',
  C: 'C',
};

/** A gin card code ('As', 'Th', '5d') as a shared Card. Throws on anything else. */
export function ginCard(code: string): Card {
  const m = /^([2-9TJQKA])([shdc])$/i.exec(code);
  const rank =
    m?.[1] === undefined ? undefined : RANK_OF_CODE[m[1].toUpperCase()];
  const suit =
    m?.[2] === undefined ? undefined : SUIT_OF_CODE[m[2].toUpperCase()];
  if (rank === undefined || suit === undefined) {
    throw new Error(`not a gin card code: '${code}'`);
  }
  return {rank, suit};
}

/** The shared scaffold's data-card value for a gin code: 'Th' -> '10H'. */
export function ginCardKey(code: string): string {
  return cardKey(ginCard(code));
}

/**
 * The board's card style. 'app' inherits the shared deck; 'modern' is gin's
 * own rank-over-suit face, the 'simple' deck.
 */
export type GinCardFace = 'app' | 'modern';
/** Every card face, in the menu's order. */
export const GIN_CARD_FACES: readonly GinCardFace[] = ['app', 'modern'];

/** Per card face, over GIN_CARD_STYLE. */
export const GIN_FACE_STYLES: Readonly<Record<GinCardFace, CardStyleOverride>> =
  {
    app: {},
    modern: {deck: 'simple'},
  };

/** Lift of a picked hand card, px. */
export const PICK_LIFT = 10;

/** Gin's CardStyle, over the app's. Its colours are tokens (tokens.css). */
export const GIN_CARD_STYLE: CardStyleOverride = {
  // The hidden opponent hand: 18px apart on a 64px card.
  fan: {faceDown: 0.28},
  radius: 0.1,
  // Flat: a soft shadow under each of the opponent's overlapped backs adds
  // up to a dark halo round the hand.
  shadow: 'none',
  // The staged discard ('selected') is a ring, never a lift: the hand's
  // spacing is only legible for unlifted cards.
  selection: {lift: 0, outline: 'var(--gin-selected-ring)'},
  // Gin's card states (components/card_state.ts ginTags). No state changes
  // a card's opacity: a ghost is washed out by a tint over an opaque face.
  tags: {
    'meld-run': {band: 'var(--gin-meld-run)'},
    'meld-set': {band: 'var(--gin-meld-set)'},
    layoff: {ring: 'var(--gin-layoff)'},
    // A staged lay-off on the knocker's meld: its 'layoff' ring plus a
    // glow, at full strength (never washed out like a ghost).
    'staged-layoff': {glow: 'var(--gin-staged-layoff-glow)'},
    alert: {ring: 'var(--gin-alert-ring)', glow: 'var(--gin-alert-glow)'},
    ghost: {filter: 'grayscale(0.7)', tint: 'var(--gin-ghost-tint)'},
    // A card a grid names without holding it (CardCellState.mark): its
    // symbol alone, centred, one flat colour whatever the suit; no body.
    dead: {symbol: 'var(--gin-card-dead-ink)'},
    known: {symbol: 'var(--gin-card-known-ink)'},
    // Shift+click (or X) marks hand cards to move together: lifted, so the
    // group reads at a glance. The hand keeps room for the lift.
    picked: {ring: 'var(--gin-picked-ring)', lift: PICK_LIFT},
    hover: {glow: 'var(--gin-hover-glow)'},
  },
};

/** The suits and ranks a gin variant deals (deck_context.tsx DeckConfig). */
export interface GinGridDeck {
  suitRows: ReadonlyArray<{readonly code: string}>;
  rankCodes: readonly string[];
}

/** Every code of a gin variant's grid, suit by suit ('As', '2s', …). */
export function gridCodes(deck: GinGridDeck): string[] {
  return deck.suitRows.flatMap(suit =>
    deck.rankCodes.map(rank => `${rank}${suit.code}`),
  );
}

/** Slot ids. Grid cells are `${GRID_PREFIX[zone]}${code}` ('me-As'). */
export const GIN_SLOT = {
  opponent: 'opponent',
  stock: 'stock',
  discard: 'discard',
  hand: 'hand',
} as const;
/** The prefix of each grid's cell slot ids. */
export const GRID_PREFIX = {opponent: 'opp-', player: 'me-'} as const;
/** A side of the table: the opponent's, or the player's. */
export type GinZone = keyof typeof GRID_PREFIX;

/** The slot id of `code`'s cell in `zone`'s grid. */
export function gridSlotId(zone: GinZone, code: string): string {
  return `${GRID_PREFIX[zone]}${code}`;
}

/** What the board shows, from the settings and the game. */
export interface GinTableOptions {
  /** The opponent's hand face down, or the suit x rank grid (revealed). */
  opponent: 'hidden' | 'grid';
  /** The player's hand fanned (Hand view: Flat), on an arc, or the grid. */
  player: 'hand-overlap' | 'hand-arc' | 'grid';
  /**
   * Grid style: 'normal' draws a panel behind each grid and outlines its
   * empty cells; 'slots' outlines the empty cells with no panel; 'no-bg'
   * draws neither; 'compact' overlaps the cells (bigger cards, no panel).
   */
  grid: GridStyle;
  deck: GinGridDeck;
  /**
   * The codes each grid holds ('As', 'Th'): a grid paints its empty cells
   * under its cards. It never moves a cell: every rank column is drawn
   * whatever the grid holds, so a drawn card only fills its own cell.
   */
  held?: Partial<Record<GinZone, readonly string[]>>;
  /**
   * Cards the fanned hand holds now, staged layoffs (ghosts) included. The
   * hand keeps room for MAX_HAND (a full hand and the drawn card), or for
   * this many when a layoff stages more. Omitted, MAX_HAND.
   */
  handCards?: number;
}

/**
 * Region ids: the action bar and the stats sit in the supply row; a grid in
 * the 'normal' style has a panel behind it.
 */
export const GIN_REGION = {
  actions: 'actions',
  stats: 'stats',
  opponentPanel: 'opp-panel',
  playerPanel: 'me-panel',
} as const;

/**
 * Most cards a gin hand holds (ten, plus the drawn card): the room a fanned
 * hand keeps through play.
 */
export const MAX_HAND = 11;
/**
 * Most staged layoffs drawn as ghosts after the fanned hand: while any are
 * staged the hand keeps room for them too (GinTableOptions.handCards).
 */
export const MAX_GHOSTS = 4;
/** Card width on the table at the largest scale it draws, px. */
export const GIN_MAX_CARD_WIDTH = 96;
/**
 * The floor of a spread grid's cells, px. Thirteen cells side by side at
 * 40 px do not fit the Gameplay panel at 1280x720 (470 px), and overlapping
 * them is the compact style the player did not choose: a spread grid draws
 * its cells smaller instead (ginMinCardWidth).
 */
export const GIN_MIN_GRID_CELL_WIDTH = 28;

/**
 * The board's card-width floor for a view (the <Table>'s minCardWidth).
 * Every table has a grid's room in it (the opponent's area is laid out for
 * the revealed grid even while the hand is face down, so the hand's end
 * moves nothing): a spread style ('normal', 'slots', 'no-bg')
 * floors its cells at GIN_MIN_GRID_CELL_WIDTH, the compact one at
 * GIN_MIN_CARD_WIDTH.
 */
export function ginMinCardWidth(o: Pick<GinTableOptions, 'grid'>): number {
  return o.grid !== 'compact' ? GIN_MIN_GRID_CELL_WIDTH : GIN_MIN_CARD_WIDTH;
}

/** The opponent's hidden hand, relative to the player's cards. */
export const SMALL = 0.84;
/** A normal grid cell, relative to the player's cards. */
export const GRID = 0.71;

/**
 * Card widths, px, the hands are measured over (measureHands): every width a
 * hand's own cards are drawn at, the opponent's smaller ones included,
 * beside a spread grid at its floor too (its overlapped last resort draws
 * every card at the grid's floor).
 */
export const GIN_HAND_WIDTHS = {
  min: Math.floor(
    Math.min(GIN_MIN_CARD_WIDTH, GIN_MIN_GRID_CELL_WIDTH) * SMALL,
  ),
  max: GIN_MAX_CARD_WIDTH,
};

/**
 * How much less gin likes a layout than its first choice, as a ratio of card
 * widths (TableVariant.cost): the library draws the layout whose cards come
 * out largest after dividing by these. A compact grid packed down to the
 * index hides more of each card than its half-card overlap ('tight');
 * wrapping a grid's ranks onto two bands breaks the rank order across the
 * table ('bands'); wrapping the fanned hand onto another row breaks the hand
 * in two ('row', again for each row). A 'normal', 'slots' or 'no-bg' grid is
 * never overlapped where a spread one can be drawn at all: its overlapped
 * shapes cost 'squeeze', so they are drawn only in a panel too narrow for
 * thirteen legible cells side by side, even on two bands (a phone). They
 * keep every cell fixed too.
 */
export const GIN_LAYOUT_COST = {
  tight: 1.3,
  bands: 1.2,
  row: 1.5,
  squeeze: 100,
} as const;

/**
 * Most rows the fanned hand wraps onto when one row does not fit the panel
 * at a legible card width (fifteen cards: five a row).
 */
export const GIN_HAND_MAX_ROWS = 3;
