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

// TableSpec: a card table as data. A game's layout is a
// TableSpec (where each pile sits and how it fans) plus a CardStyle (how cards
// look); <Table> renders the pair. Re-laying out a game means editing a spec,
// never game logic.
//
// UNITS. Every length in a spec is in card widths, measured from the table's
// top-left: x = 1 is one card width to the right. <Table> multiplies by the
// rendered card width, so one spec serves every size and every deck.
import type {Deck, IndexGeometry} from './types.ts';
import {
  indexClearance,
  PAD_FRACTION,
  PAD_MIN_PX,
  type PileMode,
} from './pile_geometry.ts';
import {isHandMode, type HandMode, type HandWrap} from './hand_geometry.ts';
import {itemAt} from './checked.ts';
import {deckIndex} from './index_scale.ts';

export {validateTableSpec} from './table_validate.ts';

/** One slot of a table: its place, mode and cards' room. */
export interface SlotSpec {
  /** Unique within the spec; the key of the slot's cards in <Table piles>. */
  id: string;
  /** Top-left of the slot's first card, card widths. */
  x: number;
  y: number;
  /** A pile mode, or a hand mode: a player's hand, flat, overlapped
   *  or on an arc, drawn by <Hand>. 'free': every item carries its own `at`
   *  (FreePlacement), card widths from the slot's origin (chips, a spread of
   *  cards in play). */
  pile: PileMode | HandMode | 'free';
  /** Longest the pile may grow along its fan axis, card widths; it compresses
   *  to index visibility, then scrolls. */
  maxExtent?: number | undefined;
  /** Clockwise degrees: 0, 90, 180 or 270 (the legibility check maps the
   *  index box exactly for right angles and cannot for anything else). */
  rotate?: number | undefined;
  /** Shown in the slot when it is empty. */
  label?: string | undefined;
  // ---- Additions to the pinned contract, all optional. ----
  /** Squared piles: depth layers drawn under the top card. */
  stagger?: number | undefined;
  /** Show the card count as a badge (stocks). */
  countBadge?: boolean | undefined;
  /** Fan modes: draw only the top N cards (a draw-three waste). */
  visibleCount?: number | undefined;
  /** This slot's cards are covered by later slots by the game's design (a
   *  card under the two that rest on it in the row below). The legibility check
   *  reports such an index as excused, never as covered, and never as
   *  checked: the declaration is the claim, and it is visible in the data. */
  coveredByDesign?: boolean | undefined;
  // ---- Additions for hands and tables, all optional. ----
  /** 'start' (default): (x, y) is the slot's top-left. 'center': (x, y) is
   *  the centre of what the slot draws, after rotation — seats and play
   *  positions, whose hands change length with the deal. 'end': (x, y) is
   *  the top-right of what the slot draws, so a hand against a table's
   *  right edge shrinks toward it (an unturned slot only). */
  anchor?: 'start' | 'center' | 'end' | undefined;
  /** Arc hands: pivot distance × card height. */
  arcRadius?: number | undefined;
  // ---- Captions, cues and badges, all optional. ----
  /** Text drawn in the CAPTION_HEIGHT band directly above the slot (a
   *  foundation's target sequence). The slot's y must leave
   *  room for it. <Table captions> replaces the text at run time (a hand
   *  score); a slot takes run-time captions only if it declares one, so
   *  the room is always reserved in the data. */
  caption?: string | undefined;
  /** Continuation cue for an ordered line that wraps: 'continues' draws an
   *  arrow in the CUE_WIDTH band right of the slot (the line goes on at the
   *  start of the next row), 'continued' one in the band left of it. */
  cue?: 'continues' | 'continued' | undefined;
  /** Slot tags that hold wherever this spec (or variant) is drawn, merged
   *  with <Table slotTags> for the render. Styled by CardStyle.tags like
   *  card tags: the look (ring, glow, cursor) draws on the EMPTY outline, and
   *  TagStyle.empty says whether an empty slot draws at all (the built-in
   *  'hide-empty': a play position that draws nothing, box kept). */
  tags?: readonly string[] | undefined;
  /** Card-size multiplier for this slot (a scaled sub-table: another seat
   *  drawn smaller). Lengths in the spec stay in the table's card widths;
   *  the slot's own cards are `scale` of that. Default 1. */
  scale?: number | undefined;
  // ---- Wrapped hands, all optional; hand slots only. ----
  /** 'auto': a hand that cannot fit maxExtent in one row at its legibility
   *  floor wraps onto the fewest rows (up to maxRows) that fit, row-major;
   *  'never': it compresses, then scrolls. Default CardStyle.hand.wrap. */
  wrap?: HandWrap | undefined;
  /** Most rows a wrapping hand takes (a whole number >= 1). Default
   *  CardStyle.hand.maxRows. Reserve the room it needs in the spec
   *  (HandMeasure's `rows` option). */
  maxRows?: number | undefined;
  /** Between the rows of a wrapped hand, × this slot's card width.
   *  Default CardStyle.hand.rowGap. */
  rowGap?: number | undefined;
  /** Hand slots: lay the hand out as if it held this many cards, the
   *  cards it holds taking the first positions. A hand that grows or
   *  shrinks below its capacity keeps every card where it was (no
   *  re-centring, no re-balanced rows, no re-spacing): a drawn card only
   *  appears at the end. A hand past its capacity is laid out as it is. */
  capacity?: number | undefined;
  /** The slot only ever holds face-down cards (another seat's hidden hand,
   *  a stock): nothing in it is read, so its `scale` does not raise the
   *  table's card-width floor (minSlotScale skips it) and the legibility
   *  check does not hold its cards to that floor. A compact seat of backs
   *  is drawn small without shrinking the cards that are read. A face-up
   *  card in it throws. */
  faceDown?: boolean | undefined;
  /** 'free' slots only: named places, card widths from the slot's origin
   *  (each a card's top-left), that an item takes by name (`at: 'n'`). So
   *  a layout that depends on the deck (the overlapping trick's pinwheel)
   *  keeps its geometry in the spec and the items only say whose card is
   *  whose; the items' order is still the paint order. */
  places?: Readonly<Record<string, {x: number; y: number}>> | undefined;
  /** Slots drawn as one block (a hand laid out by suit, one slot a suit).
   *  Every slot naming the same block `id` names the same `x`, `y` and
   *  `pack`. With `pack`, the block's slots that hold cards are drawn in
   *  its first slots, in spec order, and the empty ones last, so an empty
   *  slot (a void suit) leaves no gap. Then the block's slots all move by
   *  one offset so that the union of the drawn boxes of those that hold
   *  cards is centred on `x` and on `y`, card widths (an axis left out
   *  keeps the slots where the spec put them on it). The offset is taken
   *  from what is drawn, in px, so the block is centred exactly whatever
   *  its rows' lengths and the deck's fan steps. `alignX` / `alignY`
   *  ('start', 'center', 'end'; default 'center') put the union's left or
   *  top edge, middle, or right or bottom edge on `x` / `y` instead: a
   *  hand whose foot must sit on a line whatever it draws. A fan slot that
   *  holds only positional holes (null) counts as holding cards: its room
   *  is kept in the union and in the packing, so a hand whose rows keep
   *  the room of cards played from them (a dummy) never moves. Unturned
   *  slots, or slots half turned (180) about their middle (a hand, or a
   *  centred slot), which cover the same box; not 'free'. A block that
   *  holds no card does not move. */
  block?: SlotBlock | undefined;
  /** TableSpec.fillWidth: which edge of a widened table the slot keeps
   *  its distance from. 'left': its x stays; 'right': it moves by the
   *  whole width added. Unpinned, it moves by half (it stays centred). */
  pin?: TablePin | undefined;
  /** Fan slots: where the room of the positional holes (null) that end
   *  the slot's items is drawn: after its cards ('trailing', the default)
   *  or before them ('leading'). A row anchored at its end ('end') takes
   *  'leading', so its cards stay against its anchor while the row keeps
   *  the room of the cards it has played (a revealed dummy); the
   *  game says which cards are gone once, at the end of each row, and
   *  each slot draws that room on its free side. */
  holeRoom?: 'trailing' | 'leading' | undefined;
}

/** SlotSpec.pin, RegionSpec.pin. */
export type TablePin = 'left' | 'right';
/** Every side a widened table can pin to. */
export const TABLE_PINS: readonly TablePin[] = ['left', 'right'];

/** TableSpec.fillWidth: `layout` drawn `extra` card widths wider, the width
 *  added given out by pin. */
export function widenTableLayout<
  T extends Pick<TableSpec, 'width' | 'slots' | 'regions'>,
>(layout: T, extra: number): T {
  if (!(Number.isFinite(extra) && extra >= 0)) {
    throw new Error(
      `widenTableLayout: extra must be a finite number >= 0, got ${String(extra)}`,
    );
  }
  if (extra === 0) return layout;
  const dx = (pin: TablePin | undefined) =>
    pin === 'left' ? 0 : pin === 'right' ? extra : extra / 2;
  return {
    ...layout,
    width: layout.width + extra,
    slots: layout.slots.map(s => {
      const d = dx(s.pin);
      if (d === 0) return s;
      const b = s.block;
      return {
        ...s,
        x: s.x + d,
        ...(b?.x !== undefined ? {block: {...b, x: b.x + d}} : {}),
      };
    }),
    ...(layout.regions
      ? {
          regions: layout.regions.map(r =>
            r.x === undefined || dx(r.pin) === 0
              ? r
              : {...r, x: r.x + dx(r.pin)},
          ),
        }
      : {}),
  };
}

/** SlotSpec.block: where its edge or middle goes on an axis. */
export type BlockAlign = 'start' | 'center' | 'end';
/** SlotSpec.block. */
export interface SlotBlock {
  id: string;
  x?: number | undefined;
  y?: number | undefined;
  pack?: boolean | undefined;
  alignX?: BlockAlign | undefined;
  alignY?: BlockAlign | undefined;
}

/** A slot whose box is the same turned or not: unturned, or half turned
 *  about its middle (a hand, or a centred slot). Blocks and region unions
 *  take these. */
export function turnsInPlace(
  s: Pick<SlotSpec, 'rotate' | 'pile' | 'anchor'>,
): boolean {
  const r = (((s.rotate ?? 0) % 360) + 360) % 360;
  return (
    r === 0 || (r === 180 && (isHandMode(s.pile) || s.anchor === 'center'))
  );
}

/** Per-item offset for 'free' slots, in card widths from the slot origin, or
 *  the name of one of the slot's `places` (the layout keeps the geometry, the
 *  item says which place it takes). Required for items in 'free' slots;
 *  forbidden elsewhere (throws). */
export interface FreePlacement {
  at: {x: number; y: number} | string;
}

/** Non-card content placed on the table (an action bar, stats, a seat name):
 *  <Table regions> supplies the content, the spec the box. */
export interface RegionSpec {
  id: string;
  /** Fixed placement, in card widths (same frame as slots). Either x/y or attach. */
  x?: number | undefined;
  y?: number | undefined;
  /** Card widths. */
  width: number;
  height: number;
  /** Attach to a slot. The box is placed on `side` of the slot's box in the
   *  slot's LOCAL (unrotated) frame, `gap` away (card widths, default 0),
   *  aligned by `align` (default 'center'). Then the box centre is
   *  transformed by the slot's rotation. upright: the box is drawn unrotated
   *  about that centre (seat names read horizontally), and on a slot turned
   *  90 or 270 degrees it is placed by that on-screen footprint (h x w in
   *  the local frame), so it still sits `gap` from the slot; otherwise it
   *  rotates with the slot.
   *
   *  `also`: more slots the box is placed on with `slot`, as one: on the
   *  union of the boxes of those of them (`slot` too) that hold an item,
   *  or on `slot`'s own box when none does. So a seat's badge stays on its
   *  hand however the hand is drawn (a stack of backs in one slot, or one
   *  row a suit), and moves with it (SlotSpec.block). Every slot named is
   *  unturned (validateTableSpec). */
  attach?: {
    slot: string;
    also?: readonly string[] | undefined;
    side: 'top' | 'bottom' | 'left' | 'right';
    gap?: number | undefined;
    align?: 'start' | 'center' | 'end' | undefined;
  };
  /** Default true when attach is set (and irrelevant otherwise). */
  upright?: boolean | undefined;
  /** Counts as cover in the legibility check. Default true. */
  occludes?: boolean | undefined;
  /** A fixed (x/y) region only: SlotSpec.pin. An attached region moves
   *  with its slot. */
  pin?: TablePin | undefined;
  /** Is a drop target of kind 'region'. Default false. */
  droppable?: boolean | undefined;
  /** Class names for the region's wrapper where this spec (or variant) is
   *  drawn, so the game's CSS can lay the content out per layout (a name
   *  that hugs the end of its hand in one variant and is centred under a
   *  stack in another). The content itself is the same in every variant. */
  className?: string | undefined;
}

/** 'width': fill the container's width. 'contain': also fit its height
 *  (the container must have one). */
export type TableFit = 'width' | 'contain';

/** Height of a caption band, card widths. */
export const CAPTION_HEIGHT = 0.3;
/** Width of a continuation-cue band, card widths. */
export const CUE_WIDTH = 0.3;

/** A table's layout: its size, slots and regions, in card widths. */
export interface TableSpec {
  /** Card widths. The table is at least this big; piles that grow past
   *  `height` extend it rather than overlap what is below. */
  width: number;
  height: number;
  /** Paint order: a later slot is drawn over an earlier one. */
  slots: SlotSpec[];
  /** Responsive alternatives, in order of preference after this
   *  spec. <Table> fits each layout of [this spec, ...variants] that may be
   *  drawn in its container (see `minWidth`):
   *   - fit 'contain' (the container has a height): it draws the one whose
   *     cards come out largest, at or above the table's minimum card
   *     width, each card width divided by its layout's `cost`
   *     (TableVariant.cost; this spec's is 1); a tie goes to the earlier
   *     layout. So a bigger container never draws smaller cards for want
   *     of trying a layout, and a layout the game likes less (a wrapped
   *     hand) wins only by the margin its cost names.
   *   - fit 'width' (no height to trade against, so a narrower layout
   *     always draws larger cards): the first that fits, in order.
   *  If none fits, the one that scrolls least is drawn: one that scrolls
   *  only down before one that scrolls sideways, then the largest card.
   *  Every variant names exactly this spec's slot ids, so a game's piles
   *  never depend on which is drawn. */
  variants?: TableVariant[] | undefined;
  /** Container width in px below which this spec is not drawn and its
   *  variants are tried instead (it must still fit where it is drawn). A
   *  wide layout that would squeeze onto a phone at small cards gives way
   *  to a compact variant drawn larger. Only on a spec with variants; a
   *  variant says when it applies with TableVariant.minWidth. */
  minWidth?: number | undefined;
  /** Card width in px, as drawn (after <Table cardScale>), below which this
   *  spec is passed over for a variant drawn at or above its own. A layout
   *  whose regions hold text with a px floor (TEXT_FLOOR_PX) sizes them in
   *  card widths, so they hold that text only down to some card width (a
   *  column sized for text at 0.2 card widths holds 12 px text from a 60
   *  px card). It is passed over even for a variant that scrolls (text
   *  drawn over cards is worse than a scroll); where every layout is under
   *  its own, the usual choice is made among all. A variant names its own
   *  with TableVariant.minDrawnCardWidth. */
  minDrawnCardWidth?: number | undefined;
  /** Non-card content boxes (RegionSpec). Every variant names the same
   *  region ids; a region lies inside the table, so it counts toward fit. */
  regions?: RegionSpec[] | undefined;
  /** fit 'contain': a table drawn in a box wider than itself (its height,
   *  or the largest card, set the card width) takes the box's whole
   *  width. The width added goes to the slots and fixed regions by their
   *  `pin`: pinned 'left' keep their place, pinned 'right' move by all of
   *  it, the rest by half, so a four-seat table's side hands keep their
   *  margin from the felt's edges and the middle stays centred. The card
   *  width and the layout drawn do not change (it is chosen and fitted on
   *  `width`). A table that scrolls is drawn at `width`. */
  fillWidth?: boolean | undefined;
}

/** An alternative layout of a table, for a size it fits. */
export interface TableVariant {
  /** Container width in px below which this variant is not drawn. */
  minWidth?: number | undefined;
  /** Drawn card width in px below which this variant is passed over
   *  (TableSpec.minDrawnCardWidth). */
  minDrawnCardWidth?: number | undefined;
  /** How much less the game prefers this layout than its base spec, as a
   *  ratio of card widths (a finite number >= 1, default 1): it is drawn
   *  over another layout only when its card width divided by `cost` is
   *  larger than that layout's card width divided by its own. A variant
   *  that wraps a hand onto two rows at cost 1.5 wins only where its cards
   *  come out half again as wide as the best layout that does not. */
  cost?: number | undefined;
  spec: Omit<TableSpec, 'variants' | 'minWidth' | 'minDrawnCardWidth'>;
}

// ---- Overlap derived from the deck. ----
//
// A table that overlaps slots (overlapped rows, a wrapping ordered
// line, a narrow row of thirteen, a fanned waste beside the foundations) must not
// fix the overlap in card widths: how far a card may cover the one under it
// depends on where the ACTIVE deck draws its index. A corner index clears at
// ~0.55 card widths down; the centred 'simple' block needs ~1.1. So such a
// recipe is a TableLayout: a function of the deck's TableClearance, and
// <Table> resolves it for the deck and style it draws with. The recipe
// declares the relationship ("row r+1 overlaps row r, leaving the index
// visible": rowStep = max(preferred, clear.down)); the numbers come from the
// same IndexGeometry pileGeometry fans piles by. A layout that grows past its
// container is fitted smaller and, below the minimum card width, scrolls
// (data-table-overflow): it never covers an index to stay small.

/** How close cards may overlap on a table without covering an index, in card
 *  widths, for one deck at one CardStyle. */
export interface TableClearance {
  /** Least step from a face-up card to one drawn lower in the same column
   *  (an overlapped row below): the index box's bottom edge plus padding. */
  down: number;
  /** Least step from a face-up card to one drawn to its right (a
   *  horizontally overlapped row): the index box's right edge plus padding. */
  right: number;
  /** Most a fan-right pile advances per face-up card as <Pile> lays it out
   *  (the floor times CardStyle fan.horizontal, rounding included), so a
   *  recipe can leave room for a fanned waste. */
  fanRight: number;
}

/** A table whose overlaps are derived from the deck (see TableClearance). */
export type TableLayout = (clear: TableClearance) => TableSpec;

/** What <Table spec> and a recipe take: a fixed spec (no overlap that
 *  depends on the index) or a TableLayout. */
export type TableSource = TableSpec | TableLayout;

/**
 * The clearance of `deck` drawn at `style`, for cards `widths.min` to
 * `widths.max` px wide. The px padding floor weighs most on the smallest
 * card, so `down` and `right` are taken there and hold for every width
 * above it; `fanRight` is <Pile>'s own offset (indexClearance, rounded px),
 * the most it comes to over every whole-px width in range.
 */
export function tableClearance(
  deck: Pick<Deck, 'aspect' | 'index' | 'indexAt'>,
  style: {indexScale: number; fan: {horizontal: number}},
  widths: {min: number; max: number},
): TableClearance {
  const {min, max} = widths;
  if (!(
    Number.isFinite(min) &&
    min > 0 &&
    Number.isFinite(max) &&
    max >= min
  )) {
    throw new Error(
      `tableClearance widths must satisfy 0 < min <= max, got ${String(min)}..${String(max)}`,
    );
  }
  if (!(deck.aspect > 0)) {
    throw new Error(
      `tableClearance deck aspect must be > 0, got ${String(deck.aspect)}`,
    );
  }
  const index = deckIndex(deck, style.indexScale);
  const cardH = 1 / deck.aspect;
  const pad = (size: number) => Math.max(PAD_FRACTION * size, PAD_MIN_PX / min);
  const down = (index.top + index.height) * cardH + pad(cardH);
  const right = index.left + index.width + pad(1);
  const factor = Math.max(1, style.fan.horizontal);
  let fanRight = 0;
  for (let w = min; w <= max; w = w === min ? Math.floor(min) + 1 : w + 1) {
    const floor = indexClearance(index, w, w * cardH, 'right');
    fanRight = Math.max(
      fanRight,
      Math.max(floor, Math.round(floor * factor)) / w,
    );
  }
  const up = (v: number) => Math.ceil(v * 1000) / 1000;
  return {down: up(down), right: up(right), fanRight: up(fanRight)};
}

/** The spec to draw: `source` itself, or its layout at `clear`. */
export function resolveTableSpec(
  source: TableSource,
  clear: TableClearance,
): TableSpec {
  if (typeof source !== 'function') return source;
  for (const [k, v] of Object.entries(clear)) {
    if (!(Number.isFinite(v) && v > 0)) {
      throw new Error(
        `table clearance ${k} must be finite and > 0, got ${String(v)}`,
      );
    }
  }
  return source(clear);
}

/** How one layout fits a container (fitTableCardWidth, plus whether it
 *  scrolls sideways at the smallest card). */
export interface LayoutFit {
  cardWidth: number;
  /** The cards would come out under the minimum card width: it scrolls. */
  overflow: boolean;
  /** Its width alone does not fit at the minimum card width. */
  sideways: boolean;
  /** The card width that fits before the table's bounds clamp it, px:
   *  under the minimum, how far the layout is from fitting. */
  raw: number;
}

/** The spec <Table> draws in a container `boxWidth` px wide (see
 *  TableSpec.variants): of [spec, ...variants] that may be drawn at that
 *  width (the base spec at or above its minWidth, a variant at or above
 *  its own), with `fit` 'contain' the one that fits with the largest card
 *  width divided by its cost, the earlier on a tie, and with 'width' the
 *  first that fits; if none fits, the one that scrolls least. A layout
 *  drawn under its minDrawnCardWidth (its fitted card width times
 *  `cardScale`, as <Table> draws it) is left out of that choice while any
 *  layout is not. `exclude` names layouts (0 the base, k variant k - 1)
 *  left out altogether. `fitOf` fits one layout to the container.
 *  Unmeasured (width 0), the base spec. */
export function selectTableVariant(
  spec: TableSpec,
  boxWidth: number,
  fitOf: (layout: Omit<TableSpec, 'variants'>) => LayoutFit,
  fit: TableFit = 'width',
  cardScale = 1,
  exclude: ReadonlySet<number> = new Set(),
): {layout: Omit<TableSpec, 'variants'>; index: number} {
  if (!(cardScale > 0 && Number.isFinite(cardScale))) {
    throw new Error(
      `selectTableVariant: cardScale must be a finite number > 0, got ${String(cardScale)}`,
    );
  }
  const candidates: Array<{
    minWidth?: number | undefined;
    minDrawnCardWidth?: number | undefined;
    cost?: number | undefined;
    spec: Omit<TableSpec, 'variants'>;
  }> = [
    {minWidth: spec.minWidth, minDrawnCardWidth: spec.minDrawnCardWidth, spec},
    ...(spec.variants ?? []),
  ];
  // Unmeasured (first render, or no layout engine): the base spec.
  if (!(boxWidth > 0) || candidates.length === 1) {
    return {layout: spec, index: 0};
  }
  const drawable = candidates
    .map((c, index) => ({c, index}))
    .filter(
      ({c, index}) =>
        !exclude.has(index) &&
        (c.minWidth === undefined || boxWidth >= c.minWidth),
    );
  // Every layout gated out by its minWidth (a container narrower than all
  // of them): the last, the narrowest by convention.
  if (drawable.length === 0) {
    const index = candidates.length - 1;
    return {layout: itemAt(candidates, index).spec, index};
  }
  const fitted = drawable.map(({c, index}) => ({
    index,
    spec: c.spec,
    cost: c.cost ?? 1,
    minDrawn: c.minDrawnCardWidth,
    fit: fitOf(c.spec),
  }));
  // A layout under its minDrawnCardWidth would draw its regions' text
  // over its cards: it is passed over, even for one that scrolls, while
  // any other qualifies.
  const legible = fitted.filter(
    f =>
      f.minDrawn === undefined ||
      Math.round(f.fit.cardWidth * cardScale) >= f.minDrawn,
  );
  const pool = legible.length > 0 ? legible : fitted;
  const fits = pool.filter(f => !f.fit.overflow);
  let best: (typeof fitted)[number];
  if (fits.length > 0) {
    best =
      fit === 'contain'
        ? fits.reduce((a, b) =>
            b.fit.cardWidth / b.cost > a.fit.cardWidth / a.cost ? b : a,
          )
        : itemAt(fits, 0);
  } else {
    // None fits: scroll least. Down only before sideways, then the
    // largest card (the least scroll), the earlier on a tie.
    best = pool.reduce((a, b) => {
      if (a.fit.sideways !== b.fit.sideways) return a.fit.sideways ? b : a;
      return b.fit.raw > a.fit.raw ? b : a;
    });
  }
  return {layout: best.spec, index: best.index};
}

/** The smallest slot scale in a layout: the table's card width must be at
 *  least minCardWidth / this for every card to meet the floor. faceDown
 *  slots do not count: nothing in them is read. */
export function minSlotScale(spec: Pick<TableSpec, 'slots'>): number {
  return Math.min(
    1,
    ...spec.slots.filter(s => !s.faceDown).map(s => s.scale ?? 1),
  );
}

/** A card's index box after a clockwise right-angle rotation, as fractions of
 *  the rotated card's bounding box. */
export function rotateIndex(index: IndexGeometry, deg: number): IndexGeometry {
  const {top, left, width, height} = index;
  switch (((deg % 360) + 360) % 360) {
    case 0:
      return index;
    case 90:
      return {
        left: 1 - (top + height),
        top: left,
        width: height,
        height: width,
      };
    case 180:
      return {left: 1 - (left + width), top: 1 - (top + height), width, height};
    case 270:
      return {left: top, top: 1 - (left + width), width: height, height: width};
    default:
      throw new Error(
        `rotateIndex handles right angles only, got ${String(deg)}`,
      );
  }
}

// ---- Geometry builders. Each returns plain SlotSpecs; recipes are data. ----

type SlotExtras = Omit<SlotSpec, 'id' | 'x' | 'y'>;

/** `n` slots in a row, `step` card widths apart (1 + gap for side by side,
 *  under 1 for a horizontally overlapped row). Ids are `${prefix}${i}`. */
export function rowSlots(
  prefix: string,
  n: number,
  at: {x: number; y: number; step: number},
  extras: SlotExtras,
): SlotSpec[] {
  return Array.from({length: n}, (_, i) => ({
    id: `${prefix}${String(i)}`,
    x: at.x + i * at.step,
    y: at.y,
    ...extras,
  }));
}

/**
 * Overlapped rows, each card resting half on each of two cards in the row
 * above (a triangle of cards, or several). `rows[r]` lists the columns of row r in
 * half-card-width units, so a triangle's row r is `[R-r, R-r+2, …]`. Rows are
 * `rowStep` card widths apart. A card that another card rests on is marked
 * coveredByDesign only when `rowStep` is too small to clear any deck's index
 * — the caller says which by passing `covered`.
 */
export function overlappedRows(
  prefix: string,
  rows: ReadonlyArray<readonly number[]>,
  at: {x: number; y: number; rowStep: number},
  extras: SlotExtras & {covered?: boolean | undefined},
): SlotSpec[] {
  const {covered = false, ...rest} = extras;
  const out: SlotSpec[] = [];
  rows.forEach((cols, r) => {
    cols.forEach(c => {
      const restsOn =
        r + 1 < rows.length &&
        itemAt(rows, r + 1).some(below => Math.abs(below - c) === 1);
      out.push({
        id: `${prefix}${String(r)}-${String(c)}`,
        x: at.x + c / 2,
        y: at.y + r * at.rowStep,
        ...rest,
        ...(covered && restsOn ? {coveredByDesign: true} : {}),
      });
    });
  });
  return out;
}

/** The half-width columns of a triangle of cards with `n` rows: row r has r+1 cards. */
export function pyramidColumns(n: number): number[][] {
  return Array.from({length: n}, (_, r) =>
    Array.from({length: r + 1}, (_, i) => n - 1 - r + 2 * i),
  );
}

/**
 * `n` slots on a circle (a clock face of piles): slot i sits at angle
 * `start + i * 360/n` degrees clockwise from 12 o'clock. (cx, cy) is the
 * circle's centre and `r` its radius, card widths, both measured to the
 * centre of a card; `cardHeight` (card widths) turns centres into corners.
 */
export function radialSlots(
  prefix: string,
  n: number,
  at: {cx: number; cy: number; r: number; start?: number; cardHeight: number},
  extras: SlotExtras,
): SlotSpec[] {
  const start = at.start ?? 0;
  return Array.from({length: n}, (_, i) => {
    const a = ((start + (i * 360) / n) * Math.PI) / 180;
    const x = at.cx + at.r * Math.sin(a) - 0.5;
    const y = at.cy - at.r * Math.cos(a) - at.cardHeight / 2;
    return {id: `${prefix}${String(i)}`, x: round3(x), y: round3(y), ...extras};
  });
}

/**
 * `n` slots in reading order, `perRow` to a row (a wrapping ordered line
 * of cards played left to right). Each full row's last slot carries cue 'continues' and
 * each later row's first 'continued', so the line reads on across the wrap;
 * both cue bands are reserved (the line starts at x + CUE_WIDTH). Ids are
 * `${prefix}${i}`, i counting along the line.
 */
export function orderedLine(
  prefix: string,
  n: number,
  at: {x: number; y: number; perRow: number; step: number; rowStep: number},
  extras: SlotExtras,
): SlotSpec[] {
  if (!(at.perRow >= 1)) {
    throw new Error(
      `orderedLine perRow must be >= 1, got ${String(at.perRow)}`,
    );
  }
  const rows = Math.ceil(n / at.perRow);
  return Array.from({length: n}, (_, i) => {
    const r = Math.floor(i / at.perRow);
    const c = i % at.perRow;
    const cue =
      c === at.perRow - 1 && r < rows - 1
        ? 'continues'
        : c === 0 && r > 0
          ? 'continued'
          : undefined;
    return {
      id: `${prefix}${String(i)}`,
      x: round3(at.x + CUE_WIDTH + c * at.step),
      y: round3(at.y + r * at.rowStep),
      ...extras,
      ...(cue ? {cue} : {}),
    };
  });
}

/** Width of an orderedLine, cue bands included, card widths. */
export function orderedLineWidth(perRow: number, step: number): number {
  return 2 * CUE_WIDTH + (perRow - 1) * step + 1;
}

/**
 * `n` slots in rows of `perRow` (a group that reflows: build one per
 * variant with a different perRow). Ids are `${prefix}${i}` whatever the
 * wrap, so a game's piles are the same in every variant.
 */
export function gridSlots(
  prefix: string,
  n: number,
  at: {x: number; y: number; perRow: number; colStep: number; rowStep: number},
  extras: SlotExtras,
): SlotSpec[] {
  if (!(at.perRow >= 1)) {
    throw new Error(`gridSlots perRow must be >= 1, got ${String(at.perRow)}`);
  }
  return Array.from({length: n}, (_, i) => ({
    id: `${prefix}${String(i)}`,
    x: round3(at.x + (i % at.perRow) * at.colStep),
    y: round3(at.y + Math.floor(i / at.perRow) * at.rowStep),
    ...extras,
  }));
}

/**
 * A whole TableSpec as a scaled sub-table (one player's layout in a
 * multi-player table): its slots moved to (x, y), every length times `scale`, ids
 * `${prefix}${id}`. The result is plain slots; its footprint is
 * spec.width * scale by spec.height * scale.
 */
export function nestTable(
  prefix: string,
  spec: Omit<TableSpec, 'variants'>,
  at: {x: number; y: number; scale: number},
): SlotSpec[] {
  if (!(at.scale > 0 && at.scale <= 1)) {
    throw new Error(
      `nestTable scale must be in (0, 1], got ${String(at.scale)}`,
    );
  }
  return spec.slots.map(s => ({
    ...s,
    id: `${prefix}${s.id}`,
    x: round3(at.x + s.x * at.scale),
    y: round3(at.y + s.y * at.scale),
    scale: round3((s.scale ?? 1) * at.scale),
    ...(s.maxExtent !== undefined
      ? {maxExtent: round3(s.maxExtent * at.scale)}
      : {}),
  }));
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}
