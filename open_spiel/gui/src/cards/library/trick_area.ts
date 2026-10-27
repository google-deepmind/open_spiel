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

// The trick area (pure): where a four-seat trick game lays the cards of the
// trick in play, as table slots and items for the shared <Table>/<HandTable>.
// Two layouts, a viewer's choice any trick game may offer:
//
//   - 'toward' (Toward players): each card on its own, set out from the
//     middle toward the seat that played it; nothing overlaps.
//   - 'overlap' (Overlapping): the cards overlap in the middle, as the
//     classic desktop card games drew a trick. Each card sits a little off the
//     middle toward its seat on a pinwheel (north's a touch right, east's a
//     touch down, south's a touch left, west's a touch up), and each card
//     played lies on the one before, so the play order reads from the
//     stacking. The pinwheel spreads just enough for the deck that, whoever
//     led, every card keeps an index corner in view (overlapPinwheel).
//
// Seats are compass points, the table's own frame: 'n' is the seat at the top
// of the screen. Units are card widths, as every table spec. The trick
// collect (trick_collect.ts) measures the cards where they are drawn, so it
// sweeps toward the winner from either layout.
import type {TableCard} from './core/identity.ts';
import type {HandTableRecipe} from './hand_table.tsx';
import type {FreePlacement, SlotSpec} from './table_spec.ts';
import type {IndexGeometry} from './types.ts';
import {itemAt} from './checked.ts';

function fail(why: string): never {
  throw new Error(why);
}

/** The trick-area layouts. */
export type TrickLayout = 'toward' | 'overlap';
/** Every trick layout. */
export const TRICK_LAYOUTS: readonly TrickLayout[] = ['toward', 'overlap'];
/** Each trick layout's name in the settings. */
export const TRICK_LAYOUT_LABELS: Readonly<Record<TrickLayout, string>> = {
  toward: 'Toward players',
  overlap: 'Overlapping',
};

/** A seat round the trick, the table's frame (north at the top). */
export type TrickSeat = 'n' | 'e' | 's' | 'w';
/** The four trick seats, clockwise from north. */
export const TRICK_SEATS: readonly TrickSeat[] = ['n', 'e', 's', 'w'];

/** A pinwheel, card widths from the trick's middle to a card's centre:
 *  `along` toward its seat, `aside` a quarter turn clockwise of that. */
export interface Pinwheel {
  along: number;
  aside: number;
}

/**
 * The classic desktop pinwheel, as measured on a 139 px card (west's,
 * north's and east's cards centred 60 px toward their seats and 10 px aside,
 * to a pixel or two). The pinwheel drawn is the least spread from this, the
 * same twist, at which every card keeps an index box in view whoever led
 * (overlapPinwheel): a deck with a taller index spreads a little more.
 */
export const OVERLAP_REFERENCE: Pinwheel = {along: 0.43, aside: 0.07};
/** aside / along of the reference. */
const TWIST = OVERLAP_REFERENCE.aside / OVERLAP_REFERENCE.along;
/** The pinwheel search's step, card widths. */
export const ALONG_STEP = 0.01;
/** The pinwheel search's limit (a card width: past that it is hardly an
 *  overlapping trick), card widths. */
export const ALONG_MAX = 1;
/** Overlaps under this, card widths, are rounding, not cover. */
const COVER_EPS = 1e-3;

/** Where a seat's card sits in the overlapping trick: its centre, card
 *  widths from the trick's middle (screen axes, y down). */
export function overlapOffset(
  seat: TrickSeat,
  p: Pinwheel = OVERLAP_REFERENCE,
): {x: number; y: number} {
  const {along: a, aside: b} = p;
  switch (seat) {
    case 'n':
      return {x: b, y: -a};
    case 'e':
      return {x: a, y: b};
    case 's':
      return {x: -b, y: a};
    case 'w':
      return {x: -a, y: -b};
    default:
      throw new Error(`trick area: unknown seat ${JSON.stringify(seat)}`);
  }
}

/** Play order from a lead, clockwise. */
export function playOrder(lead: TrickSeat): TrickSeat[] {
  assertSeat(lead);
  const i = TRICK_SEATS.indexOf(lead);
  return [...TRICK_SEATS.slice(i), ...TRICK_SEATS.slice(0, i)];
}

/** A card's box, card widths. */
export interface R {
  x: number;
  y: number;
  w: number;
  h: number;
}
/** Whether two boxes overlap by more than rounding. */
export function meets(a: R, b: R): boolean {
  return (
    a.x < b.x + b.w - COVER_EPS &&
    b.x < a.x + a.w - COVER_EPS &&
    a.y < b.y + b.h - COVER_EPS &&
    b.y < a.y + a.h - COVER_EPS
  );
}

/** Every card of a full trick, whoever led, keeps one of `indexes` (index
 *  boxes as fractions of the card, HandMeasure.indexes) clear of the cards
 *  played after it. */
export function pinwheelLegible(
  p: Pinwheel,
  cardHeight: number,
  indexes: readonly IndexGeometry[],
): boolean {
  if (indexes.length === 0) {
    throw new Error('trick area: a deck with no index box');
  }
  for (const lead of TRICK_SEATS) {
    const cards = playOrder(lead).map((k): R => {
      const o = overlapOffset(k, p);
      return {x: o.x - 0.5, y: o.y - cardHeight / 2, w: 1, h: cardHeight};
    });
    for (const [i, c] of cards.entries()) {
      const seen = indexes.some(ix => {
        const box = {
          x: c.x + ix.left,
          y: c.y + ix.top * cardHeight,
          w: ix.width,
          h: ix.height * cardHeight,
        };
        return !cards.slice(i + 1).some(later => meets(box, later));
      });
      if (!seen) return false;
    }
  }
  return true;
}

/** What the pinwheel is chosen from: HandMeasure's card height and index
 *  boxes. */
export interface TrickMeasure {
  cardHeight: number;
  indexes: readonly IndexGeometry[];
}

const PINWHEELS = new Map<string, Pinwheel>();

/** The overlapping trick's pinwheel for a deck: the reference's twist, at
 *  the least spread from the reference (in ALONG_STEP) at which
 *  pinwheelLegible holds. Throws if no spread up to ALONG_MAX does. */
export function overlapPinwheel(m: TrickMeasure): Pinwheel {
  if (!(m.cardHeight > 0) || !Array.isArray(m.indexes)) {
    throw new Error(`trick area: bad measure ${JSON.stringify(m)}`);
  }
  const key = JSON.stringify([m.cardHeight, m.indexes]);
  const hit = PINWHEELS.get(key);
  if (hit) return hit;
  for (
    let k = 0;
    OVERLAP_REFERENCE.along + k * ALONG_STEP <= ALONG_MAX + 1e-9;
    k++
  ) {
    const along =
      Math.round((OVERLAP_REFERENCE.along + k * ALONG_STEP) * 1000) / 1000;
    const p = {along, aside: Math.round(along * TWIST * 1000) / 1000};
    if (pinwheelLegible(p, m.cardHeight, m.indexes)) {
      PINWHEELS.set(key, p);
      return p;
    }
  }
  throw new Error(
    `trick area: no overlapping trick up to ${String(ALONG_MAX)} card widths keeps an index of every card in view (${key})`,
  );
}

/** Where the trick goes on a table, card widths. */
export interface TrickAreaGeometry {
  /** The trick's middle. */
  x: number;
  y: number;
  /** The deck's card height, card widths (HandMeasure.cardHeight). */
  cardHeight: number;
  /** 'toward': each card's centre this far from the middle toward its
   *  seat, across (west, east) and down (north, south). */
  toward: {dx: number; dy: number};
  /** 'overlap': the pinwheel (overlapPinwheel for the deck). Omitted, the
   *  reference's. */
  pinwheel?: Pinwheel | undefined;
}

/** Slot id prefix: 'toward' slots are `${prefix}-n` .. `${prefix}-w`, the
 *  'overlap' slot is `${prefix}`. */
export const TRICK_SLOT_PREFIX = 'trick';

/** `v` rounded to 1e-6, so a spec carries no float noise. */
export function round(v: number): number {
  return Math.round(v * 1e6) / 1e6;
}

function assertLayout(layout: TrickLayout): void {
  if (!TRICK_LAYOUTS.includes(layout)) {
    throw new Error(`trick area: unknown layout ${JSON.stringify(layout)}`);
  }
}

function assertSeat(seat: TrickSeat): void {
  if (!TRICK_SEATS.includes(seat)) {
    throw new Error(`trick area: unknown seat ${JSON.stringify(seat)}`);
  }
}

/** The slot a seat's trick card lies in (and a card dragged to the trick
 *  drops on). */
export function trickSlotOf(
  layout: TrickLayout,
  seat: TrickSeat,
  prefix = TRICK_SLOT_PREFIX,
): string {
  assertLayout(layout);
  assertSeat(seat);
  return layout === 'overlap' ? prefix : `${prefix}-${seat}`;
}

/** Every slot id the layout uses. */
export function trickSlotIds(
  layout: TrickLayout,
  prefix = TRICK_SLOT_PREFIX,
): string[] {
  assertLayout(layout);
  return layout === 'overlap'
    ? [prefix]
    : TRICK_SEATS.map(k => `${prefix}-${k}`);
}

/** Where a seat's card sits in the 'toward' trick: its centre, card widths
 *  from the trick's middle. */
export function towardOffset(
  seat: TrickSeat,
  toward: {dx: number; dy: number},
): {x: number; y: number} {
  switch (seat) {
    case 'n':
      return {x: 0, y: -toward.dy};
    case 'e':
      return {x: toward.dx, y: 0};
    case 's':
      return {x: 0, y: toward.dy};
    case 'w':
      return {x: -toward.dx, y: 0};
    default:
      throw new Error(`trick area: unknown seat ${JSON.stringify(seat)}`);
  }
}

function assertGeometry(g: TrickAreaGeometry): void {
  const nums = [g.x, g.y, g.cardHeight, g.toward.dx, g.toward.dy];
  if (
    !nums.every(Number.isFinite) ||
    !(g.cardHeight > 0) ||
    g.toward.dx < 0 ||
    g.toward.dy < 0
  ) {
    throw new Error(`trick area: bad geometry ${JSON.stringify(g)}`);
  }
}

/** The box the layout's four cards take, card widths, centred on the
 *  trick's middle. */
export function trickAreaSize(
  layout: TrickLayout,
  g: Pick<TrickAreaGeometry, 'cardHeight' | 'toward' | 'pinwheel'>,
): {width: number; height: number} {
  assertLayout(layout);
  if (layout === 'overlap') {
    const a = (g.pinwheel ?? OVERLAP_REFERENCE).along;
    return {width: 2 * a + 1, height: 2 * a + g.cardHeight};
  }
  return {width: 2 * g.toward.dx + 1, height: 2 * g.toward.dy + g.cardHeight};
}

/** The overlapping trick's one slot is anchored at its top-left, so its
 *  cards stay put as the trick grows (a 'free' slot's box grows with its
 *  cards); each seat's place is its card's top-left from there. */
function overlapOrigin(g: TrickAreaGeometry): {x: number; y: number} {
  const a = (g.pinwheel ?? OVERLAP_REFERENCE).along;
  return {x: g.x - a - 0.5, y: g.y - a - g.cardHeight / 2};
}

/** The trick's slots, drawn empty as nothing ('hide-empty'). */
export function trickAreaSlots(
  layout: TrickLayout,
  g: TrickAreaGeometry,
  prefix = TRICK_SLOT_PREFIX,
): SlotSpec[] {
  assertLayout(layout);
  assertGeometry(g);
  if (layout === 'overlap') {
    const o = overlapOrigin(g);
    // Overlapping is the design: a later card may cover one of an earlier
    // card's index corners (overlapPinwheel keeps another in view).
    const p = g.pinwheel ?? OVERLAP_REFERENCE;
    const places = Object.fromEntries(
      TRICK_SEATS.map(k => {
        const off = overlapOffset(k, p);
        return [k, {x: round(p.along + off.x), y: round(p.along + off.y)}];
      }),
    );
    return [
      {
        id: prefix,
        x: o.x,
        y: o.y,
        pile: 'free',
        anchor: 'start',
        places,
        tags: ['hide-empty'],
        coveredByDesign: true,
      },
    ];
  }
  return TRICK_SEATS.map((k): SlotSpec => {
    const off = towardOffset(k, g.toward);
    return {
      id: `${prefix}-${k}`,
      x: g.x + off.x,
      y: g.y + off.y,
      pile: 'squared',
      anchor: 'center',
      tags: ['hide-empty'],
    };
  });
}

/** A card of the trick in play and the seat that played it. */
export interface TrickAreaCard<T extends TableCard = TableCard> {
  seat: TrickSeat;
  item: T;
}

/**
 * The trick's items per slot, from its cards IN PLAY ORDER. 'toward': one
 * card in each seat's slot. 'overlap': every card in the one slot in play
 * order (so each lies on the one before), each at its seat's pinwheel place.
 * A seat with two cards throws. Slots with no card are absent.
 */
export function trickAreaItems<T extends TableCard>(
  layout: TrickLayout,
  cards: ReadonlyArray<TrickAreaCard<T>>,
  prefix = TRICK_SLOT_PREFIX,
): Record<string, Array<T | (T & FreePlacement)>> {
  assertLayout(layout);
  const seen = new Set<TrickSeat>();
  for (const c of cards) {
    assertSeat(c.seat);
    if (seen.has(c.seat)) {
      throw new Error(`trick area: two trick cards for seat '${c.seat}'`);
    }
    seen.add(c.seat);
  }
  if (cards.length === 0) return {};
  if (layout === 'toward') {
    return Object.fromEntries(
      cards.map(c => [`${prefix}-${c.seat}`, [c.item]]),
    );
  }
  // Each card takes its seat's place in the one slot (SlotSpec.places).
  return {[prefix]: cards.map(c => ({...c.item, at: c.seat}))};
}

/** Where a seat's card's centre is drawn, card widths in the table's frame
 *  (layout tests, and a game placing something by the trick). */
export function trickCardCentre(
  layout: TrickLayout,
  seat: TrickSeat,
  g: TrickAreaGeometry,
): {x: number; y: number} {
  assertLayout(layout);
  assertGeometry(g);
  const off =
    layout === 'overlap'
      ? overlapOffset(seat, g.pinwheel ?? OVERLAP_REFERENCE)
      : towardOffset(seat, g.toward);
  return {x: g.x + off.x, y: g.y + off.y};
}

// ---- A table laid out with its trick 'toward' players, re-laid 'overlap'. ----

/** The spec with its four 'toward' trick slots (`${prefix}-n` .. `-w`,
 *  centre-anchored, upright, squared) replaced by the overlapping trick's
 *  one slot, centred on their middle (the centroid of the four: a cross's
 *  centre, a row's middle), its pinwheel the deck's (overlapPinwheel).
 *  Everything else, regions included, is unchanged, so the table does not
 *  move when the viewer switches.
 *
 *  The overlapping trick takes only the room the four cards took: where its
 *  pinwheel does not fit inside their box (a phone's one-row trick is a card
 *  tall), that layout's one slot keeps the four cards' own places, so the
 *  items (one slot, each card at its seat) are the same in every variant and
 *  nothing the table kept clear is covered. 'toward' returns the spec. */
export function trickLayoutSpec<S extends {slots: SlotSpec[]}>(
  spec: S,
  layout: TrickLayout,
  m: TrickMeasure,
  prefix = TRICK_SLOT_PREFIX,
): S {
  assertLayout(layout);
  if (layout === 'toward') return spec;
  const ids = trickSlotIds('toward', prefix);
  const four = ids.map(id => {
    const s = spec.slots.find(x => x.id === id);
    if (!s) {
      throw new Error(
        `trick area: the table has no slot '${id}' to re-lay as an overlapping trick`,
      );
    }
    if (
      s.anchor !== 'center' ||
      (s.rotate ?? 0) !== 0 ||
      s.pile !== 'squared' ||
      (s.scale ?? 1) !== 1
    ) {
      throw new Error(
        `trick area: slot '${id}' is not an upright, centre-anchored, squared card at scale 1`,
      );
    }
    return s;
  });
  const h = m.cardHeight;
  const middle = {
    x: four.reduce((a, s) => a + s.x / 4, 0),
    y: four.reduce((a, s) => a + s.y / 4, 0),
  };
  const box = {
    left: Math.min(...four.map(s => s.x)) - 0.5,
    right: Math.max(...four.map(s => s.x)) + 0.5,
    top: Math.min(...four.map(s => s.y)) - h / 2,
    bottom: Math.max(...four.map(s => s.y)) + h / 2,
  };
  const pinwheel = overlapPinwheel(m);
  const reach = {x: pinwheel.along + 0.5, y: pinwheel.along + h / 2};
  const EPS = 1e-9;
  const fits =
    middle.x - reach.x >= box.left - EPS &&
    middle.x + reach.x <= box.right + EPS &&
    middle.y - reach.y >= box.top - EPS &&
    middle.y + reach.y <= box.bottom + EPS;
  const slot: SlotSpec = fits
    ? (trickAreaSlots(
        'overlap',
        {...middle, cardHeight: h, toward: {dx: 0, dy: 0}, pinwheel},
        prefix,
      ).at(0) ?? fail('trick area: no overlap slot'))
    : {
        id: prefix,
        x: box.left,
        y: box.top,
        pile: 'free',
        anchor: 'start',
        tags: ['hide-empty'],
        places: Object.fromEntries(
          TRICK_SEATS.map((k, i) => [
            k,
            {
              x: round(itemAt(four, i).x - 0.5 - box.left),
              y: round(itemAt(four, i).y - h / 2 - box.top),
            },
          ]),
        ),
      };
  const at = spec.slots.findIndex(x => x.id === ids[0]);
  const slots = spec.slots.filter(x => !ids.includes(x.id));
  slots.splice(Math.min(at, slots.length), 0, slot);
  return {...spec, slots};
}

/** A whole table spec (and each variant) re-laid by trickLayoutSpec. */
export function trickLayoutTable<
  T extends {
    slots: SlotSpec[];
    variants?: Array<{spec: {slots: SlotSpec[]}}> | undefined;
  },
>(
  spec: T,
  layout: TrickLayout,
  m: TrickMeasure,
  prefix = TRICK_SLOT_PREFIX,
): T {
  if (layout === 'toward') return spec;
  const top = trickLayoutSpec(spec, layout, m, prefix);
  if (!spec.variants) return top;
  return {
    ...top,
    variants: spec.variants.map(v => ({
      ...v,
      spec: trickLayoutSpec(v.spec, layout, m, prefix),
    })),
  };
}

const OVERLAP_RECIPES = new WeakMap<HandTableRecipe, HandTableRecipe>();

/** A <HandTable> recipe with its trick laid out `layout` (the same object
 *  for the same recipe, so a table re-renders without re-resolving). Its
 *  trick slots carry the default prefix (TRICK_SLOT_PREFIX). */
export function trickLayoutRecipe(
  recipe: HandTableRecipe,
  layout: TrickLayout,
): HandTableRecipe {
  assertLayout(layout);
  if (layout === 'toward') return recipe;
  let r = OVERLAP_RECIPES.get(recipe);
  if (!r) {
    r = {
      ...recipe,
      id: `${recipe.id}--trick-overlap`,
      label: `${recipe.label}, overlapping trick`,
      layout: m => trickLayoutTable(recipe.layout(m), 'overlap', m),
    };
    OVERLAP_RECIPES.set(recipe, r);
  }
  return r;
}

export * from './trick_stack_area.ts';
