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

// The stacked trick (trick_area.ts's 'overlap' layout generalised): the
// cards of a trick stacked toward the seats that played them.
import type {TableCard} from './core/identity.ts';
import type {HandTableRecipe} from './hand_table.tsx';
import type {FreePlacement, SlotSpec} from './table_spec.ts';
import type {IndexGeometry} from './types.ts';
import {itemAt} from './checked.ts';
import {
  ALONG_MAX,
  ALONG_STEP,
  meets,
  OVERLAP_REFERENCE,
  round,
  TRICK_SLOT_PREFIX,
  type Pinwheel,
  type R,
  type TrickMeasure,
  type TrickSeat,
} from './trick_area.ts';

// ---- The stacked trick: two seats, or any ring of them (additive). ----
//
// True of cards in general, not of one game: wherever players put cards down
// in turn on one shared spot, each card lies on the one before it, a little
// toward whoever put it down, so the order of play reads from the stacking,
// as on a real table. The library knows only play positions and their
// directions from the middle; who led, who won and what a card means stay
// the game's (its items say whose card is whose, in the order played).
//
// The four-seat 'overlap' layout above is this stack with the compass
// directions and OVERLAP_REFERENCE (stackPinwheel of SEAT_TOWARD equals
// overlapPinwheel). A two-handed game stacks two seats, 'n' (across) and 's'
// (the near seat), from PAIR_OVERLAP_REFERENCE; a table of five or more
// stacks its ring, each card toward the play position it replaces. Nothing
// here changes a table that does not call it.

/** A unit vector from the trick's middle toward a seat (screen axes, y down). */
export interface Toward {
  x: number;
  y: number;
}

/** Each compass seat's direction from the trick's middle. */
export const SEAT_TOWARD: Readonly<Record<TrickSeat, Toward>> = {
  n: {x: 0, y: -1},
  e: {x: 1, y: 0},
  s: {x: 0, y: 1},
  w: {x: -1, y: 0},
};

/** The two-seat trick's seats: across the table, then the near seat. */
export const PAIR_SEATS: readonly TrickSeat[] = ['n', 's'];

/**
 * The two-seat stack's reference: each card a fifth of a card width or so
 * toward its seat and a tenth aside (the pinwheel's turn, so the two never
 * square up), so the second card lies on most of the first and the first
 * shows a strip and an index corner. The spread drawn is the least from this
 * at which both cards keep an index in view, whoever led (stackPinwheel).
 */
export const PAIR_OVERLAP_REFERENCE: Pinwheel = {along: 0.18, aside: 0.1};

function assertToward(t: Toward): void {
  const given: unknown = t;
  if (
    given === null ||
    typeof given !== 'object' ||
    !Number.isFinite(t.x) ||
    !Number.isFinite(t.y) ||
    Math.abs(Math.hypot(t.x, t.y) - 1) > 1e-6
  ) {
    throw new Error(
      `trick area: a direction must be a unit vector, got ${JSON.stringify(t)}`,
    );
  }
}

function assertPinwheel(p: Pinwheel): void {
  const given: unknown = p;
  if (
    given === null ||
    typeof given !== 'object' ||
    !(p.along > 0) ||
    !Number.isFinite(p.aside) ||
    p.aside < 0
  ) {
    throw new Error(`trick area: bad pinwheel ${JSON.stringify(p)}`);
  }
}

/** Where a card sits in a stacked trick: its centre, card widths from the
 *  middle, `along` toward its seat and `aside` a quarter turn clockwise of
 *  that (overlapOffset for the compass seats). */
export function stackOffset(
  toward: Toward,
  p: Pinwheel,
): {x: number; y: number} {
  assertToward(toward);
  return {
    x: p.along * toward.x - p.aside * toward.y,
    y: p.along * toward.y + p.aside * toward.x,
  };
}

/** Every card of a full stacked trick keeps one of `indexes` clear of the
 *  cards played after it, whoever led. `towards` is the seats' directions in
 *  play order round the table (at least two). */
export function stackLegible(
  towards: readonly Toward[],
  p: Pinwheel,
  cardHeight: number,
  indexes: readonly IndexGeometry[],
): boolean {
  if (towards.length < 2) {
    throw new Error(
      `trick area: a stacked trick needs two seats or more, got ${String(towards.length)}`,
    );
  }
  if (indexes.length === 0) {
    throw new Error('trick area: a deck with no index box');
  }
  if (!(cardHeight > 0)) {
    throw new Error(`trick area: bad card height ${String(cardHeight)}`);
  }
  const rects = towards.map((t): R => {
    const o = stackOffset(t, p);
    return {x: o.x - 0.5, y: o.y - cardHeight / 2, w: 1, h: cardHeight};
  });
  for (let lead = 0; lead < rects.length; lead++) {
    const cards = [...rects.slice(lead), ...rects.slice(0, lead)];
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

const STACK_PINWHEELS = new Map<string, Pinwheel | null>();
function r3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/** The reference's twist kept, in ALONG_STEP up to ALONG_MAX (overlapPinwheel's search). */
function* twistSpreads(reference: Pinwheel): Generator<Pinwheel> {
  const twist = reference.aside / reference.along;
  for (let k = 0; reference.along + k * ALONG_STEP <= ALONG_MAX + 1e-9; k++) {
    const along = r3(reference.along + k * ALONG_STEP);
    yield {along, aside: r3(along * twist)};
  }
}

/** Every spread from the reference up (along and aside each from the
 *  reference's, in ALONG_STEP up to ALONG_MAX), the least first: by the
 *  larger of the two, then the smaller aside, then the smaller along. So a
 *  deck whose index is cleared by setting the cards further aside keeps the
 *  reference's overlap along the seats' axis, and the reverse. */
function* leastSpreads(reference: Pinwheel): Generator<Pinwheel> {
  const steps = (from: number) =>
    Array.from(
      {
        length:
          Math.max(0, Math.floor((ALONG_MAX - from) / ALONG_STEP + 1e-9)) + 1,
      },
      (_, k) => r3(from + k * ALONG_STEP),
    );
  const all = steps(reference.along).flatMap(along =>
    steps(reference.aside).map(aside => ({along, aside})),
  );
  all.sort(
    (p, q) =>
      Math.max(p.along, p.aside) - Math.max(q.along, q.aside) ||
      p.aside - q.aside ||
      p.along - q.along,
  );
  yield* all;
}

/** The least spread from `reference` at which stackLegible holds; null if
 *  none up to ALONG_MAX does. Two seats search both components (leastSpreads):
 *  their two cards clear an index either across or along. Three or more keep
 *  the reference's twist first (so the compass gives overlapPinwheel), then
 *  search both. */
function findStackPinwheel(
  towards: readonly Toward[],
  m: TrickMeasure,
  reference: Pinwheel,
): Pinwheel | null {
  if (!(m.cardHeight > 0) || !Array.isArray(m.indexes)) {
    throw new Error(`trick area: bad measure ${JSON.stringify(m)}`);
  }
  assertPinwheel(reference);
  towards.forEach(assertToward);
  const key = JSON.stringify([towards, m.cardHeight, m.indexes, reference]);
  const known = STACK_PINWHEELS.get(key);
  if (known !== undefined) return known;
  const legible = (p: Pinwheel) =>
    stackLegible(towards, p, m.cardHeight, m.indexes);
  let found: Pinwheel | null = null;
  if (towards.length > 2) {
    for (const p of twistSpreads(reference)) {
      if (legible(p)) {
        found = p;
        break;
      }
    }
  }
  if (!found) {
    for (const p of leastSpreads(reference)) {
      if (legible(p)) {
        found = p;
        break;
      }
    }
  }
  STACK_PINWHEELS.set(key, found);
  return found;
}

/** The stacked trick's pinwheel for a deck: the least spread from
 *  `reference` (default: PAIR_OVERLAP_REFERENCE for two seats, else
 *  OVERLAP_REFERENCE) at which every card keeps an index in view, whoever
 *  led (findStackPinwheel says how "least" is searched). Throws if no spread
 *  up to ALONG_MAX does. */
export function stackPinwheel(
  towards: readonly Toward[],
  m: TrickMeasure,
  reference?: Pinwheel,
): Pinwheel {
  const ref =
    reference ??
    (towards.length === 2 ? PAIR_OVERLAP_REFERENCE : OVERLAP_REFERENCE);
  const p = findStackPinwheel(towards, m, ref);
  if (!p) {
    throw new Error(
      `trick area: no stacked trick up to ${String(ALONG_MAX)} card widths keeps an index of every card in view (${JSON.stringify([towards, m])})`,
    );
  }
  return p;
}

/** A stacked trick for an element-drawn board (not a <Table>): each card's
 *  top-left, card widths from the stack's top-left, in the order of
 *  `towards`, and the box all of them take (reserve it whole, so the board
 *  does not move as the trick fills). */
export function stackBox(
  towards: readonly Toward[],
  m: TrickMeasure,
  reference?: Pinwheel,
): {width: number; height: number; places: Array<{x: number; y: number}>} {
  const p = stackPinwheel(towards, m, reference);
  const centres = towards.map(t => stackOffset(t, p));
  const left = Math.min(...centres.map(c => c.x)) - 0.5;
  const top = Math.min(...centres.map(c => c.y)) - m.cardHeight / 2;
  const right = Math.max(...centres.map(c => c.x)) + 0.5;
  const bottom = Math.max(...centres.map(c => c.y)) + m.cardHeight / 2;
  return {
    width: round(right - left),
    height: round(bottom - top),
    places: centres.map(c => ({
      x: round(c.x - 0.5 - left),
      y: round(c.y - m.cardHeight / 2 - top),
    })),
  };
}

/** A play position re-laid into a stacked trick: its slot, and the
 *  direction toward the seat that plays to it (omitted: from the positions'
 *  middle toward the slot's centre, a ring's own geometry). */
export interface StackSeat {
  slot: string;
  toward?: Toward | undefined;
}

/** The two seats of a two-handed table: the slot across ('n') and the near
 *  seat's ('s'). */
export function pairStackSeats(across: string, near: string): StackSeat[] {
  if (across === near) {
    throw new Error(
      `trick area: the two trick slots must differ, got '${across}' twice`,
    );
  }
  return [
    {slot: across, toward: SEAT_TOWARD.n},
    {slot: near, toward: SEAT_TOWARD.s},
  ];
}

/** How a trick is stacked in one slot. */
export interface StackTrickOptions {
  /** The stacked slot's id (default TRICK_SLOT_PREFIX). */
  id?: string | undefined;
  /** The pinwheel searched from (stackPinwheel's default). */
  reference?: Pinwheel | undefined;
}

// The slots a stacked trick replaces, checked: two or more, each named once,
// each an upright squared card.
function stackedSlots(
  spec: {slots: SlotSpec[]},
  ids: readonly string[],
  id: string,
): SlotSpec[] {
  if (ids.length < 2) {
    throw new Error(
      `trick area: a stacked trick needs two seats or more, got ${String(ids.length)}`,
    );
  }
  if (new Set(ids).size !== ids.length) {
    throw new Error(
      `trick area: a stacked trick names a slot twice (${ids.join(', ')})`,
    );
  }
  if (!ids.includes(id) && spec.slots.some(x => x.id === id)) {
    throw new Error(`trick area: the table already has a slot '${id}'`);
  }
  return ids.map(sid => {
    const s = spec.slots.find(x => x.id === sid);
    if (!s) {
      throw new Error(`trick area: the table has no slot '${sid}' to stack`);
    }
    if (s.pile !== 'squared' || (s.rotate ?? 0) !== 0 || s.anchor === 'end') {
      throw new Error(
        `trick area: slot '${sid}' is not an upright, squared, start- or centre-anchored card`,
      );
    }
    return s;
  });
}

// Each seat's direction: its own, or from the stack's middle to its slot.
function towardsOf(
  seats: readonly StackSeat[],
  centres: ReadonlyArray<{x: number; y: number}>,
  middle: {x: number; y: number},
): Toward[] {
  return seats.map((k, i) => {
    if (k.toward) return k.toward;
    const centre = itemAt(centres, i);
    const d = {x: centre.x - middle.x, y: centre.y - middle.y};
    const len = Math.hypot(d.x, d.y);
    if (!(len > 1e-9)) {
      throw new Error(
        `trick area: slot '${k.slot}' is at the stack's middle, so it has no direction; give its toward`,
      );
    }
    return {x: d.x / len, y: d.y / len};
  });
}

// Each stacked card's place in the free slot, and the slot's origin.
function stackPlaces(
  ids: readonly string[],
  at: ReadonlyArray<{x: number; y: number}>,
  half: {x: number; y: number},
  scale: number,
): {
  origin: {x: number; y: number};
  places: Record<string, {x: number; y: number}>;
} {
  const origin = {
    x: Math.min(...at.map(c => c.x)) - half.x,
    y: Math.min(...at.map(c => c.y)) - half.y,
  };
  const places = Object.fromEntries(
    ids.map((sid, i) => [
      sid,
      {
        x: round((itemAt(at, i).x - half.x - origin.x) / scale),
        y: round((itemAt(at, i).y - half.y - origin.y) / scale),
      },
    ]),
  );
  return {origin, places};
}

/**
 * The spec with the seats' play slots (upright, squared, at one scale;
 * 'start' or 'center' anchored) replaced by ONE free slot `id` in which each
 * card lies on the one before, set off toward its seat by the deck's stack
 * pinwheel, centred on the slots' middle. Its places are named by the slots
 * replaced, so stackTrickItems' cards say only which position each took.
 * `seats` are in play order round the table.
 *
 * Like trickLayoutSpec, the stack takes only the room the slots took: where
 * it does not fit inside their box, the one slot keeps the slots' own places
 * (still painted in the order played). Everything else is unchanged.
 */
export function stackTrickSpec<S extends {slots: SlotSpec[]}>(
  spec: S,
  seats: readonly StackSeat[],
  m: TrickMeasure,
  opts: StackTrickOptions = {},
): S {
  const id = opts.id ?? TRICK_SLOT_PREFIX;
  const ids = seats.map(k => k.slot);
  const old = stackedSlots(spec, ids, id);
  const scale = itemAt(old, 0).scale ?? 1;
  if (old.some(s => (s.scale ?? 1) !== scale)) {
    throw new Error(
      `trick area: the stacked slots differ in scale (${ids.join(', ')})`,
    );
  }
  const half = {x: scale / 2, y: (scale * m.cardHeight) / 2};
  const centres = old.map(s =>
    s.anchor === 'center'
      ? {x: s.x, y: s.y}
      : {x: s.x + half.x, y: s.y + half.y},
  );
  const middle = {
    x: centres.reduce((a, c) => a + c.x, 0) / centres.length,
    y: centres.reduce((a, c) => a + c.y, 0) / centres.length,
  };
  const towards = towardsOf(seats, centres, middle);
  const box = {
    left: Math.min(...centres.map(c => c.x)) - half.x,
    right: Math.max(...centres.map(c => c.x)) + half.x,
    top: Math.min(...centres.map(c => c.y)) - half.y,
    bottom: Math.max(...centres.map(c => c.y)) + half.y,
  };
  const p = findStackPinwheel(
    towards,
    m,
    opts.reference ??
      (seats.length === 2 ? PAIR_OVERLAP_REFERENCE : OVERLAP_REFERENCE),
  );
  const EPS = 1e-9;
  const stacked =
    p &&
    towards.map(t => {
      const o = stackOffset(t, p);
      return {x: middle.x + scale * o.x, y: middle.y + scale * o.y};
    });
  const fits =
    !!stacked &&
    stacked.every(
      c =>
        c.x - half.x >= box.left - EPS &&
        c.x + half.x <= box.right + EPS &&
        c.y - half.y >= box.top - EPS &&
        c.y + half.y <= box.bottom + EPS,
    );
  const at = fits ? stacked : centres;
  const {origin, places} = stackPlaces(ids, at, half, scale);
  const tags = [...new Set(old.flatMap(s => s.tags ?? []))];
  const slot: SlotSpec = {
    id,
    x: round(origin.x),
    y: round(origin.y),
    pile: 'free',
    anchor: 'start',
    places,
    ...(scale !== 1 ? {scale} : {}),
    ...(tags.length ? {tags} : {}),
    // Overlapping is the design: a later card covers part of an earlier
    // one (the stack pinwheel keeps an index of each in view).
    ...(fits ? {coveredByDesign: true} : {}),
  };
  const first = spec.slots.findIndex(x => ids.includes(x.id));
  const slots = spec.slots.filter(x => !ids.includes(x.id));
  slots.splice(Math.min(first, slots.length), 0, slot);
  return {...spec, slots};
}

type StackSeatsOf =
  readonly StackSeat[] | ((spec: {slots: SlotSpec[]}) => readonly StackSeat[]);

/** A whole table spec (and each variant) re-laid by stackTrickSpec. */
export function stackTrickTable<
  T extends {
    slots: SlotSpec[];
    variants?: Array<{spec: {slots: SlotSpec[]}}> | undefined;
  },
>(
  spec: T,
  seats: StackSeatsOf,
  m: TrickMeasure,
  opts: StackTrickOptions = {},
): T {
  const of = (s: {slots: SlotSpec[]}) =>
    typeof seats === 'function' ? seats(s) : seats;
  const top = stackTrickSpec(spec, of(spec), m, opts);
  if (!spec.variants) return top;
  return {
    ...top,
    variants: spec.variants.map(v => ({
      ...v,
      spec: stackTrickSpec(v.spec, of(v.spec), m, opts),
    })),
  };
}

/** A <HandTable> recipe with its trick stacked (stackTrickTable). Make it
 *  once (module scope or a cache), so a table re-renders without
 *  re-resolving. */
export function stackTrickRecipe(
  recipe: HandTableRecipe,
  seats: StackSeatsOf,
  opts: StackTrickOptions = {},
): HandTableRecipe {
  return {
    ...recipe,
    id: `${recipe.id}--trick-stack`,
    label: `${recipe.label}, stacked trick`,
    layout: m => stackTrickTable(recipe.layout(m), seats, m, opts),
  };
}

/**
 * The stacked trick's items, from its cards IN PLAY ORDER: every card in the
 * one slot `id`, at the place of the play position it took, so each lies on
 * the one before. A position with two cards throws; no cards, no slot.
 */
export function stackTrickItems<T extends TableCard>(
  cards: ReadonlyArray<{slot: string; item: T}>,
  id = TRICK_SLOT_PREFIX,
): Record<string, Array<T & FreePlacement>> {
  const seen = new Set<string>();
  for (const c of cards) {
    if (typeof c.slot !== 'string' || !c.slot) {
      throw new Error(
        `trick area: a stacked card with no slot (${JSON.stringify(c.item.id)})`,
      );
    }
    if (seen.has(c.slot)) {
      throw new Error(`trick area: two trick cards for '${c.slot}'`);
    }
    seen.add(c.slot);
  }
  if (cards.length === 0) return {};
  return {[id]: cards.map(c => ({...c.item, at: c.slot}))};
}
