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

// The pure legibility check for a whole table: place every card of a
// TableSpec as <Table> and <Pile> draw them, so findCoveredIndexes can run on
// a spec and a deal without a browser. Each game's own layout check
// uses this one function, so every table is held to one rule.
import type {SlotSpec, TableSpec} from './table_spec.ts';
import {itemAt} from './checked.ts';
import {CAPTION_HEIGHT, CUE_WIDTH} from './table_spec.ts';
import type {Deck} from './types.ts';
import {isHandMode} from './hand_geometry.ts';
import {
  computePileGeometry,
  findCoveredIndexes,
  placePile,
  type PlacedCard,
} from './pile_geometry.ts';
import {sceneInternals, type Scene} from './core/scene.ts';
import {isToken} from './core/identity.ts';

/** A card placed for the table check, and whether its slot is covered by
 *  design. */
export type TablePlacement = PlacedCard & {excusable: boolean};

// One pile slot's cards, placed on the table.
function placeSlot(
  s: SlotSpec,
  piles: Readonly<Record<string, ReadonlyArray<{faceUp: boolean}>>>,
  deck: Deck,
  cardWidth: number,
  placed: TablePlacement[],
): void {
  // Pile tables only; hand slots are measured with handMeasure instead.
  if (isHandMode(s.pile)) {
    throw new Error(`placeTable: '${s.id}' is a hand slot (${s.pile})`);
  }
  if (s.pile === 'free') {
    throw new Error(
      `placeTable: '${s.id}' is a free slot; check it with placeScene`,
    );
  }
  // A block moves with what it draws (SlotSpec.block), which only a Scene knows.
  if (s.block !== undefined) {
    throw new Error(
      `placeTable: '${s.id}' is in block '${s.block.id}'; check it with placeScene`,
    );
  }
  const mode = s.pile;
  const cards = piles[s.id] ?? [];
  if (cards.length === 0) return;
  let drawn = cards;
  if (mode === 'squared') {
    drawn = cards
      .slice(-1 - (s.stagger ?? 0))
      .map((c, i, a) => (i < a.length - 1 ? {...c, faceUp: false} : c));
  } else if (s.visibleCount !== undefined) {
    drawn = cards.slice(-s.visibleCount);
  }
  const faceUp = drawn.map(c => c.faceUp);
  const cw = cardWidth * (s.scale ?? 1);
  const g = computePileGeometry({
    deck,
    cardWidth: cw,
    faceUp,
    mode,
    stagger: s.stagger,
    maxExtent: s.maxExtent !== undefined ? s.maxExtent * cardWidth : undefined,
  });
  // A pile that cannot fit keeps legible offsets and scrolls inside its
  // maxExtent: what lies past the frame is clipped, reachable by scrolling.
  const frame =
    s.maxExtent !== undefined && !g.fits ? s.maxExtent * cardWidth : Infinity;
  for (const p of placePile(g, faceUp, deck.index)) {
    const far =
      mode === 'fan-right' ? p.rect.x + p.rect.width : p.rect.y + p.rect.height;
    if (far > frame) continue;
    placed.push({
      ...p,
      rect: {
        ...p.rect,
        x: p.rect.x + s.x * cardWidth,
        y: p.rect.y + s.y * cardWidth,
      },
      excusable: !!s.coveredByDesign,
    });
  }
}

// Captions and cues paint after the cards they could land on, as <Table>
// draws them: model them as face-down cards at the end.
function placeBands(
  spec: TableSpec,
  deck: Deck,
  cardWidth: number,
  placed: TablePlacement[],
): void {
  const zero = {top: 0, left: 0, width: 0, height: 0};
  for (const s of spec.slots) {
    const cw = cardWidth * (s.scale ?? 1);
    const band = CAPTION_HEIGHT * cardWidth;
    const cue = CUE_WIDTH * cardWidth;
    const ch = cw / deck.aspect;
    if (s.caption !== undefined) {
      placed.push({
        rect: {
          x: s.x * cardWidth,
          y: s.y * cardWidth - band,
          width: cw,
          height: band,
        },
        faceUp: false,
        index: zero,
        excusable: false,
      });
    }
    if (s.cue) {
      const x =
        s.cue === 'continues' ? s.x * cardWidth + cw : s.x * cardWidth - cue;
      placed.push({
        rect: {
          x,
          y: s.y * cardWidth + ch / 2 - cue / 2,
          width: cue,
          height: cue,
        },
        faceUp: false,
        index: zero,
        excusable: false,
      });
    }
  }
}

/** Place a table's cards as <Table> and <Pile> draw them, for the pure check.
 *  Each pile lists its cards' faces, bottom first. */
export function placeTable(
  spec: TableSpec,
  piles: Readonly<Record<string, ReadonlyArray<{faceUp: boolean}>>>,
  deck: Deck,
  cardWidth: number,
): TablePlacement[] {
  const placed: TablePlacement[] = [];
  for (const s of spec.slots) placeSlot(s, piles, deck, cardWidth, placed);
  placeBands(spec, deck, cardWidth, placed);
  return placed;
}

/**
 * Place a computed Scene's cards for the pure check: what <Table> draws, including game-tag lift, turn,
 * hand and free slots (a wrapped hand's every row), captions, cues, and every
 * region that occludes. Tokens are not cards and are left out; library tags
 * never reach a Scene. Cards a pile or hand has scrolled past its maxExtent
 * are out of view, reachable by scrolling, and left out as placeTable leaves
 * them out.
 */
export function placeScene(
  scene: Scene,
): Array<PlacedCard & {excusable: boolean}> {
  return scenePlacements(scene).placed.map(p => ({
    rect: p.rect,
    ...(p.rotate !== undefined ? {rotate: p.rotate} : {}),
    faceUp: p.faceUp,
    index: p.index,
    excusable: p.excusable,
  }));
}

type ScenePlaced = PlacedCard & {excusable: boolean; id: string | null};

// Captions and cues, as covering face-down cards.
function placeSceneBands(
  slots: ReturnType<typeof sceneInternals>['slots'],
  cardWidth: number,
  placed: ScenePlaced[],
): void {
  const zero = {top: 0, left: 0, width: 0, height: 0};
  for (const l of slots) {
    const s = l.spec;
    const band = CAPTION_HEIGHT * cardWidth;
    const cue = CUE_WIDTH * cardWidth;
    if (s.caption !== undefined) {
      placed.push({
        id: null,
        rect: {
          x: s.x * cardWidth,
          y: s.y * cardWidth - band,
          width: l.cw,
          height: band,
        },
        faceUp: false,
        index: zero,
        excusable: false,
      });
    }
    if (s.cue) {
      const x =
        s.cue === 'continues' ? s.x * cardWidth + l.cw : s.x * cardWidth - cue;
      placed.push({
        id: null,
        rect: {
          x,
          y: s.y * cardWidth + l.ch / 2 - cue / 2,
          width: cue,
          height: cue,
        },
        faceUp: false,
        index: zero,
        excusable: false,
      });
    }
  }
}

function scenePlacements(scene: Scene): {
  placed: ScenePlaced[];
  scrolledOut: string[];
} {
  const {slots} = sceneInternals(scene);
  const cardWidth = scene.cardWidth;
  const placed: ScenePlaced[] = [];
  const scrolledOut: string[] = [];
  const zero = {top: 0, left: 0, width: 0, height: 0};
  for (const l of slots) {
    const frame =
      l.overflow && l.spec.maxExtent !== undefined
        ? l.spec.maxExtent * cardWidth
        : Infinity;
    const horizontal =
      l.spec.pile === 'fan-right' ||
      l.spec.pile === 'fan-left' ||
      l.kind === 'hand';
    for (const e of l.entries) {
      if (isToken(e.placement.item)) continue;
      const far = horizontal
        ? e.local.x + e.local.width
        : e.local.y + e.local.height;
      if (far > frame + 0.5) {
        scrolledOut.push(e.placement.id);
        continue;
      }
      const b = e.placement.box;
      placed.push({
        id: e.placement.id,
        rect: {x: b.x, y: b.y, width: b.width, height: b.height},
        ...(b.rotate ? {rotate: b.rotate} : {}),
        faceUp: e.index !== null,
        index: e.index ?? zero,
        excusable: !!l.spec.coveredByDesign,
      });
    }
  }
  placeSceneBands(slots, cardWidth, placed);
  for (const r of scene.regions) {
    if (!r.occludes) continue;
    const b = r.box;
    placed.push({
      id: null,
      rect: {x: b.x, y: b.y, width: b.width, height: b.height},
      ...(b.rotate ? {rotate: b.rotate} : {}),
      faceUp: false,
      index: zero,
      excusable: false,
    });
  }
  return {placed, scrolledOut};
}

/** What sceneLegibility found; every list is card ids (empty: legible). */
export interface SceneLegibilityReport {
  /** Face-up cards whose index another card, a caption, a cue or an
   *  occluding region covers (coveredByDesign slots excused). */
  covered: string[];
  /** Cards a pile or hand has scrolled out of view (past its maxExtent). */
  scrolledOut: string[];
  /** Cards drawn narrower than the legibility floor (minCardWidth). */
  belowFloor: string[];
  /** Slots that scroll (a pile or hand past its maxExtent). */
  scrolling: string[];
  /** The table itself is below its minimum card width and scrolls. */
  tableOverflow: boolean;
}

/**
 * The whole legibility rule on a computed Scene, by card: no face-up index
 * covered, no card scrolled out of view, none narrower than the legibility
 * floor, and which slots (and whether the table) scroll. A wrapped hand is
 * checked on every row. Pure: games' layout checks use it.
 */
export function sceneLegibility(
  scene: Scene,
  o: {minCardWidth: number},
): SceneLegibilityReport {
  if (!(o.minCardWidth > 0)) {
    throw new Error(
      `sceneLegibility minCardWidth must be > 0, got ${String(o.minCardWidth)}`,
    );
  }
  const internals = sceneInternals(scene);
  const {placed, scrolledOut} = scenePlacements(scene);
  const covered = findCoveredIndexes(placed)
    .map(i => itemAt(placed, i))
    .filter(p => !p.excusable)
    .map(p => p.id)
    .filter(id => id !== null);
  const belowFloor: string[] = [];
  for (const l of internals.slots) {
    for (const e of l.entries) {
      // A faceDown slot's backs are never read: no floor holds them.
      if (
        !isToken(e.placement.item) &&
        !l.spec.faceDown &&
        l.cw < o.minCardWidth - 0.5
      ) {
        belowFloor.push(e.placement.id);
      }
    }
  }
  return {
    covered,
    scrolledOut,
    belowFloor,
    scrolling: internals.slots.filter(l => l.overflow).map(l => l.spec.id),
    tableOverflow: internals.overflow,
  };
}
