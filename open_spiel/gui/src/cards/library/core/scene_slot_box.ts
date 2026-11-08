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

// The box a slot draws on the table: its pile or hand at full extent, where
// its anchor puts it, and how it turns.
import type {CardStyle} from '../style.ts';
import type {SlotSpec} from '../table_spec.ts';
import {computePileGeometry, type PileMode} from '../pile_geometry.ts';
import {
  computeHandGeometry,
  isHandMode,
  type HandGeometry,
  type HandMode,
  type HandWrap,
} from '../hand_geometry.ts';
import {deckIndex} from '../index_scale.ts';
import {fanTurns} from './scene_geometry.ts';
import {definedFields} from './defined.ts';
import {
  OVERFLOW_MARKER_PX,
  VERTICAL,
  type SceneEnv,
  type SlotBox,
  type SlotCardInfo,
} from './scene_types.ts';

interface Size {
  width: number;
  height: number;
}

// A hand slot's drawn size (an overflowing one clipped to maxExtent plus its
// scroll marker).
function handSize(
  s: SlotSpec & {pile: HandMode},
  cards: ReadonlyArray<SlotCardInfo | null>,
  deck: SceneEnv['deck'],
  style: CardStyle,
  cw: number,
  maxExtent: number | undefined,
  handLift: number,
): Size {
  const known = cards.filter((c): c is SlotCardInfo => c !== null);
  if (known.length !== cards.length) {
    throw new Error(
      `slot '${s.id}' is a '${s.pile}' hand; only fan slots take positional holes (null)`,
    );
  }
  const g = handGeometry(
    deck,
    style,
    toCapacity(s, known),
    s.pile,
    cw,
    maxExtent,
    s.arcRadius,
    handLift,
    handWrapOf(s, style, cw),
  );
  return {
    width:
      g.fits || maxExtent === undefined
        ? g.width
        : Math.min(g.width, maxExtent),
    height: g.height + (g.fits ? 0 : OVERFLOW_MARKER_PX),
  };
}

// A pile slot's drawn size, from the cards it shows.
function pileSize(
  s: SlotSpec,
  cards: ReadonlyArray<SlotCardInfo | null>,
  deck: SceneEnv['deck'],
  style: CardStyle,
  cw: number,
  maxExtent: number | undefined,
): Size {
  const mode = s.pile as PileMode;
  const shown =
    mode !== 'squared' && s.visibleCount !== undefined
      ? cards.slice(-s.visibleCount)
      : cards;
  const g = computePileGeometry({
    deck,
    cardWidth: cw,
    faceUp: shown.map(c => (c ? c.faceUp : null)),
    // A vertical fan's quarter-turned position 0 is its base, which
    // the box has always measured upright.
    ...definedFields({
      turns:
        mode === 'squared'
          ? undefined
          : fanTurns(
              shown.map(c => c ?? {}),
              VERTICAL.includes(mode) &&
                shown[0] === cards[0] &&
                (cards[0]?.turn === 90 || cards[0]?.turn === 270),
            ),
      maxExtent,
      stagger: s.stagger,
    }),
    mode,
    spread: mode === 'fan-right' ? style.fan.horizontal : style.fan.faceUp,
    faceDownSpread: style.fan.faceDown,
    squaredStagger: style.squaredStagger,
    indexScale: style.indexScale,
  });
  const along =
    (maxExtent !== undefined ? Math.min(g.extent, maxExtent) : g.extent) +
    (g.fits ? 0 : OVERFLOW_MARKER_PX);
  if (mode === 'fan-right') return {width: along, height: g.crossExtent};
  if (mode === 'fan-down') return {width: g.crossExtent, height: along};
  return {width: g.crossExtent, height: g.extent};
}

/**
 * The box a slot draws on the table: its pile or hand at full extent (an
 * overflowing one clipped to maxExtent plus its scroll marker), where the
 * anchor puts it, and how it turns. The same computation <Table> has always
 * used, so slot positions do not move.
 */
export function computeSlotBox(
  s: SlotSpec,
  cards: ReadonlyArray<SlotCardInfo | null>,
  deck: SceneEnv['deck'],
  style: CardStyle,
  cardWidth: number,
  free: Size | null,
  handLift: number,
): SlotBox {
  const maxExtent =
    s.maxExtent !== undefined ? s.maxExtent * cardWidth : undefined;
  const scale = s.scale ?? 1;
  const cw = cardWidth * scale;
  const cardHeight = (cardWidth / deck.aspect) * scale;
  const pile = s.pile;
  const hand = isHandMode(pile);
  let size: Size = {width: cw, height: cardHeight};
  if (free) {
    size = free;
  } else if (isHandMode(pile)) {
    size = handSize({...s, pile}, cards, deck, style, cw, maxExtent, handLift);
  } else if (cards.length > 0) {
    size = pileSize(s, cards, deck, style, cw, maxExtent);
  }
  const {width, height} = size;
  const center = s.anchor === 'center';
  // Hands and centred slots turn about their middle; other piles about their
  // first card's centre, so a pile slot's centre stays where the spec put it.
  // (An end-anchored slot is never turned: validateTableSpec.)
  const origin =
    hand || center
      ? {x: width / 2, y: height / 2}
      : {x: cw / 2, y: cardHeight / 2};
  return {
    left:
      s.x * cardWidth - (center ? width / 2 : s.anchor === 'end' ? width : 0),
    top: s.y * cardWidth - (center ? height / 2 : 0),
    width,
    height,
    rotate: s.rotate ?? 0,
    origin,
  };
}

/**
 * The box a slot draws on the table (the pre-Scene helper, kept for layout
 * specs that check slot overlap without a browser): each card is a face and
 * an optional selection (null: a positional hole in a fan), the selection
 * lift is the style's.
 */
export function slotBox(
  s: SlotSpec,
  cards: ReadonlyArray<{faceUp: boolean; selected?: boolean} | null>,
  deck: SceneEnv['deck'],
  style: CardStyle,
  cardWidth: number,
): SlotBox {
  if (s.pile === 'free') {
    throw new Error(
      `slotBox: '${s.id}' is a free slot; use layoutScene(...).slotBox`,
    );
  }
  return computeSlotBox(
    s,
    cards.map(c =>
      c ? {faceUp: c.faceUp, lifted: c.selected === true} : null,
    ),
    deck,
    style,
    cardWidth,
    null,
    style.selection.lift,
  );
}

/** A hand's cards padded to the slot's capacity (SlotSpec.capacity) with
 *  unlifted stand-ins, so the geometry of the cards it holds is the
 *  geometry of the first positions of a full hand: nothing moves as it
 *  fills. The stand-ins are measured, never drawn (only the first
 *  cards.length placements are used). */
export function toCapacity(
  s: SlotSpec,
  cards: readonly SlotCardInfo[],
): readonly SlotCardInfo[] {
  const cap = s.capacity;
  const last = cards[cards.length - 1];
  if (cap === undefined || last === undefined || cards.length >= cap) {
    return cards;
  }
  // Stand-ins face the way the last card does: a face-down hand is
  // reserved at face-down spacing, a face-up one at face-up spacing.
  const faceUp = last.faceUp;
  return [
    ...cards,
    ...Array.from({length: cap - cards.length}, (): SlotCardInfo => ({
      faceUp,
      lifted: false,
    })),
  ];
}

/** A hand slot's wrapping: the slot's wrap, maxRows and rowGap, else the
 *  style's (CardStyle.hand); rowGap in px of the slot's cards. */
export function handWrapOf(
  s: SlotSpec,
  style: CardStyle,
  cw: number,
): {wrap: HandWrap; maxRows: number; rowGap: number} {
  const wrap = s.wrap ?? style.hand.wrap;
  if (wrap === 'never' && (s.maxRows !== undefined || s.rowGap !== undefined)) {
    throw new Error(
      `slot '${s.id}' sets maxRows/rowGap but does not wrap (CardStyle.hand.wrap is 'never'): give the slot wrap: 'auto'`,
    );
  }
  return {
    wrap,
    maxRows: s.maxRows ?? style.hand.maxRows,
    rowGap: (s.rowGap ?? style.hand.rowGap) * cw,
  };
}

/** A hand's geometry from the cards a slot holds. */
export function handGeometry(
  deck: SceneEnv['deck'],
  style: CardStyle,
  cards: readonly SlotCardInfo[],
  mode: HandMode,
  cardWidth: number,
  maxExtent: number | undefined,
  arcRadius: number | undefined,
  lift: number,
  wrap: {wrap: HandWrap; maxRows: number; rowGap: number},
): HandGeometry {
  return computeHandGeometry({
    // The index is already at the style's scale; the twin (if the deck
    // draws one) is its mirror, which a turned card may keep visible.
    deck: {
      aspect: deck.aspect,
      index: deckIndex(deck, style.indexScale),
      ...(deck.rotatedTwin === undefined
        ? {}
        : {rotatedTwin: deck.rotatedTwin}),
      ...(deck.id === undefined ? {} : {id: deck.id}),
    },
    cardWidth,
    faceUp: cards.map(c => c.faceUp),
    turns: cards.map(c => c.turn ?? 0),
    selected: cards.map(c => c.lifted),
    mode,
    ...definedFields({maxExtent, arcRadius}),
    spread: style.fan.horizontal,
    faceDownSpread: style.fan.faceDown,
    lift,
    ...wrap,
  });
}
