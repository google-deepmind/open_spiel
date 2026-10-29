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

// <Hand>: a player's hand — flat, overlapped, or on an arc — with placements
// from handGeometry.ts, so a face-up card's rank and suit are never covered.
// Spacing, face-down spacing and selected-card lift come from the CardStyle in
// effect (fan.horizontal, fan.faceDown, selection.lift). A hand that cannot
// fit its maxExtent keeps legible spacing and scrolls, with a visible marker,
// exactly like <Pile>. It is a one-slot <Table> (react/OneSlot.tsx):
// TableCards with required ids, tags for state, a policy for interaction.
import type {JSX} from 'react';
import type {Deck} from './types.ts';
import type {CardStyle} from './style.ts';
import {deckIndex} from './index_scale.ts';
import {
  computeHandGeometry,
  type HandCardShape,
  type HandGeometry,
  type HandMode,
  type HandWrap,
} from './hand_geometry.ts';
import {OneSlotTable, type OneSlotProps} from './react/one_slot.tsx';

/** The geometry a hand slot draws, for a deck and style: <Table> sizes hand
 *  slots with it, so a slot's box and its drawing cannot disagree. Each card
 *  is its face and whether it is selected (lifted). */
export function handGeometryFor(
  deck: Deck,
  style: CardStyle,
  cards: readonly HandCardShape[],
  opts: {
    mode: HandMode;
    cardWidth: number;
    maxExtent?: number | undefined;
    spread?: number | undefined;
    lift?: number | undefined;
    arcRadius?: number | undefined;
    gap?: number | undefined;
    /** Wrapping, as on a SlotSpec (rowGap × card width); each defaults to CardStyle.hand. */
    wrap?: HandWrap | undefined;
    maxRows?: number | undefined;
    rowGap?: number | undefined;
  },
): HandGeometry {
  return computeHandGeometry({
    deck: {aspect: deck.aspect, index: deckIndex(deck, style.indexScale)},
    cardWidth: opts.cardWidth,
    faceUp: cards.map(c => c.faceUp),
    selected: cards.map(c => !!c.selected),
    mode: opts.mode,
    maxExtent: opts.maxExtent,
    spread: opts.spread ?? style.fan.horizontal,
    faceDownSpread: style.fan.faceDown,
    lift: opts.lift ?? style.selection.lift,
    arcRadius: opts.arcRadius,
    gap: opts.gap,
    wrap: opts.wrap ?? style.hand.wrap,
    maxRows: opts.maxRows ?? style.hand.maxRows,
    rowGap: (opts.rowGap ?? style.hand.rowGap) * opts.cardWidth,
  });
}

/** <Hand items>: the hand as a one-slot <Table>: TableCards, a policy, renderOverlay and renderCard, with the same
 *  meanings as on <Table>. Spacing and lift come from the CardStyle; a hand
 *  takes no positional holes (a null entry throws). */
export interface HandProps extends OneSlotProps {
  mode: HandMode;
  cardWidth: number;
  /** Available width in px; the hand compresses to index visibility, then scrolls. */
  maxExtent?: number | undefined;
  /** Arc hands: pivot distance × card height. */
  arcRadius?: number | undefined;
  /** Wrapping onto more rows when the hand cannot fit maxExtent (SlotSpec
   *  wrap, maxRows, rowGap × card width); each defaults to CardStyle.hand. */
  wrap?: HandWrap | undefined;
  maxRows?: number | undefined;
  rowGap?: number | undefined;
  emptyLabel?: string | undefined;
}

/** A player's hand: one fanned slot of a <Table>. */
export function Hand(p: HandProps): JSX.Element {
  if ((p.items as unknown) === undefined || (p.items as unknown) === null) {
    throw new Error('<Hand> needs items');
  }
  const slot = {
    id: p.slotId ?? 'hand',
    pile: p.mode,
    ...(p.maxExtent !== undefined
      ? {maxExtent: Math.max(1, p.maxExtent / p.cardWidth)}
      : {}),
    ...(p.arcRadius !== undefined ? {arcRadius: p.arcRadius} : {}),
    ...(p.wrap !== undefined ? {wrap: p.wrap} : {}),
    ...(p.maxRows !== undefined ? {maxRows: p.maxRows} : {}),
    ...(p.rowGap !== undefined ? {rowGap: p.rowGap} : {}),
    ...(p.emptyLabel !== undefined ? {label: p.emptyLabel} : {}),
  };
  return <OneSlotTable {...p} slot={slot} cardWidth={p.cardWidth} />;
}
