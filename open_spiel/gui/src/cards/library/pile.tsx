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

// <Pile>: a stack of cards laid out squared or fanned, with offsets from
// pileGeometry.ts so that a face-up card's rank and suit are never covered.
// When a maxExtent cannot be met without covering an index, the pile keeps
// its legible offsets and scrolls, with a visible marker — it never overlaps
// further. It is a one-slot <Table> (react/OneSlot.tsx): TableCards with
// required ids, tags for state, a policy for interaction. fan-left/fan-up,
// positional holes (null entries in a fan) and a quarter-turned base card
// (turn 90/270 at position 0 of a vertical fan) are the survey's pile shapes
//.
import type {JSX} from 'react';
import type {PileMode} from './pile_geometry.ts';
import {OneSlotTable, type OneSlotProps} from './react/one_slot.tsx';

/** <Pile items>: the pile as a one-slot <Table>: TableCards, a policy, renderOverlay and renderCard, with the same
 *  meanings as on <Table>. */
export interface PileProps extends OneSlotProps {
  mode: PileMode;
  cardWidth: number;
  /** Available px along the fan axis; the pile compresses to index visibility, then scrolls. */
  maxExtent?: number | undefined;
  /** Squared mode: depth layers drawn under the top card (0: the top card only). */
  stagger?: number | undefined;
  /** Show the card count as a badge. */
  countBadge?: boolean | undefined;
  /** Fan modes: draw only the top N cards (e.g. a draw-three waste). */
  visibleCount?: number | undefined;
  emptyLabel?: string | undefined;
}

/** A pile of cards: one slot of a <Table>. */
export function Pile(p: PileProps): JSX.Element {
  if ((p.items as unknown) === undefined || (p.items as unknown) === null) {
    throw new Error('<Pile> needs items');
  }
  if (p.maxExtent !== undefined && !(p.maxExtent > 0)) {
    throw new Error(`<Pile> maxExtent must be > 0, got ${String(p.maxExtent)}`);
  }
  const slot = {
    id: p.slotId ?? 'pile',
    pile: p.mode,
    ...(p.maxExtent !== undefined
      ? {maxExtent: Math.max(1, p.maxExtent / p.cardWidth)}
      : {}),
    ...(p.stagger !== undefined ? {stagger: p.stagger} : {}),
    ...(p.countBadge ? {countBadge: true} : {}),
    ...(p.visibleCount !== undefined ? {visibleCount: p.visibleCount} : {}),
    ...(p.emptyLabel !== undefined ? {label: p.emptyLabel} : {}),
  };
  return <OneSlotTable {...p} slot={slot} cardWidth={p.cardWidth} />;
}
