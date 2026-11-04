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

// The meld cycle of one card during the laydown: a double-click steps the
// card through its candidate melds, then unstaged, then (in the knocker's
// discard stage) staged as THE discard, and back. Pure; the composition hook
// applies the result.

import type {PlayerMeld} from '../types.ts';
import {
  candidatesForCard,
  overlapping,
  reachable,
  serializeSignature,
  stagedLayoffSignatures,
  type Composition,
  type LaydownStage,
} from './gin_laydown_logic.ts';

/** What one step of a card's cycle does. */
export type CycleResult =
  /** The card has no candidate melds: the gesture toggles the discard. */
  | {readonly kind: 'toggle-discard'}
  /** Nothing changes (the card is in a laid meld, or no gesture applies). */
  | {readonly kind: 'none'}
  /**
   * The staged melds become `staged`; the staged discard is cleared, set to
   * the card, or kept; `flashed` when a slot was refused on the way.
   */
  | {
      readonly kind: 'stage';
      readonly staged: Set<string>;
      readonly discard: 'clear' | 'set' | 'keep';
      readonly flashed: boolean;
    };

/** Where the cycle starts: the composition and what is staged now. */
export interface CycleInput extends Composition {
  readonly stagedMeldIds: ReadonlySet<string>;
  readonly stagedDiscard: string | null;
  readonly laydownStage: LaydownStage | null;
  /** The card can be the knocker's discard at this stage. */
  readonly discardEligible: boolean;
}

const DISCARD_SLOT = 'discard';
type Slot = PlayerMeld | null | typeof DISCARD_SLOT;

/**
 * Whether the engine can lay `slot`: at the meld stage a meld must be in
 * one of the plans (8s8c8d is a meld of the hand, yet no lay-down that
 * passes holds it when 8d is needed in 6d7d8d); at the lay-off stage,
 * staging lay-off `slot` after displacing `conflicting` must leave a lay-off
 * set the engine can reach (the mirror of undoLayoff's chain rule).
 */
function slotReachable(
  c: CycleInput,
  next: ReadonlySet<string>,
  slot: PlayerMeld,
  conflicting: readonly PlayerMeld[],
): boolean {
  if (slot.locked) return true;
  if (c.laydownStage === 'melds' && slot.cards.length >= 3) {
    const signature = serializeSignature(slot.cards);
    return c.laydownPlans.some(plan => plan.signatureSet.has(signature));
  }
  if (c.laydownStage !== 'layoffs' || slot.cards.length >= 3) {
    return true;
  }
  const displaced = new Set(next);
  for (const other of conflicting) displaced.delete(other.id);
  const target = stagedLayoffSignatures(c.playerMelds, displaced);
  target.add(serializeSignature(slot.cards));
  return reachable(c.laydownPlans, target);
}

function startIndex(
  c: CycleInput,
  cardCode: string,
  candidates: readonly PlayerMeld[],
  current: PlayerMeld | undefined,
  cycleLength: number,
): number {
  if (current !== undefined) {
    return candidates.findIndex(meld => meld.id === current.id);
  }
  if (c.discardEligible && c.stagedDiscard === cardCode) return cycleLength - 1;
  return candidates.length;
}

/**
 * One step of `cardCode`'s cycle. A card belongs to one staged meld, so
 * forming a candidate displaces the staged melds it overlaps; that keeps
 * every meld a card belongs to reachable by cycling it. A laid (locked)
 * meld is never displaced: a candidate overlapping one is refused with the
 * pinned flash, and so is a meld or lay-off the engine cannot lay.
 */
export function cycleMeld(c: CycleInput, cardCode: string): CycleResult {
  const candidates = candidatesForCard(c.playerMelds, cardCode);
  if (candidates.length === 0) {
    return c.discardEligible ? {kind: 'toggle-discard'} : {kind: 'none'};
  }
  const current = candidates.find(meld => c.stagedMeldIds.has(meld.id));
  // Never cycle away from an already-laid meld.
  if (current !== undefined && c.playerLockedMeldIds.has(current.id)) {
    return {kind: 'none'};
  }
  const cycle: Slot[] = [...candidates, null];
  if (c.discardEligible) cycle.push(DISCARD_SLOT);
  const from = startIndex(c, cardCode, candidates, current, cycle.length);
  const next = new Set(c.stagedMeldIds);
  if (current !== undefined) next.delete(current.id);
  let flashed = false;
  for (let step = 1; step <= cycle.length; step += 1) {
    const slot = cycle[(from + step) % cycle.length] ?? null;
    if (slot === null) {
      // "none": the card is unstaged (and not the discard).
      const discard = c.stagedDiscard === cardCode ? 'clear' : 'keep';
      return {kind: 'stage', staged: next, discard, flashed};
    }
    if (slot === DISCARD_SLOT) {
      return {kind: 'stage', staged: next, discard: 'set', flashed};
    }
    const conflicting = overlapping(c.playerMelds, next, slot);
    if (
      conflicting.some(other => c.playerLockedMeldIds.has(other.id)) ||
      !slotReachable(c, next, slot, conflicting)
    ) {
      flashed = true;
      continue;
    }
    for (const other of conflicting) next.delete(other.id);
    next.add(slot.id);
    // A card cannot be both melded and the discard: staging a meld that
    // holds the staged discard (any of its cards) clears the discard.
    const clears =
      c.stagedDiscard !== null && slot.cards.includes(c.stagedDiscard);
    return {
      kind: 'stage',
      staged: next,
      discard: clears ? 'clear' : 'keep',
      flashed,
    };
  }
  return {kind: 'stage', staged: next, discard: 'keep', flashed};
}
