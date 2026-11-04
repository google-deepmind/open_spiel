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

// The per-seat fields of the gin view, read: which hands are open, each
// seat's laid melds, its meld analysis, and the lay-down in progress.

import type {GinState} from './gin_session.ts';
import {
  asObject,
  normalisedCodes,
  normalisedMelds,
  seatOf,
  serializeMeldSignature,
} from './gin_derived_parse.ts';
import type {LaydownPlan, LaydownStep, MeldType} from './types.ts';

/** The lay-down sub-stage: the knocker's discard, lay-offs, or melds. */
export type LaydownStage = 'discard' | 'layoffs' | 'melds';

/** The seats whose hands the view lays open. */
export function visibleHandsOf(state: GinState | undefined): Set<number> {
  const visible = new Set<number>();
  const raw = asObject(state?.ginVisibleHands);
  if (raw === null) return visible;
  for (const [key, flag] of Object.entries(raw)) {
    const seat = seatOf(key);
    if (flag === true && seat !== null) visible.add(seat);
  }
  return visible;
}

/** Appends `melds`, normalised, to `seat`'s laid melds in `bySeat`. */
export function appendLayedMelds(
  bySeat: Map<number, string[][]>,
  seat: number,
  melds: unknown,
): void {
  const normalized = normalisedMelds(melds);
  if (normalized.length === 0) return;
  bySeat.set(seat, (bySeat.get(seat) ?? []).concat(normalized));
}

/** Each seat's laid melds, as the view's ginLayedMelds gives them. */
export function layedMeldsOf(
  state: GinState | undefined,
): Map<number, string[][]> {
  const bySeat = new Map<number, string[][]>();
  const raw = asObject(state?.ginLayedMelds);
  if (raw === null) return bySeat;
  for (const [key, melds] of Object.entries(raw)) {
    const seat = seatOf(key);
    if (seat !== null) appendLayedMelds(bySeat, seat, melds);
  }
  return bySeat;
}

/** A meld's kind: one suit is a run, else a set. */
export function meldTypeOf(meld: readonly string[]): MeldType {
  return new Set(meld.map(card => card.slice(-1))).size === 1 ? 'run' : 'set';
}

/** Each card of each seat's laid melds, by the kind of meld it is in. */
export function meldCardTypes(
  bySeat: ReadonlyMap<number, string[][]>,
): Map<number, Map<string, MeldType>> {
  const out = new Map<number, Map<string, MeldType>>();
  bySeat.forEach((melds, seat) => {
    const types = new Map<string, MeldType>();
    for (const meld of melds) {
      const type = meldTypeOf(meld);
      for (const card of meld) types.set(card, type);
    }
    if (types.size > 0) out.set(seat, types);
  });
  return out;
}

/** Every meld in each visible hand (the view's ginAnalysisMeldsBySeat). */
export function analysisMeldsOf(
  state: GinState | undefined,
): Map<number, string[][]> {
  const out = new Map<number, string[][]>();
  const raw = asObject(state?.ginAnalysisMeldsBySeat);
  if (raw === null) return out;
  for (const [key, melds] of Object.entries(raw)) {
    const seat = seatOf(key);
    const normalized = normalisedMelds(melds);
    if (seat !== null && normalized.length > 0) out.set(seat, normalized);
  }
  return out;
}

function wholeIndex(value: unknown): number | null {
  const n =
    typeof value === 'string'
      ? Number(value)
      : typeof value === 'number'
        ? value
        : NaN;
  if (!Number.isFinite(n)) return null;
  const index = Math.trunc(n);
  return index >= 0 ? index : null;
}

/** Which of each seat's melds make its best arrangement, by index. */
export function analysisBestIndicesOf(
  state: GinState | undefined,
): Map<number, number[]> {
  const out = new Map<number, number[]>();
  const raw = asObject(state?.ginAnalysisBestIndicesBySeat);
  if (raw === null) return out;
  for (const [key, indices] of Object.entries(raw)) {
    const seat = seatOf(key);
    if (seat === null || !Array.isArray(indices)) continue;
    const cleaned = indices
      .map(wholeIndex)
      .filter((value): value is number => value !== null);
    if (cleaned.length > 0) out.set(seat, cleaned);
  }
  return out;
}

/** Each seat's best arrangement, as melds. */
export function analysisBestMeldsOf(
  melds: ReadonlyMap<number, string[][]>,
  best: ReadonlyMap<number, number[]>,
): Map<number, string[][]> {
  const out = new Map<number, string[][]>();
  melds.forEach((seatMelds, seat) => {
    const chosen = (best.get(seat) ?? [])
      .map(index => seatMelds[index])
      .filter(
        (meld): meld is string[] => meld !== undefined && meld.length > 0,
      );
    if (chosen.length > 0) out.set(seat, chosen);
  });
  return out;
}

function stepOf(raw: unknown): LaydownStep | null {
  const step = asObject(raw);
  const cards = Array.isArray(step?.['cards'])
    ? normalisedCodes(step['cards'])
    : [];
  const action = step?.['action'];
  // Melds are 3+ cards; single-card layoffs are 1 card. Only empty or
  // action-less steps are dropped, so the defender's lay-offs survive.
  if (cards.length === 0 || typeof action !== 'number') return null;
  return {cards, action, signature: serializeMeldSignature(cards)};
}

/** The lay-down plans the view offers the seat laying down. */
export function laydownPlansOf(value: unknown): LaydownPlan[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((plan: unknown): LaydownPlan | null => {
      if (!Array.isArray(plan)) return null;
      const sequence = plan
        .map(stepOf)
        .filter((step): step is LaydownStep => step !== null);
      if (sequence.length === 0) return null;
      return {
        sequence,
        signatureSet: new Set(sequence.map(step => step.signature)),
      };
    })
    .filter((plan): plan is LaydownPlan => plan !== null);
}

/** The lay-down in progress: its seat, pass action and sub-stage. */
export interface Laydown {
  readonly laydownSeat: number | null;
  readonly laydownPassAction: number | null;
  readonly laydownStage: LaydownStage | null;
  readonly laydownPlans: LaydownPlan[];
}

/** The lay-down fields of the view. */
export function laydownOf(state: GinState | undefined): Laydown {
  const stage = state?.ginLaydownStage;
  return {
    laydownSeat:
      typeof state?.ginLaydownSeat === 'number' ? state.ginLaydownSeat : null,
    laydownPassAction:
      typeof state?.ginLaydownPassAction === 'number'
        ? state.ginLaydownPassAction
        : null,
    laydownStage:
      stage === 'discard' || stage === 'layoffs' || stage === 'melds'
        ? stage
        : null,
    laydownPlans: laydownPlansOf(state?.ginLaydownPlans),
  };
}
