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

// The viewer's melds, read from the view: every candidate meld of its hand
// (the analysis), its laid melds (locked), and in the defender's lay-off
// stage its legal lay-offs, each with a stable id; which of them make the
// best arrangement; and which cards each could take.

import {
  asNumberArray,
  normaliseCardCode,
  normalisedCodes,
  serializeMeldSignature,
} from './gin_derived_parse.ts';
import {appendLayedMelds, meldTypeOf} from './gin_derived_seats.ts';
import type {GinStructRecord, LaydownPlan, PlayerMeld} from './types.ts';

/** The viewer's melds and what they cover. */
export interface PlayerMelds {
  playerMelds: PlayerMeld[];
  playerMeldOptions: Map<string, number[]>;
  playerMeldCards: Set<string>;
  playerLockedMeldIds: Set<string>;
  playerBestMeldIds: string[];
}

/** What the viewer's melds are read from. */
export interface MeldSources {
  readonly struct: GinStructRecord | null;
  readonly viewer: number;
  readonly analysisMelds: ReadonlyMap<number, string[][]>;
  readonly analysisBest: ReadonlyMap<number, number[]>;
  /** Each seat's laid melds; the struct's are appended to it here. */
  readonly layedMelds: Map<number, string[][]>;
  readonly laydownSeat: number | null;
  readonly laydownPlans: readonly LaydownPlan[];
}

/** Melds gathered once each by signature, in the order they were added. */
class MeldList {
  readonly melds: Array<{cards: string[]; locked: boolean}> = [];
  private readonly seen = new Set<string>();

  add(source: unknown, locked: boolean): boolean {
    if (!Array.isArray(source)) return false;
    let added = false;
    for (const meld of source) {
      if (!Array.isArray(meld)) continue;
      const cards = normalisedCodes(meld);
      if (cards.length === 0) continue;
      const signature = serializeMeldSignature(cards);
      if (this.seen.has(signature)) continue;
      this.seen.add(signature);
      this.melds.push({cards, locked});
      added = true;
    }
    return added;
  }
}

/**
 * In the defender's lay-off sub-stage, its legal single-card lay-offs (the
 * 1-2 card plan steps) join the melds, so that they are highlighted and
 * submitted through the same machinery. Only for the seat laying down.
 */
function addLayoffs(src: MeldSources, list: MeldList): Set<string> {
  const signatures = new Set<string>();
  if (src.laydownSeat === null || src.laydownSeat !== src.viewer) {
    return signatures;
  }
  for (const plan of src.laydownPlans) {
    for (const step of plan.sequence) {
      if (step.cards.length >= 3 || signatures.has(step.signature)) continue;
      signatures.add(step.signature);
      list.add([step.cards], false);
    }
  }
  return signatures;
}

function gather(src: MeldSources): {list: MeldList; layoffs: Set<string>} {
  const list = new MeldList();
  const struct = src.struct;
  if (!list.add(src.analysisMelds.get(src.viewer), false)) {
    list.add(struct?.['analysisMelds'], false);
  }
  list.add(struct?.['melds'], false);
  const layedStruct = struct?.['layed_melds'];
  if (Array.isArray(layedStruct)) {
    layedStruct.forEach((melds: unknown, seat) => {
      appendLayedMelds(src.layedMelds, seat, melds);
    });
    list.add(layedStruct[src.viewer], true);
  }
  list.add(src.layedMelds.get(src.viewer), true);
  return {list, layoffs: addLayoffs(src, list)};
}

function indexMelds(list: MeldList): Omit<PlayerMelds, 'playerBestMeldIds'> {
  const out: Omit<PlayerMelds, 'playerBestMeldIds'> = {
    playerMelds: [],
    playerMeldOptions: new Map<string, number[]>(),
    playerMeldCards: new Set<string>(),
    playerLockedMeldIds: new Set<string>(),
  };
  for (const {cards, locked} of list.melds) {
    const index = out.playerMelds.length;
    // A content id (the meld's signature), not its position: a positional
    // id would rebind a staged selection to another meld when the candidate
    // list churns between positions. Signatures are unique in the list.
    const id = `meld:${serializeMeldSignature(cards)}`;
    out.playerMelds.push({id, type: meldTypeOf(cards), cards, locked});
    for (const card of cards) {
      out.playerMeldCards.add(card);
      if (locked) continue;
      const options = out.playerMeldOptions.get(card) ?? [];
      options.push(index);
      out.playerMeldOptions.set(card, options);
    }
    if (locked) out.playerLockedMeldIds.add(id);
  }
  return out;
}

/**
 * Auto-selects legal lay-offs (they only lower the layer's own deadwood),
 * except one whose card is in an auto-selected best meld: the defender lays
 * off before the finish-layoffs pass and melds after, so laying the card off
 * first would break the (larger) meld.
 */
function selectLayoffs(
  melds: readonly PlayerMeld[],
  best: string[],
  layoffs: ReadonlySet<string>,
): void {
  if (layoffs.size === 0) return;
  const chosen = new Set(best);
  const byId = new Map(melds.map(meld => [meld.id, meld]));
  const bestCards = new Set<string>();
  for (const id of best) {
    for (const card of byId.get(id)?.cards ?? []) bestCards.add(card);
  }
  for (const meld of melds) {
    if (meld.locked || chosen.has(meld.id)) continue;
    const signature = serializeMeldSignature(meld.cards);
    if (layoffs.has(signature) && !meld.cards.some(c => bestCards.has(c))) {
      best.push(meld.id);
      chosen.add(meld.id);
    }
  }
}

/** The viewer's melds, from the view's analysis, laid melds and plans. */
export function playerMeldsOf(src: MeldSources): PlayerMelds {
  const {list, layoffs} = gather(src);
  const indexed = indexMelds(list);
  const bestIndices =
    src.analysisBest.get(src.viewer) ??
    asNumberArray(src.struct?.['analysisBestMeldIndices']);
  const best: string[] = [];
  for (const value of bestIndices) {
    const meld = indexed.playerMelds[Math.trunc(value)];
    if (meld !== undefined) best.push(meld.id);
  }
  selectLayoffs(indexed.playerMelds, best, layoffs);
  return {...indexed, playerBestMeldIds: best};
}

/** `hand` with the cards of `seat`'s laid melds it does not hold. */
export function mergeHandWithLayed(
  hand: readonly string[],
  layed: readonly string[][] | undefined,
): string[] {
  if (layed === undefined || layed.length === 0) return [...hand];
  const seen = new Set(hand.map(card => normaliseCardCode(card) ?? card));
  const merged = [...hand];
  for (const meld of layed) {
    for (const card of meld) {
      const key = normaliseCardCode(card) ?? card;
      if (key === '' || seen.has(key)) continue;
      merged.push(key);
      seen.add(key);
    }
  }
  return merged;
}
