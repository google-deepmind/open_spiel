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

// Lay-offs onto the knocker's melds: which meld a dropped card extends, and
// what the cards a Done would forfeit are worth. Pure.
//
// Rules: the defender may lay
// off a deadwood card that extends one of the knocker's sets to four or one of
// its runs at either end, and a card that fits only after another lay-off has
// extended the run may be laid off too. Aces are low (A-2-3, never Q-K-A). The
// engine's laydown plans decide what is legal; this module only decides which
// meld a drop lands on, so a card is accepted on the meld it extends.

const RANKS = 'A23456789TJQK';

function rankOf(card: string): number {
  return RANKS.indexOf(card.charAt(0));
}

function suitOf(card: string): string {
  return card.charAt(1);
}

/** A card's deadwood count: ace 1, two to nine their pips, ten and court 10. */
export function ginCardPoints(card: string): number {
  const rank = rankOf(card);
  if (rank < 0) throw new Error(`not a gin card: ${card}`);
  return Math.min(rank + 1, 10);
}

/** Does `card` extend `meld`: a set (one rank) to at most four, or a run (one
 *  suit, consecutive) at either end? */
export function ginExtendsMeld(card: string, meld: readonly string[]): boolean {
  const first = meld[0];
  if (first === undefined || meld.includes(card)) return false;
  const isSet = meld.every(c => c.startsWith(first.charAt(0)));
  if (isSet && meld.length >= 2 && meld.length < 4) {
    return card.startsWith(first.charAt(0));
  }
  if (
    !meld.every(c => suitOf(c) === suitOf(first)) ||
    suitOf(card) !== suitOf(first)
  ) {
    return false;
  }
  const ranks = meld.map(rankOf);
  const r = rankOf(card);
  return r === Math.min(...ranks) - 1 || r === Math.max(...ranks) + 1;
}

function isRun(meld: readonly string[]): boolean {
  const first = meld[0];
  return (
    first !== undefined &&
    meld.every(c => suitOf(c) === suitOf(first)) &&
    !meld.every(c => c.startsWith(first.charAt(0)))
  );
}

/** The knocker's melds with the staged lay-offs on them: each staged card
 *  joins a meld it extends, repeatedly, so a chained lay-off (9s after 8s)
 *  rides on the one before it. A card that extends both a set and a run
 *  joins the run, as the engine counts it, so the next card of the run's
 *  chain lands on it. */
export function ginMeldsWithLayoffs(
  melds: ReadonlyArray<readonly string[]>,
  staged: Iterable<string>,
): string[][] {
  const out = melds.map(m => [...m]);
  const left = new Set(staged);
  let placed = true;
  while (placed && left.size > 0) {
    placed = false;
    for (const card of [...left]) {
      const fits = out.filter(m => ginExtendsMeld(card, m));
      const meld = fits.find(isRun) ?? fits[0];
      if (meld !== undefined) {
        meld.push(card);
        left.delete(card);
        placed = true;
      }
    }
  }
  return out;
}

/** Is a drop of `card` on `onto` (a card of the knocker's spread, a staged
 *  lay-off included) a drop on a meld that `card` extends? */
export function ginLayoffDropFits(
  card: string,
  onto: string,
  knockerMelds: ReadonlyArray<readonly string[]>,
  staged: Iterable<string>,
): boolean {
  const meld = ginMeldsWithLayoffs(knockerMelds, staged).find(m =>
    m.includes(onto),
  );
  return meld !== undefined && ginExtendsMeld(card, meld);
}

/** One of the engine's laydown plans in the lay-off stage: the lay-off steps
 *  of one reachable set of lay-offs (every reachable subset is a plan). */
export interface GinLayoffPlan {
  sequence: ReadonlyArray<{cards: readonly string[]}>;
}

/** The lay-offs a Done now would forfeit, and their deadwood: the most
 *  valuable reachable plan that keeps every staged lay-off, less the staged
 *  ones, and leaving out any plan that would take a card from a staged meld.
 *  A chain counts in full (4c and 5c when only 4c lays off yet), since
 *  Done gives up both. Null when lay-offs are staged that no plan holds: the
 *  composition and the engine's plans disagree, so what Done forfeits is not
 *  known (the caller says so rather than read it as nothing). */
export function ginForfeitableLayoffs(
  plans: readonly GinLayoffPlan[],
  stagedLayoffs: ReadonlySet<string>,
  meldedCards: ReadonlySet<string>,
): {cards: string[]; points: number} | null {
  let best: {cards: string[]; points: number} | null =
    stagedLayoffs.size === 0 ? {cards: [], points: 0} : null;
  for (const plan of plans) {
    const planCards = plan.sequence.flatMap(step => step.cards);
    if (
      planCards.length === 0 ||
      ![...stagedLayoffs].every(c => planCards.includes(c))
    ) {
      continue;
    }
    const cards = planCards.filter(c => !stagedLayoffs.has(c));
    if (cards.some(c => meldedCards.has(c))) continue;
    const points = cards.reduce((sum, c) => sum + ginCardPoints(c), 0);
    if (
      best === null ||
      points > best.points ||
      (points === best.points && cards.length > best.cards.length)
    ) {
      best = {cards, points};
    }
  }
  return best;
}

/** What the knocker's discard does to the composed deadwood. */
export interface ComposedDiscard {
  /** The staged discard, which leaves the hand (null when none is picked). */
  readonly staged: string | null;
  /**
   * Whether a discard is still owed (the knocker's discard sub-stage). With
   * none picked, the count leaves out the highest unmelded card, the best
   * discard the staged melds allow: the figure the engine showed for the
   * eleven-card hand before the knock.
   */
  readonly pending: boolean;
}

const NO_DISCARD: ComposedDiscard = {staged: null, pending: false};

/** The deadwood a Done now leaves: the hand's cards that are in no staged
 *  meld, not laid off and not the staged discard, at their card values (the
 *  engine's count after the laydown, TotalCardValue of what is left). */
export function ginComposedDeadwood(
  hand: readonly string[],
  meldedCards: ReadonlySet<string>,
  stagedLayoffs: ReadonlySet<string>,
  discard: ComposedDiscard = NO_DISCARD,
): number {
  const left = hand
    .filter(
      c => !meldedCards.has(c) && !stagedLayoffs.has(c) && c !== discard.staged,
    )
    .map(c => ginCardPoints(c));
  const total = left.reduce((sum, v) => sum + v, 0);
  if (!discard.pending || discard.staged !== null || left.length === 0) {
    return total;
  }
  return total - Math.max(...left);
}
