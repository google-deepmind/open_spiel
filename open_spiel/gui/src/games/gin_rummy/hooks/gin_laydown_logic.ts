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

// The laydown composition's rules, as pure functions: which melds a card
// can cycle through, which lay-offs the engine can reach, what a Done
// submits at each sub-stage, and the driver that walks the sub-stages.
// use_gin_laydown_composition.ts holds the state they act on.

import type {LaydownStage} from '../gin_derived_seats.ts';
import {ginCardPoints} from '../gin_layoffs.ts';
import type {SessionPayload} from '../gin_session.ts';
import {ginDeadwoodOverKnock, ginNotLayable} from '../gin_turn_line.ts';
import type {LaydownPlan, PlayerMeld} from '../types.ts';

export type {LaydownStage} from '../gin_derived_seats.ts';

/**
 * A staged meld or lay-off is identified by the SIGNATURE of its cards
 * (lower-cased, sorted, comma-joined), the scheme the adapter uses for plan
 * steps, so a staged selection survives candidate-list churn: the id is
 * content-based, not positional.
 */
export function serializeSignature(cards: readonly string[]): string {
  return cards
    .map(card => card.toLowerCase())
    .sort()
    .join(',');
}

function setsEqual(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  if (a.size !== b.size) return false;
  for (const value of a) if (!b.has(value)) return false;
  return true;
}

/** Whether the staged lay-off set `target` is one the engine can reach. */
export function reachable(
  plans: readonly LaydownPlan[],
  target: ReadonlySet<string>,
): boolean {
  return plans.some(plan => setsEqual(plan.signatureSet, target));
}

function stepActions(
  step: unknown,
): {signature: string; action: number} | null {
  if (typeof step !== 'object' || step === null) return null;
  const cardsValue: unknown = Reflect.get(step, 'cards');
  const action: unknown = Reflect.get(step, 'action');
  const cards = Array.isArray(cardsValue)
    ? cardsValue.map(card => String(card).toLowerCase()).filter(c => c !== '')
    : [];
  if (cards.length === 0 || typeof action !== 'number') return null;
  return {signature: serializeSignature(cards), action};
}

/**
 * A signature -> action map from a raw `ginLaydownPlans` view. The Done
 * driver resolves actions from each response, not from the rendered
 * snapshot (which lags a render behind), so it can walk the engine's
 * sub-stages within one async function.
 */
export function buildSignatureActionMap(plans: unknown): Map<string, number> {
  const map = new Map<string, number>();
  if (!Array.isArray(plans)) return map;
  for (const plan of plans) {
    if (!Array.isArray(plan)) continue;
    for (const step of plan) {
      const resolved = stepActions(step);
      if (resolved !== null && !map.has(resolved.signature)) {
        map.set(resolved.signature, resolved.action);
      }
    }
  }
  return map;
}

/**
 * Each plan's set of step signatures, from a raw `ginLaydownPlans` view: the
 * sets of melds and lay-offs the engine lets the seat lay before passing.
 */
export function buildPlanSets(plans: unknown): Array<Set<string>> {
  if (!Array.isArray(plans)) return [];
  const out: Array<Set<string>> = [];
  for (const plan of plans) {
    if (!Array.isArray(plan)) continue;
    const set = new Set<string>();
    for (const step of plan) {
      const resolved = stepActions(step);
      if (resolved !== null) set.add(resolved.signature);
    }
    if (set.size > 0) out.push(set);
  }
  return out;
}

/** The candidate melds of a card, in playerMelds order. */
export function candidatesForCard(
  playerMelds: readonly PlayerMeld[],
  cardCode: string,
): PlayerMeld[] {
  const seen = new Set<string>();
  const out: PlayerMeld[] = [];
  for (const meld of playerMelds) {
    if (meld.cards.includes(cardCode) && !seen.has(meld.id)) {
      seen.add(meld.id);
      out.push(meld);
    }
  }
  return out;
}

/** A card's lay-off candidate: the unlocked 1-2 card pseudo-meld holding it. */
export function layoffFor(
  playerMelds: readonly PlayerMeld[],
  cardCode: string,
): PlayerMeld | undefined {
  return playerMelds.find(
    meld =>
      meld.cards.length < 3 && !meld.locked && meld.cards.includes(cardCode),
  );
}

/** The signatures of the staged (unlocked, 1-2 card) lay-offs in `staged`. */
export function stagedLayoffSignatures(
  playerMelds: readonly PlayerMeld[],
  staged: ReadonlySet<string>,
  except?: string,
): Set<string> {
  const out = new Set<string>();
  for (const meld of playerMelds) {
    if (
      staged.has(meld.id) &&
      meld.cards.length < 3 &&
      !meld.locked &&
      meld.id !== except
    ) {
      out.add(serializeSignature(meld.cards));
    }
  }
  return out;
}

/** The staged melds `meld` overlaps (shares a card with). */
export function overlapping(
  playerMelds: readonly PlayerMeld[],
  staged: ReadonlySet<string>,
  meld: PlayerMeld,
): PlayerMeld[] {
  return playerMelds.filter(
    other =>
      other.id !== meld.id &&
      staged.has(other.id) &&
      other.cards.some(card => meld.cards.includes(card)),
  );
}

/** What the composition's rules read. */
export interface Composition {
  readonly playerMelds: readonly PlayerMeld[];
  readonly playerLockedMeldIds: ReadonlySet<string>;
  readonly laydownPlans: readonly LaydownPlan[];
}

/**
 * The staged set after staging lay-off `slot` on top of `staged`, displacing
 * the staged melds that share a card with it, when the engine can reach it
 * (8h cannot ride before 7h); null when it cannot, or when it would displace
 * a laid meld.
 */
export function withLayoff(
  c: Composition,
  staged: ReadonlySet<string>,
  slot: PlayerMeld,
): Set<string> | null {
  const conflicting = overlapping(c.playerMelds, staged, slot);
  if (conflicting.some(other => c.playerLockedMeldIds.has(other.id))) {
    return null;
  }
  const next = new Set(staged);
  for (const other of conflicting) next.delete(other.id);
  const target = stagedLayoffSignatures(c.playerMelds, next);
  target.add(serializeSignature(slot.cards));
  if (!reachable(c.laydownPlans, target)) return null;
  next.add(slot.id);
  return next;
}

/** Hand cards whose lay-off could be staged now, and those in no staged meld. */
export function layoffChoices(
  c: Composition,
  staged: ReadonlySet<string>,
): {layableCards: Set<string>; freeLayoffCards: string[]} {
  const layable = new Set<string>();
  const free: string[] = [];
  for (const meld of c.playerMelds) {
    if (meld.cards.length >= 3 || meld.locked || staged.has(meld.id)) continue;
    if (withLayoff(c, staged, meld) === null) continue;
    for (const card of meld.cards) {
      if (layable.has(card)) continue;
      layable.add(card);
      const inStagedMeld = c.playerMelds.some(
        other =>
          other.cards.length >= 3 &&
          staged.has(other.id) &&
          other.cards.includes(card),
      );
      if (!inStagedMeld) free.push(card);
    }
  }
  return {layableCards: layable, freeLayoffCards: free};
}

/** The cards in staged melds (3+ cards, laid ones included) and staged lay-offs. */
export function stagedCardsOf(
  playerMelds: readonly PlayerMeld[],
  staged: ReadonlySet<string>,
): {melded: Set<string>; layoffs: Set<string>} {
  const layoffs = new Set<string>();
  const melded = new Set<string>();
  for (const meld of playerMelds) {
    if (!staged.has(meld.id)) continue;
    const into = meld.cards.length >= 3 ? melded : meld.locked ? null : layoffs;
    for (const card of meld.cards) into?.add(card);
  }
  return {melded, layoffs};
}

/** The pass-free first staging of a laydown: laid melds and the best melds. */
export function initialStaging(
  playerMelds: readonly PlayerMeld[],
  bestMeldIds: readonly string[],
  lockedIds: ReadonlySet<string>,
): Set<string> {
  const next = new Set<string>(lockedIds);
  const byId = new Map(playerMelds.map(meld => [meld.id, meld]));
  for (const id of bestMeldIds) {
    const meld = byId.get(id);
    // MELDS only. Laying off is a deliberate choice (an expert may HOLD a
    // lay-off, hoping to undercut), so it is never pre-staged.
    if (meld !== undefined && !meld.locked && meld.cards.length >= 3) {
      next.add(id);
    }
  }
  return next;
}

/** What a Done is checked against before anything is sent. */
export interface DoneCheck {
  readonly stage: LaydownStage | null;
  /** The acting seat is the knocker, whose deadwood must reach the card. */
  readonly knocker: boolean;
  readonly knockCard: number | null;
  /** The acting seat's hand. */
  readonly hand: readonly string[];
  /** The cards in staged melds (laid ones included). */
  readonly melded: ReadonlySet<string>;
  readonly stagedDiscard: string | null;
  readonly plans: readonly LaydownPlan[];
  /** The signatures of the staged melds not yet laid (3+ cards). */
  readonly meldSignatures: ReadonlySet<string>;
  /** The signatures of the staged lay-offs not yet laid (1-2 cards). */
  readonly layoffSignatures: ReadonlySet<string>;
}

/**
 * Why Done cannot lay the staged composition down, or null when it can.
 * The engine lets the knocker pass only once the cards left in hand count
 * no more than the knock card, so the knocker's discard and melds are
 * checked against it before the discard is sent; at the meld and lay-off
 * stages the staged set must be one of the engine's plans. Nothing is sent
 * while this says no, so the table never ends past a step the rest of the
 * composition cannot follow.
 */
export function compositionProblem(c: DoneCheck): string | null {
  if (c.stage === null) return null;
  if (c.stage === 'discard' && c.stagedDiscard === null) return null;
  if (c.knocker && c.stage !== 'layoffs' && c.knockCard !== null) {
    const deadwood = c.hand
      .filter(card => card !== c.stagedDiscard && !c.melded.has(card))
      .reduce((sum, card) => sum + ginCardPoints(card), 0);
    if (deadwood > c.knockCard) {
      return ginDeadwoodOverKnock(deadwood, c.knockCard, c.stage === 'discard');
    }
  }
  const staged = c.stage === 'layoffs' ? c.layoffSignatures : c.meldSignatures;
  if (c.stage !== 'discard' && staged.size > 0 && !reachable(c.plans, staged)) {
    return ginNotLayable(c.stage === 'layoffs');
  }
  return null;
}

/** What a Done submits next: its actions, or the error that stops it. */
export type Slice =
  | {readonly kind: 'submit'; readonly actions: number[]}
  | {readonly kind: 'error'; readonly message: string}
  | {readonly kind: 'none'};

/** The staged composition a Done captures once, as it starts. */
export interface DoneComposition {
  readonly stagedMelds: readonly PlayerMeld[];
  readonly stagedLayoffs: readonly PlayerMeld[];
  readonly discardActionId: number | null;
}

/** The actions of the sub-stage `stage`, or why it cannot be submitted. */
export function sliceFor(
  stage: LaydownStage | null,
  passAction: number | null,
  signatureActions: ReadonlyMap<string, number>,
  composition: DoneComposition,
  planSets: ReadonlyArray<ReadonlySet<string>> = [],
): Slice {
  if (stage === 'discard') {
    if (composition.discardActionId === null) {
      return {
        kind: 'error',
        message: 'Pick a card to discard before laying down.',
      };
    }
    // The discard alone: it advances the engine to the meld sub-stage but
    // does not close the turn (no pass).
    return {kind: 'submit', actions: [composition.discardActionId]};
  }
  if (stage === null) return {kind: 'none'};
  const wanted =
    stage === 'layoffs' ? composition.stagedLayoffs : composition.stagedMelds;
  const actions: number[] = [];
  const unresolved: string[] = [];
  for (const meld of wanted) {
    const action = signatureActions.get(serializeSignature(meld.cards));
    if (action === undefined) unresolved.push(meld.cards.join(''));
    else actions.push(action);
  }
  // A staged set must be one the engine can lay and then pass: each of its
  // melds may be legal alone (and resolve to an action) while the set
  // leaves the knocker over the knock card.
  const signatures = new Set(
    wanted.map(meld => serializeSignature(meld.cards)),
  );
  const inPlan =
    signatures.size === 0 || planSets.some(plan => setsEqual(plan, signatures));
  if (unresolved.length > 0 || !inPlan) {
    // A staged card that cannot be laid at this sub-stage (a meld the chosen
    // discard or an earlier lay-off broke) is never silently skipped while
    // the pass closes the turn: nothing of this slice is submitted.
    const named =
      unresolved.length > 0
        ? unresolved.join(', ')
        : wanted.map(meld => meld.cards.join('')).join(', ');
    return {
      kind: 'error',
      message:
        `Cannot lay ${named} at this stage. ` +
        'Adjust your composition, then press Done again ' +
        '— nothing was submitted.',
    };
  }
  if (passAction !== null) actions.push(passAction);
  return actions.length > 0 ? {kind: 'submit', actions} : {kind: 'none'};
}

/** Where the driver is: the seat, its sub-stage, pass and plan actions. */
export interface DriveState {
  readonly seat: number | null;
  readonly stage: LaydownStage | null;
  readonly passAction: number | null;
  readonly signatureActions: ReadonlyMap<string, number>;
  /** Each plan's set of step signatures. */
  readonly planSets: ReadonlyArray<ReadonlySet<string>>;
}

/** The driver's state after `payload`, or null when the laydown is over. */
export function nextDriveState(
  seat: number | null,
  payload: SessionPayload,
): DriveState | null {
  const {state} = payload;
  if (state.isTerminal) return null;
  const nextSeat =
    typeof state.ginLaydownSeat === 'number' ? state.ginLaydownSeat : null;
  // The turn handed off: the laydown is complete.
  if (nextSeat !== seat) return null;
  const stageValue = state.ginLaydownStage;
  const stage =
    stageValue === 'discard' ||
    stageValue === 'layoffs' ||
    stageValue === 'melds'
      ? stageValue
      : null;
  if (stage === null) return null;
  return {
    seat,
    stage,
    passAction:
      typeof state.ginLaydownPassAction === 'number'
        ? state.ginLaydownPassAction
        : null,
    signatureActions: buildSignatureActionMap(state.ginLaydownPlans),
    planSets: buildPlanSets(state.ginLaydownPlans),
  };
}

/** The notice for a sub-stage the server refused. */
export function refusalMessage(err: unknown): string {
  const raw = err instanceof Error ? err.message.trim() : '';
  const reason = raw === '' ? 'the move was refused' : raw;
  const stop = /[.!?]$/.test(reason) ? '' : '.';
  return (
    `The lay-down was not accepted: ${reason}${stop} ` +
    'The table now shows where the hand stands; check your melds, ' +
    'then press Done.'
  );
}

/**
 * Walks the engine's sub-stages: submits the current sub-stage's slice,
 * reads the new stage and plans from the response, and goes on until the
 * turn hands off. On a failure it stops and reports through `onError`; the
 * session then shows the position the server is at (the actions before a
 * refusal stay played), and Done resumes from there.
 */
export async function driveLaydown(
  start: DriveState,
  composition: DoneComposition,
  submit: (actions: readonly number[]) => Promise<SessionPayload>,
  onError: (message: string) => void,
): Promise<void> {
  let at: DriveState | null = start;
  // At most: discard -> melds (knocker) or layoffs -> melds (defender). The
  // bound is a backstop against a malformed stage signal.
  for (let step = 0; step < 4 && at !== null; step += 1) {
    const slice = sliceFor(
      at.stage,
      at.passAction,
      at.signatureActions,
      composition,
      at.planSets,
    );
    if (slice.kind === 'error') {
      onError(slice.message);
      return;
    }
    if (slice.kind === 'none') return;
    let payload: SessionPayload;
    try {
      payload = await submit(slice.actions);
    } catch (err) {
      onError(refusalMessage(err));
      return;
    }
    at = nextDriveState(at.seat, payload);
  }
}
