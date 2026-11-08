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

// Single-stage laydown composition. The player composes the WHOLE laydown
// (the knocker's mandatory discard and melds, or the defender's lay-offs and
// own melds) as local state, then commits it with ONE Done. The engine's
// laydown is sub-staged (knocker: discard -> melds; defender: lay-offs ->
// melds), and a later sub-stage's action ids are not enumerated until the
// boundary action applies, so Done walks the sub-stages as a straight-line
// async driver (gin_laydown_logic.ts driveLaydown): each sub-stage's slice is
// submitted all or nothing, and the response says what comes next. There is
// no cross-render deferral, and a manual choice is never reverted to the best
// melds.

import {
  type Dispatch,
  type RefObject,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import {ginForfeitableLayoffs} from '../gin_layoffs.ts';
import type {SessionPayload} from '../gin_session.ts';
import type {LaydownPlan, PlayerMeld} from '../types.ts';
import {cycleMeld} from './gin_laydown_cycle.ts';
import {
  buildSignatureActionMap,
  driveLaydown,
  initialStaging,
  layoffChoices,
  layoffFor,
  reachable,
  stagedCardsOf,
  stagedLayoffSignatures,
  withLayoff,
  type Composition,
  type LaydownStage,
} from './gin_laydown_logic.ts';
import {useDoneBlocker} from './use_gin_done_blocker.ts';

export type {LaydownStage} from './gin_laydown_logic.ts';

/**
 * How long the "pinned" flash stays on a card the player tried to peel out
 * of order (a chained lay-off cannot be removed before the cards riding on
 * it), ms. Not a move, so no animation speed scales it.
 */
const PINNED_FLASH_MS = 700;

/** What the composition is built from. */
export interface GinLaydownCompositionInput {
  /** The melds (ids are content-based signatures), from the derived state. */
  playerMelds: readonly PlayerMeld[];
  playerBestMeldIds: readonly string[];
  playerLockedMeldIds: ReadonlySet<string>;
  playerHandKey: string;
  /**
   * The position's lay-down plans and the engine's sub-stage. Plans check a
   * lay-off undo against the reachable lay-off sets and resolve the FIRST
   * sub-stage's actions; later sub-stages are resolved from each response.
   */
  laydownPlans: readonly LaydownPlan[];
  laydownPassAction: number | null;
  laydownStage: LaydownStage | null;
  laydownSeat: number | null;
  /**
   * The knocker's mandatory discard is part of the composition: each hand
   * card's discard action id (only in the discard sub-stage).
   */
  discardActions: ReadonlyMap<string, number>;
  viewerIsKnocker: boolean;
  /** The acting seat's hand, which the knocker's deadwood is counted on. */
  playerHand: readonly string[];
  /** The knock card the knocker's deadwood must not exceed. */
  knockCard: number | null;
  shouldShowMeldHints: boolean;
  /**
   * Whether this composition may act on the current laydown; the board
   * decides (the seat itself, or the oracle for any seat).
   */
  canAct: boolean;
  isBusy: boolean;
  /** Submits one sub-stage, all or nothing; rejects naming the failed step. */
  submitAtomicSequence: (actions: readonly number[]) => Promise<SessionPayload>;
}

/** The composition, and what the board can do with it. */
export interface GinLaydownCompositionResult {
  /** Staged melds and lay-offs (content-based ids); drives the highlights. */
  stagedMeldIds: Set<string>;
  /** The knocker's staged discard (null until picked, or not applicable). */
  stagedDiscard: string | null;
  /** A card flashing because it was refused (a pinned lay-off, a laid meld). */
  pinnedFlash: string | null;
  /** Whether Done can fire now. */
  canDone: boolean;
  /**
   * Why Done cannot lay the staged composition down (the knocker's deadwood
   * over the knock card, a staged set in no plan), or null.
   */
  doneBlocker: string | null;
  /** Why the last Done stopped; never silently swallowed. */
  submitError: string | null;
  /** Steps a card through its meld cycle. */
  toggleMeldForCard: (cardCode: string) => void;
  /** Peels a staged lay-off off (chains peel outside-in; else it flashes). */
  undoLayoff: (cardCode: string) => void;
  /** Stages or unstages the knocker's discard. */
  stageDiscard: (cardCode: string) => void;
  /** Hand cards whose lay-off could be staged now. */
  layableCards: Set<string>;
  /** Of those, the ones in no staged meld: what a Done forfeits for nothing. */
  freeLayoffCards: string[];
  /** Stages a card's lay-off; false when it is not legal now. */
  stageLayoff: (cardCode: string) => boolean;
  /**
   * What a Done now would forfeit, chains included, and its deadwood; null
   * when the staged lay-offs are in no engine plan (logged), so the board
   * asks before Done, saying it could not check.
   */
  forfeitableLayoffs: {cards: string[]; points: number} | null;
  /** The cards in staged melds (3+ cards, laid ones included) and lay-offs. */
  stagedCards: {melded: Set<string>; layoffs: Set<string>};
  /** Submits the composition, sub-stage by sub-stage; resumable on failure. */
  done: () => void;
}

interface Staging {
  stagedMeldIds: Set<string>;
  setStagedMeldIds: Dispatch<SetStateAction<Set<string>>>;
  stagedDiscard: string | null;
  setStagedDiscard: Dispatch<SetStateAction<string | null>>;
  setSubmitError: Dispatch<SetStateAction<string | null>>;
}

/**
 * Stages the best melds once per laydown turn, keyed on the hand, so the
 * common case is one click; a manual edit is never reverted. A committed
 * sub-stage changes the hand but is not a new laydown, so once Done starts
 * the staging is frozen until the acting seat changes.
 */
function useInitialStaging(
  input: GinLaydownCompositionInput,
  s: Staging,
  driveStartedRef: RefObject<boolean>,
): void {
  const {laydownSeat, shouldShowMeldHints, playerHandKey} = input;
  const {playerBestMeldIds, playerLockedMeldIds, playerMelds} = input;
  const {setStagedMeldIds, setStagedDiscard, setSubmitError} = s;
  const initialisedForHand = useRef<string | null>(null);
  const frozenForSeat = useRef<number | null>(null);
  useEffect(() => {
    if (laydownSeat !== frozenForSeat.current) {
      // A new acting seat (or a laydown left or entered) starts fresh.
      frozenForSeat.current = laydownSeat;
      driveStartedRef.current = false;
      initialisedForHand.current = null;
    }
    if (laydownSeat === null || !shouldShowMeldHints) return;
    if (
      driveStartedRef.current ||
      initialisedForHand.current === playerHandKey
    ) {
      return;
    }
    initialisedForHand.current = playerHandKey;
    setStagedMeldIds(
      initialStaging(playerMelds, playerBestMeldIds, playerLockedMeldIds),
    );
    setStagedDiscard(null);
    setSubmitError(null);
  }, [
    laydownSeat,
    shouldShowMeldHints,
    playerHandKey,
    playerBestMeldIds,
    playerLockedMeldIds,
    playerMelds,
    setStagedMeldIds,
    setStagedDiscard,
    setSubmitError,
    driveStartedRef,
  ]);
  // Laid melds are always part of the composition.
  useEffect(() => {
    if (playerLockedMeldIds.size === 0) return;
    setStagedMeldIds(prev => {
      const missing = [...playerLockedMeldIds].filter(id => !prev.has(id));
      return missing.length === 0 ? prev : new Set([...prev, ...missing]);
    });
  }, [playerLockedMeldIds, setStagedMeldIds]);
}

function usePinnedFlash(): [string | null, (cardCode: string) => void] {
  const [pinnedFlash, setPinnedFlash] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current);
    },
    [],
  );
  const flash = useCallback((cardCode: string) => {
    setPinnedFlash(cardCode);
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      setPinnedFlash(null);
      timer.current = null;
    }, PINNED_FLASH_MS);
  }, []);
  return [pinnedFlash, flash];
}

function useStageDiscard(
  input: GinLaydownCompositionInput,
  s: Staging,
): (cardCode: string) => void {
  const {viewerIsKnocker, laydownStage, discardActions} = input;
  const {playerMelds, playerLockedMeldIds} = input;
  const {stagedDiscard, setStagedDiscard, setStagedMeldIds, setSubmitError} = s;
  return useCallback(
    (cardCode: string) => {
      if (!viewerIsKnocker || laydownStage !== 'discard') return;
      if (!discardActions.has(cardCode)) return;
      setSubmitError(null);
      const marking = stagedDiscard !== cardCode;
      setStagedDiscard(prev => (prev === cardCode ? null : cardCode));
      // A card cannot be both melded and the discard: MARKING it as the
      // discard evicts it from any staged (not laid) meld. Un-marking leaves
      // the melds alone.
      if (!marking) return;
      setStagedMeldIds(prev => {
        const evict = playerMelds.filter(
          meld =>
            prev.has(meld.id) &&
            !playerLockedMeldIds.has(meld.id) &&
            meld.cards.includes(cardCode),
        );
        if (evict.length === 0) return prev;
        const next = new Set(prev);
        for (const meld of evict) next.delete(meld.id);
        return next;
      });
    },
    [
      viewerIsKnocker,
      laydownStage,
      discardActions,
      stagedDiscard,
      playerMelds,
      playerLockedMeldIds,
      setStagedDiscard,
      setStagedMeldIds,
      setSubmitError,
    ],
  );
}

function useToggleMeld(
  input: GinLaydownCompositionInput,
  s: Staging,
  flash: (cardCode: string) => void,
  stageDiscard: (cardCode: string) => void,
): (cardCode: string) => void {
  const {shouldShowMeldHints, laydownStage, viewerIsKnocker} = input;
  const {discardActions, playerMelds, playerLockedMeldIds, laydownPlans} =
    input;
  const {stagedMeldIds, stagedDiscard} = s;
  const {setStagedMeldIds, setStagedDiscard, setSubmitError} = s;
  return useCallback(
    (cardCode: string) => {
      if (!shouldShowMeldHints) return;
      const result = cycleMeld(
        {
          playerMelds,
          playerLockedMeldIds,
          laydownPlans,
          stagedMeldIds,
          stagedDiscard,
          laydownStage,
          discardEligible:
            laydownStage === 'discard' &&
            viewerIsKnocker &&
            discardActions.has(cardCode),
        },
        cardCode,
      );
      if (result.kind === 'toggle-discard') {
        stageDiscard(cardCode);
        return;
      }
      if (result.kind === 'none') return;
      setSubmitError(null);
      if (result.flashed) flash(cardCode);
      if (result.discard === 'clear') setStagedDiscard(null);
      if (result.discard === 'set') setStagedDiscard(cardCode);
      setStagedMeldIds(result.staged);
    },
    [
      shouldShowMeldHints,
      laydownStage,
      viewerIsKnocker,
      discardActions,
      playerMelds,
      playerLockedMeldIds,
      laydownPlans,
      stagedMeldIds,
      stagedDiscard,
      flash,
      stageDiscard,
      setStagedMeldIds,
      setStagedDiscard,
      setSubmitError,
    ],
  );
}

function useUndoLayoff(
  input: GinLaydownCompositionInput,
  s: Staging,
  flash: (cardCode: string) => void,
): (cardCode: string) => void {
  const {playerMelds, laydownPlans} = input;
  const {stagedMeldIds, setStagedMeldIds, setSubmitError} = s;
  return useCallback(
    (cardCode: string) => {
      // The staged lay-off (1-2 card pseudo-meld) that holds this card.
      const target = playerMelds.find(
        meld =>
          stagedMeldIds.has(meld.id) &&
          meld.cards.length < 3 &&
          !meld.locked &&
          meld.cards.includes(cardCode),
      );
      if (target === undefined) return;
      // In the lay-off sub-stage every reachable lay-off set is a plan, so
      // the set left after peeling is legal iff it is empty or a plan. If it
      // is not, the card holds a chain (a later card rides on it): it flashes.
      const remaining = stagedLayoffSignatures(
        playerMelds,
        stagedMeldIds,
        target.id,
      );
      if (remaining.size > 0 && !reachable(laydownPlans, remaining)) {
        flash(cardCode);
        return;
      }
      setSubmitError(null);
      setStagedMeldIds(prev => {
        const next = new Set(prev);
        next.delete(target.id);
        return next;
      });
    },
    [
      playerMelds,
      stagedMeldIds,
      laydownPlans,
      flash,
      setStagedMeldIds,
      setSubmitError,
    ],
  );
}

function useStageLayoff(
  input: GinLaydownCompositionInput,
  s: Staging,
  composition: Composition,
  flash: (cardCode: string) => void,
): (cardCode: string) => boolean {
  const {laydownStage, canAct, playerMelds} = input;
  const {stagedMeldIds, setStagedMeldIds, setSubmitError} = s;
  return useCallback(
    (cardCode: string): boolean => {
      if (laydownStage !== 'layoffs' || !canAct) return false;
      const slot = layoffFor(playerMelds, cardCode);
      if (slot === undefined || stagedMeldIds.has(slot.id)) return false;
      const next = withLayoff(composition, stagedMeldIds, slot);
      if (next === null) {
        flash(cardCode);
        return false;
      }
      setSubmitError(null);
      setStagedMeldIds(next);
      return true;
    },
    [
      laydownStage,
      canAct,
      playerMelds,
      stagedMeldIds,
      composition,
      flash,
      setStagedMeldIds,
      setSubmitError,
    ],
  );
}

function useForfeitable(
  input: GinLaydownCompositionInput,
  stagedCards: {melded: Set<string>; layoffs: Set<string>},
): {cards: string[]; points: number} | null {
  const {laydownStage, canAct, laydownPlans} = input;
  return useMemo(() => {
    if (laydownStage !== 'layoffs' || !canAct) return {cards: [], points: 0};
    const forfeit = ginForfeitableLayoffs(
      laydownPlans,
      stagedCards.layoffs,
      stagedCards.melded,
    );
    if (forfeit === null) {
      console.error(
        `gin lay-offs: the staged lay-offs ${[...stagedCards.layoffs].join(' ')} are in none of the engine's ${String(laydownPlans.length)} laydown plans`,
      );
    }
    return forfeit;
  }, [laydownStage, canAct, stagedCards, laydownPlans]);
}

function canDoneNow(
  input: GinLaydownCompositionInput,
  stagedDiscard: string | null,
  blocker: string | null,
): boolean {
  if (!input.canAct || input.isBusy || input.laydownSeat === null) return false;
  if (blocker !== null) return false;
  // The mandatory discard must be chosen before the turn can commit; in the
  // lay-off and meld stages a pass must be legal (a naked pass is a valid
  // Done for a player with nothing to lay).
  if (input.laydownStage === 'discard') return stagedDiscard !== null;
  return input.laydownPassAction !== null;
}

function useDone(
  input: GinLaydownCompositionInput,
  s: Staging,
  canDone: boolean,
  driveStartedRef: RefObject<boolean>,
): () => void {
  const {playerMelds, discardActions, laydownSeat, laydownStage} = input;
  const {laydownPassAction, laydownPlans, submitAtomicSequence} = input;
  const {stagedMeldIds, stagedDiscard, setSubmitError} = s;
  // Guards a drive so a re-render mid-drive cannot launch a second one
  // (which would double-submit a committed sub-stage).
  const driveInFlight = useRef(false);
  return useCallback(() => {
    if (!canDone || driveInFlight.current) return;
    driveInFlight.current = true;
    // From here the composition is the player's across every sub-stage of
    // this laydown: committed sub-stages change the hand, but must not
    // re-stage the best melds.
    driveStartedRef.current = true;
    setSubmitError(null);
    // The composition is captured once. Melds are 3+ cards; lay-offs 1-2.
    // A laid (locked) meld is already on the table: the engine's plans at a
    // resumed sub-stage no longer hold it, so it is not sent again.
    const staged = playerMelds.filter(
      meld => stagedMeldIds.has(meld.id) && !meld.locked,
    );
    const composition = {
      stagedMelds: staged.filter(meld => meld.cards.length >= 3),
      stagedLayoffs: staged.filter(meld => meld.cards.length < 3),
      discardActionId:
        stagedDiscard === null
          ? null
          : (discardActions.get(stagedDiscard) ?? null),
    };
    const start = {
      seat: laydownSeat,
      stage: laydownStage,
      passAction: laydownPassAction,
      signatureActions: buildSignatureActionMap(
        laydownPlans.map(plan => plan.sequence),
      ),
      planSets: laydownPlans.map(plan => plan.signatureSet),
    };
    void driveLaydown(
      start,
      composition,
      submitAtomicSequence,
      setSubmitError,
    ).finally(() => {
      driveInFlight.current = false;
    });
  }, [
    canDone,
    playerMelds,
    stagedMeldIds,
    stagedDiscard,
    discardActions,
    laydownSeat,
    laydownStage,
    laydownPassAction,
    laydownPlans,
    submitAtomicSequence,
    setSubmitError,
    driveStartedRef,
  ]);
}

/** The single-stage laydown composition of the acting seat. */
export function useGinLaydownComposition(
  input: GinLaydownCompositionInput,
): GinLaydownCompositionResult {
  const [stagedMeldIds, setStagedMeldIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [stagedDiscard, setStagedDiscard] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // Set when a Done drive starts; frozen until the laydown hands off.
  const driveStartedRef = useRef(false);
  const s: Staging = {
    stagedMeldIds,
    setStagedMeldIds,
    stagedDiscard,
    setStagedDiscard,
    setSubmitError,
  };
  const [pinnedFlash, flash] = usePinnedFlash();
  useInitialStaging(input, s, driveStartedRef);
  const {playerMelds, playerLockedMeldIds, laydownPlans} = input;
  const composition = useMemo(
    () => ({playerMelds, playerLockedMeldIds, laydownPlans}),
    [playerMelds, playerLockedMeldIds, laydownPlans],
  );
  const {laydownStage, canAct} = input;
  const choices = useMemo(
    () =>
      laydownStage !== 'layoffs' || !canAct
        ? {layableCards: new Set<string>(), freeLayoffCards: []}
        : layoffChoices(composition, stagedMeldIds),
    [laydownStage, canAct, composition, stagedMeldIds],
  );
  const stagedCards = useMemo(
    () => stagedCardsOf(playerMelds, stagedMeldIds),
    [playerMelds, stagedMeldIds],
  );
  const stageDiscard = useStageDiscard(input, s);
  const doneBlocker = useDoneBlocker(
    input,
    stagedMeldIds,
    stagedDiscard,
    stagedCards.melded,
  );
  const canDone = canDoneNow(input, stagedDiscard, doneBlocker);
  return {
    stagedMeldIds,
    stagedDiscard,
    pinnedFlash,
    canDone,
    doneBlocker,
    submitError,
    toggleMeldForCard: useToggleMeld(input, s, flash, stageDiscard),
    undoLayoff: useUndoLayoff(input, s, flash),
    stageDiscard,
    ...choices,
    stageLayoff: useStageLayoff(input, s, composition, flash),
    forfeitableLayoffs: useForfeitable(input, stagedCards),
    stagedCards,
    done: useDone(input, s, canDone, driveStartedRef),
  };
}
