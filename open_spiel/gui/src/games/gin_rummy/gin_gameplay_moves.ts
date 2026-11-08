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

// The gin board's moves: the laydown composition of the acting seat, the
// draw, pass and knock handlers, the Done that asks before forfeiting
// lay-offs, and the double-click on a card.

import {useCallback, useMemo, useState} from 'react';

import type {GinView} from './gin_gameplay_view.ts';
import type {GinTurn} from './gin_gameplay_turn.ts';
import type {SessionContextValue} from './gin_session.ts';
import {
  useGinLaydownComposition,
  type GinLaydownCompositionResult,
} from './hooks/use_gin_laydown_composition.ts';
import {
  useGinRummyDerivedState,
  type GinRummyDerivedState,
} from './use_gin_rummy_derived_state.ts';
import type {PlayerMeld} from './types.ts';

const KNOCK_PHASES = new Set(['knock', 'layoff', 'show', 'showdown', 'score']);

/** The laydown in progress, as the board composes it. */
export interface GinCompose {
  composition: GinLaydownCompositionResult;
  composeMelds: readonly PlayerMeld[];
  shouldShowMeldHints: boolean;
  laydownActive: boolean;
  composeOnPlayerGrid: boolean;
  composeOnOpponentGrid: boolean;
  /** Staged lay-offs: the cards leave the hand for the knocker's spread. */
  stagedLayoffCards: Set<string>;
}

/**
 * The laydown composition. Compose data follows the ACTING seat, not the
 * viewed one: a second derived-state pass keyed on the laydown seat supplies
 * the actor's melds, laid melds, hand key and discard candidates. When the
 * viewed seat is the actor the two passes coincide. The seat itself acts
 * (the oracle, in analysis, acts for any seat).
 */
export function useGinCompose(
  s: SessionContextValue,
  d: GinRummyDerivedState,
  view: GinView,
  turn: GinTurn,
): GinCompose {
  const {laydownSeat, laydownStage} = d;
  const shouldShowMeldHints = d.knocked || KNOCK_PHASES.has(d.normalizedPhase);
  const canAct =
    laydownSeat !== null &&
    !s.isBusy &&
    !d.isTerminal &&
    (view.isAnalysisMode ||
      (turn.canInteract && turn.viewerSeatForRole === laydownSeat));
  const composeDerived = useGinRummyDerivedState(s.session, {
    preferObservationStruct: view.preferObservationStruct,
    viewerSeat: laydownSeat ?? view.effectiveViewSeat,
  });
  const composition = useGinLaydownComposition({
    playerMelds: composeDerived.playerMelds,
    playerBestMeldIds: composeDerived.playerBestMeldIds,
    playerLockedMeldIds: composeDerived.playerLockedMeldIds,
    playerHandKey: composeDerived.playerHandKey,
    laydownPlans: d.laydownPlans,
    laydownPassAction: d.laydownPassAction,
    laydownStage,
    laydownSeat,
    discardActions: composeDerived.discardActions,
    viewerIsKnocker: laydownSeat !== null && turn.knockerSeats.has(laydownSeat),
    playerHand: composeDerived.playerHand,
    knockCard: d.knockCard,
    shouldShowMeldHints,
    canAct,
    isBusy: s.isBusy,
    submitAtomicSequence: s.submitAtomicSequence,
  });
  // Which grid carries the compose gesture depends on where the actor's
  // hand renders: the player grid when the viewed seat is the actor, the
  // opponent grid otherwise (analysis cross-seat control).
  const laydownActive = laydownStage !== null && canAct;
  const {layoffs} = composition.stagedCards;
  const stagedLayoffCards = useMemo(
    () => (laydownActive ? layoffs : new Set<string>()),
    [laydownActive, layoffs],
  );
  return {
    composition,
    composeMelds: composeDerived.playerMelds,
    shouldShowMeldHints,
    laydownActive,
    composeOnPlayerGrid: laydownActive && laydownSeat === d.playerHandSlot,
    composeOnOpponentGrid:
      laydownActive &&
      view.isAnalysisMode &&
      laydownSeat === d.opponentHandSlot,
    stagedLayoffCards,
  };
}

/** The board's move handlers. */
export interface GinMoves {
  handleActionDoubleClick: (actionId: number | null) => void;
  drawUpcardHandler: (() => void) | null;
  drawStockHandler: (() => void) | null;
  passHandler: (() => void) | null;
  handleKnock: () => void;
  canAttemptKnock: boolean;
}

/** Plays an action when the viewer may, and the draw, pass and knock moves. */
export function useGinMoves(
  s: SessionContextValue,
  d: GinRummyDerivedState,
  view: GinView,
  turn: GinTurn,
): GinMoves {
  const {isBusy, submitAction} = s;
  const {isTerminal, knockAction} = d;
  const {isAnalysisMode} = view;
  const {viewerIsActiveSeat} = turn;
  const handleActionDoubleClick = useCallback(
    (actionId: number | null) => {
      if (actionId === null || isBusy || isTerminal) return;
      if (!isAnalysisMode && !viewerIsActiveSeat) return;
      void submitAction(actionId);
    },
    [submitAction, isBusy, isTerminal, isAnalysisMode, viewerIsActiveSeat],
  );
  const handleKnock = useCallback(() => {
    handleActionDoubleClick(knockAction);
  }, [knockAction, handleActionDoubleClick]);
  const may = isAnalysisMode || turn.canInteract;
  const play = (action: number | null): (() => void) | null =>
    may && action !== null
      ? () => {
          handleActionDoubleClick(action);
        }
      : null;
  return {
    handleActionDoubleClick,
    drawUpcardHandler: play(d.drawUpcardAction),
    drawStockHandler: play(d.drawStockAction),
    // The Pass button serves the first-upcard pass and the wall's pass
    // (use_gin_actions.ts, PASS_BUTTON_PHASES): every laydown commit goes
    // through Done.
    passHandler: play(d.passAction),
    handleKnock,
    canAttemptKnock: may,
  };
}

/** The Done button, and whether it first asks about forfeited lay-offs. */
export interface GinDone {
  doneHandler: (() => void) | undefined;
  skipArmed: boolean;
  forfeitCount: number;
}

/**
 * Done would forfeit legal lay-offs: the first press says how many and what
 * they are worth on the turn line, chains included; a second press is the
 * Done. Any change to the composition asks again. When the staged lay-offs
 * are in no plan, what Done forfeits is unknown: Done still asks first, and
 * says so.
 */
export function useGinDone(d: GinRummyDerivedState, c: GinCompose): GinDone {
  const {composition, laydownActive} = c;
  const {stagedMeldIds, forfeitableLayoffs, canDone, done} = composition;
  const {laydownStage, laydownSeat} = d;
  // Armed for one composition: any change to it disarms.
  const [armedFor, setArmedFor] = useState<{
    staged: Set<string>;
    stage: string | null;
    seat: number | null;
  } | null>(null);
  const armed =
    armedFor !== null &&
    armedFor.staged === stagedMeldIds &&
    armedFor.stage === laydownStage &&
    armedFor.seat === laydownSeat;
  const setArmed = useCallback(
    (on: boolean) => {
      setArmedFor(
        on
          ? {staged: stagedMeldIds, stage: laydownStage, seat: laydownSeat}
          : null,
      );
    },
    [stagedMeldIds, laydownStage, laydownSeat],
  );
  const forfeitCount = forfeitableLayoffs?.cards.length ?? 0;
  const askBeforeDone = forfeitableLayoffs === null || forfeitCount > 0;
  const handleDoneChecked = useCallback(() => {
    if (askBeforeDone && !armed) {
      setArmed(true);
      return;
    }
    setArmed(false);
    done();
  }, [askBeforeDone, armed, done, setArmed]);
  return {
    doneHandler: laydownActive && canDone ? handleDoneChecked : undefined,
    skipArmed: askBeforeDone && armed,
    forfeitCount,
  };
}

/**
 * The double-click on a hand card during the laydown: in the knocker's
 * discard sub-stage the toggle cycle owns the whole gesture (candidate melds
 * -> unstaged -> staged as THE discard -> back); in the lay-off sub-stage a
 * staged lay-off is peeled (chain-checked), a free lay-off card lays off,
 * and any other card cycles its meld; in the meld sub-stage it cycles a meld.
 */
export function useComposeCard(
  d: GinRummyDerivedState,
  c: GinCompose,
): (cardCode: string) => void {
  const {laydownStage} = d;
  const {composeMelds} = c;
  const {
    stagedMeldIds,
    toggleMeldForCard,
    undoLayoff,
    stageLayoff,
    freeLayoffCards,
  } = c.composition;
  const free = useMemo(() => new Set(freeLayoffCards), [freeLayoffCards]);
  return useCallback(
    (cardCode: string) => {
      if (laydownStage !== 'layoffs') {
        toggleMeldForCard(cardCode);
        return;
      }
      const inStagedLayoff = composeMelds.some(
        meld =>
          stagedMeldIds.has(meld.id) &&
          meld.cards.length < 3 &&
          meld.cards.includes(cardCode),
      );
      if (inStagedLayoff) undoLayoff(cardCode);
      else if (free.has(cardCode)) stageLayoff(cardCode);
      else toggleMeldForCard(cardCode);
    },
    [
      laydownStage,
      composeMelds,
      stagedMeldIds,
      toggleMeldForCard,
      undoLayoff,
      free,
      stageLayoff,
    ],
  );
}
