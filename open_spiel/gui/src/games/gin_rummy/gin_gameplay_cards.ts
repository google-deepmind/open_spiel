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

// What the gin board draws for each grid: the cards each zone shows and
// their states, the knocker's spread for lay-offs, the turn line during the
// laydown, and the finished hand's accounting.

import {useCallback, useMemo} from 'react';

import type {LayoffDrop, BoardHandResult} from './gin_board_parts.ts';
import type {GinCompose, GinDone, GinMoves} from './gin_gameplay_moves.ts';
import type {GinTurn} from './gin_gameplay_turn.ts';
import type {GinView} from './gin_gameplay_view.ts';
import {
  ginHandResult,
  ginResultFigure,
  type GinSeatNamer,
} from './gin_hand_result.ts';
import {ginComposedDeadwood, ginLayoffDropFits} from './gin_layoffs.ts';
import type {SessionContextValue} from './gin_session.ts';
import {
  GIN_LAYOFFS_UNCHECKED,
  ginLaydownPrompt,
  ginLayoffSkipPrompt,
} from './gin_turn_line.ts';
import {
  useGinCardStates,
  type GinCardStatesResult,
} from './hooks/use_gin_card_states.ts';
import type {MeldType} from './types.ts';
import type {GinRummyDerivedState} from './use_gin_rummy_derived_state.ts';

const LAYOFF_PHASES = new Set(['layoff', 'show', 'showdown', 'score']);
const NO_TYPES: ReadonlyMap<string, MeldType> = new Map();
const NO_CARDS: ReadonlySet<string> = new Set();

/** Whether the opponent's cards are shown face up, and as what. */
export interface OpponentShown {
  showOpponentCards: boolean;
  manualOpponentControl: boolean;
}

/** Whether the opponent's hand is shown: revealed, chosen (analysis), or laid off onto. */
export function opponentShown(
  d: GinRummyDerivedState,
  view: GinView,
  turn: GinTurn,
): OpponentShown {
  const opponentAutoReveal =
    d.revealHands ||
    (d.opponentHandSlot !== null && d.visibleHands.has(d.opponentHandSlot));
  const forceLayoffReveal =
    turn.viewerIsKnocker && LAYOFF_PHASES.has(d.normalizedPhase);
  return {
    showOpponentCards:
      forceLayoffReveal || view.manualOpponentReveal || opponentAutoReveal,
    manualOpponentControl: view.manualOpponentReveal || opponentAutoReveal,
  };
}

function meldTypes(melds: readonly string[][]): Map<string, MeldType> {
  const types = new Map<string, MeldType>();
  for (const meld of melds) {
    const type: MeldType =
      new Set(meld.map(card => card.slice(-1))).size === 1 ? 'run' : 'set';
    for (const card of meld) types.set(card, type);
  }
  return types;
}

/** The opponent grid's meld bands: its laid melds, and the defender's pending ones. */
function useOpponentMeldTypes(d: GinRummyDerivedState): {
  laid: ReadonlyMap<string, MeldType>;
  pending: ReadonlyMap<string, MeldType>;
} {
  const {opponentHandSlot, laydownSeat, normalizedPhase} = d;
  const {
    layedMeldCardTypesBySeat,
    analysisBestMeldsBySeat,
    analysisMeldsBySeat,
  } = d;
  const showBands =
    laydownSeat !== null &&
    opponentHandSlot === laydownSeat &&
    LAYOFF_PHASES.has(normalizedPhase);
  const laid = useMemo(
    () =>
      opponentHandSlot === null
        ? NO_TYPES
        : (layedMeldCardTypesBySeat.get(opponentHandSlot) ?? NO_TYPES),
    [layedMeldCardTypesBySeat, opponentHandSlot],
  );
  const pending = useMemo(() => {
    if (!showBands) return NO_TYPES;
    return meldTypes(
      analysisBestMeldsBySeat.get(opponentHandSlot) ??
        analysisMeldsBySeat.get(opponentHandSlot) ??
        [],
    );
  }, [
    analysisBestMeldsBySeat,
    analysisMeldsBySeat,
    opponentHandSlot,
    showBands,
  ]);
  return {laid, pending};
}

function useKnownCards(
  d: GinRummyDerivedState,
  show: boolean,
  seat: number | null,
): ReadonlySet<string> {
  const {knownCardsBySeat} = d;
  return useMemo(
    () =>
      !show || seat === null
        ? NO_CARDS
        : new Set(knownCardsBySeat.get(seat) ?? []),
    [knownCardsBySeat, seat, show],
  );
}

/** Both grids' cell states, from everything the board knows. */
export function useBoardCardStates(
  d: GinRummyDerivedState,
  view: GinView,
  shown: OpponentShown,
  c: GinCompose,
  handlers: {
    normalDiscard: (card: string) => void;
    composeCard: (card: string) => void;
    canInteract: boolean;
  },
): GinCardStatesResult & {opponentHandDisplay: readonly string[]} {
  const hints = view.showKnowledgeHints;
  const {opponentHand, discardHighlights} = d;
  const opponentHandDisplay = useMemo(
    () =>
      shown.showOpponentCards ? opponentHand : opponentHand.map(() => 'XX'),
    [opponentHand, shown.showOpponentCards],
  );
  const opponentHandSet = useMemo(() => new Set(opponentHand), [opponentHand]);
  const effectiveDiscardHighlights = useMemo(
    () => (hints ? discardHighlights : NO_CARDS),
    [discardHighlights, hints],
  );
  const types = useOpponentMeldTypes(d);
  const {composition} = c;
  const knocker = d.knockerSeat;
  const opponentViewKnownCards = useKnownCards(d, hints, d.playerHandSlot);
  const playerViewKnownCards = useKnownCards(d, hints, d.opponentHandSlot);
  const {freeLayoffCards} = composition;
  const layoffOptionCards = useMemo(
    () => new Set(freeLayoffCards),
    [freeLayoffCards],
  );
  const states = useGinCardStates({
    opponentHandDisplay,
    opponentHandSet,
    manualOpponentControl: shown.manualOpponentControl,
    showOpponentCards: shown.showOpponentCards,
    opponentViewKnownCards,
    opponentLayedTypes: types.laid,
    opponentPendingMeldTypes: types.pending,
    showLayoffsInOpponentHand:
      knocker !== null && knocker === d.opponentHandSlot,
    playerHandSet: d.playerHandSet,
    playerMeldCards: d.playerMeldCards,
    playerMelds: d.playerMelds,
    activeMeldIds: composition.stagedMeldIds,
    effectiveDiscardHighlights,
    showLayoffsInPlayerHand: knocker !== null && knocker === d.playerHandSlot,
    playerViewKnownCards,
    discardActions: d.discardActions,
    layoffCards: d.layoffCards,
    shouldShowMeldHints: c.shouldShowMeldHints,
    isAnalysisMode: view.isAnalysisMode,
    canInteract: handlers.canInteract,
    laydownComposeActive: c.composeOnPlayerGrid && c.shouldShowMeldHints,
    stagedDiscard: composition.stagedDiscard,
    pinnedFlash: composition.pinnedFlash,
    handlePlayerCardAction: handlers.normalDiscard,
    handleComposeCard: handlers.composeCard,
    // Analysis cross-seat control: the actor's hand renders on the OPPONENT
    // grid; the compose gesture and staged bands follow it.
    opponentComposeActive: c.composeOnOpponentGrid && c.shouldShowMeldHints,
    composeMelds: c.composeMelds,
    composeStagedMeldIds: composition.stagedMeldIds,
    stagedLayoffCards: c.stagedLayoffCards,
    layoffOptionCards,
  });
  return {...states, opponentHandDisplay};
}

/** Discards a card outside the laydown (draw, then discard). */
export function useNormalDiscard(
  d: GinRummyDerivedState,
  moves: GinMoves,
): (card: string) => void {
  const {discardActions} = d;
  const {handleActionDoubleClick} = moves;
  return useCallback(
    (cardCode: string) => {
      handleActionDoubleClick(discardActions.get(cardCode) ?? null);
    },
    [discardActions, handleActionDoubleClick],
  );
}

/**
 * The laydown prompt, which is the turn line while the hand is laid down
 * (Done sits in the action region, so nothing above the table appears
 * mid-knock).
 */
export function laydownPrompt(
  d: GinRummyDerivedState,
  c: GinCompose,
  done: GinDone,
): string | null {
  if (!c.laydownActive) return null;
  // What stops Done is said where the player looks for the next step.
  if (c.composition.doneBlocker !== null) return c.composition.doneBlocker;
  const forfeit = c.composition.forfeitableLayoffs;
  if (done.skipArmed) {
    return forfeit === null
      ? GIN_LAYOFFS_UNCHECKED
      : ginLayoffSkipPrompt(done.forfeitCount, forfeit.points);
  }
  return ginLaydownPrompt(
    d.laydownStage ?? 'melds',
    c.composition.stagedDiscard,
    done.forfeitCount,
    c.stagedLayoffCards.size,
    c.composition.freeLayoffCards.length,
  );
}

/**
 * The Deadwood line while the viewer's own hand is laid down after the knock
 * (the knocker's discard, then melds; the defender's lay-offs, then melds):
 * the count the composition leaves, which is what the engine counts after
 * Done. The engine's own figure in the Knock and Layoff phases counts every
 * card not yet laid (74 for a gin hand), and a staged lay-off leaves it
 * stale, so it is never shown here.
 */
export function composedDeadwood(
  d: GinRummyDerivedState,
  c: GinCompose,
): number | null {
  if (!c.composeOnPlayerGrid) return null;
  return ginLaydownDeadwood(
    d.laydownStage,
    d.playerHand,
    c.composition.stagedCards,
    c.composition.stagedDiscard,
  );
}

/**
 * The deadwood the composition leaves at a laydown sub-stage: the hand,
 * less the staged discard (in the knocker's discard sub-stage, the best
 * discard while none is picked), the staged melds and the staged lay-offs.
 * Null outside a laydown.
 */
export function ginLaydownDeadwood(
  stage: GinRummyDerivedState['laydownStage'],
  hand: readonly string[],
  staged: {
    readonly melded: ReadonlySet<string>;
    readonly layoffs: ReadonlySet<string>;
  },
  stagedDiscard: string | null,
): number | null {
  if (stage === null) return null;
  return ginComposedDeadwood(hand, staged.melded, staged.layoffs, {
    staged: stagedDiscard,
    pending: stage === 'discard',
  });
}

/**
 * Dragging a card onto the knocker's meld it extends lays it off. The
 * actor's hand is the near zone, or the far one in analysis cross-seat
 * control; the knocker's spread is the other.
 */
export function layoffDropOf(
  d: GinRummyDerivedState,
  c: GinCompose,
): LayoffDrop | null {
  const knocker = d.knockerSeat;
  const melds = knocker === null ? [] : (d.layedMeldsBySeat.get(knocker) ?? []);
  if (!c.laydownActive || d.laydownStage !== 'layoffs' || melds.length === 0) {
    return null;
  }
  if (!c.composeOnPlayerGrid && !c.composeOnOpponentGrid) return null;
  const {layableCards, stageLayoff} = c.composition;
  return {
    actorZone: c.composeOnPlayerGrid ? 'player' : 'opponent',
    canLayOff: card => layableCards.has(card),
    fits: (card, onto) =>
      ginLayoffDropFits(card, onto, melds, c.stagedLayoffCards),
    layOff: card => {
      stageLayoff(card);
    },
  };
}

/**
 * The finished hand's accounting. 'You' is the seat a player holds in game
 * mode; any other seat is 'Opponent', or 'Player N' in analysis. Seats are
 * named by where they sit, never by controller.
 */
export function useHandOutcome(
  s: SessionContextValue,
  d: GinRummyDerivedState,
  view: GinView,
): {value: BoardHandResult | null; error: string | null} {
  const youSeat = s.session.mode === 'game' ? view.viewerSeatIndex : null;
  const {structRecord, returns} = d;
  return useMemo(() => {
    const name: GinSeatNamer = seat => {
      if (seat === youSeat) return 'You';
      return youSeat === null ? `Player ${String(seat)}` : 'Opponent';
    };
    const figureName: GinSeatNamer = seat =>
      seat === youSeat ? 'You' : `Player ${String(seat)}`;
    try {
      const result = ginHandResult(structRecord, returns);
      return {
        value:
          result === null
            ? null
            : {
                result,
                figure: ginResultFigure(result, youSeat, figureName),
                name,
              },
        error: null,
      };
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      console.error(message);
      return {value: null, error: message};
    }
  }, [structRecord, returns, youSeat]);
}
