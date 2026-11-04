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

// The cell state of every card of both grids: the opponent's (or, while the
// knocker's spread is shown, the knocker's) and the player's.

import {useMemo} from 'react';

import type {CardCellState} from '../components/index.ts';
import {useDeck, type DeckConfig} from '../deck_context.tsx';
import type {MeldType, PlayerMeld} from '../types.ts';

/** What the opponent's grid is drawn from. */
export interface OpponentCardsInput {
  opponentHandDisplay: readonly string[];
  opponentHandSet: ReadonlySet<string>;
  manualOpponentControl: boolean;
  showOpponentCards: boolean;
  opponentViewKnownCards: ReadonlySet<string>;
  opponentLayedTypes: ReadonlyMap<string, MeldType>;
  opponentPendingMeldTypes: ReadonlyMap<string, MeldType>;
  showLayoffsInOpponentHand: boolean;
  discardActions: ReadonlyMap<string, number>;
  layoffCards: ReadonlySet<string>;
  laydownComposeActive: boolean;
  stagedDiscard: string | null;
  pinnedFlash: string | null;
  /**
   * Analysis cross-seat control: when the ACTING seat's hand renders on the
   * opponent grid, the compose gesture and staged-meld bands follow it
   * there. `composeMelds`/`composeStagedMeldIds` are the actor's
   * composition (they equal playerMelds/activeMeldIds when the viewed seat
   * is the actor).
   */
  opponentComposeActive?: boolean | undefined;
  composeMelds?: readonly PlayerMeld[] | undefined;
  composeStagedMeldIds?: ReadonlySet<string> | undefined;
  /**
   * Cards of STAGED (uncommitted) layoffs. Drawn like the physical gesture:
   * the card leaves the defender's hand (slot) and sits on the knocker's
   * grid as a blue 'layoff' ghost; double-click there takes it back
   * (chain-checked). Bands are for melds only, never layoffs.
   */
  stagedLayoffCards?: ReadonlySet<string> | undefined;
  /**
   * The actor's deadwood cards that can be laid off now: lit 'layoff' in the
   * hand, and their double-click lays them off.
   */
  layoffOptionCards?: ReadonlySet<string> | undefined;
  handlePlayerCardAction: (cardCode: string) => void;
  handleComposeCard: (cardCode: string) => void;
}

/** What the player's grid is drawn from. */
export interface PlayerCardsInput {
  playerHandSet: ReadonlySet<string>;
  playerMeldCards: ReadonlySet<string>;
  playerMelds: readonly PlayerMeld[];
  activeMeldIds: ReadonlySet<string>;
  effectiveDiscardHighlights: ReadonlySet<string>;
  showLayoffsInPlayerHand: boolean;
  playerViewKnownCards: ReadonlySet<string>;
  discardActions: ReadonlyMap<string, number>;
  layoffCards: ReadonlySet<string>;
  shouldShowMeldHints: boolean;
  isAnalysisMode: boolean;
  canInteract: boolean;
  /**
   * The single-stage laydown compose. When active, EVERY player hand card is
   * interactive (stage a meld/layoff, pick the discard, or peel a staged
   * layoff) via `handleComposeCard`. Outside the compose, the normal
   * draw-then-discard double-click (`handlePlayerCardAction`) applies.
   */
  laydownComposeActive: boolean;
  stagedDiscard: string | null;
  pinnedFlash: string | null;
  opponentComposeActive?: boolean | undefined;
  stagedLayoffCards?: ReadonlySet<string> | undefined;
  layoffOptionCards?: ReadonlySet<string> | undefined;
  handlePlayerCardAction: (cardCode: string) => void;
  handleComposeCard: (cardCode: string) => void;
}

/** What both grids are drawn from. */
export type GinCardStatesInput = OpponentCardsInput &
  PlayerCardsInput & {playerMeldOptions?: ReadonlyMap<string, number[]>};

/** Both grids' cell states, by card code. */
export interface GinCardStatesResult {
  opponentCardStates: Map<string, CardCellState>;
  playerCardStates: Map<string, CardCellState>;
}

function eachCode(
  deck: DeckConfig,
  cell: (code: string) => CardCellState,
): Map<string, CardCellState> {
  const states = new Map<string, CardCellState>();
  for (const suit of deck.suitRows) {
    for (const rank of deck.rankCodes) {
      const code = `${rank}${suit.code}`;
      states.set(code, cell(code));
    }
  }
  return states;
}

/** A staged lay-off, marked staged (never washed out); double-click takes it back. */
function stagedLayoffCell(
  code: string,
  pinnedFlash: string | null,
  handleComposeCard: (cardCode: string) => void,
): CardCellState {
  return {
    type: 'card',
    isGhost: true,
    isStagedLayoff: true,
    highlight: pinnedFlash === code ? 'alert' : 'layoff',
    onDoubleClick: () => {
      handleComposeCard(code);
    },
    action: 'take-back',
  };
}

function discardHandler(
  available: boolean,
  code: string,
  handle: (cardCode: string) => void,
): Pick<CardCellState, 'onDoubleClick' | 'action'> {
  if (!available) return {};
  return {
    onDoubleClick: () => {
      handle(code);
    },
    action: 'discard',
  };
}

/** A card the opponent's grid names without holding. */
function opponentEmptyCell(
  i: OpponentCardsInput,
  code: string,
  actionable: boolean,
): CardCellState {
  if (i.stagedLayoffCards?.has(code) === true && i.laydownComposeActive) {
    // Viewing the defender: this grid is the KNOCKER's, and the staged
    // layoff sits here.
    return stagedLayoffCell(code, i.pinnedFlash, i.handleComposeCard);
  }
  if (i.showOpponentCards && i.opponentViewKnownCards.has(code)) {
    return {
      type: 'card',
      mark: 'known',
      isGhost: true,
      highlight: 'alert',
      ...discardHandler(actionable, code, i.handlePlayerCardAction),
    };
  }
  if (
    i.showOpponentCards &&
    i.layoffCards.has(code) &&
    i.showLayoffsInOpponentHand
  ) {
    return {
      type: 'card',
      isGhost: true,
      highlight: 'layoff',
      ...discardHandler(actionable, code, i.handlePlayerCardAction),
    };
  }
  return {type: 'slot'};
}

/**
 * Analysis cross-seat control: the ACTING seat's hand is on this grid, so
 * compose gestures and staged bands follow it. Bands show only staged MELDS
 * (3+ cards) and laid locked melds, never the analysis suggestions: what is
 * banded is exactly what Done will play.
 */
function opponentComposeCell(
  i: OpponentCardsInput,
  code: string,
): CardCellState {
  // The staged layoff has jumped to the knocker's spread: its cell is empty.
  if (i.stagedLayoffCards?.has(code) === true) return {type: 'slot'};
  const staged = i.composeStagedMeldIds;
  const stagedMeld =
    staged === undefined
      ? undefined
      : (i.composeMelds ?? []).find(
          meld =>
            meld.cards.length >= 3 &&
            staged.has(meld.id) &&
            meld.cards.includes(code),
        );
  const lockedType = i.opponentLayedTypes.get(code);
  const canLayOff = i.layoffOptionCards?.has(code) === true;
  return {
    type: 'card',
    isActionable: true,
    isMelded: stagedMeld !== undefined || lockedType !== undefined,
    meldType: stagedMeld?.type ?? lockedType,
    highlight:
      i.pinnedFlash === code
        ? 'alert'
        : i.stagedDiscard === code
          ? 'selected'
          : canLayOff
            ? 'layoff'
            : null,
    onDoubleClick: () => {
      i.handleComposeCard(code);
    },
    action: canLayOff ? 'lay-off' : 'compose',
  };
}

function opponentCell(i: OpponentCardsInput, code: string): CardCellState {
  const actionable =
    i.manualOpponentControl &&
    i.opponentHandSet.has(code) &&
    i.discardActions.has(code);
  if (!i.opponentHandDisplay.includes(code)) {
    return opponentEmptyCell(i, code, actionable);
  }
  const isLayoffCard = i.showOpponentCards && i.layoffCards.has(code);
  if (isLayoffCard && !i.showLayoffsInOpponentHand) return {type: 'slot'};
  if (i.opponentComposeActive === true) return opponentComposeCell(i, code);
  const known = i.showOpponentCards && i.opponentViewKnownCards.has(code);
  const meldType = i.showOpponentCards
    ? (i.opponentLayedTypes.get(code) ?? i.opponentPendingMeldTypes.get(code))
    : undefined;
  return {
    type: 'card',
    isActionable: actionable,
    isGhost: known,
    highlight: known ? 'alert' : isLayoffCard ? 'layoff' : null,
    isMelded: meldType !== undefined,
    meldType,
    ...discardHandler(actionable, code, i.handlePlayerCardAction),
  };
}

/** A card the player holds. */
function playerHeldCell(i: PlayerCardsInput, code: string): CardCellState {
  const isLayoffCard = i.layoffCards.has(code);
  // The staged layoff has jumped to the knocker's spread: its cell is empty.
  if (i.stagedLayoffCards?.has(code) === true && i.laydownComposeActive) {
    return {type: 'slot'};
  }
  if (isLayoffCard && !i.showLayoffsInPlayerHand) return {type: 'slot'};
  // Bands are for MELDS only (3+ cards); staged layoffs are ghosts on the
  // knocker's grid instead.
  const activeMeld =
    i.shouldShowMeldHints && i.playerMeldCards.has(code)
      ? i.playerMelds.find(
          meld =>
            meld.cards.length >= 3 &&
            i.activeMeldIds.has(meld.id) &&
            meld.cards.includes(code),
        )
      : undefined;
  const canLayOff =
    i.laydownComposeActive && i.layoffOptionCards?.has(code) === true;
  let handler: Pick<CardCellState, 'onDoubleClick' | 'action'>;
  if (i.laydownComposeActive) {
    // Every hand card is interactive during the compose: stage or cycle a
    // meld, pick the discard, lay off, or peel a staged layoff (the handler
    // routes by sub-stage).
    handler = {
      onDoubleClick: () => {
        i.handleComposeCard(code);
      },
      action: canLayOff ? 'lay-off' : 'compose',
    };
  } else {
    const canDiscard =
      (i.isAnalysisMode || i.canInteract) && i.discardActions.has(code);
    handler = discardHandler(canDiscard, code, i.handlePlayerCardAction);
  }
  return {
    type: 'card',
    isMelded: activeMeld !== undefined,
    meldType: activeMeld?.type,
    highlight:
      i.pinnedFlash === code
        ? 'alert'
        : i.stagedDiscard === code
          ? 'selected'
          : canLayOff || (isLayoffCard && i.showLayoffsInPlayerHand)
            ? 'layoff'
            : null,
    ...handler,
  };
}

function playerCell(i: PlayerCardsInput, code: string): CardCellState {
  if (i.playerHandSet.has(code)) return playerHeldCell(i, code);
  if (
    i.stagedLayoffCards?.has(code) === true &&
    i.opponentComposeActive === true
  ) {
    // Cross-seat compose (viewing the knocker while the defender lays
    // down): the defender's staged layoff sits on THIS grid.
    return stagedLayoffCell(code, i.pinnedFlash, i.handleComposeCard);
  }
  if (i.layoffCards.has(code) && i.showLayoffsInPlayerHand) {
    return {type: 'card', isGhost: true, highlight: 'layoff'};
  }
  if (i.playerViewKnownCards.has(code)) {
    return {type: 'card', mark: 'known', isGhost: true, highlight: 'alert'};
  }
  if (i.effectiveDiscardHighlights.has(code)) {
    return {type: 'card', mark: 'dead', isGhost: true};
  }
  return {type: 'slot'};
}

/** The opponent grid's cell states (a pure function of its input). */
export function opponentCardStatesOf(
  deck: DeckConfig,
  input: OpponentCardsInput,
): Map<string, CardCellState> {
  return eachCode(deck, code => opponentCell(input, code));
}

/** The player grid's cell states (a pure function of its input). */
export function playerCardStatesOf(
  deck: DeckConfig,
  input: PlayerCardsInput,
): Map<string, CardCellState> {
  return eachCode(deck, code => playerCell(input, code));
}

function useOpponentCardStates(
  deck: DeckConfig,
  i: OpponentCardsInput,
): Map<string, CardCellState> {
  const {opponentHandDisplay, opponentHandSet, manualOpponentControl} = i;
  const {showOpponentCards, opponentViewKnownCards, opponentLayedTypes} = i;
  const {opponentPendingMeldTypes, showLayoffsInOpponentHand} = i;
  const {discardActions, layoffCards, laydownComposeActive} = i;
  const {stagedDiscard, pinnedFlash, opponentComposeActive, composeMelds} = i;
  const {composeStagedMeldIds, stagedLayoffCards, layoffOptionCards} = i;
  const {handlePlayerCardAction, handleComposeCard} = i;
  return useMemo(
    () =>
      opponentCardStatesOf(deck, {
        opponentHandDisplay,
        opponentHandSet,
        manualOpponentControl,
        showOpponentCards,
        opponentViewKnownCards,
        opponentLayedTypes,
        opponentPendingMeldTypes,
        showLayoffsInOpponentHand,
        discardActions,
        layoffCards,
        laydownComposeActive,
        stagedDiscard,
        pinnedFlash,
        opponentComposeActive,
        composeMelds,
        composeStagedMeldIds,
        stagedLayoffCards,
        layoffOptionCards,
        handlePlayerCardAction,
        handleComposeCard,
      }),
    [
      deck,
      opponentHandDisplay,
      opponentHandSet,
      manualOpponentControl,
      showOpponentCards,
      opponentViewKnownCards,
      opponentLayedTypes,
      opponentPendingMeldTypes,
      showLayoffsInOpponentHand,
      discardActions,
      layoffCards,
      laydownComposeActive,
      stagedDiscard,
      pinnedFlash,
      opponentComposeActive,
      composeMelds,
      composeStagedMeldIds,
      stagedLayoffCards,
      layoffOptionCards,
      handlePlayerCardAction,
      handleComposeCard,
    ],
  );
}

function usePlayerCardStates(
  deck: DeckConfig,
  i: PlayerCardsInput,
): Map<string, CardCellState> {
  const {playerHandSet, playerMeldCards, playerMelds, activeMeldIds} = i;
  const {effectiveDiscardHighlights, showLayoffsInPlayerHand} = i;
  const {playerViewKnownCards, discardActions, layoffCards} = i;
  const {shouldShowMeldHints, isAnalysisMode, canInteract} = i;
  const {laydownComposeActive, stagedDiscard, pinnedFlash} = i;
  const {opponentComposeActive, stagedLayoffCards, layoffOptionCards} = i;
  const {handlePlayerCardAction, handleComposeCard} = i;
  return useMemo(
    () =>
      playerCardStatesOf(deck, {
        playerHandSet,
        playerMeldCards,
        playerMelds,
        activeMeldIds,
        effectiveDiscardHighlights,
        showLayoffsInPlayerHand,
        playerViewKnownCards,
        discardActions,
        layoffCards,
        shouldShowMeldHints,
        isAnalysisMode,
        canInteract,
        laydownComposeActive,
        stagedDiscard,
        pinnedFlash,
        opponentComposeActive,
        stagedLayoffCards,
        layoffOptionCards,
        handlePlayerCardAction,
        handleComposeCard,
      }),
    [
      deck,
      playerHandSet,
      playerMeldCards,
      playerMelds,
      activeMeldIds,
      effectiveDiscardHighlights,
      showLayoffsInPlayerHand,
      playerViewKnownCards,
      discardActions,
      layoffCards,
      shouldShowMeldHints,
      isAnalysisMode,
      canInteract,
      laydownComposeActive,
      stagedDiscard,
      pinnedFlash,
      opponentComposeActive,
      stagedLayoffCards,
      layoffOptionCards,
      handlePlayerCardAction,
      handleComposeCard,
    ],
  );
}

/**
 * Computes card cell states for both the opponent's and the player's grid.
 * Each is recomputed only when its input changes, field by field.
 */
export function useGinCardStates(
  input: GinCardStatesInput,
): GinCardStatesResult {
  const deck = useDeck();
  return {
    opponentCardStates: useOpponentCardStates(deck, input),
    playerCardStates: usePlayerCardStates(deck, input),
  };
}
