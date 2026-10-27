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

// The board's last-turn staging. When a turn's draw and discard arrive
// together, the board draws it leg by leg (useGinTurnStaging): the previous
// seat's discard landing on the pile ('prior', when it came in the same
// update: the viewer's own discard, answered by the bot at once), then the
// drawn card in the hand ('drawn'), then the final frame. Each leg lands and
// rests before the next leaves, at the viewer's Animation speed; Instant and
// reduced motion show the final frame and its marks at once.
//
// No move is sent from a staged frame: until the final frame is drawn, the
// board has not shown the pile the viewer chooses from, and a draw would be
// blind. A stock draw or card action during staging instead skips the turn
// to its final frame and sends nothing, so the input is never dead and
// motion never holds the viewer up. Reordering the hand stays live.

/* eslint-disable react-hooks/refs -- the board reads the frame it last drew
   (the deal on screen, the grid it last rested on) while it renders the
   next; no state holds what is on the screen. */

import {
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from 'react';

import {
  motionDuration,
  useCardStyle,
  type TableItem,
} from '../../cards/library/index.ts';
import {animationScale, scaleMs} from '../../motion/index.ts';
import type {CardCellState} from './components/card_state.ts';
import type {GinRummyBoardSurfaceProps} from './gin_board_parts.ts';
import {cardUnder, cuedItemIds, type GinBoardCue} from './gin_move_cues.ts';
import {settledStartFits} from './gin_placement.ts';
import {GIN_SLOT} from './gin_rummy_layout.ts';
import {GIN_STATUS} from './gin_turn_line.ts';
import {
  GIN_REST_MS,
  useGinTurnStaging,
  type GinStagePhase,
} from './hooks/use_gin_turn_staging.ts';
import {normaliseCardCode} from './types.ts';

const NO_PILE: readonly string[] = [];

/** What the board draws for the frame on show. */
export interface BoardFrame {
  /** The staged frame, or null for the final one. */
  phase: GinStagePhase | null;
  staging: boolean;
  /** Skips the staged turn to its final frame. */
  settle: () => void;
  cue: GinBoardCue | null;
  cued: {stock: string | null; taken: string | null; discard: string | null};
  /** The upcard a face-down hand took, face up at its end until next turn. */
  takenFaceUp: string | null;
  shownDiscard: string | null;
  shownStockCount: number;
  shownOpponentCount: number;
  shownHiddenHandCount: number;
  shownOpponentStates: Map<string, CardCellState>;
  shownStatusText: string;
  drawStock: (() => void) | undefined;
  drawDiscard: (() => void) | undefined;
  /** The deal last drawn; the board writes it after each draw. */
  lastDealRef: RefObject<Record<string, readonly TableItem[]> | null>;
}

interface Cue {
  readonly moveCue: GinBoardCue | null;
  readonly handHidden: boolean;
  readonly cueHandHidden: boolean;
  readonly cueFaceUpOpponent: boolean;
}

function cueOf(p: GinRummyBoardSurfaceProps, handHidden: boolean): Cue {
  const moveCue = p.moveCue ?? null;
  // The opponent's hand may be face down (a count) or face up (its grid); a
  // spectator's near hand is face down.
  const cueHandHidden =
    moveCue?.zone === 'opponent'
      ? !p.showOpponentCards
      : moveCue?.zone === 'player' && handHidden;
  return {
    moveCue,
    handHidden,
    cueHandHidden,
    cueFaceUpOpponent: moveCue?.zone === 'opponent' && p.showOpponentCards,
  };
}

/** The pile's top before the turn drew: the upcard taken, or the card under the discard. */
function priorTopOf(
  moveCue: GinBoardCue | null,
  pile: readonly string[],
): string | null {
  if (moveCue?.discarded === null || moveCue?.discarded === undefined) {
    return null;
  }
  return moveCue.drew === 'upcard'
    ? moveCue.taken
    : cardUnder(pile, moveCue.discarded);
}

function useStagePhase(
  p: GinRummyBoardSurfaceProps,
  c: Cue,
  lastDealRef: RefObject<Record<string, readonly TableItem[]> | null>,
): GinStagePhase | null {
  const style = useCardStyle();
  const {moveCue} = c;
  const canStage =
    moveCue !== null &&
    (c.cueHandHidden || c.cueFaceUpOpponent) &&
    moveCue.discarded !== null &&
    normaliseCardCode(p.discardCard) === moveCue.discarded;
  const priorTop = priorTopOf(moveCue, p.discardPile ?? NO_PILE);
  // The pile as last drawn: if it does not show priorTop yet, the previous
  // seat's discard is new too.
  const lastDeal = lastDealRef.current;
  const lastTop = lastDeal?.[GIN_SLOT.discard]?.[0]?.id ?? null;
  const priorLeg =
    priorTop !== null &&
    lastDeal !== null &&
    normaliseCardCode(lastTop) !== normaliseCardCode(priorTop);
  return useGinTurnStaging(moveCue, canStage, {
    legMs: motionDuration(style.motion),
    restMs: scaleMs(
      GIN_REST_MS,
      animationScale({reducedMotion: style.motion.reducedMotion}),
    ),
    priorLeg,
  });
}

function useOpponentFrame(
  p: GinRummyBoardSurfaceProps,
  c: Cue,
  stagedNow: GinStagePhase | null,
): {phase: GinStagePhase | null; states: Map<string, CardCellState>} {
  // The opponent's face-up grid as last settled: the 'prior' frame draws it
  // (the turn has not drawn yet). It is this turn's start only if the board
  // last rested on the position just before it; after a jump (a scrub, a
  // reset, a turn that coalesced over one still staging) it is another
  // position's, and may hold a card now in the viewer's hand: then the
  // 'prior' frame is drawn as 'drawn' instead.
  const settledRef = useRef(p.opponentCardStates);
  const settled = settledRef.current;
  const discarded = c.moveCue?.discarded ?? null;
  const fits =
    !c.cueFaceUpOpponent ||
    discarded === null ||
    settledStartFits(
      settled,
      p.opponentCardStates,
      discarded,
      c.moveCue?.taken ?? null,
    );
  const phase = stagedNow === 'prior' && !fits ? 'drawn' : stagedNow;
  const staging = phase !== null;
  useLayoutEffect(() => {
    if (!staging) settledRef.current = p.opponentCardStates;
  }, [staging, p.opponentCardStates]);
  const {opponentCardStates} = p;
  const faceUp = c.cueFaceUpOpponent;
  const states = useMemo(() => {
    if (!faceUp || phase === null || discarded === null) {
      return opponentCardStates;
    }
    // 'prior' is drawn only when the settled grid is this turn's start.
    if (phase === 'prior') return settled;
    // 'drawn': the final hand plus the card it is about to discard.
    const held = fits ? settled.get(discarded) : undefined;
    const out = new Map(opponentCardStates);
    out.set(discarded, held?.type === 'card' ? held : {type: 'card'});
    return out;
  }, [faceUp, phase, fits, discarded, opponentCardStates, settled]);
  return {phase, states};
}

function shownCue(
  phase: GinStagePhase | null,
  moveCue: GinBoardCue | null,
): GinBoardCue | null {
  // 'prior': none of this turn's marks yet; 'drawn': the draw's, not the discard's.
  if (phase === 'prior') return null;
  if (phase === 'drawn' && moveCue !== null) {
    return {...moveCue, discarded: null, discardIndex: null};
  }
  return moveCue;
}

function shownDiscardOf(
  p: GinRummyBoardSurfaceProps,
  phase: GinStagePhase | null,
  moveCue: GinBoardCue | null,
): string | null {
  if (phase === 'prior') return priorTopOf(moveCue, p.discardPile ?? NO_PILE);
  if (
    phase === 'drawn' &&
    moveCue?.discarded !== null &&
    moveCue?.discarded !== undefined
  ) {
    return cardUnder(p.discardPile ?? NO_PILE, moveCue.discarded);
  }
  return p.discardCard;
}

function counts(
  p: GinRummyBoardSurfaceProps,
  phase: GinStagePhase | null,
  moveCue: GinBoardCue | null,
): Pick<
  BoardFrame,
  'shownStockCount' | 'shownOpponentCount' | 'shownHiddenHandCount'
> {
  const drawn = phase === 'drawn';
  return {
    shownStockCount:
      phase === 'prior' && moveCue?.drew === 'stock'
        ? p.stockCount + 1
        : p.stockCount,
    shownOpponentCount:
      drawn && moveCue?.zone === 'opponent'
        ? p.opponentHandCount + 1
        : p.opponentHandCount,
    shownHiddenHandCount:
      drawn && moveCue?.zone === 'player'
        ? p.playerHand.length + 1
        : p.playerHand.length,
  };
}

/** The frame the board draws now: staged or final, and what it shows. */
export function useBoardFrame(
  p: GinRummyBoardSurfaceProps,
  handHidden: boolean,
): BoardFrame {
  const c = cueOf(p, handHidden);
  const {moveCue} = c;
  const lastDealRef = useRef<Record<string, readonly TableItem[]> | null>(null);
  const stagedPhase = useStagePhase(p, c, lastDealRef);
  // The skipped turn, by its draw's history index: cleared once that turn's
  // staging is over, so a later hand's turn at the same index stages.
  const [settledDraw, setSettledDraw] = useState<number | null>(null);
  if (stagedPhase === null && settledDraw !== null) setSettledDraw(null);
  const stagedNow =
    stagedPhase !== null && moveCue?.drawIndex !== settledDraw
      ? stagedPhase
      : null;
  const {phase, states} = useOpponentFrame(p, c, stagedNow);
  const staging = phase !== null;
  const settle = (): void => {
    setSettledDraw(moveCue?.drawIndex ?? null);
  };
  const cue = shownCue(phase, moveCue);
  const shown = counts(p, phase, moveCue);
  const shownDiscard = shownDiscardOf(p, phase, moveCue);
  const cued = cuedItemIds(cue, {
    stockCount: shown.shownStockCount,
    discardCard: shownDiscard,
  });
  // Nor does the turn line hand the viewer the move while the opponent's
  // turn is still being drawn: it says whose turn the board shows.
  const shownStatusText =
    staging &&
    moveCue?.zone === 'opponent' &&
    p.statusText.startsWith(GIN_STATUS.yourTurn)
      ? GIN_STATUS.opponentTurn
      : p.statusText;
  return {
    phase,
    staging,
    settle,
    cue,
    cued,
    takenFaceUp: c.cueHandHidden ? cued.taken : null,
    shownDiscard,
    ...shown,
    shownOpponentStates: states,
    shownStatusText,
    drawStock: staging && p.onDrawStock !== undefined ? settle : p.onDrawStock,
    drawDiscard:
      staging && p.onDrawDiscard !== undefined ? settle : p.onDrawDiscard,
    lastDealRef,
  };
}
