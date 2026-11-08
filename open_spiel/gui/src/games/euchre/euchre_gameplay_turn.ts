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
// The euchre Gameplay panel's turn and moves (euchre_gameplay_panel.tsx):
// whose turn it is and whether the viewer may act, which cards it may play,
// the presses that play a card or a call, the plays still landing, and the
// panel's status and phase lines.

import {useCallback, useLayoutEffect, useMemo} from 'react';

import {
  motionDuration,
  useCardStyle,
  type TrickCollectState,
} from '../../cards/library/index.ts';
import {
  animationScale,
  scaleMs,
  useAnimationScale,
} from '../../motion/index.ts';
import {PLAY_REST_MS, useStagedPlays} from '../shared/use_staged_plays.ts';
import {isHumanController, type SessionContextValue} from './euchre_session.ts';
import {COMPASS_NAME} from './hand_result.ts';
import {euchreHeldView, type EuchreView} from './held_trick.ts';
import type {EuchreUIState} from './hooks/use_euchre_state.ts';
import {trickKey} from './trick_seats.ts';
import type {CardCode, PlayerPosition, TrickCard} from './types.ts';

const NO_PLAYS: readonly string[] = [];

/** A finished trick on show, as the table holds and collects it. */
export type EuchreTrickCollect = TrickCollectState<CardCode, PlayerPosition>;

/** Whose turn it is, and whether the viewer may act. */
export interface EuchreTurn {
  currentAutoPlay: boolean;
  waitingOnBot: boolean;
  isAnalysisMode: boolean;
  notMyTurn: boolean;
  notParticipant: boolean;
  canInteract: boolean;
}

/** Whose turn it is in the session, and whether the viewer may act. */
export function useEuchreTurn(s: SessionContextValue): EuchreTurn {
  const {session, isBusy, viewer} = s;
  // Get current player info
  const players = session.players;
  const currentPlayerSlot = session.state.currentPlayer;
  const currentPlayerEntry =
    currentPlayerSlot >= 0
      ? players.find(p => p.slot === currentPlayerSlot)
      : undefined;
  const currentController = currentPlayerEntry?.controller;
  const currentAutoPlay = Boolean(currentPlayerEntry?.autoPlay);

  const isHumanTurn =
    currentPlayerSlot >= 0
      ? isHumanController(currentController, currentPlayerSlot)
      : false;
  const waitingOnBot =
    !session.state.isTerminal && currentPlayerSlot >= 0 && !isHumanTurn;

  // In game mode, only allow actions when it's the viewer's turn
  const isGameMode = session.mode === 'game';
  const isAnalysisMode = session.mode === 'analysis';
  const notMyTurn = isGameMode && !viewer.isActiveSeat;
  const notParticipant = isGameMode && !viewer.isParticipant;
  const turnLocked = notMyTurn || notParticipant;

  const canInteract =
    !session.state.isTerminal && !waitingOnBot && !isBusy && !turnLocked;
  return {
    currentAutoPlay,
    waitingOnBot,
    isAnalysisMode,
    notMyTurn,
    notParticipant,
    canInteract,
  };
}

/** The cards the viewer may play now, and the presses that move. */
export interface EuchreMoves {
  playableCardsSet: Set<string>;
  handleCardClick: (card: CardCode) => void;
  handleBiddingAction: (actionId: number) => void;
}

/**
 * The viewer's moves. `tableBehind`: a finished trick waits its turn on the
 * table (useTrickCollect pending), so no card is played until it has caught
 * up, and a move is never made on a stale table.
 */
export function useEuchreMoves(
  s: SessionContextValue,
  euchreState: EuchreUIState | null,
  turn: EuchreTurn,
  tableBehind: boolean,
): EuchreMoves {
  const {session, submitAction} = s;
  const {canInteract, isAnalysisMode} = turn;
  // Get playable cards from the state
  const playableCardsSet = useMemo(() => {
    if (tableBehind || (!canInteract && !isAnalysisMode)) {
      return new Set<string>();
    }
    return euchreState?.playableCards ?? new Set<string>();
  }, [tableBehind, canInteract, isAnalysisMode, euchreState?.playableCards]);

  // Handle card click (only during play phase)
  const handleCardClick = useCallback(
    (card: CardCode) => {
      if (!euchreState) return;
      if (euchreState.phase !== 'playing') return;
      if (tableBehind || (!canInteract && !isAnalysisMode)) return;

      // Find the action for this card
      const action = session.state.legalActions.find(a => a.cardCode === card);

      if (action) {
        submitAction(action.action);
      }
    },
    [
      session,
      euchreState,
      canInteract,
      isAnalysisMode,
      submitAction,
      tableBehind,
    ],
  );

  // Handle bidding action
  const handleBiddingAction = useCallback(
    (actionId: number) => {
      if (!canInteract && !isAnalysisMode) return;
      submitAction(actionId);
    },
    [canInteract, isAnalysisMode, submitAction],
  );
  return {playableCardsSet, handleCardClick, handleBiddingAction};
}

/** The table with a finished trick on show, and its plays still landing. */
export interface EuchreStaged {
  heldView: EuchreView | null;
  shown: number;
  settling: boolean;
  flightMs: number;
  easing: string;
}

/**
 * Plays that arrive in one update (the bots answer at once) join the
 * trick one at a time, each flying from its hand after the one before
 * has landed, at the viewer's Animation speed. `setSettling` hears when the
 * last of them has landed, which the trick hold waits for.
 */
export function useEuchreStaged(
  euchreState: EuchreUIState | null,
  trickCollect: EuchreTrickCollect,
  setSettling: (settling: boolean) => void,
): EuchreStaged {
  const heldView = euchreState
    ? euchreHeldView(euchreState, trickCollect.trick, trickCollect.pending)
    : null;
  const cardMotion = useCardStyle().motion;
  useAnimationScale(); // re-render when the speed changes
  const flightMs = motionDuration(cardMotion);
  const staged = useStagedPlays(
    heldView ? heldView.trickPile.map(trickKey) : NO_PLAYS,
    {
      legMs: flightMs,
      restMs: scaleMs(
        PLAY_REST_MS,
        animationScale({reducedMotion: cardMotion.reducedMotion}),
      ),
    },
  );
  useLayoutEffect(() => {
    setSettling(staged.settling);
  }, [staged.settling, setSettling]);
  return {
    heldView,
    shown: staged.shown,
    settling: staged.settling,
    flightMs,
    easing: cardMotion.easing,
  };
}

/**
 * The last staged play of a held trick while it is still in the air (the
 * trick's winner waits for it to land); null otherwise.
 */
export function flyingCard(
  view: EuchreView,
  settling: boolean,
  trickCollect: EuchreTrickCollect,
): TrickCard | null {
  return settling && trickCollect.trick !== null
    ? (view.trickPile.at(-1) ?? null)
    : null;
}

/**
 * Build status message. `nextStaged` is the card the table is waiting on: the
 * next staged play, or the last one while it is still in the air.
 */
export function statusLine(
  euchreState: EuchreUIState,
  view: EuchreView,
  trickCollect: EuchreTrickCollect,
  nextStaged: TrickCard | null,
  turn: EuchreTurn,
): string {
  // A card still to land comes first: the trick's winner is named only once
  // the trick's last card is down, never under cards still flying in.
  if (nextStaged) {
    return `${COMPASS_NAME[euchreState.compassAt[nextStaged.position]]} to play`;
  }
  // While a finished trick is held, the line follows the table (the
  // trick's winner), not the live turn underneath.
  if (trickCollect.trick) return view.statusMessage;
  if (euchreState.isTerminal) return '';
  if (turn.waitingOnBot) {
    return turn.currentAutoPlay ? 'Bot is thinking...' : 'Waiting for bot...';
  }
  if (turn.notParticipant) return 'Spectating';
  if (turn.notMyTurn) return 'Waiting for your turn...';
  return euchreState.statusMessage;
}

const PHASE_LABELS: Readonly<Record<string, string>> = {
  dealing: 'Dealing...',
  bidding1: 'Bidding (Round 1)',
  bidding2: 'Bidding (Round 2)',
  alone: 'Go Alone?',
  playing: 'Playing',
  terminal: 'Game Over',
};

/** The phase as the meta line names it. */
export function phaseLabelOf(euchreState: EuchreUIState): string {
  return euchreState.discarding
    ? 'Dealer discards'
    : (PHASE_LABELS[euchreState.phase] ?? 'Unknown');
}

/** The trick being collected, for the trick pile; null while none is. */
export function collectOf(
  trickCollect: EuchreTrickCollect,
): {key: string; winner: PlayerPosition; durationMs: number} | null {
  return trickCollect.phase === 'collect' &&
    trickCollect.trick &&
    trickCollect.key
    ? {
        key: trickCollect.key,
        winner: trickCollect.trick.winner,
        durationMs: trickCollect.collectMs,
      }
    : null;
}
