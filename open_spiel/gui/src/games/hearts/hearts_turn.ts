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

// Whose turn it is at the hearts table and what the viewer may do there:
// who is on move, whether a bot is, whether the viewer may act, and whether
// the hand on move is out of reach (analysis, its cards hidden). Also the
// status line under the table and the board's names for cards and actions.

import {cardLabel} from '../../cards/library/index.ts';
import type {HeartsUIState} from './hooks/use_hearts_state.ts';
import {
  isHumanController,
  type LegalAction,
  type SessionPayload,
  type ViewerContext,
} from './hearts_session.ts';
import {toSharedCard, type CardCode} from './types.ts';

/** "the Queen of Spades, the King of Spades and the Queen of Hearts". */
export function cardList(cards: readonly CardCode[]): string {
  const names = cards.map(c => `the ${cardLabel(toSharedCard(c))}`);
  return names.length < 2
    ? names.join('')
    : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1] ?? ''}`;
}

/** The legal action that plays or passes `card` ("Th" is labelled "10H"). */
export function actionFor(
  legal: readonly LegalAction[],
  card: CardCode,
): number | null {
  const label = card.toUpperCase().replace(/^T/, '10');
  const a = legal.find(
    e =>
      e.cardCode === card ||
      e.label === label ||
      e.label === card.toUpperCase(),
  );
  return a ? a.action : null;
}

/** Whose turn it is, and what the viewer may do. */
export interface HeartsTurn {
  readonly currentPlayerSlot: number;
  readonly currentAutoPlay: boolean;
  readonly waitingOnBot: boolean;
  readonly isGameMode: boolean;
  readonly isAnalysisMode: boolean;
  readonly notMyTurn: boolean;
  readonly notParticipant: boolean;
  readonly canInteract: boolean;
  /** The hand on move is hidden from the analysis viewer. */
  readonly outOfReach: boolean;
  /** The name of the seat on move, when out of reach. */
  readonly onMoveName: string;
}

/** The turn at `session`'s position, for `viewer` (pure). */
export function heartsTurn(
  session: SessionPayload,
  viewer: ViewerContext,
  heartsState: HeartsUIState | null,
  showAllCards: boolean,
  isBusy: boolean,
): HeartsTurn {
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

  // Analysis with the opponents' hands hidden and another seat on move
  // (a bottom seat other than theirs): their hand is backs, so nothing at
  // the table can pass or play for them. Say so, with the way out, rather
  // than leave a table where nothing answers.
  const outOfReach =
    !!heartsState &&
    isAnalysisMode &&
    !showAllCards &&
    !heartsState.isTerminal &&
    !waitingOnBot &&
    currentPlayerSlot >= 0 &&
    heartsState.currentPlayer !== 'bottom' &&
    !(heartsState.phase === 'pass' && heartsState.passDirection === 'none');
  const onMoveName = outOfReach
    ? (heartsState.players.find(p => p.position === heartsState.currentPlayer)
        ?.name ?? `Player ${String(currentPlayerSlot)}`)
    : '';
  return {
    currentPlayerSlot,
    currentAutoPlay,
    waitingOnBot,
    isGameMode,
    isAnalysisMode,
    notMyTurn,
    notParticipant,
    canInteract,
    outOfReach,
    onMoveName,
  };
}

/**
 * The status message under the table (not used during the pass phase or at
 * the end: those have their own UI).
 */
export function heartsStatusMessage(
  heartsState: HeartsUIState | null,
  turn: HeartsTurn,
  sending: CardCode | null,
): string {
  if (!heartsState) return '';
  if (heartsState.phase === 'pass') return '';
  if (heartsState.isTerminal) return ''; // Game over UI shown in center
  if (sending) return `Playing ${cardList([sending])}...`;
  if (turn.waitingOnBot) {
    return turn.currentAutoPlay ? 'Bot is thinking...' : 'Waiting for bot...';
  }
  if (turn.notParticipant) return 'Spectating';
  if (turn.notMyTurn) return 'Waiting for your turn...';
  return '';
}
