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

import {useMemo} from 'react';
import type {CompletedTrick} from '../../../cards/library/index.ts';
import type {SessionPayload} from '../hearts_session.ts';
import type {
  CardCode,
  Player,
  PlayerPosition,
  TrickCard,
  GamePhase,
  PassDirection,
} from '../types.ts';
import {HIDDEN_CARD, isConcealedHand, sortCards} from '../types.ts';

/**
 * OpenSpiel hearts struct format from the server (hearts_adapter.py)
 */
interface HeartsStruct {
  // N, E, S, W -> card codes; a hand the viewer may not see is one null per
  // card.
  hands: Record<string, Array<string | null> | undefined>;
  currentTrick: Array<{card: string; player?: number; position?: string}>;
  completedTricks: number;
  /** The last finished trick and its winner (null before the first). */
  lastTrick?: {
    index: number;
    cards: Array<{card: string; player: number}>;
    winner: string;
  } | null;
  points: Record<string, number | undefined>;
  passDirection: string;
  phase: string;
  currentPlayer: number;
}

/**
 * State derived from OpenSpiel session for Hearts UI
 */
export interface HeartsUIState {
  players: Player[];
  trickPile: TrickCard[];
  phase: GamePhase;
  passDirection: PassDirection;
  currentPlayer: PlayerPosition;
  playableCards: Set<string>;
  isTerminal: boolean;
  statusMessage: string;
  /** The last finished trick, for the trick collect. */
  lastTrick: CompletedTrick<CardCode, PlayerPosition> | null;
}

const COMPASSES = ['N', 'E', 'S', 'W'] as const;

/**
 * Map OpenSpiel player index to UI position
 * OpenSpiel uses: 0=N, 1=E, 2=S, 3=W
 * Our UI uses: bottom (viewer), top (across), left, right
 *
 * We rotate so the viewer's seat is at bottom
 */
function mapPlayerToPosition(
  playerIndex: number,
  viewerSeat: number,
): PlayerPosition {
  // Positions in clockwise order: N=0, E=1, S=2, W=3
  const positions: readonly PlayerPosition[] = [
    'top',
    'right',
    'bottom',
    'left',
  ];

  // Rotate based on viewer seat so viewer is at bottom (index 2)
  const offset = (playerIndex - viewerSeat + 4) % 4;
  // N->top, E->right, S->bottom, W->left when viewerSeat=2 (South)
  // But we want viewer at bottom, so:
  // If viewer is seat 0 (N), then N->bottom, E->left, S->top, W->right
  const position = positions[(2 + offset) % 4];
  if (position === undefined) {
    throw new Error(`hearts: no seat ${String(playerIndex)}`);
  }
  return position;
}

function indexToCompass(index: number): string {
  return COMPASSES[index % 4] ?? '?';
}

function isValidName(n: string | undefined): n is string {
  return !!n && n !== 'terminal' && n.length < 50;
}

/** The four players, rotated so the viewer's seat is at the bottom. */
function playersOf(
  session: SessionPayload,
  struct: HeartsStruct,
  viewerSeat: number,
): Player[] {
  const players: Player[] = [];
  for (let i = 0; i < 4; i++) {
    const compass = indexToCompass(i);
    const hand = struct.hands[compass] ?? [];
    const position = mapPlayerToPosition(i, viewerSeat);
    const score = struct.points[compass] ?? 0;

    // The seat's served label, never its controller: the table names
    // seats by where they sit; who plays them is the Players panel's.
    const label = session.players.find(p => p.slot === i)?.label;
    const name =
      (isValidName(label) ? label : null) ?? `Player ${String(i + 1)}`;

    players.push({
      name,
      hand: isConcealedHand(hand)
        ? hand.map(() => HIDDEN_CARD)
        : sortCards(hand as CardCode[]),
      position,
      score,
      hidden: isConcealedHand(hand),
    });
  }
  return players;
}

/** The trick in play, each card at its player's position. */
function trickPileOf(struct: HeartsStruct, viewerSeat: number): TrickCard[] {
  // Each trick card includes the player index from the server
  return struct.currentTrick.map(entry => {
    // Map the player index to UI position based on viewer seat
    const playerIdx = typeof entry.player === 'number' ? entry.player : 0;
    return {
      card: entry.card as CardCode,
      position: mapPlayerToPosition(playerIdx, viewerSeat),
    };
  });
}

/** Playable cards from legal actions: the server's cardCode, else the label
 *  ("2C" -> "2c", "10H" -> "Th"). */
function playableCardsOf(session: SessionPayload): Set<string> {
  return new Set<string>(
    session.state.legalActions
      .map(a => {
        if (a.cardCode) return a.cardCode;
        if (a.label.length >= 2) {
          let rank = a.label.slice(0, -1);
          const suit = a.label.slice(-1).toLowerCase();
          // Convert "10" to "T"
          if (rank === '10') rank = 'T';
          return rank + suit;
        }
        return null;
      })
      .filter((c): c is string => c !== null),
  );
}

/** The last finished trick, rotated to the viewer's table. */
function lastTrickOf(
  struct: HeartsStruct,
  viewerSeat: number,
): CompletedTrick<CardCode, PlayerPosition> | null {
  const served = struct.lastTrick;
  if (served && !/^[NESW]$/.test(served.winner)) {
    throw new Error(
      `hearts: last trick winner is not a seat: ${JSON.stringify(served.winner)}`,
    );
  }
  return served
    ? {
        index: served.index,
        cards: served.cards.map(c => ({
          card: c.card as CardCode,
          seat: mapPlayerToPosition(c.player, viewerSeat),
        })),
        winner: mapPlayerToPosition('NESW'.indexOf(served.winner), viewerSeat),
      }
    : null;
}

/** The Hearts UI state of `session`'s position (pure). */
export function heartsUIState(
  session: SessionPayload,
  viewerSeat: number,
): HeartsUIState | null {
  const state = session.state;
  const struct = state.struct as HeartsStruct | undefined;
  if (!struct) {
    return null;
  }

  // Determine phase
  const phase: GamePhase = struct.phase === 'passing' ? 'pass' : 'play';

  // Map pass direction
  const passDirectionMap: Record<string, PassDirection | undefined> = {
    left: 'left',
    right: 'right',
    across: 'across',
    none: 'none',
  };
  const passDirection: PassDirection =
    passDirectionMap[struct.passDirection] ?? 'none';

  // Current player position
  const currentPlayerIndex = struct.currentPlayer;
  const currentPlayer =
    currentPlayerIndex >= 0
      ? mapPlayerToPosition(currentPlayerIndex, viewerSeat)
      : 'bottom';

  // Status message
  let statusMessage = '';
  if (state.isTerminal) {
    statusMessage = 'Game Over';
  } else if (phase === 'pass') {
    statusMessage = `Select 3 cards to pass ${passDirection}`;
  } else if (currentPlayerIndex >= 0) {
    const isViewerTurn = currentPlayerIndex === viewerSeat;
    statusMessage = isViewerTurn
      ? 'Your turn to play'
      : `Waiting for ${indexToCompass(currentPlayerIndex)}`;
  }

  return {
    lastTrick: lastTrickOf(struct, viewerSeat),
    players: playersOf(session, struct, viewerSeat),
    trickPile: trickPileOf(struct, viewerSeat),
    phase,
    passDirection,
    currentPlayer,
    playableCards: playableCardsOf(session),
    isTerminal: state.isTerminal,
    statusMessage,
  };
}

/**
 * Hook to transform OpenSpiel session state into Hearts UI state
 */
export function useHeartsState(
  session: SessionPayload,
  viewerSeat: number,
): HeartsUIState | null {
  return useMemo(
    () => heartsUIState(session, viewerSeat),
    [session, viewerSeat],
  );
}
