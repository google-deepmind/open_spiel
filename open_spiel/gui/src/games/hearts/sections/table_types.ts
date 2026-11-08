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

// The hearts table's props and the pass's shape (table_layout.tsx).
import type {ReactNode} from 'react';

import type {TrickLayout} from '../../../cards/library/index.ts';
import type {HeartsCardSize, HeartsHandShape} from '../hearts_layout.ts';
import type {
  CardCode,
  GamePhase,
  PassDirection,
  Player,
  PlayerPosition,
  TrickCard,
} from '../types.ts';

/** A card size setting of the table. */
export type CardSize = HeartsCardSize;

/** The hearts table's props. */
export interface TableLayoutProps {
  /** Array of 4 players */
  players: Player[];
  /** Cards in the current trick */
  trickPile: TrickCard[];
  /** Current game phase */
  phase: GamePhase;
  /** Pass direction in pass phase */
  passDirection: PassDirection;
  /** Which player's turn it is; null once the game is over (no seat is). */
  currentPlayer: PlayerPosition | null;
  /** Whether the game has ended */
  isTerminal?: boolean;
  /** Final rewards/scores when game ends */
  rewards?: number[];
  /** Whether to show opponent cards face-up */
  showOpponentCards?: boolean;
  /** Show all cards can be switched here (analysis): the table is laid out
   *  once for the opponents' backs and their faces alike (HeartsDress
   *  'hidden' / 'open'), so switching it moves nothing. Off (play), hidden
   *  opponents draw the table sized for backs ('closed'). */
  canShowAll?: boolean;
  /** Whether all hands are interactive (analysis mode) */
  allHandsInteractive?: boolean;
  /** Set of selected card codes (for pass phase) */
  selectedCards?: Set<string>;
  /** Set of playable card codes (for play phase) */
  playableCards?: Set<string>;
  /** A card chosen by click, tap, Enter, Space, or a drag onto the trick. */
  onCardClick?: (card: CardCode) => void;
  /** Callback when pass button is clicked */
  onPassCards?: () => void;
  /** Your card on its way to the server: it stays lifted, out of the hand,
   *  and its tooltip says it is being played, until the move lands. Nothing
   *  else in the hand changes: no card is dimmed. */
  sendingCard?: CardCode | null;
  /** Your three cards on their way to the server: they stay lifted and the
   *  pass button says it is passing. */
  passPending?: boolean;
  /** The pass that has left your hand: your cards fly to the seat they went
   *  to and turn face down there, and the cards you received fly in from the
   *  seat that passed them (heartsItems). */
  passFlight?: PassFlight | null;
  /** Largest card size; the table fits its container up to it. */
  cardSize?: CardSize;
  /** Your hand's shape: a straight fan (default) or an arc. */
  handShape?: HeartsHandShape;
  /** The trick: each card toward its player, or overlapping in the middle
   *  (cards/library trick_area). Omitted, the table's default
   *  (DEFAULT_HEARTS_TRICK_LAYOUT: toward). */
  trickLayout?: TrickLayout;
  /** Registered deck id; omitted, the app's CardDeckProvider decides. */
  deck?: string;
  /** A finished trick being collected (cards/library trick_collect): its
   *  cards, which are `trickPile`, slide toward the winner's hand and fade. */
  collect?: {key: string; winner: PlayerPosition; durationMs: number} | null;
  /** Custom center content (replaces trick pile) */
  centerContent?: ReactNode;
  /** Status message to display */
  message?: string;
  /** Additional CSS class */
  className?: string;
}

/** Cards passed per round. */
export const PASS_COUNT = 3;

/** A pass that has left your hand, for the flight (heartsItems). */
export interface PassFlight {
  /** The seat your cards went to, and the seat that passed to you (the same
   *  seat when passing across). */
  to: PlayerPosition;
  from: PlayerPosition;
  /** The cards you passed, and those you received (none until every seat
   *  has passed). */
  sent: readonly CardCode[];
  received: readonly CardCode[];
}

/** The seat you pass to and the seat that passes to you: passing left goes
 *  to the next seat clockwise (yours is the bottom, the left seat is next). */
export function passSeats(
  direction: PassDirection,
): {to: PlayerPosition; from: PlayerPosition} | null {
  switch (direction) {
    case 'left':
      return {to: 'left', from: 'right'};
    case 'right':
      return {to: 'right', from: 'left'};
    case 'across':
      return {to: 'top', from: 'top'};
    case 'none':
      return null;
    default:
      throw new Error(
        `hearts: unknown pass direction ${JSON.stringify(direction)}`,
      );
  }
}

/** What the items and the policy read of the props. */
export type SeatState = Pick<
  TableLayoutProps,
  | 'players'
  | 'trickPile'
  | 'phase'
  | 'currentPlayer'
  | 'showOpponentCards'
  | 'allHandsInteractive'
  | 'selectedCards'
  | 'playableCards'
  | 'onCardClick'
  | 'sendingCard'
  | 'passFlight'
>;
