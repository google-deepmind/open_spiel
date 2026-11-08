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
import type {SessionPayload} from '../euchre_session.ts';
import type {
  CardCode,
  Suit,
  Player,
  PlayerPosition,
  TrickCard,
  GamePhase,
  ActionType,
} from '../types.ts';
import {HIDDEN_CARD, isConcealedHand, parseCard, sortCards} from '../types.ts';
import {
  COMPASS_NAME,
  TEAM_NAME,
  euchreOutcome,
  partnerOf,
  pointsText,
  teamOf,
  type Compass,
  type EuchreOutcome,
  type Team,
} from '../hand_result.ts';

/**
 * OpenSpiel euchre struct format from backend
 */
interface EuchreStruct {
  // N, E, S, W -> card codes; in game mode a hand the viewer may not see is
  // one null per card.
  hands: Record<string, Array<string | null>>;
  upcard: string | null;
  trumpSuit: string | null;
  dealer: string;
  phase: string;
  currentTrick: Array<{card: string; player?: number; position?: string}>;
  /** The last finished trick and its winner (null before the first). */
  lastTrick?: {
    index: number;
    cards: Array<{card: string; player: number}>;
    winner: string;
  } | null;
  tricksWon: {NS: number; EW: number};
  declarer: string | null;
  goingAlone: boolean;
  currentPlayer: number;
  /** The dealer is choosing a card to discard (phase still reads "playing"). */
  discarding?: boolean;
}

/**
 * Augmented legal action from backend
 */
interface AugmentedAction {
  action: number;
  label: string;
  cardCode?: string | null;
  actionType?: ActionType;
  calledSuit?: string | null;
}

/**
 * State derived from OpenSpiel session for Euchre UI
 */
export interface EuchreUIState {
  players: Player[];
  trickPile: TrickCard[];
  phase: GamePhase;
  upcard: CardCode | null;
  trumpSuit: Suit | null;
  dealer: PlayerPosition;
  declarer: PlayerPosition | null;
  goingAlone: boolean;
  currentPlayer: PlayerPosition;
  tricksWon: {NS: number; EW: number};
  /** Each table position's side (the viewer's seat is at the bottom). */
  teamOf: Record<PlayerPosition, Team>;
  /** Each table position's compass seat. */
  compassAt: Record<PlayerPosition, Compass>;
  /** The dealer is discarding: the card actions are a discard, not a play. */
  discarding: boolean;
  playableCards: Set<string>;
  biddingActions: AugmentedAction[];
  isTerminal: boolean;
  statusMessage: string;
  /** The last finished trick, for the trick collect. */
  lastTrick: CompletedTrick<CardCode, PlayerPosition> | null;
  /** The seat playing alone, by compass name ("East"), or null. */
  aloneSeat: string | null;
  /** The lone player's partner, who sits the hand out, or null. */
  sittingOut: PlayerPosition | null;
  /** At game over: what the hand scored and why; null before. */
  outcome: EuchreOutcome | null;
  /** The up-card, ordered up and in the dealer's hand until the discard
   *  (drawn there from the call on); null otherwise. */
  pickup: {dealer: PlayerPosition; card: CardCode} | null;
}

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
  const positions: PlayerPosition[] = ['top', 'right', 'bottom', 'left'];
  const offset = (playerIndex - viewerSeat + 4) % 4;
  return positions[(2 + offset) % 4] ?? 'bottom';
}

function compassToIndex(compass: string): number {
  const map: Record<string, number> = {N: 0, E: 1, S: 2, W: 3};
  return map[compass.toUpperCase()] ?? 0;
}

function indexToCompass(index: number): string {
  const compasses = ['N', 'E', 'S', 'W'];
  return compasses[index % 4] ?? '';
}

const COMPASSES: readonly Compass[] = ['N', 'E', 'S', 'W'];

// The seat's compass word: the seat, never its controller.
const COMPASS_NAMES: Record<string, string> = {
  N: 'North',
  E: 'East',
  S: 'South',
  W: 'West',
};

/** Build players array */
function playersOf(
  struct: EuchreStruct,
  viewerSeat: number,
  handPlayedOut: boolean,
  sittingOutSeat: Compass | null,
): Player[] {
  const trumpSuit = struct.trumpSuit as Suit | null;
  return COMPASSES.map((compass, i) => {
    const hand =
      handPlayedOut && compass !== sittingOutSeat
        ? []
        : (struct.hands[compass] ?? []);
    const position = mapPlayerToPosition(i, viewerSeat);
    const name = COMPASS_NAMES[compass] ?? compass;

    // Team tricks won
    const team = compass === 'N' || compass === 'S' ? 'NS' : 'EW';
    const tricksWon = struct.tricksWon[team];

    // Dealer and declarer flags
    const isDealer = struct.dealer === compass;
    const isDeclarer = struct.declarer === compass;

    return {
      name,
      hand: isConcealedHand(hand)
        ? hand.map(() => HIDDEN_CARD)
        : sortCards(hand as CardCode[], trumpSuit),
      hidden: isConcealedHand(hand),
      position,
      tricksWon,
      isDealer,
      isDeclarer,
    };
  });
}

/** Build trick pile */
function trickPileOf(struct: EuchreStruct, viewerSeat: number): TrickCard[] {
  return struct.currentTrick.map(entry => {
    const playerIdx = typeof entry.player === 'number' ? entry.player : 0;
    return {
      card: entry.card as CardCode,
      position: mapPlayerToPosition(playerIdx, viewerSeat),
    };
  });
}

/**
 * An up-card ordered up is the dealer's from the call: a person sees
 * the dealer take it. The engine moves it into the dealer's hand only
 * at the discard step, after Go Alone, so at Go Alone it is drawn in
 * that hand here (face down in a hand the viewer cannot see); at the
 * discard step the engine's hand is drawn as served. It was ordered
 * up exactly when trump is its suit: the second round may not name
 * the turned-down suit. Nothing hidden is read: the up-card
 * and the call are public.
 */
function pickupOf(
  struct: EuchreStruct,
  phase: GamePhase,
  dealer: PlayerPosition,
  players: Player[],
): EuchreUIState['pickup'] {
  const upcard = struct.upcard as CardCode | null;
  const trumpSuit = struct.trumpSuit as Suit | null;
  if (
    !upcard ||
    !trumpSuit ||
    parseCard(upcard).suit !== trumpSuit ||
    !(phase === 'alone' || struct.discarding === true)
  ) {
    return null;
  }
  if (phase === 'alone') {
    const i = players.findIndex(p => p.position === dealer);
    const d = players[i];
    if (d === undefined) {
      throw new Error(
        `euchre: the dealer (${struct.dealer}) has no seat at the table`,
      );
    }
    // A face-up hand that already shows it (an engine that moves
    // it at the call) is drawn as served.
    if (d.hidden === true || !d.hand.includes(upcard)) {
      players[i] = {
        ...d,
        hand:
          d.hidden === true
            ? [...d.hand, HIDDEN_CARD]
            : sortCards([...d.hand, upcard], trumpSuit),
      };
    }
  }
  return {dealer, card: upcard};
}

/** Playable cards and bidding actions from legal actions */
function actionsOf(legalActions: readonly AugmentedAction[]): {
  playableCards: Set<string>;
  biddingActions: AugmentedAction[];
} {
  const playableCards = new Set<string>();
  const biddingActions: AugmentedAction[] = [];
  for (const action of legalActions) {
    if (action.cardCode) {
      playableCards.add(action.cardCode);
    } else if (action.actionType && action.actionType !== 'card') {
      biddingActions.push(action);
    }
  }
  return {playableCards, biddingActions};
}

/** Status message */
function statusOf(
  struct: EuchreStruct,
  phase: GamePhase,
  viewerSeat: number,
  outcome: EuchreOutcome | null,
): string {
  const currentPlayerIndex = struct.currentPlayer;
  const isViewerTurn = currentPlayerIndex === viewerSeat;
  const who = indexToCompass(currentPlayerIndex);
  if (outcome) {
    return outcome.scorer
      ? `${TEAM_NAME[outcome.scorer]} score ${pointsText(outcome.points[outcome.scorer])}`
      : 'No score';
  } else if (phase === 'bidding1') {
    if (struct.upcard) {
      return isViewerTurn
        ? 'Order up the upcard or pass?'
        : `Waiting for ${who} to bid`;
    }
  } else if (phase === 'bidding2') {
    return isViewerTurn
      ? 'Name trump or pass?'
      : `Waiting for ${who} to call trump`;
  } else if (phase === 'alone') {
    return isViewerTurn
      ? 'Go alone or play with partner?'
      : `Waiting for ${who} to choose`;
  } else if (phase === 'playing' && struct.discarding === true) {
    return isViewerTurn ? 'Discard a card' : `Waiting for ${who} to discard`;
  } else if (phase === 'playing') {
    return isViewerTurn ? 'Your turn to play' : `Waiting for ${who}`;
  }
  return '';
}

/** The last finished trick, for the trick collect. */
function lastTrickOf(
  struct: EuchreStruct,
  viewerSeat: number,
): CompletedTrick<CardCode, PlayerPosition> | null {
  const served = struct.lastTrick;
  if (served && !/^[NESW]$/.test(served.winner)) {
    throw new Error(
      `euchre: last trick winner is not a seat: ${JSON.stringify(served.winner)}`,
    );
  }
  return served
    ? {
        index: served.index,
        cards: served.cards.map(c => ({
          card: c.card as CardCode,
          seat: mapPlayerToPosition(c.player, viewerSeat),
        })),
        winner: mapPlayerToPosition(compassToIndex(served.winner), viewerSeat),
      }
    : null;
}

/** Each table position's value of `f` for the seat sitting there. */
function byPosition<T>(
  viewerSeat: number,
  f: (c: Compass) => T,
): Record<PlayerPosition, T> {
  return Object.fromEntries(
    COMPASSES.map(c => [
      mapPlayerToPosition(compassToIndex(c), viewerSeat),
      f(c),
    ]),
  ) as Record<PlayerPosition, T>;
}

/** Map phase */
const PHASE_MAP: Record<string, GamePhase> = {
  dealing: 'dealing',
  bidding1: 'bidding1',
  bidding2: 'bidding2',
  alone: 'alone',
  playing: 'playing',
  terminal: 'terminal',
};

/** The declarer's seat, checked to be one. */
function declarerOf(struct: EuchreStruct): Compass | null {
  const declarerSeat = struct.declarer as Compass | null;
  if (declarerSeat !== null && !(declarerSeat in COMPASS_NAME)) {
    throw new Error(
      `euchre: declarer is not a seat: ${JSON.stringify(struct.declarer)}`,
    );
  }
  return declarerSeat;
}

/**
 * At game over the engine prints the original deal, but every seat
 * that played has played all its cards: draw those
 * hands empty. Only the lone player's partner still holds its cards.
 * With allow_lone_defender a defender's partner may sit out too, and
 * the struct does not say which one: keep the served hands
 * there rather than draw a hand that never played as played out.
 */
function handPlayedOutOf(
  session: SessionPayload,
  struct: EuchreStruct,
  declarerSeat: Compass | null,
): boolean {
  const loneDefenderPossible =
    session.gameParams['allow_lone_defender'] === true;
  const handPlayedOut =
    session.state.isTerminal && declarerSeat !== null && !loneDefenderPossible;
  const total = struct.tricksWon.NS + struct.tricksWon.EW;
  if (handPlayedOut && total !== 5) {
    throw new Error(`euchre: game over after ${String(total)} tricks, not 5`);
  }
  return handPlayedOut;
}

/** The table's state for the viewer at `viewerSeat`, from the session. */
export function euchreUIState(
  session: SessionPayload,
  viewerSeat: number,
): EuchreUIState | null {
  const state = session.state;
  const struct = state.struct as EuchreStruct | undefined;
  if (!struct) {
    return null;
  }
  const declarerSeat = declarerOf(struct);
  const loneSeat = struct.goingAlone && declarerSeat ? declarerSeat : null;
  const sittingOutSeat = loneSeat ? partnerOf(loneSeat) : null;
  const handPlayedOut = handPlayedOutOf(session, struct, declarerSeat);
  const players = playersOf(struct, viewerSeat, handPlayedOut, sittingOutSeat);
  const phase: GamePhase = PHASE_MAP[struct.phase] ?? 'playing';
  const at = (compass: string): PlayerPosition =>
    mapPlayerToPosition(compassToIndex(compass), viewerSeat);
  const dealer = at(struct.dealer);
  const pickup = pickupOf(struct, phase, dealer, players);
  let outcome: EuchreOutcome | null = null;
  if (state.isTerminal) {
    if (!state.returns) throw new Error('euchre: game over without returns');
    outcome = euchreOutcome({
      declarer: declarerSeat,
      goingAlone: struct.goingAlone,
      tricksWon: struct.tricksWon,
      returns: state.returns,
    });
  }
  return {
    aloneSeat: loneSeat ? COMPASS_NAME[loneSeat] : null,
    sittingOut: sittingOutSeat ? at(sittingOutSeat) : null,
    outcome,
    pickup,
    lastTrick: lastTrickOf(struct, viewerSeat),
    players,
    trickPile: trickPileOf(struct, viewerSeat),
    phase,
    upcard: struct.upcard as CardCode | null,
    trumpSuit: struct.trumpSuit as Suit | null,
    dealer,
    declarer: struct.declarer ? at(struct.declarer) : null,
    goingAlone: struct.goingAlone,
    currentPlayer:
      struct.currentPlayer >= 0
        ? mapPlayerToPosition(struct.currentPlayer, viewerSeat)
        : 'bottom',
    tricksWon: struct.tricksWon,
    compassAt: byPosition(viewerSeat, c => c),
    teamOf: byPosition(viewerSeat, teamOf),
    discarding: struct.discarding ?? false,
    ...actionsOf(state.legalActions as readonly AugmentedAction[]),
    isTerminal: state.isTerminal,
    statusMessage: statusOf(struct, phase, viewerSeat, outcome),
  };
}

/**
 * Hook to transform OpenSpiel session state into Euchre UI state
 */
export function useEuchreState(
  session: SessionPayload | null | undefined,
  viewerSeat: number,
): EuchreUIState | null {
  return useMemo(
    () => (session ? euchreUIState(session, viewerSeat) : null),
    [session, viewerSeat],
  );
}
