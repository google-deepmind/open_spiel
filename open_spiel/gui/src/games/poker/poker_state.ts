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

// The poker table as the board draws it, derived from the session: the
// seats, the board, the pot, the street, the actions grouped into fold, call
// and bets, and the history in poker's own words.

import {
  type BettingToken,
  type PokerEnding,
  deriveEnding,
  describeHistoryActions,
  foldedSeatsFromHistory,
  levelOf,
  parseBettingHistory,
  streetBaseOf,
} from './poker_betting.ts';
import type {LegalAction, SessionPayload} from './poker_session.ts';

export type {PokerEnding} from './poker_betting.ts';

/** A card: its code ("Ks"), rank and suit. */
export interface PokerCard {
  code: string;
  rank: string;
  suit: 'c' | 'd' | 'h' | 's';
}

/** One seat at the table, as the board draws its plate. */
export interface PokerSeat {
  slot: number;
  name: string;
  cards: PokerCard[];
  contribution: number;
  startingStack: number;
  stack: number;
  isActive: boolean;
  /** The seat folded this hand (read off the public action history). */
  hasFolded: boolean;
  isDealer: boolean;
  isSmallBlind: boolean;
  isBigBlind: boolean;
  bestHand?: string | undefined;
  bestFive: PokerCard[];
  returnValue?: number | undefined;
  isWinner: boolean;
}

/** The legal actions, grouped as the board's buttons offer them. */
export interface PokerActionGroups {
  fold?: LegalAction | undefined;
  call?: LegalAction | undefined;
  bets: Array<LegalAction & {amount: number}>;
}

/** The table, derived from a session. */
export interface PokerUiState {
  seats: PokerSeat[];
  board: PokerCard[];
  blinds: number[];
  /** Each seat's ante, in chips; 0 at a table with blinds. */
  ante: number;
  pot: number;
  currentPlayer: number;
  currentStreet: string;
  bettingHistory: string;
  isTerminal: boolean;
  /**
   * How a finished hand ended: 'fold' when every seat but one folded (no
   * showdown happened, so no hand is ranked or shown), 'showdown' otherwise;
   * null while the hand is in play.
   */
  ending: PokerEnding | null;
  returns: number[] | null;
  actionGroups: PokerActionGroups;
  /**
   * What each seat had in when this street's betting began (hand-total
   * chips, the engine's unit): the ante preflop, where the blinds are this
   * street's bets; afterwards the last raise-to of an earlier street, or
   * the opening level. Street amounts (a bet of 100, a raise to 300) are
   * hand totals minus this, as every poker table names them.
   */
  streetBase: number;
  /**
   * The amount to match now (hand-total chips): the last raise-to, or the
   * opening level. Nobody has bet this street while it equals `streetBase`,
   * so the next wager is a Bet, otherwise a Raise to. The buttons and the
   * history's replay use this one rule.
   */
  streetLevel: number;
  history: PokerHistoryEntry[];
}

/** One row of the Action History. */
export interface PokerHistoryEntry {
  label: string;
  player: number | null;
  /** The action in poker's own words ("Check", "Call 50", "Bet 100",
   *  "Raise to 300", "all-in"), when the betting string accounts for it. */
  action?: string;
}

interface UniversalPokerStruct {
  board_cards?: string;
  player_hands?: string[];
  player_contributions?: number[];
  starting_stacks?: number[];
  pot_size?: number;
  current_player?: number;
  betting_history?: string;
  blinds?: number[];
  ante?: number;
  best_hand_rank_types?: string[];
  best_five_card_hands?: string[];
}

const SUIT_LABELS: Record<PokerCard['suit'], string> = {
  c: 'Clubs',
  d: 'Diamonds',
  h: 'Hearts',
  s: 'Spades',
};

const RANK_LABELS: Record<string, string> = {
  A: 'Ace',
  K: 'King',
  Q: 'Queen',
  J: 'Jack',
  T: 'Ten',
};

/** The card's accessible name: "King of Spades". */
export function cardAriaLabel(card: PokerCard): string {
  const rank = RANK_LABELS[card.rank] ?? card.rank;
  return `${rank} of ${SUIT_LABELS[card.suit]}`;
}

/** The cards of a card string ("KsQh"); anything unreadable is skipped. */
export function parseCards(cards: unknown): PokerCard[] {
  if (typeof cards !== 'string' || cards.length === 0) {
    return [];
  }
  const parsed: PokerCard[] = [];
  for (let index = 0; index + 1 < cards.length; index += 2) {
    const rawRank = cards[index]?.toUpperCase();
    const rawSuit = cards[index + 1]?.toLowerCase();
    if (!rawRank || !rawSuit || !['c', 'd', 'h', 's'].includes(rawSuit)) {
      continue;
    }
    parsed.push({
      code: `${rawRank}${rawSuit}`,
      rank: rawRank,
      suit: rawSuit as PokerCard['suit'],
    });
  }
  return parsed;
}

/**
 * The board in the order its cards were dealt. The engine's `board_cards` is
 * a sorted card set, so a turn or river that ranks between two flop cards
 * would be drawn between them and push the others along. `previous` is the
 * order shown last (card codes); while the board only grows (the next
 * street) or only shrinks (a replay stepping back), the cards already shown
 * keep their places and a new card goes on the end. A board that does
 * neither is a different hand, drawn as the engine gives it (the flop is a
 * set dealt at once, so its own order carries nothing).
 */
export function boardInDealOrder(
  previous: readonly string[],
  board: readonly PokerCard[],
): PokerCard[] {
  const byCode = new Map(board.map(card => [card.code, card]));
  const shown = new Set(previous);
  const grows = previous.every(code => byCode.has(code));
  const shrinks = board.every(card => shown.has(card.code));
  if (!grows && !shrinks) {
    return [...board];
  }
  const kept = previous.flatMap(code => {
    const card = byCode.get(code);
    return card ? [card] : [];
  });
  return [...kept, ...board.filter(card => !shown.has(card.code))];
}

function asStruct(session: SessionPayload | undefined): UniversalPokerStruct {
  const value = session?.state.struct;
  if (!value || typeof value !== 'object') {
    return {};
  }
  return value;
}

function parseBetAmount(action: LegalAction): number | null {
  const match = /\b(?:Bet|Raise)(\d+)\b/i.exec(action.label);
  if (match) {
    return Number(match[1]);
  }
  if (action.action >= 2) {
    return action.action;
  }
  return null;
}

/** The legal actions as the buttons offer them: fold, call and bets. */
export function groupPokerActions(
  legalActions: readonly LegalAction[],
): PokerActionGroups {
  const bets: Array<LegalAction & {amount: number}> = [];
  let fold: LegalAction | undefined;
  let call: LegalAction | undefined;

  for (const action of legalActions) {
    if (/\bFold\b/i.test(action.label)) {
      fold = action;
      continue;
    }
    if (/\b(?:Call|Check)\b/i.test(action.label)) {
      call = action;
      continue;
    }
    const amount = parseBetAmount(action);
    if (amount !== null && Number.isFinite(amount)) {
      bets.push({...action, amount});
    }
  }

  bets.sort((left, right) => left.amount - right.amount);
  return {fold, call, bets};
}

const HOLDEM_STREETS = ['Preflop', 'Flop', 'Turn', 'River'];
// Hold'em's public cards before each street: none, the flop, the turn and
// the river.
const HOLDEM_BOARD_SIZES = [0, 3, 1, 1];

/**
 * Each betting round's name. Hold'em's streets have names of their own; any
 * other game's rounds are numbered ("Round 1", "Round 2", as Leduc's are),
 * and a game of one round (Kuhn) has just "Betting".
 */
function roundNames(boardSizes: readonly number[]): readonly string[] {
  const holdem =
    boardSizes.length === HOLDEM_BOARD_SIZES.length &&
    boardSizes.every((size, index) => size === HOLDEM_BOARD_SIZES[index]);
  if (holdem) return HOLDEM_STREETS;
  if (boardSizes.length <= 1) return ['Betting'];
  return boardSizes.map((_, index) => `Round ${String(index + 1)}`);
}

/**
 * The betting round the board has reached, by the public cards turned:
 * "Preflop" to "River" in Hold'em, "Round 1" and "Round 2" in Leduc,
 * "Betting" in Kuhn. `boardSizes` is empty for a game that does not say,
 * read as Hold'em.
 */
export function roundName(
  boardCards: readonly PokerCard[],
  gameBoardSizes: readonly number[],
): string {
  const boardSizes =
    gameBoardSizes.length > 0 ? gameBoardSizes : HOLDEM_BOARD_SIZES;
  let turned = 0;
  let round = 0;
  boardSizes.forEach((size, index) => {
    turned += size;
    if (index > 0 && boardCards.length >= turned) round = index;
  });
  const names = roundNames(boardSizes);
  const name = names[round];
  if (name === undefined) {
    throw new Error(`poker: no name for round ${String(round)}`);
  }
  return name;
}

function inferStreet(
  boardCards: PokerCard[],
  ending: PokerEnding | null,
  boardSizes: readonly number[],
): string {
  if (ending === 'fold') {
    return 'Won by fold';
  }
  if (ending === 'showdown') {
    return 'Showdown';
  }
  return roundName(boardCards, boardSizes);
}

// One name per seat for the whole session. The server labels a seat
// "player N" (0-based); the fallback uses the same numbering, so a seat
// whose label is missing does not change its name at showdown. The plates,
// the hand bar, the showdown line and the Action History all read this one
// name. The table names a seat by where it sits, never by its controller:
// that is the Players panel's.
function playerName(session: SessionPayload, slot: number): string {
  const player = session.players.find(entry => entry.slot === slot);
  const name = player?.label;
  return name !== undefined &&
    name.length > 0 &&
    name.toLowerCase() !== 'terminal'
    ? name
    : `player ${String(slot)}`;
}

function blindSeat(
  blinds: readonly number[],
  better: (a: number, b: number) => boolean,
): number | undefined {
  let best: {amount: number; slot: number} | undefined;
  blinds.forEach((blind, slot) => {
    const amount = blind;
    if (amount > 0 && (!best || better(amount, best.amount))) {
      best = {amount, slot};
    }
  });
  return best?.slot;
}

interface SeatInputs {
  hands: readonly string[];
  contributions: readonly number[];
  startingStacks: readonly number[];
  blinds: readonly number[];
  bestHands: readonly string[];
  bestFiveHands: readonly string[];
  numPlayers: number;
  currentPlayer: number;
  foldedSeats: ReadonlySet<number>;
  ending: PokerEnding | null;
}

function deriveSeats(session: SessionPayload, s: SeatInputs): PokerSeat[] {
  const returns = session.state.returns;
  const maxReturn =
    returns && returns.length > 0 ? Math.max(...returns) : undefined;
  const smallBlind = blindSeat(s.blinds, (a, b) => a < b);
  const bigBlind = blindSeat(s.blinds, (a, b) => a > b);
  // A table without blinds (an ante game) has no button.
  const hasButton = smallBlind !== undefined;
  return Array.from({length: s.numPlayers}, (_, slot) => {
    const contribution = s.contributions[slot] ?? 0;
    const startingStack = s.startingStacks[slot] ?? 0;
    const returnValue = returns?.[slot];
    // A hand's rank is a showdown fact. A folded hand never contends, and
    // a hand won by fold shows no hand at all, so neither gets a rank
    // label or a best five (the hole cards themselves are whatever the
    // served struct lets this viewer see).
    const ranked = !s.foldedSeats.has(slot) && s.ending !== 'fold';
    const bestHand = s.bestHands[slot];
    return {
      slot,
      name: playerName(session, slot),
      cards: parseCards(s.hands[slot] ?? ''),
      contribution,
      startingStack,
      stack: Math.max(0, startingStack - contribution),
      isActive: s.currentPlayer === slot,
      hasFolded: s.foldedSeats.has(slot),
      isDealer:
        hasButton &&
        (s.numPlayers > 2 ? slot === s.numPlayers - 1 : slot === 1),
      isSmallBlind: smallBlind === slot,
      isBigBlind: bigBlind === slot,
      bestHand: ranked && bestHand ? bestHand : undefined,
      bestFive: ranked ? parseCards(s.bestFiveHands[slot] ?? '') : [],
      returnValue,
      isWinner:
        session.state.isTerminal &&
        typeof returnValue === 'number' &&
        typeof maxReturn === 'number' &&
        maxReturn > 0 &&
        returnValue === maxReturn,
    };
  });
}

function list<T>(value: T[] | undefined): T[] {
  return Array.isArray(value) ? value : [];
}

function deriveHistory(
  session: SessionPayload,
  rounds: BettingToken[][],
  table: {blinds: number[]; startingStacks: number[]; ante: number},
): PokerHistoryEntry[] {
  const historyPlayers = session.state.historyPlayers;
  const rows = session.state.historyActionStrings.map((label, index) => ({
    label,
    player:
      typeof historyPlayers[index] === 'number' ? historyPlayers[index] : null,
  }));
  const described = describeHistoryActions(rows, rounds, table);
  return rows.map((row, index) => {
    const action = described.get(index);
    return action === undefined ? row : {...row, action};
  });
}

/** The table the board draws for `session`; null with no session. */
export function derivePokerState(
  session: SessionPayload | undefined,
): PokerUiState | null {
  if (!session) {
    return null;
  }
  const struct = asStruct(session);
  const hands = list(struct.player_hands);
  const contributions = list(struct.player_contributions);
  const startingStacks = list(struct.starting_stacks).map(Number);
  const blinds = list(struct.blinds).map(Number);
  const ante = struct.ante ?? 0;
  const numPlayers = Math.max(
    session.players.length,
    hands.length,
    contributions.length,
    startingStacks.length,
    2,
  );
  const currentPlayer =
    typeof struct.current_player === 'number'
      ? struct.current_player
      : session.state.currentPlayer;
  const board = parseCards(struct.board_cards ?? '');
  const bettingHistory = struct.betting_history ?? '';
  const foldedSeats = foldedSeatsFromHistory(
    session.state.historyActionStrings,
    session.state.historyPlayers,
    numPlayers,
  );
  const ending = deriveEnding(
    session.state.isTerminal,
    numPlayers,
    foldedSeats,
    bettingHistory,
  );
  const seats = deriveSeats(session, {
    hands,
    contributions,
    startingStacks,
    blinds,
    bestHands: list(struct.best_hand_rank_types),
    bestFiveHands: list(struct.best_five_card_hands),
    numPlayers,
    currentPlayer,
    foldedSeats,
    ending,
  });
  const rounds = parseBettingHistory(bettingHistory);
  const boardSizes = session.game.boardSizes;
  return {
    seats,
    board,
    blinds,
    ante,
    pot: struct.pot_size ?? 0,
    currentPlayer,
    currentStreet: inferStreet(board, ending, boardSizes),
    bettingHistory,
    isTerminal: session.state.isTerminal,
    ending,
    returns: session.state.returns,
    actionGroups: groupPokerActions(session.state.legalActions),
    streetBase: streetBaseOf(rounds, blinds, ante),
    streetLevel: levelOf(rounds.flat(), blinds, ante),
    history: deriveHistory(session, rounds, {blinds, startingStacks, ante}),
  };
}

/**
 * One Action History row as the table names things. The seat plates say
 * "player 0", so the history says "player 0" too, not "P1"; a chance row
 * keeps its card ("Deal Qh" read "Deal Deal"); a sized action spaces its
 * amount ("Bet8287" reads "Bet 8,287").
 */
export function historyEntryText(
  entry: PokerHistoryEntry,
  seats: ReadonlyArray<Pick<PokerSeat, 'name'>>,
): {who: string; what: string} {
  const move = /\bmove=(.+)$/.exec(entry.label)?.[1]?.trim() ?? entry.label;
  if (entry.player === null || entry.player < 0) {
    const card = /^Deal\s+(\S.*)$/.exec(move)?.[1];
    return {who: 'Deal', what: card ?? move};
  }
  const seat = seats[entry.player];
  if (!seat) {
    throw new Error(
      `poker history: player ${String(entry.player)} has no seat (${String(seats.length)} seats)`,
    );
  }
  if (entry.action !== undefined) {
    return {who: seat.name, what: entry.action};
  }
  const sized = /^([A-Za-z]+)(\d+)$/.exec(move);
  const what = sized
    ? `${sized[1] ?? ''} ${Number(sized[2]).toLocaleString()}`
    : move;
  return {who: seat.name, what};
}
