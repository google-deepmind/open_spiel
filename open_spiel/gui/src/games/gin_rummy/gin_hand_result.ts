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

// The end of a gin hand, accounted for: who knocked, the hand type (gin,
// knock, undercut or a draw), each player's melds, lay-offs and deadwood,
// and the points with their arithmetic. Read from the GameOver struct; the
// points are the engine's returns, never recomputed.
//
// Scoring (the engine's GinRummyState::Returns):
//  - gin (a knock with zero deadwood): the knocker scores the defender's
//    deadwood plus the gin bonus; the defender may not lay off;
//  - knock: the knocker's deadwood is lower than the defender's after
//    lay-offs, and the knocker scores the difference;
//  - undercut: the defender's deadwood is equal to or lower than the
//    knocker's, and the defender scores the difference plus the undercut
//    bonus;
//  - draw: no one knocked (the stock reached its last two cards, or the
//    engine's 50-draw or repeated-throw-back stops): no score;
//  - Oklahoma with spade_doubles: a spade initial upcard doubles the hand's
//    whole score, bonuses included. The struct carries the factor as
//    `score_multiplier` (present only under spade_doubles; absent is 1).
// The bonus is read off the returns (points divided by the multiplier,
// minus the difference), so a game played with other bonus parameters is
// still shown truly.

import {normaliseCardCode, SUIT_META} from './types.ts';

/** How a hand ended. */
export type GinHandKind = 'gin' | 'knock' | 'undercut' | 'draw';

/** One seat at the end of a hand. */
export interface GinSeatResult {
  seat: number;
  /** The melds the seat laid down, as engine card codes. */
  melds: string[][];
  /**
   * The cards left unmelded (its deadwood), or null when the viewer is not
   * shown them (masked).
   */
  deadwoodCards: string[] | null;
  /** The seat's deadwood count, or null when the viewer is not shown it. */
  deadwood: number | null;
}

/** The accounting of a finished hand. */
export interface GinHandResult {
  kind: GinHandKind;
  /** The seat that knocked; null for a draw. */
  knocker: number | null;
  /** The seat that scored; null for a draw. */
  winner: number | null;
  /**
   * Points the winner scored: the engine's return. Null for a draw, or when
   * the returns are not this hand's.
   */
  points: number | null;
  /**
   * The deadwood difference the points start from (gin: the defender's
   * deadwood). Null for a draw.
   */
  difference: number | null;
  /**
   * The gin or undercut bonus in the points, before the multiplier; null
   * when there is none (a knock, a draw) or the points are unknown.
   */
  bonus: number | null;
  /**
   * What the hand's score is multiplied by: 2 for an Oklahoma hand whose
   * initial upcard was a spade under spade_doubles, else 1.
   */
  multiplier: number;
  /** Cards the defender laid off on the knocker's melds. */
  layoffs: string[];
  /** Both seats, by seat index; empty for a draw (no hand is shown). */
  seats: GinSeatResult[];
  /** A draw: true when the stock reached its last two cards. */
  stockExhausted: boolean;
}

function asCards(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((c): c is string => typeof c === 'string')
    : [];
}

function asMelds(value: unknown): string[][] {
  return Array.isArray(value)
    ? value.map(asCards).filter(m => m.length > 0)
    : [];
}

/**
 * The stock size at which the hand ends without a knock (the engine's
 * kWallStockSize).
 */
export const GIN_WALL_STOCK_SIZE = 2;

function multiplierOf(struct: Record<string, unknown>): number {
  const raw = struct['score_multiplier'] ?? 1;
  if (typeof raw !== 'number' || !Number.isInteger(raw) || raw < 1) {
    throw new Error(
      `gin hand result: score_multiplier is not a positive integer (${JSON.stringify(raw)})`,
    );
  }
  return raw;
}

function drawResult(
  struct: Record<string, unknown>,
  returns: readonly number[] | null,
  multiplier: number,
): GinHandResult {
  if (returns?.some(r => r !== 0) === true) {
    throw new Error(
      `gin hand result: no one knocked, yet the returns are ${JSON.stringify(returns)}`,
    );
  }
  const stock = struct['stock_size'];
  const stockSize = typeof stock === 'number' ? stock : null;
  return {
    kind: 'draw',
    knocker: null,
    winner: null,
    points: null,
    difference: null,
    bonus: null,
    multiplier,
    layoffs: [],
    seats: [],
    stockExhausted: stockSize !== null && stockSize <= GIN_WALL_STOCK_SIZE,
  };
}

function seatResults(struct: Record<string, unknown>): GinSeatResult[] {
  const deadwoodRaw = Array.isArray(struct['deadwood'])
    ? struct['deadwood']
    : [];
  const hands = Array.isArray(struct['hands'])
    ? struct['hands'].map(asCards)
    : [];
  const layed = Array.isArray(struct['layed_melds'])
    ? struct['layed_melds'].map(asMelds)
    : [];
  return [0, 1].map(seat => {
    const d: unknown = deadwoodRaw[seat];
    const hand = hands[seat] ?? [];
    // A masked hand ('XX' backs) is one the viewer is not shown.
    const masked = hand.some(c => normaliseCardCode(c) === null);
    return {
      seat,
      melds: layed[seat] ?? [],
      deadwoodCards: masked ? null : hand,
      deadwood: typeof d === 'number' && d >= 0 ? d : null,
    };
  });
}

interface Points {
  points: number | null;
  bonus: number | null;
}

function pointsOf(
  kind: GinHandKind,
  winner: number,
  difference: number,
  multiplier: number,
  returns: readonly number[] | null,
  deadwoods: string,
): Points {
  if (returns === null) return {points: null, bonus: null};
  const r = returns[winner];
  if (r === undefined || returns[1 - winner] !== -r) {
    throw new Error(
      `gin hand result: returns ${JSON.stringify(returns)} are not a zero-sum score for seat ${String(winner)}`,
    );
  }
  if (r % multiplier !== 0) {
    throw new Error(
      `gin hand result: returns ${JSON.stringify(returns)} are not a multiple of the score multiplier ${String(multiplier)}`,
    );
  }
  const extra = r / multiplier - difference;
  if (kind === 'knock' ? extra !== 0 : extra < 0) {
    const times =
      multiplier === 1 ? '' : ` (score multiplier ${String(multiplier)})`;
    throw new Error(
      `gin hand result: a ${kind} with deadwoods ${deadwoods}${times} returns ${JSON.stringify(returns)}`,
    );
  }
  return {points: r, bonus: kind === 'knock' ? null : extra};
}

/**
 * The hand's result, or null while it is not over (the struct's phase is not
 * GameOver). `returns` are the engine's returns for this hand, or null when
 * they are not. Throws when the struct contradicts itself (both seats
 * knocked, the returns disagree with the deadwoods), so a wrong sum is never
 * drawn.
 */
export function ginHandResult(
  struct: Record<string, unknown> | null,
  returns: readonly number[] | null,
): GinHandResult | null {
  const phase = struct?.['phase'];
  if (struct === null || typeof phase !== 'string') return null;
  if (phase.toLowerCase() !== 'gameover') return null;
  const knocked = Array.isArray(struct['knocked'])
    ? struct['knocked'].map(Boolean)
    : [];
  const knockers = knocked.flatMap((k, seat) => (k ? [seat] : []));
  if (knockers.length > 1) {
    throw new Error(
      `gin hand result: both seats knocked (${JSON.stringify(struct['knocked'])})`,
    );
  }
  const multiplier = multiplierOf(struct);
  const knocker = knockers[0];
  if (knocker === undefined) return drawResult(struct, returns, multiplier);
  const defender = 1 - knocker;
  const seats = seatResults(struct);
  const dK = seats[knocker]?.deadwood ?? null;
  const dD = seats[defender]?.deadwood ?? null;
  if (dK === null || dD === null) {
    throw new Error(
      `gin hand result: a deadwood is not shown at the end of the hand (${JSON.stringify(struct['deadwood'])})`,
    );
  }
  const kind: GinHandKind = dK === 0 ? 'gin' : dK < dD ? 'knock' : 'undercut';
  const winner = kind === 'undercut' ? defender : knocker;
  const difference = kind === 'gin' ? dD : Math.abs(dD - dK);
  const deadwoods = `${String(dK)} (knocker) and ${String(dD)}`;
  return {
    kind,
    knocker,
    winner,
    ...pointsOf(kind, winner, difference, multiplier, returns, deadwoods),
    difference,
    multiplier,
    layoffs: asCards(struct['layoffs']),
    seats,
    stockExhausted: false,
  };
}

/** How the hand type is named on screen. */
export const GIN_HAND_KIND_LABEL: Readonly<Record<GinHandKind, string>> = {
  gin: 'Gin',
  knock: 'Knock',
  undercut: 'Undercut',
  draw: 'Draw',
};

/** A card as a player reads it: 10♦, Q♠. */
export function ginCardText(code: string): string {
  const c = normaliseCardCode(code);
  if (c === null) return code;
  const rank = c.slice(0, -1);
  const symbol = SUIT_META[c.slice(-1)]?.symbol ?? '';
  return `${rank === 'T' ? '10' : rank}${symbol}`;
}

/** Names a seat: 'You' for the viewer's own seat, else its name. */
export type GinSeatNamer = (seat: number) => string;

/**
 * The result in a few words, for the stats region: its label (the hand
 * type) and the outcome ('You lose 72', 'Player 1 wins 13', 'no score').
 */
export function ginResultFigure(
  result: GinHandResult,
  you: number | null,
  name: GinSeatNamer,
): {label: string; outcome: string} {
  const label = GIN_HAND_KIND_LABEL[result.kind];
  if (result.kind === 'draw' || result.winner === null) {
    return {label, outcome: 'no score'};
  }
  const pts = result.points === null ? '' : ` ${String(result.points)}`;
  if (you === null) {
    return {label, outcome: `${name(result.winner)} wins${pts}`};
  }
  return {
    label,
    outcome: result.winner === you ? `You win${pts}` : `You lose${pts}`,
  };
}

/**
 * The longest stats figure ginResultFigure makes, for the layout to size
 * the stats region by (gin_layout_stats.ts): a spectator's, at three-digit
 * points. The board names a seat that is not the viewer's 'Player N' in the
 * figure.
 */
export const GIN_RESULT_FIGURE_LONGEST = `${GIN_HAND_KIND_LABEL.undercut}: Player 1 wins 125`;

/**
 * The points as a sum, in words: '47 (your deadwood) + 25 gin bonus = 72'.
 * A doubled hand brackets the sum and doubles it: '(47 (your deadwood) + 25
 * gin bonus) × 2 = 144'. `whose` names a seat's deadwood ('your', "Player
 * 1's"). Null for a draw.
 */
export function ginPointsSum(
  result: GinHandResult,
  whose: GinSeatNamer,
): string | null {
  const {knocker, difference} = result;
  if (result.kind === 'draw' || knocker === null || difference === null) {
    return null;
  }
  const defender = 1 - knocker;
  const dK = String(result.seats[knocker]?.deadwood ?? null);
  const dD = String(result.seats[defender]?.deadwood ?? null);
  const total = result.points === null ? '' : ` = ${String(result.points)}`;
  const bonus = (what: string): string =>
    result.bonus === null
      ? ` + the ${what} bonus`
      : ` + ${String(result.bonus)} ${what} bonus`;
  const times = (sum: string, end: string): string =>
    result.multiplier === 1
      ? `${sum}${end}`
      : `(${sum}) × ${String(result.multiplier)}${end}`;
  if (result.kind === 'gin') {
    return times(`${dD} (${whose(defender)} deadwood)${bonus('gin')}`, total);
  }
  if (result.kind === 'knock') {
    const diff = `${dD} (${whose(defender)} deadwood) − ${dK} (${whose(knocker)})`;
    return result.multiplier === 1
      ? `${diff} = ${String(difference)}`
      : times(diff, total);
  }
  return times(
    `${dK} (${whose(knocker)} deadwood) − ${dD} (${whose(defender)})${bonus('undercut')}`,
    total,
  );
}
