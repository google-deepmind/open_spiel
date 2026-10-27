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

// The betting of a poker hand, read from the engine's ACPC betting string:
// the street amounts, the history's rows in poker's own words, and how the
// hand ended. Split from poker_state.ts.

/** How a finished hand ended. */
export type PokerEnding = 'fold' | 'showdown';

/** One action of the ACPC betting string. */
export interface BettingToken {
  kind: 'f' | 'c' | 'r';
  amount: number;
}

/**
 * The engine's ACPC betting string, one array of actions per street:
 * "r200c/cr400f" -> [[r200, c], [c, r400, f]]. A raise's amount is the
 * raiser's total in the hand ("raise to", across streets). Fails loud on
 * anything else.
 */
export function parseBettingHistory(bettingHistory: string): BettingToken[][] {
  return bettingHistory.split('/').map(round => {
    const tokens: BettingToken[] = [];
    const re = /f|c|r(\d+)|(.)/g;
    let match: RegExpExecArray | null;
    while ((match = re.exec(round)) !== null) {
      if (match[2] !== undefined) {
        throw new Error(
          `poker: unknown action '${match[2]}' in betting string '${bettingHistory}'`,
        );
      }
      const text = match[0];
      if (text === 'f' || text === 'c') {
        tokens.push({kind: text, amount: 0});
      } else {
        tokens.push({kind: 'r', amount: Number(match[1])});
      }
    }
    return tokens;
  });
}

/**
 * What each seat had in when the current street's betting began: preflop,
 * the ante (the blinds are preflop bets); after it, the last raise-to of an
 * earlier street, or the opening level when nobody raised.
 */
export function streetBaseOf(
  rounds: BettingToken[][],
  blinds: readonly number[],
  ante = 0,
): number {
  const current = rounds.length - 1;
  return current <= 0
    ? ante
    : levelOf(rounds.slice(0, current).flat(), blinds, ante);
}

/**
 * The amount to match after `tokens`: the last raise-to, or the big blind
 * plus the ante.
 */
export function levelOf(
  tokens: readonly BettingToken[],
  blinds: readonly number[],
  ante = 0,
): number {
  let level = Math.max(0, ...blinds) + ante;
  for (const token of tokens) {
    if (token.kind === 'r') {
      level = token.amount;
    }
  }
  return level;
}

function chips(value: number): string {
  return value.toLocaleString();
}

const LABEL_KIND: Array<[RegExp, BettingToken['kind']]> = [
  [/^Fold$/i, 'f'],
  [/^(?:Call|Check)$/i, 'c'],
  [/^(?:Bet|Raise)\d+$/i, 'r'],
];

/** A history row: its label and the seat that acted (null for chance). */
export interface HistoryRow {
  label: string;
  player: number | null;
}

/** The table a history is replayed at: blinds, stacks and ante. */
export interface BettingTable {
  blinds: readonly number[];
  startingStacks: readonly number[];
  ante: number;
}

function bettingString(rounds: BettingToken[][]): string {
  return rounds
    .map(r =>
      r.map(t => (t.kind === 'r' ? `r${String(t.amount)}` : t.kind)).join(''),
    )
    .join('/');
}

/** The move of a history row, checked against its betting action. */
function checkedMove(
  row: HistoryRow,
  index: number,
  token: BettingToken,
  ordinal: number,
  rounds: BettingToken[][],
): void {
  const move = /\bmove=(.+)$/.exec(row.label)?.[1]?.trim() ?? row.label;
  const kind = LABEL_KIND.find(([re]) => re.test(move))?.[1];
  const amount = Number(/(\d+)$/.exec(move)?.[1] ?? 0);
  if (kind !== token.kind || (kind === 'r' && amount !== token.amount)) {
    throw new Error(
      `poker history: row ${String(index)} '${move}' does not match betting action ${String(ordinal)} of '${bettingString(rounds)}'`,
    );
  }
}

/** One betting action in poker's words, updating `contribution`. */
function describeToken(
  token: BettingToken,
  seat: number,
  level: number,
  base: number,
  contribution: number[],
  stack: number,
): string {
  const allIn = (): string =>
    stack > 0 && (contribution[seat] ?? 0) >= stack ? ' all-in' : '';
  if (token.kind === 'f') return 'Fold';
  if (token.kind === 'c') {
    const owed = level - (contribution[seat] ?? 0);
    if (owed <= 0) return 'Check';
    const paid =
      stack > 0 ? Math.min(owed, stack - (contribution[seat] ?? 0)) : owed;
    contribution[seat] = (contribution[seat] ?? 0) + paid;
    return `Call ${chips(paid)}${allIn()}`;
  }
  const verb = level <= base ? 'Bet' : 'Raise to';
  contribution[seat] = token.amount;
  return `${verb} ${chips(token.amount - base)}${allIn()}`;
}

/**
 * Each player row of the history in poker's own words, replayed from the
 * engine's betting string (the authority on amounts and streets): "Check"
 * when nothing is owed (the engine says "Call" for both), "Call 50" (the
 * chips put in), "Bet 100" when nobody has bet this street, "Raise to 300"
 * otherwise, both in street chips, and "all-in" when the stack is gone. The
 * rows name the seat; a row whose move disagrees with the string fails loud.
 * A history that does not hold every action (a trimmed record) gets no
 * words, and its rows keep the engine's labels.
 */
export function describeHistoryActions(
  rows: readonly HistoryRow[],
  rounds: BettingToken[][],
  table: BettingTable,
): Map<number, string> {
  const {blinds, startingStacks, ante} = table;
  const described = new Map<number, string>();
  const playerRows = rows
    .map((row, index) => ({row, index}))
    .filter(({row}) => row.player !== null && row.player >= 0);
  const tokenCount = rounds.reduce((sum, round) => sum + round.length, 0);
  if (tokenCount === 0 || tokenCount !== playerRows.length) {
    return described;
  }
  const seats = Math.max(
    blinds.length,
    startingStacks.length,
    ...playerRows.map(({row}) => (row.player ?? 0) + 1),
  );
  const contribution = Array.from(
    {length: seats},
    (_, seat) => (blinds[seat] ?? 0) + ante,
  );
  const played: BettingToken[] = [];
  let next = 0;
  rounds.forEach((round, roundIndex) => {
    const base = streetBaseOf(rounds.slice(0, roundIndex + 1), blinds, ante);
    for (const token of round) {
      const entry = playerRows[next];
      if (entry === undefined) throw new Error('poker history: rows ran out');
      next += 1;
      checkedMove(entry.row, entry.index, token, next, rounds);
      const seat = entry.row.player ?? 0;
      const level = levelOf(played, blinds, ante);
      played.push(token);
      const stack = startingStacks[seat] ?? 0;
      described.set(
        entry.index,
        describeToken(token, seat, level, base, contribution, stack),
      );
    }
  });
  return described;
}

/**
 * Folds in the engine's ACPC betting string ("r200c/cr400f"): each action is
 * 'f', 'c' or 'r<amount>', so every 'f' is one fold.
 */
function foldsInBettingHistory(bettingHistory: string): number {
  return (bettingHistory.match(/f/g) ?? []).length;
}

/**
 * A finished hand ended by fold when at most one seat is still in it. The
 * engine's betting string is the authority on the count; the history only
 * names which seats folded, and may be shorter (a trimmed record).
 */
export function deriveEnding(
  isTerminal: boolean,
  numPlayers: number,
  foldedSeats: ReadonlySet<number>,
  bettingHistory: string,
): PokerEnding | null {
  if (!isTerminal) {
    return null;
  }
  const folds = Math.max(
    foldedSeats.size,
    foldsInBettingHistory(bettingHistory),
  );
  if (folds > numPlayers - 1) {
    throw new Error(
      `poker: ${String(folds)} folds at a ${String(numPlayers)}-seat table`,
    );
  }
  return folds === numPlayers - 1 ? 'fold' : 'showdown';
}

const FOLD_MOVE = /(?:^|\bmove=)Fold\b/i;

/**
 * The seats that folded, from the public action history (a fold is public at
 * the table, so it is never redacted). Fails loud on a fold with no seat or by
 * a seat the table does not have.
 */
export function foldedSeatsFromHistory(
  labels: readonly string[],
  players: ReadonlyArray<number | null | undefined>,
  numPlayers: number,
): Set<number> {
  const folded = new Set<number>();
  labels.forEach((label, index) => {
    if (!FOLD_MOVE.test(label)) {
      return;
    }
    const player = players[index];
    if (typeof player !== 'number' || player < 0) {
      throw new Error(
        `poker history: row ${String(index)} is a fold with no seat (player ${String(player)})`,
      );
    }
    if (player >= numPlayers) {
      throw new Error(
        `poker history: player ${String(player)} folded but has no seat (${String(numPlayers)} seats)`,
      );
    }
    folded.add(player);
  });
  return folded;
}
