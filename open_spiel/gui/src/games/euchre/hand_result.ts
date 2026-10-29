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

// What a finished euchre hand scored, and why.
// The points are the engine's returns; the cause is the scoring case of
// the rules that the hand reached: the
// makers made it (3 or 4 tricks), marched (all 5) or were euchred (0-2),
// with or without a lone maker. A hand no one named trump is thrown in.

/** A seat by its compass point. */
export type Compass = 'N' | 'E' | 'S' | 'W';
/** A side: North-South or East-West. */
export type Team = 'NS' | 'EW';

/** Each seat's name. */
export const COMPASS_NAME: Record<Compass, string> = {
  N: 'North',
  E: 'East',
  S: 'South',
  W: 'West',
};
/** Each side's name. */
export const TEAM_NAME: Record<Team, string> = {
  NS: 'North-South',
  EW: 'East-West',
};

/** The side `seat` plays for. */
export function teamOf(seat: Compass): Team {
  return seat === 'N' || seat === 'S' ? 'NS' : 'EW';
}

/** The seat across the table from `seat`. */
export function partnerOf(seat: Compass): Compass {
  return ({N: 'S', S: 'N', E: 'W', W: 'E'} as const)[seat];
}

/** What a finished hand scored, and why. */
export interface EuchreOutcome {
  /** Points each side scored this hand; the side that did not score has 0. */
  points: Record<Team, number>;
  /** The side that scored, or null when no one did (a thrown-in hand). */
  scorer: Team | null;
  /** One plain sentence: how the hand ended. */
  cause: string;
}

function tricks(n: number): string {
  return n === 0 ? 'no trick' : n === 1 ? '1 trick' : `${String(n)} tricks`;
}

/** The outcome of a finished hand, from the engine's returns. */
export function euchreOutcome(args: {
  declarer: Compass | null;
  goingAlone: boolean;
  tricksWon: Record<Team, number>;
  returns: readonly number[];
}): EuchreOutcome {
  const {declarer, goingAlone, tricksWon, returns} = args;
  if (returns.length !== 4) {
    throw new Error(
      `euchre: expected 4 returns, got ${JSON.stringify(returns)}`,
    );
  }
  const [n = 0, e = 0, s = 0, w = 0] = returns;
  if (n !== s || e !== w || n !== -e) {
    throw new Error(
      `euchre: returns are not one zero-sum score per side: ${JSON.stringify(returns)}`,
    );
  }
  const points: Record<Team, number> = {NS: Math.max(n, 0), EW: Math.max(e, 0)};
  const scorer: Team | null = n > 0 ? 'NS' : e > 0 ? 'EW' : null;

  if (declarer === null) {
    return {points, scorer, cause: 'Thrown in: no one named trump.'};
  }
  const makers = teamOf(declarer);
  const taken = tricksWon[makers];
  const who = goingAlone
    ? `${COMPASS_NAME[declarer]} went alone and`
    : `${TEAM_NAME[makers]}, the makers,`;
  let cause: string;
  if (taken === 5) cause = `${who} won all 5 tricks: a march.`;
  else if (taken >= 3) cause = `${who} won ${tricks(taken)}.`;
  else {
    cause = `${who} ${goingAlone ? 'was' : 'were'} euchred with ${tricks(taken)}.`;
  }
  return {points, scorer, cause};
}

/** "1 point", "2 points". */
export function pointsText(points: number): string {
  return points === 1 ? '1 point' : `${String(points)} points`;
}
