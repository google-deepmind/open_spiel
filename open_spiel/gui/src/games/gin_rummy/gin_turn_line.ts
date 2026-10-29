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

// Gin's turn line (the stats region's last line: the status, then a bot note
// and a block reason), its words in one place. The panel writes them, and the
// layout sizes the stats region for the longest line they make
// (gin_layout_stats.ts): a longer phrase added here grows the region,
// instead of drawing text over the stock.

/** Why the viewer cannot act now (the board's interaction block reason). */
export const GIN_BLOCK_REASON = {
  finished: 'Game is finished',
  busy: 'Action in progress…',
  noSeat: 'Claim a seat to make moves',
  notGame: 'Gameplay is disabled in this mode',
  botSeat: 'Seat is controlled by a bot',
  otherView: 'Switch view back to your seat to act',
} as const;
/** The block reason while another seat is to move. */
export function ginWaitingFor(slot: number): string {
  return `Waiting for player ${String(slot)}`;
}

/** The note while the bot is to move (GameStats). */
export const GIN_BOT_NOTE = {
  thinking: 'Bot is thinking…',
  waiting: 'Waiting for bot step…',
} as const;

/**
 * The status's phrases (the derived state's statusText, the board's turn
 * text), joined with GIN_STATUS_JOIN.
 */
export const GIN_STATUS = {
  gameOver: 'Game over',
  ready: 'Ready for the next move',
  yourTurn: 'Your turn',
  opponentTurn: 'Opponent turn',
  waiting: 'Waiting for player',
  knock: 'Knock',
  discard: 'Discard',
  /** The forced stock draw (only Draw stock is legal). */
  drawStock: 'Draw from the stock',
  layoff: 'Layoff',
  pressDone: 'Press Done when finished',
} as const;
/** What joins the status's phrases. */
export const GIN_STATUS_JOIN = ' • ';
/** A spectator's turn text: whose move it is, named by its seat. */
export function ginToMove(seat: string): string {
  return `${seat} to move`;
}

/**
 * The laydown compose prompt, which replaces the status while the hand is
 * laid down.
 */
export function ginLaydownPrompt(
  stage: 'discard' | 'layoffs' | 'melds',
  stagedDiscard: string | null = null,
  layoffs = 0,
  stagedLayoffs = 0,
  layoffsNow = layoffs,
): string {
  if (stage === 'discard') {
    return stagedDiscard !== null
      ? `Discarding ${stagedDiscard}. Stage melds, then Done.`
      : 'Double-click your discard, stage melds, then Done.';
  }
  // The defender's lay-offs are said with their gesture: a page
  // that only said "Lay off" left a player dragging onto the meld in vain.
  // `layoffs` counts chains (5c after 4c); `layoffsNow` the cards lit and
  // draggable now. When they differ the line gives both, so the count
  // matches what the board offers.
  if (stage === 'layoffs' && layoffs > 0) {
    return layoffsNow < layoffs
      ? `${ginCardCount(layoffs)} lay off (${String(layoffsNow)} now): drag onto the knocker's meld or double-click. Then Done.`
      : `You can lay off ${ginCardCount(layoffs)}: drag onto the knocker's meld or double-click. Then Done.`;
  }
  // Every lay-off staged: still the lay-off stage, so the line says so and
  // how to take one back (not the meld stage's wording).
  if (stage === 'layoffs' && stagedLayoffs > 0) {
    return `Laying off ${ginCardCount(stagedLayoffs)} (double-click ${stagedLayoffs === 1 ? 'it' : 'one'} to take it back). Then Done.`;
  }
  return 'Stage your melds, then Done.';
}

/**
 * The line while the knocker's composition leaves more deadwood than the
 * knock card: Done waits, since the engine would refuse the pass.
 */
export function ginDeadwoodOverKnock(
  deadwood: number,
  knockCard: number,
  discardStage: boolean,
): string {
  const discard = discardStage ? ' or discard another card' : '';
  return `Deadwood ${String(deadwood)} is over the knock card (${String(knockCard)}): stage more melds${discard}, then Done.`;
}

/** The line while the staged melds or lay-offs are in none of the plans. */
export function ginNotLayable(layoffs: boolean): string {
  return `These ${layoffs ? 'lay-offs' : 'melds'} cannot be laid together: change them, then Done.`;
}

function ginCardCount(n: number): string {
  return n === 1 ? '1 card' : `${String(n)} cards`;
}

/**
 * The line after a Done that would forfeit legal lay-offs: Done waits for a
 * second press, and says what it would cost.
 */
export function ginLayoffSkipPrompt(layoffs: number, points: number): string {
  return `${ginCardCount(layoffs)} still ${layoffs === 1 ? 'lays' : 'lay'} off (${String(points)} points of deadwood). Press Done again to keep them.`;
}

/**
 * The line after a Done whose lay-offs could not be checked: the staged
 * lay-offs are in none of the engine's plans (logged), so Done asks first
 * rather than pass as if nothing were forfeited.
 */
export const GIN_LAYOFFS_UNCHECKED =
  "Lay-offs don't match the engine's moves. Press Done again to go on.";

/**
 * The longest turn line, as its pieces: the status (or laydown prompt), the
 * bot note and the block reason, each the longest of its kind. The status's
 * longest form is the turn text, a phase and "Press Done": a spectator's
 * turn text names the seat ("Player 1 to move"), and the longest phase it
 * shows is "Showdown" (the derived state's layoff phases).
 */
export function ginLongestTurnLine(): {status: string; notes: string[]} {
  const longest = (xs: readonly string[]): string =>
    xs.reduce((a, b) => (b.length > a.length ? b : a));
  const turnText = longest([
    GIN_STATUS.yourTurn,
    GIN_STATUS.opponentTurn,
    GIN_STATUS.waiting,
    ginToMove('Player 1'),
  ]);
  const status = longest([
    [turnText, 'Showdown', GIN_STATUS.pressDone].join(GIN_STATUS_JOIN),
    [turnText, GIN_STATUS.knock, GIN_STATUS.pressDone].join(GIN_STATUS_JOIN),
    ginLaydownPrompt('discard'),
    ginLaydownPrompt('discard', '10H'),
    ginLaydownPrompt('layoffs', null, 10),
    ginLaydownPrompt('layoffs', null, 0, 10),
    ginLaydownPrompt('layoffs', null, 10, 0, 9),
    GIN_LAYOFFS_UNCHECKED,
    ginLaydownPrompt('melds'),
    ginDeadwoodOverKnock(100, 10, true),
    ginNotLayable(true),
    ginLayoffSkipPrompt(10, 100),
    GIN_STATUS.ready,
  ]);
  return {
    status,
    notes: [
      longest(Object.values(GIN_BOT_NOTE)),
      longest([...Object.values(GIN_BLOCK_REASON), ginWaitingFor(1)]),
    ],
  };
}
