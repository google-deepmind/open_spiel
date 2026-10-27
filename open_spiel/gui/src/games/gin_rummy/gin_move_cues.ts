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

// What the last turn did, for the board's resting-state cues (legible at a
// glance, without animation). Pure: the history in,
// one cue out, so the board can show the turn after the viewer looked away and
// tests can pin it without a browser.
//
// A turn is the trailing run of one seat's moves (chance outcomes skipped). It
// names the pile the seat drew from, the upcard it took (public: it was face
// up) and the card it discarded. A card drawn from the stock is never named:
// the history the server sends masks it, and the cue does not look for it.
// A seat whose actions the server hides from the viewer (the label names no
// action) gives no cue.
import {normaliseCardCode} from './types.ts';
import type {GinZone} from './gin_rummy_layout.ts';

/** One history entry as the session serves it. */
export interface GinHistoryStep {
  /** Acting seat; negative for chance. */
  player: number;
  /** The action's label ("Player: 1 Action: Draw stock", "... Action: Qd"). */
  label: string;
}

/** What the last turn did. */
export interface GinTurnCue {
  seat: number;
  /** The pile drawn from this turn, if the seat has drawn. */
  drew: 'stock' | 'upcard' | null;
  /** History index of the draw (identifies the turn across renders). */
  drawIndex: number | null;
  /** The upcard taken, when drew === 'upcard' and it is known. */
  taken: string | null;
  /** The card discarded this turn, once it has been. */
  discarded: string | null;
  discardIndex: number | null;
}

/** The action part of a label: what follows "Action: ", or the whole label. */
function actionText(label: string): string {
  const i = label.lastIndexOf('Action:');
  return (i >= 0 ? label.slice(i + 'Action:'.length) : label).trim();
}

/** Zips the session's parallel history arrays; null when they disagree. */
export function ginHistorySteps(
  players: readonly number[] | null | undefined,
  labels: readonly string[] | null | undefined,
): GinHistoryStep[] | null {
  if (players === null || players === undefined) return null;
  if (labels === null || labels === undefined) return null;
  if (players.length !== labels.length) {
    console.error(
      `ginHistorySteps: historyPlayers has ${String(players.length)} entries, historyActionStrings ${String(labels.length)}; no move cues`,
    );
    return null;
  }
  return players.map((player, i) => ({player, label: labels[i] ?? ''}));
}

/**
 * The last turn's cue, or null when there is nothing to show: no turn yet, a
 * turn that did not draw (a pass, a layoff), or a turn by
 * `excludeSeat` (the viewer's own seat: they know what they did).
 *
 * `prevUpcard` is the struct's prev_upcard: the card the last upcard draw
 * took (the engine sets it at the draw and the discard leaves it alone).
 */
export function lastTurnCue(
  steps: readonly GinHistoryStep[],
  opts: {excludeSeat?: number | null; prevUpcard?: string | null} = {},
): GinTurnCue | null {
  let seat: number | null = null;
  let start = steps.length;
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const p = steps[i]?.player ?? -1;
    if (p < 0) continue;
    if (seat === null) seat = p;
    else if (p !== seat) break;
    start = i;
  }
  if (seat === null || seat === opts.excludeSeat) return null;

  let drew: GinTurnCue['drew'] = null;
  let drawIndex: number | null = null;
  let discarded: string | null = null;
  let discardIndex: number | null = null;
  for (let i = start; i < steps.length; i += 1) {
    const step = steps[i];
    if (step?.player !== seat) continue;
    const text = actionText(step.label).toLowerCase();
    if (drew === null && (text === 'draw upcard' || text === 'draw stock')) {
      drew = text === 'draw upcard' ? 'upcard' : 'stock';
      drawIndex = i;
      continue;
    }
    // The turn's discard is its first single-card action after the draw:
    // a knock comes between the two, melds (several cards) after it. A
    // single card with no draw before it is a layoff, not a discard.
    const card = normaliseCardCode(actionText(step.label));
    if (card !== null && drew !== null && discarded === null) {
      discarded = card;
      discardIndex = i;
    }
  }
  if (drew === null) return null;
  const taken =
    drew === 'upcard' ? normaliseCardCode(opts.prevUpcard ?? null) : null;
  return {seat, drew, drawIndex, taken, discarded, discardIndex};
}

/**
 * The card under `discarded` on the discard pile: what the pile showed
 * before the discard landed (the frame between the draw and the discard).
 */
export function cardUnder(
  discardPile: readonly string[],
  discarded: string,
): string | null {
  const rest = discardPile.filter(c => normaliseCardCode(c) !== discarded);
  return rest.at(-1) ?? null;
}

/** A turn's cue and the board zone its seat sits in. */
export type GinBoardCue = GinTurnCue & {zone: GinZone};

/**
 * Which items carry the last turn's mark: the stock's top card when it drew
 * from the stock, the upcard it took (face up wherever it now sits) and the
 * discard it made while it is still the upcard.
 */
export function cuedItemIds(
  cue: GinBoardCue | null,
  shown: {stockCount: number; discardCard: string | null},
): {stock: string | null; taken: string | null; discard: string | null} {
  if (cue === null) return {stock: null, taken: null, discard: null};
  return {
    stock:
      cue.drew === 'stock' && shown.stockCount > 0
        ? `stock-${String(shown.stockCount - 1)}`
        : null,
    taken:
      cue.drew === 'upcard' && cue.taken !== null && cue.taken !== cue.discarded
        ? cue.taken
        : null,
    // The pile's card as the board draws it (its item id); compared as a
    // canonical code, since the cue's cards are normalised.
    discard:
      cue.discarded !== null &&
      shown.discardCard !== null &&
      normaliseCardCode(shown.discardCard) === cue.discarded
        ? shown.discardCard
        : null,
  };
}
