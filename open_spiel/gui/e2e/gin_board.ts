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

// The gin board as the gin_rummy end-to-end tests read and press it: the
// view's parts they read, seat 0's moves and the presses that make them.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import {expect} from './fixtures.ts';

/** The game the tests play. */
export const GAME = 'gin_rummy';
/**
 * Seat 0's moves in one hand: a draw and a discard per turn, and at most a
 * few dozen turns before the stock runs down to the wall.
 */
export const MAX_MOVES = 200;
/** Where the table is captured, or '' for nowhere. */
export const SCREENSHOTS = process.env['GIN_SCREENSHOTS'] ?? '';

/** The parts of the gin_rummy view this test reads. */
export interface GinView {
  readonly phase: string;
  readonly hands: ReadonlyArray<readonly string[]>;
  readonly upcard: string | null;
  readonly discardPile: readonly string[];
  readonly melded: ReadonlySet<string>;
  readonly visible: ReadonlySet<number>;
  readonly stage: string | null;
  readonly knockCard: number;
  readonly view: Record<string, unknown>;
}

/** `value` as an object, or a failure naming `what`. */
export function record(value: unknown, what: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${what} is not an object.`);
  }
  return value as Record<string, unknown>;
}

/** `value` as a list of strings, or a failure naming `what`. */
export function strings(value: unknown, what: string): string[] {
  if (!Array.isArray(value)) throw new Error(`${what} is not a list.`);
  return value.map(item => {
    if (typeof item !== 'string') {
      throw new Error(`${what} holds a non-string.`);
    }
    return item;
  });
}

/** Every meld of the seat's hand, from the per-seat analysis. */
export function allMelds(
  view: Record<string, unknown>,
  seat: number,
): string[][] {
  const melds = record(view['ginAnalysisMeldsBySeat'], 'melds')[String(seat)];
  if (!Array.isArray(melds)) return [];
  return melds.map(meld => strings(meld, 'meld'));
}

/** The seat's best melds, from the per-seat analysis. */
export function bestMelds(
  view: Record<string, unknown>,
  seat: number,
): string[][] {
  const melds = allMelds(view, seat);
  const best = record(view['ginAnalysisBestIndicesBySeat'], 'best')[
    String(seat)
  ];
  if (!Array.isArray(best)) return [];
  return best.flatMap(index =>
    typeof index === 'number' && melds[index] !== undefined
      ? [melds[index]]
      : [],
  );
}

/** The cards of the seat's best melds. */
export function bestMelded(
  view: Record<string, unknown>,
  seat: number,
): Set<string> {
  return new Set(bestMelds(view, seat).flat());
}

/** The lay-down plans the view gives the seat laying down. */
export function plansOf(
  view: Record<string, unknown>,
): Array<Array<{cards: string[]; action: number}>> {
  const plans = view['ginLaydownPlans'];
  if (!Array.isArray(plans)) return [];
  return plans.map(plan => {
    if (!Array.isArray(plan)) throw new Error('A plan is not a list.');
    return plan.map(step => {
      const fields = record(step, 'plan step');
      const action = fields['action'];
      if (typeof action !== 'number') throw new Error('A step has no action.');
      return {cards: strings(fields['cards'], 'step cards'), action};
    });
  });
}

/** The parts of `frame`'s gin view the tests read. */
export function readView(frame: Frame): GinView {
  if (frame.presentation === null) throw new Error('No gin presentation.');
  const view = record(frame.presentation.view, 'view');
  const struct = record(view['struct'], 'struct');
  const handsValue = struct['hands'];
  if (!Array.isArray(handsValue)) throw new Error('No hands.');
  const visible = new Set(
    Object.keys(record(view['ginVisibleHands'], 'visible')).map(Number),
  );
  const stage = view['ginLaydownStage'];
  const knockCard = struct['knock_card'];
  if (typeof knockCard !== 'number') throw new Error('No knock card.');
  return {
    phase: String(struct['phase']),
    hands: handsValue.map(hand => strings(hand, 'hand')),
    upcard: typeof struct['upcard'] === 'string' ? struct['upcard'] : null,
    discardPile: strings(struct['discard_pile'], 'discard pile'),
    melded: bestMelded(view, 0),
    visible,
    stage: typeof stage === 'string' ? stage : null,
    knockCard,
    view,
  };
}

/** The deadwood count of the ranks that are not their pips. */
export const POINTS: Readonly<Record<string, number>> = {
  A: 1,
  T: 10,
  J: 10,
  Q: 10,
};

/** A card's deadwood count. */
export function points(code: string): number {
  const rank = code.slice(0, -1);
  return POINTS[rank] ?? (rank === 'K' ? 10 : Number(rank));
}

/** The legal action labelled `text` after "Action: ", if any. */
export function legal(frame: Frame, text: string): number | null {
  const found = (frame.legal_actions ?? []).find(
    action => action.label.split('Action: ')[1] === text,
  );
  return found?.id ?? null;
}

/** Seat 0's discard: the highest deadwood card it may discard. */
export function discardChoice(frame: Frame, view: GinView): string {
  const hand = view.hands[0] ?? [];
  const legalCards = hand.filter(code => legal(frame, code) !== null);
  const deadwood = legalCards.filter(code => !view.melded.has(code));
  const pool = deadwood.length > 0 ? deadwood : legalCards;
  const best = [...pool].sort((a, b) => points(b) - points(a))[0];
  if (best === undefined) throw new Error('Seat 0 has no card to discard.');
  return best;
}

/** The board's action button labelled `text`. */
export function button(board: Locator, text: string): Locator {
  return board.locator('.gin-action', {hasText: new RegExp(`^${text}$`)});
}

/** Seat 0's card `code`, in its hand. */
export function handCard(board: Locator, code: string): Locator {
  return board.locator(`[data-card-id="${code}"][data-slot="me-${code}"]`);
}

/** The turn line: the status, or the lay-down prompt. */
export function turnLine(board: Locator): Locator {
  return board.locator('.gin-stats__turn > span').first();
}

/**
 * Checks the scene's live region (the page's `status`) says the turn line:
 * it must not keep an earlier drop's "Discarding ..." through the knock and
 * the lay-down. A repeat is marked by a trailing no-break space.
 */
export async function expectLiveIsTurnLine(
  board: Locator,
  what: string,
): Promise<void> {
  const read = async (): Promise<[string, string]> => [
    ((await board.locator('[data-card-live]').textContent()) ?? '').replace(
      /\u00a0$/,
      '',
    ),
    (await turnLine(board).textContent()) ?? '',
  ];
  await expect
    .poll(
      async () => {
        const [live, line] = await read();
        return line !== '' && live === line;
      },
      {message: `${what}: the live region says the turn line`},
    )
    .toBe(true);
}

/** The press that makes seat 0's move at `frame`, and what it is. */
export function chooseMove(
  board: Locator,
  frame: Frame,
  view: GinView,
  knocks = true,
): {target: Locator; what: string; double: boolean} {
  if (knocks && legal(frame, 'Knock') !== null) {
    return {target: button(board, 'Knock'), what: 'knock', double: false};
  }
  if (view.stage === 'discard') {
    const code = discardChoice(frame, view);
    return {
      target: handCard(board, code),
      what: `stage ${code} as the knocker's discard`,
      double: true,
    };
  }
  if (view.stage !== null) {
    return {target: button(board, 'Done'), what: 'done', double: false};
  }
  if (legal(frame, 'Pass') !== null) {
    return {target: button(board, 'Pass'), what: 'pass', double: false};
  }
  if (legal(frame, 'Draw stock') !== null) {
    return {
      target: board.locator('[data-slot="stock"][data-card-id]').last(),
      what: 'draw from the stock',
      double: true,
    };
  }
  const code = discardChoice(frame, view);
  return {target: handCard(board, code), what: `discard ${code}`, double: true};
}

/** Captures the page as `name`, when GIN_SCREENSHOTS names a directory. */
export async function capture(page: Page, name: string): Promise<void> {
  if (SCREENSHOTS === '') return;
  // The board settles its cards in flight before the shot.
  await page.waitForTimeout(900);
  await page.screenshot({path: path.join(SCREENSHOTS, `kernel-${name}.png`)});
}

/**
 * Milliseconds a press waits for its move to be sent before pressing again:
 * a card still flying into place may take the first press elsewhere.
 */
export const PRESS_WAIT_MS = 4000;
/** How many times a press is tried. */
export const PRESS_TRIES = 4;

/** Presses `move`'s target until a move is sent, or fails. */
export async function press(
  page: Page,
  move: {target: Locator; what: string; double: boolean},
): Promise<void> {
  await expect(move.target, move.what).toBeVisible();
  for (let attempt = 0; attempt < PRESS_TRIES; attempt++) {
    const posted = page
      .waitForResponse(
        response =>
          response.request().method() === 'POST' &&
          new URL(response.url()).pathname.endsWith('/actions'),
        {timeout: PRESS_WAIT_MS},
      )
      .then(
        () => true,
        () => false,
      );
    if (move.double) await move.target.dblclick({force: true});
    else await move.target.click();
    if (await posted) return;
  }
  throw new Error(
    `${move.what}: no move was sent after ${String(PRESS_TRIES)} presses.`,
  );
}
