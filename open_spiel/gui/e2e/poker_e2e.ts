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

// Whole hands of kuhn_poker and leduc_poker played on the poker board, from
// seat 0 against Random, through the board's own buttons (Check, Call, Bet,
// Fold and New hand), never the legal list. Hands are dealt until one has
// ended at a showdown and one by a fold. Every frame seat 0 receives is
// checked to hold none of seat 1's cards until a showdown tables them, and
// the board's plate shows seat 1's card only then.
//
// With POKER_SCREENSHOTS set to a directory, the table is captured there at
// POKER_VIEWPORT_WIDTH (default 1280) by 800 pixels: the opening decision,
// mid-hand, a showdown and a hand won by fold.

/// <reference types="node" />

import path from 'node:path';
import process from 'node:process';

import type {Locator, Page} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import type {App} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';

const SCREENSHOTS = process.env['POKER_SCREENSHOTS'] ?? '';
// Hands dealt before both endings must have been seen.
const MAX_HANDS = 30;
// Seat 0's moves in one hand: kuhn bets at most twice, and leduc's two
// rounds allow two raises each.
const MAX_MOVES = 8;

interface PokerStruct {
  readonly player_hands: readonly string[];
  readonly board_cards: string;
  readonly current_player: number;
  readonly betting_history: string;
}

function struct(frame: Frame): PokerStruct {
  const view = frame.presentation?.view as {struct?: PokerStruct} | undefined;
  if (view?.struct === undefined) throw new Error('No poker struct.');
  return view.struct;
}

/** Whether `frame` ends a hand at a showdown, two seats still in. */
function isShowdown(frame: Frame): boolean {
  return (
    frame.status === 'terminal' && !struct(frame).betting_history.includes('f')
  );
}

/**
 * Checks no frame seat 0 received showed seat 1's card, except at a
 * showdown, where it must.
 */
function expectNoPeek(frames: readonly Frame[], game: string): number {
  let checked = 0;
  for (const frame of frames) {
    if (frame.session.game !== game || !isViewer(frame.viewer, 0)) continue;
    if (frame.presentation === null) continue;
    const hands = struct(frame).player_hands;
    const what = `${game} ply ${String(frame.ply)}`;
    if (isShowdown(frame)) {
      expect(hands[1], what).toMatch(/^[2-9TJQKA][shdc]$/);
    } else {
      expect(hands[1], what).toBe('');
    }
    // The cards dealt to the seats are chance outcomes no seat reads.
    for (const entry of frame.history) {
      if (entry.seat === null) expect(entry.label, what).toBeNull();
    }
    checked++;
  }
  return checked;
}

async function capture(page: Page, name: string): Promise<void> {
  if (SCREENSHOTS === '') return;
  // The chips and cards in flight land before the shot.
  await page.waitForTimeout(1200);
  await page.screenshot({path: path.join(SCREENSHOTS, `kernel-${name}.png`)});
}

/**
 * The frame where seat 0 next moves, or the hand's end, after `ply`, among
 * the frames from the `from`th on.
 */
function nextTurn(
  app: App,
  game: string,
  ply: number,
  from: number,
): Promise<Frame> {
  return app.frames.waitFor(
    frame =>
      frame.session.game === game &&
      isViewer(frame.viewer, 0) &&
      frame.ply > ply &&
      (frame.status === 'terminal' ||
        (frame.to_move.seat === 0 && frame.legal_actions !== null)),
    from,
    `seat 0's turn after ply ${String(ply)}`,
  );
}

/**
 * How seat 0 plays a hand: check and call to a showdown, or fold when it
 * can, opening with a check or a bet. Random's choices are seeded by the ply,
 * so a new hand meets the same choices; the two openings meet both answers.
 */
type Style = 'showdown' | 'fold-after-check' | 'fold-after-bet';

/** The button seat 0 presses in `style`. */
async function choose(board: Locator, style: Style): Promise<Locator> {
  const passive = board.locator('.poker-action--call');
  if (style === 'showdown') return passive;
  const fold = board.locator('.poker-action--fold');
  if ((await fold.count()) > 0) return fold;
  return style === 'fold-after-bet'
    ? board.locator('.poker-action--bet')
    : passive;
}

/** Plays seat 0's moves to the end of one hand; returns its last frame. */
async function playHand(
  app: App,
  game: string,
  board: Locator,
  plan: {style: Style; shots: Set<string>; from: number},
): Promise<Frame> {
  // Seat 0 opens every hand of both games, so a hand starts at its turn
  // (never at the last hand's end, which the page may be sent again).
  let frame = await app.frames.waitFor(
    next =>
      next.session.game === game &&
      isViewer(next.viewer, 0) &&
      next.status !== 'terminal' &&
      next.to_move.seat === 0 &&
      next.legal_actions !== null,
    plan.from,
    'the first turn of a hand',
  );
  for (let move = 0; move < MAX_MOVES; move++) {
    if (frame.status === 'terminal') return frame;
    const cards = board.locator(
      '.poker-seat--0 .poker-card:not(.poker-card--empty)',
    );
    await expect(cards).toHaveCount(1);
    await expect(
      board.locator('.poker-seat--1 .poker-card--empty'),
    ).toHaveCount(1);
    if (!plan.shots.has('opening')) {
      plan.shots.add('opening');
      await capture(app.page, `${game}-01-opening`);
    } else if (!plan.shots.has('mid') && frame.history.length > 3) {
      plan.shots.add('mid');
      await capture(app.page, `${game}-02-midhand`);
    }
    const button = await choose(board, plan.style);
    await expect(button).toBeEnabled();
    const from = app.frames.count;
    await button.click();
    frame = await nextTurn(app, game, frame.ply, from);
  }
  if (frame.status !== 'terminal') {
    throw new Error(
      `The ${game} hand did not end in ${String(MAX_MOVES)} moves.`,
    );
  }
  return frame;
}

async function expectEnding(board: Locator, end: Frame): Promise<void> {
  const result = board.locator('.poker-terminal strong');
  if (isShowdown(end)) {
    await expect(
      board.locator('.poker-seat--1 .poker-card:not(.poker-card--empty)'),
    ).toHaveCount(1);
    // Seat 0 is the viewer: its own win reads "You win".
    await expect(result).toHaveText(
      /^(You win|player \d wins|.+ split) \+\d+$|split pot/,
    );
  } else {
    await expect(
      board.locator('.poker-seat--1 .poker-card--empty'),
    ).toHaveCount(1);
    await expect(result).toHaveText(/by fold$/);
  }
}

for (const game of ['kuhn_poker', 'leduc_poker']) {
  test(`${game}: seat 0 plays hands on the poker board against Random`, async ({
    app,
    page,
  }) => {
    test.setTimeout(300_000);
    const width = Number(process.env['POKER_VIEWPORT_WIDTH'] ?? '1280');
    await page.setViewportSize({width, height: 800});
    await app.openSession(
      {
        game,
        parameters: {},
        seats: [
          {controller: 'human', config: {}, autoplay: false},
          {controller: 'random', config: {}, autoplay: true},
        ],
        chance_policy: 'auto',
        // Each new hand is dealt and played afresh.
        seed: null,
      },
      0,
    );
    const board = (await app.gameplay()).locator('.poker-gameplay');
    await expect(board.locator('.poker-handbar strong')).toHaveText(
      game === 'kuhn_poker' ? 'Kuhn Poker' : "Leduc Hold'em",
    );
    const shots = new Set<string>();
    const endings = new Set<string>();
    let from = 0;
    for (let hand = 0; hand < MAX_HANDS && endings.size < 2; hand++) {
      const style: Style = !endings.has('showdown')
        ? 'showdown'
        : hand % 2 === 0
          ? 'fold-after-check'
          : 'fold-after-bet';
      const end = await playHand(app, game, board, {style, shots, from});
      await expectEnding(board, end);
      const ending = isShowdown(end) ? 'showdown' : 'fold';
      if (!endings.has(ending)) {
        await capture(
          page,
          `${game}-${ending === 'showdown' ? '03-showdown' : '04-fold'}`,
        );
      }
      endings.add(ending);
      if (endings.size < 2) {
        from = app.frames.count;
        await board
          .locator('.poker-terminal button', {hasText: 'New hand'})
          .click();
      }
    }
    expect([...endings].sort()).toEqual(['fold', 'showdown']);
    // The board's own View section, as the View tab shows it.
    const animation = await app.showSection('View', 'Animation');
    await expect(animation.getByText('Animation speed')).toBeVisible();
    await capture(page, `${game}-05-view`);
    expect(expectNoPeek(app.frames.all(), game)).toBeGreaterThan(2);
  });
}
