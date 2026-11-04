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

// A game that hides information, viewed from a seat and as the oracle: the
// seat is sent and shown its own card and never its opponent's, and the
// oracle is shown both. A game at a seat (the address names the seat) is
// never sent another seat's hand, whatever its View settings say: its
// Perspective is locked and Show all cards is off.

import type {Locator} from '@playwright/test';

import type {Frame} from '../src/contract/contract.ts';
import type {App} from './app.ts';
import {isViewer} from './frames.ts';
import {expect, test} from './fixtures.ts';

const GAME = 'kuhn_poker';

// Kuhn poker deals one card to each of its two seats.
const DEALS = 2;

/**
 * Deals the cards as the oracle, where the session leaves chance to it: the
 * outcomes are the rows of Actions › Legal, each with its probability.
 */
async function dealIfExposed(app: App): Promise<void> {
  const legal = await app.legal();
  for (let deal = 0; deal < DEALS; deal++) {
    const first = app.legalButtons(legal).first();
    await expect(first).toBeVisible();
    if (!((await first.textContent()) ?? '').includes('p=')) return;
    const before = app.frames.count;
    await first.click();
    await app.frames.waitFor(frame => frame.num_plies === deal + 1, before);
  }
}

// The name the poker board gives kuhn_poker's card `card`, a rank index in
// the state text; the cards are drawn as spades.
function cardName(card: string): string {
  const rank = ['Jack', 'Queen', 'King'][Number(card)];
  if (rank === undefined) throw new Error(`"${card}" is not a kuhn card.`);
  return `${rank} of Spades`;
}

// The code the poker board writes for kuhn_poker's card `card` ("Ks").
function cardCode(card: string): string {
  const rank = ['J', 'Q', 'K'][Number(card)];
  if (rank === undefined) throw new Error(`"${card}" is not a kuhn card.`);
  return `${rank}s`;
}

/** The face-up cards on `seat`'s plate, by their accessible names. */
function faceUp(board: Locator, seat: number): Locator {
  return board.locator(
    `.poker-seat--${String(seat)} .poker-card:not(.poker-card--empty)`,
  );
}

function requireText(text: string | null, what: string): string {
  if (text === null) throw new Error(`The frame carries no ${what}.`);
  return text;
}

/** Checks what CONTRACT.md withholds from a seat is absent from `frame`. */
function expectSeatFrame(frame: Frame, dealt: Frame): void {
  expect(frame.omniscient).toBe(false);
  expect(frame.state_text).toBeNull();
  expect(frame.struct).toBeNull();
  expect(frame.session.seed).toBeNull();
  for (const entry of frame.history.slice(0, DEALS)) {
    expect(entry).toMatchObject({seat: null, action: null, label: null});
  }
  const body = JSON.stringify(frame);
  for (const entry of dealt.history.slice(0, DEALS)) {
    expect(body).not.toContain(requireText(entry.label, 'deal label'));
  }
}

test('kuhn_poker: seat 0 sees its own card only; the oracle sees both', async ({
  app,
  page,
}) => {
  await app.openGame(GAME);
  await app.setViewer('oracle');
  await dealIfExposed(app);
  const dealt = await app.frames.waitFor(
    frame =>
      frame.session.game === GAME &&
      isViewer(frame.viewer, 'oracle') &&
      frame.to_move.kind === 'seat',
    0,
    'the oracle frame after the deal',
  );
  const state = requireText(dealt.state_text, 'state text');
  const [ownCard, opponentCard] = state.trim().split(/\s+/);
  if (ownCard === undefined || opponentCard === undefined) {
    throw new Error(`The state text ${JSON.stringify(state)} names no cards.`);
  }

  const seen = await app.setViewer(0);
  expectSeatFrame(seen, dealt);
  const own = requireText(seen.information_state_text, 'information state');
  expect(own.startsWith(ownCard)).toBe(true);

  // The poker board shows seat 0 its own card face up and the other's slot
  // empty.
  const board = await app.gameplay();
  await expect(faceUp(board, 0)).toHaveAttribute(
    'aria-label',
    cardName(ownCard),
  );
  await expect(faceUp(board, 1)).toHaveCount(0);
  await expect(board.locator('.poker-seat--1 .poker-card--empty')).toHaveCount(
    1,
  );
  await app.showSection('Actions', 'History');
  await app.showPanel('State');
  for (const frame of app.frames.all()) {
    if (isViewer(frame.viewer, 0)) expectSeatFrame(frame, dealt);
  }
  // Neither deal reaches seat 0 as a label, and the full state not at all,
  // anywhere in the document: text, attributes and hidden panels. An icon's
  // path data is drawing coordinates, not text, and is left out.
  const exposed = await page.evaluate(() => {
    const parts = [document.body.textContent];
    for (const element of document.body.querySelectorAll('*')) {
      if (element.closest('svg') !== null) continue;
      for (const attribute of element.attributes) parts.push(attribute.value);
    }
    return parts.join('\n');
  });
  expect(exposed).not.toContain('Deal:');
  expect(exposed).not.toContain(state);

  const oracle = await app.setViewer('oracle');
  const full = requireText(oracle.state_text, 'state text');
  expect(full.startsWith(`${ownCard} ${opponentCard}`)).toBe(true);
  const table = await app.gameplay();
  for (const [seat, card] of [
    [0, ownCard],
    [1, opponentCard],
  ] as const) {
    await expect(faceUp(table, seat)).toHaveAttribute(
      'aria-label',
      cardName(card),
    );
  }
  const history = (await app.showSection('Actions', 'History')).getByRole(
    'list',
    {name: 'History'},
  );
  // The oracle's History names each deal in the table's words ("Deal Ks"),
  // as the table's Action History does.
  for (const card of [ownCard, opponentCard]) {
    await expect(history).toContainText(`Deal ${cardCode(card)}`);
  }
});

test('euchre: a game at seat 0 is never sent or shown another hand', async ({
  app,
  page,
}) => {
  const human = {controller: 'human', config: {}, autoplay: false};
  const bot = {controller: 'random', config: {}, autoplay: true};
  // Analysis first, with every hand shown, so that nothing it chose carries
  // into the game.
  await app.openSession(
    {
      game: 'euchre',
      parameters: {},
      seats: [human, bot, bot, bot],
      chance_policy: 'auto',
      seed: 4,
    },
    'oracle',
  );
  await app.sitAt(2);
  const euchre = await app.showSection('View', 'Euchre');
  await euchre.getByRole('checkbox', {name: 'Show all cards'}).check();

  const before = app.frames.count;
  await app.setViewer(0);
  const view = await app.showSection('View', 'Perspective');
  await expect(view.getByTestId('perspective-locked')).toHaveText(
    'Seat 0 (you)',
  );
  await expect(view.getByTestId('perspective-lock-reason')).toHaveText(
    'Locked to your seat during a game',
  );
  await expect(view.getByRole('combobox')).toHaveCount(0);
  const section = await app.showSection('View', 'Euchre');
  await expect(
    section.getByRole('checkbox', {name: 'Show all cards'}),
  ).toBeDisabled();
  await expect(
    section.getByText('Cards can only be revealed in analysis mode.'),
  ).toBeVisible();

  const board = (await app.gameplay()).locator('.euchre-gameplay-wrapper');
  await expect(board.locator('.euchre-player-name--bottom')).toHaveText(
    'North',
  );
  for (const position of ['left', 'top', 'right']) {
    await expect(
      board.locator(`.euchre-player-section--${position} [data-card-id]`),
    ).toHaveCount(0);
  }
  const seen = app.frames.all().slice(before);
  expect(seen.length).toBeGreaterThan(0);
  for (const frame of seen) {
    expect(isViewer(frame.viewer, 0)).toBe(true);
    expect(frame.omniscient).toBe(false);
    expect(frame.state_text).toBeNull();
    const struct = (
      frame.presentation?.view as {
        struct: {hands: Record<string, Array<string | null>>};
      }
    ).struct;
    for (const compass of ['E', 'S', 'W']) {
      expect(struct.hands[compass]?.every(card => card === null)).toBe(true);
    }
    expect(struct.hands['N']).not.toContain(null);
  }
  expect(new URL(page.url()).searchParams.get('viewSeat')).toBeNull();
});
