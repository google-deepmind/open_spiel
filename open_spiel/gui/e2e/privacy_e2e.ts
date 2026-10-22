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
// oracle is shown both.

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

  const board = await app.gameplay();
  await expect(
    board.getByRole('region', {name: "Seat 0's information state"}),
  ).toHaveText(own);
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
  await expect(
    (await app.gameplay()).getByRole('region', {name: 'State'}),
  ).toHaveText(full);
  const history = (await app.showSection('Actions', 'History')).getByRole(
    'list',
    {name: 'History'},
  );
  for (const entry of dealt.history.slice(0, DEALS)) {
    await expect(history).toContainText(requireText(entry.label, 'deal label'));
  }
});
