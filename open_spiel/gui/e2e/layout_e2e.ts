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

// Where things are drawn in the running client: a copy icon never covers the
// text it copies, and a notice over Gameplay never moves the board.

import {expect, test} from './fixtures.ts';

test('a copy icon sits beside its text, never over it', async ({app, page}) => {
  // The width at which go's first line runs past a State panel.
  await page.setViewportSize({width: 1440, height: 900});
  await app.openGame('go');
  const state = await app.showSection('State', 'Text');
  const text = state.getByRole('group', {name: 'Text', exact: true});
  await expect(text).toContainText('history.size()=0');
  // The line is longer than the text's box, so it scrolls sideways.
  expect(
    await text.evaluate(element => element.scrollWidth > element.clientWidth),
  ).toBe(true);
  const copy = state.getByRole('button', {name: 'Copy to clipboard'});
  const textBox = await text.boundingBox();
  const copyBox = await copy.boundingBox();
  if (textBox === null || copyBox === null) {
    throw new Error('The text or its copy icon is not drawn.');
  }
  expect(copyBox.x).toBeGreaterThanOrEqual(textBox.x + textBox.width);
});

test('the hand-over notice is drawn over Gameplay and moves nothing', async ({
  app,
}) => {
  // Kuhn poker opens as seat 0, every seat played by a person.
  await app.openGame('kuhn_poker');
  const gameplay = await app.gameplay();
  const board = gameplay.locator('.gameplay__board');
  const status = gameplay.locator('.gameplay__status');
  const boardBefore = await board.boundingBox();
  const statusBefore = await status.boundingBox();

  const legal = await app.legal();
  const before = app.frames.count;
  await app.legalButtons(legal).first().click();
  await app.frames.waitFor(
    frame => frame.to_move.kind === 'seat' && frame.to_move.seat === 1,
    before,
    'Player 1 to move',
  );
  await app.gameplay();
  const notice = gameplay.getByRole('status').filter({
    hasText: 'Player 1 is to move, and plays from their own view.',
  });
  await expect(notice).toBeVisible();
  await expect(
    notice.getByRole('button', {name: 'Hand over to Player 1'}),
  ).toBeVisible();
  expect(
    await notice.evaluate(
      element => getComputedStyle(element.parentElement ?? element).position,
    ),
  ).toBe('absolute');
  expect(await board.boundingBox()).toEqual(boardBefore);
  expect(await status.boundingBox()).toEqual(statusBefore);
});
