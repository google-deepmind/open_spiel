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
// text it copies.

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
