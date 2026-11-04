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

// The test function every end-to-end test uses. It points the page at the
// server, gives the test the `App` and the frames the page received, and
// fails the test if the page logged an error or a request failed.

/// <reference types="node" />

import process from 'node:process';

import {test as base, expect} from '@playwright/test';

import {App} from './app.ts';
import {FrameLog} from './frames.ts';
import {BASE_URL_VARIABLE} from './global_setup.ts';
import {PageHealth} from './page_health.ts';

interface Fixtures {
  readonly app: App;
  readonly frames: FrameLog;
  readonly health: PageHealth;
}

/** Playwright's `test`, with the client's fixtures. */
export const test = base.extend<Fixtures>({
  // Playwright reads a fixture's dependencies from its first parameter,
  // which must be an object pattern even when empty.
  // eslint-disable-next-line no-empty-pattern
  baseURL: async ({}, provide) => {
    const url = process.env[BASE_URL_VARIABLE];
    if (url === undefined || url === '') {
      throw new Error(`${BASE_URL_VARIABLE} is unset: no server was started.`);
    }
    await provide(url);
  },
  health: [
    async ({page}, provide) => {
      const health = new PageHealth(page);
      await provide(health);
      expect(health.list(), 'console errors and failed requests').toEqual([]);
    },
    {auto: true},
  ],
  frames: async ({page}, provide) => {
    await provide(new FrameLog(page));
  },
  app: async ({page, frames}, provide) => {
    await provide(new App(page, frames));
  },
});

export {expect};
