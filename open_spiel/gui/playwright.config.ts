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

// The end-to-end tests: a headless Chromium driving the built client, served
// by the Python server that `e2e/global_setup.ts` starts. Run them with
// `npm run e2e`.

/// <reference types="node" />

import process from 'node:process';

import {defineConfig, devices} from '@playwright/test';

const CI = process.env['CI'] !== undefined;

export default defineConfig({
  testDir: 'e2e',
  testMatch: '**/*_e2e.ts',
  globalSetup: './e2e/global_setup.ts',
  // The tests share one server, and a search controller's moves use a whole
  // core, so they run one at a time.
  workers: 1,
  fullyParallel: false,
  forbidOnly: CI,
  retries: 0,
  // Milliseconds. A game against a search controller plays several searches.
  timeout: 180_000,
  expect: {timeout: 15_000},
  reporter: CI ? [['list'], ['html', {open: 'never'}]] : 'list',
  use: {
    ...devices['Desktop Chrome'],
    // The dock lays out three columns from 1600 pixels.
    viewport: {width: 1680, height: 1050},
    headless: true,
    permissions: ['clipboard-read', 'clipboard-write'],
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{name: 'chromium'}],
});
