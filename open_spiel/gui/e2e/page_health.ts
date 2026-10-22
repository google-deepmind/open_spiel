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

// Collects what a page does wrong while a test drives it: console errors,
// uncaught exceptions, requests that fail and responses that refuse.

import type {Page} from '@playwright/test';

/** What went wrong on a page, one sentence each. */
export class PageHealth {
  private readonly problems: string[] = [];

  constructor(page: Page) {
    page.on('console', message => {
      if (message.type() === 'error') {
        this.problems.push(`console error: ${message.text()}`);
      }
    });
    page.on('pageerror', error => {
      this.problems.push(`uncaught: ${error.message}`);
    });
    page.on('requestfailed', request => {
      const failure = request.failure()?.errorText ?? 'unknown failure';
      // A reload or navigation cancels the requests in flight; that is the
      // browser's doing, not a failure of the page or the server.
      if (failure === 'net::ERR_ABORTED') return;
      this.problems.push(
        `request failed: ${request.method()} ${request.url()}: ${failure}`,
      );
    });
    page.on('response', response => {
      if (response.status() >= 400) {
        const request = response.request();
        this.problems.push(
          `response ${String(response.status())}: ${request.method()} ${response.url()}`,
        );
      }
    });
  }

  /** Every problem seen so far. */
  list(): readonly string[] {
    return this.problems;
  }
}
