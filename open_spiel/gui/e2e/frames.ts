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

// Every frame the page receives, in the order the responses arrive, so a
// test can compare what the page shows with what the server sent it.

import {type Page, type Response, expect} from '@playwright/test';

import type {Frame, Viewer} from '../src/contract/contract.ts';

/** Who a test views a game as: the oracle, a spectator or a seat index. */
export type ViewerChoice = 'oracle' | 'spectator' | number;

/** Whether `viewer` is the viewer `choice` names. */
export function isViewer(viewer: Viewer, choice: ViewerChoice): boolean {
  if (typeof choice === 'number') {
    return viewer.kind === 'seat' && viewer.seat === choice;
  }
  return viewer.kind === choice;
}

function isSessionResponse(response: Response): boolean {
  const {pathname} = new URL(response.url());
  return pathname.startsWith('/api/sessions') && response.ok();
}

/** The frames the page has received. */
export class FrameLog {
  private readonly frames: Frame[] = [];
  private readonly pending = new Set<Promise<void>>();

  constructor(page: Page) {
    page.on('response', response => {
      if (!isSessionResponse(response)) return;
      const read = response
        .json()
        .then((body: unknown) => {
          this.frames.push(body as Frame);
        })
        // The page navigated away and the body is gone; the page never
        // used that frame either.
        .catch(() => undefined)
        .finally(() => {
          this.pending.delete(read);
        });
      this.pending.add(read);
    });
  }

  /** How many frames have arrived. */
  get count(): number {
    return this.frames.length;
  }

  /** Every frame received, oldest first. */
  all(): readonly Frame[] {
    return this.frames;
  }

  /**
   * Waits for a frame received after the first `after` that satisfies
   * `predicate`, and returns the latest such frame.
   *
   * @throws Error when none arrives within the expect timeout.
   */
  async waitFor(
    predicate: (frame: Frame) => boolean,
    after = 0,
    what = 'a matching frame',
  ): Promise<Frame> {
    let found: Frame | undefined;
    await expect
      .poll(
        async () => {
          await Promise.all(this.pending);
          found = this.frames.slice(after).reverse().find(predicate);
          return found !== undefined;
        },
        {message: `waiting for ${what}`},
      )
      .toBe(true);
    if (found === undefined) throw new Error(`No ${what} arrived.`);
    return found;
  }

  /** The latest frame of `game` for `viewer`, waiting for one if need be. */
  latest(game: string, viewer: ViewerChoice): Promise<Frame> {
    return this.waitFor(
      frame => frame.session.game === game && isViewer(frame.viewer, viewer),
      0,
      `a ${game} frame for viewer ${String(viewer)}`,
    );
  }
}
