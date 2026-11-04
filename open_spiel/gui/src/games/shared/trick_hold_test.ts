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

// The session-to-feed glue of the trick collect: what counts as scrubbing.
import {describe, expect, it} from 'vitest';
import {
  detectCompletedTrick,
  type CompletedTrick,
} from '../../cards/library/index.ts';
import {replayScrubbing, trickFeedOf} from './trick_hold.ts';
import type {TrickSession} from './trick_hold.ts';

function trick(index: number): CompletedTrick {
  return {index, cards: [{card: '2c', seat: 'bottom'}], winner: 'bottom'};
}
function payload(history: number, o: Partial<TrickSession> = {}): TrickSession {
  return {
    sessionId: 's1',
    state: {history: Array.from({length: history}, (_, i) => i)},
    ...o,
  };
}

describe('replayScrubbing', () => {
  it('is a replay short of its end, not one at the live end', () => {
    expect(replayScrubbing({isActive: false, step: 0, totalSteps: 0})).toBe(
      false,
    );
    // Analysis mode: the Actions panel keeps a replay active at the end.
    expect(replayScrubbing({isActive: true, step: 60, totalSteps: 60})).toBe(
      false,
    );
    // A live move arrives before the replay catches up: still live.
    expect(replayScrubbing({isActive: true, step: 64, totalSteps: 60})).toBe(
      false,
    );
    expect(replayScrubbing({isActive: true, step: 40, totalSteps: 60})).toBe(
      true,
    );
  });
});

describe('trickFeedOf', () => {
  it('reads session, history length and the history view', () => {
    expect(trickFeedOf(null, false, null)).toBeNull();
    expect(trickFeedOf(payload(5), false, trick(1))).toEqual({
      session: 's1',
      history: 5,
      scrubbing: false,
      last: trick(1),
    });
    expect(
      trickFeedOf(payload(5, {historyView: {step: 3, length: 9}}), false, null)
        ?.scrubbing,
    ).toBe(true);
    expect(trickFeedOf(payload(5), true, null)?.scrubbing).toBe(true);
  });

  it('a live move in analysis mode collects; stepping back to the end does not', () => {
    const live = trickFeedOf(payload(60), false, trick(4));
    const moved = trickFeedOf(payload(64), false, trick(5));
    expect(detectCompletedTrick(live, moved)?.index).toBe(5);
    const viewing = trickFeedOf(
      payload(40, {historyView: {step: 40, length: 64}}),
      false,
      trick(3),
    );
    expect(detectCompletedTrick(viewing, moved)).toBeNull();
  });
});
