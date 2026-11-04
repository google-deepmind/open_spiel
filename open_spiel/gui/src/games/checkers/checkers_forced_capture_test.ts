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

// A forced capture is marked at rest, not only under the pointer.
import {describe, expect, it} from 'vitest';

import {captureIsForced} from './checkers_gameplay_state.ts';

describe('captureIsForced', () => {
  it('is true when every legal move is a jump', () => {
    expect(
      captureIsForced([
        {action: 1, label: 'e3g5', actionType: 'jump'},
        {action: 2, label: 'c3e5', actionType: 'jump'},
      ]),
    ).toBe(true);
  });

  it('is false with a plain move among them, or with no move at all', () => {
    expect(
      captureIsForced([
        {action: 1, label: 'e3f4', actionType: 'move'},
        {action: 2, label: 'c3e5', actionType: 'jump'},
      ]),
    ).toBe(false);
    expect(captureIsForced([])).toBe(false);
  });
});
