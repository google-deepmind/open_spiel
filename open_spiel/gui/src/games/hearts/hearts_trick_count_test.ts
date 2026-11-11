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

// The trick counter follows the table, not the engine.
import {describe, expect, it} from 'vitest';

import {shownTrickNumber} from './hearts_gameplay_panel.tsx';

describe('shownTrickNumber', () => {
  it('does not count a finished trick still on show', () => {
    // The engine has completed four tricks; the fourth is still on the table.
    expect(shownTrickNumber(4, 1, 13)).toBe(4);
    expect(shownTrickNumber(4, 0, 13)).toBe(5);
  });

  it('stops at the last trick of the hand', () => {
    expect(shownTrickNumber(13, 0, 13)).toBe(13);
  });

  it('fails loud on more tricks on show than completed', () => {
    expect(() => shownTrickNumber(0, 1, 13)).toThrow(/on show/);
  });
});
