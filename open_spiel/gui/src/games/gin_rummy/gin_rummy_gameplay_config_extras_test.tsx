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

import {act} from 'react';
import {afterEach, describe, expect, it} from 'vitest';

import {getDeck} from '../../cards/library/index.ts';
import {only, type Rendered, render} from '../../testing/render.tsx';
import {GinRummyVisualsSettings} from './gin_rummy_gameplay_config_extras.tsx';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
  localStorage.clear();
});

describe('GinRummyVisualsSettings card style', () => {
  it('names the shared deck and the simple deck by its label', () => {
    rendered = render(<GinRummyVisualsSettings />);
    const label = [...rendered.container.querySelectorAll('label')].find(
      l => l.querySelector('span')?.textContent === 'Card style',
    );
    const select = label?.querySelector('select');
    if (!select) throw new Error('no Card style select');
    expect([...select.options].map(o => [o.value, o.textContent])).toEqual([
      ['app', 'As the card deck setting'],
      ['modern', 'Centred rank and suit'],
    ]);
  });

  it('shows the picked face\'s "best for" line under the picker', () => {
    rendered = render(<GinRummyVisualsSettings />);
    const {container} = rendered;
    const select = only(container, 'select:has(option[value="modern"])');
    if (!(select instanceof HTMLSelectElement)) throw new Error('no select');
    const note = (): string | null =>
      only(container, '[data-testid="gin-face-description"]').textContent;
    expect(note()).toMatch(/Card deck/);
    act(() => {
      select.value = 'modern';
      select.dispatchEvent(new Event('change', {bubbles: true}));
    });
    expect(note()).toBe(getDeck('simple').description);
    act(() => {
      select.value = 'app';
      select.dispatchEvent(new Event('change', {bubbles: true}));
    });
  });
});
