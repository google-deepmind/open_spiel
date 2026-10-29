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

import {InGame} from '../../testing/game.tsx';
import {only, type Rendered, render} from '../../testing/render.tsx';
import {EuchreGameplayConfigExtras} from './euchre_gameplay_config_extras.tsx';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

describe('EuchreGameplayConfigExtras card style', () => {
  it("names each style by its deck's label and shows its best-for line", () => {
    rendered = render(
      <InGame viewer={{kind: 'oracle', seat: null}}>
        <EuchreGameplayConfigExtras />
      </InGame>,
    );
    const {container} = rendered;
    const select = only(container, 'select:has(option[value="basic"])');
    if (!(select instanceof HTMLSelectElement)) throw new Error('no select');
    expect([...select.options].map(o => [o.value, o.textContent])).toEqual([
      ['basic', 'Corner index + large pip'],
      ['simple-corner', 'Large corner index'],
      ['simple', 'Centred rank and suit'],
    ]);
    expect(select.value).toBe('basic');
    const hint = (): string | null =>
      only(container, '[data-testid="deck-description"]').textContent;
    expect(hint()).toMatch(/^Best for table piles and tricks/);

    act(() => {
      select.value = 'simple-corner';
      select.dispatchEvent(new Event('change', {bubbles: true}));
    });
    expect(hint()).toMatch(/^Best for fanned hands/);
    act(() => {
      select.value = 'basic';
      select.dispatchEvent(new Event('change', {bubbles: true}));
    });
  });
});

describe('EuchreGameplayConfigExtras Show all cards', () => {
  function showAll(container: HTMLElement): HTMLInputElement {
    const label = [...container.querySelectorAll('label')].find(
      candidate => candidate.textContent === 'Show all cards',
    );
    const input = label?.querySelector('input');
    if (!(input instanceof HTMLInputElement)) throw new Error('No toggle.');
    return input;
  }

  it('may be turned on in analysis', () => {
    rendered = render(
      <InGame viewer={{kind: 'oracle', seat: null}}>
        <EuchreGameplayConfigExtras />
      </InGame>,
    );
    expect(showAll(rendered.container).disabled).toBe(false);
    expect(rendered.container.textContent).not.toContain(
      'Cards can only be revealed in analysis mode.',
    );
  });

  it('is disabled in a game, with the reason', () => {
    rendered = render(
      <InGame viewer={{kind: 'seat', seat: 0}}>
        <EuchreGameplayConfigExtras />
      </InGame>,
    );
    expect(showAll(rendered.container).disabled).toBe(true);
    expect(rendered.container.textContent).toContain(
      'Cards can only be revealed in analysis mode.',
    );
  });
});
