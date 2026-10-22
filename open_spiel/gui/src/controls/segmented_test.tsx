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
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {Rendered} from '../testing/render.tsx';
import {only, render} from '../testing/render.tsx';
import {Segmented} from './segmented.tsx';

const OPTIONS = [
  {value: 'strings', label: 'Strings'},
  {value: 'ints', label: 'Ints'},
] as const;

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  vi.restoreAllMocks();
});

describe('Segmented', () => {
  it('is a group named by its visible caption', () => {
    rendered = render(
      <Segmented
        label="Values"
        value="ints"
        options={OPTIONS}
        onChange={vi.fn()}
      />,
    );
    const group = only(rendered.container, '[role="group"]');
    const caption = document.getElementById(
      group.getAttribute('aria-labelledby') ?? '',
    );
    expect(caption?.textContent).toBe('Values');
  });

  it('presses the chosen option only', () => {
    rendered = render(
      <Segmented
        label="Values"
        value="ints"
        options={OPTIONS}
        onChange={vi.fn()}
      />,
    );
    const pressed = Array.from(
      rendered.container.querySelectorAll('button'),
      button => [button.textContent, button.getAttribute('aria-pressed')],
    );
    expect(pressed).toEqual([
      ['Strings', 'false'],
      ['Ints', 'true'],
    ]);
  });

  it('reports the option clicked', () => {
    const onChange = vi.fn();
    const view = render(
      <Segmented
        label="Values"
        value="ints"
        options={OPTIONS}
        onChange={onChange}
      />,
    );
    rendered = view;
    act(() => {
      only(view.container, 'button[aria-pressed="false"]').click();
    });
    expect(onChange).toHaveBeenCalledExactlyOnceWith('strings');
  });

  it('refuses a value that is not an option', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    expect(() =>
      render(
        <Segmented
          label="Values"
          value={'floats' as 'ints'}
          options={OPTIONS}
          onChange={vi.fn()}
        />,
      ),
    ).toThrow('Values: "floats" is not one of the options.');
  });
});
