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
import {restoreProperties, stubProperty} from '../testing/stub_property.ts';
import {StatePanel} from './state_panel.tsx';
import {recordedFrame} from './testing/recorded_frames.ts';

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  restoreProperties();
});

async function show(name: string): Promise<HTMLElement> {
  rendered = render(<StatePanel frame={await recordedFrame(name)} />);
  return rendered.container;
}

function tabs(container: HTMLElement): Array<string | null> {
  return Array.from(
    container.querySelectorAll('[role="tab"]'),
    tab => tab.textContent,
  );
}

function openTab(container: HTMLElement, title: string): HTMLElement {
  const tab = Array.from(
    container.querySelectorAll<HTMLElement>('[role="tab"]'),
  ).find(candidate => candidate.textContent === title);
  if (tab === undefined) throw new Error(`No tab ${title}.`);
  act(() => {
    tab.click();
  });
  return only(container, '[role="tabpanel"]:not([hidden])');
}

describe('StatePanel', () => {
  it('offers a tab per section the frame has, Returns before Rewards', async () => {
    const container = await show('kuhn_poker/oracle/live');
    expect(only(container, '[role="tablist"]').getAttribute('aria-label')).toBe(
      'State sections',
    );
    expect(tabs(container)).toEqual(['Text', 'Returns', 'Rewards']);
    const text = only(container, '[role="tabpanel"]:not([hidden])');
    expect(only(text, 'pre').getAttribute('aria-label')).toBe('Text');
    expect(only(text, 'pre').tabIndex).toBe(0);
  });

  it('shows returns and rewards as compact text with no copy icon', async () => {
    const container = await show('kuhn_poker/oracle/live');
    const returns = openTab(container, 'Returns');
    expect(returns.querySelector('button')).toBeNull();
    const text = only(returns, 'pre');
    expect(text.classList).toContain('copyable__text--compact');
    expect(text.textContent).toBe('Player 0: -2\nPlayer 1: 2');
  });

  it('copies the text with the icon in its card', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    stubProperty(navigator, 'clipboard', {writeText});
    const container = await show('kuhn_poker/0/3');
    expect(tabs(container)).toEqual(['Observation', 'Rewards']);
    const copy = only(container, '.copy-button');
    expect(copy.getAttribute('aria-label')).toBe('Copy to clipboard');
    await act(async () => {
      copy.click();
      await Promise.resolve();
    });
    expect(writeText).toHaveBeenCalledExactlyOnceWith('011');
  });

  it('says why the text is missing, with nothing to copy', async () => {
    const container = await show('kuhn_poker/spectator/3');
    expect(tabs(container)).toEqual(['Text']);
    expect(container.querySelector('pre')).toBeNull();
    expect(container.querySelector('.copy-button')).toBeNull();
    expect(only(container, 'p').textContent).toBe(
      'No public state text for this game.',
    );
  });

  it('keeps a board’s columns as the game laid them out', async () => {
    const container = await show('tic_tac_toe/oracle/live');
    expect(tabs(container)).toEqual(['Text', 'Struct', 'Rewards']);
    expect(
      only(container, '[role="tabpanel"]:not([hidden]) .copyable pre')
        .textContent,
    ).toBe('o..\n.x.\n..x');
  });
});
