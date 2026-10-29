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
import {CompactText, CopyableText, CopyButton} from './copy_button.tsx';
import {COPY_FEEDBACK_MS} from './use_copy.ts';

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  vi.useRealTimers();
  restoreProperties();
});

function names(button: HTMLElement): [string | null, string | null] {
  return [button.getAttribute('aria-label'), button.getAttribute('title')];
}

describe('CopyButton', () => {
  it('copies its text, says so, and announces it politely', async () => {
    vi.useFakeTimers();
    const writeText = vi.fn(() => Promise.resolve());
    stubProperty(navigator, 'clipboard', {writeText});
    rendered = render(<CopyButton text="[0, 4, 8]" />);
    const button = only(rendered.container, 'button');
    const status = only(rendered.container, '[role="status"]');
    expect(status.getAttribute('aria-live')).toBe('polite');
    expect(names(button)).toEqual(['Copy to clipboard', 'Copy to clipboard']);
    expect(button.textContent).toBe('');
    expect(button.querySelector('.icon--copy')).not.toBeNull();

    await act(async () => {
      button.click();
      await Promise.resolve();
    });
    expect(writeText).toHaveBeenCalledExactlyOnceWith('[0, 4, 8]');
    expect(names(button)).toEqual(['Copied', 'Copied!']);
    expect(button.dataset['status']).toBe('copied');
    expect(button.querySelector('.icon--check')).not.toBeNull();
    expect(status.textContent).toBe('Copied to the clipboard.');

    act(() => {
      vi.advanceTimersByTime(COPY_FEEDBACK_MS);
    });
    expect(names(button)).toEqual(['Copy to clipboard', 'Copy to clipboard']);
    expect(status.textContent).toBe('');
  });

  it('names a failed copy', async () => {
    stubProperty(navigator, 'clipboard', {
      writeText: () => Promise.reject(new Error('refused')),
    });
    stubProperty(document, 'execCommand', () => false);
    rendered = render(<CopyButton text="x" />);
    const button = only(rendered.container, 'button');
    await act(async () => {
      button.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(names(button)).toEqual(['Copy failed', 'Copy failed']);
    expect(button.dataset['status']).toBe('failed');
  });
});

describe('CopyableText', () => {
  it('shows the text as a named, focusable group with its copy button', () => {
    rendered = render(<CopyableText text="x.o" label="Board" />);
    const card = only(rendered.container, '.copyable');
    const text = only(card, 'pre');
    expect(text.textContent).toBe('x.o');
    expect(text.getAttribute('role')).toBe('group');
    expect(text.getAttribute('aria-label')).toBe('Board');
    expect(text.tabIndex).toBe(0);
    expect(only(card, 'button').getAttribute('aria-label')).toBe(
      'Copy to clipboard',
    );
  });
});

describe('CompactText', () => {
  it('shows the text as a named, focusable group with its copy button', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    stubProperty(navigator, 'clipboard', {writeText});
    rendered = render(<CompactText text={'P0: 1\nP1: -1'} label="Returns" />);
    const text = only(rendered.container, 'pre');
    expect(text.textContent).toBe('P0: 1\nP1: -1');
    expect(text.getAttribute('role')).toBe('group');
    expect(text.getAttribute('aria-label')).toBe('Returns');
    expect(text.tabIndex).toBe(0);
    expect(text.classList).toContain('copyable__text--compact');
    const card = only(rendered.container, '.copyable');
    expect(card.classList).toContain('copyable--compact');
    const button = only(card, 'button');
    expect(names(button)).toEqual(['Copy to clipboard', 'Copy to clipboard']);
    await act(async () => {
      button.click();
      await Promise.resolve();
    });
    expect(writeText).toHaveBeenCalledExactlyOnceWith('P0: 1\nP1: -1');
    expect(names(button)).toEqual(['Copied', 'Copied!']);
  });
});

describe('CopyableText as a region', () => {
  it('names the text as a region when asked, and keeps its copy button', () => {
    rendered = render(
      <CopyableText text="x.o" label="State" role="region" className="mine" />,
    );
    const text = only(rendered.container, 'pre');
    expect(text.getAttribute('role')).toBe('region');
    expect(text.classList).toContain('mine');
    expect(text.classList).not.toContain('copyable__text--compact');
    expect(only(rendered.container, '.copyable .copy-button')).toBeTruthy();
  });
});
