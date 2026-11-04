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
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {fakeColorScheme} from '../testing/color_scheme.ts';
import type {Rendered} from '../testing/render.tsx';
import {only, render} from '../testing/render.tsx';
import {restoreProperties, stubProperty} from '../testing/stub_property.ts';
import {ThemeProvider} from '../theme/theme_provider.tsx';
import {CopyButton} from './copy_button.tsx';
import {SettingsMenu} from './settings_menu.tsx';

let rendered: Rendered;

function trigger(): HTMLElement {
  return only(document, 'button[aria-haspopup="dialog"]');
}

function dialog(): HTMLElement | null {
  return document.querySelector('[role="dialog"]');
}

function open(): HTMLElement {
  act(() => {
    trigger().click();
  });
  const opened = dialog();
  if (opened === null) {
    throw new Error('The settings dialog did not open.');
  }
  return opened;
}

function press(key: string): void {
  act(() => {
    document.activeElement?.dispatchEvent(
      new KeyboardEvent('keydown', {key, bubbles: true}),
    );
  });
}

function pointerDown(target: Element): void {
  act(() => {
    // jsdom has no PointerEvent; the menu reads only the target.
    target.dispatchEvent(new MouseEvent('pointerdown', {bubbles: true}));
  });
}

function labels(parent: ParentNode): Array<string | null> {
  return Array.from(
    parent.querySelectorAll('button'),
    button => button.textContent,
  );
}

beforeEach(() => {
  fakeColorScheme(false);
  rendered = render(
    <ThemeProvider>
      <button id="outside" type="button">
        Outside
      </button>
      <SettingsMenu>
        <p id="extra">More settings</p>
        <CopyButton text="[0, 4, 8]" />
        <button id="last" type="button">
          Last
        </button>
      </SettingsMenu>
    </ThemeProvider>,
  );
});

afterEach(() => {
  rendered.unmount();
  window.localStorage.clear();
  delete document.documentElement.dataset['theme'];
  vi.unstubAllGlobals();
  restoreProperties();
});

describe('SettingsMenu', () => {
  it('opens a dialog it names and controls', () => {
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(dialog()).toBeNull();
    const opened = open();
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(trigger().getAttribute('aria-controls')).toBe(opened.id);
    expect(opened.getAttribute('aria-label')).toBe('Settings');
    expect(opened.querySelector('#extra')?.textContent).toBe('More settings');
  });

  it('closes on a click outside, and not on one inside', () => {
    const opened = open();
    pointerDown(only(opened, '#extra'));
    expect(dialog()).not.toBeNull();
    pointerDown(only(document, '#outside'));
    expect(dialog()).toBeNull();
  });

  it('toggles from its button', () => {
    open();
    act(() => {
      trigger().click();
    });
    expect(dialog()).toBeNull();
  });

  it('sets the theme', () => {
    const opened = open();
    const theme = only(opened, '[role="group"]');
    expect(labels(theme)).toEqual(['System', 'Light', 'Dark']);
    const dark = Array.from(theme.querySelectorAll('button')).find(
      button => button.textContent === 'Dark',
    );
    act(() => {
      dark?.click();
    });
    expect(document.documentElement.dataset['theme']).toBe('dark');
    expect(dark?.getAttribute('aria-pressed')).toBe('true');
  });
});

describe('SettingsMenu focus and keys', () => {
  it('closes on Escape and returns focus to its button', () => {
    const opened = open();
    only(opened, 'button[aria-pressed="true"]').focus();
    press('Escape');
    expect(dialog()).toBeNull();
    expect(document.activeElement).toBe(trigger());
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
  });

  it('moves focus into the dialog as it opens', () => {
    const opened = open();
    expect(document.activeElement).toBe(opened.querySelector('button'));
  });

  it('handles an Escape pressed inside it, and only that one', () => {
    open();
    const reached: Element[] = [];
    const listener = (event: KeyboardEvent): void => {
      if (event.target instanceof Element) {
        reached.push(event.target);
      }
    };
    document.addEventListener('keydown', listener);
    const outside = only(document, '#outside');
    act(() => {
      outside.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
      );
    });
    expect(dialog()).not.toBeNull();
    press('Escape');
    document.removeEventListener('keydown', listener);
    expect(dialog()).toBeNull();
    expect(reached).toEqual([outside]);
  });

  it('closes when focus moves outside it, and not within it', () => {
    const opened = open();
    act(() => {
      only(opened, '#last').focus();
    });
    expect(dialog()).not.toBeNull();
    act(() => {
      only(document, '#outside').focus();
    });
    expect(dialog()).toBeNull();
  });

  it('stays open through a copy that borrows the focus', async () => {
    stubProperty(navigator, 'clipboard', undefined);
    const execCommand = vi.fn(() => true);
    stubProperty(document, 'execCommand', execCommand);
    const copyButton = only(open(), '.copy-button');
    copyButton.focus();
    await act(async () => {
      copyButton.click();
      await Promise.resolve();
    });
    expect(execCommand).toHaveBeenCalledExactlyOnceWith('copy');
    expect(dialog()).not.toBeNull();
    expect(document.activeElement).toBe(copyButton);
  });

  it('ignores other keys', () => {
    open();
    press('Enter');
    expect(dialog()).not.toBeNull();
  });
});
