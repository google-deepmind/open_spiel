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

import {act, useState, type JSX} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {button, only, render, type Rendered} from '../testing/render.tsx';
import {PanelDrawer, type ClosedPanel} from './panel_drawer.tsx';

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
});

function show(closed: readonly ClosedPanel[], onOpen = vi.fn()): void {
  rendered = render(<PanelDrawer closed={closed} onOpen={onOpen} />);
}

function trigger(): HTMLElement {
  return only(document, 'button[aria-haspopup="dialog"]');
}

function sheet(): HTMLElement {
  return only(document, '[role="dialog"]');
}

function closeButton(): HTMLElement {
  return only(sheet(), 'button[aria-label="Close panel options"]');
}

function press(element: Element): void {
  act(() => {
    element.dispatchEvent(new MouseEvent('pointerdown', {bubbles: true}));
  });
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

function escape(element: Element): void {
  act(() => {
    element.dispatchEvent(
      new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
    );
  });
}

/** A drawer whose reopened panels leave its list, as the dock's do. */
function Reopening({titles}: {titles: readonly string[]}): JSX.Element {
  const [closed, setClosed] = useState(titles);
  return (
    <PanelDrawer
      closed={closed.map(title => ({id: title, title}))}
      onOpen={id => {
        setClosed(closed.filter(title => title !== id));
      }}
    />
  );
}

describe('PanelDrawer', () => {
  it('starts closed, its button naming the drawer it controls', () => {
    show([]);
    expect(trigger().getAttribute('aria-label')).toBe('Panel options');
    expect(trigger().getAttribute('title')).toBe('Panel options');
    expect(trigger().querySelector('.icon--kebab')).not.toBeNull();
    expect(trigger().getAttribute('aria-expanded')).toBe('false');
    expect(trigger().getAttribute('aria-controls')).toBe(sheet().id);
    expect(sheet().hidden).toBe(true);
  });

  it('lists the closed panels in order when opened', () => {
    show([
      {id: 'actions', title: 'Actions'},
      {id: 'session', title: 'Session'},
    ]);
    click(trigger());
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(sheet().hidden).toBe(false);
    const items = [...sheet().querySelectorAll('li button')];
    expect(items.map(item => item.textContent)).toEqual(['Actions', 'Session']);
    expect(only(sheet(), 'h3').textContent).toBe('Closed panels');
  });

  it('says so when every panel is open', () => {
    show([]);
    click(trigger());
    expect(sheet().textContent).toContain('All panels are open');
    expect(sheet().querySelector('li')).toBeNull();
  });

  it('reopens a panel and stays open', () => {
    const onOpen = vi.fn();
    show([{id: 'actions', title: 'Actions'}], onOpen);
    click(trigger());
    click(only(sheet(), 'li button'));
    expect(onOpen).toHaveBeenCalledExactlyOnceWith('actions');
    expect(sheet().hidden).toBe(false);
  });

  it('closes on its labelled close button', () => {
    show([]);
    click(trigger());
    click(only(sheet(), 'button[aria-label="Close panel options"]'));
    expect(sheet().hidden).toBe(true);
  });

  it('closes on Escape inside it and returns focus to its button', () => {
    show([]);
    click(trigger());
    escape(only(sheet(), 'button'));
    expect(sheet().hidden).toBe(true);
    expect(document.activeElement).toBe(trigger());
  });

  it('ignores Escape pressed elsewhere', () => {
    show([]);
    click(trigger());
    escape(document.body);
    expect(sheet().hidden).toBe(false);
  });

  it('is labelled by its heading', () => {
    show([]);
    const heading = only(sheet(), 'h2');
    expect(heading.textContent).toBe('Panels');
    expect(sheet().getAttribute('aria-labelledby')).toBe(heading.id);
  });
});

describe('PanelDrawer focus', () => {
  it('moves focus to the first panel listed when opened', () => {
    show([
      {id: 'actions', title: 'Actions'},
      {id: 'session', title: 'Session'},
    ]);
    click(trigger());
    expect(document.activeElement).toBe(button('Actions'));
  });

  it('moves focus to its close button when opened with nothing listed', () => {
    show([]);
    click(trigger());
    expect(document.activeElement).toBe(closeButton());
  });

  it('moves focus to the panel listed next, then to its close button', () => {
    rendered = render(<Reopening titles={['Actions', 'State', 'Session']} />);
    click(trigger());
    click(button('State'));
    expect(document.activeElement).toBe(button('Session'));
    click(button('Session'));
    expect(document.activeElement).toBe(button('Actions'));
    click(button('Actions'));
    expect(document.activeElement).toBe(closeButton());
  });

  it('closes on a press outside it, not inside', () => {
    show([{id: 'actions', title: 'Actions'}]);
    click(trigger());
    press(only(sheet(), 'li button'));
    expect(sheet().hidden).toBe(false);
    press(document.body);
    expect(sheet().hidden).toBe(true);
  });

  it('toggles from its button', () => {
    show([]);
    click(trigger());
    click(trigger());
    expect(sheet().hidden).toBe(true);
  });
});
