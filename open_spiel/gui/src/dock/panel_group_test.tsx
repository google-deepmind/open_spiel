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

import {act, type JSX, useState} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {press} from '../testing/dom.ts';
import {button, only, render, type Rendered} from '../testing/render.tsx';
import {MemberPanel, PanelGroup, isPaneEmpty} from './panel_group.tsx';
import {
  type Member,
  type PanelGroupController,
  type PanelGroupDefinition,
  PanelGroupContext,
  memberId,
  membersOf,
  usePaneHidden,
} from './panel_groups.ts';

let rendered: Rendered | undefined;
let shown = true;
let count = 3;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  shown = true;
  count = 3;
});

function Hidden(): JSX.Element {
  return <p>{usePaneHidden() ? 'hidden' : 'shown'}</p>;
}

const GROUP: PanelGroupDefinition = {
  id: 'actions',
  title: 'Actions',
  sections: [
    {id: 'history', title: 'History', Content: Hidden},
    {
      id: 'legal',
      title: 'Legal',
      Content: () => <p>Moves</p>,
      Label: () => `Legal (${String(count)})`,
    },
    {
      id: 'text',
      title: 'Text',
      Content: () => (shown ? <input aria-label="Draft" /> : null),
    },
  ],
};

function tabs(): Array<[string, string | null]> {
  return [...document.querySelectorAll('[role="tab"]')].map(tab => [
    tab.textContent,
    tab.getAttribute('aria-selected'),
  ]);
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

/** A controller that records what the group asks of the dock. */
function controller(popped: string[] = []): PanelGroupController {
  return {
    popped: new Set(popped),
    active: () => undefined,
    setActive: vi.fn(),
    popOut: vi.fn(),
    dockBack: vi.fn(),
  };
}

describe('PanelGroup', () => {
  it('shows its sections as tabs, the first open, the rest mounted', () => {
    rendered = render(<PanelGroup group={GROUP} />);
    const list = only(document, '[role="tablist"]');
    expect(list.getAttribute('aria-label')).toBe('Actions sections');
    expect(tabs()).toEqual([
      ['History', 'true'],
      ['Legal (3)', 'false'],
      ['Text', 'false'],
    ]);
    const panes = [...document.querySelectorAll('[role="tabpanel"]')];
    expect(panes.map(pane => (pane as HTMLElement).hidden)).toEqual([
      false,
      true,
      true,
    ]);
    expect(only(document, '[role="tabpanel"]:not([hidden])').textContent).toBe(
      'shown',
    );
    expect(document.querySelector('[aria-label^="Pop out"]')).toBeNull();
  });

  it('moves between its tabs with the arrow keys, wrapping', () => {
    rendered = render(<PanelGroup group={GROUP} />);
    press(button('History'), 'ArrowLeft');
    expect(tabs()[2]).toEqual(['Text', 'true']);
    expect(document.activeElement).toBe(button('Text'));
    press(button('Text'), 'ArrowRight');
    expect(tabs()[0]).toEqual(['History', 'true']);
    const history = only(document, '[role="tabpanel"]:not([hidden])');
    click(button('Legal (3)'));
    expect(history.textContent).toBe('hidden');
  });

  it('draws the footer of the open section only, under its pane', () => {
    const footed: PanelGroupDefinition = {
      ...GROUP,
      sections: GROUP.sections.map(section =>
        section.id === 'history'
          ? {...section, Footer: () => <p>Replay</p>}
          : section,
      ),
    };
    rendered = render(<PanelGroup group={footed} />);
    expect(only(document, '.panel-group__footer').textContent).toBe('Replay');
    click(button('Legal (3)'));
    expect(document.querySelector('.panel-group__footer')).toBeNull();
  });

  it('keeps the section chosen by its caller', () => {
    const onChoose = vi.fn();
    rendered = render(
      <PanelGroup group={GROUP} chosen="text" onChoose={onChoose} />,
    );
    expect(tabs()[2]).toEqual(['Text', 'true']);
    click(button('History'));
    expect(onChoose).toHaveBeenCalledExactlyOnceWith('history');
  });

  it('gives a section that draws nothing no tab', () => {
    shown = false;
    rendered = render(<PanelGroup group={GROUP} />);
    expect(tabs().map(([title]) => title)).toEqual(['History', 'Legal (3)']);
  });
});

describe('PanelGroup, inside a dock', () => {
  it('offers to pop the open section out', () => {
    const dock = controller();
    rendered = render(
      <PanelGroupContext value={dock}>
        <PanelGroup group={GROUP} />
      </PanelGroupContext>,
    );
    click(button('Legal (3)'));
    expect(dock.setActive).toHaveBeenCalledWith('actions', 'legal');
    click(only(document, '[aria-label="Pop out History"]'));
    expect(dock.popOut).toHaveBeenCalledWith({
      group: GROUP,
      section: GROUP.sections[0],
      id: 'actions.history',
    });
  });

  it('leaves out the sections popped out of it', () => {
    rendered = render(
      <PanelGroupContext value={controller(['actions.history'])}>
        <PanelGroup group={GROUP} />
      </PanelGroupContext>,
    );
    expect(tabs()).toEqual([
      ['Legal (3)', 'true'],
      ['Text', 'false'],
    ]);
  });

  it('says so when every section is popped out', () => {
    const all = GROUP.sections.map(({id}) => memberId('actions', id));
    rendered = render(
      <PanelGroupContext value={controller(all)}>
        <PanelGroup group={GROUP} />
      </PanelGroupContext>,
    );
    expect(document.body.textContent).toContain(
      'Every Actions section is open as its own panel.',
    );
  });

  it('keeps a section’s state as tabs change', () => {
    function Keeper(): JSX.Element {
      const [, rerender] = useState(0);
      return (
        <>
          <button
            type="button"
            onClick={() => {
              rerender(value => value + 1);
            }}
          >
            Again
          </button>
          <PanelGroup group={GROUP} />
        </>
      );
    }
    rendered = render(<Keeper />);
    const draft = only(document, 'input');
    click(button('Text'));
    click(button('History'));
    click(button('Again'));
    expect(only(document, 'input')).toBe(draft);
  });
});

describe('MemberPanel', () => {
  const member = (): Member => {
    const found = membersOf([GROUP]).get('actions.legal');
    if (found === undefined) throw new Error('No member.');
    return found;
  };

  it('says the group it came from, and docks back', () => {
    const dock = controller(['actions.legal']);
    const setTitle = vi.fn();
    rendered = render(
      <PanelGroupContext value={dock}>
        <MemberPanel member={member()} setTitle={setTitle} />
      </PanelGroupContext>,
    );
    expect(only(document, '.panel-group__from').textContent).toBe(
      'From Actions',
    );
    expect(document.body.textContent).toContain('Moves');
    expect(setTitle).toHaveBeenCalledWith('Legal (3)');
    click(only(document, '[aria-label="Dock Legal into Actions"]'));
    expect(dock.dockBack).toHaveBeenCalledWith(member());
  });
});

describe('membersOf', () => {
  it('refuses a group without sections and a section defined twice', () => {
    expect(() => membersOf([{id: 'x', title: 'X', sections: []}])).toThrow(
      'Panel group "x" has no sections.',
    );
    expect(() => membersOf([GROUP, GROUP])).toThrow(
      'Section "actions.history" is defined twice.',
    );
  });
});

describe('isPaneEmpty', () => {
  it('finds a pane empty with no element and no text', () => {
    const pane = document.createElement('div');
    expect(isPaneEmpty(pane)).toBe(true);
    pane.textContent = ' ';
    expect(isPaneEmpty(pane)).toBe(true);
    pane.textContent = 'x';
    expect(isPaneEmpty(pane)).toBe(false);
  });
});
