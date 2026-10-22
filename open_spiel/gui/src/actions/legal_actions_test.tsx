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

import {act, type JSX} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {Action, Frame, Viewer} from '../contract/contract.ts';
import {SessionClient} from '../session/client.ts';
import {FakeServer} from '../session/testing/fake_server.ts';
import type {Session} from '../session/use_session.ts';
import {useSession} from '../session/use_session.ts';
import {
  layOut,
  restoreLayout,
  scrollTo,
  stubResizeObserver,
} from '../testing/dom.ts';
import {actions, entries, frame, session} from '../testing/frames.ts';
import {only} from '../testing/render.tsx';
import {OVERSCAN_ROWS, ROW_GAP_PX, ROW_STRIDE_PX} from './action_list.ts';
import {LegalActions, legalLabel} from './legal_actions.tsx';

const SPECTATOR: Viewer = {kind: 'spectator', seat: null};
const VIEW_ROWS = 16;
const COLUMNS = 'abcdefghijklmnopqrst';

// A 20x20 board's placements, "B a1" to "B t20", listed in reverse id order.
const BOARD: Action[] = actions(400, id => {
  const column = COLUMNS[Math.floor(id / 20)] ?? '?';
  return `B ${column}${String((id % 20) + 1)}`;
}).reverse();

let container: HTMLElement;
let root: Root;

function show(shown: Session, hidden = false): void {
  act(() => {
    root.render(<LegalActions session={shown} hidden={hidden} />);
  });
}

function rows(): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>('.legal__row'));
}

function idOf(label: string): number {
  const action = BOARD.find(candidate => candidate.label === label);
  if (action === undefined) throw new Error(`No action ${label}.`);
  return action.id;
}

function text(selector: string): string | null {
  return only(container, selector).textContent;
}

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  stubResizeObserver();
  layOut(HTMLDivElement.prototype, {
    clientHeight: VIEW_ROWS * ROW_STRIDE_PX,
    scrollHeight: 400 * ROW_STRIDE_PX,
  });
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  restoreLayout();
  vi.unstubAllGlobals();
});

describe('LegalActions with 400 actions', () => {
  it('renders only the rows in view, with no filter', () => {
    const shown = session(frame({legal_actions: BOARD}));
    show(shown);
    expect(container.querySelector('input')).toBeNull();
    const scroller = only(container, '[role="list"]');
    expect(scroller.getAttribute('aria-label')).toBe('Legal (400)');
    expect(rows()).toHaveLength(VIEW_ROWS + OVERSCAN_ROWS);
    expect(rows()[0]?.textContent).toBe('0B a1');

    scrollTo(scroller, 200 * ROW_STRIDE_PX);
    expect(rows()).toHaveLength(VIEW_ROWS + 2 * OVERSCAN_ROWS);
    expect(rows()[0]?.textContent).toBe(`${String(200 - OVERSCAN_ROWS)}B j9`);
    act(() => {
      rows()[OVERSCAN_ROWS]?.click();
    });
    expect(shown.commands.act).toHaveBeenCalledExactlyOnceWith(200);
  });

  it('gives each windowed row the stride less the gap', () => {
    show(session(frame({legal_actions: BOARD})));
    const slot = container.querySelector<HTMLElement>('.legal-window__slot');
    expect(slot?.style.height).toBe(`${String(ROW_STRIDE_PX - ROW_GAP_PX)}px`);
  });
});

describe('LegalActions', () => {
  it('lists a short list in id order as buttons', () => {
    const shown = session(frame({legal_actions: actions(3).reverse()}));
    show(shown);
    expect(only(container, 'ul').getAttribute('aria-label')).toBe('Legal (3)');
    expect(rows().map(row => [row.tagName, row.textContent])).toEqual([
      ['BUTTON', '0action 0'],
      ['BUTTON', '1action 1'],
      ['BUTTON', '2action 2'],
    ]);
    act(() => {
      rows()[2]?.click();
    });
    expect(shown.commands.act).toHaveBeenCalledExactlyOnceWith(2);
  });

  it('draws up to 200 actions whole', () => {
    show(session(frame({legal_actions: actions(200)})));
    expect(container.querySelector('[role="list"]')).toBeNull();
    expect(rows()).toHaveLength(200);
  });

  it('lists the chance outcomes the client picks, with their odds', () => {
    const outcomes = [
      {id: 1, label: 'Deal Q', probability: 1 / 3},
      {id: 0, label: 'Deal J', probability: 2 / 3},
    ];
    const shown = session(
      frame({
        to_move: {kind: 'chance', seat: null},
        legal_actions: [],
        chance_outcomes: outcomes,
      }),
    );
    show(shown);
    expect(only(container, 'ul').getAttribute('aria-label')).toBe('Legal (2)');
    expect(rows().map(row => row.textContent)).toEqual([
      '0Deal Jp=0.67',
      '1Deal Qp=0.33',
    ]);
    act(() => {
      rows()[1]?.click();
    });
    expect(shown.commands.chooseChance).toHaveBeenCalledExactlyOnceWith(1);
    expect(shown.commands.act).not.toHaveBeenCalled();
    expect(legalLabel(shown.frame)).toBe('Legal (2)');
  });
});

describe('LegalActions refusals', () => {
  it('refuses a press while a request is in flight, keeping the focus', () => {
    const shown = session(frame({legal_actions: actions(40)}), true);
    show(shown);
    const row = rows()[1];
    expect(row?.getAttribute('aria-disabled')).toBe('true');
    act(() => {
      row?.focus();
      row?.click();
    });
    expect(document.activeElement).toBe(row);
    expect(shown.commands.act).not.toHaveBeenCalled();
  });

  it('shows a spectator the actions as text, with why they cannot play', () => {
    const shown = session(
      frame({viewer: SPECTATOR, legal_actions: actions(40)}),
    );
    show(shown);
    expect(text('.legal > [role="status"]')).toBe(
      'Spectating. Claim a seat to act.',
    );
    expect(container.querySelector('button')).toBeNull();
    expect(rows()).toHaveLength(40);
  });

  it('lets an omniscient viewer play from an earlier position', () => {
    const history = entries(['a', 'b']);
    const shown = session(frame({history, ply: 1, legal_actions: actions(2)}));
    show(shown);
    act(() => {
      rows()[0]?.click();
    });
    expect(shown.commands.act).toHaveBeenCalledExactlyOnceWith(0);
  });
});

describe('LegalActions with nothing to play', () => {
  it('says why a seat whose list is withheld cannot play', () => {
    const waiting: Frame = frame({
      viewer: {kind: 'seat', seat: 1},
      omniscient: false,
      legal_actions: null,
    });
    show(session(waiting));
    expect(container.textContent).toBe('Waiting for your turn…');
    expect(legalLabel(null)).toBe('Legal');
    expect(legalLabel(waiting)).toBe('Legal');
  });

  it('says a list is withheld when nothing blocks the viewer', () => {
    const seat0: Viewer = {kind: 'seat', seat: 0};
    show(session(frame({viewer: seat0, legal_actions: null})));
    expect(container.textContent).toBe('The legal actions are withheld.');
  });

  it('says the game is over rather than that nothing is legal', () => {
    const over = {kind: 'terminal', seat: null} as const;
    show(session(frame({to_move: over, legal_actions: []})));
    expect(container.textContent).toBe('Game finished. No legal actions.');
    expect(legalLabel(frame({to_move: over, legal_actions: []}))).toBe('Legal');
  });

  it('says so when nothing is legal', () => {
    show(session(frame({legal_actions: []})));
    expect(container.textContent).toBe('No legal actions available.');
  });

  it('says so when it has no frame', () => {
    show(session(null));
    expect(container.textContent).toBe('No position to show.');
  });
});

describe('LegalActions rows', () => {
  it('flags labels that several actions share', () => {
    const shared = [
      {id: 4, label: 'Draw'},
      {id: 9, label: 'Draw'},
      {id: 5, label: 'Pass'},
    ];
    show(session(frame({legal_actions: shared})));
    expect(container.querySelector('.legal > p')?.textContent).toBe(
      '2 actions share their label with another; their ids tell them apart.',
    );
    expect(rows().map(row => row.getAttribute('data-duplicate'))).toEqual([
      'true',
      null,
      'true',
    ]);
  });

  it('does no work while hidden', () => {
    const unreadable = Object.assign(actions(3), {
      [Symbol.iterator]: () => {
        throw new Error('A hidden list read its actions.');
      },
    });
    show(session(frame({legal_actions: unreadable})), true);
    // Its box stays, empty, so that its group keeps offering its tab.
    expect(container.querySelector('.legal')?.childNodes).toHaveLength(0);
    expect(container.textContent).toBe('');
  });
});

describe('LegalActions focus after a play', () => {
  // Focuses the row of `label`, then shows `next`, as the frame after a play.
  function playFrom(label: string, next: Frame): void {
    act(() => {
      rows()
        .find(row => row.textContent.endsWith(label))
        ?.focus();
    });
    show(session(next));
  }

  function without(list: readonly Action[], id: number): Action[] {
    return list.filter(action => action.id !== id);
  }

  it('moves to the row now at the same index', () => {
    show(session(frame({legal_actions: actions(5)})));
    playFrom('action 1', frame({legal_actions: without(actions(5), 1)}));
    expect(document.activeElement?.textContent).toBe('2action 2');
  });

  it('moves to the last row when the last row goes', () => {
    show(session(frame({legal_actions: actions(3)})));
    playFrom('action 2', frame({legal_actions: actions(2)}));
    expect(document.activeElement?.textContent).toBe('1action 1');
  });

  it('moves to the row at the same index of a windowed list', () => {
    show(session(frame({legal_actions: BOARD})));
    playFrom('B a5', frame({legal_actions: without(BOARD, idOf('B a5'))}));
    expect(document.activeElement?.textContent).toBe('5B a6');
  });

  it('moves to the section when a note or text replaces the rows', () => {
    const section = (): HTMLElement => only(container, '.legal');
    show(session(frame({legal_actions: actions(3)})));
    const over = {kind: 'terminal', seat: null} as const;
    playFrom('action 1', frame({to_move: over, legal_actions: []}));
    expect(document.activeElement).toBe(section());

    show(session(frame({legal_actions: actions(3)})));
    const waiting = frame({viewer: {kind: 'seat', seat: 1}});
    playFrom('action 1', waiting);
    expect(rows().map(row => row.tagName)).toEqual(['SPAN', 'SPAN']);
    expect(document.activeElement).toBe(section());
  });

  it('stays where the user moved it', () => {
    const elsewhere = document.createElement('button');
    document.body.append(elsewhere);
    show(session(frame({legal_actions: actions(3)})));
    act(() => {
      rows()[1]?.focus();
      elsewhere.focus();
    });
    show(session(frame({legal_actions: without(actions(3), 1)})));
    expect(document.activeElement).toBe(elsewhere);
    elsewhere.remove();
  });
});

describe('LegalActions under useSession', () => {
  it('plays through the session', async () => {
    const server = new FakeServer();
    const client = new SessionClient({fetch: server.fetch});
    function Panel(): JSX.Element | null {
      const live = useSession(client, {kind: 'oracle', seat: null});
      return live.frame === null ? (
        <button
          type="button"
          onClick={() => {
            live.commands.create({
              game: 'count',
              parameters: {},
              seats: [],
              chance_policy: 'auto',
              seed: null,
            });
          }}
        >
          Start
        </button>
      ) : (
        <LegalActions session={live} />
      );
    }
    const settle = (): Promise<void> =>
      act(() => new Promise<void>(resolve => setTimeout(resolve, 0)));
    act(() => {
      root.render(<Panel />);
    });
    act(() => {
      only(container, 'button').click();
    });
    await settle();
    act(() => {
      rows()[1]?.click();
    });
    await settle();
    expect(server.history('session-1')).toEqual([1]);
  });
});
