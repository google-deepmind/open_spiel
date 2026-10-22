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
import {type Root, createRoot} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {Frame, SeatConfig} from '../contract/contract.ts';
import {stubResizeObserver} from '../testing/dom.ts';
import {enter, labelled} from '../testing/input.ts';
import {only} from '../testing/render.tsx';
import {PlayersPanel} from './players_panel.tsx';
import {MCTS, RANDOM, SCRIPTED, frameWith, seatView} from './testing/seats.ts';

let container: HTMLElement;
let root: Root;
let setSeat: ReturnType<
  typeof vi.fn<(seat: number, config: SeatConfig) => void>
>;
let advance: ReturnType<typeof vi.fn<(maxPlies?: number | null) => void>>;

beforeEach(() => {
  stubResizeObserver();
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  setSeat = vi.fn<(seat: number, config: SeatConfig) => void>();
  advance = vi.fn<(maxPlies?: number | null) => void>();
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  vi.unstubAllGlobals();
});

const TWO_HUMANS = frameWith([seatView(0), seatView(1)]);

function show(frame: Frame, busy = false): void {
  act(() => {
    root.render(
      <PlayersPanel frame={frame} busy={busy} commands={{setSeat, advance}} />,
    );
  });
}

function seat(index: number): HTMLElement {
  const row = container.querySelectorAll<HTMLElement>('li.seat')[index];
  if (row === undefined) throw new Error(`No seat ${String(index)}.`);
  return row;
}

// The one button under `parent` named `name`, by its label or its text.
function button(parent: ParentNode, name: string): HTMLButtonElement {
  const found = Array.from(parent.querySelectorAll('button')).filter(
    candidate =>
      (candidate.getAttribute('aria-label') ?? candidate.textContent) === name,
  );
  const [element] = found;
  if (found.length !== 1 || element === undefined) {
    throw new Error(
      `Expected one ${name} button, found ${String(found.length)}.`,
    );
  }
  return element;
}

function click(element: HTMLElement): void {
  act(() => {
    element.click();
  });
}

function open(index: number): HTMLElement {
  click(button(seat(index), 'Settings'));
  return seat(index);
}

function saveState(row: HTMLElement): string {
  return only(row, '[role="status"]').textContent;
}

describe('PlayersPanel seats', () => {
  it('heads the panel and lists every seat collapsed', () => {
    show(TWO_HUMANS);
    expect(only(container, 'h2').textContent).toBe('Players');
    expect(
      Array.from(
        container.querySelectorAll('.seat__name'),
        name => name.textContent,
      ),
    ).toEqual(['Player 0', 'Player 1']);
    const toggle = button(seat(1), 'Settings');
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    const settings = document.getElementById(
      toggle.getAttribute('aria-controls') ?? '',
    );
    expect(settings?.hidden).toBe(true);
    click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(toggle.textContent).toBe('Hide settings');
    expect(settings?.hidden).toBe(false);
  });

  it('shows the controller on its own line, autoplay or not', () => {
    show(frameWith([seatView(0), seatView(1, RANDOM, {autoplay: true})]));
    expect(only(seat(1), '.seat__controller').textContent).toBe('Random');
    expect(only(seat(0), '.seat__controller').textContent).toBe('Human');
  });

  it('offers the session’s controllers, saying which cannot play here', () => {
    const unavailable = {...MCTS, unavailable_reason: 'MCTS needs X.'};
    const frame = frameWith([seatView(0), seatView(1)]);
    show({
      ...frame,
      session: {
        ...frame.session,
        controllers: frame.session.controllers.map(controller =>
          controller.name === 'mcts' ? unavailable : controller,
        ),
      },
    });
    const row = open(0);
    const select = labelled(row, 'Controller') as HTMLSelectElement;
    expect(Array.from(select.options, option => option.text)).toEqual([
      'Human',
      'Random',
      'MCTS — not available here',
      'Scripted',
    ]);
    expect(select.value).toBe('human');
    expect(row.querySelector('[role="note"]')).toBeNull();
    enter(select, 'mcts');
    expect(only(row, '[role="note"]').textContent).toBe('MCTS needs X.');
  });

  it('loads a chosen controller’s defaults and saves them, autoplaying', () => {
    show(TWO_HUMANS);
    const row = open(1);
    enter(labelled(row, 'Controller'), 'mcts');
    expect(labelled(row, 'Max Simulations').value).toBe('1000');
    expect(labelled(row, 'Uct C').value).toBe('2');
    expect((labelled(row, 'Solve') as HTMLInputElement).checked).toBe(true);
    expect((labelled(row, 'Autoplay') as HTMLInputElement).checked).toBe(true);
    enter(labelled(row, 'Max Simulations'), '200');
    click(button(row, 'Save settings for Player 1'));
    expect(setSeat.mock.calls).toEqual([
      [
        1,
        {
          controller: 'mcts',
          config: {max_simulations: 200, uct_c: 2, solve: true},
          autoplay: true,
        },
      ],
    ]);
  });
});

describe('PlayersPanel settings form', () => {
  it('puts each setting’s description behind its info toggle', () => {
    show(frameWith([seatView(0), seatView(1, MCTS)]));
    const row = open(1);
    expect(row.textContent).not.toContain('Simulations per move');
    const infos = row.querySelectorAll('button[aria-label="More info"]');
    expect(infos).toHaveLength(3);
    click(infos[0] as HTMLButtonElement);
    expect(only(document.body, '[role="tooltip"]').textContent).toBe(
      'Simulations per move, each ending in a random rollout.',
    );
  });

  it('shows nothing for a controller without settings', () => {
    show(TWO_HUMANS);
    const row = open(0);
    enter(labelled(row, 'Controller'), 'random');
    expect(row.querySelector('.field-form, textarea')).toBeNull();
    expect(row.textContent).not.toContain('takes no settings');
  });

  it('offers no autoplay to a human', () => {
    show(TWO_HUMANS);
    expect(() => labelled(open(0), 'Autoplay')).toThrow();
  });
});

describe('PlayersPanel save bars', () => {
  it('puts Save and Reset above the controller and after the settings', () => {
    show(frameWith([seatView(0), seatView(1, MCTS, {autoplay: true})]));
    const row = open(1);
    const bars = row.querySelectorAll('.seat-editor__save-bar');
    expect(bars).toHaveLength(2);
    const [top, bottom] = Array.from(bars);
    expect(top?.classList).toContain('seat-editor__save-bar--top');
    expect(
      Array.from(top?.querySelectorAll('button') ?? [], b => b.textContent),
    ).toEqual(['Save', 'Reset']);
    expect(
      Array.from(bottom?.querySelectorAll('button') ?? [], b => b.textContent),
    ).toEqual(['Save settings', 'Reset changes']);
    expect(bottom?.querySelector('[role="status"]')).toBeNull();
    const controller = labelled(row, 'Controller');
    expect(
      top?.compareDocumentPosition(controller) ===
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBe(true);
  });

  it('tracks unsaved changes by value, with a dot on Save', () => {
    show(frameWith([seatView(0), seatView(1, MCTS, {autoplay: true})]));
    const row = open(1);
    const save = button(row, 'Save settings for Player 1');
    const reset = button(row, 'Reset changes for Player 1');
    expect([saveState(row), save.disabled, reset.disabled]).toEqual([
      'No changes',
      false,
      false,
    ]);
    expect(row.querySelector('.seat-editor__unsaved-dot')).toBeNull();
    enter(labelled(row, 'Uct C'), '3');
    expect(saveState(row)).toBe('Unsaved changes');
    expect(row.querySelectorAll('.seat-editor__unsaved-dot')).toHaveLength(2);
    enter(labelled(row, 'Uct C'), '2.0');
    expect(saveState(row)).toBe('No changes');
    enter(labelled(row, 'Controller'), 'random');
    enter(labelled(row, 'Controller'), 'mcts');
    expect(saveState(row)).toBe('No changes');
    click(labelled(row, 'Autoplay'));
    expect(saveState(row)).toBe('Unsaved changes');
  });

  it('holds no changes for a seat saved without its settings', () => {
    show(frameWith([seatView(0), seatView(1, MCTS, {config: {}})]));
    const row = open(1);
    expect(labelled(row, 'Max Simulations').value).toBe('1000');
    expect(saveState(row)).toBe('No changes');
  });

  it('starts the draft afresh each time the settings open', () => {
    show(TWO_HUMANS);
    const row = open(0);
    enter(labelled(row, 'Controller'), 'random');
    expect(saveState(row)).toBe('Unsaved changes');
    click(button(row, 'Hide settings'));
    expect(row.textContent).not.toContain('Unsaved changes');
    click(button(row, 'Settings'));
    expect(labelled(row, 'Controller').value).toBe('human');
    expect(saveState(row)).toBe('No changes');
  });

  it('resets a draft to the saved settings', () => {
    show(TWO_HUMANS);
    const row = open(0);
    enter(labelled(row, 'Controller'), 'mcts');
    click(button(row, 'Reset changes'));
    expect(labelled(row, 'Controller').value).toBe('human');
    expect(saveState(row)).toBe('No changes');
  });

  it('saves and resets nothing while a request is in flight', () => {
    show(TWO_HUMANS);
    enter(labelled(open(0), 'Controller'), 'random');
    show(TWO_HUMANS, true);
    expect(button(seat(0), 'Save settings for Player 0').disabled).toBe(true);
    expect(button(seat(0), 'Save settings').disabled).toBe(true);
    expect(button(seat(0), 'Reset changes for Player 0').disabled).toBe(true);
  });
});

describe('PlayersPanel saving', () => {
  it('refuses to save invalid settings, and says which', () => {
    show(frameWith([seatView(0), seatView(1, MCTS)]));
    const row = open(1);
    enter(labelled(row, 'Max Simulations'), '0');
    expect(row.textContent).not.toContain('A whole number from 1 to 10000.');
    click(button(row, 'Save settings'));
    expect(setSeat).not.toHaveBeenCalled();
    expect(labelled(row, 'Max Simulations').getAttribute('aria-invalid')).toBe(
      'true',
    );
    expect(row.textContent).toContain('A whole number from 1 to 10000.');
    enter(labelled(row, 'Max Simulations'), '10');
    expect(row.textContent).not.toContain('A whole number');
  });

  it('drops a draft once the server holds new settings, and keeps it otherwise', () => {
    show(TWO_HUMANS);
    const row = open(1);
    enter(labelled(row, 'Controller'), 'random');
    click(button(row, 'Save settings for Player 1'));
    show({...TWO_HUMANS});
    expect(saveState(seat(1))).toBe('Unsaved changes');
    show(frameWith([seatView(0), seatView(1, RANDOM, {autoplay: true})]));
    expect(saveState(seat(1))).toBe('No changes');
    expect(labelled(seat(1), 'Controller').value).toBe('random');
  });

  it('drops an unsaved draft after a move', () => {
    show(TWO_HUMANS);
    enter(labelled(open(1), 'Controller'), 'random');
    show({...TWO_HUMANS});
    expect(saveState(seat(1))).toBe('Unsaved changes');
    show({...TWO_HUMANS, ply: 1, num_plies: 1});
    expect(saveState(seat(1))).toBe('No changes');
    expect(labelled(seat(1), 'Controller').value).toBe('human');
  });

  it('drops every draft with a new session', () => {
    show(TWO_HUMANS);
    enter(labelled(open(0), 'Controller'), 'random');
    show(
      frameWith([seatView(0), seatView(1)], {
        session: {...TWO_HUMANS.session, id: 'session-2'},
      }),
    );
    expect(labelled(open(0), 'Controller').value).toBe('human');
  });

  it('edits settings no form can show as JSON', () => {
    show(
      frameWith([seatView(0), seatView(1, SCRIPTED, {config: {moves: '1 2'}})]),
    );
    const row = open(1);
    const text = labelled(row, 'Configuration (JSON)');
    expect(JSON.parse(text.value)).toEqual({moves: '1 2'});
    enter(text, '{"moves": ');
    click(button(row, 'Save settings for Player 1'));
    expect(setSeat).not.toHaveBeenCalled();
    expect(text.getAttribute('aria-invalid')).toBe('true');
    enter(text, '{"moves": "3"}');
    click(button(row, 'Save settings for Player 1'));
    expect(setSeat.mock.calls).toEqual([
      [1, {controller: 'scripted', config: {moves: '3'}, autoplay: false}],
    ]);
  });
});

describe('PlayersPanel stepping', () => {
  const stepped = (to: number, overrides: Partial<Frame> = {}): Frame =>
    frameWith([seatView(0), seatView(1, RANDOM)], {
      to_move: {kind: 'seat', seat: to},
      ...overrides,
    });

  it('steps a controller that waits to be advanced, on its turn', () => {
    show(stepped(1));
    expect(seat(0).textContent).not.toContain('Step');
    const step = button(seat(1), 'Step');
    expect(step.disabled).toBe(false);
    click(step);
    expect(advance).toHaveBeenCalledExactlyOnceWith(1);
  });

  it('offers no Step off its turn, while busy, or in a replay', () => {
    show(stepped(0));
    expect(button(seat(1), 'Step').disabled).toBe(true);
    expect(button(seat(1), 'Step').title).toBe('Not this player’s turn');
    show(stepped(1), true);
    expect(button(seat(1), 'Step').disabled).toBe(true);
    show(stepped(1, {ply: 0, num_plies: 2}));
    expect(button(seat(1), 'Step').disabled).toBe(true);
  });

  it('steps an autoplaying seat only where autoplay left it to move', () => {
    const autoplaying = (to: number): Frame =>
      frameWith([seatView(0), seatView(1, RANDOM, {autoplay: true})], {
        to_move: {kind: 'seat', seat: to},
      });
    show(autoplaying(0));
    expect(seat(1).textContent).not.toContain('Step');
    show(autoplaying(1));
    const step = button(seat(1), 'Step');
    expect(step.disabled).toBe(false);
    click(step);
    expect(advance).toHaveBeenCalledExactlyOnceWith(1);
  });

  it('says why a seat stopped, and retries it on its turn', () => {
    const failure = {error: 'ValueError', message: 'No legal move.'};
    show(
      frameWith([seatView(0), seatView(1, RANDOM, {failure})], {
        status: 'paused',
        to_move: {kind: 'seat', seat: 1},
      }),
    );
    const alert = only(seat(1), '[role="alert"]');
    expect(alert.textContent).toContain(
      'Player 1’s controller stopped with ValueError.',
    );
    expect(alert.textContent).toContain('No legal move.');
    click(button(seat(1), 'Retry'));
    expect(setSeat.mock.calls).toEqual([
      [1, {controller: 'random', config: {}, autoplay: false}],
    ]);
    expect(advance).toHaveBeenCalledExactlyOnceWith(1);
  });
});

describe('PlayersPanel to a spectator', () => {
  const failure = {error: 'ValueError', message: null};
  const frame = frameWith(
    [seatView(0), seatView(1, MCTS, {autoplay: true, failure})],
    {
      viewer: {kind: 'spectator', seat: null},
      omniscient: false,
      status: 'paused',
    },
  );

  it('shows every seat’s settings read-only and open, behind a chevron', () => {
    show(frame);
    expect(only(container, '.players__status').textContent).toContain(
      'Spectating',
    );
    const row = seat(1);
    expect(row.querySelector('select, input, textarea')).toBeNull();
    const toggle = button(row, 'Hide settings for Player 1');
    expect(toggle.textContent).toBe('▲');
    const terms = Array.from(
      row.querySelectorAll('.seat__summary-row'),
      entry => [
        entry.querySelector('dt')?.textContent,
        entry.querySelector('dd')?.textContent,
      ],
    );
    expect(terms).toEqual([
      ['Seat', 'Bot'],
      ['Controller', 'MCTS'],
      ['Autoplay', 'On'],
      ['Max Simulations', '1000'],
      ['Uct C', '2'],
      ['Solve', 'Yes'],
    ]);
    expect(row.textContent).not.toContain('Retry');
    click(toggle);
    expect(button(row, 'Show settings for Player 1').textContent).toBe('▼');
  });

  it('says a seat stopped without a message it may not see', () => {
    show(frame);
    const alert = only(seat(1), '[role="alert"]');
    expect(alert.querySelector('pre')).toBeNull();
  });
});

describe('PlayersPanel to a seat of a game that hides information', () => {
  it('lets the seat change who plays every seat', () => {
    show(
      frameWith([seatView(0), seatView(1)], {
        viewer: {kind: 'seat', seat: 0},
        omniscient: false,
      }),
    );
    expect(container.querySelector('.players__status')).toBeNull();
    expect(labelled(open(1), 'Controller')).toBeInstanceOf(HTMLSelectElement);
  });
});

describe('PlayersPanel defects', () => {
  it('refuses a seat played by a controller the session does not list', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const stray = frameWith([
      seatView(0),
      {...seatView(1), controller: 'alpha_zero'},
    ]);
    expect(() => {
      show(stray);
    }).toThrow('The session lists no controller alpha_zero.');
    vi.restoreAllMocks();
  });
});
