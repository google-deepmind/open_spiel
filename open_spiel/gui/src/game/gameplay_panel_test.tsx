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

import type {CatalogueEntry} from '../contract/contract.ts';
import {TransportError} from '../session/errors.ts';
import type {Session} from '../session/use_session.ts';
import {entries, frame, session} from '../testing/frames.ts';
import {type Rendered, button, only, render} from '../testing/render.tsx';
import {PanelGroup} from '../dock/panel_group.tsx';
import {ACTIONS_GROUP} from './actions_panel.tsx';
import {type Game, GameContext} from './game_context.ts';
import {GameplayPanel} from './gameplay_panel.tsx';

const ENTRY: CatalogueEntry = {
  name: 'game',
  long_name: 'Game',
  dynamics: 'sequential',
  chance_mode: 'explicit_stochastic',
  information: 'imperfect_information',
  min_seats: 2,
  max_seats: 2,
  num_players: 2,
  parameters: [],
  loadable: true,
  unavailable_reason: null,
  controllers: ['human'],
  renderer: null,
};

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  window.localStorage.clear();
});

function game(shown: Session): Game {
  return {
    entry: ENTRY,
    session: shown,
    viewer: shown.frame?.viewer ?? {kind: 'oracle', seat: null},
  };
}

function showGameplay(value: Game): HTMLElement {
  rendered?.unmount();
  rendered = render(
    <GameContext value={value}>
      <GameplayPanel />
    </GameContext>,
  );
  return rendered.container;
}

describe('GameplayPanel', () => {
  it('is a status line over the board, with no toolbar, list or stepper', () => {
    const container = showGameplay(
      game(session(frame({history: entries(['a', 'b'])}))),
    );
    expect(only(container, '.gameplay__status').textContent).toBe(
      'Turn 2 · Player 0',
    );
    expect(container.querySelector('select')).toBeNull();
    expect(container.querySelector('[aria-label="Replay"]')).toBeNull();
    expect(container.querySelector('.legal__row')).toBeNull();
    expect(container.querySelector('input')).toBeNull();
    // The board's text has its own copy button, and that is the one button.
    expect(
      Array.from(container.querySelectorAll('button'), b => b.className),
    ).toEqual(['button button--icon copy-button']);
  });

  it('shows the session error over the board, until dismissed', () => {
    const failing: Session = {
      ...session(frame()),
      error: new TransportError('Seat 1 is to move.', 409),
    };
    const container = showGameplay(game(failing));
    const notices = only(container, '.gameplay__notices');
    expect(only(notices, '[role="alert"]').textContent).toContain(
      'Seat 1 is to move.',
    );
    act(() => {
      button('×', notices).click();
    });
    expect(container.querySelector('[role="alert"]')).toBeNull();
  });

  it('offers no hand-over when another person is to move in a game', () => {
    const shown = frame({
      viewer: {kind: 'seat', seat: 0},
      omniscient: false,
      to_move: {kind: 'seat', seat: 1},
      legal_actions: null,
      state_text: null,
      observation_text: 'Card: K',
    });
    const container = showGameplay(
      game(
        session({
          ...shown,
          session: {...shown.session, hides_information: true},
        }),
      ),
    );
    expect(container.textContent).not.toContain('Hand over');
    // The observation's copy button is the one button.
    expect(
      Array.from(container.querySelectorAll('button'), b =>
        b.getAttribute('aria-label'),
      ),
    ).toEqual(['Copy to clipboard']);
  });
});

describe('ACTIONS_GROUP', () => {
  function showActions(value: Game): HTMLElement {
    rendered?.unmount();
    rendered = render(
      <GameContext value={value}>
        <PanelGroup group={ACTIONS_GROUP} />
      </GameContext>,
    );
    return rendered.container;
  }

  it('steps through the history from its footer, shown with History only', () => {
    const shown = session(frame({history: entries(['a', 'b', 'c']), ply: 3}));
    const container = showActions(game(shown));
    const tabs = Array.from(
      container.querySelectorAll('[role="tab"]'),
      tab => tab.textContent,
    );
    expect(tabs).toEqual(['History', 'Legal (2)', 'Text']);
    const footer = only(container, '.panel-group__footer');
    act(() => {
      only(footer, 'button[aria-label="Previous move"]').click();
    });
    expect(shown.commands.goTo).toHaveBeenCalledExactlyOnceWith(2);
    expect(only(footer, 'label').textContent).toBe('Skip chance');
    act(() => {
      button('Text', container).click();
    });
    expect(container.querySelector('.panel-group__footer')).toBeNull();
  });
});
