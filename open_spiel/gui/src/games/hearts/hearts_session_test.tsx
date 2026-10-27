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

import {act, type ReactNode} from 'react';
import {afterEach, describe, expect, it} from 'vitest';

import type {Frame, Presentation} from '../../contract/contract.ts';
import {ANALYSIS_VIEW_SEAT_KEY} from '../../game/bottom_seat.ts';
import {RENDERERS} from '../../renderers/registry.ts';
import {frame} from '../../testing/frames.ts';
import {enter, labelled} from '../../testing/input.ts';
import {type Rendered, only, render} from '../../testing/render.tsx';
import {SetViewSetting} from '../../testing/view_settings.tsx';
import {Perspective} from '../../view/perspective.tsx';
import {
  heartsReplay,
  heartsSession,
  heartsViewer,
  isHumanController,
} from './hearts_session.ts';
import {HeartsRenderer} from './hearts_renderer.tsx';
import {heartsUIState} from './hooks/use_hearts_state.ts';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
});

const HAND = ['2c', '5c', 'Td', 'Qs', 'Ah'];
const BACKS = [null, null, null, null, null];

// Seat 0's view while passing left, as the hearts adapter writes it.
function passView(): Record<string, unknown> {
  return {
    struct: {
      hands: {N: HAND, E: BACKS, S: BACKS, W: BACKS},
      currentTrick: [],
      completedTricks: 0,
      lastTrick: null,
      points: {N: 0, E: 0, S: 0, W: 0},
      passDirection: 'left',
      phase: 'passing',
      currentPlayer: 0,
    },
  };
}

function shown(view: Record<string, unknown>): Presentation {
  return {
    renderer: 'hearts',
    view,
    actions: [
      {id: 0, hint: {cardCode: '2c'}},
      {id: 34, hint: {cardCode: 'Td'}},
    ],
  };
}

function seatFrame(presentation: Presentation): Frame {
  const seats = [0, 1, 2, 3].map(seat => ({
    seat,
    controller: seat === 0 ? 'human' : 'random',
    config: {},
    autoplay: seat !== 0,
    failure: null,
    status: 'ready' as const,
    message: null,
  }));
  return frame({
    viewer: {kind: 'seat', seat: 0},
    omniscient: false,
    seats,
    legal_actions: [
      {id: 0, label: '2C'},
      {id: 34, label: 'TD'},
    ],
    presentation,
  });
}

describe('heartsSession', () => {
  it('builds the session shape the board reads', () => {
    const p = shown(passView());
    const session = heartsSession(seatFrame(p), p);
    expect(session.mode).toBe('game');
    expect(session.state.currentPlayer).toBe(0);
    expect(session.state.legalActions).toEqual([
      {action: 0, label: '2C', cardCode: '2c'},
      {action: 34, label: 'TD', cardCode: 'Td'},
    ]);
    expect(session.players.map(player => player.label)).toEqual([
      'player 0',
      'player 1',
      'player 2',
      'player 3',
    ]);
    expect(session.players[1]?.autoPlay).toBe(true);
  });

  it('is analysis for the oracle and names the viewer for a seat', () => {
    const p = shown(passView());
    const oracle = frame({presentation: p});
    expect(heartsSession(oracle, p).mode).toBe('analysis');
    expect(heartsViewer(seatFrame(p))).toEqual({
      seat: 0,
      viewSeat: 0,
      isParticipant: true,
      isActiveSeat: true,
    });
    expect(heartsReplay(oracle).isActive).toBe(false);
  });

  it('refuses a view that is not an object', () => {
    const p: Presentation = {renderer: 'hearts', view: [], actions: null};
    expect(() => heartsSession(seatFrame(p), p)).toThrow('not an object');
  });

  it('takes only human seats for people at the table', () => {
    expect(isHumanController('human', 1)).toBe(true);
    expect(isHumanController('random', 0)).toBe(false);
  });
});

describe('heartsUIState', () => {
  it('puts the viewer at the bottom with its hand sorted and the rest backs', () => {
    const p = shown(passView());
    const state = heartsUIState(heartsSession(seatFrame(p), p), 0);
    if (state === null) throw new Error('No state.');
    expect(state.phase).toBe('pass');
    expect(state.passDirection).toBe('left');
    const bottom = state.players.find(player => player.position === 'bottom');
    expect(bottom?.hand).toEqual(['2c', '5c', 'Td', 'Qs', 'Ah']);
    expect(bottom?.name).toBe('player 0');
    const left = state.players.find(player => player.position === 'left');
    expect(left?.hidden).toBe(true);
    expect([...state.playableCards].sort()).toEqual(['2c', 'Td']);
  });
});

describe('HeartsRenderer', () => {
  it('is registered under the name the adapter gives', () => {
    expect(RENDERERS.get('hearts')).toBe(HeartsRenderer);
    expect(RENDERERS.options('hearts')).toMatchObject({
      keepsState: true,
      drawsStatus: true,
    });
  });

  it('asks for three cards to pass, with the pass button waiting', () => {
    const p = shown(passView());
    rendered = render(
      <HeartsRenderer
        frame={seatFrame(p)}
        presentation={p}
        play={() => undefined}
      />,
    );
    const {container} = rendered;
    expect(container.textContent).toContain('Select three cards to pass left');
    const pass = container.querySelector('.hearts-pass-button');
    if (!(pass instanceof HTMLButtonElement)) throw new Error('No Pass.');
    expect(pass.textContent).toBe('Pass Left');
    expect(pass.disabled).toBe(true);
  });
});

// Every hand, as the oracle is sent them, with North (seat 0) to lead.
const HANDS = {
  N: ['2c', '5c', 'Td', 'Qs', 'Ah'],
  E: ['3c', '6d', 'Jd', 'Ks', '2h'],
  S: ['4c', '7d', 'Qd', 'As', '3h'],
  W: ['8c', '8d', 'Kd', '9s', '4h'],
};

function analysisFrame(): Frame {
  const p = shown({
    struct: {
      hands: HANDS,
      currentTrick: [],
      completedTricks: 0,
      lastTrick: null,
      points: {N: 0, E: 0, S: 0, W: 0},
      passDirection: 'none',
      phase: 'playing',
      currentPlayer: 0,
    },
  });
  return {
    ...seatFrame(p),
    viewer: {kind: 'oracle', seat: null},
    omniscient: true,
  };
}

/** The hearts board in analysis, beside View › Perspective. */
function analysisBoard(seat: number | null, showAll: boolean): ReactNode {
  const shownFrame = analysisFrame();
  const {presentation} = shownFrame;
  if (presentation === null) throw new Error('No presentation.');
  return (
    <>
      <SetViewSetting name={ANALYSIS_VIEW_SEAT_KEY} value={seat} />
      <SetViewSetting name="hearts.showAllCards" value={showAll} />
      <Perspective viewer={{kind: 'oracle', seat: null}} numSeats={4} />
      <HeartsRenderer
        frame={shownFrame}
        presentation={presentation}
        play={() => undefined}
      />
    </>
  );
}

/** The cards drawn face up in hand slot `slot`, by card code; a back's id
 *  names its slot ("west-0"). */
function faceUp(root: HTMLElement, slot: string): string[] {
  return [...root.querySelectorAll(`[data-slot="${slot}"][data-card-id]`)]
    .map(card => card.getAttribute('data-card-id') ?? '')
    .filter(id => /^[2-9TJQKA][cdhs]$/.test(id));
}

describe('HeartsRenderer in analysis (View › Perspective)', () => {
  it('sits at the seat chosen, the other hands face down', () => {
    rendered = render(analysisBoard(null, false));
    const {container} = rendered;
    expect(faceUp(container, 'south').sort()).toEqual([...HANDS.N].sort());
    enter(labelled(container, 'Perspective'), '2');
    expect(faceUp(container, 'south').sort()).toEqual([...HANDS.S].sort());
    for (const slot of ['west', 'north', 'east']) {
      expect(faceUp(container, slot)).toEqual([]);
    }
  });

  it('offers to take the seat to move when its hand is hidden', () => {
    rendered = render(analysisBoard(2, false));
    const {container} = rendered;
    const take = only(container, '[data-testid="hearts-turn-to-mover"]');
    expect(take.textContent).toBe('Take their seat');
    act(() => {
      take.click();
    });
    expect(
      (labelled(container, 'Perspective') as HTMLSelectElement).value,
    ).toBe('0');
    expect(faceUp(container, 'south').sort()).toEqual([...HANDS.N].sort());
  });

  it('turns every hand face up with Show all cards, at any seat', () => {
    rendered = render(analysisBoard(1, true));
    const {container} = rendered;
    expect(faceUp(container, 'south').sort()).toEqual([...HANDS.E].sort());
    expect(faceUp(container, 'west').sort()).toEqual([...HANDS.S].sort());
    expect(faceUp(container, 'north').sort()).toEqual([...HANDS.W].sort());
    expect(faceUp(container, 'east').sort()).toEqual([...HANDS.N].sort());
    expect(
      container.querySelector('[data-testid="hearts-turn-to-mover"]'),
    ).toBeNull();
  });
});
