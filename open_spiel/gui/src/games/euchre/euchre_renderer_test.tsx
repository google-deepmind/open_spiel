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
// The euchre board as drawn (cards named like the card library, the lone hand, the trick badges, the
// hand over, the up-card at Go Alone), and the kernel renderer: a frame's
// view drawn as the board, the viewer's presses played, no hidden card drawn
// face up, and analysis drawn from the seat View › Perspective chooses.

import {act, type ReactNode} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {Presentation} from '../../contract/contract.ts';
import {ANALYSIS_VIEW_SEAT_KEY} from '../../game/bottom_seat.ts';
import {RENDERERS} from '../../renderers/registry.ts';
import {frame} from '../../testing/frames.ts';
import {enter, labelled} from '../../testing/input.ts';
import {type Rendered, only, render} from '../../testing/render.tsx';
import {SetViewSetting} from '../../testing/view_settings.tsx';
import {Perspective} from '../../view/perspective.tsx';
import {BiddingOverlay} from './components/bidding_overlay.tsx';
import {Card} from './components/card.tsx';
import {euchreCardLabel} from './components/card_a11y.ts';
import {EuchreRenderer} from './euchre_renderer.tsx';
import {euchreOutcome} from './hand_result.ts';
import {TableLayout, type TableLayoutProps} from './sections/table_layout.tsx';
import type {CardCode, Player} from './types.ts';

let rendered: Rendered | null = null;

afterEach(() => {
  rendered?.unmount();
  rendered = null;
  vi.restoreAllMocks();
});

function show(tree: ReactNode): HTMLElement {
  rendered = render(tree);
  return rendered.container;
}

function named(root: HTMLElement, role: string, name: string): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(`[role="${role}"]`)].filter(
    element => element.getAttribute('aria-label') === name,
  );
}

describe('hand cards are named like the card library', () => {
  it('names cards with cardLabel', () => {
    expect(euchreCardLabel('Ah')).toBe('Ace of Hearts');
    expect(euchreCardLabel('Td')).toBe('Ten of Diamonds');
    expect(euchreCardLabel('9c')).toBe('Nine of Clubs');
    expect(() => euchreCardLabel('Xz' as CardCode)).toThrow();
  });

  it('a playable card is a named button that answers Enter', () => {
    const onClick = vi.fn();
    const root = show(<Card card="Js" faceUp playable onClick={onClick} />);
    const [button] = named(root, 'button', 'Jack of Spades');
    if (button === undefined) throw new Error('No card button.');
    act(() => {
      button.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
      );
      button.click();
    });
    expect(onClick).toHaveBeenCalledTimes(2);
  });

  it('a face-up card that cannot be played is a named image; a face-down one is decorative', () => {
    const root = show(
      <>
        <Card card="Qh" faceUp />
        <Card card="Kd" faceUp={false} />
      </>,
    );
    expect(named(root, 'img', 'Queen of Hearts')).toHaveLength(1);
    expect(
      root.querySelectorAll('.euchre-card')[1]?.getAttribute('aria-hidden'),
    ).toBe('true');
    expect(root.textContent).not.toMatch(/King/);
  });
});

function player(
  name: string,
  position: Player['position'],
  hand: CardCode[],
  hidden = false,
): Player {
  return {name, position, hand, hidden};
}

function table(over: Partial<TableLayoutProps>): TableLayoutProps {
  return {
    players: [
      player('North', 'bottom', ['9s', 'Ah']),
      player('East', 'left', ['9c', '9c'], true),
      player('South', 'top', ['9c', '9c'], true),
      player('West', 'right', ['9c', '9c'], true),
    ],
    trickPile: [],
    phase: 'bidding2',
    upcard: null,
    trumpSuit: 'h',
    dealer: 'top',
    declarer: 'left',
    goingAlone: true,
    aloneSeat: 'East',
    sittingOut: 'right',
    currentPlayer: 'bottom',
    tricksWon: {NS: 2, EW: 1},
    teamOf: {bottom: 'NS', top: 'NS', left: 'EW', right: 'EW'},
    ...over,
  };
}

describe('the lone player and the partner who sits out', () => {
  it('names the lone seat in text and marks its partner', () => {
    const root = show(<TableLayout {...table({})} />);
    expect(only(root, '[data-testid="euchre-alone"]').textContent).toBe(
      'East alone',
    );
    expect(
      only(root, '[data-testid="euchre-sits-out-right"]').textContent,
    ).toBe('Sits out');
    expect(
      root.querySelector('[data-testid="euchre-sits-out-left"]'),
    ).toBeNull();
    expect(named(root, 'img', 'Declarer, playing alone')).toHaveLength(1);
    expect(
      named(
        root,
        'group',
        'Hand of West, 2 cards, face down, sits out this hand',
      ),
    ).toHaveLength(1);
  });

  it('no alone text and no sit-out mark when the declarer plays with a partner', () => {
    const root = show(
      <TableLayout
        {...table({goingAlone: false, aloneSeat: null, sittingOut: null})}
      />,
    );
    expect(root.querySelector('[data-testid="euchre-alone"]')).toBeNull();
    expect(root.textContent).not.toMatch(/Sits out/);
    expect(named(root, 'img', 'Declarer')).toHaveLength(1);
  });
});

describe('trick badges are labelled; game over shows points', () => {
  it('labels the badges and each seat by its side', () => {
    const root = show(<TableLayout {...table({})} />);
    expect(only(root, '[data-testid="euchre-tricks-NS"]').textContent).toBe(
      'N-S tricks2',
    );
    expect(named(root, 'img', 'North-South: 2 tricks')).toHaveLength(3);
    expect(named(root, 'img', 'East-West: 1 trick')).toHaveLength(3);
  });

  it("game over shows the scoring case and each side's points", () => {
    const outcome = euchreOutcome({
      declarer: 'E',
      goingAlone: true,
      tricksWon: {NS: 3, EW: 2},
      returns: [2, -2, 2, -2],
    });
    const root = show(
      <TableLayout
        {...table({
          isTerminal: true,
          phase: 'terminal',
          tricksWon: {NS: 3, EW: 2},
          outcome,
        })}
      />,
    );
    expect(only(root, '[data-testid="euchre-outcome-cause"]').textContent).toBe(
      'East went alone and was euchred with 2 tricks.',
    );
    expect(only(root, '[data-testid="euchre-outcome-NS"]').textContent).toBe(
      'North-South2 points3 tricksScores',
    );
    expect(only(root, '[data-testid="euchre-outcome-EW"]').textContent).toBe(
      'East-West0 points2 tricks',
    );
  });
});

describe('the up-card at the Go Alone step', () => {
  function overlay(upcard: CardCode, trumpSuit: 'h' | 'd'): HTMLElement {
    return show(
      <BiddingOverlay
        phase="alone"
        upcard={upcard}
        actions={[
          {action: 29, label: 'Alone', actionType: 'alone'},
          {action: 30, label: 'Partner', actionType: 'partner'},
        ]}
        isViewerTurn
        trumpSuit={trumpSuit}
        onAction={() => undefined}
      />,
    );
  }

  it('shows the card the dealer picked up when it was ordered up', () => {
    const root = overlay('Jh', 'h');
    expect(root.textContent).toMatch(/Dealer picked up:/);
    expect(named(root, 'img', 'Jack of Hearts')).toHaveLength(1);
  });

  it('does not show an up-card that was turned down', () => {
    const root = overlay('Jh', 'd');
    expect(root.textContent).not.toMatch(/Dealer picked up:/);
    expect(named(root, 'img', 'Jack of Hearts')).toHaveLength(0);
  });
});

function nulls(n: number): null[] {
  return Array.from({length: n}, () => null);
}

// North (seat 0) to play the first card of a trick, hearts trump.
function presentation(
  fields: Record<string, unknown> = {},
  hints: ReadonlyArray<[number, string]> = [],
): Presentation {
  return {
    renderer: 'euchre',
    view: {
      struct: {
        hands: {N: ['9s', 'Ah', 'Kh'], E: nulls(3), S: nulls(3), W: nulls(3)},
        upcard: 'Jd',
        trumpSuit: 'h',
        dealer: 'W',
        phase: 'playing',
        currentTrick: [],
        lastTrick: null,
        tricksWon: {NS: 1, EW: 1},
        declarer: 'N',
        goingAlone: false,
        currentPlayer: 0,
        discarding: false,
        ...fields,
      },
      currentPosition: 'N',
    },
    actions: hints.map(([id, code]) => ({
      id,
      hint: {
        cardCode: code,
        actionType: 'card',
        calledSuit: null,
        displayLabel: code,
      },
    })),
  };
}

function euchreFrame(shown: Presentation) {
  return frame({
    session: {
      ...frame().session,
      game: 'euchre',
      num_seats: 4,
      hides_information: true,
    },
    viewer: {kind: 'seat', seat: 0},
    omniscient: false,
    seats: [0, 1, 2, 3].map(seat => ({
      seat,
      controller: seat === 0 ? 'human' : 'random',
      config: {},
      autoplay: seat !== 0,
      failure: null,
      status: 'ready',
      message: null,
    })),
    legal_actions: (shown.actions ?? []).map(({id}) => ({
      id,
      label: String(id),
    })),
    state_text: null,
    presentation: shown,
  });
}

describe('the euchre renderer', () => {
  it('is registered under its name and keeps its state across plies', () => {
    expect(RENDERERS.get('euchre')).toBe(EuchreRenderer);
    expect(RENDERERS.options('euchre').keepsState).toBe(true);
    expect(RENDERERS.options('euchre').drawsStatus).toBe(true);
  });

  it("draws the viewer's hand face up and every other hand as backs", () => {
    const shown = presentation({}, [
      [20, 'Ah'],
      [16, 'Kh'],
      [3, '9s'],
    ]);
    const root = show(
      <EuchreRenderer
        frame={euchreFrame(shown)}
        presentation={shown}
        play={vi.fn()}
      />,
    );
    expect(only(root, '.euchre-player-name--bottom').textContent).toBe('North');
    const bottom = only(root, '.euchre-player-section--bottom');
    expect(bottom.querySelectorAll('[data-card-id]')).toHaveLength(3);
    for (const position of ['left', 'top', 'right']) {
      const hand = only(root, `.euchre-player-section--${position}`);
      expect(hand.querySelectorAll('[data-card-id]')).toHaveLength(0);
      expect(hand.querySelectorAll('.euchre-card--back')).toHaveLength(3);
    }
    expect(root.textContent).toMatch(/Phase: Playing/);
  });
});

describe("the euchre renderer's presses", () => {
  it('plays the card the viewer presses', () => {
    const play = vi.fn();
    const shown = presentation({}, [
      [20, 'Ah'],
      [16, 'Kh'],
      [3, '9s'],
    ]);
    const root = show(
      <EuchreRenderer
        frame={euchreFrame(shown)}
        presentation={shown}
        play={play}
      />,
    );
    const [card] = named(root, 'button', 'King of Hearts');
    if (card === undefined) throw new Error('No playable King of Hearts.');
    act(() => {
      card.click();
    });
    expect(play).toHaveBeenCalledWith(16);
  });

  it('plays a call from the bidding panel', () => {
    const play = vi.fn();
    const shown: Presentation = {
      ...presentation({phase: 'bidding1', trumpSuit: null, declarer: null}),
      actions: [
        {
          id: 24,
          hint: {
            cardCode: null,
            actionType: 'pass',
            calledSuit: null,
            displayLabel: 'Pass',
          },
        },
        {
          id: 26,
          hint: {
            cardCode: null,
            actionType: 'call_suit',
            calledSuit: 'd',
            displayLabel: 'Diamonds',
          },
        },
      ],
    };
    const root = show(
      <EuchreRenderer
        frame={euchreFrame(shown)}
        presentation={shown}
        play={play}
      />,
    );
    expect(root.textContent).toMatch(/First Round/);
    const orderUp = [...root.querySelectorAll('button')].find(
      b => b.textContent === 'Order Up',
    );
    if (orderUp === undefined) throw new Error('No Order Up button.');
    act(() => {
      orderUp.click();
    });
    expect(play).toHaveBeenCalledWith(26);
  });
});

// Every hand, as the oracle is sent them.
const ALL_HANDS = {
  N: ['9s', 'Ah', 'Kh'],
  E: ['9c', 'Tc', 'Jc'],
  S: ['9d', 'Td', 'Qd'],
  W: ['Qs', 'Ks', 'As'],
};

/** The oracle's frame of `shown`: analysis. */
function analysisFrame(shown: Presentation) {
  return {
    ...euchreFrame(shown),
    viewer: {kind: 'oracle', seat: null} as const,
    omniscient: true,
  };
}

/** The euchre board in analysis at `seat`, every other hand shown or not. */
function analysisTable(seat: number | null, showAll: boolean): ReactNode {
  const shown = presentation({hands: ALL_HANDS});
  return (
    <>
      <SetViewSetting name={ANALYSIS_VIEW_SEAT_KEY} value={seat} />
      <SetViewSetting name="euchre.showAllCards" value={showAll} />
      <EuchreRenderer
        frame={analysisFrame(shown)}
        presentation={shown}
        play={vi.fn()}
      />
    </>
  );
}

function faceUpCards(root: HTMLElement, position: string): number {
  return only(root, `.euchre-player-section--${position}`).querySelectorAll(
    '[data-card-id]',
  ).length;
}

describe('the euchre renderer in analysis (View › Perspective)', () => {
  it('sits at the seat chosen, the other hands face down', () => {
    const root = show(analysisTable(2, false));
    expect(only(root, '.euchre-player-name--bottom').textContent).toBe('South');
    expect(faceUpCards(root, 'bottom')).toBe(3);
    for (const position of ['left', 'top', 'right']) {
      expect(faceUpCards(root, position)).toBe(0);
    }
  });

  it('turns every hand face up with Show all cards, at any seat', () => {
    const root = show(analysisTable(1, true));
    expect(only(root, '.euchre-player-name--bottom').textContent).toBe('East');
    for (const position of ['bottom', 'left', 'top', 'right']) {
      expect(faceUpCards(root, position)).toBe(3);
    }
  });

  it('switches seat as View › Perspective changes, and plays nothing', () => {
    const shown = presentation({hands: ALL_HANDS});
    const play = vi.fn();
    const root = show(
      <>
        <SetViewSetting name={ANALYSIS_VIEW_SEAT_KEY} value={null} />
        <SetViewSetting name="euchre.showAllCards" value={false} />
        <Perspective viewer={{kind: 'oracle', seat: null}} numSeats={4} />
        <EuchreRenderer
          frame={analysisFrame(shown)}
          presentation={shown}
          play={play}
        />
      </>,
    );
    expect(only(root, '.euchre-player-name--bottom').textContent).toBe('North');
    enter(labelled(root, 'Perspective'), '3');
    expect(only(root, '.euchre-player-name--bottom').textContent).toBe('West');
    expect(faceUpCards(root, 'bottom')).toBe(3);
    expect(faceUpCards(root, 'right')).toBe(0);
    expect(play).not.toHaveBeenCalled();
  });

  it("keeps a game's seat at the bottom and its hidden hands hidden", () => {
    const shown = presentation({}, []);
    const root = show(
      <>
        <SetViewSetting name={ANALYSIS_VIEW_SEAT_KEY} value={2} />
        <SetViewSetting name="euchre.showAllCards" value={true} />
        <EuchreRenderer
          frame={euchreFrame(shown)}
          presentation={shown}
          play={vi.fn()}
        />
      </>,
    );
    expect(only(root, '.euchre-player-name--bottom').textContent).toBe('North');
    for (const position of ['left', 'top', 'right']) {
      expect(faceUpCards(root, position)).toBe(0);
    }
  });
});
