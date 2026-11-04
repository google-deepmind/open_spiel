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
// The euchre board's state: the status line, the hand at game over, the scoring cases, the held trick and
// the sides an East viewer sees.

import {describe, expect, it} from 'vitest';

import type {CompletedTrick} from '../../cards/library/index.ts';
import type {SessionPayload} from './euchre_session.ts';
import {euchreOutcome} from './hand_result.ts';
import {
  type EuchreTrickCollect,
  type EuchreTurn,
  statusLine,
} from './euchre_gameplay_turn.ts';
import {euchreHeldView, type EuchreView} from './held_trick.ts';
import {type EuchreUIState, euchreUIState} from './hooks/use_euchre_state.ts';
import type {CardCode, PlayerPosition} from './types.ts';

function nulls(n: number): null[] {
  return Array.from({length: n}, () => null);
}

function session(
  struct: unknown,
  options: {
    currentPlayer?: number;
    returns?: number[] | null;
    isTerminal?: boolean;
    gameParams?: Record<string, boolean>;
    mode?: 'game' | 'analysis';
  } = {},
): SessionPayload {
  return {
    sessionId: 's',
    mode: options.mode ?? 'game',
    gameParams: options.gameParams ?? {},
    historyView: false,
    players: [],
    state: {
      struct,
      currentPlayer: options.currentPlayer ?? 0,
      isTerminal: options.isTerminal ?? false,
      legalActions: [],
      returns: options.returns ?? null,
      history: [],
      historyPlayers: [],
    },
  };
}

function struct(phase: string, currentPlayer: number, discarding: boolean) {
  return {
    hands: {
      N: ['9s', 'Jh', 'Ac', 'Kd', 'Qd', 'Td'],
      E: nulls(5),
      S: nulls(5),
      W: nulls(5),
    },
    upcard: 'Td',
    trumpSuit: 'd',
    dealer: 'N',
    phase,
    currentTrick: [],
    tricksWon: {NS: 0, EW: 0},
    declarer: 'S',
    goingAlone: false,
    currentPlayer,
    discarding,
  };
}

function state(s: SessionPayload, viewerSeat: number) {
  const result = euchreUIState(s, viewerSeat);
  if (result === null) throw new Error('No euchre state.');
  return result;
}

describe('euchre status', () => {
  it('asks the dealer to discard', () => {
    const s = state(session(struct('playing', 0, true)), 0);
    expect(s.discarding).toBe(true);
    expect(s.statusMessage).toBe('Discard a card');
  });

  it('tells another seat whom it is waiting for to discard', () => {
    expect(state(session(struct('playing', 0, true)), 1).statusMessage).toBe(
      'Waiting for N to discard',
    );
  });

  it('keeps the play status when no one is discarding', () => {
    const s = state(session(struct('playing', 0, false)), 0);
    expect(s.discarding).toBe(false);
    expect(s.statusMessage).toBe('Your turn to play');
  });

  it('names the seat choosing whether to go alone', () => {
    const s = state(session(struct('alone', 3, false), {currentPlayer: 3}), 0);
    expect(s.statusMessage).toBe('Waiting for W to choose');
  });

  it('draws the up-card the dealer took into a hidden hand as a back', () => {
    // East deals and was ordered up in diamonds: at Go Alone, North sees
    // East take the up-card, but East's hand stays all backs.
    const s = state(
      session({...struct('alone', 1, false), dealer: 'E', upcard: 'Td'}),
      0,
    );
    const east = s.players.find(p => p.name === 'East');
    expect(east?.hidden).toBe(true);
    expect(east?.hand).toHaveLength(6);
    expect(east?.hand).not.toContain('Td');
    expect(s.pickup).toEqual({dealer: 'left', card: 'Td'});
  });
});

const endStruct = {
  hands: {N: [], E: [], S: [], W: nulls(5)},
  upcard: 'Td',
  trumpSuit: 'h',
  dealer: 'N',
  phase: 'terminal',
  currentTrick: [],
  tricksWon: {NS: 3, EW: 2},
  declarer: 'E',
  goingAlone: true,
  currentPlayer: -4,
};

describe('the hand at game over', () => {
  it('draws played-out hands empty, keeps the sitting-out partner, and scores from returns', () => {
    const s = state(
      session(endStruct, {returns: [2, -2, 2, -2], isTerminal: true}),
      0,
    );
    const hand = (name: string) =>
      s.players.find(p => p.name.startsWith(name))?.hand;
    expect(hand('North')).toEqual([]);
    expect(hand('East')).toEqual([]);
    expect(hand('South')).toEqual([]);
    expect(hand('West')).toHaveLength(5);
    expect(s.aloneSeat).toBe('East');
    expect(s.sittingOut).toBe('right');
    expect(s.outcome?.points).toEqual({NS: 2, EW: 0});
    expect(s.statusMessage).toBe('North-South score 2 points');
  });

  it('fails loud at game over without returns', () => {
    expect(() =>
      euchreUIState(session(endStruct, {returns: null, isTerminal: true}), 0),
    ).toThrow(/without returns/);
  });

  it('names no lone seat and no outcome mid-hand', () => {
    const s = state(
      session({
        ...endStruct,
        hands: {N: ['9s', 'Jh'], E: nulls(2), S: nulls(2), W: nulls(2)},
        phase: 'playing',
        goingAlone: false,
        tricksWon: {NS: 1, EW: 0},
      }),
      0,
    );
    expect(s.aloneSeat).toBeNull();
    expect(s.sittingOut).toBeNull();
    expect(s.outcome).toBeNull();
    expect(s.players[0]?.hand).toHaveLength(2);
  });
});

describe('euchreOutcome: the scoring cases, points from the engine', () => {
  it('makers made it', () => {
    expect(
      euchreOutcome({
        declarer: 'N',
        goingAlone: false,
        tricksWon: {NS: 3, EW: 2},
        returns: [1, -1, 1, -1],
      }),
    ).toEqual({
      points: {NS: 1, EW: 0},
      scorer: 'NS',
      cause: 'North-South, the makers, won 3 tricks.',
    });
  });

  it('a march and a lone march', () => {
    expect(
      euchreOutcome({
        declarer: 'W',
        goingAlone: false,
        tricksWon: {NS: 0, EW: 5},
        returns: [-2, 2, -2, 2],
      }).cause,
    ).toBe('East-West, the makers, won all 5 tricks: a march.');
    const lone = euchreOutcome({
      declarer: 'S',
      goingAlone: true,
      tricksWon: {NS: 5, EW: 0},
      returns: [4, -4, 4, -4],
    });
    expect(lone.points).toEqual({NS: 4, EW: 0});
    expect(lone.cause).toBe('South went alone and won all 5 tricks: a march.');
  });

  it('makers euchred, and a thrown-in hand', () => {
    expect(
      euchreOutcome({
        declarer: 'E',
        goingAlone: false,
        tricksWon: {NS: 5, EW: 0},
        returns: [2, -2, 2, -2],
      }).cause,
    ).toBe('East-West, the makers, were euchred with no trick.');
    expect(
      euchreOutcome({
        declarer: null,
        goingAlone: false,
        tricksWon: {NS: 0, EW: 0},
        returns: [0, 0, 0, 0],
      }),
    ).toEqual({
      points: {NS: 0, EW: 0},
      scorer: null,
      cause: 'Thrown in: no one named trump.',
    });
  });

  it('refuses returns that are not one zero-sum score per side', () => {
    const args = {declarer: 'N', goingAlone: false, tricksWon: {NS: 3, EW: 2}};
    expect(() =>
      euchreOutcome({...args, declarer: 'N', returns: [1, -1, 2, -1]}),
    ).toThrow(/zero-sum/);
    expect(() =>
      euchreOutcome({...args, declarer: 'N', returns: [1, -1]}),
    ).toThrow(/expected 4 returns/);
  });
});

describe('counts and turn line follow the held trick', () => {
  const teamOf: EuchreView['teamOf'] = {
    bottom: 'NS',
    top: 'NS',
    left: 'EW',
    right: 'EW',
  };
  const live: EuchreView = {
    players: [
      {name: 'North', position: 'bottom', hand: ['9s']},
      {name: 'East', position: 'left', hand: ['9c']},
      {name: 'South', position: 'top', hand: ['9d']},
      {name: 'West', position: 'right', hand: ['9h']},
    ],
    trickPile: [],
    isTerminal: false,
    phase: 'playing',
    trumpSuit: 'h',
    tricksWon: {NS: 3, EW: 1},
    teamOf,
    statusMessage: 'Your turn to play',
  };
  const trick = (
    index: number,
    winner: PlayerPosition,
  ): CompletedTrick<CardCode, PlayerPosition> => ({
    index,
    cards: [
      {card: 'Ah', seat: 'bottom'},
      {card: 'Kh', seat: 'left'},
      {card: 'Qh', seat: 'top'},
      {card: 'Th', seat: 'right'},
    ],
    winner,
  });

  it('leaves the live counts and turn line alone when nothing is held', () => {
    const view = euchreHeldView(live, null);
    expect(view.tricksWon).toEqual({NS: 3, EW: 1});
    expect(view.statusMessage).toBe('Your turn to play');
  });

  it('does not count the held trick, or queued ones, until collected', () => {
    expect(euchreHeldView(live, trick(3, 'top')).tricksWon).toEqual({
      NS: 2,
      EW: 1,
    });
    expect(
      euchreHeldView(live, trick(2, 'top'), [trick(3, 'left')]).tricksWon,
    ).toEqual({NS: 2, EW: 0});
  });

  it('names the winner only once the trick has landed', () => {
    const held = trick(3, 'left');
    const state = {
      ...live,
      compassAt: {bottom: 'N', left: 'E', top: 'S', right: 'W'},
    } as unknown as EuchreUIState;
    const collect = {trick: held, pending: []} as unknown as EuchreTrickCollect;
    const view = euchreHeldView(live, held);
    const turn = {} as EuchreTurn;
    // A card of the trick still to fly in: the line names its player.
    expect(
      statusLine(state, view, collect, {card: 'Th', position: 'right'}, turn),
    ).toBe('West to play');
    expect(statusLine(state, view, collect, null, turn)).toBe(
      'East-West win the trick',
    );
  });

  it('names the trick winner while a trick is held', () => {
    expect(euchreHeldView(live, trick(3, 'left')).statusMessage).toBe(
      'East-West win the trick',
    );
  });

  it('fails loud when the live counts cannot hold the held trick', () => {
    expect(() =>
      euchreHeldView({...live, tricksWon: {NS: 0, EW: 1}}, trick(0, 'top')),
    ).toThrow();
  });
});

describe('sides follow the compass seat, not the place at the table', () => {
  const midHand = {
    hands: {N: ['9s'], E: nulls(1), S: nulls(1), W: nulls(1)},
    upcard: 'Td',
    trumpSuit: 'h',
    dealer: 'W',
    phase: 'playing',
    currentTrick: [],
    lastTrick: {
      index: 3,
      cards: [
        {card: 'Ah', player: 0},
        {card: 'Kh', player: 1},
        {card: 'Jh', player: 2},
        {card: 'Th', player: 3},
      ],
      winner: 'S',
    },
    tricksWon: {NS: 3, EW: 1},
    declarer: 'N',
    goingAlone: false,
    currentPlayer: 1,
    discarding: false,
  };

  it('an East viewer sits with West at bottom and top, East-West', () => {
    const s = state(session(midHand, {currentPlayer: 1, mode: 'analysis'}), 1);
    expect(s.teamOf).toEqual({
      bottom: 'EW',
      top: 'EW',
      left: 'NS',
      right: 'NS',
    });
    const last = s.lastTrick;
    if (last === null) throw new Error('No last trick.');
    const won = {...last, winner: 'bottom' as const};
    expect(euchreHeldView(s, won).tricksWon).toEqual({NS: 3, EW: 0});
  });
});
