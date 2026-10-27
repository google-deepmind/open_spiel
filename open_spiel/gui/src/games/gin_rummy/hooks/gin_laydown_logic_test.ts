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

import {describe, expect, it} from 'vitest';

import type {LaydownPlan, PlayerMeld} from '../types.ts';
import {cycleMeld} from './gin_laydown_cycle.ts';
import {
  buildPlanSets,
  compositionProblem,
  type DoneCheck,
  driveLaydown,
  refusalMessage,
  serializeSignature,
  sliceFor,
} from './gin_laydown_logic.ts';

function meld(cards: string[], locked = false): PlayerMeld {
  const type = cards.every(c => c.startsWith(cards[0]?.charAt(0) ?? ''))
    ? 'set'
    : 'run';
  return {id: serializeSignature(cards), type, cards, locked};
}

function plan(...steps: Array<[string[], number]>): LaydownPlan {
  const sequence = steps.map(([cards, action]) => ({
    cards,
    action,
    signature: serializeSignature(cards),
  }));
  return {sequence, signatureSet: new Set(sequence.map(s => s.signature))};
}

function signatures(...melds: string[][]): Set<string> {
  return new Set(melds.map(serializeSignature));
}

const SEVENS = ['7s', '7c', '7d'];
const CLUB_RUN = ['8c', '9c', 'Tc'];
const DEADWOOD = ['2s', 'Ah', '6h', 'Ac'];

// A knocker's hand after the knock, with 4h to discard and the knock card
// 10.
function knockerCheck(over: Partial<DoneCheck>): DoneCheck {
  return {
    stage: 'discard',
    knocker: true,
    knockCard: 10,
    hand: [...DEADWOOD, ...SEVENS, ...CLUB_RUN, '4h'],
    melded: new Set(),
    stagedDiscard: '4h',
    plans: [],
    meldSignatures: new Set(),
    layoffSignatures: new Set(),
    ...over,
  };
}

describe('compositionProblem', () => {
  it("stops the knocker's Done while the deadwood is over the knock card", () => {
    const problem = compositionProblem(knockerCheck({melded: new Set(SEVENS)}));
    expect(problem).toBe(
      'Deadwood 37 is over the knock card (10): stage more melds or ' +
        'discard another card, then Done.',
    );
  });

  it('lets the knocker lay down once the deadwood reaches the knock card', () => {
    const melded = new Set([...SEVENS, ...CLUB_RUN]);
    expect(compositionProblem(knockerCheck({melded}))).toBeNull();
  });

  it('waits for the discard before judging the discard stage', () => {
    expect(compositionProblem(knockerCheck({stagedDiscard: null}))).toBeNull();
  });

  it('judges a resumed meld stage on the cards left in hand', () => {
    // 7s7c7d is laid; the engine's one plan is 8c9cTc.
    const check = knockerCheck({
      stage: 'melds',
      hand: [...DEADWOOD, ...CLUB_RUN],
      melded: new Set(SEVENS),
      stagedDiscard: null,
      plans: [plan([CLUB_RUN, 139])],
    });
    expect(compositionProblem(check)).toMatch(/^Deadwood 37 is over/);
    expect(
      compositionProblem({
        ...check,
        melded: new Set([...SEVENS, ...CLUB_RUN]),
        meldSignatures: signatures(CLUB_RUN),
      }),
    ).toBeNull();
  });

  it('refuses a staged meld set that is in no plan', () => {
    const problem = compositionProblem({
      ...knockerCheck({}),
      stage: 'melds',
      knocker: false,
      plans: [plan([CLUB_RUN, 139])],
      meldSignatures: signatures(SEVENS),
    });
    expect(problem).toBe(
      'These melds cannot be laid together: change them, then Done.',
    );
  });

  it('refuses a staged lay-off set that is in no plan', () => {
    const problem = compositionProblem({
      ...knockerCheck({}),
      stage: 'layoffs',
      knocker: false,
      plans: [plan([['5c'], 4])],
      layoffSignatures: signatures(['6c']),
    });
    expect(problem).toMatch(/^These lay-offs cannot be laid together/);
  });
});

describe('sliceFor', () => {
  const actions = new Map([
    [serializeSignature(SEVENS), 89],
    [serializeSignature(CLUB_RUN), 139],
  ]);
  const planSets = [signatures(SEVENS, CLUB_RUN)];

  it('sends nothing for melds that each resolve but are in no plan together', () => {
    const slice = sliceFor(
      'melds',
      54,
      actions,
      {stagedMelds: [meld(SEVENS)], stagedLayoffs: [], discardActionId: null},
      planSets,
    );
    expect(slice).toMatchObject({kind: 'error'});
  });

  it('sends a plan, then the pass', () => {
    const slice = sliceFor(
      'melds',
      54,
      actions,
      {
        stagedMelds: [meld(SEVENS), meld(CLUB_RUN)],
        stagedLayoffs: [],
        discardActionId: null,
      },
      planSets,
    );
    expect(slice).toEqual({kind: 'submit', actions: [89, 139, 54]});
  });

  it('sends a bare pass for an empty composition', () => {
    const slice = sliceFor(
      'melds',
      54,
      actions,
      {stagedMelds: [], stagedLayoffs: [], discardActionId: null},
      planSets,
    );
    expect(slice).toEqual({kind: 'submit', actions: [54]});
  });
});

describe('buildPlanSets', () => {
  it("reads each raw plan's step signatures", () => {
    const sets = buildPlanSets([
      [{cards: CLUB_RUN, action: 139}],
      [],
      [{cards: [], action: 1}],
    ]);
    expect(sets).toEqual([signatures(CLUB_RUN)]);
  });
});

describe('refusalMessage', () => {
  it('says the refusal once, with no doubled stop', () => {
    const message = refusalMessage(
      new Error('Action 54 is not legal at ply 70.'),
    );
    expect(message).toContain('Action 54 is not legal at ply 70. The table');
    expect(message).not.toContain('..');
  });
});

describe('driveLaydown', () => {
  it('reports a refusal and stops', async () => {
    const errors: string[] = [];
    await driveLaydown(
      {
        seat: 0,
        stage: 'melds',
        passAction: 54,
        signatureActions: new Map([[serializeSignature(CLUB_RUN), 139]]),
        planSets: [signatures(CLUB_RUN)],
      },
      {stagedMelds: [meld(CLUB_RUN)], stagedLayoffs: [], discardActionId: null},
      () => Promise.reject(new Error('Action 54 is not legal at ply 70.')),
      message => {
        errors.push(message);
      },
    );
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(/^The lay-down was not accepted: Action 54/);
  });
});

describe('cycleMeld at the meld stage', () => {
  // 8s8c8d is a meld of the hand, but the only lay-down
  // that passes uses 8s8c8h, 6d7d8d and 5h6h7h.
  const melds = [
    meld(['8s', '8c', '8d']),
    meld(['8s', '8c', '8h']),
    meld(['6d', '7d', '8d']),
    meld(['5h', '6h', '7h']),
  ];
  const plans = [
    plan(
      [['8s', '8c', '8h'], 120],
      [['6d', '7d', '8d'], 121],
      [['5h', '6h', '7h'], 122],
    ),
  ];

  it('skips a meld that no plan holds', () => {
    const result = cycleMeld(
      {
        playerMelds: melds,
        playerLockedMeldIds: new Set(),
        laydownPlans: plans,
        stagedMeldIds: new Set(),
        stagedDiscard: null,
        laydownStage: 'melds',
        discardEligible: false,
      },
      '8c',
    );
    expect(result).toMatchObject({kind: 'stage', flashed: true});
    if (result.kind !== 'stage') throw new Error('not staged');
    expect([...result.staged]).toEqual([
      serializeSignature(['8s', '8c', '8h']),
    ]);
  });
});
