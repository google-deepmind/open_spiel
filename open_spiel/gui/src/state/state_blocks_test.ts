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

import {stateBlocks} from './state_blocks.ts';
import {recordedFrame} from './testing/recorded_frames.ts';

const ZERO_REWARDS = {
  id: 'rewards',
  title: 'Rewards',
  text: 'Player 0: 0\nPlayer 1: 0',
};
const KUHN_RESULT = 'Player 0: -2\nPlayer 1: 2';

async function blocks(name: string): Promise<ReturnType<typeof stateBlocks>> {
  return stateBlocks(await recordedFrame(name));
}

describe('stateBlocks in a game that hides information', () => {
  it('shows the oracle the full state, returns and rewards, in that order', async () => {
    expect(await blocks('kuhn_poker/oracle/live')).toEqual([
      {id: 'text', title: 'Text', text: '0 2 pbb'},
      {id: 'returns', title: 'Returns', text: KUHN_RESULT},
      {id: 'rewards', title: 'Rewards', text: KUHN_RESULT},
    ]);
  });

  it('shows a seat its observation, and no information state', async () => {
    expect(await blocks('kuhn_poker/0/3')).toEqual([
      {id: 'text', title: 'Observation', text: '011'},
      ZERO_REWARDS,
    ]);
  });

  it('omits a reward hidden with the action that earned it', async () => {
    expect(await blocks('kuhn_poker/1/3')).toEqual([
      {id: 'text', title: 'Observation', text: '211'},
    ]);
    // Returns go to every viewer.
    expect(
      (await blocks('kuhn_poker/1/live')).find(block => block.id === 'returns'),
    ).toEqual({id: 'returns', title: 'Returns', text: KUHN_RESULT});
  });

  it('says so when the game gives a seat no observation text', async () => {
    const frame = await recordedFrame('kuhn_poker/0/3');
    expect(stateBlocks({...frame, observation_text: null})[0]).toEqual({
      id: 'text',
      title: 'Observation',
      missing: 'Observation unavailable for your seat.',
    });
  });

  it('tells a spectator of a game that hides information there is no public text', async () => {
    expect(await blocks('kuhn_poker/spectator/3')).toEqual([
      {
        id: 'text',
        title: 'Text',
        missing: 'No public state text for this game.',
      },
    ]);
  });

  it('reads an empty text as none, and gives no rewards at the start', async () => {
    expect(await blocks('kuhn_poker/oracle/0')).toEqual([
      {id: 'text', title: 'Text', missing: 'State text unavailable.'},
    ]);
    expect(await blocks('kuhn_poker/0/0')).toEqual([
      {
        id: 'text',
        title: 'Observation',
        missing: 'Observation unavailable for your seat.',
      },
    ]);
  });
});

describe('stateBlocks in a game that hides nothing', () => {
  it('shows the struct as indented JSON when the game has one', async () => {
    const frame = await recordedFrame('tic_tac_toe/oracle/2');
    expect(stateBlocks(frame)).toEqual([
      {id: 'text', title: 'Text', text: 'o..\n.x.\n...'},
      {
        id: 'struct',
        title: 'Struct',
        text: JSON.stringify(frame.struct, null, 2),
      },
      ZERO_REWARDS,
    ]);
  });

  it('shows every viewer the full state of a game that hides nothing', async () => {
    for (const viewer of ['0', 'spectator']) {
      const shown = await blocks(`tic_tac_toe/${viewer}/2`);
      expect(shown[0]).toEqual({
        id: 'text',
        title: 'Text',
        text: 'o..\n.x.\n...',
      });
      expect(shown.map(block => block.id)).toEqual([
        'text',
        'struct',
        'rewards',
      ]);
    }
  });

  it('refuses an oracle frame that is not omniscient', async () => {
    const frame = await recordedFrame('tic_tac_toe/oracle/2');
    expect(() => stateBlocks({...frame, omniscient: false})).toThrow(
      'An oracle frame is omniscient.',
    );
  });

  it('refuses an omniscient frame without the state text', async () => {
    const frame = await recordedFrame('tic_tac_toe/oracle/2');
    expect(() => stateBlocks({...frame, state_text: null})).toThrow(
      'An omniscient frame carries the state text.',
    );
  });
});
