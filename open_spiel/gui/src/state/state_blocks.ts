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

import type {Frame} from '../contract/contract.ts';

/** What a section of the State panel shows. */
export type StateBlockId = 'text' | 'struct' | 'returns' | 'rewards';

/**
 * One section of the State panel: its text, or, for the state text alone,
 * a sentence saying why it has none.
 */
export type StateBlock = {
  readonly id: StateBlockId;
  readonly title: string;
} & ({readonly text: string} | {readonly missing: string});

// One line per seat, as "Player i: v", each value exactly as the game gives
// it.
function perSeat(values: readonly number[]): string {
  return values
    .map((value, seat) => `Player ${String(seat)}: ${String(value)}`)
    .join('\n');
}

// The state as text, or the viewer's observation when the viewer may not see
// the full state. A text the game leaves empty, as at a position before the
// deal, reads as none.
function textBlock(frame: Frame): StateBlock {
  if (frame.omniscient) {
    if (frame.state_text === null) {
      throw new Error('An omniscient frame carries the state text.');
    }
    return frame.state_text === ''
      ? {id: 'text', title: 'Text', missing: 'State text unavailable.'}
      : {id: 'text', title: 'Text', text: frame.state_text};
  }
  if (frame.viewer.kind === 'oracle') {
    throw new Error('An oracle frame is omniscient.');
  }
  if (frame.viewer.kind === 'seat') {
    const text = frame.observation_text ?? '';
    return text === ''
      ? {
          id: 'text',
          title: 'Observation',
          missing: 'Observation unavailable for your seat.',
        }
      : {id: 'text', title: 'Observation', text};
  }
  return {
    id: 'text',
    title: 'Text',
    missing: 'No public state text for this game.',
  };
}

/**
 * The sections the State panel shows for `frame`, in order: the state as
 * text (the viewer's observation when the viewer may not see the full
 * state), then the struct, the returns and the rewards, each only when the
 * frame carries it.
 */
export function stateBlocks(frame: Frame): StateBlock[] {
  const blocks = [textBlock(frame)];
  // Null when the viewer may not see the full state, and when the game
  // defines no struct.
  if (frame.struct !== null) {
    blocks.push({
      id: 'struct',
      title: 'Struct',
      text: JSON.stringify(frame.struct, null, 2),
    });
  }
  if (frame.returns !== null) {
    blocks.push({
      id: 'returns',
      title: 'Returns',
      text: perSeat(frame.returns),
    });
  }
  if (frame.rewards !== null) {
    blocks.push({
      id: 'rewards',
      title: 'Rewards',
      text: perSeat(frame.rewards),
    });
  }
  return blocks;
}
