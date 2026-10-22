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
import {createContext, type JSX, useContext} from 'react';

import {NoFrame} from '../actions/note.tsx';
import type {Frame} from '../contract/contract.ts';
import {CompactText, CopyableText} from '../controls/copy_button.tsx';
import {PanelGroup} from '../dock/panel_group.tsx';
import type {
  PanelGroupDefinition,
  SectionDefinition,
} from '../dock/panel_groups.ts';
import {GameContext} from '../game/game_context.ts';
import type {StateBlock, StateBlockId} from './state_blocks.ts';
import {stateBlocks} from './state_blocks.ts';

import './state.css';

// The frame a StatePanel was given; the page's frame when undefined.
const StateFrame = createContext<Frame | null | undefined>(undefined);

// The frame the State sections show.
function useStateFrame(): Frame | null {
  const given = useContext(StateFrame);
  const game = useContext(GameContext);
  if (given !== undefined) return given;
  if (game === null) {
    throw new Error('A State section is drawn outside a GamePage.');
  }
  return game.session.frame;
}

/**
 * One section's content: the state text or the struct in a card with its
 * copy icon; the returns or rewards sized to their lines; or why the state
 * text is missing.
 */
export function StateBlockView({block}: {block: StateBlock}): JSX.Element {
  if ('missing' in block) {
    return <p className="state-panel__missing">{block.missing}</p>;
  }
  return block.id === 'returns' || block.id === 'rewards' ? (
    <CompactText text={block.text} label={block.title} />
  ) : (
    <CopyableText text={block.text} label={block.title} />
  );
}

// The section of block `id`, titled `title` when the frame lacks it. It
// draws nothing, so has no tab, when the frame lacks the block; Text says
// there is no position while there is no frame.
function blockSection(id: StateBlockId, title: string): SectionDefinition {
  const find = (frame: Frame | null): StateBlock | undefined =>
    frame === null
      ? undefined
      : stateBlocks(frame).find(block => block.id === id);
  return {
    id,
    title,
    Content: () => {
      const frame = useStateFrame();
      if (frame === null) return id === 'text' ? <NoFrame /> : null;
      const block = find(frame);
      if (block === undefined) return null;
      return (
        <div className="state-panel__section">
          <StateBlockView block={block} />
        </div>
      );
    },
    Label: () => find(useStateFrame())?.title ?? title,
  };
}

/**
 * The State panel's sections: Text (Observation for a seat that may not see
 * the full state), Struct, Returns and Rewards. A section the frame lacks
 * has no tab.
 */
export const STATE_GROUP: PanelGroupDefinition = {
  id: 'state',
  title: 'State',
  sections: [
    blockSection('text', 'Text'),
    blockSection('struct', 'Struct'),
    blockSection('returns', 'Returns'),
    blockSection('rewards', 'Rewards'),
  ],
};

/** The position in `frame` as the State panel's group of sections. */
export function StatePanel({frame}: {frame: Frame}): JSX.Element {
  return (
    <StateFrame value={frame}>
      <PanelGroup group={STATE_GROUP} />
    </StateFrame>
  );
}
