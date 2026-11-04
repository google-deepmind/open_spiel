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
import type {JSX} from 'react';

import {History} from '../actions/history.tsx';
import {HistoryText} from '../actions/history_text.tsx';
import {LegalActions, legalLabel} from '../actions/legal_actions.tsx';
import {
  type PanelGroupDefinition,
  usePaneHidden,
} from '../dock/panel_groups.ts';
import {ReplayControl} from '../state/replay_control.tsx';
import {useGame} from './game_context.ts';

import './game.css';

function HistoryContent(): JSX.Element {
  return <History session={useGame().session} />;
}

// History's footer: the replay control, which steps through the positions.
function HistoryFooter(): JSX.Element | null {
  const {entry, session} = useGame();
  const {frame} = session;
  if (frame === null) return null;
  return (
    <ReplayControl
      frame={frame}
      busy={session.busy}
      chanceMode={entry.chance_mode}
      onGoTo={session.commands.goTo}
    />
  );
}

function LegalContent(): JSX.Element {
  return <LegalActions session={useGame().session} hidden={usePaneHidden()} />;
}

function LegalLabel(): string {
  return legalLabel(useGame().session.frame);
}

function TextContent(): JSX.Element {
  return <HistoryText session={useGame().session} />;
}

/**
 * The Actions panel: the history, the legal actions (with a chance node's
 * outcomes) and the history as text, a tab each. The history steps through
 * the positions from its footer, the replay control.
 */
export const ACTIONS_GROUP: PanelGroupDefinition = {
  id: 'actions',
  title: 'Actions',
  sections: [
    {
      id: 'history',
      title: 'History',
      Content: HistoryContent,
      Footer: HistoryFooter,
    },
    {id: 'legal', title: 'Legal', Content: LegalContent, Label: LegalLabel},
    {id: 'text', title: 'Text', Content: TextContent},
  ],
};
