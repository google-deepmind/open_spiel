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

import type {IDockviewPanelProps} from 'dockview-core';
import type {FunctionComponent, JSX, ReactNode} from 'react';

import {Note} from '../actions/note.tsx';
import {useShowsSessionError} from '../dock/dock_context.ts';
import {useGroupMinimumHeight} from '../dock/group_minimum.ts';
import {PanelGroup} from '../dock/panel_group.tsx';
import type {PanelGroupDefinition} from '../dock/panel_groups.ts';
import {
  SETUP_GROUP,
  SeatsOverflowContext,
  SetupPanel,
} from '../setup/setup_panel.tsx';
import {STATE_GROUP} from '../state/state_panel.tsx';
import {VIEW_GROUP} from '../view/view_panel.tsx';
import {ACTIONS_GROUP} from './actions_panel.tsx';
import {GAME_PANELS, type GamePanelId} from './game_panels.ts';
import {useGame} from './game_context.ts';
import {GameplayPanel} from './gameplay_panel.tsx';
import {SESSION_GROUP} from './session_panel.tsx';

import './game.css';

// The panels that show the position, and so have nothing to show without
// one: while the session has failed, they and the sections popped out of
// them point to its error.
const POSITION_PANELS: ReadonlySet<string> = new Set(['actions', 'state']);

// The panel `id` is, or the group a section popped out as `id` came from.
function groupOf(id: string): string {
  const [group] = id.split('.');
  if (group === undefined) throw new Error(`No panel id in "${id}".`);
  return group;
}

/**
 * A panel's frame: the panel's content, which scrolls within the panel. The
 * session's error is shown once: over the Gameplay board, or, while the
 * Gameplay panel is not on screen, above the content of the one panel that
 * shows it. A panel that shows the position, left without one by the
 * error, says that the Gameplay panel has it.
 */
export function PanelFrame({
  id,
  children,
}: {
  readonly id: string;
  readonly children: ReactNode;
}): JSX.Element {
  const {error, frame} = useGame().session;
  const gameplayShowsError = useShowsSessionError('gameplay');
  const showsError = useShowsSessionError(id) && id !== 'gameplay';
  const pointsToError =
    error !== null &&
    frame === null &&
    gameplayShowsError &&
    POSITION_PANELS.has(groupOf(id));
  return (
    <div className="panel">
      {showsError && error !== null && (
        <p className="panel__error" role="alert">
          {error.message}
        </p>
      )}
      <div className="panel__content">
        {pointsToError ? (
          <Note>Session unavailable. The Gameplay panel has the details.</Note>
        ) : (
          children
        )}
      </div>
    </div>
  );
}

function SetupContent(props: IDockviewPanelProps): JSX.Element {
  // While Players is open, Setup's dock group stays tall enough for every
  // seat.
  const onSeatsOverflow = useGroupMinimumHeight(
    props,
    GAME_PANELS.get('setup').minimumHeight,
  );
  return (
    <SeatsOverflowContext value={onSeatsOverflow}>
      <SetupPanel />
    </SeatsOverflowContext>
  );
}

// A dock component that draws `Content` in panel `id`'s frame.
function framed(
  id: GamePanelId,
  Content: (props: IDockviewPanelProps) => JSX.Element,
): FunctionComponent<IDockviewPanelProps> {
  const Panel = (props: IDockviewPanelProps): JSX.Element => (
    <PanelFrame id={id}>
      <Content {...props} />
    </PanelFrame>
  );
  Panel.displayName = `${id} panel`;
  return Panel;
}

// A dock component that draws `group` in its panel's frame.
function grouped(
  id: GamePanelId,
  group: PanelGroupDefinition,
): FunctionComponent<IDockviewPanelProps> {
  return framed(id, () => <PanelGroup group={group} />);
}

/** The game page's panel groups, whose sections can be popped out. */
export const PANEL_GROUPS: readonly PanelGroupDefinition[] = [
  ACTIONS_GROUP,
  STATE_GROUP,
  SESSION_GROUP,
  SETUP_GROUP,
  VIEW_GROUP,
];

/** What each panel of a game page draws, keyed by panel id. */
export const PANEL_COMPONENTS: Readonly<
  Record<GamePanelId, FunctionComponent<IDockviewPanelProps>>
> = {
  gameplay: framed('gameplay', GameplayPanel),
  actions: grouped('actions', ACTIONS_GROUP),
  state: grouped('state', STATE_GROUP),
  session: grouped('session', SESSION_GROUP),
  setup: framed('setup', SetupContent),
  view: grouped('view', VIEW_GROUP),
};
