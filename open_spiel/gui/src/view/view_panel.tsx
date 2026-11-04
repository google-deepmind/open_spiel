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

import type {
  PanelGroupDefinition,
  SectionDefinition,
} from '../dock/panel_groups.ts';
import {useGame} from '../game/game_context.ts';
import {BOARD_VIEW_SECTIONS} from '../renderers/registry.ts';
import {Perspective} from './perspective.tsx';

import './view.css';

// View › Perspective: the seat the board is drawn from.
function PerspectiveContent(): JSX.Element {
  const {entry, session, viewer} = useGame();
  const numSeats = session.frame?.session.num_seats ?? entry.num_players;
  if (numSeats === null) {
    throw new Error(`${entry.name} is not loadable, so it has no seats.`);
  }
  return (
    <div className="view">
      <Perspective viewer={viewer} numSeats={numSeats} />
    </div>
  );
}

// A board's own section, drawn only while that board draws the game: a
// section that draws nothing has no tab.
function boardSection(
  renderer: string,
  section: SectionDefinition,
): SectionDefinition {
  const {Content} = section;
  function BoardContent(): JSX.Element | null {
    const {session} = useGame();
    const shown = session.frame?.presentation?.renderer ?? null;
    return shown === renderer ? <Content /> : null;
  }
  return {...section, Content: BoardContent};
}

const BOARD_SECTIONS = BOARD_VIEW_SECTIONS.map(({renderer, section}) =>
  boardSection(renderer, section),
);

/**
 * The View panel: how the game is shown, one section per inner tab. Board
 * display settings come first, should a board have any, then Perspective,
 * then the board's Animation section, should it animate. The page's theme is
 * an app setting, in the header's Settings.
 */
export const VIEW_GROUP: PanelGroupDefinition = {
  id: 'view',
  title: 'View',
  sections: [
    ...BOARD_SECTIONS.filter(section => !section.id.endsWith('_animation')),
    {id: 'perspective', title: 'Perspective', Content: PerspectiveContent},
    ...BOARD_SECTIONS.filter(section => section.id.endsWith('_animation')),
  ],
};
