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

import type {PanelGroupDefinition} from '../dock/panel_groups.ts';
import {useGame} from '../game/game_context.ts';
import {Perspective} from './perspective.tsx';

import './view.css';

// View › Perspective: whose view of the game the page shows.
function PerspectiveContent(): JSX.Element {
  const {entry, session, viewer, setViewer} = useGame();
  const numSeats = session.frame?.session.num_seats ?? entry.num_players;
  if (numSeats === null) {
    throw new Error(`${entry.name} is not loadable, so it has no seats.`);
  }
  return (
    <div className="view">
      <Perspective viewer={viewer} numSeats={numSeats} onChange={setViewer} />
    </div>
  );
}

/**
 * The View panel: how the game is shown, one section per inner tab. Board
 * display settings come first, should a board have any, then Perspective.
 * The page's theme is an app setting, in the header's Settings.
 */
export const VIEW_GROUP: PanelGroupDefinition = {
  id: 'view',
  title: 'View',
  sections: [
    {id: 'perspective', title: 'Perspective', Content: PerspectiveContent},
  ],
};
