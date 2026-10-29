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

import {definePanels} from '../dock/panels.ts';

/**
 * The game page's panels. Gameplay is on the left; beside it, one column of
 * Actions | State over Setup | View. Session is closed at first, and opens
 * from Panel options as a tab of Actions.
 */
export const GAME_PANELS = definePanels([
  {
    id: 'gameplay',
    title: 'Gameplay',
    primary: true,
    minimumWidth: 320,
    minimumHeight: 240,
  },
  {
    id: 'actions',
    title: 'Actions',
    position: {referencePanel: 'gameplay', direction: 'right'},
    minimumWidth: 260,
    minimumHeight: 200,
    initialWidth: 400,
  },
  {
    id: 'state',
    title: 'State',
    position: {referencePanel: 'actions', direction: 'within'},
    minimumWidth: 260,
    minimumHeight: 200,
  },
  {
    id: 'session',
    title: 'Session',
    position: {referencePanel: 'actions', direction: 'within'},
    minimumWidth: 260,
    minimumHeight: 200,
    initiallyOpen: false,
  },
  {
    id: 'setup',
    title: 'Setup',
    position: {referencePanel: 'actions', direction: 'below'},
    minimumWidth: 260,
    minimumHeight: 200,
  },
  {
    id: 'view',
    title: 'View',
    position: {referencePanel: 'setup', direction: 'within'},
    minimumWidth: 260,
    minimumHeight: 200,
  },
]);

/** The id of a game page panel: a key of the components the page docks. */
export type GamePanelId = (typeof GAME_PANELS.list)[number]['id'];
