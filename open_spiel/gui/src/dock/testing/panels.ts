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

import {definePanels} from '../panels.ts';

/**
 * Panels for the dock's tests, laid out in three columns: Gameplay; Actions |
 * State; Setup | View over Players | Session, Session closed at first. Where
 * the dock is too narrow for three columns, the third moves under the
 * second.
 */
export const PANELS = definePanels([
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
    initialWidth: 360,
  },
  {
    id: 'state',
    title: 'State',
    position: {referencePanel: 'actions', direction: 'within'},
    minimumWidth: 260,
    minimumHeight: 200,
  },
  {
    id: 'setup',
    title: 'Setup',
    position: {referencePanel: 'actions', direction: 'right'},
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
  {
    id: 'players',
    title: 'Players',
    position: {referencePanel: 'setup', direction: 'below'},
    minimumWidth: 260,
    minimumHeight: 200,
  },
  {
    id: 'session',
    title: 'Session',
    position: {referencePanel: 'players', direction: 'within'},
    minimumWidth: 260,
    minimumHeight: 200,
    initiallyOpen: false,
  },
]);

/** The id of a test panel. */
export type PanelId = (typeof PANELS.list)[number]['id'];
