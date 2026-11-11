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

// STRUCT-GLUE(euchre StateStruct): the board reads the adapter's
// struct-shaped view of the table, filtered per viewer (euchre_adapter.py).
//
// The euchre renderer: the euchre board, given the frame as the session it
// reads (euchre_session.ts) and the viewer's moves. It keeps its state from
// one ply to the next, as the trick on show, the plays still landing and the
// cards in flight need, and says whose turn it is on its own table.

import {type JSX, useMemo} from 'react';

import type {Frame} from '../../contract/contract.ts';
import type {SectionDefinition} from '../../dock/panel_groups.ts';
import type {RendererEntry, RendererProps} from '../../renderers/renderer.ts';
import {AnimationSpeedField} from '../../view/animation_speed_field.tsx';
import {EuchreGameplayConfigExtras} from './euchre_gameplay_config_extras.tsx';
import {createEuchreGameplayPanel} from './euchre_gameplay_panel.tsx';
import {
  EuchreSessionContext,
  type SessionContextValue,
  euchreReplay,
  euchreSession,
  euchreViewer,
  useEuchreSession,
} from './euchre_session.ts';

// One panel component for every frame, so the board keeps its state.
const EuchreGameplay = createEuchreGameplayPanel(useEuchreSession);

function toMoveIsViewer(frame: Frame): boolean {
  const {to_move: toMove, viewer} = frame;
  return toMove.kind === 'seat' && toMove.seat === viewer.seat;
}

function cannotMove(): void {
  throw new Error('You cannot move at this position.');
}

/** Draws a euchre frame with the euchre board. */
export function EuchreRenderer({
  frame,
  presentation,
  play,
  busy = false,
}: RendererProps): JSX.Element {
  const value = useMemo(
    (): SessionContextValue => ({
      session: euchreSession(frame, presentation),
      error: null,
      viewer: euchreViewer(frame),
      replay: euchreReplay(frame),
      // A seat to move that cannot play here (a replay, a paused or finished
      // game) is busy to the board, as it is while a move is in flight.
      isBusy: busy || (play === null && toMoveIsViewer(frame)),
      submitAction: play ?? cannotMove,
    }),
    [frame, presentation, play, busy],
  );
  return (
    <EuchreSessionContext.Provider value={value}>
      <EuchreGameplay />
    </EuchreSessionContext.Provider>
  );
}

const VIEW_SECTIONS: readonly SectionDefinition[] = [
  {id: 'euchre', title: 'Euchre', Content: EuchreGameplayConfigExtras},
  {id: 'euchre_animation', title: 'Animation', Content: AnimationSpeedField},
];

/** The euchre renderer's registry entry. */
export const EUCHRE_RENDERER: RendererEntry = {
  name: 'euchre',
  render: EuchreRenderer,
  keepsState: true,
  drawsStatus: true,
  viewSections: VIEW_SECTIONS,
};
