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

import type {JSX, ReactNode} from 'react';

import type {CatalogueEntry, Viewer} from '../contract/contract.ts';
import {GameContext} from '../game/game_context.ts';
import {frame, session} from './frames.ts';

const ENTRY: CatalogueEntry = {
  name: 'game',
  long_name: 'Game',
  dynamics: 'sequential',
  chance_mode: 'explicit_stochastic',
  information: 'imperfect_information',
  min_seats: 2,
  max_seats: 2,
  num_players: 2,
  parameters: [],
  loadable: true,
  unavailable_reason: null,
  controllers: ['human'],
  renderer: null,
};

/**
 * A game page's context around `children`, shown to `viewer`: the oracle
 * in analysis, a seat or a spectator in a game.
 */
export function InGame({
  viewer,
  children,
}: {
  readonly viewer: Viewer;
  readonly children: ReactNode;
}): JSX.Element {
  return (
    <GameContext
      value={{entry: ENTRY, session: session(frame({viewer})), viewer}}
    >
      {children}
    </GameContext>
  );
}
