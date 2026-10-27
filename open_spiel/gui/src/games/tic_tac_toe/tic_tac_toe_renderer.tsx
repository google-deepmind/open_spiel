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

// The tic_tac_toe renderer: the game's Gameplay panel (tic_tac_toe_gameplay_panel.tsx),
// reading the frame as its session (../shared/board_session.ts).

import type {RendererEntry} from '../../renderers/renderer.ts';
import {boardRendererEntry} from '../shared/board_renderer.tsx';
import {useBoardSession} from '../shared/board_session.ts';
import {createTicTacToeGameplayPanel} from './tic_tac_toe_gameplay_panel.tsx';

/** The tic_tac_toe renderer's registry entry. */
export const TIC_TAC_TOE_RENDERER: RendererEntry = boardRendererEntry(
  'tic_tac_toe',
  createTicTacToeGameplayPanel(useBoardSession),
);
