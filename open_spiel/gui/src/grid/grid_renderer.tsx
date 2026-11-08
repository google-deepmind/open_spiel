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

import {actionLabels} from '../renderers/hints.ts';
import type {RendererProps} from '../renderers/renderer.ts';
import {Lines} from '../renderers/lines.tsx';
import {field} from '../renderers/struct_reader.ts';
import {GridBoard} from './grid_board.tsx';
import {readGridMoves, readGridView} from './grid_view.ts';

/**
 * The `grid` renderer, for any adapter whose view is {"grid": ...} and,
 * optionally, "lines": text shown above the board. A game whose board is a
 * plain grid needs no renderer of its own.
 */
export function GridRenderer({
  frame,
  presentation,
  play,
}: RendererProps): JSX.Element {
  const view = readGridView(
    field(presentation.view, 'view', 'grid', value => value),
  );
  return (
    <div className="renderer">
      <Lines view={presentation.view} />
      <GridBoard
        // A new position starts with nothing selected.
        key={frame.ply}
        view={view}
        moves={readGridMoves(presentation)}
        play={play}
        labels={actionLabels(frame.legal_actions)}
      />
    </div>
  );
}
