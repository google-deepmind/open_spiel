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

// Every renderer the client has, by the name a server adapter gives it. A
// game's renderer is added here, one line each, in name order; a game whose
// adapter uses a shared renderer ("grid", "cards") adds nothing.

import {CardTableRenderer} from '../cards/card_table_renderer.tsx';
import {CHECKERS_RENDERER} from '../games/checkers/checkers_renderer.tsx';
import {CHESS_RENDERER} from '../games/chess/chess_renderer.tsx';
import {CONNECT_FOUR_RENDERER} from '../games/connect_four/connect_four_renderer.tsx';
import {EUCHRE_RENDERER} from '../games/euchre/euchre_renderer.tsx';
import {GIN_RUMMY_RENDERER} from '../games/gin_rummy/gin_rummy_renderer.tsx';
import {HEARTS_RENDERER} from '../games/hearts/hearts_renderer.tsx';
import {POKER_RENDERER} from '../games/poker/poker_renderer.tsx';
import {TIC_TAC_TOE_RENDERER} from '../games/tic_tac_toe/tic_tac_toe_renderer.tsx';
import {GridRenderer} from '../grid/grid_renderer.tsx';
import type {SectionDefinition} from '../dock/panel_groups.ts';
import {
  type Renderer,
  type RendererEntry,
  type Renderers,
  defineRenderers,
} from './renderer.ts';

const ENTRIES: readonly RendererEntry[] = [
  {name: 'cards', render: CardTableRenderer},
  CHECKERS_RENDERER,
  CHESS_RENDERER,
  CONNECT_FOUR_RENDERER,
  EUCHRE_RENDERER,
  GIN_RUMMY_RENDERER,
  {name: 'grid', render: GridRenderer},
  HEARTS_RENDERER,
  POKER_RENDERER,
  TIC_TAC_TOE_RENDERER,
];

// Each name's drawing component; an add-on may replace one before the app
// is drawn (replaceRenderer).
const renderers = new Map<string, Renderer>(
  ENTRIES.map(entry => [entry.name, entry.render]),
);
const defined = defineRenderers(ENTRIES);

/** The registered renderers. */
export const RENDERERS: Renderers = {
  /** The component that draws `name`, as an add-on may have replaced it. */
  get: name => renderers.get(name) ?? null,
  /** The options of the entry named `name`. */
  options: name => defined.options(name),
  names: defined.names,
};

/**
 * Draws the presentations of renderer `name` with `render` in place of the
 * built-in component, keeping the entry's options (its View sections, its
 * History list, `keepsState` and `drawsStatus`). An add-on calls this from
 * its `registerInto`, before the app is drawn (src/addons.ts).
 *
 * @returns The component it replaced, so a test can put it back.
 * @throws Error when no renderer is named `name`.
 */
export function replaceRenderer(name: string, render: Renderer): Renderer {
  const previous = renderers.get(name);
  if (previous === undefined) {
    throw new Error(
      `There is no renderer "${name}" to replace; the renderers are ${defined.names.join(', ')}.`,
    );
  }
  renderers.set(name, render);
  return previous;
}

/** Each board's own View sections, with the renderer they belong to. */
export const BOARD_VIEW_SECTIONS: ReadonlyArray<{
  readonly renderer: string;
  readonly section: SectionDefinition;
}> = ENTRIES.flatMap(entry =>
  (entry.viewSections ?? []).map(section => ({
    renderer: entry.name,
    section,
  })),
);
