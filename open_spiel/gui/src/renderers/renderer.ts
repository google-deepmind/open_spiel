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

// The renderer registry. A game's server adapter names the renderer that
// draws its presentation; the board looks the name up here and falls back to
// the game's text when no renderer of that name is registered.

import type {JSX} from 'react';

import type {Frame, Presentation} from '../contract/contract.ts';
import type {HistoryRenderer} from '../actions/history_renderer.ts';
import type {SectionDefinition} from '../dock/panel_groups.ts';

/** What a renderer is given to draw one frame. */
export interface RendererProps {
  /** The frame, as the server built it for the viewer. */
  readonly frame: Frame;
  /** The frame's presentation: the adapter's view and action hints. */
  readonly presentation: Presentation;
  /**
   * Plays an action id at the position shown; null when the viewer cannot
   * play there (not its turn, a replay, a spectator, a finished game).
   */
  readonly play: ((action: number) => void) | null;
  /**
   * Plays several actions in order as one move, resolving with the frame
   * after the last; null or absent whenever `play` is null.
   */
  readonly playSequence?:
    ((actions: readonly number[]) => Promise<Frame>) | null;
  /** Whether a request is in flight, during which a press is ignored. */
  readonly busy?: boolean;
}

/** A component that draws a presentation. */
export type Renderer = (props: RendererProps) => JSX.Element;

/** How a renderer sits in the Gameplay panel. */
export interface RendererOptions {
  /**
   * The renderer keeps its component state from one position to the next
   * (a hand's order, a move being composed, cards in flight), so it is not
   * drawn afresh at each ply.
   */
  readonly keepsState?: boolean;
  /**
   * The renderer says whose turn it is and what the turn asks on its own
   * table, so the panel shows no status line above it.
   */
  readonly drawsStatus?: boolean;
  /** The board's own settings, shown first in the View panel. */
  readonly viewSections?: readonly SectionDefinition[];
  /** The board's own list in Actions › History, in place of the rows. */
  readonly History?: HistoryRenderer;
}

/** A renderer, and the name adapters give it. */
export interface RendererEntry extends RendererOptions {
  readonly name: string;
  readonly render: Renderer;
}

/** A validated set of renderers, looked up by name. */
export interface Renderers {
  /** The renderer named `name`, or null when there is none. */
  get(name: string): Renderer | null;
  /** The options of the renderer named `name`; none when there is none. */
  options(name: string): RendererOptions;
  /** Every name, in definition order. */
  readonly names: readonly string[];
}

/** Validates `entries` and returns them by name; throws on a repeated name. */
export function defineRenderers(entries: readonly RendererEntry[]): Renderers {
  const byName = new Map<string, RendererEntry>();
  for (const entry of entries) {
    const {name} = entry;
    if (name.trim() === '') throw new Error('A renderer has a blank name.');
    if (byName.has(name)) {
      throw new Error(`Renderer "${name}" is defined twice.`);
    }
    byName.set(name, entry);
  }
  return {
    get: name => byName.get(name)?.render ?? null,
    options: name => byName.get(name) ?? {},
    names: [...byName.keys()],
  };
}
