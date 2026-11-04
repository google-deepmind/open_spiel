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

// The internal runtime seam: what dnd/, motion/ and a11y/ see of a
// mounted CardScene. Pinned; not exported from index.ts.
import {createContext} from 'react';
import type {CardId, Point} from '../core/identity.ts';
import type {EventResult, TableEvent} from '../core/events.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import type {Scene} from '../core/scene.ts';
import type {Target} from '../core/targets.ts';
import type {CardStyle} from '../style.ts';
import type {Ticker} from '../motion/ticker.ts';
import type {MotionService} from '../motion/service.ts';

/** A table mounted in a scene: its id, scene, root and policy. */
export interface MountedTable {
  tableId: string;
  scene: Scene;
  root: HTMLElement;
  policy?: InteractionPolicy | undefined;
}
/** Who sets a library tag on a card. */
export type LibraryTagOwner = 'dnd' | 'focus' | 'hover' | 'motion';
/** The runtime a scene's tables share: tables, elements and tags. */
export interface SceneRuntime {
  tables(): readonly MountedTable[];
  /** Throws if unknown. */
  table(tableId: string): MountedTable;
  /** Affine. */
  clientToTable(tableId: string, p: Point): Point;
  tableToClient(tableId: string, p: Point): Point;
  /** hitTest on the computed scene. */
  hit(client: Point): {tableId: string; target: Target} | null;
  /** Registry (pinned nodes included). */
  element(id: CardId): HTMLElement | null;
  /** Transient library tags; owner-scoped so dnd, focus and hover never clobber
   *  each other. Keys are card ids or target keys ('slot:t3', 'insert:hand:2',
   *  'region:pot'); the runtime writes them to the DOM without a React render. */
  setLibraryTags(
    owner: LibraryTagOwner,
    tags: ReadonlyMap<CardId, readonly string[]>,
  ): void;
  /** policy.onEvent + announce. */
  emit(e: TableEvent): EventResult;
  /** aria-live polite text on the scene's live region (additive, dnd: the
   *  neutral per-step announcements of a keyboard drag). */
  announce(text: string): void;
  policy(tableId: string): InteractionPolicy | undefined;
  style(): CardStyle;
  ticker(): Ticker;
  motion: MotionService;
  onCommit(
    cb: (
      prev: ReadonlyMap<string, Scene>,
      next: ReadonlyMap<string, Scene>,
    ) => void,
  ): () => void;
}
/** The scene runtime of the enclosing <CardScene>. */
export const SceneRuntimeContext = createContext<SceneRuntime | null>(null);
