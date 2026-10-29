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

// runFlip. Owned by the motion module.
import type {Transition} from '../core/motion_plan.ts';
import type {CardStyle} from '../style.ts';
import type {SceneRuntime} from '../react/runtime.ts';
import {motionInternals} from './service.ts';

/** Called by <Table> in useLayoutEffect after each commit: animates
 *  planTransitions(prev, next) by writing `transform` to the pinned card
 *  nodes on the scene's ticker, honouring CardStyle.motion (reducedMotion
 *  'respect' -> duration 0; flip:false -> faces change without turning;
 *  staggerMs between cards). The inverted first frame is written before this
 *  returns, so the browser never paints a card at its new place first.
 *
 *  What animates: moves (and rotations, and size changes) of an id or a `was`
 *  pair; flips (faceUp or card-null-ness changed); an id that entered this
 *  Table as it left another Table of the same CardScene. Enters with no
 *  source, exits, freed cards, suppressed ids (motion.suppressNext), tables
 *  with no layout box and whole-table rescales (card width changed) snap.
 *  A card already in flight re-targets from where it is. Resolves when every
 *  animation started here has finished; motion.settled() covers all of them. */
export function runFlip(
  rt: SceneRuntime,
  plan: readonly Transition[],
  style: CardStyle['motion'],
): Promise<void> {
  return motionInternals(rt.motion).runPlan(plan, style);
}
