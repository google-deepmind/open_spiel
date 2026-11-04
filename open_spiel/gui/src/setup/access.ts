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

import type {Frame} from '../contract/contract.ts';

/**
 * Why the viewer of `frame` may not set up the game, or null when they may.
 *
 * Who the game is shown to is a perspective, not a permission: the person at
 * the page sets the game up whichever seat they look from. Only a spectator,
 * who watches, is never offered the controls, and the server refuses them.
 */
export function setupRefusal(frame: Frame): string | null {
  return frame.viewer.kind === 'spectator'
    ? 'Spectators watch the game; they cannot change it.'
    : null;
}
