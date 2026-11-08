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

import type {SessionPayload} from './board_session.ts';

/** The returns as the panels print them, or a dash before the end. */
export function formatReturns(payload: SessionPayload): string {
  if (!payload.state.returns) {
    return '—';
  }
  return payload.state.returns.map(value => value.toFixed(2)).join(', ');
}

/**
 * Whether `controller` is a person; with no controller, seat 0 is taken to
 * be the person.
 */
export function isHumanController(
  controller?: string | null,
  slot?: number,
): boolean {
  if (!controller) {
    return slot === 0;
  }
  return controller === 'human' || controller === 'manual';
}
