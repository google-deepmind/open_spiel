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

// The board's turn line, said on the scene's live region.
import {useEffect, useRef} from 'react';

import {useCardScene} from '../../../cards/library/index.ts';

/**
 * Says the turn line (the status, or the lay-down prompt) on the scene's
 * live region each time it changes. The region otherwise keeps the last
 * drop's words ("Discarding Queen of Spades") through every later move,
 * the knock and the lay-down, where a reader of the page takes it for the
 * current state; now it reads what the stats region shows in each phase.
 * Renders nothing.
 */
export function TurnLineAnnouncer({text}: {text: string}): null {
  const scene = useCardScene();
  const said = useRef<string | null>(null);
  useEffect(() => {
    if (text === said.current) return;
    said.current = text;
    if (text !== '') scene.announce(text);
  });
  return null;
}
