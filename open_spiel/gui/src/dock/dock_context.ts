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

import {createContext, useContext} from 'react';

/** What a panel may know about the dock it is in. */
export interface DockState {
  /**
   * The one panel that shows the session's error: the first panel, in
   * definition order, whose content is on screen, so the primary panel
   * whenever it is. Undefined when no panel is on screen.
   */
  readonly sessionErrorPanel: string | undefined;
}

/** The dock a panel is rendered in; provided by `Dock`. */
export const DockContext = createContext<DockState | null>(null);

/** Whether panel `id` is the one that shows the session's error. */
export function useShowsSessionError(id: string): boolean {
  const dock = useContext(DockContext);
  if (dock === null) {
    throw new Error('useShowsSessionError is called outside a Dock.');
  }
  return id === dock.sessionErrorPanel;
}
