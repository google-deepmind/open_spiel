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

import {useEffect, useState, useSyncExternalStore} from 'react';

import type {Viewer} from '../contract/contract.ts';
import {type SessionClient, sameViewer} from './client.ts';
import {
  type SessionCommands,
  type SessionSnapshot,
  SessionStore,
} from './session_store.ts';

/** A session as a component sees it. */
export interface Session extends SessionSnapshot {
  /** Whether the frame shown is the live position rather than a replay. */
  readonly live: boolean;
  readonly commands: SessionCommands;
}

/**
 * Holds one session for `viewer`, who is sent with every request. When the
 * viewer changes, the shown position is fetched again for them, and no frame
 * built for another viewer is returned meanwhile.
 *
 * @throws Error when `client` differs from the first render's: the session
 *     belongs to that client, so a new client needs a new component.
 */
export function useSession(client: SessionClient, viewer: Viewer): Session {
  const [store] = useState(() => new SessionStore(client, viewer));
  if (store.client !== client) {
    throw new Error('useSession keeps the client of its first render.');
  }
  useEffect(() => {
    store.setViewer(viewer);
  }, [store, viewer]);
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot);
  // Until the effect passes a new viewer on, the store holds the old one's
  // frame.
  const frame =
    snapshot.frame !== null && sameViewer(snapshot.frame.viewer, viewer)
      ? snapshot.frame
      : null;
  const live = frame !== null && frame.ply === frame.num_plies;
  return {...snapshot, frame, live, commands: store};
}
