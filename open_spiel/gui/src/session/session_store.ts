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

import type {
  CreateSession,
  Frame,
  SeatConfig,
  Viewer,
} from '../contract/contract.ts';
import {type SessionClient, sameViewer} from './client.ts';
import {ContractError, type RequestError, TransportError} from './errors.ts';

/** What a session shows at one moment. */
export interface SessionSnapshot {
  /**
   * The position shown, always built for the current viewer; null until a
   * session is open, and from a change of viewer until its frame arrives.
   */
  readonly frame: Frame | null;
  /** Whether a request is in flight or queued. */
  readonly busy: boolean;
  /** Why the latest request failed, until the next command is issued. */
  readonly error: RequestError | null;
}

/** The commands a session accepts; each may be passed as a callback. */
export interface SessionCommands {
  /** Starts a new session, which replaces the current one. */
  readonly create: (body: CreateSession) => void;
  /** Shows the live position of an existing session. */
  readonly open: (id: string) => void;
  /** Fetches the shown position again. */
  readonly refresh: () => void;
  /**
   * Fetches the shown position again, quietly: the session is not marked
   * busy, and nothing is sent while another command is pending. For waiting
   * on the server, as while a seat's controller is preparing.
   */
  readonly poll: () => void;
  /**
   * Shows the position after `ply` actions, or the live position when `ply`
   * is past it.
   */
  readonly goTo: (ply: number) => void;
  /** Moves the shown position by `delta` plies, stopping at either end. */
  readonly step: (delta: number) => void;
  /**
   * Plays `action` at the position shown. At an earlier position this plays
   * from it, taking back every later ply, as only an omniscient viewer may.
   */
  readonly act: (action: number) => void;
  /**
   * Plays `actions` in order from the position shown, as one command: the
   * frame after the last, or a rejection with the first refusal, after which
   * none of the rest is sent. A board that composes several actions into one
   * move (a gin lay-down) waits on it, and says why it failed: a refusal is
   * the board's to show, not the snapshot's error. The actions before the
   * refusal stay played, so after one the store fetches the position the
   * server is at and shows it, and drops the commands queued behind.
   */
  readonly actSequence: (actions: readonly number[]) => Promise<Frame>;
  /** Chooses a chance outcome at the position shown, as `act` does. */
  readonly chooseChance: (outcome: number) => void;
  /** Lets controllers play; `maxPlies` null leaves the bound to the server. */
  readonly advance: (maxPlies?: number | null) => void;
  readonly undo: () => void;
  /**
   * Restarts the game and replays `history`. With a prefix of the live
   * history it takes back every later ply; with any other sequence it sets
   * up that position.
   */
  readonly reset: (history?: readonly number[]) => void;
  readonly setSeat: (seat: number, config: SeatConfig) => void;
}

// Where the session stands, as the latest answer for any viewer left it.
// Every viewer is sent these fields, so a command may build on an answer the
// store does not show.
interface Position {
  readonly id: string;
  readonly ply: number;
  readonly numPlies: number;
}

// Issues one request from the position the previous command left, as the
// viewer the command was issued under, or returns null when there is nothing
// to request.
type Command = (
  position: Position | null,
  viewer: Viewer,
) => Promise<Frame> | null;

const IDLE: SessionSnapshot = {frame: null, busy: false, error: null};

/**
 * One session, shown to one viewer at a time.
 *
 * Commands run one at a time, in the order they are issued, each from the
 * position the previous one left. Each is sent as the viewer current when it
 * was issued, so an action issued for seat 0 is sent as seat 0 even when the
 * viewer changes before it runs. A command that changes the game answers with
 * the live position.
 *
 * The snapshot never holds a frame built for another viewer: a change of
 * viewer clears the frame and fetches the position again, and an answer built
 * for another viewer moves the position without being shown.
 *
 * A failed command drops the commands queued behind it, since they were
 * issued on the expectation that it would succeed; the fetch that follows a
 * change of viewer is kept, since the frame it replaces is already cleared. A refusal or a failed
 * transport becomes the snapshot's error; any other exception is a defect,
 * rethrown uncaught. The client bounds every request, so a server that stops
 * answering fails the command rather than stalling the queue.
 *
 * The store checks the plies it navigates to, which it resolves itself; game
 * values such as an action id are the server's to check.
 */
export class SessionStore implements SessionCommands {
  /** The client every request goes through. */
  readonly client: SessionClient;
  private viewer: Viewer;
  private position: Position | null = null;
  private snapshot = IDLE;
  private readonly listeners = new Set<() => void>();
  private queue = Promise.resolve();
  private pending = 0;
  // Bumped by each failure; a command issued before it is dropped, unless
  // it is kept.
  private generation = 0;

  constructor(client: SessionClient, viewer: Viewer) {
    this.client = client;
    this.viewer = viewer;
  }

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  readonly getSnapshot = (): SessionSnapshot => this.snapshot;

  /** Shows the session to `viewer`, fetching the shown position for them. */
  setViewer(viewer: Viewer): void {
    if (sameViewer(viewer, this.viewer)) return;
    this.viewer = viewer;
    // Cleared without notice, so that listeners hear of the change once,
    // already busy when a refresh follows.
    this.snapshot = {...this.snapshot, frame: null};
    // With no position yet, a pending command may still open a session; its
    // answer is for the old viewer, so the fetch behind it brings the new
    // viewer's frame. No failure drops that fetch: when the command fails, it
    // shows the position the session is at, if there is one.
    if (this.position !== null || this.pending > 0) {
      this.enqueue(
        (position, current) =>
          position === null ? null : this.load(position, current, position.ply),
        {keep: true},
      );
    } else {
      this.update({});
    }
  }

  readonly create = (body: CreateSession): void => {
    this.enqueue((_, viewer) => this.client.create(viewer, body));
  };

  readonly open = (id: string): void => {
    this.enqueue((_, viewer) => this.client.frame(id, viewer));
  };

  readonly refresh = (): void => {
    this.enqueue(this.reload);
  };

  readonly poll = (): void => {
    if (this.pending > 0 || this.position === null) return;
    this.enqueue(this.reload, {quiet: true});
  };

  readonly goTo = (ply: number): void => {
    if (!Number.isInteger(ply) || ply < 0) {
      throw new RangeError(
        `A ply is a non-negative integer, not ${String(ply)}.`,
      );
    }
    this.show(current => Math.min(ply, current.numPlies));
  };

  readonly step = (delta: number): void => {
    if (!Number.isInteger(delta)) {
      throw new RangeError(
        `A step is a whole number of plies, not ${String(delta)}.`,
      );
    }
    this.show(current =>
      Math.min(Math.max(current.ply + delta, 0), current.numPlies),
    );
  };

  readonly act = (action: number): void => {
    this.enqueue((position, viewer) => {
      const current = requireSession(position);
      return this.client.act(current.id, viewer, action, earlierPly(current));
    });
  };

  readonly actSequence = (actions: readonly number[]): Promise<Frame> => {
    if (actions.length === 0) {
      return Promise.reject(new RangeError('An action sequence is empty.'));
    }
    return new Promise<Frame>((resolve, reject) => {
      const command: Command = async (position, viewer) => {
        let current = requireSession(position);
        let frame: Frame | null = null;
        try {
          for (const action of actions) {
            frame = await this.client.act(
              current.id,
              viewer,
              action,
              earlierPly(current),
            );
            current = {
              id: frame.session.id,
              ply: frame.ply,
              numPlies: frame.num_plies,
            };
          }
        } catch (e) {
          reject(e instanceof Error ? e : new Error(String(e)));
          if (!(e instanceof ContractError || e instanceof TransportError)) {
            throw e;
          }
          // The commands queued behind were issued on the expectation that
          // the sequence would succeed.
          this.generation += 1;
          // The actions before the refusal were played: the frame shown must
          // be the server's, or every later move is built on a stale one.
          return this.load(current, viewer, current.ply);
        }
        if (frame === null) throw new Error('An action sequence is empty.');
        resolve(frame);
        return frame;
      };
      this.enqueue(command, {
        onDrop: () => {
          reject(new Error('An earlier command failed; nothing was sent.'));
        },
      });
    });
  };

  readonly chooseChance = (outcome: number): void => {
    this.enqueue((position, viewer) => {
      const current = requireSession(position);
      return this.client.chooseChance(
        current.id,
        viewer,
        outcome,
        earlierPly(current),
      );
    });
  };

  readonly advance = (maxPlies: number | null = null): void => {
    this.enqueue((position, viewer) =>
      this.client.advance(requireSession(position).id, viewer, maxPlies),
    );
  };

  readonly undo = (): void => {
    this.enqueue((position, viewer) =>
      this.client.undo(requireSession(position).id, viewer),
    );
  };

  readonly reset = (history: readonly number[] = []): void => {
    this.enqueue((position, viewer) =>
      this.client.reset(requireSession(position).id, viewer, history),
    );
  };

  readonly setSeat = (seat: number, config: SeatConfig): void => {
    this.enqueue((position, viewer) =>
      this.client.setSeat(requireSession(position).id, viewer, seat, config),
    );
  };

  // Shows the ply `target` picks from the current position, sending nothing
  // when that position is already shown.
  private show(target: (current: Position) => number): void {
    this.enqueue((position, viewer) => {
      const current = requireSession(position);
      const ply = target(current);
      if (ply === current.ply && this.snapshot.frame !== null) return null;
      return this.load(current, viewer, ply);
    });
  }

  // Fetches the position the previous command left.
  private readonly reload: Command = (position, viewer) => {
    const current = requireSession(position);
    return this.load(current, viewer, current.ply);
  };

  // The live position comes by the live route, so that refreshing a live
  // frame picks up the plies played since.
  private load(current: Position, viewer: Viewer, ply: number): Promise<Frame> {
    return ply === current.numPlies
      ? this.client.frame(current.id, viewer)
      : this.client.replay(current.id, viewer, ply);
  }

  // Queues `command`, which a failure before it drops unless `keep`;
  // `onDrop` hears of the drop. A `quiet` command neither marks the session
  // busy nor clears its error, and counts for nothing in `pending`.
  private enqueue(
    command: Command,
    {
      keep = false,
      quiet = false,
      onDrop,
    }: {keep?: boolean; quiet?: boolean; onDrop?: () => void} = {},
  ): void {
    const viewer = this.viewer;
    const generation = keep ? null : this.generation;
    if (!quiet) {
      this.pending += 1;
      this.update({busy: true, error: null});
    }
    this.queue = this.queue.then(() =>
      this.run(command, viewer, generation, onDrop, quiet),
    );
  }

  private async run(
    command: Command,
    viewer: Viewer,
    generation: number | null,
    onDrop: (() => void) | undefined,
    quiet: boolean,
  ): Promise<void> {
    try {
      if (generation !== null && generation !== this.generation) {
        onDrop?.();
        return;
      }
      const request = command(this.position, viewer);
      if (request === null) return;
      const frame = await request;
      this.position = {
        id: frame.session.id,
        ply: frame.ply,
        numPlies: frame.num_plies,
      };
      if (sameViewer(frame.viewer, this.viewer)) this.update({frame});
    } catch (e) {
      // A quiet command was issued by no one, so nothing queued rests on it.
      if (!quiet) this.generation += 1;
      if (e instanceof ContractError || e instanceof TransportError) {
        this.update({error: e});
      } else {
        queueMicrotask(() => {
          throw e;
        });
      }
    } finally {
      if (!quiet) {
        this.pending -= 1;
        if (this.pending === 0) this.update({busy: false});
      }
    }
  }

  private update(change: Partial<SessionSnapshot>): void {
    this.snapshot = {...this.snapshot, ...change};
    for (const listener of this.listeners) listener();
  }
}

// The ply of the position shown when it is earlier than the live one, where
// a choice plays from it, taking back the later plies; null when it is live.
function earlierPly({ply, numPlies}: Position): number | null {
  return ply < numPlies ? ply : null;
}

function requireSession(position: Position | null): Position {
  if (position === null) throw new Error('No session is open.');
  return position;
}
