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

import {type JSX, useEffect, useId, useRef, useState} from 'react';

import {sessionPath} from '../app/route.ts';
import type {CatalogueEntry, Viewer} from '../contract/contract.ts';
import {Dock} from '../dock/dock.tsx';
import type {SessionClient} from '../session/client.ts';
import type {SessionCommands} from '../session/session_store.ts';
import {useSession} from '../session/use_session.ts';
import {defaultSession, initialViewer} from './game_defaults.ts';
import {type Game, GameContext} from './game_context.ts';
import {GAME_PANELS} from './game_panels.ts';
import {PANEL_COMPONENTS, PANEL_GROUPS, PanelFrame} from './panels.tsx';

import './game.css';

/** The properties of a GamePage. */
export interface GamePageProps {
  readonly entry: CatalogueEntry;
  readonly client: SessionClient;
  /** The session to open; null starts a game with the defaults. */
  readonly session: string | null;
  /**
   * Who the game is shown to first; the game's default when null or a seat
   * the game does not have.
   */
  readonly viewer: Viewer | null;
}

/**
 * The page that plays one game: its panels, docked, sharing one session.
 * The address keeps the session and the viewer, so that reloading the page
 * shows the same game to the same viewer.
 */
export function GamePage(props: GamePageProps): JSX.Element {
  const {entry, client} = props;
  const [viewer, setViewer] = useState<Viewer>(() =>
    initialViewer(entry, props.viewer),
  );
  const [screened, setScreened] = useState<number | null>(null);
  const session = useSession(client, viewer);
  useStartSession(session.commands, entry, props.session);
  const id = session.frame?.session.id ?? null;
  useEffect(() => {
    if (id !== null) {
      window.history.replaceState(
        null,
        '',
        sessionPath(entry.name, id, viewer),
      );
    }
  }, [entry.name, id, viewer]);

  const game: Game = {
    entry,
    session,
    viewer,
    setViewer,
    handOver: seat => {
      setScreened(seat);
      setViewer({kind: 'seat', seat});
    },
  };
  return (
    <GameContext value={game}>
      <div className="game-page" inert={screened !== null}>
        <Dock
          panels={GAME_PANELS}
          components={PANEL_COMPONENTS}
          groups={PANEL_GROUPS}
          game={entry.name}
          frame={PanelFrame}
        />
      </div>
      {screened !== null && (
        <HandOverScreen
          seat={screened}
          onShow={() => {
            setScreened(null);
          }}
        />
      )}
    </GameContext>
  );
}

// Opens session `id`, or starts a game of `entry` when it is null, once.
function useStartSession(
  commands: SessionCommands,
  entry: CatalogueEntry,
  id: string | null,
): void {
  const started = useRef(false);
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    if (id === null) commands.create(defaultSession(entry));
    else commands.open(id);
  }, [commands, entry, id]);
}

/**
 * Covers the page while the device passes to the person who plays `seat`,
 * so that the one handing it over does not see that seat's view.
 */
function HandOverScreen({
  seat,
  onShow,
}: {
  readonly seat: number;
  readonly onShow: () => void;
}): JSX.Element {
  const headingId = useId();
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    button.current?.focus();
  }, []);
  const name = `Player ${String(seat)}`;
  return (
    <div
      className="hand-over"
      role="dialog"
      aria-modal="true"
      aria-labelledby={headingId}
    >
      <h2 id={headingId}>{`Pass the device to ${name}`}</h2>
      <p>{`${name}’s view stays hidden until they are ready.`}</p>
      <button ref={button} type="button" className="button" onClick={onShow}>
        {`Show ${name}’s view`}
      </button>
    </div>
  );
}
