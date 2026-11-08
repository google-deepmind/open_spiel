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

import {type JSX, useEffect, useLayoutEffect, useRef, useState} from 'react';

import {sessionPath} from '../app/route.ts';
import type {CatalogueEntry, Viewer} from '../contract/contract.ts';
import {Dock} from '../dock/dock.tsx';
import type {SessionClient} from '../session/client.ts';
import type {SessionCommands} from '../session/session_store.ts';
import {usePreparingPoll} from '../session/use_preparing_poll.ts';
import {useSession} from '../session/use_session.ts';
import {sessionModeOf, useAnalysisViewSeat} from './bottom_seat.ts';
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
   * Who the game is shown to: the oracle (analysis, the default when null),
   * or a seat or a spectator (a game). A seat the game does not have is
   * shown the spectator's view.
   */
  readonly viewer: Viewer | null;
  /** The seat analysis opens drawn from; null for the first. */
  readonly viewSeat?: number | null;
}

/**
 * The page that plays one game: its panels, docked, sharing one session.
 * It opens in analysis: the oracle's
 * frame, drawn from the seat chosen in View › Perspective. An address that
 * names a seat or a spectator opens a game instead, shown only what that
 * viewer may see. The address keeps the session, the viewer and the seat
 * analysis is drawn from, so that reloading the page shows the same game the
 * same way.
 */
export function GamePage(props: GamePageProps): JSX.Element {
  const {entry, client} = props;
  const [viewer] = useState<Viewer>(() => initialViewer(entry, props.viewer));
  const session = useSession(client, viewer);
  useStartSession(session.commands, entry, props.session);
  usePreparingPoll(session);
  const viewSeat = useOpeningViewSeat(entry, viewer, props.viewSeat ?? null);
  const id = session.frame?.session.id ?? null;
  useEffect(() => {
    if (id !== null) {
      window.history.replaceState(
        null,
        '',
        sessionPath(entry.name, id, viewer, viewSeat),
      );
    }
  }, [entry.name, id, viewer, viewSeat]);

  const game: Game = {entry, session, viewer};
  return (
    <GameContext value={game}>
      <div className="game-page">
        <Dock
          panels={GAME_PANELS}
          components={PANEL_COMPONENTS}
          groups={PANEL_GROUPS}
          game={entry.name}
          frame={PanelFrame}
        />
      </div>
    </GameContext>
  );
}

/**
 * The seat analysis is drawn from, starting at `asked` (the address's
 * `viewSeat`) when the game has it. Every page starts afresh: a seat chosen
 * on another page never carries over, and none is kept outside analysis.
 */
function useOpeningViewSeat(
  entry: CatalogueEntry,
  viewer: Viewer,
  asked: number | null,
): number | null {
  const [seat, setSeat] = useAnalysisViewSeat();
  const analysis = sessionModeOf(viewer) === 'analysis';
  const opening =
    analysis && asked !== null && asked < (entry.num_players ?? 0)
      ? asked
      : null;
  useLayoutEffect(() => {
    setSeat(opening);
  }, [opening, setSeat]);
  return analysis ? seat : null;
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
