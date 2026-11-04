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
import {createContext, type JSX, useContext, useState} from 'react';

import {NoFrame} from '../actions/note.tsx';
import {PanelGroup} from '../dock/panel_group.tsx';
import type {PanelGroupDefinition} from '../dock/panel_groups.ts';
import {useGame} from '../game/game_context.ts';
import {PlayersPanel} from '../players/players_panel.tsx';
import {setupRefusal} from './access.ts';
import {GameOptions} from './game_options.tsx';
import {setupParameters} from './parameters.ts';
import {Position} from './position.tsx';

import '../dock/panel_group.css';
import './setup.css';

/**
 * Told how far the Players seat list overflows its scroll while Players is
 * the open section, and null otherwise; see `PlayersPanel`. Provided by the
 * Setup dock panel, so that its dock group grows to fit the seats.
 */
export const SeatsOverflowContext = createContext<
  ((pixels: number | null) => void) | undefined
>(undefined);

// Setup › Game options; nothing for a game that offers no parameters.
function GameOptionsContent(): JSX.Element | null {
  const {entry, session} = useGame();
  const {frame, busy, error, commands} = session;
  if (setupParameters(entry).length === 0) return null;
  if (frame === null) return <NoFrame />;
  return (
    <div className="setup">
      <GameOptions
        entry={entry}
        frame={frame}
        busy={busy}
        error={error}
        onCreate={commands.create}
      />
    </div>
  );
}

// Setup › Position; nothing for a spectator, who cannot change the game.
function PositionContent(): JSX.Element | null {
  const {frame, busy, error, commands} = useGame().session;
  if (frame === null) return <NoFrame />;
  if (setupRefusal(frame) !== null) return null;
  return (
    <div className="setup">
      <Position
        frame={frame}
        busy={busy}
        error={error}
        onReset={commands.reset}
      />
    </div>
  );
}

// Setup › Players: every seat and its controller.
function PlayersContent(): JSX.Element {
  const {frame, busy, commands} = useGame().session;
  const onSeatsOverflow = useContext(SeatsOverflowContext);
  if (frame === null) return <NoFrame />;
  return (
    <PlayersPanel
      frame={frame}
      busy={busy}
      commands={commands}
      onSeatsOverflow={onSeatsOverflow}
    />
  );
}

/**
 * The Setup panel's sections: Game options, Position and Players, an inner
 * tab each. Game options draws nothing for a game that offers no
 * parameters, and Position nothing for a spectator, so neither has a tab
 * then.
 */
export const SETUP_GROUP: PanelGroupDefinition = {
  id: 'setup',
  title: 'Setup',
  sections: [
    {id: 'options', title: 'Game options', Content: GameOptionsContent},
    {id: 'position', title: 'Position', Content: PositionContent},
    {id: 'players', title: 'Players', Content: PlayersContent},
  ],
};

/**
 * The Setup panel: its sections once a session is open; until then, why
 * none is.
 */
export function SetupPanel(): JSX.Element {
  const {frame, error} = useGame().session;
  // A session, once open, stays open: a new one replaces it.
  const [opened, setOpened] = useState(false);
  // Kept here, so that outside a dock the open section survives a change of
  // viewer, while which the frame is fetched again and the group is not
  // drawn. Inside one the dock remembers it.
  const [chosen, setChosen] = useState<string | undefined>(undefined);
  if (frame !== null && !opened) setOpened(true);
  if (frame === null) {
    return (
      <p className="setup__notice">
        {!opened
          ? 'No game is open.'
          : error === null
            ? 'Loading the position.'
            : `The position could not be loaded. ${error.message}`}
      </p>
    );
  }
  return (
    <PanelGroup group={SETUP_GROUP} chosen={chosen} onChoose={setChosen} />
  );
}
