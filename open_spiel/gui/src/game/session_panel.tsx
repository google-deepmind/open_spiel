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

import type {JSX} from 'react';

import {NoFrame} from '../actions/note.tsx';
import type {Frame} from '../contract/contract.ts';
import {CopyableText} from '../controls/copy_button.tsx';
import type {PanelGroupDefinition} from '../dock/panel_groups.ts';
import {useGame} from './game_context.ts';
import {viewerLabel} from './game_defaults.ts';

import './game.css';

const STATUS: Record<Frame['status'], string> = {
  playing: 'In play',
  paused: 'Paused: a seat’s controller failed',
  terminal: 'Over',
};

const CHANCE: Record<Frame['session']['chance_policy'], string> = {
  auto: 'Sampled by the server',
  expose: 'Chosen in the browser',
};

/** Session › Details: the session's fixed properties and its status. */
function SessionDetails(): JSX.Element {
  const {entry, session} = useGame();
  const {frame} = session;
  if (frame === null) return <NoFrame />;
  const info = frame.session;
  const rows: Array<readonly [term: string, value: string]> = [
    ['Game', `${entry.long_name} (${info.game})`],
    ['Session', info.id],
    ['Viewer', viewerLabel(frame.viewer)],
    ['Status', STATUS[frame.status]],
    ['Seats', String(info.num_seats)],
    ['Chance', CHANCE[info.chance_policy]],
    [
      'Seed',
      info.seed === null ? 'Shown to the oracle only' : String(info.seed),
    ],
    ['Parameters', JSON.stringify(info.parameters)],
  ];
  return (
    <dl className="session-panel__details">
      {rows.map(([term, value]) => (
        <div key={term}>
          <dt>{term}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Session › Seats: each seat's controller and its settings. */
function SessionSeats(): JSX.Element | null {
  const {frame} = useGame().session;
  if (frame === null || frame.seats.length === 0) return null;
  return (
    <ul className="session-panel__seats" aria-label="Seat controllers">
      {frame.seats.map(seat => {
        const config =
          Object.keys(seat.config).length === 0
            ? '—'
            : JSON.stringify(seat.config);
        return (
          <li key={seat.seat}>
            <strong>{`Player ${String(seat.seat)}:`}</strong> {seat.controller}
            {seat.autoplay ? ', autoplay' : ''} <em>{`(config: ${config})`}</em>
            {seat.failure !== null && (
              <span className="session-panel__failure">
                {` Failed: ${seat.failure.error}${
                  seat.failure.message === null
                    ? ''
                    : `: ${seat.failure.message}`
                }`}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/** Session › JSON: the frame shown, as the JSON the server sent, to copy. */
function SessionJson(): JSX.Element | null {
  const {frame} = useGame().session;
  if (frame === null) return null;
  return (
    <CopyableText text={JSON.stringify(frame, null, 2)} label="Frame JSON" />
  );
}

/**
 * The Session panel: the session's fixed properties, its seats, and the
 * frame shown as JSON, a tab each.
 */
export const SESSION_GROUP: PanelGroupDefinition = {
  id: 'session',
  title: 'Session',
  sections: [
    {id: 'details', title: 'Details', Content: framedBody(SessionDetails)},
    {id: 'seats', title: 'Seats', Content: framedBody(SessionSeats)},
    {id: 'json', title: 'JSON', Content: framedBody(SessionJson)},
  ],
};

// `Content` in the group's inset; nothing at all when it draws nothing, so
// that its tab is hidden.
function framedBody(
  Content: () => JSX.Element | null,
): () => JSX.Element | null {
  const Body = (): JSX.Element | null => {
    const content = Content();
    return content === null ? null : (
      <div className="panel-group__body">{content}</div>
    );
  };
  Body.displayName = Content.name;
  return Body;
}
