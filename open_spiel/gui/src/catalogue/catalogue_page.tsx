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

import {type JSX, useId, useState} from 'react';

import {gamePath} from '../app/route.ts';
import type {Catalogue, CatalogueEntry} from '../contract/contract.ts';
import {Segmented, type SegmentedOption} from '../controls/segmented.tsx';
import {
  type GameView,
  type ViewFilter,
  filterGames,
  gameFacts,
  gameView,
} from './games.ts';

import '../form/form.css';
import './catalogue.css';

/** Each game view's badge, and its filter's name. */
const VIEW_LABEL: Record<GameView, string> = {
  board: 'Board UI',
  text: 'Text only',
  unavailable: 'Unavailable',
};

/** The views in the order the filter offers them. */
const VIEWS: readonly GameView[] = ['board', 'text', 'unavailable'];

/** What a badge says about the game, for a reader that cannot see it. */
const VIEW_TITLE: Record<GameView, string> = {
  board: 'Played on a drawn board',
  text: 'Played from its text; no board is drawn for it yet',
  unavailable: 'The server cannot serve this game',
};

/**
 * The catalogue: every game the server knows, found by name, and narrowed
 * to the games drawn as a board, played from their text, or unavailable.
 * Each game says which at a glance, its badge in a column of its own. A game the server can serve links to its page; any
 * other says why it cannot.
 */
export function CataloguePage({
  catalogue,
}: {
  readonly catalogue: Catalogue;
}): JSX.Element {
  const searchId = useId();
  const [query, setQuery] = useState('');
  const [view, setView] = useState<ViewFilter>('all');
  const {games} = catalogue;
  const shown = filterGames(games, query, view);
  const count = (wanted: GameView): number =>
    games.filter(entry => gameView(entry) === wanted).length;
  const views: ReadonlyArray<SegmentedOption<ViewFilter>> = [
    {value: 'all', label: 'All'},
    ...VIEWS.map(wanted => ({
      value: wanted,
      label: `${VIEW_LABEL[wanted]} (${String(count(wanted))})`,
    })),
  ];
  return (
    <div className="catalogue">
      <div className="catalogue__search">
        <label htmlFor={searchId}>Find a game</label>
        <input
          id={searchId}
          className="control"
          type="search"
          value={query}
          onChange={event => {
            setQuery(event.currentTarget.value);
          }}
        />
        <Segmented
          label="Show"
          value={view}
          options={views}
          onChange={setView}
        />
        <p className="catalogue__count" role="status">
          {`${String(shown.length)} of ${String(games.length)} games`}
        </p>
      </div>
      <ul className="catalogue__games" aria-label="Games">
        {shown.map(entry => (
          <GameRow key={entry.name} entry={entry} />
        ))}
      </ul>
    </div>
  );
}

function GameRow({entry}: {readonly entry: CatalogueEntry}): JSX.Element {
  const view = gameView(entry);
  return (
    <li className="catalogue__game">
      <span
        className={`catalogue__badge catalogue__badge--${view}`}
        data-view={view}
        title={VIEW_TITLE[view]}
      >
        {VIEW_LABEL[view]}
      </span>
      <div className="catalogue__about">
        {entry.loadable ? (
          <a className="catalogue__title" href={gamePath(entry.name)}>
            {entry.long_name}
          </a>
        ) : (
          <span className="catalogue__title">{entry.long_name}</span>
        )}
        <code className="catalogue__name">{entry.name}</code>
        <span className="catalogue__facts">{gameFacts(entry)}</span>
        {entry.unavailable_reason !== null && (
          <p className="catalogue__reason">
            {`Cannot be played here: ${entry.unavailable_reason}`}
          </p>
        )}
      </div>
    </li>
  );
}
