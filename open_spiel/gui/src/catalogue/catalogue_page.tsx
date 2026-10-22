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
import {filterGames, gameFacts} from './games.ts';

import '../form/form.css';
import './catalogue.css';

/**
 * The catalogue: every game the server knows, found by name. A game the
 * server can serve links to its page; any other says why it cannot.
 */
export function CataloguePage({
  catalogue,
}: {
  readonly catalogue: Catalogue;
}): JSX.Element {
  const searchId = useId();
  const [query, setQuery] = useState('');
  const {games} = catalogue;
  const shown = filterGames(games, query);
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
  return (
    <li className="catalogue__game">
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
    </li>
  );
}
