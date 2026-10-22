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

import {type JSX, type ReactNode, useState} from 'react';

import {routeOf} from './app/route.ts';
import {type CatalogueState, useCatalogue} from './app/use_catalogue.ts';
import {CataloguePage} from './catalogue/catalogue_page.tsx';
import {SettingsMenu} from './controls/settings_menu.tsx';
import {HeaderActionsContext} from './dock/header_actions.ts';
import {GamePage} from './game/game_page.tsx';
import {SessionClient} from './session/client.ts';

import './app.css';

/** The properties of the App. */
export interface AppProps {
  /** The page's address; the window's when omitted. */
  readonly url?: URL;
  /** The client of the server; one of the page's own origin when omitted. */
  readonly client?: SessionClient;
}

/**
 * The OpenSpiel GUI: the catalogue at `/`, and each game's page at
 * `/game/<name>`. Every link loads its page afresh.
 */
export function App(props: AppProps): JSX.Element {
  const [url] = useState(() => props.url ?? new URL(window.location.href));
  const [client] = useState(() => props.client ?? new SessionClient());
  const catalogue = useCatalogue(client);
  const route = routeOf(url);
  if (route.page === 'missing') {
    return <Page title="Page not found">{`Nothing is at ${route.path}.`}</Page>;
  }
  if (catalogue.status !== 'loaded') {
    return <Page title="Games">{catalogueProblem(catalogue)}</Page>;
  }
  if (route.page === 'catalogue') {
    return (
      <Page title="Games">
        <CataloguePage catalogue={catalogue.catalogue} />
      </Page>
    );
  }
  const entry = catalogue.catalogue.games.find(
    game => game.name === route.game,
  );
  if (entry === undefined) {
    return (
      <Page title={route.game}>{`There is no game "${route.game}".`}</Page>
    );
  }
  if (!entry.loadable) {
    return (
      <Page title={entry.long_name}>
        {`${entry.long_name} cannot be played here: ${entry.unavailable_reason ?? 'the server cannot load it'}.`}
      </Page>
    );
  }
  return (
    <Page title={entry.long_name}>
      <GamePage
        key={entry.name}
        entry={entry}
        client={client}
        session={route.session}
        viewer={route.viewer}
      />
    </Page>
  );
}

function catalogueProblem(state: CatalogueState): string {
  return state.status === 'failed'
    ? `The games could not be loaded. ${state.error.message}`
    : 'Loading the games.';
}

/**
 * The page's frame: the header, then `children`, a sentence or a page. At
 * the header's right is an element the page fills with its own controls,
 * then the settings.
 */
function Page({
  title,
  children,
}: {
  readonly title: string;
  readonly children: ReactNode;
}): JSX.Element {
  const [actions, setActions] = useState<HTMLElement | null>(null);
  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <a className="app__home" href="/">
            OpenSpiel
          </a>
          <span className="app__separator" aria-hidden="true">
            •
          </span>
          <h1 className="app__title">{title}</h1>
        </div>
        <div className="app__actions">
          <div className="app__page-actions" ref={setActions} />
          <SettingsMenu />
        </div>
      </header>
      <main className="app__main">
        <HeaderActionsContext value={actions}>
          {typeof children === 'string' ? (
            <p className="app__message">{children}</p>
          ) : (
            children
          )}
        </HeaderActionsContext>
      </main>
    </div>
  );
}
