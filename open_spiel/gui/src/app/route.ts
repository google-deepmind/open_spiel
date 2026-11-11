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

import type {Viewer} from '../contract/contract.ts';
import {viewerParameter} from '../session/client.ts';

/** The page a URL shows. */
export type Route =
  | {readonly page: 'catalogue'}
  | {
      readonly page: 'game';
      readonly game: string;
      /** The session to open; a new one is started when null. */
      readonly session: string | null;
      /** The viewer the URL names, or null for the game's default. */
      readonly viewer: Viewer | null;
      /** The seat analysis is drawn from (`viewSeat`), or null for none. */
      readonly viewSeat: number | null;
    }
  | {readonly page: 'missing'; readonly path: string};

const GAME_PATH = /^\/game\/([^/]+)\/?$/;

/**
 * The page `url` shows: the catalogue at `/`, a game at `/game/<name>`, with
 * the session and viewer it names in its `session` and `viewer` parameters,
 * and the seat analysis is drawn from in `viewSeat`.
 */
export function routeOf(url: URL): Route {
  const {pathname, searchParams} = url;
  if (pathname === '/' || pathname === '') return {page: 'catalogue'};
  const match = GAME_PATH.exec(pathname);
  const name = match?.[1];
  const game = name === undefined ? null : decodedName(name);
  if (game === null) return {page: 'missing', path: pathname};
  return {
    page: 'game',
    game,
    session: searchParams.get('session'),
    viewer: addressedViewer(searchParams.get('viewer')),
    viewSeat: seatOf(searchParams.get('viewSeat')),
  };
}

// A game's name from its path segment, or null when the segment is not a
// valid percent-encoding, such as `%`.
function decodedName(segment: string): string | null {
  try {
    return decodeURIComponent(segment);
  } catch (e) {
    if (e instanceof URIError) return null;
    throw e;
  }
}

/**
 * The viewer a page's `viewer` parameter names: null when it is absent (the
 * game's default, analysis), and the spectator when it names no viewer, such
 * as `viewer=-1` or `viewer=seat`. An address that asks for a viewer is never
 * shown the oracle's every hand by mistake.
 */
function addressedViewer(parameter: string | null): Viewer | null {
  if (parameter === null) return null;
  return viewerOf(parameter) ?? {kind: 'spectator', seat: null};
}

/** The viewer a `viewer` parameter names; null when absent or unknown. */
export function viewerOf(parameter: string | null): Viewer | null {
  if (parameter === 'oracle' || parameter === 'spectator') {
    return {kind: parameter, seat: null};
  }
  const seat = seatOf(parameter);
  return seat === null ? null : {kind: 'seat', seat};
}

/** The seat a parameter names; null when absent or not a seat index. */
export function seatOf(parameter: string | null): number | null {
  if (parameter !== null && /^(0|[1-9][0-9]{0,3})$/.test(parameter)) {
    return Number(parameter);
  }
  return null;
}

/** The path of the page that plays `game`. */
export function gamePath(game: string): string {
  return `/game/${encodeURIComponent(game)}`;
}

/**
 * The URL of the page that shows `session` of `game` to `viewer`, drawn from
 * seat `viewSeat` in analysis when it is not null.
 */
export function sessionPath(
  game: string,
  session: string,
  viewer: Viewer,
  viewSeat: number | null = null,
): string {
  const query = new URLSearchParams({session, viewer: viewerParameter(viewer)});
  if (viewSeat !== null) query.set('viewSeat', String(viewSeat));
  return `${gamePath(game)}?${query.toString()}`;
}
