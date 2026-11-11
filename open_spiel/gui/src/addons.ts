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

// Client add-ons: code the user puts in the client's addons/ directory
// (which git ignores), one module per add-on, to change what the client
// draws through its public extension points without editing the client.
// Each module exports `registerInto(api)`, which the entry point calls once,
// in path order, before the app is drawn. src/INTERFACES.md, "Client
// add-ons" and "Replacing the chess board", is the guide.

import {getDeck, registerDeck} from './cards/library/registry.ts';
import {chessRenderer} from './games/chess/chess_renderer.tsx';
import {registerPieceSet} from './games/chess/pieces/piece_sets.ts';
import {replaceRenderer} from './renderers/registry.ts';

/** The extension points an add-on's registerInto is given. */
export interface AddOnApi {
  /** Draws a renderer's presentations with another component. */
  readonly replaceRenderer: typeof replaceRenderer;
  /** The "chess" renderer drawing with another board. */
  readonly chessRenderer: typeof chessRenderer;
  /** Registers a chess piece set, offered in View › Board › Pieces. */
  readonly registerPieceSet: typeof registerPieceSet;
  /** Registers a card deck, offered as a Card deck in the card games' View. */
  readonly registerDeck: typeof registerDeck;
  /** A registered card deck, to build a new deck or back from. */
  readonly getDeck: typeof getDeck;
}

/** The extension points the client gives its add-ons. */
export const ADD_ON_API: AddOnApi = {
  replaceRenderer,
  chessRenderer,
  registerPieceSet,
  registerDeck,
  getDeck,
};

/**
 * Calls each add-on module's `registerInto` with `api`, in path order.
 * `modules` maps each module's path to the module.
 *
 * @throws Error naming the module when one has no registerInto function,
 *     or when its registerInto throws.
 */
export function loadAddOns(
  modules: Readonly<Record<string, unknown>>,
  api: AddOnApi = ADD_ON_API,
): void {
  for (const path of Object.keys(modules).sort()) {
    const module = modules[path];
    const registerInto: unknown =
      typeof module === 'object' && module !== null
        ? Reflect.get(module, 'registerInto')
        : undefined;
    if (typeof registerInto !== 'function') {
      throw new Error(`Add-on ${path} exports no registerInto function.`);
    }
    try {
      Reflect.apply(registerInto, undefined, [api]);
    } catch (error: unknown) {
      throw new Error(`Add-on ${path} failed to register.`, {cause: error});
    }
  }
}
