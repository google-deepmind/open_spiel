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

import {useEffect, useState} from 'react';

import type {Catalogue} from '../contract/contract.ts';
import type {SessionClient} from '../session/client.ts';
import {
  ContractError,
  type RequestError,
  TransportError,
} from '../session/errors.ts';

/** The catalogue as a page sees it: loading, loaded, or why it failed. */
export type CatalogueState =
  | {readonly status: 'loading'}
  | {readonly status: 'loaded'; readonly catalogue: Catalogue}
  | {readonly status: 'failed'; readonly error: RequestError};

/**
 * Fetches the catalogue from `client` once. A refusal or a failed transport
 * is the state's error; anything else is a defect, thrown from the render.
 */
export function useCatalogue(client: SessionClient): CatalogueState {
  const [state, setState] = useState<CatalogueState>({status: 'loading'});
  useEffect(() => {
    let current = true;
    client.catalogue().then(
      catalogue => {
        if (current) setState({status: 'loaded', catalogue});
      },
      (e: unknown) => {
        if (!current) return;
        if (e instanceof ContractError || e instanceof TransportError) {
          setState({status: 'failed', error: e});
        } else {
          setState(() => {
            throw e;
          });
        }
      },
    );
    return () => {
      current = false;
    };
  }, [client]);
  return state;
}
