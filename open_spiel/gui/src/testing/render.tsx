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

import type {ReactNode} from 'react';
import {act} from 'react';
import {createRoot} from 'react-dom/client';

declare global {
  // React reads this to know that updates are wrapped in act().
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

/** A tree rendered into the document by render(). */
export interface Rendered {
  container: HTMLElement;
  unmount: () => void;
}

/** Renders `tree` into a fresh element of the document, inside act(). */
export function render(tree: ReactNode): Rendered {
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  act(() => {
    root.render(tree);
  });
  return {
    container,
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

/** The one element matching `selector` under `parent`; throws otherwise. */
export function only(parent: ParentNode, selector: string): HTMLElement {
  const found = parent.querySelectorAll(selector);
  const [element] = found;
  if (found.length !== 1 || !(element instanceof HTMLElement)) {
    throw new Error(`Expected one ${selector}, found ${String(found.length)}.`);
  }
  return element;
}

/** The one button under `parent` whose text is `name`; throws otherwise. */
export function button(
  name: string,
  parent: ParentNode = document,
): HTMLButtonElement {
  const found = [...parent.querySelectorAll('button')].filter(
    candidate => candidate.textContent === name,
  );
  const [element] = found;
  if (found.length !== 1 || element === undefined) {
    throw new Error(
      `Expected one "${name}" button, found ${String(found.length)}.`,
    );
  }
  return element;
}
