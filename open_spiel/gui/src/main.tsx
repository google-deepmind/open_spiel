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

import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';

import {loadAddOns} from './addons.ts';
import {App} from './app.tsx';
import {ThemeProvider} from './theme/theme_provider.tsx';

import './styles/tokens.css';
import './styles/base.css';
import './styles/dock.css';

// The user's own add-ons, should addons/ hold any (src/addons.ts).
loadAddOns(import.meta.glob('../addons/*.{ts,tsx}', {eager: true}));

const container = document.getElementById('root');
if (container === null) {
  throw new Error('The page has no #root element to mount the app in.');
}
/** The React root the app is rendered into. */
export const root = createRoot(container);
root.render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
);
