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

// What the dock needs from the theme. The theme's stylesheets define the
// dockview theme class and every token below, in light and dark.

import type {DockviewTheme} from 'dockview-core';

/** The dockview theme the dock is drawn with. */
export const DOCK_THEME: DockviewTheme = {
  name: 'open-spiel',
  className: 'dockview-theme-open-spiel',
};

/** The theme tokens dock.css reads, and the only ones. */
export const DOCK_TOKENS = [
  '--color-accent',
  '--color-border',
  '--color-border-strong',
  '--color-danger',
  '--color-none',
  '--color-surface',
  '--color-surface-raised',
  '--color-surface-sunken',
  '--color-text',
  '--color-text-secondary',
  '--color-warning',
  '--radius',
] as const;
