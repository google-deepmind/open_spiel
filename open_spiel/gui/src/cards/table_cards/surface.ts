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

// The table surface as CSS variables: what
// useTableSurface returns for a game to spread on its own root element. Set
// there only, never on :root, so a game that does not adopt sees nothing.
import type {CSSProperties} from 'react';
import feltNoise from './felt_noise.svg?raw';
import type {Shading, SurfaceSettings, Texture} from './settings.ts';
import woodGrain from './wood_grain.svg?raw';

/** An SVG file's markup, without its comments, as a CSS url() layer. */
export function svgLayer(markup: string): string {
  const svg = markup.replace(/<!--[\s\S]*?-->/g, '').trim();
  if (!svg.startsWith('<svg')) {
    throw new Error(`svgLayer: not an SVG: ${svg.slice(0, 40)}`);
  }
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}

/** The pattern each texture paints over the colour (CSS background layers):
 *  no light fall-off, so it holds under either shading. The noise tiles are
 *  SVG files beside this one (felt_noise.svg, wood_grain.svg); the rest are
 *  gradients. No image files, no network. */
export const TEXTURE_LAYERS: Readonly<Record<Texture, string>> = {
  plain: 'none',
  // Fine short fibres.
  felt: svgLayer(feltNoise),
  // A woven cloth: two fine diagonal threads.
  cloth:
    'repeating-linear-gradient(45deg, var(--felt-cloth-light) 0 1px, var(--color-none) 1px 4px), ' +
    'repeating-linear-gradient(-45deg, var(--felt-cloth-dark) 0 1px, var(--color-none) 1px 4px)',
  // Long grain along the width, and plank seams.
  wood: `${svgLayer(woodGrain)}, repeating-linear-gradient(0deg, var(--felt-wood-seam) 0 1px, var(--color-none) 1px 96px)`,
};

/** A texture's own light fall-off, painted under 'soft' shading only (felt's
 *  soft light towards the edges). */
export const TEXTURE_SHADE: Readonly<Partial<Record<Texture, string>>> = {
  felt: 'radial-gradient(ellipse at 50% 40%, var(--felt-shade-light), var(--felt-shade-dark))',
};

/**
 * What each shading sets, as the variables a game reads:
 *   --table-shading  a background layer between the texture and the colour
 *                    (a game that had a gradient of its own paints it here:
 *                    Hearts' diagonal fall-off to the darker felt);
 *   --table-shadow   the table's box-shadow (an inset vignette, a drop shadow);
 *   --table-grain    the opacity of a game's own faint grain overlay.
 * 'flat' sets every one to nothing: with the plain texture, one colour.
 */
export const SHADING_VARS: Readonly<
  Record<Shading, Readonly<Record<string, string>>>
> = {
  soft: {
    '--table-shading':
      'linear-gradient(145deg, var(--color-felt) 0%, var(--color-felt-dark) 100%)',
    '--table-shadow':
      'inset 0 0 100px var(--felt-vignette), 0 4px 20px var(--felt-drop-shadow)',
    '--table-grain': '0.03',
  },
  flat: {
    '--table-shading': 'none',
    '--table-shadow': 'none',
    '--table-grain': '0',
  },
};

/** Each shading's name in the settings. */
export const SHADING_LABELS: Readonly<Record<Shading, string>> = {
  soft: 'Soft',
  flat: 'Flat',
};

/** Each texture's name in the settings. */
export const TEXTURE_LABELS: Readonly<Record<Texture, string>> = {
  plain: 'Plain',
  felt: 'Felt',
  cloth: 'Cloth',
  wood: 'Wood',
};

/** The variables for a surface. A null colour leaves every --color-felt*
 *  variable unset, so the theme's own felt (and its text colours) show. The
 *  colour is validated where it is read and written (settings.ts). */
export function surfaceVars(s: SurfaceSettings): CSSProperties {
  const shade = s.shading === 'soft' ? TEXTURE_SHADE[s.texture] : undefined;
  const pattern = TEXTURE_LAYERS[s.texture];
  const texture =
    shade === undefined
      ? pattern
      : pattern === 'none'
        ? shade
        : `${pattern}, ${shade}`;
  const vars: Record<string, string> = {
    '--table-texture': texture,
    ...SHADING_VARS[s.shading],
  };
  if (s.color !== null) {
    vars['--color-felt'] = s.color;
    vars['--color-felt-dark'] =
      `color-mix(in srgb, ${s.color}, var(--felt-shade-ink) 25%)`;
    vars['--color-felt-border'] =
      `color-mix(in srgb, ${s.color}, var(--felt-shade-ink) 45%)`;
    // Always light: a colour too light for white text is refused (settings.ts).
    vars['--color-felt-text'] = 'var(--felt-picked-text)';
    vars['--color-felt-muted'] = 'var(--felt-picked-muted)';
  }
  return vars;
}

/** Colour swatches offered besides the theme's felt and a custom pick.
 *  Midnight and slate were Gin's table backgrounds. All dark: see
 *  MIN_TEXT_CONTRAST. */
export const SURFACE_SWATCHES: ReadonlyArray<{color: string; label: string}> = [
  {color: '#1a6b35', label: 'Green'},
  {color: '#1d4e89', label: 'Blue'},
  {color: '#7a1f2b', label: 'Burgundy'},
  {color: '#6b4a2b', label: 'Walnut'},
  {color: '#081522', label: 'Midnight'},
  {color: '#1f232c', label: 'Slate'},
];
