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

// The outlines of the Staunton piece set (staunton.tsx), drawn for this
// client in a 100 by 100 box: each piece stands on the line y = 87.5 and is
// symmetric about x = 50, but for the knight, which faces left.

/** A kind of piece, as the set names its drawings. */
export type StauntonRole =
  'pawn' | 'knight' | 'bishop' | 'rook' | 'queen' | 'king';

/**
 * How a path is painted: 'body' in the piece's fill with a dark edge;
 * 'line' as a detail line in the piece's line colour; 'inner' as a light
 * detail line drawn on the dark pieces only, where the edges between their
 * parts would otherwise vanish; 'dot' filled in the line colour; 'edge' as a
 * dark line on either colour.
 */
export type StauntonPaint = 'body' | 'line' | 'inner' | 'dot' | 'edge';

/** The box every piece is drawn in. */
export const STAUNTON_VIEW_BOX = '0 0 100 100';

/** The width of a piece's lines, in the box's units. */
export const STAUNTON_STROKE = 3.3;

/** A circle as a path: centre `cx`, `cy` and radius `r`. */
function circle(cx: number, cy: number, r: number): string {
  const arc = `a${String(r)} ${String(r)} 0 1 0`;
  const across = String(2 * r);
  return `M${String(cx - r)} ${String(cy)}${arc} ${across} 0${arc} -${across} 0z`;
}

/**
 * The base the pieces stand on (the bishop's foot flares instead): a slab
 * with rounded corners from y = 80.5 to the line y = 87.5, `half` either
 * side of x = 50.
 */
function slab(half: number): string {
  const left = 50 - half;
  const right = 50 + half;
  const n = (value: number) => String(value);
  return (
    `M50 87.5H${n(right - 3)}Q${n(right)} 87.5 ${n(right)} 84.5V83.5` +
    `Q${n(right)} 80.5 ${n(right - 3)} 80.5H${n(left + 3)}` +
    `Q${n(left)} 80.5 ${n(left)} 83.5V84.5Q${n(left)} 87.5 ${n(left + 3)} 87.5Z`
  );
}

/** Each piece's paths, painted in order. */
export const STAUNTON_PATHS: Readonly<
  Record<StauntonRole, ReadonlyArray<readonly [string, StauntonPaint]>>
> = {
  pawn: [
    [slab(28), 'body'],
    [
      'M40 51.5C37.5 61.5 32 71 24.5 80.5H75.5C68 71 62.5 61.5 60 51.5Z',
      'body',
    ],
    [
      'M36 43.5H64Q68 43.5 68 47.5Q68 51.5 64 51.5H36' +
        'Q32 51.5 32 47.5Q32 43.5 36 43.5Z',
      'body',
    ],
    [circle(50, 30, 13.5), 'body'],
    ['M36 51.5H64M24.5 80.5H75.5', 'inner'],
  ],
  rook: [
    [slab(29), 'body'],
    [
      'M24 80.5H76V76.5C70 75.5 64.5 72 64.5 67L63.5 40.5' +
        'C63.5 36.5 68 34 72 33.5V18H64.5V24H55.5V18H44.5V24H35.5V18H28V33.5' +
        'C32 34 36.5 36.5 36.5 40.5L35.5 67C35.5 72 30 75.5 24 76.5Z',
      'body',
    ],
    ['M24 80.5H76M36.5 40.5H63.5M28 33.5H72', 'line'],
  ],
  bishop: [
    [
      'M22 87.5H78Q81 87.5 81 85C81 82.5 77.5 81.5 73.5 80.5H26.5' +
        'C22.5 81.5 19 82.5 19 85Q19 87.5 22 87.5Z',
      'body',
    ],
    ['M40 67.5C40 74 34.5 78 26.5 80.5H73.5C65.5 78 60 74 60 67.5Z', 'body'],
    [
      'M36 60.5H64Q67.5 60.5 67.5 64Q67.5 67.5 64 67.5H36' +
        'Q32.5 67.5 32.5 64Q32.5 60.5 36 60.5Z',
      'body',
    ],
    [
      'M38.5 60.5C27 54 26.5 40.5 35.5 31.5C40.5 26.5 46.5 22 50 17' +
        'C53.5 22 59.5 26.5 64.5 31.5C73.5 40.5 73 54 61.5 60.5Z',
      'body',
    ],
    [circle(50, 12, 5.5), 'body'],
    ['M59 34.5L45.5 50', 'line'],
    ['M36 60.5H64M36 67.5H64M26.5 80.5H73.5', 'inner'],
  ],
  knight: [
    [slab(28), 'body'],
    [
      'M31 80.5C31.5 76.5 33 72.5 37 68.5C41 64.5 45 59.5 45.5 52.5' +
        'C42.5 55.5 38.5 58.5 33 59C30 59.3 27.5 58 26 56' +
        'C23 57 19 57.2 16.5 56C13.5 54.5 12.5 51 13.5 48.5' +
        'C15 45 18.5 42.5 22 39.5C28.5 35 34.5 29.5 38 21.5' +
        'C38.5 16 39 12 40.5 9C43 12 45 14.5 45.5 17.5' +
        'C46.5 15 48.5 12.5 50.5 11.5C51.5 14 52 16.5 52.5 19.5' +
        'C62 21.5 70.5 29 74 40C78 52 77 66 75 80.5Z',
      'body',
    ],
    [circle(37, 31.5, 2.4), 'dot'],
    [circle(17.5, 50, 1.5), 'dot'],
    [
      'M54 25.5C62 28 67.5 34 70 42.5C72.5 51 72.5 62 71 74' +
        'M45.5 52.5C44.5 48 41.5 45 37.5 43.5',
      'line',
    ],
    ['M31 80.5H75', 'inner'],
  ],
  queen: [
    [slab(30), 'body'],
    [
      'M23 80.5H77C70 79 65.5 74 64.5 68C64 64.5 64.5 62 66 59.5' +
        'C73 53.5 81 44 85 30Q78 40 73 42.5Q71 34 68.5 23' +
        'Q63.5 34 58.5 40Q54 32 50 19Q46 32 41.5 40Q36.5 34 31.5 23' +
        'Q29 34 27 42.5Q22 40 15 30C19 44 27 53.5 34 59.5' +
        'C35.5 62 36 64.5 35.5 68C34.5 74 30 79 23 80.5Z',
      'body',
    ],
    [
      [
        circle(15, 27, 4.6),
        circle(31.5, 19.5, 4.6),
        circle(50, 14.5, 4.6),
        circle(68.5, 19.5, 4.6),
        circle(85, 27, 4.6),
      ].join(''),
      'body',
    ],
    [
      'M34 59.5C44.5 56.5 55.5 56.5 66 59.5M35.5 68C45 66 55 66 64.5 68' +
        'M23 80.5H77',
      'line',
    ],
  ],
  king: [
    [slab(30), 'body'],
    ['M47.5 22V15H42V10H47.5V4.5H52.5V10H58V15H52.5V22Z', 'body'],
    [
      'M25 80.5C29 79 31 76 30.5 72L28 64C20 60.5 15 54 15 46.5' +
        'C15 37.9 21.9 31 30.5 31C37 31 42.5 34.5 45 40' +
        'C47 41.5 53 41.5 55 40C57.5 34.5 63 31 69.5 31' +
        'C78.1 31 85 37.9 85 46.5C85 54 80 60.5 72 64' +
        'L69.5 72C69 76 71 79 75 80.5Z',
      'body',
    ],
    [
      'M50 21.5C54.5 23.5 58 27.5 58 32.5C58 37 54.5 40.5 50 41.5' +
        'C45.5 40.5 42 37 42 32.5C42 27.5 45.5 23.5 50 21.5Z',
      'body',
    ],
    [
      'M21 49C21 42 25.5 37.5 31 37M79 49C79 42 74.5 37.5 69 37' +
        'M44.5 42C38 47 34.5 55 34 62.5M55.5 42C62 47 65.5 55 66 62.5',
      'line',
    ],
    [
      'M28 64C42.5 60.5 57.5 60.5 72 64M30.5 72C43.5 70 56.5 70 69.5 72' +
        'M25 80.5H75',
      'line',
    ],
  ],
};
