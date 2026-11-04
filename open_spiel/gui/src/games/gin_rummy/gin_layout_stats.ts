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

// The stats region's room. The stats' text has a px floor (TEXT_FLOOR_PX)
// and their region is sized in card widths, so the region is sized here for
// the stats' longest text at the font it is drawn at: gin_rummy.css draws it
// at max(12 px, GIN_STATS_FONT card widths). Each layout sizes its region for
// 12 px at some card width and is drawn only from that card width
// (TableSpec.minDrawnCardWidth): the wide layout's column for each of
// GIN_STATS_CARD_PX, and wider where the panel has width to spare (a wider
// column is shorter); the narrow layout's full-width row for 12 px at the
// smallest card that layout draws, ungated, and at the larger
// GIN_STATS_CARD_PX, gated.

import {TEXT_FLOOR_PX} from '../../styles/text_floor.ts';
import {
  GIN_BLOCK_REASON,
  GIN_STATUS,
  ginLongestTurnLine,
} from './gin_turn_line.ts';

/**
 * Card widths on the table at the smallest scale it draws, px. The floor is
 * the smallest card's (the grid's), so a fitted board never draws a card
 * narrower than 40 px: below it, it scrolls.
 */
export const GIN_MIN_CARD_WIDTH = 40;

/** The stats' font, card widths (gin_rummy.css .gin-stats). */
export const GIN_STATS_FONT = 0.2;
/**
 * The smallest card the wide layout sized for GIN_STATS_FONT is drawn at,
 * px: its stats hold 12 px text from here.
 */
export const GIN_WIDE_MIN_CARD_PX = Math.ceil(TEXT_FLOOR_PX / GIN_STATS_FONT);
/**
 * The card widths, px, the stats' room is also offered for, largest first.
 * A region sized for its text at TEXT_FLOOR_PX / g card widths holds the
 * stats' max(12 px, GIN_STATS_FONT card widths) at any card of g px or more
 * (TableVariant.minDrawnCardWidth g): smaller cards still get the wide
 * layout, its column a little larger, rather than the narrow row under the
 * supply. The last is GIN_MIN_CARD_WIDTH, the board's floor.
 */
export const GIN_STATS_CARD_PX = [
  GIN_WIDE_MIN_CARD_PX,
  48,
  GIN_MIN_CARD_WIDTH,
] as const;

/**
 * The stats' type, in ems of the stats font (gin_rummy.css): line height; a
 * figure line's (its value is 1.12 em); the gap between the figures and
 * before the turn line; the gap between the figures and the turn line in
 * the narrow row; the turn line's font (of the stats font) and the gap
 * between its pieces (turn-line ems).
 */
const STATS_TYPE = {
  line: 1.25,
  figureLine: 1.25 * 1.12,
  gap: 0.35,
  rowGap: 1.2,
  turn: 1.05,
  pieceGap: 0.4,
} as const;

/**
 * Ems a character of the turn line takes, at most: its status is bold, its
 * notes regular. Measured in Chromium at 0.52 and 0.44, with the font stack
 * fallen back to a wider sans than the fonts it names; rounded up.
 */
const TURN_EM_PER_CHAR = {status: 0.55, note: 0.48} as const;

/** A turn-line piece's width, ems of the turn line's font. */
function statusEms(text: string): number {
  return text.length * TURN_EM_PER_CHAR.status;
}

/** A note is drawn after '· '. */
function noteEms(text: string): number {
  return (text.length + 2) * TURN_EM_PER_CHAR.note;
}

interface StatsContent {
  readonly figures: readonly number[];
  readonly turn: readonly number[];
}

/**
 * What the stats hold, ems: the figures (each line's width, measured as
 * above; the hand's result, shown when the game is over, sizes the narrow
 * row's figures column whole: 13.3 ems was measured for a 27-character
 * line, and GIN_RESULT_FIGURE_LONGEST, the longest result, is held to that
 * length) and the turn line as its pieces (the status, then each note;
 * turn-line ems): while a hand is played the longest gin writes
 * (gin_turn_line.ts), and at the end of the game the game-over line under
 * the returns.
 */
function statsContent(): readonly StatsContent[] {
  const longest = ginLongestTurnLine();
  return [
    {
      figures: [6.8, 7.3],
      turn: [statusEms(longest.status), ...longest.notes.map(noteEms)],
    },
    {
      figures: [6.8, 7.3, 13.3],
      turn: [
        statusEms(GIN_STATUS.gameOver),
        noteEms(GIN_BLOCK_REASON.finished),
      ],
    },
  ];
}

const STATS_CONTENT = statsContent();

/**
 * A piece wider than its line wraps at word breaks, so it fills each line
 * only to about this share.
 */
const STATS_WRAP = 0.8;

/** Lines of `pieces` (ems) laid out as a wrapping flex row `width` ems wide. */
function statsLines(pieces: readonly number[], width: number): number {
  let lines = 0;
  let line = 0;
  for (const p of pieces) {
    if (p > width) {
      lines += (line > 0 ? 1 : 0) + Math.ceil(p / (width * STATS_WRAP));
      line = 0;
    } else if (line > 0 && line + STATS_TYPE.pieceGap + p <= width) {
      line += STATS_TYPE.pieceGap + p;
    } else {
      if (line > 0) lines += 1;
      line = p;
    }
  }
  return lines + (line > 0 ? 1 : 0);
}

function rowHeight(c: StatsContent, width: number): number {
  const {line, figureLine, gap, rowGap, turn} = STATS_TYPE;
  const figureWidth = Math.max(...c.figures);
  const room = width - figureWidth - rowGap;
  if (!(room > 0)) return Infinity;
  return Math.max(
    c.figures.length * figureLine + (c.figures.length - 1) * gap,
    statsLines(c.turn, room / turn) * turn * line,
  );
}

function columnHeight(c: StatsContent, width: number): number {
  const {line, figureLine, gap, turn} = STATS_TYPE;
  if (Math.max(...c.figures.slice(0, 2)) > width) return Infinity;
  const figureLines = c.figures.reduce(
    (n, f) => n + (f > width ? Math.ceil(f / (width * STATS_WRAP)) : 1),
    0,
  );
  return (
    figureLines * figureLine +
    (c.figures.length - 1) * gap +
    gap +
    statsLines(c.turn, width / turn) * turn * line
  );
}

/**
 * Height, ems, of the stats `width` ems wide: the figures above the turn
 * line ('column'), or beside it ('row', the stats box at 3:1 or wider).
 */
function statsTextHeight(width: number, mode: 'column' | 'row'): number {
  return Math.max(
    ...STATS_CONTENT.map(c =>
      mode === 'row' ? rowHeight(c, width) : columnHeight(c, width),
    ),
  );
}

function roundUp3(v: number): number {
  return Math.ceil(v * 1000 - 1e-6) / 1000;
}

/**
 * Height, card widths, a stats region `width` card widths wide needs for its
 * text at `font` card widths, never under `least`. The stats box lays its
 * text out as a row at 3:1 or wider (gin_rummy.css @container), so the
 * height is the least one at which the layout that shape gets holds it.
 */
export function ginStatsHeight(
  width: number,
  font: number,
  least: number,
): number {
  if (!(width > 0 && font > 0 && least > 0)) {
    throw new Error(
      `ginStatsHeight: width, font and least must be > 0, got ${String(width)}, ${String(font)}, ${String(least)}`,
    );
  }
  const em = width / font;
  const row = Math.max(least, statsTextHeight(em, 'row') * font);
  const column = Math.max(least, statsTextHeight(em, 'column') * font);
  if (width >= 3 * row) return roundUp3(row);
  if (width < 3 * column) return roundUp3(column);
  // Too tall for a row at a row's height, short enough for a column only
  // under 3:1: just over a third of the width.
  if (!Number.isFinite(column)) {
    throw new Error(
      `ginStatsHeight: no stats layout holds the text ${String(width)} card widths wide at font ${String(font)}`,
    );
  }
  return roundUp3(width / 3 + 0.01);
}
