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

// A reader for the stylesheets the client's style gates check. It knows as
// much CSS as those gates need: rules, nested at-rules, declarations, comments
// and strings. It throws on unbalanced braces, so a stylesheet it misreads
// fails the gate instead of passing it unread.

/** One `property: value` declaration, with the selectors it sits under. */
export interface Declaration {
  /** Every enclosing selector and at-rule prelude, outermost first. */
  selectors: string[];
  property: string;
  value: string;
  /** 1-based line of the property in the stylesheet. */
  line: number;
}

/** Blanks every comment, keeping offsets and line breaks where they were. */
function blankComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, comment =>
    comment.replace(/[^\n]/g, ' '),
  );
}

/** The offset just past the string that opens at `start`. */
function skipString(css: string, start: number): number {
  const quote = css[start];
  for (let offset = start + 1; offset < css.length; offset++) {
    if (css[offset] === '\\') {
      offset++;
    } else if (css[offset] === quote) {
      return offset + 1;
    }
  }
  throw new Error(`Unterminated string at offset ${String(start)}.`);
}

class Reader {
  private readonly css: string;
  private readonly declarations: Declaration[] = [];
  private position = 0;

  constructor(css: string) {
    this.css = css;
  }

  read(): Declaration[] {
    this.block([]);
    if (this.position < this.css.length) {
      throw new Error(
        `Unmatched } on line ${String(this.lineAt(this.position))}.`,
      );
    }
    return this.declarations;
  }

  private lineAt(offset: number): number {
    return this.css.slice(0, offset).split('\n').length;
  }

  // Reads items up to the `}` that closes the current block, or the end.
  private block(selectors: string[]): void {
    let start = this.position;
    let depth = 0;
    while (this.position < this.css.length) {
      const character = this.css[this.position];
      if (character === '"' || character === "'") {
        this.position = skipString(this.css, this.position);
        continue;
      }
      if (character === '(') {
        depth++;
      } else if (character === ')') {
        depth--;
      } else if (character === ';' && depth === 0) {
        this.declaration(selectors, start, this.position);
        start = this.position + 1;
      } else if (character === '{') {
        const prelude = this.css.slice(start, this.position).trim();
        this.position++;
        this.block([...selectors, prelude]);
        if (this.css[this.position] !== '}') {
          throw new Error(`Unclosed { after ${JSON.stringify(prelude)}.`);
        }
        start = this.position + 1;
      } else if (character === '}') {
        this.declaration(selectors, start, this.position);
        return;
      }
      this.position++;
    }
    this.declaration(selectors, start, this.position);
  }

  private declaration(selectors: string[], start: number, end: number): void {
    const text = this.css.slice(start, end);
    if (text.trim() === '') {
      return;
    }
    const colon = text.indexOf(':');
    if (colon < 0) {
      throw new Error(
        `Not a declaration on line ${String(this.lineAt(start))}: ${text.trim()}`,
      );
    }
    const offset = start + text.length - text.trimStart().length;
    this.declarations.push({
      selectors,
      property: text.slice(0, colon).trim().toLowerCase(),
      value: text.slice(colon + 1).trim(),
      line: this.lineAt(offset),
    });
  }
}

/** Every declaration in `css`, in order. */
export function readDeclarations(css: string): Declaration[] {
  return new Reader(blankComments(css)).read();
}
