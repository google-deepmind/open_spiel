# OpenSpiel GUI style

These rules cover both halves of the GUI: the Python server in
`open_spiel/python/gui/` and the browser client in `open_spiel/gui/`. Every
rule marked _checked_ is checked by the commands below; review enforces the
rest.

## Style guides

- Python follows the
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
  with OpenSpiel's two-space indentation.
- TypeScript follows the
  [Google TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html).

|                    | Python                                 | TypeScript                                                                                            |
| ------------------ | -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Format (_checked_) | pyink                                  | Prettier                                                                                              |
| Lint (_checked_)   | pylint                                 | ESLint: the gts rules on typescript-eslint's strict, type-checked set, and the TypeScript rules below |
| Types (_checked_)  | mypy, strict                           | tsc, strict                                                                                           |
| Configuration      | `open_spiel/python/gui/pyproject.toml` | `open_spiel/gui/`                                                                                     |

Run the same checks locally before sending a change:

```sh
# Python, from the repository root.
pip install --group open_spiel/python/gui/pyproject.toml:check
pyink --check --config open_spiel/python/gui/pyproject.toml open_spiel/python/gui
pylint --rcfile open_spiel/python/gui/pyproject.toml open_spiel/python/gui
mypy --config-file open_spiel/python/gui/pyproject.toml open_spiel/python/gui

# TypeScript, from open_spiel/gui.
npm ci
npm run check
```

## TypeScript rules

These rules from the Google TypeScript Style Guide are in
`eslint.config.mjs` on top of the gts rule set (_checked_).

- `T[]` and `readonly T[]` only when `T` is a primitive or a dotted name;
  otherwise `Array<T>` and `ReadonlyArray<T>`: `Array<string | number>`,
  `ReadonlyArray<{id: string}>`.
- Every exported function, class, constant, type and interface has a
  `/** ... */` comment that says what it is for.
- A top-level function is a declaration, `function f() {}`, not
  `const f = () => {}`. Arrow functions stay for callbacks and nested
  functions.
- Every `switch` has a `default`. On a union, the default is a compile-time
  exhaustiveness check:

  ```ts
  default: {
    const unreachable: never = field;
    throw new Error(`Unhandled field kind: ${String(unreachable)}`);
  }
  ```

- Numbers are parsed with `Number()`, which is NaN for malformed text. No
  `parseFloat`; `parseInt` only with a radix other than 10. A CSS length is
  read with `pixels()` (`src/styles/pixels.ts`).
- No `Array()` or `new Array()`: an array literal, or
  `Array.from({length: n}, () => value)`.
- A body without braces sits on the line of its `if`, `else`, `for` or
  `while`; a body on its own line has braces.
- Exports are named, not default.
- DOM code reads `textContent`, not `innerText`.
- Production code types an object literal by its declaration,
  `const x: T = {...}`, not `{...} as T`, except as an argument.

The deliberate exceptions:

| Pattern                                  | Where                                         | Why                                                                   |
| ---------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| Default exports                          | `playwright.config.ts`, `e2e/global_setup.ts` | Playwright requires them.                                             |
| `locator.innerText()`                    | `e2e/`                                        | Playwright's rendered, visible text, which the privacy tests read.    |
| Object literals cast with `as`           | tests (`*_test.ts*`) and `testing/`           | Fakes build partial objects, such as events jsdom does not implement. |
| A type parameter used only in the return | `useStickToBottom<T extends HTMLElement>`     | The React ref pattern, as in `useRef<T>`; callers always name `T`.    |

## Size

- A file is at most 500 lines (_checked_). TypeScript does not count blank
  lines and comments; a Python file, where docstrings and comments count, is
  at most 600.
- A TypeScript function is at most 80 lines, not counting blank lines and
  comments (_checked_). A Python function is at most 50 statements
  (_checked_).
- A file has one responsibility. When a file nears the limit, split it along
  that responsibility rather than compressing it.

## Types and the wire format

- Every Python function is fully annotated, and every exported TypeScript
  function declares its parameter and return types (_checked_). No `any`; no
  `Any` outside code that parses untrusted input.
- The wire format between server and client is defined once, in Python. The
  TypeScript types are generated from it and never edited by hand.

## Comments and names

- Comments say what the code cannot: why it is this way, an invariant, a unit,
  a published rule or paper. They never narrate the code.
- Comments and docstrings describe the code as it is. They carry no history,
  no issue or ticket numbers, no author names, and no `TODO` or `FIXME`
  (_checked_ in Python).
- Nothing refers to infrastructure, repositories, services or people outside
  this repository.
- Names are complete words. Abbreviate only where the abbreviation is the
  common term (`id`, `url`, `rng`), plus the Google Python Style Guide's two
  single letters: `e` for a caught exception and `f` for an open file.

## Tests

- Tests live beside the code they test: `foo.py` is tested by `foo_test.py`,
  written with `absltest`; `foo.ts` is tested by `foo_test.ts`.
- A bug fix comes with the test that would have caught it.
- Tests are deterministic. Every random source is seeded.

## Panels

- A panel with several parts is a panel group (`src/dock/panel_group.tsx`):
  its sections as tabs on a sunken track, one shown at a time, each of which
  can be popped out into a panel of its own and docked back. A section that
  draws nothing has no tab, so a section that must keep its tab while it
  draws nothing (a hidden list) keeps an empty box. A section's `Footer`
  (History's replay control) sits under it, outside its scroller.
- Text a reader copies is a `CopyableText` (`src/controls/copy_button.tsx`):
  a card with an icon copy button in its top right corner. A few numbers
  (rewards, returns) are a `CompactText`, the same card with no copy
  button.
- Inset content sits in a `.card` (`--radius-card`, `--color-surface-raised`
  on a `--color-border` edge); a status line is a `.status-bar`.
- Icons are drawn in CSS (`.icon` in `src/controls/controls.css`); a control
  that shows only an icon is named by its `aria-label` and `title`.

## Dependencies and assets

- The server imports only the standard library, OpenSpiel (`pyspiel` and
  `open_spiel`), `absl` and `numpy`.
- The client imports only its own files under `open_spiel/gui/` and the
  packages `package.json` declares (_checked_ for relative and absolute
  paths).
- A relative import names the file with its extension: `./app.tsx`, not
  `./app` (_checked_).
- Runtime dependencies are pinned to an exact version.
- Art is SVG (drawn in code, shipped as `.svg` files, or an SVG `data:` URI)
  or a small PNG `data:` URI. The client ships no fonts and no other images,
  and loads nothing from another host, SVG included (_checked_).
- Art is drawn in code for this client. Each `.svg` file starts with the
  Apache-2.0 header and a comment that says what it draws and that it is
  drawn in code.
- Third-party notices are in `licenses/`.
- Every source file starts with the OpenSpiel Apache-2.0 licence header.
