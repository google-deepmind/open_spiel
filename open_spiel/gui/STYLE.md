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

|                    | Python                                 | TypeScript                                                            |
| ------------------ | -------------------------------------- | --------------------------------------------------------------------- |
| Format (_checked_) | pyink                                  | Prettier                                                              |
| Lint (_checked_)   | pylint                                 | ESLint: the gts rules on typescript-eslint's strict, type-checked set |
| Types (_checked_)  | mypy, strict                           | tsc, strict                                                           |
| Configuration      | `open_spiel/python/gui/pyproject.toml` | `open_spiel/gui/`                                                     |

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

## Size

- A file is at most 500 lines (_checked_). TypeScript does not count blank
  lines and comments.
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
- Icons are drawn in CSS (`.icon` in `src/controls/controls.css`), since the
  client ships no images; a control that shows only an icon is named by its
  `aria-label` and `title`.

## Dependencies and assets

- The server imports only the standard library, OpenSpiel (`pyspiel` and
  `open_spiel`), `absl` and `numpy`.
- The client imports only its own files under `open_spiel/gui/` and the
  packages `package.json` declares (_checked_ for relative and absolute
  paths).
- A relative import names the file with its extension: `./app.tsx`, not
  `./app` (_checked_).
- Runtime dependencies are pinned to an exact version.
- The client ships no images, fonts or card art. Every game is drawn with text
  and CSS.
- Third-party notices are in `licenses/`.
- Every source file starts with the OpenSpiel Apache-2.0 licence header.
