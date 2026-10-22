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

import {act} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {enter, labelled} from '../testing/input.ts';
import type {Rendered} from '../testing/render.tsx';
import {only, render} from '../testing/render.tsx';
import type {Field} from './field.ts';
import {FieldForm} from './field_form.tsx';

const FIELDS: Field[] = [
  {
    name: 'max_simulations',
    label: 'max_simulations',
    description: 'Simulations per move.',
    kind: 'integer',
    default: 1000,
    minimum: 1,
    maximum: 10000,
  },
  {
    name: 'uct_c',
    label: 'uct_c',
    description: null,
    kind: 'number',
    default: 2,
    minimum: 0,
    maximum: null,
  },
  {
    name: 'solve',
    label: 'solve',
    description: null,
    kind: 'boolean',
    default: true,
  },
  {
    name: 'model',
    label: 'Model',
    description: null,
    kind: 'string',
    default: '',
  },
  {
    name: 'size',
    label: 'size',
    description: null,
    kind: 'choice',
    default: 9,
    choices: [9, 13],
  },
];
const DRAFTS = {
  max_simulations: '1000',
  uct_c: '2',
  solve: true,
  model: 'small',
  size: '1',
};

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
});

function control(label: string): HTMLInputElement | HTMLSelectElement {
  const element = labelled(rendered?.container ?? document, label);
  if (element instanceof HTMLTextAreaElement) {
    throw new Error(`${label} labels a text area.`);
  }
  return element;
}

describe('FieldForm', () => {
  it('gives each field the control of its kind, labelled', () => {
    rendered = render(
      <FieldForm
        fields={FIELDS}
        drafts={DRAFTS}
        errors={{}}
        onChange={vi.fn()}
      />,
    );
    const simulations = control('max_simulations') as HTMLInputElement;
    expect([
      simulations.type,
      simulations.min,
      simulations.max,
      simulations.step,
    ]).toEqual(['number', '1', '10000', '1']);
    expect((control('uct_c') as HTMLInputElement).step).toBe('any');
    expect((control('uct_c') as HTMLInputElement).max).toBe('');
    expect((control('solve') as HTMLInputElement).checked).toBe(true);
    expect(control('Model').value).toBe('small');
    const size = control('size') as HTMLSelectElement;
    expect(Array.from(size.options, option => option.text)).toEqual([
      '9',
      '13',
    ]);
    expect(size.value).toBe('1');
  });

  it('describes a control by its error, and its field behind an info toggle', () => {
    rendered = render(
      <FieldForm
        fields={FIELDS}
        drafts={{...DRAFTS, max_simulations: '0'}}
        errors={{max_simulations: 'A whole number from 1 to 10000.'}}
        onChange={vi.fn()}
      />,
    );
    const {container} = rendered;
    const simulations = control('max_simulations');
    expect(
      document.getElementById(
        simulations.getAttribute('aria-describedby') ?? '',
      )?.textContent,
    ).toBe('A whole number from 1 to 10000.');
    expect(simulations.getAttribute('aria-invalid')).toBe('true');
    expect(control('uct_c').getAttribute('aria-invalid')).toBe('false');
    expect(control('uct_c').hasAttribute('aria-describedby')).toBe(false);
    expect(container.textContent).not.toContain('Simulations per move.');
    const info = only(container, 'button[aria-label="More info"]');
    act(() => {
      info.click();
    });
    expect(info.getAttribute('aria-expanded')).toBe('true');
    const tooltip = only(document.body, '[role="tooltip"]');
    expect(tooltip.textContent).toBe('Simulations per move.');
    expect(info.getAttribute('aria-controls')).toBe(tooltip.id);
    act(() => {
      document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));
    });
    expect(document.querySelector('[role="tooltip"]')).toBeNull();
  });

  it('puts each label beside its control, in one row per field', () => {
    rendered = render(
      <FieldForm
        fields={FIELDS}
        drafts={DRAFTS}
        errors={{}}
        onChange={vi.fn()}
      />,
    );
    const rows = rendered.container.querySelectorAll('.field-form__field');
    expect(rows).toHaveLength(FIELDS.length);
    for (const row of Array.from(rows)) {
      expect(row.children[0]?.classList).toContain('field-form__label');
      expect(row.children[1]?.matches('input, select, div')).toBe(true);
    }
  });
});

describe('FieldForm names', () => {
  it('names a control whose setting name has a space', () => {
    const field: Field = {
      name: 'max depth',
      label: 'max depth',
      description: null,
      kind: 'string',
      default: '',
    };
    rendered = render(
      <FieldForm
        fields={[field]}
        drafts={{'max depth': ''}}
        errors={{}}
        onChange={vi.fn()}
      />,
    );
    expect(control('max depth')).toBeInstanceOf(HTMLInputElement);
  });
});

describe('FieldForm presets', () => {
  it('offers presets as chips that fill the box', () => {
    const onChange = vi.fn();
    const field: Field = {...FIELDS[0], presets: [100, 1000]} as Field;
    rendered = render(
      <FieldForm
        fields={[field]}
        drafts={{max_simulations: '1000'}}
        errors={{}}
        onChange={onChange}
      />,
    );
    const group = only(rendered.container, '[role="group"]');
    expect(group.getAttribute('aria-label')).toBe('max_simulations presets');
    const chips = Array.from(group.querySelectorAll('button'));
    expect(chips.map(chip => chip.getAttribute('aria-pressed'))).toEqual([
      'false',
      'true',
    ]);
    act(() => {
      chips[0]?.click();
    });
    expect(onChange).toHaveBeenCalledExactlyOnceWith('max_simulations', '100');
  });
});

describe('FieldForm edits', () => {
  it('reports each edit by field name', () => {
    const onChange = vi.fn();
    rendered = render(
      <FieldForm
        fields={FIELDS}
        drafts={DRAFTS}
        errors={{}}
        onChange={onChange}
      />,
    );
    enter(control('max_simulations'), '50');
    enter(control('size'), '0');
    act(() => {
      control('solve').click();
    });
    expect(onChange.mock.calls).toEqual([
      ['max_simulations', '50'],
      ['size', '0'],
      ['solve', false],
    ]);
  });

  it('offers no choice when the draft is none of the choices', () => {
    rendered = render(
      <FieldForm
        fields={FIELDS}
        drafts={{...DRAFTS, size: ''}}
        errors={{}}
        onChange={vi.fn()}
      />,
    );
    const size = control('size') as HTMLSelectElement;
    expect(size.value).toBe('');
    expect(size.options[0]?.text).toBe('Choose…');
  });

  it('refuses a field without a draft', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    expect(() =>
      render(
        <FieldForm
          fields={FIELDS}
          drafts={{}}
          errors={{}}
          onChange={vi.fn()}
        />,
      ),
    ).toThrow('The field max_simulations has no draft.');
    vi.restoreAllMocks();
  });

  it('labels a checkbox by its field', () => {
    rendered = render(
      <FieldForm
        fields={FIELDS}
        drafts={DRAFTS}
        errors={{}}
        onChange={vi.fn()}
      />,
    );
    const box = only(rendered.container, 'input[type="checkbox"]');
    expect((box as HTMLInputElement).labels?.[0]?.textContent).toBe('solve');
  });
});
